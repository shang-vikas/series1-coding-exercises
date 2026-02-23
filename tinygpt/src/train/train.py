import os
import math
import random
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import wandb

from src.model.gpt import TinyGPT


# ==========================
# CONFIG
# ==========================

def load_config(config_name="local"):
    config_path = f"configs/{config_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ==========================
# DATA
# ==========================

def load_memmap(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, context_size):
    ix = np.random.randint(0, len(data) - context_size - 1, size=batch_size)
    x = np.stack([data[i:i+context_size] for i in ix])
    y = np.stack([data[i+1:i+1+context_size] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ==========================
# CHECKPOINTING
# ==========================

def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    # Restore RNG safely
    try:
        torch_rng = checkpoint["torch_rng"]
        if isinstance(torch_rng, torch.Tensor):
            torch.set_rng_state(torch_rng.cpu())
    except Exception as e:
        print("Skipping torch RNG restore:", e)

    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])

    return checkpoint["step"]


# ==========================
# VALIDATION
# ==========================

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_size, vocab_size, eval_iters=20):
    model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_size)
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# ==========================
# TRAIN
# ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="local", help="Config name (local or cloud)")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    wandb.init(project="tinygpt-resume")

    train_data = load_memmap(config["data_path"])
    val_data = load_memmap(config["val_path"])

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        context_size=config["context_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    ).to(DEVICE)

    effective_tokens = config["batch_size"] * config["context_size"] * config["grad_accum_steps"]
    print(f"Effective tokens per optimizer step: {effective_tokens}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["max_steps"]
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if DEVICE.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
    else:
        scaler = torch.amp.GradScaler(enabled=False)

    start_step = 0

    if os.path.exists(config["checkpoint_path"]):
        print("Loading checkpoint...")
        start_step = load_checkpoint(model, optimizer, scheduler, scaler, config["checkpoint_path"])
        print(f"Resuming from step {start_step}")

    criterion = nn.CrossEntropyLoss()
    model.train()

    start_time = time.time()
    tokens_processed = 0

    for step in range(start_step, config["max_steps"]):

        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(config["grad_accum_steps"]):
            x, y = get_batch(train_data, config["batch_size"], config["context_size"])
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=config["use_amp"]):
                logits = model(x)
                loss = criterion(
                    logits.view(-1, config["vocab_size"]),
                    y.view(-1)
                )
                loss = loss / config["grad_accum_steps"]

            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tokens_this_step = config["batch_size"] * config["context_size"] * config["grad_accum_steps"]
        tokens_processed += tokens_this_step

        if step % 50 == 0 and step > start_step:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed

            val_loss = estimate_loss(
                model, val_data,
                config["batch_size"],
                config["context_size"],
                config["vocab_size"]
            )

            print(
                f"Step {step} | "
                f"Train Loss: {total_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Tok/s: {tokens_per_sec:.2f}"
            )

            wandb.log({
                "train_loss": total_loss,
                "val_loss": val_loss,
                "tokens_per_sec": tokens_per_sec,
                "step": step
            })

        if step % config["save_every"] == 0 and step > 0:
            print("Saving checkpoint...")
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }, config["checkpoint_path"])

    print("Training complete.")


if __name__ == "__main__":
    main()