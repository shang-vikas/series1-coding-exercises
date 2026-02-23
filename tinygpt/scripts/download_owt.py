import os
from datasets import load_dataset
from tqdm import tqdm

# ==============================
# CONFIG (local dev)
# ==============================
TARGET_CHAR_COUNT = 200_000_000  # 200MB approx
OUTPUT_PATH = "data/raw/raw.txt"

os.makedirs("data/raw", exist_ok=True)

def main():
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    total_chars = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Downloading"):
            text = example["text"].strip()

            if not text:
                continue

            f.write(text + "\n\n")
            total_chars += len(text)

            if total_chars >= TARGET_CHAR_COUNT:
                break

    print(f"\nDone. Total characters written: {total_chars}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
