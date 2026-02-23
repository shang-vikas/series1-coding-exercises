Instruction Fine-Tuning TinyGPT

After base LM training, we add alignment.

We do NOT jump to RLHF immediately.

We do:

Supervised instruction fine-tuning

Optional tiny reward model

Optional preference optimization

1️⃣ Why Instruction Tuning?

Base LM learns:

Predict next token.

Instruction tuning teaches:

Follow user intent.

Without it, model:

Continues prompts

Ignores formatting

Doesn’t answer directly

2️⃣ Dataset Choice

Small curated instruction dataset:

Options:

Alpaca-style 50k

Dolly subset

Self-generated synthetic instructions

For 15M model:

10k–50k examples is enough.

More will overfit quickly.

3️⃣ Format

We standardize prompt format:

### Instruction:
{instruction}

### Response:
{answer}

Model learns response boundary.

4️⃣ Fine-Tuning Strategy

Reuse pretrained weights.

Change:

Lower LR (e.g., 1e-4)

No weight decay change

Fewer steps (500–2000)

No need for long runs.

Small model adapts quickly.

5️⃣ What Changes After Instruction Tuning?

Model will:

Answer directly

Stop continuing prompt endlessly

Respect structure

But it will NOT:

Become magically intelligent

Gain reasoning depth

Capacity is still 15M parameters.

6️⃣ Reward Model (Optional)

For learning:

We can train tiny reward model:

Input:
(prompt, response A, response B)

Output:
Which is better?

Train small classifier on top of frozen LM embeddings.

This teaches:

Preference learning

Pairwise ranking loss

But for 15M model, this is mostly educational.

7️⃣ What You Learn From This Phase

Difference between pretraining and alignment

Data efficiency of fine-tuning

Catastrophic forgetting

Overfitting in small models

Scaling law limitations

Final Reality Check

A 15M model:

Is not GPT-2 XL.
Is not LLaMA.
Is not ChatGPT.

But training it from scratch teaches you:

Optimization dynamics

Tokenization impact

Throughput engineering

Resume safety

Scaling tradeoffs

Evaluation discipline

This is not about model size.

It’s about understanding the machine.