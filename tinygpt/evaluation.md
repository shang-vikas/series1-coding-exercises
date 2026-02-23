Evaluation Strategy for TinyGPT (15M)

This document explains how we evaluate the base language model after pretraining.

We focus on:

Loss

Perplexity

Bits-Per-Character (BPC)

Zero-shot likelihood scoring

We do NOT chase leaderboard metrics.
We measure what matters for understanding scaling.

1️⃣ Training Loss

Training objective:

Next-token prediction.

Cross-entropy:

L=−log⁡P(xt∣x<t)
L=−logP(x
t
	​

∣x
<t
	​

)

Random baseline:

If vocab = 8192:

log⁡(8192)≈8.99
log(8192)≈8.99

If your initial loss ≈ 9 → model is random.

A healthy 15M model trained on ~200M tokens should reach:

Loss ≈ 5.5–6.0

Perplexity ≈ 245–400

2️⃣ Perplexity

Perplexity:

ppl=eloss
ppl=e
loss

Interpretation:

ppl ≈ 8000 → random

ppl ≈ 500 → learning

ppl ≈ 200 → strong small model

Perplexity measures uncertainty.

Lower = better compression of language.

3️⃣ Bits Per Character (BPC)

Why BPC?

Tokenization can distort evaluation.
Character-level metric is more stable.

BPC=lossln⁡(2)×tokenscharacters
BPC=
ln(2)
loss
	​

×
characters
tokens
	​


Lower BPC = better compression.

Expected BPC for tiny models:

~1.5–2.0 range on web text.

4️⃣ WikiText-2 Perplexity

Add a small external benchmark.

Install:

pip install datasets

Use HuggingFace WikiText-2 validation set.

Procedure:

Tokenize with our tokenizer

Compute log-likelihood

Report perplexity

Important:
No fine-tuning on WikiText.

We measure generalization.

5️⃣ Zero-Shot Likelihood Scoring

We implement simple multiple-choice scoring:

Given prompt:

The capital of France is

Score candidates:

Paris

London

Berlin

Compute:

log⁡P(candidate∣prompt)
logP(candidate∣prompt)

Pick highest.

This demonstrates understanding of:

Conditional likelihood

No sampling

Pure probability ranking

This is how serious evaluation harnesses work internally.

6️⃣ What We Expect From 15M Model

It will:

Generate grammatical English

Fail at long reasoning

Show frequency bias

Over-repeat at times

Hallucinate facts

That’s not failure.
That’s capacity limitation.

Small models compress language.
They do not understand the world