TinyGPT – From Scratch Training (15M Params)

This repo trains a ~15–17M parameter GPT-style language model from scratch on OpenWebText.

Goal:

Understand full LM pipeline

Tokenizer → Dataset → Model → Training → Resume → Eval

Run locally (MPS) and scale to 3090 cloud

Train up to ~200M tokens under $20

🔧 System Requirements

Cloud:

RTX 3090 (24GB VRAM)

CUDA available

50GB+ disk

Local:

Mac (MPS) or CPU

For debugging only

📁 Pipeline Overview

We rebuild everything on cloud in this order:

Download raw text (~1GB)

Clean text

Train SentencePiece tokenizer (8k unigram)

Tokenize + pack to .bin

Train model (pilot → full)

🚀 Cloud Training Instructions
0️⃣ Setup
git clone <repo>
cd tinygpt

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

Verify CUDA:

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

Must show RTX 3090.

1️⃣ Download 1GB OpenWebText

Edit:

scripts/download_owt.py

TARGET_CHAR_COUNT = 1_000_000_000

Run:

python scripts/download_owt.py

Output:

data/raw/raw.txt

Expected size: ~1GB

2️⃣ Clean Text
python scripts/clean_text.py

Output:

data/cleaned/clean.txt

Expected:
~700M–850M characters.

3️⃣ Train Tokenizer (Unigram 8k)
python scripts/train_tokenizer.py

Output:

data/spm/spm.model
data/spm/spm.vocab

Time: ~20–40 minutes.

4️⃣ Tokenize + Pack
python scripts/tokenize_dataset.py

Output:

data/tokenized/train.bin
data/tokenized/val.bin

Check printed token count.

Target:
~200M tokens.

If <150M:
Increase raw text size and repeat.

🧠 Model Specs

~17M parameters

d_model = 384

layers = 6

heads = 6

vocab = 8192

context = 512

weight tying enabled

🧪 Pilot Training (50M Tokens)

Edit train.py config:

CONTEXT_SIZE = 512
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4
USE_AMP = True
MAX_STEPS = 800
LR = 5e-4
SAVE_EVERY = 100

Run:

PYTHONPATH=. python src/train/train.py

Monitor GPU:

watch -n 1 nvidia-smi

Expect:

20k–40k tokens/sec

<8GB VRAM usage

Loss decreasing smoothly

🧠 Full Training (200M Tokens)

After pilot is stable:

MAX_STEPS = 3200

Then:

PYTHONPATH=. python src/train/train.py

Estimated runtime:
~2–4 hours on 3090.

🔁 Resume Training

Training auto-resumes if:

checkpoint.pt

exists.

If spot instance dies:

PYTHONPATH=. python src/train/train.py

Resume happens automatically.

📊 Metrics Logged

Train loss

Val loss

Tokens/sec

Step count

WandB logging enabled

Perplexity:

ppl = exp(loss)
💰 Cost Estimate

Pilot (50M tokens):
~30–45 minutes

Full (200M tokens):
~2–4 hours

On Vast.ai 3090 spot:
<$10 likely
<$20 worst case

🧠 Learning Outcomes

You will understand:

Scaling laws (Chinchilla intuition)

Tokenizer impact

Batch size vs LR

Resume safety

Throughput estimation

Cloud cost control

Small model capacity limits

🔬 Next Steps After Base LM

Compute real perplexity

Add text generation script

Instruction fine-tuning

Tiny reward model

Log-likelihood evaluation harness

This project is small in parameters, but large in understanding.

You are not building a toy.
You are building the mental model of how large models actually train