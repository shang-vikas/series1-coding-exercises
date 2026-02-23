import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

INPUT_PATH = "data/cleaned/clean.txt"
SPM_PATH = "data/spm/spm.model"

TRAIN_OUTPUT = "data/tokenized/train.bin"
VAL_OUTPUT = "data/tokenized/val.bin"

VAL_SPLIT = 0.05  # 5%

os.makedirs("data/tokenized", exist_ok=True)

def main():
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_PATH)

    bos_id = sp.bos_id()

    print("Loading cleaned text...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")

    all_tokens = []

    print("Tokenizing...")
    for doc in tqdm(docs):
        if not doc.strip():
            continue

        tokens = sp.encode(doc)
        all_tokens.append(bos_id)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    total_tokens = len(all_tokens)
    print("Total tokens:", total_tokens)

    split_idx = int(total_tokens * (1 - VAL_SPLIT))

    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    train_tokens.tofile(TRAIN_OUTPUT)
    val_tokens.tofile(VAL_OUTPUT)

    print("Saved:")
    print("Train tokens:", len(train_tokens))
    print("Val tokens:", len(val_tokens))


if __name__ == "__main__":
    main()