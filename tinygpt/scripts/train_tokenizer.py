""" Very tight specification:

vocab_size = 8192

model_type = unigram

character_coverage = 0.9995

bos_id = 1

eos_id = 2

pad_id = 0

unk_id = 3

No hard_vocab_limit (important) """
import os
import sentencepiece as spm

INPUT_PATH = "data/cleaned/clean.txt"
MODEL_PREFIX = "data/spm/spm"
VOCAB_SIZE = 8192

os.makedirs("data/spm", exist_ok=True)

def main():
    spm.SentencePieceTrainer.train(
        input=INPUT_PATH,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        hard_vocab_limit=False
    )

    print("Tokenizer training complete.")
    print("Saved to data/spm/")

if __name__ == "__main__":
    main()