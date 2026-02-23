import os
import re
from tqdm import tqdm

INPUT_PATH = "data/raw/raw.txt"
OUTPUT_PATH = "data/cleaned/clean.txt"

MIN_CHARS = 200
MAX_CHARS = 50_000

os.makedirs("data/cleaned", exist_ok=True)

def clean_text(text: str) -> str:
    # Remove simple HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    total_docs = 0
    kept_docs = 0
    total_chars_kept = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for doc in tqdm(docs, desc="Cleaning"):
            total_docs += 1

            doc = clean_text(doc)

            if len(doc) < MIN_CHARS:
                continue
            if len(doc) > MAX_CHARS:
                continue

            out.write(doc + "\n\n")
            kept_docs += 1
            total_chars_kept += len(doc)

    print("\nCleaning complete.")
    print(f"Total docs: {total_docs}")
    print(f"Kept docs: {kept_docs}")
    print(f"Total characters kept: {total_chars_kept}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()