import os
import random

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences")
OUTPUT_FILE = os.path.join(SCRIPT_FOLDER, "randomPhrases.txt")

def read_phrases_from_folder(folder: str) -> list:
    phrases = []
    if not os.path.isdir(folder):
        return phrases
    for root, _, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            phrases.append(s)
            except Exception:
                continue
    return phrases

def sample_phrases(phrases: list, k: int) -> list:
    if len(phrases) <= k:
        random.seed()
        out = phrases[:]
        random.shuffle(out)
        return out
    return random.sample(phrases, k)

def write_phrases_to_file(phrases: list, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for p in phrases:
            f.write(p + "\n")

if __name__ == "__main__":
    phrases = read_phrases_from_folder(INPUT_FOLDER)
    selected = sample_phrases(phrases, 500)
    write_phrases_to_file(selected, OUTPUT_FILE)
    print(f"Wrote {len(selected)} phrases to: {OUTPUT_FILE}")
