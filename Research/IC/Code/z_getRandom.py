import os
import random
import base64

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences_selected")
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
                            phrases.append((fname, s))
            except Exception:
                continue
    print(f"Read {len(phrases)} phrases from: {folder}")
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
        for item in phrases:
            if isinstance(item, tuple) and len(item) == 2:
                fname, phrase = item
            else:
                fname, phrase = ("", str(item))
            enc = base64.b64encode(fname.encode("utf-8")).decode("ascii")
            f.write(enc + " | " + phrase + "\n")

if __name__ == "__main__":
    phrases = read_phrases_from_folder(INPUT_FOLDER)
    selected = sample_phrases(phrases, 500)
    write_phrases_to_file(selected, OUTPUT_FILE)
    print(f"Wrote {len(selected)} phrases to: {OUTPUT_FILE}")
