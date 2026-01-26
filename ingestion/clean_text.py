import re
from pathlib import Path

def clean_text(text: str) -> str:
  
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)

    # remove citations like (200 BCE), [1], etc. (light touch)
    text = re.sub(r"\[[0-9]+\]", "", text)

    return text.strip()


def clean_directory(input_dir: Path):
    for file in input_dir.glob("chapter_*.txt"):
        raw = file.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        file.write_text(cleaned, encoding="utf-8")
        print(f" Cleaned {file.name}")


if __name__ == "__main__":
    clean_directory(Path("data/history-of-ancient-and-early-medieval-india"))
    clean_directory(Path("data/india-after-gandhi"))

