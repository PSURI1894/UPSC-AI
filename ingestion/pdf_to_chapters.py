import fitz  # PyMuPDF
from pathlib import Path

BOOKS = {
    "history-of-ancient-and-early-medieval-india": {
        "pdf_path": "raw_books/history-of-ancient-and-early-medieval-india.pdf",
        "output_dir": "data/history-of-ancient-and-early-medieval-india",
        "toc": [
            (1, 20, "Introduction: Ideas of the Early Indian Past"),
            (21, 60, "Understanding Literary and Archaeological Sources"),
            (61, 110, "Hunter-Gatherers of the Palaeolithic and Mesolithic Ages"),
            (111, 170, "Neolithic and Chalcolithic Villages"),
            (171, 260, "The Harappan Civilization"),
            (261, 340, "Cultural Transitions: 2000–600 BCE"),
            (341, 420, "Cities, Kings, and Renunciants"),
            (421, 480, "The Maurya Empire"),
            (481, 520, "Interaction and Innovation"),
            (521, 600, "Aesthetics and Empire"),
            (601, 680, "Emerging Regional Configurations")
        ]
    },

    "india-after-gandhi": {
        "pdf_path": "raw_books/india-after-gandhi.pdf",
        "output_dir": "data/india-after-gandhi",
        "toc": [
            (1, 30, "Prologue: Unnatural Nation"),
            (31, 60, "Freedom and Parricide"),
            (61, 90, "The Logic of Division"),
            (91, 120, "Apples in the Basket"),
            (121, 150, "A Valley Bloody and Beautiful"),
            (151, 180, "Refugees and the Republic"),
            (181, 210, "Ideas of India"),
            (211, 250, "The Biggest Gamble in History"),
            (251, 300, "Redrawing the Map"),
            (301, 350, "Securing Kashmir"),
            (351, 400, "The Southern Challenge"),
            (401, 450, "The Experience of Defeat"),
            (451, 500, "The Rise of Populism"),
            (501, 550, "Democracy in Disarray"),
            (551, 590, "Why India Survives")
        ]
    }
}


def extract_book(book_key, config):
    pdf_path = config["pdf_path"]
    output_dir = Path(config["output_dir"])
    toc = config["toc"]

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    for idx, (start, end, title) in enumerate(toc, start=1):
        chapter_text = ""

        for page in range(start - 1, end):
            chapter_text += doc[page].get_text()

        chapter_file = output_dir / f"chapter_{idx:02d}.txt"

        with open(chapter_file, "w", encoding="utf-8") as f:
            f.write(f"BOOK: {book_key}\n")
            f.write(f"CHAPTER: {title}\n")
            f.write(f"PAGES: {start}-{end}\n\n")
            f.write(chapter_text)

        print(f" {book_key} | Chapter {idx:02d} extracted")




if __name__ == "__main__":
    for book_key, config in BOOKS.items():
        print(f"\n Processing: {book_key}")
        extract_book(book_key, config)

    print("\n All books processed successfully.")
