"""
Generates tests/fixtures/notes_sample.pdf — a small synthetic 2-page
"textbook notes" document with chapter/section-style headings, to
verify chunking generalizes beyond research-paper structure.
"""
import fitz
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "notes_sample.pdf"


def generate():
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Chapter 3", fontsize=14)
    page1.insert_text((72, 100), "Cellular Respiration", fontsize=13)
    page1.insert_text((72, 130),
        "Cellular respiration is the process by which cells break down glucose\n"
        "to produce energy in the form of ATP.", fontsize=11)
    page1.insert_text((72, 190), "Key Terms", fontsize=13)
    page1.insert_text((72, 220),
        "Mitochondria: the organelle where respiration occurs.\n"
        "ATP: adenosine triphosphate, the energy currency of the cell.", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Photosynthesis", fontsize=13)
    page2.insert_text((72, 100),
        "Photosynthesis converts light energy into chemical energy stored in glucose.\n"
        "It occurs in the chloroplasts of plant cells.", fontsize=11)
    page2.insert_text((72, 160), "Review Questions", fontsize=13)
    page2.insert_text((72, 190),
        "1. What is the main product of cellular respiration?\n"
        "2. Where does photosynthesis occur in a plant cell?", fontsize=11)

    doc.save(str(OUTPUT_PATH))
    doc.close()
    print(f"Saved notes fixture PDF to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()