"""
Generates tests/fixtures/sample.pdf — a small synthetic 3-page "paper"
with clear sections, used by Stage 1 tests.

Run once:
    python tests/fixtures/generate_sample_pdf.py
"""
import fitz
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "sample.pdf"


def generate():
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Abstract", fontsize=14)
    page1.insert_text(
        (72, 100),
        "This paper proposes a novel method for retrieval augmented generation.\n"
        "We show that our method improves faithfulness by 15 percent over baselines.",
        fontsize=11,
    )
    page1.insert_text((72, 160), "1. Introduction", fontsize=14)
    page1.insert_text(
        (72, 190),
        "Retrieval augmented generation combines retrieval with generation.\n"
        "Hallucination remains a key challenge in these systems.\n"
        "In this work we address this problem directly using a verification loop.",
        fontsize=11,
    )

    page2 = doc.new_page()
    page2.insert_text((72, 72), "2. Method", fontsize=14)
    page2.insert_text(
        (72, 100),
        "Our method uses a two-stage pipeline: generation followed by verification.\n"
        "The verifier checks each claim against retrieved source chunks.\n"
        "If a claim cannot be verified, it is flagged as unsupported.",
        fontsize=11,
    )
    page2.insert_text((72, 200), "3. Experiments", fontsize=14)
    page2.insert_text(
        (72, 230),
        "We evaluate on a dataset of 50 research papers with hand labeled Q&A pairs.\n"
        "Our faithfulness score improved from 0.72 to 0.91 after adding verification.",
        fontsize=11,
    )

    page3 = doc.new_page()
    page3.insert_text((72, 72), "References", fontsize=14)
    page3.insert_text(
        (72, 100),
        "[1] Smith et al. 2023. Retrieval methods.\n"
        "[2] Doe et al. 2024. Verification loops in RAG.",
        fontsize=11,
    )

    doc.save(str(OUTPUT_PATH))
    doc.close()
    print(f"Saved fixture PDF to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()