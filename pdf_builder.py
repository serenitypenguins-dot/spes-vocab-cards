from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color

GRADE_LABELS = {
    "k": "Kindergarten",
    "1": "1st Grade",
    "2": "2nd Grade",
    "3": "3rd Grade",
    "4": "4th Grade",
    "5": "5th Grade",
}

CARDS_PER_PAGE = 3
IMG_SIZE = 150


def _font_size_for_word(word: str) -> float:
    length = len(word)
    if length <= 6:
        return 48
    elif length <= 10:
        return 40
    elif length <= 14:
        return 32
    elif length <= 20:
        return 26
    else:
        return 20


def build_pdf(
    words: list[str],
    image_paths: dict[str, Path | None],
    grade: str,
    output_path: Path,
):
    page_w, page_h = letter
    margin = 36
    header_h = 24
    usable_h = page_h - 2 * margin - header_h
    usable_w = page_w - 2 * margin
    cell_h = usable_h / CARDS_PER_PAGE

    c = canvas.Canvas(str(output_path), pagesize=letter)
    grade_label = GRADE_LABELS.get(grade, "Kindergarten")
    header_text = f"Vocabulary Cards — Grade: {grade_label}"

    for i, word in enumerate(words):
        slot = i % CARDS_PER_PAGE
        if slot == 0:
            if i > 0:
                c.showPage()
            c.setFont("Helvetica", 9)
            c.setFillColor(Color(0.5, 0.5, 0.5))
            c.drawString(margin, page_h - margin + 5, header_text)
            c.setFillColor(Color(0, 0, 0))

        x = margin
        cell_top = page_h - margin - header_h - slot * cell_h
        cell_bottom = cell_top - cell_h
        center_y = (cell_top + cell_bottom) / 2

        # Card border
        c.setStrokeColor(Color(0.85, 0.85, 0.85))
        c.setLineWidth(0.5)
        c.rect(x + 4, cell_bottom + 4, usable_w - 8, cell_h - 8)

        # Image — centered on center_y
        img_x = x + 20
        img_y = center_y - IMG_SIZE / 2
        img_path = image_paths.get(word)
        if img_path and img_path.exists():
            try:
                c.drawImage(
                    str(img_path), img_x, img_y, IMG_SIZE, IMG_SIZE,
                    preserveAspectRatio=False, mask="auto",
                )
            except Exception:
                c.setFillColor(Color(0.93, 0.93, 0.93))
                c.rect(img_x, img_y, IMG_SIZE, IMG_SIZE, fill=1, stroke=0)
                c.setFillColor(Color(0, 0, 0))
        else:
            c.setFillColor(Color(0.93, 0.93, 0.93))
            c.rect(img_x, img_y, IMG_SIZE, IMG_SIZE, fill=1, stroke=0)
            c.setFillColor(Color(0, 0, 0))

        # Text — baseline positioned so cap-height center aligns with center_y
        # Cap height ≈ 70% of font size for Helvetica Bold
        font_size = _font_size_for_word(word)
        cap_height = font_size * 0.70
        text_baseline = center_y - cap_height / 2

        c.setFont("Helvetica-Bold", font_size)
        text_x = img_x + IMG_SIZE + 24
        c.drawString(text_x, text_baseline, word)

    c.showPage()
    c.save()
