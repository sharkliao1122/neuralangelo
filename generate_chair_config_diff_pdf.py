from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(r"C:\research\neuralangelo_test")
OFFICIAL = ROOT / r"outputs\chair\chair_from_nerf2mesh_official\config.yaml"
BETTER = ROOT / r"outputs\chair\chair_from_nerf2mesh_better\config.yaml"
OUTPUT = ROOT / "chair_official_vs_better_parameter_diff_table.pdf"


def read_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def extract_value(lines, key, parent=None):
    if parent is None:
        needle = f"{key}:"
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(needle):
                return stripped.split(":", 1)[1].strip()
        raise KeyError(f"Could not find key {key}")

    parent_indent = None
    in_parent = False
    parent_needle = f"{parent}:"
    key_needle = f"{key}:"

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped == parent_needle:
            in_parent = True
            parent_indent = indent
            continue
        if in_parent:
            if stripped and indent <= parent_indent:
                in_parent = False
                parent_indent = None
            elif stripped.startswith(key_needle):
                return stripped.split(":", 1)[1].strip()

    raise KeyError(f"Could not find key {key} under parent {parent}")


def build_rows():
    official_lines = read_lines(OFFICIAL)
    better_lines = read_lines(BETTER)

    rows = [
        (
            "checkpoint.save_iter",
            extract_value(official_lines, "save_iter", parent="checkpoint"),
            extract_value(better_lines, "save_iter", parent="checkpoint"),
            "Better saves checkpoints less often.",
        ),
        (
            "trainer.amp_config.enabled",
            extract_value(official_lines, "enabled", parent="amp_config"),
            extract_value(better_lines, "enabled", parent="amp_config"),
            "Better enables AMP mixed precision for speed and VRAM savings.",
        ),
        (
            "validation_iter",
            extract_value(official_lines, "validation_iter"),
            extract_value(better_lines, "validation_iter"),
            "Better runs validation half as often.",
        ),
        (
            "logdir",
            extract_value(official_lines, "logdir"),
            extract_value(better_lines, "logdir"),
            "Different run name and log path; this does not change model math.",
        ),
    ]

    return rows


def make_pdf():
    rows = build_rows()

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.5,
        leading=10,
    )
    title = ParagraphStyle(
        "TitleSmall",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.black,
        spaceAfter=8,
    )

    elements = [
        Paragraph("Neuralangelo Training Config Comparison", title),
        Paragraph(
            f"Official: {OFFICIAL.parent}<br/>Better: {BETTER.parent}",
            small,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Only four saved config entries differ between these two runs. "
            "The dataset path, image sizes, batch size, model structure, render sampling, "
            "optimizer, scheduler, and loss weights are the same.",
            body,
        ),
        Spacer(1, 3 * mm),
    ]

    header = [
        Paragraph("<b>Parameter</b>", small),
        Paragraph("<b>Official</b>", small),
        Paragraph("<b>Better</b>", small),
        Paragraph("<b>Meaning</b>", small),
    ]
    table_data = [header]
    for param, official, better, meaning in rows:
        table_data.append(
            [
                Paragraph(param, small),
                Paragraph(official, small),
                Paragraph(better, small),
                Paragraph(meaning, small),
            ]
        )

    col_widths = [60 * mm, 55 * mm, 70 * mm, 88 * mm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e7f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.Color(0.96, 0.98, 1)]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 4 * mm))
    elements.append(
        Paragraph(
            "Key readout: the main behavior difference is that the better run enables AMP, "
            "and it also validates and saves checkpoints less frequently.",
            body,
        )
    )

    doc.build(elements)


if __name__ == "__main__":
    make_pdf()
