#!/usr/bin/env python3
"""Generate a Persian RTL Word report with correct and failed OCR samples."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "samples"
OUT = ROOT / "docs" / "REPORT_correct_and_fail_samples.docx"
FONT = "Vazir"

GOOD = [
    {
        "title": "تطابق کامل — واژهٔ حسابداری",
        "image": "01_exact_hesab.jpg",
        "gold": "حساب دریافتی",
        "pred": "حساب دریافتی",
        "cer": "۰٫۰۰۰",
        "note": "مدل کل خط را بدون هیچ خطای کاراکتری خوانده است.",
    },
    {
        "title": "تطابق کامل — عنوان ستون جدول",
        "image": "15_exact_sharh.jpg",
        "gold": "شرح",
        "pred": "شرح",
        "cer": "۰٫۰۰۰",
        "note": "کلمهٔ کوتاه ولی خوانا؛ یکی از موارد نادر تطابق کامل روی مجموعهٔ طلایی.",
    },
    {
        "title": "خط بلند مشاوره — خطای جزئی نقطه‌گذاری",
        "image": "02_good_ellis.jpg",
        "gold": "الیس: در درمان رفتاری از شیوه تعیین تکلیف استفاده میکند",
        "pred": "الیس در درمان رفتاری از شیوه تعیین تکلیف استفاده میکند.",
        "cer": "۰٫۰۳۶",
        "note": "جمله کامل و درست است؛ تفاوت عمدتاً در دونقطه و نقطهٔ پایان است.",
    },
    {
        "title": "خط بلند مشاوره — یک حرف اول اشتباه",
        "image": "16_good_zaruri.jpg",
        "gold": "ضروری برای رفع مشکلات آنان",
        "pred": "مزوری برای رفع مشکلات آنان",
        "cer": "۰٫۰۷۷",
        "note": "ساختار جمله حفظ شده؛ فقط حرف اول (ض ← م) عوض شده است.",
    },
    {
        "title": "خط بلند مشاوره — چند حرف نزدیک به هم",
        "image": "03_good_pieri.jpg",
        "gold": "متوجه این تحلیل می‌شوند و پیری را روند ناگهانی و غیر پیوسته می‌دانند",
        "pred": "متوجه این تکلیل می‌سوند و پیرک را روند نالحاتی و غیر پیوسته می‌دانند",
        "cer": "۰٫۰۸۸",
        "note": "خواننده جمله را می‌فهمد؛ چند حرف شبیه (تحلیل←تکلیل، پیری←پیرک) جابه‌جا شده‌اند.",
    },
    {
        "title": "خط بلند مشاوره — معنی حفظ‌شده",
        "image": "17_good_info.jpg",
        "gold": "ارائه‌ی اطلاعات در زمینه‌های آموزشی، شغلی یا اجتماعی",
        "pred": "ارائه می اطلاعات در زمینه های آموزشی، شغلی، را اجتماعی",
        "cer": "۰٫۰۹۶",
        "note": "تقریباً همهٔ واژه‌ها درست‌اند؛ چند حرف اضافه/جابه‌جا و «یا» به «را» تبدیل شده است.",
    },
]

FAIL = [
    {
        "title": "کراپ تک‌حرف — حدس ساختگی",
        "image": "07_fail_short.jpg",
        "gold": "ص",
        "pred": "میک",
        "cer": "۱٫۰۰۰",
        "note": "ورودی یک حرف است؛ مدل چند حرف ساختگی تولید کرده. کراپ‌های خیلی کوتاه دقت کل را پایین می‌کشند.",
    },
    {
        "title": "توهم متن اضافه",
        "image": "08_fail_hallucination.jpg",
        "gold": "مبانی مشاوره و راهنمایی: فصل ۱۱",
        "pred": "میانی مشاوره دور اهلنای فصل۱۱ تاربرد سارح در میدانهای جنگ، برده‌تر است.",
        "cer": "۱٫۰۰۰",
        "note": "بخشی از عنوان درست است؛ ادامهٔ جمله از خط کناری صفحه به خروجی چسبیده یا ساختگی است.",
    },
    {
        "title": "واژهٔ کوتاه حسابداری با خط‌چین",
        "image": "09_fail_bedehkar.jpg",
        "gold": "بدهکار ۸ ماه",
        "pred": "به بحثیار ۱۸۰۶۸",
        "cer": "۰٫۹۱۷",
        "note": "خط‌چین دفتر و حروف به‌هم‌چسبیده باعث شده «بدهکار» به متن بی‌معنی تبدیل شود.",
    },
    {
        "title": "فرمول ریاضی تکه‌تکه",
        "image": "10_fail_math.jpg",
        "gold": "ax+b x>1",
        "pred": "121",
        "cer": "۰٫۸۷۵",
        "note": "نمادها و حروف لاتین دست‌نویس به رقم تقلیل یافته‌اند. جزوهٔ ریاضی سخت‌ترین سند مجموعه است.",
    },
    {
        "title": "فرمول چندبخشی با کادر",
        "image": "11_fail_formula.jpg",
        "gold": "y=f(a) ; (0) = (y=0)",
        "pred": "٥ = ٥",
        "cer": "۰٫۸۵۰",
        "note": "مدل تقریباً کل عبارت را از دست داده و فقط دو رقم عربی برگردانده است.",
    },
    {
        "title": "خط بلند دفتر کل با عدد",
        "image": "12_fail_long_acc.jpg",
        "gold": "درمانده گیری باید در آخر فقط یک ردیف یا بدهکار یا بستانکار باقی بماند در دفتر کل اگر بعد از",
        "pred": "1075000,000 1075000,000 در مالندیکه باید در آخر فقط بدرین نیکی بدرانه",
        "cer": "۰٫۷۶۹",
        "note": "اعداد ستون دفتر به متن چسبیده‌اند و نیمهٔ فارسی جمله هم خراب شده است.",
    },
    {
        "title": "عبارت کوتاه حسابداری (خرید)",
        "image": "13_fail_kharid.jpg",
        "gold": "تحقیقات نقدی خرید",
        "pred": "که عیهان نعیک حسر",
        "cer": "۰٫۷۶۵",
        "note": "دست‌نویس فشرده؛ خروجی مدل تقریباً هیچ واژه‌ای از برچسب طلایی را حفظ نکرده است.",
    },
    {
        "title": "پیکان و واژه‌های کوتاه حسابداری",
        "image": "14_fail_sod.jpg",
        "gold": "سود سرمایه",
        "pred": "کنور ۱ سرما",
        "cer": "۰٫۷۰۰",
        "note": "پیکان و خط‌چین بین دو واژه، تشخیص را به هم زده؛ «سود» کلاً از بین رفته است.",
    },
]


def _set_run_font(run, *, size=12, bold=False, color=None, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    r_pr = run._element.get_or_add_rPr()
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), font)
    r_fonts.set(qn("w:hAnsi"), font)
    r_fonts.set(qn("w:cs"), font)
    r_fonts.set(qn("w:eastAsia"), font)
    existing = r_pr.find(qn("w:rFonts"))
    if existing is not None:
        r_pr.remove(existing)
    r_pr.insert(0, r_fonts)
    if bold:
        b_cs = OxmlElement("w:bCs")
        r_pr.append(b_cs)


def _rtl_para(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    if p_pr.find(qn("w:bidi")) is None:
        p_pr.append(OxmlElement("w:bidi"))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.35


def add_p(doc, text, *, size=12, bold=False, color=None, space_after=6):
    p = doc.add_paragraph()
    _rtl_para(p)
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    _set_run_font(run, size=size, bold=bold, color=color)
    return p


def shade_cell(cell, hex_color: str):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)


def set_cell_rtl(cell):
    for p in cell.paragraphs:
        _rtl_para(p)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def add_kv_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    table.autofit = True
    for i, (k, v) in enumerate(rows):
        table.cell(i, 0).text = k
        table.cell(i, 1).text = v
        shade_cell(table.cell(i, 0), "F0F0F0")
        for j in range(2):
            cell = table.cell(i, j)
            set_cell_rtl(cell)
            for p in cell.paragraphs:
                for run in p.runs:
                    _set_run_font(run, size=11, bold=(j == 0))
    tbl = table._tbl
    tbl_pr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    bidi = OxmlElement("w:bidiVisual")
    bidi.set(qn("w:val"), "1")
    tbl_pr.append(bidi)
    for row in table.rows:
        row.cells[0].width = Cm(3.2)
        row.cells[1].width = Cm(13.5)
    return table


def add_image(doc, path: Path):
    with Image.open(path) as im:
        w, h = im.size
    max_w = 15.5
    width_cm = min(max_w, max(4.5, w / 38))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))


def add_sample(doc, item, kind: str):
    color = RGBColor(0x1B, 0x6B, 0x3A) if kind == "good" else RGBColor(0x9B, 0x1C, 0x1C)
    add_p(doc, item["title"], size=13, bold=True, color=color, space_after=4)
    add_image(doc, SAMPLES / item["image"])
    add_kv_table(
        doc,
        [
            ("متن طلایی (انسان)", item["gold"]),
            ("خروجی مدل", item["pred"]),
            ("CER", item["cer"]),
        ],
    )
    add_p(doc, item["note"], size=11, color=RGBColor(0x33, 0x33, 0x33), space_after=14)


def configure_doc(doc: Document):
    section = doc.sections[0]
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(1.6)
    section.right_margin = Cm(1.6)
    sect_pr = section._sectPr
    if sect_pr.find(qn("w:bidi")) is None:
        sect_pr.append(OxmlElement("w:bidi"))
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_pr = normal._element.get_or_add_pPr()
    if p_pr.find(qn("w:bidi")) is None:
        p_pr.append(OxmlElement("w:bidi"))


def main():
    doc = Document()
    configure_doc(doc)

    add_p(doc, "گزارش خلاصهٔ پروژهٔ OCR دست‌نویس فارسی", size=20, bold=True, space_after=4)
    add_p(
        doc,
        "تاریخ: ۱۹ سپتامبر ۲۰۲۶  |  معیار: CER روی مجموعهٔ طلایی دستی (۱۶۸ خط دارای متن)",
        size=11,
        color=RGBColor(0x44, 0x44, 0x44),
        space_after=12,
    )

    add_p(doc, "۱. موضوع پروژه", size=16, bold=True, space_after=8)
    add_p(
        doc,
        "این سامانه برگه‌ها و جزوه‌های دست‌نویس فارسی (اسکن یا عکس موبایل) را به خطوط جدا می‌کند و "
        "متن هر خط را با مدل بینایی می‌خواند. هدف عملی، تبدیل صفحهٔ آزمون/جزوه به فایل Word است: "
        "متن چاپی با OCR، و جدول/شکل/دست‌نویس نامرتب به‌صورت تصویر کراپ‌شده.",
    )
    add_p(
        doc,
        "مسیر پردازش: بارگذاری فایل ← پیش‌پردازش تصویر ← تشخیص بلوک/خط ← OCR فارسی ← خروجی Word راست‌به‌چپ.",
    )

    add_p(doc, "۲. داده و روش آزمایش", size=16, bold=True, space_after=8)
    add_p(
        doc,
        "از ۹۴۵ صفحه، ۲۲٬۲۵۰ کراپ سطح‌خط ساخته شد. تقسیم train/val/test در سطح سند است و نشتی بین splitها صفر است. "
        "مجموعهٔ طلایی ۲۴۰ خط از split تست است که انسان رونویسی کرده؛ ۱۶۸ خط متن دارند و ۷۲ خط ناخوانا از نمره حذف شده‌اند. "
        "هر مدل همان کراپ را جداگانه خوانده است. عدد گزارش‌شده توافق دو مدل نیست؛ مقایسه با انسان است.",
    )
    add_p(
        doc,
        "CER = (جایگزینی + حذف + درج) ÷ طول متن طلایی. دقت کاراکتری = ۱ − CER.",
        size=11,
        color=RGBColor(0x33, 0x33, 0x33),
    )

    add_p(doc, "۳. نتیجهٔ کمّی", size=16, bold=True, space_after=8)
    add_p(
        doc,
        "بهترین مدل محلی (Qwen3-VL-2B مخصوص تک‌خط فارسی/عربی) به دقت کاراکتری ۶۲٫۳٪ رسید (CER = ۰٫۳۷۷). "
        "مدل قبلی محلی ۷ میلیارد پارامتری ۵۴٫۶٪ بود. نرخ تطابق کامل خط حدود ۳٫۶٪ است.",
        bold=True,
    )

    cmp = doc.add_table(rows=4, cols=4)
    cmp.alignment = WD_TABLE_ALIGNMENT.RIGHT
    headers = ["مدل", "CER", "دقت کاراکتری", "WER"]
    data = [
        ["Qwen3-VL-2B Persian line-OCR (فعلی)", "۰٫۳۷۷", "۶۲٫۳٪", "۰٫۷۴"],
        ["Qwen2.5-VL-7B محلی (قبلی)", "۰٫۴۵۴", "۵۴٫۶٪", "۰٫۷۷"],
        ["تشخیص خط Surya (بدون مدل خطی دست‌نویس)", "—", "برای دست‌نویس قابل استفاده نبود", "—"],
    ]
    for j, h in enumerate(headers):
        cmp.cell(0, j).text = h
        shade_cell(cmp.cell(0, j), "E8E8E8")
    for i, row in enumerate(data, start=1):
        for j, val in enumerate(row):
            cmp.cell(i, j).text = val
    for row in cmp.rows:
        for cell in row.cells:
            set_cell_rtl(cell)
            for p in cell.paragraphs:
                for run in p.runs:
                    _set_run_font(run, size=10, bold=(row == cmp.rows[0]))
    tbl_pr = cmp._tbl.tblPr
    bidi = OxmlElement("w:bidiVisual")
    bidi.set(qn("w:val"), "1")
    tbl_pr.append(bidi)

    add_p(doc, "", space_after=4)
    add_p(
        doc,
        "علت اصلی عدد ۶۲٪، کراپ‌های خیلی کوتاه است. روی خطوط بلندتر از ۲۰ کاراکتر دقت حدود ۶۸٪ است؛ "
        "روی خطوط تا ۴ کاراکتر حدود ۱۹٪.",
        space_after=4,
    )

    len_t = doc.add_table(rows=4, cols=3)
    for j, h in enumerate(["طول برچسب طلایی", "دقت کاراکتری", "تعداد خط"]):
        len_t.cell(0, j).text = h
        shade_cell(len_t.cell(0, j), "E8E8E8")
    for i, row in enumerate(
        [
            ["تا ۴ کاراکتر", "۱۸٫۹٪", "۲۳"],
            ["۵ تا ۲۰ کاراکتر", "۳۶٫۱٪", "۷۳"],
            ["بیش از ۲۰ کاراکتر", "۶۷٫۷٪", "۷۲"],
        ],
        start=1,
    ):
        for j, val in enumerate(row):
            len_t.cell(i, j).text = val
    for row in len_t.rows:
        for cell in row.cells:
            set_cell_rtl(cell)
            for p in cell.paragraphs:
                for run in p.runs:
                    _set_run_font(run, size=10, bold=(row == len_t.rows[0]))
    bidi2 = OxmlElement("w:bidiVisual")
    bidi2.set(qn("w:val"), "1")
    len_t._tbl.tblPr.append(bidi2)

    add_p(doc, "", space_after=6)
    add_p(doc, "۴. نمونه‌های درست تشخیص‌داده‌شده", size=16, bold=True, space_after=6)
    add_p(
        doc,
        "شش نمونهٔ زیر از مجموعهٔ طلایی‌اند. تصویر همان ورودی مدل است. "
        "معیار «درست» این‌جا تطابق کامل یا CER کمتر از ۰٫۱۰ است (جمله خوانا و تقریباً مطابق انسان).",
        space_after=10,
    )
    for item in GOOD:
        add_sample(doc, item, "good")

    add_p(doc, "۵. نمونه‌های اشتباه تشخیص‌داده‌شده", size=16, bold=True, space_after=6)
    add_p(
        doc,
        "هشت نمونهٔ زیر همان مدل و همان مجموعهٔ طلایی هستند، با CER بالا (حدود ۰٫۷۰ تا ۱٫۰). "
        "الگوی خطا تکرارشونده است: کراپ کوتاه، توهم متن، فرمول ریاضی، و دفتر حسابداری با خط‌چین و عدد.",
        space_after=10,
    )
    for item in FAIL:
        add_sample(doc, item, "fail")

    add_p(doc, "۶. جمع‌بندی", size=16, bold=True, space_after=8)
    add_p(
        doc,
        "پیکرهٔ خطی و مجموعهٔ طلایی آماده است و عدد ۶۲٫۳٪ روی hold-out دستی قابل دفاع است. "
        "روی خطوط جمله‌دار فارسی مدل نزدیک ۶۸٪ کاراکتر را درست می‌خواند و چند نمونهٔ تطابق کامل هم وجود دارد. "
        "گلوگاه فعلی کراپ‌های کوتاه، فرمول ریاضی، و چسبیدن عدد/خط‌چین دفتر به متن است.",
    )
    add_p(
        doc,
        "قدم بعد پیشنهادی: فیلتر کراپ‌های خیلی کوتاه پیش از نمره‌دهی/استنتاج، سپس fine-tune روی همین خطوط دست‌نویس "
        "برای رسیدن به هدف میانی حدود ۷۹٪ دقت کاراکتری.",
        space_after=12,
    )
    add_p(
        doc,
        "منبع اعداد: dataset/report_qwen3-vl-2b-persian-arabic-ocr-v1.0.json و "
        "dataset/pred_qwen3-vl-2b-persian-arabic-ocr-v1.0.jsonl روی ۱۶۸ خط دارای متن.",
        size=10,
        color=RGBColor(0x55, 0x55, 0x55),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
