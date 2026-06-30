"""End-to-end pipeline smoke test (layout + mocked VLM + export)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _make_exam_page(path: Path) -> None:
    """Create a synthetic exam-like page with text lines and a table grid."""
    h, w = 1200, 900
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Title band
    cv2.rectangle(img, (200, 80), (700, 140), (0, 0, 0), 2)
    cv2.putText(img, "Sample Exam", (220, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Text lines
    for y in (200, 260, 320, 380):
        cv2.rectangle(img, (120, y), (780, y + 40), (0, 0, 0), 1)
        cv2.putText(img, "Question text line", (140, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Table grid
    x0, y0, tw, th = 120, 450, 360, 220
    rows, cols = 4, 3
    for r in range(rows + 1):
        y = y0 + r * (th // rows)
        cv2.line(img, (x0, y), (x0 + tw, y), (0, 0, 0), 2)
    for c in range(cols + 1):
        x = x0 + c * (tw // cols)
        cv2.line(img, (x, y0), (x, y0 + th), (0, 0, 0), 2)

    # Figure box
    cv2.rectangle(img, (520, 450), (780, 670), (0, 0, 0), 2)
    cv2.circle(img, (650, 560), 60, (0, 0, 0), 2)

    cv2.imwrite(str(path), img)


class PipelineIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_path = ROOT / "test_fixtures" / "sample_exam.png"
        cls.image_path.parent.mkdir(exist_ok=True)
        _make_exam_page(cls.image_path)

    def test_layout_detects_blocks(self):
        from pipeline.layout import detect_blocks

        bgr = cv2.imread(str(self.image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blocks = detect_blocks(rgb)
        self.assertGreater(len(blocks), 0, "layout should detect at least one block")
        for block in blocks:
            self.assertIn("type", block)
            self.assertIn("bbox", block)
            self.assertEqual(len(block["bbox"]), 4)

    def test_layout_fallback_on_blank_page(self):
        from pipeline.layout import detect_blocks

        white = np.full((800, 600, 3), 255, dtype=np.uint8)
        blocks = detect_blocks(white)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[0]["bbox"], [0, 0, 600, 800])

    @patch("pipeline.ocr.extract_text_with_vlm")
    def test_invalid_api_key_falls_back_to_crop(self, mock_vlm):
        from pipeline.export import export_to_word
        from pipeline.layout import detect_blocks
        from pipeline.ocr import run_ocr

        mock_vlm.side_effect = AssertionError("VLM should not be called with invalid key")

        bgr = cv2.imread(str(self.image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blocks = detect_blocks(rgb)

        with patch.dict("os.environ", {"VLM_API_KEY": "your_api_key_here"}, clear=False):
            blocks = run_ocr(rgb, blocks)

        self.assertGreater(
            sum(1 for b in blocks if b.get("crop_image") is not None),
            0,
            "invalid API key should fall back to crop images",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fallback.docx"
            export_to_word(blocks, str(out))
            self.assertTrue(out.exists())

    @patch("pipeline.ocr.extract_text_with_vlm")
    def test_full_pipeline_with_mock_vlm(self, mock_vlm):
        from pipeline.export import export_to_word
        from pipeline.layout import detect_blocks
        from pipeline.ocr import run_ocr

        mock_vlm.return_value = "متن نمونه آزمون"

        bgr = cv2.imread(str(self.image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blocks = detect_blocks(rgb)
        self.assertGreater(len(blocks), 0)

        blocks = run_ocr(rgb, blocks)
        has_text = any(b.get("text") for b in blocks)
        has_crop = any(b.get("crop_image") is not None for b in blocks)
        self.assertTrue(has_text or has_crop, "pipeline should produce text or crops")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.docx"
            export_to_word(blocks, str(out))
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_export_rtl_document(self):
        from pipeline.export import export_to_word
        from PIL import Image

        blocks = [
            {"type": "text", "text": "سلام دنیا", "illegible": False, "crop_image": None},
            {
                "type": "figure",
                "text": None,
                "crop_image": Image.new("RGB", (200, 100), color=(200, 200, 200)),
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rtl.docx"
            export_to_word(blocks, str(out))
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
