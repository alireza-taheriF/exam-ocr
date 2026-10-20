from unittest.mock import patch

import pipeline.ocr as ocr


def test_remote_is_off_by_default(monkeypatch):
    monkeypatch.delenv("OCR_ALLOW_REMOTE", raising=False)
    assert "OCR_ALLOW_REMOTE" in ocr._validate_vlm_config()
    assert ocr.get_ocr_status()["engine"] == "local"


def test_remote_client_refused_without_opt_in(monkeypatch):
    monkeypatch.delenv("OCR_ALLOW_REMOTE", raising=False)
    with patch("pipeline.ocr.create_openai_client") as client:
        try:
            ocr.extract_text_with_vlm("x")
        except ValueError:
            pass
        client.assert_not_called()
