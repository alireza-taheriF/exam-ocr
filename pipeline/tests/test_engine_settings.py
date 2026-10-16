from PIL import Image

from pipeline.engines import QwenSettings, is_repetition_loop, preprocess_crop


def test_defaults_are_the_measured_baseline():
    s = QwenSettings()
    assert s.prompt == "Transcribe the Persian text in this image. Output only the text."
    assert (s.max_new_tokens, s.px_per_token, s.repetition_penalty) == (96, 0.0, 1.0)
    assert (s.min_height, s.clahe, s.route_min_width, s.drop_loops) == (0, False, 0, False)


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("QWEN_REPETITION_PENALTY", "1.05")
    monkeypatch.setenv("OCR_CLAHE", "1")
    monkeypatch.setenv("OCR_ROUTE_MIN_WIDTH", "120")
    s = QwenSettings.from_env()
    assert s.repetition_penalty == 1.05 and s.clahe and s.route_min_width == 120


def test_repetition_loops_are_detected():
    assert is_repetition_loop("٥" * 40)
    assert is_repetition_loop("۱۶" + "۲۶" * 30)
    assert is_repetition_loop("1" + "0" * 80)
    assert is_repetition_loop("گروه " + "ها" * 20)


def test_real_lines_are_not_loops():
    assert not is_repetition_loop("دارایی‌هایی که در شرکت وجود دارد به سمت چپ معادله اساسی حسابداری بدهکار است.")
    assert not is_repetition_loop("۵۰/۰۰۰/۰۰۰")  # accounting figures: short, legitimate
    assert not is_repetition_loop("۱۰۰۰۰۰ ریال")
    assert not is_repetition_loop("")


def test_preprocess_upscales_short_crops_only():
    s = QwenSettings(min_height=64)
    small = preprocess_crop(Image.new("RGB", (200, 32), "white"), s)
    assert small.size == (400, 64)
    tall = Image.new("RGB", (200, 80), "white")
    assert preprocess_crop(tall, s).size == (200, 80)


def test_preprocess_clahe_keeps_size_and_mode():
    out = preprocess_crop(Image.new("RGB", (120, 40), (200, 200, 200)), QwenSettings(clahe=True))
    assert out.size == (120, 40) and out.mode == "RGB"
