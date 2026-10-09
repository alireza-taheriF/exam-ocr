from labeling.validate import find_invalid_chars, validate_correction


def test_whitelist_ok():
    assert find_invalid_chars("سلام ۱۲۳ Date:") == []
    assert validate_correction("متن فارسی", action="edited")["ok"] is True


def test_empty_warns_unless_illegible():
    w = validate_correction("", action="edited")
    assert w["ok"] is False
    assert "empty_text_without_illegible" in w["warnings"]
    assert validate_correction("", action="marked_illegible")["ok"] is True


def test_bad_char():
    bad = find_invalid_chars("سلام😀")
    assert "😀" in bad
