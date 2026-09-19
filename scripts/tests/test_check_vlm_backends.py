"""Tests for scripts/check_vlm_backends.py — the Step 3 capability gate.

The gate exists to catch one specific failure: a call that succeeds, silently
drops the image, and returns plausible Persian anyway. So the tests focus on the
two decisions that carry weight — did the control code come back, and is a
failure a capability fact or an access problem.
"""

import pytest

from scripts.tests.conftest import load_script

cvb = load_script("check_vlm_under_test", "scripts/check_vlm_backends.py")


# ── Control image ───────────────────────────────────────────────────────────


def test_control_image_renders_at_a_readable_size():
    img = cvb.make_control_image("KX7WQ3")
    assert img.size == (720, 200)
    assert img.mode == "RGB"


def test_control_image_actually_contains_dark_pixels():
    """A blank image would make every backend fail for the wrong reason."""
    import numpy as np

    arr = np.asarray(cvb.make_control_image("KX7WQ3").convert("L"))
    assert (arr < 128).sum() > 500, "the code must actually be drawn"


def test_control_alphabet_excludes_confusable_glyphs():
    """O/0, I/1, S/5, Z/2 would turn a correct read into an argument."""
    for ch in "O0I1S5Z2":
        assert ch not in cvb._CODE_ALPHABET


def test_control_code_space_is_large_enough_to_rule_out_guessing():
    assert len(cvb._CODE_ALPHABET) ** 6 > 10**8


# ── Answer normalization ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("KX7WQ3", "KX7WQ3"),
        ("kx7wq3", "KX7WQ3"),
        ('"KX7WQ3"', "KX7WQ3"),
        ("KX7WQ3.", "KX7WQ3"),
        ("K X 7 W Q 3", "KX7WQ3"),
        ("", ""),
    ],
)
def test_normalize_code_strips_decoration(raw, expected):
    assert cvb.normalize_code(raw) == expected


def test_normalize_code_keeps_surrounding_prose_visible():
    """A chatty answer still contains the code; the caller checks containment."""
    out = cvb.normalize_code("The characters are KX7WQ3.")
    assert "KX7WQ3" in out
    assert out != "KX7WQ3", "prose is preserved, so exact-match alone won't pass it"


# ── Verdict classification — the distinction that drives the next step ──────


def test_a_backend_that_read_the_code_is_usable():
    assert cvb.classify({"accepts_image": True}) == "usable"


def test_text_works_but_image_rejected_is_a_capability_fact():
    row = {"accepts_image": False, "text_only": {"ok": True, "text": "ok"}}
    assert cvb.classify(row) == "no_vision"


def test_text_also_failing_means_vision_is_untested_not_absent():
    """A 402 quota error says nothing about whether the model has vision.
    Reporting it as 'no vision' would be a claim the check has not earned."""
    row = {"accepts_image": False,
           "text_only": {"ok": False, "error": "APIStatusError: 402 quota exhausted"}}
    assert cvb.classify(row) == "unreachable"


def test_missing_text_probe_is_treated_as_unreachable():
    assert cvb.classify({"accepts_image": False}) == "unreachable"


# ── Crop selection ─────────────────────────────────────────────────────────


def test_explicit_crop_path_is_used_when_it_exists(tmp_path):
    crop = tmp_path / "c.jpg"
    crop.touch()
    assert cvb.pick_crop(crop) == crop


def test_missing_explicit_crop_returns_none(tmp_path):
    assert cvb.pick_crop(tmp_path / "nope.jpg") is None


def test_default_models_cover_the_three_vision_candidates():
    assert cvb.DEFAULT_MODELS == [
        "claude-opus-4-8",
        "claude-opus-5",
        "gpt-5.6-sol",
    ]
    assert "glm-5.3" not in cvb.DEFAULT_MODELS
    assert "deepseek-v4-flash" not in cvb.DEFAULT_MODELS


# ── Text-only probe ────────────────────────────────────────────────────────


class _FakeClient:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return self._behaviour(kwargs)


def _reply(text):
    class Msg:
        content = text

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]
        usage = None
        model = "m"

    return Resp()


def test_probe_text_only_reports_success():
    client = _FakeClient(lambda kw: _reply("ok"))
    assert cvb.probe_text_only(client, "m") == {"ok": True, "text": "ok"}


def test_probe_text_only_captures_the_error_type():
    def boom(kw):
        raise RuntimeError("quota exhausted")

    result = cvb.probe_text_only(_FakeClient(boom), "m")
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]
    assert "quota exhausted" in result["error"]


def test_probe_text_only_sends_no_image():
    seen = {}

    def capture(kw):
        seen.update(kw)
        return _reply("ok")

    cvb.probe_text_only(_FakeClient(capture), "m")
    assert seen["messages"] == [
        {"role": "user", "content": "Reply with the single word: ok"}
    ]


# ── ask() shapes one image per call ────────────────────────────────────────


def test_ask_sends_exactly_one_image_and_one_prompt():
    seen = {}

    def capture(kw):
        seen.update(kw)
        return _reply("KX7WQ3")

    img = cvb.make_control_image("KX7WQ3")
    out = cvb.ask(_FakeClient(capture), "m", "sys", "user", img)
    assert out["text"] == "KX7WQ3"
    assert seen["temperature"] == 0
    content = seen["messages"][1]["content"]
    assert sum(1 for c in content if c["type"] == "image_url") == 1
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_ask_records_latency_and_usage_slots():
    out = cvb.ask(_FakeClient(lambda kw: _reply("x")), "m", "s", "u",
                  cvb.make_control_image("AB34CD"))
    assert out["seconds"] >= 0
    assert set(out["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
