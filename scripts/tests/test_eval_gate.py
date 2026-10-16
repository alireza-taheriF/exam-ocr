import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eval_gate", ROOT / "scripts" / "eval_gate.py")
eval_gate = importlib.util.module_from_spec(spec)
sys.modules["eval_gate"] = eval_gate
spec.loader.exec_module(eval_gate)


def _gold(doc, text, i):
    return {"image_path": f"c{i}.jpg", "source_document": doc, "text": text}


def _pred(text, i):
    return {"image_path": f"c{i}.jpg", "text": text, "seconds": 1.0}


def test_line_counts_clamp_and_normalize():
    assert eval_gate.line_counts("سلام", "سلام") == (0, 4, 0, 1)
    cd, cl, _, _ = eval_gate.line_counts("ab", "x" * 50)
    assert (cd, cl) == (2, 2)  # clamped to the reference length


def test_report_scores_legible_and_counts_routing():
    gold = [_gold("d1", "سلام دنیا", 0), _gold("d1", "", 1), _gold("d2", "کتاب", 2)]
    preds = {"c0.jpg": _pred("سلام دنیا", 0), "c1.jpg": _pred("", 1), "c2.jpg": _pred("", 2)}
    report, scored = eval_gate.build_report(gold, preds)
    assert report["lines"] == 2
    assert report["by_document"]["d1"]["cer"] == 0.0
    assert report["by_document"]["d2"]["cer"] == 1.0
    assert report["routing"] == {"legible_lines_returned_empty": 1, "illegible_lines": 1,
                                 "illegible_lines_returned_empty": 1, "illegible_lines_with_text": 0}
    assert report["legible_without_routed"]["lines"] == 1
    assert set(scored) == {"c0.jpg", "c2.jpg"}


def test_length_buckets():
    assert [eval_gate.length_bucket(n) for n in (1, 4, 5, 20, 21)] == ["<=4", "<=4", "5-20", "5-20", ">20"]


def test_paired_delta_negative_when_challenger_better():
    a = {str(i): {"cd": 5, "cl": 10} for i in range(50)}
    b = {str(i): {"cd": 2, "cl": 10} for i in range(50)}
    d = eval_gate.paired_delta_ci(a, b)
    assert d["delta_cer"] == -0.3 and d["ci95"][1] < 0
