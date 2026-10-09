from eval.new_gold_report import _pearson, _score_pairs


def test_score_pairs_perfect():
    r = _score_pairs(["سلام"], ["سلام"], n_boot=200)
    assert r["n"] == 1
    assert r["mean_sentence_similarity"] == 1.0
    assert r["cer"] == 0.0


def test_pearson_identity():
    xs = [0.1, 0.5, 0.9]
    assert abs(_pearson(xs, xs) - 1.0) < 1e-9
