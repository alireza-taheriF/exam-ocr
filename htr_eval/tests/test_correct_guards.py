from collections import Counter

from htr_eval.correct import apply_guards, key

LEX = {key(w) for w in ["مثلاً", "به", "مدرسه", "رفتم", "من", "خانه"]}


def run(orig, new):
    rej = Counter()
    return apply_guards(orig, new, LEX, rej), rej


def test_accepts_vocab_word_close_to_original():
    assert run("من مصلن به مدرسه رفتم", "من مثلاً به مدرسه رفتم")[0] == "من مثلاً به مدرسه رفتم"


def test_rejects_out_of_vocabulary_and_keeps_other_edits():
    out, rej = run("من مصلن به حانه", "من مسلن به خانه")
    assert out == "من مصلن به خانه" and rej["vocabulary"] == 1


def test_rejects_new_digit_word_count_and_far_edit():
    assert run("من به", "من به ۱۲")[1]["new_digit_or_latin"] == 1
    assert run("من به", "من به مدرسه رفتم")[1]["word_count"] == 1
    assert run("من مصلن", "من مدرسه")[1]["edit_distance"] == 1
