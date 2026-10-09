from labeling.form_guess import guess_region_type


def test_form_keywords_and_latin():
    assert guess_region_type("Date:") == "form_field"
    assert guess_region_type("Month:") == "form_field"
    assert guess_region_type("Subject;") == "form_field"
    assert guess_region_type("Year") == "form_field"


def test_persian_body():
    assert guess_region_type("سلام دنیا این یک خط بدنه است") == "body"
    assert guess_region_type("تاریخ امتحان فردا است و باید بخوانیم") == "body"


def test_empty_defaults_body():
    assert guess_region_type("") == "body"
