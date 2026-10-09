import pytest
from yian.nodes.human_input.questions import (
    OptionSpec, QuestionSpec, parse_questions, QuestionSpecError,
)


def test_parse_questions_minimal_single_select():
    raw = [{"question": "Which framework?", "header": "Framework",
            "options": [{"label": "React", "description": "UI lib"},
                        {"label": "Vue", "description": "progressive"}]}]
    qs = parse_questions(raw)
    assert qs[0].header == "Framework" and qs[0].multi_select is False
    assert [o.label for o in qs[0].options] == ["React", "Vue"]
    assert qs[0].options[0].description == "UI lib"


def test_parse_questions_multiselect_and_freetext_only():
    raw = [{"question": "Pick areas", "header": "Areas", "multi_select": True,
            "options": [{"label": "API"}, {"label": "UI"}]},
           {"question": "Anything else?", "header": "Notes"}]
    qs = parse_questions(raw)
    assert qs[0].multi_select is True and qs[0].options[0].description == ""
    assert qs[1].options == []


def test_parse_questions_rejects_empty_too_many_dups_badtype():
    with pytest.raises(QuestionSpecError): parse_questions([])
    with pytest.raises(QuestionSpecError):
        parse_questions([{"question": f"q{i}", "header": f"H{i}"} for i in range(5)])
    with pytest.raises(QuestionSpecError):
        parse_questions([{"question": "a", "header": "D"}, {"question": "b", "header": "D"}])
    with pytest.raises(QuestionSpecError): parse_questions([{"question": "no header"}])
    with pytest.raises(QuestionSpecError): parse_questions("nope")  # type: ignore[arg-type]
