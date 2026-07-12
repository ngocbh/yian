import pytest
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.stages.parse import parse_nodes
from agent_composer.compose.errors import LoadError


def _node(body):
    return parse_nodes(NodeReader.spanless({"ask": {"kind": "human_input", **body}}))["ask"]


def test_static_questions_ok_without_prompt():
    d = _node({"questions": [{"question": "Q", "header": "H",
                             "options": [{"label": "A"}, {"label": "B"}]}], "output": "object"})
    assert d.questions and d.prompt is None and d.adaptive_questions is None


def test_adaptive_questions_block_ok():
    d = _node({"input": {"ctx": "${a.output}"},
               "adaptive_questions": {"prompt": "make questions from ${ctx}"}})
    assert d.adaptive_questions["prompt"] and d.questions is None


def test_adaptive_questions_block_full_fields():
    d = _node({"input": {"ctx": "${a.output}"},
               "adaptive_questions": {"prompt": "from ${ctx}", "mode": "plain",
                                  "llm_config": {"model": "claude-opus-4-8"}, "retries": 3}})
    assert d.adaptive_questions["mode"] == "plain" and d.adaptive_questions["retries"] == 3


def test_questions_ref_form_ok():
    d = _node({"input": {"qs": "${c.output}"}, "questions": "${qs}", "output": "object"})
    assert d.questions == "${qs}"


def test_rejects_neither_prompt_questions_nor_auto():
    with pytest.raises(LoadError):
        _node({"output": "object"})


def test_rejects_adaptive_and_questions_together():
    with pytest.raises(LoadError, match="adaptive_questions.*questions|questions.*adaptive_questions"):
        _node({"adaptive_questions": {"prompt": "x"},
               "questions": [{"question": "Q", "header": "H"}]})


def test_rejects_adaptive_questions_without_prompt():
    with pytest.raises(LoadError, match="prompt"):
        _node({"adaptive_questions": {"mode": "plain"}})


def test_legacy_prompt_only_still_ok():
    d = _node({"prompt": "Approve ${x}?", "input": {"x": "${a.output}"}, "output": "str"})
    assert d.prompt and d.questions is None and d.adaptive_questions is None
