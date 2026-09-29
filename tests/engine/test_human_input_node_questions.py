"""HUMAN_INPUT gate: run() resolves, validates, and renders questions.

Covers the three question-bearing forms (literal baked-in, runtime input-ref,
bad fed shape) plus the prompt-only path, asserting the trusted host payload and
the returned ``Output``.
"""

import pytest

from agent_composer.nodes.base import Output
from agent_composer.nodes.human_input.node import HumanInputNode
from agent_composer.nodes.human_input.questions import QuestionSpecError
from agent_composer.tools import ToolSet


def test_literal_questions_render_and_invoke_host():
    node = HumanInputNode(
        "ask",
        questions=[
            {"question": "Pick ${x}", "header": "H", "options": [{"label": "A"}]}
        ],
    )
    requests = []
    result = node.run(
        {"x": "framework"},
        tools=ToolSet({"ask_user": lambda **request: requests.append(request) or {"H": "A"}}),
    )

    assert result == Output({"H": "A"})
    assert len(requests[0]["questions"]) == 1
    q = requests[0]["questions"][0]
    assert q["question"] == "Pick framework"
    assert q["header"] == "H"


def test_questions_input_ref_resolves():
    node = HumanInputNode("ask", questions_input="qs")
    requests = []
    result = node.run(
        {"qs": [{"question": "Q", "header": "H", "options": [{"label": "A"}]}]},
        tools=ToolSet({"ask_user": lambda **request: requests.append(request) or {"H": "A"}}),
    )

    assert result == Output({"H": "A"})
    assert len(requests[0]["questions"]) == 1
    assert requests[0]["questions"][0]["question"] == "Q"


def test_bad_fed_type_raises():
    node = HumanInputNode("ask", questions_input="qs")
    with pytest.raises(QuestionSpecError):
        # An empty list: parse_questions rejects it before invoking the host.
        node.run({"qs": []}, tools=ToolSet({"ask_user": lambda **request: None}))


def test_prompt_only_invokes_host():
    node = HumanInputNode("ask", prompt="Approve ${x}?")
    requests = []
    result = node.run(
        {"x": "deploy"},
        tools=ToolSet({"ask_user": lambda **request: requests.append(request) or "yes"}),
    )

    assert result == Output("yes")
    assert requests[0]["question"] == "Approve deploy?"
    assert requests[0]["questions"] is None
