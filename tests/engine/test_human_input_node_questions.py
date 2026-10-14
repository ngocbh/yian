"""HUMAN_INPUT gate: run() resolves, validates, and renders questions.

Covers the three question-bearing forms (literal baked-in, runtime input-ref,
bad fed shape) plus the prompt-only path, asserting the trusted host payload and
the returned ``Output``.
"""

import pytest

from yian.nodes.base import Output
from yian.nodes.human_input.node import HumanInputNode, site_label, template_sites
from yian.nodes.human_input.questions import QuestionSpecError
from yian.tools import ToolSet


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


def test_template_sites_lists_every_rendered_string():
    questions = [
        {
            "question": "Q ${a}",
            "header": "H",
            "options": [{"label": "L ${b}", "description": "D"}, "plain", {"label": 3}],
        },
        "not a mapping",
        {"question": 7, "header": "H2"},
    ]
    sites = template_sites("P ${c}", questions)
    assert [(site_label(path), text) for path, text in sites] == [
        ("prompt", "P ${c}"),
        ("questions[0].question", "Q ${a}"),
        ("questions[0].options[0].label", "L ${b}"),
        ("questions[0].options[0].description", "D"),
    ]


def test_template_sites_skips_absent_prompt_and_questions():
    assert template_sites(None, None) == []
    assert template_sites("", None) == [(("prompt",), "")]


def test_forward_renders_exactly_the_template_sites():
    # Every string field carries `${m}`; `forward` must render the template sites and
    # leave every other string verbatim, so the load check and the renderer agree.
    questions = [
        {
            "question": "${m}",
            "header": "${m}",
            "options": [
                {"label": "${m}", "description": "${m}"},
                {"label": "${m}", "description": "${m}"},
            ],
        }
    ]
    node = HumanInputNode("ask", prompt="${m}", questions=questions)
    requests = []
    node.run(
        {"m": "X"},
        tools=ToolSet({"ask_user": lambda **request: requests.append(request) or {}}),
    )
    sent = requests[0]
    rendered = {path for path, _ in template_sites("${m}", questions)}

    authored_paths = [("prompt",)]
    for i, question in enumerate(questions):
        authored_paths += [("questions", i, "question"), ("questions", i, "header")]
        for j, _ in enumerate(question["options"]):
            authored_paths += [
                ("questions", i, "options", j, "label"),
                ("questions", i, "options", j, "description"),
            ]

    def sent_value(path):
        # The prompt is sent under `question`; question fields keep their authored path.
        if path == ("prompt",):
            return sent["question"]
        value = sent
        for step in path:
            value = value[step]
        return value

    for path in authored_paths:
        expected = "X" if path in rendered else "${m}"
        assert sent_value(path) == expected, site_label(path)
    assert ("questions", 0, "header") not in rendered
