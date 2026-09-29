"""HUMAN_INPUT invokes the run's ask_user tool and returns its answer."""

from __future__ import annotations

import pytest

from agent_composer.nodes.base import Output
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.tools import ToolSet


class _PromptFailed(RuntimeError):
    """Distinct host failure used to prove exception identity is preserved."""


def test_prompt_only_returns_host_answer_with_trusted_metadata():
    requests = []

    def answer(**request):
        requests.append(request)
        return "approve"

    node = HumanInputNode(
        "approval",
        prompt="Approve ${plan}?",
        answer_schema=[
            {
                "name": "answer",
                "type": "str",
                "required": True,
                "enum": ["approve", "revise"],
            }
        ],
        title="Plan approval",
    )

    result = node.run({"plan": "release"}, tools=ToolSet({"ask_user": answer}))

    assert result == Output("approve")
    assert requests == [
        {
            "question": "Approve release?",
            "questions": None,
            "answer_schema": [
                {
                    "name": "answer",
                    "type": "str",
                    "required": True,
                    "enum": ["approve", "revise"],
                }
            ],
            "title": "Plan approval",
            "node_id": "approval",
            "options": None,
        }
    ]


def test_none_answer_is_an_ordinary_output():
    """A host ``None`` is data, not cancellation or a request to pause again."""
    result = HumanInputNode("optional", prompt="Anything to add?").run(
        {}, tools=ToolSet({"ask_user": lambda **request: None})
    )

    assert result == Output(None)


def test_literal_questions_are_rendered_validated_and_injected():
    requests = []

    def answer(**request):
        requests.append(request)
        return {"Framework": "Vue"}

    node = HumanInputNode(
        "choose",
        prompt="Choose for ${project}",
        questions=[
            {
                "question": "Which framework for ${project}?",
                "header": "Framework",
                "options": [{"label": "${candidate}", "description": "Use ${candidate}"}],
            }
        ],
    )

    result = node.run(
        {"project": "Atlas", "candidate": "Vue"}, tools=ToolSet({"ask_user": answer})
    )

    assert result == Output({"Framework": "Vue"})
    assert requests == [
        {
            "question": "Choose for Atlas",
            "questions": [
                {
                    "question": "Which framework for Atlas?",
                    "header": "Framework",
                    "options": [{"label": "Vue", "description": "Use Vue"}],
                    "multi_select": False,
                }
            ],
            "answer_schema": None,
            "title": None,
            "node_id": "choose",
            "options": None,
        }
    ]


def test_questions_input_is_validated_before_trusted_invocation():
    seen = []

    def answer(**request):
        seen.append(request)
        return {"Choice": "A"}

    node = HumanInputNode("choose", questions_input="questions")
    questions = [{"question": "Pick", "header": "Choice", "options": [{"label": "A"}]}]

    result = node.run({"questions": questions}, tools=ToolSet({"ask_user": answer}))

    assert result == Output({"Choice": "A"})
    assert seen[0]["questions"] == [
        {
            "question": "Pick",
            "header": "Choice",
            "options": [{"label": "A", "description": ""}],
            "multi_select": False,
        }
    ]


def test_host_exception_propagates_unchanged():
    failure = _PromptFailed("terminal closed")

    def fail(**request):
        raise failure

    with pytest.raises(_PromptFailed) as caught:
        HumanInputNode("ask", prompt="Question?").run({}, tools=ToolSet({"ask_user": fail}))

    assert caught.value is failure
