"""Legacy artificial-pause payloads retain optional question metadata on the wire."""

from agent_composer.suspension.pause import HumanInputRequired


def test_questions_defaults_to_empty_list():
    """Omitting `questions` yields the legacy compatibility form: an empty list."""
    reason = HumanInputRequired(prompt="?")
    assert reason.questions == []


def test_questions_round_trips_through_pydantic():
    """A non-empty `questions` list survives a model_dump/model_validate round-trip."""
    questions = [
        {
            "question": "Q",
            "header": "H",
            "options": [{"label": "A"}],
            "multi_select": False,
        }
    ]
    reason = HumanInputRequired(prompt="?", questions=questions)
    assert reason.questions == questions

    dumped = reason.model_dump()
    assert dumped["questions"] == questions

    restored = HumanInputRequired.model_validate(dumped)
    assert restored.questions == questions
