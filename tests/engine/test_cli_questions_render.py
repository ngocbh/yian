"""Unit tests for the CLI host's human_input question assembly.

`assemble_question_answers` is pure: it keys each question's elicited answer by the
question's `header`. The `ask` callable is scripted here so the test runs without a TTY.
"""

from yian.cli.run import assemble_question_answers


def test_single_select_assembly():
    questions = [{"question": "Which?", "header": "Framework",
                  "options": [{"label": "React", "description": ""}], "multi_select": False}]
    rec = assemble_question_answers(questions, ask=lambda q: "React")
    assert rec == {"Framework": "React"}


def test_multi_select_assembly():
    questions = [
        {"question": "Pick areas", "header": "Areas",
         "options": [{"label": "API"}, {"label": "UI"}], "multi_select": True},
        {"question": "Notes?", "header": "Notes", "options": [], "multi_select": False},
    ]
    scripted = {"Areas": ["API", "UI"], "Notes": "ship it"}
    rec = assemble_question_answers(questions, ask=lambda q: scripted[q["header"]])
    assert rec == {"Areas": ["API", "UI"], "Notes": "ship it"}
