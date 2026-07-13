"""HUMAN_INPUT — a structural gate that suspends the run for a person.

Deliver-as-Output model: on its single run the node ALWAYS emits
`PauseRequested(HumanInputRequired)` and parks. The host satisfies the pause with a
`DeliverAnswerCommand(node_id=this_node, value=answer)`; the engine writes that value
as the node's `Output` (`pool.set` + fire the out-edges) — the node never re-runs. Its
single output value is the delivered answer.
"""

from typing import Any, Optional

from agent_composer.expr.template import render_template_record
from agent_composer.nodes.base import Node, NodeKind, Pause
from agent_composer.nodes.human_input.questions import parse_questions
from agent_composer.suspension.pause import HumanInputRequired


class HumanInputNode(Node):
    """
    A structural gate that suspends the run for a person (deliver-as-Output model).

    On its single run the node always emits `Pause(HumanInputRequired)` and parks. The host
    satisfies it with a `DeliverAnswerCommand`, and the engine writes that answer as this node's
    Output — the node never re-runs.

    The node carries questions in one of three ways: a literal list baked in
    (`questions`), a runtime read from a declared input param (`questions_input`),
    or neither (the legacy prompt-only form). At most one of the two question
    sources is set.

    Args:
        node_id (`str`):
            The node's unique id.
        prompt (`str`, *optional*, defaults to `None`):
            The intro text to show the human (rendered against the bound input
            record). `None` for a questions-only node with no human-facing intro.
        answer_schema (`list[dict]`, *optional*, defaults to `None`):
            IOField-shaped description of the expected answer; defaults to `[]`.
        questions (`list`, *optional*, defaults to `None`):
            A LITERAL question list baked into the node. Each question's string
            fields (`question`, and each option's `label`/`description`) are
            `${...}`-rendered against the bound input record at run time.
        questions_input (`str`, *optional*, defaults to `None`):
            The name of a declared input param to read the questions list FROM at
            run time (an upstream-produced list — taken as-is, not re-templated).
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.HUMAN_INPUT

    def __init__(
        self,
        node_id: str,
        *,
        prompt: Optional[str] = None,
        answer_schema: Optional[list[dict[str, Any]]] = None,
        questions: Optional[list] = None,
        questions_input: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.prompt = prompt
        self.answer_schema = answer_schema or []
        self.questions = questions
        self.questions_input = questions_input

    def _run_impl(self, inputs: dict):
        prompt = render_template_record(self.prompt, inputs) if self.prompt is not None else None

        if self.questions is not None:
            # A literal list baked into the node: render `${...}` in each question's
            # human-facing strings against the bound record (header is a key, not text;
            # multi_select is a bool — both left verbatim).
            raw = [self._render_question(q, inputs) for q in self.questions]
        elif self.questions_input is not None:
            # An upstream-produced list, already resolved — taken as-is, no templating.
            raw = inputs[self.questions_input]
        else:
            # Legacy prompt-only form — no questions.
            return Pause(
                HumanInputRequired(
                    prompt=prompt,
                    answer_schema=self.answer_schema,
                    node_title=self.title,
                    node_id=self.id,
                )
            )

        parsed = parse_questions(raw)  # raises QuestionSpecError on a bad shape
        return Pause(
            HumanInputRequired(
                # HumanInputRequired.prompt is a required str — a questions-only node
                # (no intro) passes "" rather than None.
                prompt=prompt or "",
                questions=[q.model_dump() for q in parsed],
                answer_schema=self.answer_schema,
                node_title=self.title,
                node_id=self.id,
            )
        )

    def _render_question(self, question: dict, inputs: dict) -> dict:
        """Render `${...}` spans in a literal question's human-facing strings.

        Renders the `question` text and each option's `label`/`description` against
        `inputs`; leaves `header` (a key) and `multi_select` (a bool) untouched and
        carries any other keys through verbatim for `parse_questions` to validate."""
        rendered = dict(question)
        if isinstance(rendered.get("question"), str):
            rendered["question"] = render_template_record(rendered["question"], inputs)
        options = rendered.get("options")
        if isinstance(options, list):
            rendered["options"] = [self._render_option(opt, inputs) for opt in options]
        return rendered

    def _render_option(self, option: dict, inputs: dict) -> dict:
        """Render `${...}` in one option's `label`/`description` against `inputs`."""
        if not isinstance(option, dict):
            return option
        rendered = dict(option)
        for key in ("label", "description"):
            if isinstance(rendered.get(key), str):
                rendered[key] = render_template_record(rendered[key], inputs)
        return rendered
