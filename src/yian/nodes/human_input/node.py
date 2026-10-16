"""HUMAN_INPUT — a blocking call to the run's ``ask_user`` tool.

The node renders and validates its authored question surface, invokes the reserved
tool through the trusted metadata path, and returns the handler's answer as its one
``Output``. The worker remains occupied until the host answers or raises.
"""

from typing import Any, Optional

from yian.expr.template import render_template_record
from yian.nodes.base import Node, NodeKind, Output, _check_named_prompt
from yian.nodes.human_input.questions import parse_questions
from yian.tools import ToolSet


class HumanInputNode(Node):
    """
    A structural gate that blocks its worker until a person answers.

    On its single run the node invokes the run's ``ask_user`` tool through the
    trusted path and returns the answer as this node's ``Output``. Tool
    validation and handler exceptions propagate to the ordinary node-failure boundary.

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
            Field schema describing the expected answer; defaults to `[]`.
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
    needs_tools = True

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

    def forward(self, inputs: dict, *, tools: ToolSet | None = None) -> Output:
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
            parsed_questions = None
        if self.questions is not None or self.questions_input is not None:
            parsed = parse_questions(raw)  # raises QuestionSpecError on a bad shape
            parsed_questions = [question.model_dump() for question in parsed]

        tools = tools if tools is not None else ToolSet()
        answer = tools.invoke_trusted(
            "ask_user",
            {
                "question": prompt or "",
                "questions": parsed_questions,
                "answer_schema": self.answer_schema or None,
                "title": self.title,
                "node_id": self.id,
            },
            node=self.id,
        )
        return Output(answer)

    def validate_prompts(self, *, path: str | None = None) -> None:
        """Check the prompt and each literal question's text templates.

        Args:
            path (`str`, *optional*, defaults to `None`):
                The static node path to name in the error instead of the node's id,
                used by a pre-run check over a whole flow.

        Raises:
            `ExpressionError`:
                Raised when a template's `${...}` span doesn't parse, calls `isinstance` or
                a function that isn't a builtin, or calls a builtin with arguments it doesn't
                take; the message names the node and the site.
        """
        for site_path, text in template_sites(self.prompt, self.questions):
            _check_named_prompt(text, f"node {(path or self.id)!r} {site_label(site_path)}")

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


def template_sites(
    prompt: Optional[str], questions: Optional[list]
) -> list[tuple[tuple[str | int, ...], str]]:
    """Every string a HUMAN_INPUT gate renders as a template, as `(path, text)` pairs.

    `path` is the key path from the node body: `("prompt",)`, `("questions", i, "question")`,
    or `("questions", i, "options", j, "label" | "description")`. Only string values count,
    matching `HumanInputNode.forward`: `header`, `multi_select`, string options, and an
    upstream `questions_input` list are not templates.

    Args:
        prompt (`str`, *optional*):
            The node's authored prompt; a site only when it is a `str`.
        questions (`list`, *optional*):
            The node's literal question list; non-list values and non-dict items
            contribute no sites.

    Returns:
        `list[tuple[tuple[str | int, ...], str]]`:
            The `(path, text)` pairs in body order: the prompt, then each question's
            `question` followed by its options' `label` and `description`.
    """
    sites: list[tuple[tuple[str | int, ...], str]] = []
    if isinstance(prompt, str):
        sites.append((("prompt",), prompt))
    if not isinstance(questions, list):
        return sites
    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        if isinstance(question.get("question"), str):
            sites.append((("questions", i, "question"), question["question"]))
        options = question.get("options")
        if not isinstance(options, list):
            continue
        for j, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            for key in ("label", "description"):
                if isinstance(option.get(key), str):
                    sites.append((("questions", i, "options", j, key), option[key]))
    return sites


def site_label(path: tuple[str | int, ...]) -> str:
    """A key path written as an author reads it, such as `questions[0].options[1].label`.

    Args:
        path (`tuple[str | int, ...]`):
            A key path; string steps are field names and int steps are list indexes.

    Returns:
        `str`:
            The string steps joined with `.`, each int step written as `[i]`.
    """
    label = ""
    for step in path:
        if isinstance(step, int):
            label += f"[{step}]"
        else:
            label += f".{step}" if label else step
    return label
