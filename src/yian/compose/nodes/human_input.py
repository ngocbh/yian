"""kind=human_input — the blocking human gate's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `human_input` node's
frozen `HumanInputDescriptor`. It wraps the template `parse` from `NodeDescriptor` (which
records each field's `SourceRange`) to reject the removed `adaptive_questions.mode:` key and
any other `adaptive_questions:` key it does not read, and to record one
`("text", "questions[i]...")` range per question string and a
`("text", "adaptive_questions.prompt")` range for the adaptive brief, overrides the
`_validate_fields` hook to enforce the one-of-three question-surface legality, and overrides
`build` (which constructs the runtime `HumanInputNode` and derives the answer schema). A
`prompt:`, an `adaptive_questions.prompt:` brief, or a literal question's `question`/`header`/
option `label`/`description` YAML read as non-text is rejected at that value.
`_finish` stamps only the descriptor's opaque origin onto that node and keeps concrete ranges
on the Compose-owned `BuildResult`. Registered into `KINDS`/`BINDING_FIELDS` at import so the
shared dispatch routes here.

The `adaptive_questions:` EXPANSION (LLM-authored questions -> a synthesized agent feeding a
plain questions gate) is a cross-node desugar and stays a graph stage; this handle only
validates the block's shape and carries it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.agent import MODE_REMOVED_FIX, not_text_error, retries_error
from yian.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from yian.compose.source import SourceAccessError, SourceCursor
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError
from yian.expr.grammar import is_bare_name
from yian.expr.template import Span, scan_template
from yian.nodes.contracts import ValueContract
from yian.nodes.human_input import HumanInputNode
from yian.nodes.human_input.node import site_label, template_sites
from yian.typesys.values import Type, ValueKind

# The keys the adaptive desugar reads from an `adaptive_questions:` block.
_ADAPTIVE_KEYS = frozenset({"prompt", "llm_config", "retries"})


@dataclass(frozen=True, kw_only=True)
class HumanInputDescriptor(NodeDescriptor):
    """kind=human_input — block for a person; the typed answer is the node's output.

    Three author surfaces, exactly one of which carries the question(s):
    - `questions`: static list (`[{question, header, options, multi_select}, ...]`) OR
      a `${...}` ref string to a declared input (the manual-ref form);
    - `adaptive_questions`: a nested block (dict) `{prompt: <LLM brief>, llm_config?,
      retries?}` — the engine asks an LLM to author the questions;
    - `prompt`: the legacy free-text approve/answer surface.
    Legality (exactly-one, adaptive needs its own `prompt:`) is enforced in `_validate_fields`.

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata), the `input:`/`output:` signature surface, and the
    template `parse` from `NodeDescriptor`, which it wraps to reject the removed
    `adaptive_questions.mode:` key, any `adaptive_questions:` key other than `prompt`,
    `llm_config`, or `retries`, a non-text `adaptive_questions.prompt:` brief, and a
    non-text `question`/`header`/option `label`/`description` in a literal `questions:` list,
    and to add a `("text", "questions[i]...")` range for each question string and a
    `("text", "adaptive_questions.prompt")` range for the adaptive brief to `field_ranges`. It
    overrides the `_validate_fields` hook (the one-of-three legality and a text `prompt:`,
    calling `super()` for the default asserts-list check); only the human_input-specific
    fields are declared here.

    Class attrs: `kind="human_input"`; `allowed` adds `prompt`/`questions`/`adaptive_questions`,
    normalized `inputs`/`outputs`, and `asserts` to the commons; `required` is empty (no single flat field is
    required — the question surface is one-of-three, enforced in `_validate_fields`);
    `binding_fields=("inputs",)` — HUMAN_INPUT's ordinary input bindings participate in
    inline-call and case-output desugaring.
    """

    kind = "human_input"
    allowed = NodeDescriptor.allow(
        "prompt", "questions", "adaptive_questions", "inputs", "outputs", "asserts"
    )
    required = NodeDescriptor.require()
    binding_fields = ("inputs",)

    prompt: Optional[str] = None
    questions: Any = None
    adaptive_questions: Optional[dict] = None
    asserts: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, node_id: str, cursor: SourceCursor) -> "HumanInputDescriptor":
        """Reject unknown `adaptive_questions:` keys and non-text brief or question text; range text.

        After the template `parse`, rejects the first `question`/`header`/option `label`/
        `description` in a literal `questions:` list that is present but not a string, at that
        value. On top of the template `parse`'s per-field ranges, records a
        `("text", "adaptive_questions.prompt")` range for an `adaptive_questions:` block's
        `prompt:` brief and one `("text", site_label(path))` range for each string
        `template_sites` finds in a literal `questions:` list (`question`, option `label`/
        `description`), so a text error points at the string rather than the whole
        `adaptive_questions:` or `questions:` block.

        Raises:
            `LoadError`: The `adaptive_questions:` block authors `mode:` or a key other than
                `prompt`/`llm_config`/`retries`, its `prompt:` brief is not text (YAML read
                it as a number, boolean, date, list, or mapping), the
                template `parse` rejects the node, or a literal question's `question`,
                `header`, option `label`, or option `description` is present but not text.
        """
        adaptive = cursor.get("adaptive_questions")
        if isinstance(adaptive, dict):
            if "mode" in adaptive:
                raise LoadDiagnostics.from_cursor(cursor).error(
                    f"node {node_id!r} (kind=human_input): `adaptive_questions.mode:` was "
                    f"removed: {MODE_REMOVED_FIX}",
                    range=_key_range(cursor.child("adaptive_questions"), "mode")
                    or cursor.key_range_of("adaptive_questions"),
                )
            for key in adaptive:
                if key not in _ADAPTIVE_KEYS:
                    raise LoadDiagnostics.from_cursor(cursor).error(
                        f"node {node_id!r} (kind=human_input): `adaptive_questions:` field "
                        f"{key!r} is not allowed (allowed: {', '.join(sorted(_ADAPTIVE_KEYS))})",
                        range=_key_range(cursor.child("adaptive_questions"), key)
                        or cursor.key_range_of("adaptive_questions"),
                    )
        brief = adaptive.get("prompt") if isinstance(adaptive, dict) else None
        if brief is not None and not isinstance(brief, str):
            raise LoadDiagnostics.from_cursor(cursor).error(
                f"node {node_id!r} (kind=human_input): "
                + not_text_error("adaptive_questions.prompt", brief),
                range=_range_at(cursor, ("adaptive_questions", "prompt"))
                or cursor.key_range_of("adaptive_questions"),
            )
        desc = super().parse(node_id, cursor)
        ranges: dict[tuple[str, str], SourceRange] = {}
        if isinstance(adaptive, dict) and "prompt" in adaptive:
            rng = _range_at(cursor, ("adaptive_questions", "prompt"))
            if rng is not None:
                ranges[("text", "adaptive_questions.prompt")] = rng
        if not isinstance(desc.questions, list):
            return replace(desc, field_ranges={**desc.field_ranges, **ranges})
        found = _non_text_question_field(desc.questions)
        if found is not None:
            path, value = found
            raise LoadDiagnostics.from_cursor(cursor).error(
                f"node {node_id!r} (kind=human_input): "
                + not_text_error(site_label(path), value),
                range=_range_at(cursor, path)
                or desc.field_ranges.get(("field", "questions"))
                or desc.range,
            )
        for path, _ in template_sites(None, desc.questions):
            rng = _range_at(cursor, path)
            if rng is not None:
                ranges[("text", site_label(path))] = rng
        return replace(desc, field_ranges={**desc.field_ranges, **ranges})

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[str, SourceRange],
        own_range: Optional[SourceRange],
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Enforce the one-of-three question-surface legality, on top of the default checks.

        Overrides the base hook, run by the template `parse` after the own-key loop. Reads the
        three question surfaces from `kwargs` (already populated by the loop — human_input has no
        `keyword_remap`, so the kwargs keys equal the surface keys), NOT the raw body. Runs the
        one-of-three legality check FIRST (matching the pre-refactor precedence): exactly one of
        prompt/questions/adaptive_questions is required, `questions` and `adaptive_questions`
        are mutually exclusive, and an adaptive block must itself be a dict carrying its own
        `prompt:` (the LLM brief). The "needs a question surface" error anchors on `own_range`
        (the node body) since no question field is present to carry a range; the other two
        anchor on the `adaptive_questions:` field range where it exists. Then it checks that a
        present `prompt:` is text, anchored on its own field range, and finally calls `super()`
        for the default asserts-list check."""
        # One-of-three legality: needs a question surface, `questions`/`adaptive_questions` are
        # mutually exclusive, and an adaptive block needs its own `prompt:` (the LLM brief).
        has_q = kwargs.get("questions") is not None
        auto = kwargs.get("adaptive_questions")
        if kwargs.get("prompt") is None and not has_q and auto is None:
            raise diagnostics.error(
                f"node {node_id!r} (kind=human_input): needs `prompt:`, "
                f"`questions:`, or `adaptive_questions:`",
                range=own_range,
            )
        if has_q and auto is not None:
            raise diagnostics.error(
                f"node {node_id!r} (kind=human_input): `questions:` and "
                f"`adaptive_questions:` are mutually exclusive",
                range=ranges.get(("field", "adaptive_questions")),
            )
        if auto is not None:
            if not isinstance(auto, dict) or not auto.get("prompt"):
                raise diagnostics.error(
                    f"node {node_id!r} (kind=human_input): `adaptive_questions:` "
                    f"requires a `prompt:` (the LLM brief)",
                    range=ranges.get(("field", "adaptive_questions")),
                )
            problem = retries_error(auto["retries"]) if "retries" in auto else None
            if problem is not None:
                raise diagnostics.error(
                    f"node {node_id!r} (kind=human_input): "
                    f"`adaptive_questions.retries:` {problem}",
                    range=ranges.get(("field", "adaptive_questions")),
                )
        prompt = kwargs.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise diagnostics.error(
                f"node {node_id!r} (kind=human_input): " + not_text_error("prompt", prompt),
                range=ranges.get(("field", "prompt")) or own_range,
            )
        super()._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)

    def _build_gate(self, diagnostics: LoadDiagnostics) -> HumanInputNode:
        """Build a `human_input` gate, carrying its `questions:` config in one of three forms.

        - `self.questions` is a **list** (static/literal form): carried verbatim as
          `questions` (`questions_input=None`).
        - `self.questions` is a **string** (a bare `"${name}"` input ref): the named INPUT
          param is recorded as `questions_input` (`questions=None`), so the gate resolves the
          list from its bound input at run time. The string must be exactly one whole
          `${name}` span — no surrounding text, dots, calls, or operators (a pool ref like
          `"${a.output}"` is rejected; whether the name is a declared input is checked by
          `validate_human_questions`).
        - `self.questions` is `None` (legacy form): a plain prompt gate with no questions.

        Raises:
            `LoadError`: `questions:` is neither a list nor a string, or the string is not
                exactly one whole `${name}`.
        """
        questions = self.questions
        if isinstance(questions, list):
            return HumanInputNode(
                self.id,
                prompt=self.prompt or "",
                questions=questions,
                questions_input=None,
                title=self.node_name,
            )
        questions_range = self.field_ranges.get(("field", "questions")) or self.range
        if isinstance(questions, str):
            name = _whole_ref_name(questions)
            if name is None:
                raise diagnostics.error(
                    f"node {self.id!r} (kind=human_input): `questions:` as a string must be "
                    f"exactly one `${{name}}` naming a declared input (e.g. `questions: "
                    f"${{qs}}`), got {questions!r}; put any text in `prompt:` instead",
                    range=questions_range,
                )
            return HumanInputNode(
                self.id,
                prompt=self.prompt or "",
                questions=None,
                questions_input=name,
                title=self.node_name,
            )
        if questions is not None:
            raise diagnostics.error(
                f"node {self.id!r} (kind=human_input): `questions:` must be a list of "
                f"questions (each item `- question: ...`) or exactly `${{name}}` naming a "
                f"declared input, got {type(questions).__name__} {questions!r}",
                range=questions_range,
            )
        return HumanInputNode(self.id, prompt=self.prompt or "", title=self.node_name)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `HumanInputNode` (+ wiring) for this `HumanInputDescriptor`.

        Builds the human_input leaf: build the gate (one of three
        question forms), derive `output_contract` from authored `output:` (a questions gate with none
        defaults to a bare open OBJECT — the host returns a free-form answer record), compute the
        `answer_schema`, and wire node-side `params` + flow-owned `inputs`. `_finish` stamps the
        descriptor origin on the node and keeps its ranges on the Compose-owned `BuildResult`.
        `ctx.resolver` is unused (leaf kind), and the inline-CODE anchors on `ctx` are read only
        by the CODE build.
        """
        node = self._build_gate(ctx.diagnostics)
        node.output_contract = self.output_contract(ctx)
        # A questions gate with no explicit `output:` defaults to a bare open OBJECT (the host
        # returns a free-form answer record). Legacy gates derive their schema from the (possibly
        # absent) declared output type.
        if node.output_contract is None and (
            node.questions is not None or node.questions_input is not None
        ):
            node.output_contract = ValueContract(Type.scalar(ValueKind.OBJECT))
        output_type = (
            None if node.output_contract is None else node.output_contract.type
        )
        node.answer_schema = _answer_schema(output_type)
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)


def _non_text_question_field(questions: list) -> tuple[tuple, Any] | None:
    """The first question text field that is present and not a `str`, as `(path, value)`.

    Checks each dict question's `question` and `header`, then the `label` and `description` of
    each dict option in a list `options`; path steps are as in `template_sites`. Other shapes
    are left to the question validator.

    Args:
        questions (`list`):
            The node's literal `questions:` list.

    Returns:
        `tuple[tuple, Any] | None`:
            The offending field's key path from the node body and its value, or `None` when
            every present text field is a `str`.
    """
    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        for key in ("question", "header"):
            if key in question and not isinstance(question[key], str):
                return ("questions", i, key), question[key]
        options = question.get("options")
        if not isinstance(options, list):
            continue
        for j, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            for key in ("label", "description"):
                if key in option and not isinstance(option[key], str):
                    return ("questions", i, "options", j, key), option[key]
    return None


def _range_at(cursor: SourceCursor, path: tuple[str | int, ...]) -> Optional[SourceRange]:
    """The `SourceRange` of the value at `path` under `cursor`, or `None` when it has none.

    `None` also covers a path the cursor cannot walk (e.g. past a recursive alias cut), so
    the caller's error falls back to an enclosing range."""
    try:
        for step in path:
            cursor = cursor.child(step)
    except SourceAccessError:
        return None
    return cursor.range


def _key_range(cursor: SourceCursor, key: Any) -> Optional[SourceRange]:
    """The `SourceRange` of `key` in the mapping under `cursor`, or `None` when it has none.

    `None` also covers a mapping past a recursive alias cut, so the caller's error falls back
    to an enclosing range."""
    try:
        return cursor.key_range_of(key)
    except SourceAccessError:
        return None


def _whole_ref_name(text: str) -> Optional[str]:
    """The input name when `text` is exactly one `${name}` span, else `None`.

    `None` also covers surrounding text, a dotted path, a call, an operator, and a span that
    does not parse."""
    try:
        segments = scan_template(text)
    except ExpressionError:
        return None
    if len(segments) != 1 or not isinstance(segments[0], Span):
        return None
    tree = segments[0].tree
    if not is_bare_name(tree):
        return None
    return str(tree.children[0])


def _field_schema(name: str, typ: Type, required: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "type": typ.kind.value, "required": required}
    if typ.tags:  # a Literal[...] enum -> the allowed values
        entry["enum"] = sorted(typ.tags)
    elif (
        typ.element is not None and typ.element.tags
    ):  # list[Literal[...]] -> element enum
        entry["enum"] = sorted(typ.element.tags)
    if typ.kind == ValueKind.OBJECT and typ.fields:  # nested record -> sub-fields
        req = typ.required or frozenset()
        entry["fields"] = [_field_schema(k, f, k in req) for k, f in typ.fields.items()]
    return entry


def _answer_schema(typ: Optional[Type]) -> list[dict[str, Any]]:
    """A lightweight answer-field schema for HUMAN_INPUT, derived from its output `Type`.

    A record output -> one entry per field (recursively); any other (scalar / enum / list,
    incl. list-of-enum) -> a single `answer` entry. Lets a host renderer prompt against the
    expected answer type and pass that schema to the host interaction renderer."""
    if typ is None:
        return []
    if typ.kind == ValueKind.OBJECT and typ.fields:
        required = typ.required or frozenset()
        return [_field_schema(k, f, k in required) for k, f in typ.fields.items()]
    return [_field_schema("answer", typ, not typ.nullable)]
