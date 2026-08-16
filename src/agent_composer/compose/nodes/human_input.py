"""kind=human_input — the blocking human gate's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `human_input` node's
frozen `HumanInputDescriptor`. It inherits the template `parse` from `NodeDescriptor` (which
records each field's `SourceRange`), overriding only the `_validate_fields` hook to enforce the
one-of-three question-surface legality, and overriding `build` (which constructs the runtime
`HumanInputNode` and derives the answer schema). `_finish` stamps only the descriptor's opaque
origin onto that node and keeps concrete ranges on the Compose-owned `BuildResult`. Registered
into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.

The `adaptive_questions:` EXPANSION (LLM-authored questions -> a synthesized agent feeding a
plain questions gate) is a cross-node desugar and stays a graph stage; this handle only
validates the block's shape and carries it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.diagnostics import SourceRange
from agent_composer.expr.template import prompt_refs
from agent_composer.nodes.contracts import ValueContract
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.typesys.values import Type, ValueKind


@dataclass(frozen=True, kw_only=True)
class HumanInputDescriptor(NodeDescriptor):
    """kind=human_input — block for a person; the typed answer is the node's output.

    Three author surfaces, exactly one of which carries the question(s):
    - `questions`: static list (`[{question, header, options, multi_select}, ...]`) OR
      a `${...}` ref string to a declared input (the manual-ref form);
    - `adaptive_questions`: a nested block (dict) `{prompt: <LLM brief>, mode?,
      llm_config?, retries?}` — the engine asks an LLM to author the questions;
    - `prompt`: the legacy free-text approve/answer surface.
    Legality (exactly-one, adaptive needs its own `prompt:`) is enforced in `_validate_fields`.

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata), the `inputs:`/`outputs:` signature surface, and the
    template `parse` from `NodeDescriptor`. It overrides ONLY the `_validate_fields` hook (the
    one-of-three legality, calling `super()` first for the default asserts-list check); only the
    human_input-specific fields are declared here.

    Class attrs: `kind="human_input"`; `allowed` adds `prompt`/`questions`/`adaptive_questions`/
    `inputs`/`outputs`/`asserts` to the commons; `required` is empty (no single flat field is
    required — the question surface is one-of-three, enforced in `_validate_fields`);
    `binding_fields=()` — this PRESERVES the retired `map_binding_strings_in_descriptor` ladder
    behavior, which never matched a human_input, so its `inputs:` were NOT inline-call-desugared
    (making them desugarable would be a behavior change, out of scope here).
    """

    kind = "human_input"
    allowed = NodeDescriptor.allow(
        "prompt", "questions", "adaptive_questions", "inputs", "outputs", "asserts"
    )
    required = NodeDescriptor.require()
    binding_fields = ()

    prompt: Optional[str] = None
    questions: Any = None
    adaptive_questions: Optional[dict] = None
    asserts: list[str] = field(default_factory=list)

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
        one-of-three legality check FIRST (matching the pre-refactor precedence), then calls
        `super()` for the default asserts-list check: exactly one of prompt/questions/
        adaptive_questions is required, `questions` and `adaptive_questions` are mutually
        exclusive, and an adaptive block must itself be a dict carrying its own `prompt:` (the
        LLM brief). The "needs a question surface" error anchors on `own_range` (the node body)
        since no question field is present to carry a range; the other two anchor on the
        `adaptive_questions:` field range where it exists."""
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
        super()._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)

    def _build_gate(self, diagnostics: LoadDiagnostics) -> HumanInputNode:
        """Build a `human_input` gate, carrying its `questions:` config in one of three forms.

        - `self.questions` is a **list** (static/literal form): carried verbatim as
          `questions` (`questions_input=None`).
        - `self.questions` is a **string** (a bare `"${name}"` input ref): the named INPUT
          param is recorded as `questions_input` (`questions=None`), so the gate resolves the
          list from its bound input at run time. The ref must name a single declared input —
          exactly one ref, no dots (a pool ref like `"${a.output}"` is rejected; deeper
          pool-ref validation is a later pass).
        - `self.questions` is `None` (legacy form): a plain prompt gate with no questions.
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
        if isinstance(questions, str):
            refs = prompt_refs(questions)
            if len(refs) != 1 or "." in refs[0]:
                raise diagnostics.error(
                    f"node {self.id!r}: questions ref must be a single bare input name "
                    f"(e.g. ${{qs}}), got {questions!r}",
                    range=(self.field_ranges.get(("field", "questions")) or self.range),
                )
            return HumanInputNode(
                self.id,
                prompt=self.prompt or "",
                questions=None,
                questions_input=refs[0],
                title=self.node_name,
            )
        return HumanInputNode(self.id, prompt=self.prompt or "", title=self.node_name)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `HumanInputNode` (+ wiring) for this `HumanInputDescriptor`.

        Builds the human_input leaf: build the gate (one of three
        question forms), derive `output_contract` from `outputs:` (a questions gate with none
        defaults to a bare open OBJECT — the host returns a free-form answer record), compute the
        `answer_schema`, and wire node-side `params` + flow-owned `inputs`. `_finish` stamps the
        descriptor origin on the node and keeps its ranges on the Compose-owned `BuildResult`.
        `ctx.resolver` is unused (leaf kind), and the inline-CODE anchors on `ctx` are read only
        by the CODE build.
        """
        node = self._build_gate(ctx.diagnostics)
        node.output_contract = self.output_contract(ctx)
        # A questions gate with no explicit `outputs:` defaults to a bare open OBJECT (the host
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
    """A light IOField-shaped schema for a HUMAN_INPUT answer, from its output `Type`.

    A record output -> one entry per field (recursively); any other (scalar / enum / list,
    incl. list-of-enum) -> a single `answer` entry. Lets a host renderer prompt against the
    expected answer type and pass that schema to the host interaction renderer."""
    if typ is None:
        return []
    if typ.kind == ValueKind.OBJECT and typ.fields:
        required = typ.required or frozenset()
        return [_field_schema(k, f, k in required) for k, f in typ.fields.items()]
    return [_field_schema("answer", typ, not typ.nullable)]
