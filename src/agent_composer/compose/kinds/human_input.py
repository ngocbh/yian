"""kind=human_input — the suspend-for-a-person node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `human_input` node's
frozen `HumanInputDescriptor`, its located `parse(node_id, reader)` (which enforces the
one-of-three question-surface legality and records each field's `SourceRange`), and its
`build(desc, *, registry, resolver)` (which constructs the runtime `HumanInputNode`, derives
the answer schema, and stamps span metadata). Registered into `KINDS`/`BINDING_FIELDS` at
import so the shared dispatch routes here.

The `adaptive_questions:` EXPANSION (LLM-authored questions -> a synthesized agent feeding a
plain questions gate) is a cross-node desugar and stays a graph stage; this handle only
validates the block's shape and carries it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange
from agent_composer.expr.template import prompt_refs
from agent_composer.typesys.values import Type, ValueKind

# Human_input's own field surface (was the `human_input` row of parser._KIND_SPECS): no single
# required flat field — the question surface is one-of-three (prompt/questions/adaptive_questions),
# enforced in `parse`. These flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
HUMAN_INPUT_REQUIRED: frozenset[str] = frozenset()
HUMAN_INPUT_ALLOWED: frozenset[str] = frozenset(
    {"prompt", "questions", "adaptive_questions", "inputs", "outputs", "asserts"}
)
# Descriptor fields carrying `${...}` binding strings for the kind-blind walk in bindings.py.
# Empty to PRESERVE the retired `calls.py:map_binding_strings_in_descriptor` ladder behavior:
# that isinstance ladder never matched `HumanInputDescriptor`, so a human_input's `inputs:` were
# NOT inline-call-desugared (nor case-output-expanded). Empty keeps that exact behavior — making
# human_input inputs desugarable would be a behavior change, out of scope for the #70 refactor.
HUMAN_INPUT_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class HumanInputDescriptor:
    """kind=human_input — suspend for a person; the typed answer is the node's output.

    Three author surfaces, exactly one of which carries the question(s):
    - `questions`: static list (`[{question, header, options, multi_select}, ...]`) OR
      a `${...}` ref string to a declared input (the manual-ref form);
    - `adaptive_questions`: a nested block (dict) `{prompt: <LLM brief>, mode?,
      llm_config?, retries?}` — the engine asks an LLM to author the questions;
    - `prompt`: the legacy free-text approve/answer surface.
    Legality (exactly-one, adaptive needs its own `prompt:`) is enforced in `parse`.

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    prompt: Optional[str] = None
    questions: Any = None
    adaptive_questions: Optional[dict] = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> HumanInputDescriptor:
    """Read one `kind: human_input` node body (via a located `NodeReader`) into a descriptor.

    Enforces the one-of-three question-surface legality (`prompt`/`questions`/
    `adaptive_questions`; `questions` and `adaptive_questions` mutually exclusive; an adaptive
    block needs its own `prompt:`), then records the node body's `SourceRange` (`desc.range`)
    and each present field's `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors
    the legacy `parser._parse_node` human_input path.
    """
    from agent_composer.compose.parser import _COMMON_FIELDS

    body = reader.value
    allowed_keys = HUMAN_INPUT_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=human_input): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )

    # One-of-three legality: needs a question surface, `questions`/`adaptive_questions` are
    # mutually exclusive, and an adaptive block needs its own `prompt:` (the LLM brief).
    has_q = body.get("questions") is not None
    auto = body.get("adaptive_questions")
    if body.get("prompt") is None and not has_q and auto is None:
        raise LoadError(
            f"node {node_id!r} (kind=human_input): needs `prompt:`, "
            f"`questions:`, or `adaptive_questions:`",
            range=reader._own_range(),
        )
    if has_q and auto is not None:
        raise LoadError(
            f"node {node_id!r} (kind=human_input): `questions:` and "
            f"`adaptive_questions:` are mutually exclusive",
            range=reader.range_of("adaptive_questions"),
        )
    if auto is not None:
        if not isinstance(auto, dict) or not auto.get("prompt"):
            raise LoadError(
                f"node {node_id!r} (kind=human_input): `adaptive_questions:` "
                f"requires a `prompt:` (the LLM brief)",
                range=reader.range_of("adaptive_questions"),
            )

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in HUMAN_INPUT_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind=human_input): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

    for key in ("node_name", "depends_on", "runs_after"):
        if key in body:
            kwargs[key] = body[key]
            rng = reader.range_of(key)
            if rng is not None:
                ranges[key] = rng
    if "env" in body:
        env = body["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
            raise LoadError(
                f"node {node_id!r} (kind=human_input): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return HumanInputDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def _field_schema(name: str, typ: Type, required: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "type": typ.kind.value, "required": required}
    if typ.tags:  # a Literal[...] enum -> the allowed values
        entry["enum"] = sorted(typ.tags)
    elif typ.element is not None and typ.element.tags:  # list[Literal[...]] -> element enum
        entry["enum"] = sorted(typ.element.tags)
    if typ.kind == ValueKind.OBJECT and typ.fields:  # nested record -> sub-fields
        req = typ.required or frozenset()
        entry["fields"] = [_field_schema(k, f, k in req) for k, f in typ.fields.items()]
    return entry


def _answer_schema(typ: Optional[Type]) -> list[dict[str, Any]]:
    """A light IOField-shaped schema for a HUMAN_INPUT answer, from its output `Type`.

    A record output -> one entry per field (recursively); any other (scalar / enum / list,
    incl. list-of-enum) -> a single `answer` entry. Lets a host renderer prompt against the
    expected answer type, and the pause reason self-describe."""
    if typ is None:
        return []
    if typ.kind == ValueKind.OBJECT and typ.fields:
        required = typ.required or frozenset()
        return [_field_schema(k, f, k in required) for k, f in typ.fields.items()]
    return [_field_schema("answer", typ, not typ.nullable)]


def _build_human_input(desc: HumanInputDescriptor):
    """Build a `human_input` gate, carrying its `questions:` config in one of three forms.

    - `desc.questions` is a **list** (static/literal form): carried verbatim as
      `questions` (`questions_input=None`).
    - `desc.questions` is a **string** (a bare `"${name}"` input ref): the named INPUT
      param is recorded as `questions_input` (`questions=None`), so the gate resolves the
      list from its bound input at run time. The ref must name a single declared input —
      exactly one ref, no dots (a pool ref like `"${a.output}"` is rejected; deeper
      pool-ref validation is a later pass).
    - `desc.questions` is `None` (legacy form): a plain prompt gate with no questions.
    """
    from agent_composer.nodes.human_input import HumanInputNode

    questions = desc.questions
    if isinstance(questions, list):
        return HumanInputNode(
            desc.id, prompt=desc.prompt or "", questions=questions,
            questions_input=None, title=desc.node_name,
        )
    if isinstance(questions, str):
        refs = prompt_refs(questions)
        if len(refs) != 1 or "." in refs[0]:
            raise LoadError(
                f"node {desc.id!r}: questions ref must be a single bare input name "
                f"(e.g. ${{qs}}), got {questions!r}"
            )
        return HumanInputNode(
            desc.id, prompt=desc.prompt or "", questions=None,
            questions_input=refs[0], title=desc.node_name,
        )
    return HumanInputNode(desc.id, prompt=desc.prompt or "", title=desc.node_name)


def build(desc: HumanInputDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `HumanInputNode` (+ wiring) for one `HumanInputDescriptor`.

    Mirrors the legacy `build.build_leaf_node` human_input path: build the gate (one of three
    question forms), derive `output_type` from `outputs:` (a questions gate with none defaults
    to a bare open OBJECT — the host returns a free-form answer record), compute the
    `answer_schema`, and wire node-side `params` + flow-owned `inputs`. Additionally stamps
    `.range`/`.field_ranges` from the descriptor. `resolver` is unused (leaf kind) and
    `_loader_hints` absorbs the inline-CODE traceback anchors that only the CODE build reads.
    """
    from agent_composer.compose.build import _output_type, _sink_params, _sink_wiring

    node = _build_human_input(desc)
    node.output_type = _output_type(desc.outputs, registry)
    # A questions gate with no explicit `outputs:` defaults to a bare open OBJECT (the host
    # returns a free-form answer record). Legacy gates derive their schema from the (possibly
    # absent) declared output type.
    if node.output_type is None and (
        node.questions is not None or node.questions_input is not None
    ):
        node.output_type = Type.scalar(ValueKind.OBJECT)
    node.answer_schema = _answer_schema(node.output_type)
    node.params = _sink_params(desc.inputs)
    wiring = _sink_wiring(desc.inputs)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `human_input` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["human_input"] = KindSpec(descriptor_cls=HumanInputDescriptor, parse=parse, build=build)
    BINDING_FIELDS["human_input"] = HUMAN_INPUT_BINDING_FIELDS


register()
