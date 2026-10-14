"""Graph-stage: `human_input` `adaptive_questions:` block desugar — block → synth agent + gate.

A `human_input` node may carry an `adaptive_questions:` nested block instead of a
static `questions:` list or a free-text `prompt:`. The block IS the spec of an LLM
that AUTHORS the questions: `{prompt: <LLM brief, required>, llm_config?,
retries?}`. This pass lowers that single descriptor into TWO ordinary descriptors:

1. a synth **AgentDescriptor** (id `<gate>__compose`) that composes a
   `list[Question]` — its codomain is the code-built `question_list_type()` set via
   the `output_contract_override` seam (no surface type-string), its prompt is the
   block's brief, and it inherits the gate's context `input:` with its declared
   types, and records the gate as its `host`, so the host-edge step gives it the gate's
   CASE routes, `depends_on`, `runs_after`, and data gates on the gate's other inputs (no LLM
   call when the gate can't run);
2. the original **HumanInputDescriptor** gate (same id), rewritten to read its
   questions from the composer: `questions: "${__questions}"`, with the composer's
   output wired into the reserved `__questions` input.

WHY a descriptor-level pass (modeled on `desugar_inline_calls`): it runs BEFORE the build
loop, so the synth agent flows through the kind-blind `desc.build(ctx)` + prompt-scope
validation + ref wiring exactly like any author-written agent. A brief that references an
undeclared `${...}` is therefore caught by the same prompt-scope check that guards every agent.
A post-build pass would skip that validation entirely.

`__questions` is a reserved input name: the wire from the composer's output into the
gate's questions input. Authors do not write it.

Located parse metadata: the synth composer agent inherits the gate descriptor's
`range`/`field_ranges` (so its downstream errors locate at the gate), except that its
`("field", "prompt")` range is the brief's own range and its `("field", "llm_config")` range
is the block's `llm_config:` value range. Its `field_labels` names its `prompt` and
`llm_config` as the gate's `adaptive_questions.prompt` and `adaptive_questions.llm_config`,
so a bad brief or config is reported on the gate at the authored value, and names each
copied `input 'p'` as the gate's `input 'p'`, so a bad binding reads exactly like the gate's
own error at the same range and is reported once. The rewritten gate keeps its own ranges
via `replace`.
"""

from __future__ import annotations

from dataclasses import replace

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.parse.file import (
    AgentDescriptor,
    HumanInputDescriptor,
    NodeDescriptor,
)
from yian.nodes.contracts import ValueContract
from yian.nodes.human_input.questions import question_list_type

# Reserved gate input name: the wire from the synth composer's output into the gate's
# questions input. Authors never write it; the desugar owns the namespace.
QPARAM = "__questions"


def desugar_adaptive_questions(
    descriptors: dict[str, NodeDescriptor],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> dict[str, NodeDescriptor]:
    """Lower each `adaptive_questions:` human_input into a synth agent + rewritten gate.

    Returns a NEW descriptor map. A non-adaptive descriptor passes through untouched;
    an adaptive `human_input` is replaced IN PLACE by two entries emitted in order —
    the synth composer agent first (id `<gate>__compose`), then the rewritten gate —
    so the agent precedes the gate in insertion order (build/edge order).

    Args:
        descriptors (`dict[str, NodeDescriptor]`):
            Parsed descriptors keyed by node id in authored graph order.

    Returns:
        `dict[str, NodeDescriptor]`:
            New ordered mapping with each adaptive gate lowered to two nodes.

    Raises:
        `LoadError`:
            Raised for a synthesized node-id collision or an authored reserved
            `__questions` input on an adaptive gate.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    new_descriptors: dict[str, NodeDescriptor] = {}
    for gid, desc in descriptors.items():
        if not (
            isinstance(desc, HumanInputDescriptor)
            and desc.adaptive_questions is not None
        ):
            new_descriptors[gid] = desc
            continue

        auto = desc.adaptive_questions
        synth_id = f"{gid}__compose"
        if synth_id in descriptors:
            raise diagnostics.error(
                f"node {gid!r} (kind=human_input): synthesized composer id "
                f"{synth_id!r} collides with an existing node id",
                range=desc.range,
            )
        if QPARAM in desc.inputs:
            raise diagnostics.error(
                f"node {gid!r} (kind=human_input): input {QPARAM!r} is reserved "
                "for adaptive_questions",
                range=desc.field_ranges.get(("input", QPARAM), desc.range),
            )

        # The block IS the agent spec. Pass optional fields only when present so the
        # AgentDescriptor defaults (e.g. retries=2, llm_config={}) apply otherwise. The
        # synth agent inherits the gate's ranges so its downstream errors locate there,
        # but its prompt and llm_config are the block's, so their errors locate at the
        # block's values and name the gate's fields. Its copied inputs are named as the
        # gate's, so their errors read as the gate's own. Its internal contract has no
        # authorable type string.
        field_ranges = {
            k: v for k, v in desc.field_ranges.items() if k != ("field", "prompt")
        }
        brief_range = desc.field_ranges.get(
            ("text", "adaptive_questions.prompt")
        ) or desc.field_ranges.get(("field", "adaptive_questions"))
        if brief_range is not None:
            field_ranges[("field", "prompt")] = brief_range
        llm_config_range = desc.field_ranges.get(
            ("field", "adaptive_questions.llm_config")
        ) or desc.field_ranges.get(("field", "adaptive_questions"))
        if llm_config_range is not None:
            field_ranges[("field", "llm_config")] = llm_config_range
        agent_kwargs = {
            "id": synth_id,
            "prompt": auto["prompt"],
            # The composer reads the gate's context inputs, so it checks their
            # declared types too.
            "inputs": dict(desc.inputs),
            "input_types": dict(desc.input_types),
            "output_contract_override": ValueContract(question_list_type()),
            "outputs": None,
            # `host` gives the composer the gate's routes and ordering; its copied inputs
            # already make it wait like the gate.
            "host": gid,
            "range": desc.range,
            "field_ranges": field_ranges,
            "field_labels": {
                "prompt": f"node {gid!r} adaptive_questions.prompt",
                "llm_config": f"node {gid!r} adaptive_questions.llm_config",
                # Keyed like `Node.wiring_ref_label`'s default, which validators look up.
                **{f"input {p!r}": f"node {gid!r} input {p!r}" for p in desc.inputs},
            },
        }
        if "llm_config" in auto:
            agent_kwargs["llm_config"] = auto["llm_config"]
        if "retries" in auto:
            agent_kwargs["retries"] = auto["retries"]
        new_descriptors[synth_id] = AgentDescriptor(**agent_kwargs)

        # Rewrite the gate: read questions from the composer via the reserved wire.
        new_descriptors[gid] = replace(
            desc,
            adaptive_questions=None,
            questions="${%s}" % QPARAM,
            inputs={**desc.inputs, QPARAM: "${%s.output}" % synth_id},
        )
    return new_descriptors
