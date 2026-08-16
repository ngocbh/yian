"""Unit tests for reference-wiring validation.

`validate_references` runs the reused leaf checkers (`compile.validation._classify_path`
+ `_walk_record_fields`) over every binding site of a built flow — node input
`from:` bindings, the flow `outputs:` bindings, and (via the DESUGARED `CaseNode.inputs`)
each `case`'s `on:`/`when:` data refs. It accumulates ALL located errors and raises a
single `LoadError`. Three loud mechanisms are pinned:

- **e01** — a dangling flow-output ref (`${scor.output}` typo) -> loud.
- **e03** — a dotted-field miss on an ANONYMOUS producer record (`{rating, rationale}`,
  read natively via `read_type`) -> loud (anonymous records ARE checked).
- **prompt** — an AGENT prompt interpolating `${X.output}`/`${input.X}` instead of a
  bare declared input -> loud.

A clean flow (seed 01) passes. Case data-refs are validated from the desugared
`CaseNode.inputs` (the `__rN`/`__on` sources, e.g. a dangling `${typo.output}`); the
rewritten node-local `${__rN}`/`${__on}` are EXCLUDED from `_classify_path`.
"""

from pathlib import Path

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.contracts import ContractOrigins, ValueContract
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.parse.file import (
    CaseDescriptor,
    parse_file,
)
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.helpers.types import read_flow_inputs
from agent_composer.compose.source import SourceDocument
from agent_composer.compose.validate.references import validate_references
from agent_composer.events import SourceRange
from tests.engine._fakes import derive_wiring, stamp_reads

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _seed_diagnostics(seed: str) -> LoadDiagnostics:
    """Return the source authority for one seed file."""
    path = _SEEDS / seed
    return LoadDiagnostics(label=str(path), text=path.read_text())


def _load(seed: str):
    """(nodes, flow_inputs, producers, outputs, flow_wiring) for a seed — every descriptor
    built kind-blind via `desc.build(...)`. A `case`'s lowering (its `__rN`/`__on` split) is
    part of its own build, so its rewritten `.node`/`.wiring` come out of the same call."""
    path = _SEEDS / seed
    document = SourceDocument.parse(path.read_text(), label=str(path))
    f = parse_file(document)
    descriptors = parse_nodes(document.root.child("nodes"))
    # First pass: build leaf nodes to learn the producer output contracts.
    nodes: dict = {}
    flow_wiring: dict = {}
    for nid, d in descriptors.items():
        if not isinstance(d, CaseDescriptor):
            result = d.build(BuildContext({}))
            nodes[nid] = result.node
            flow_wiring[nid] = result.wiring
    producers = {
        nid: n.output_contract
        for nid, n in nodes.items()
        if n.output_contract is not None
    }
    # Second pass: build the cases (their on:/when: type-legality reads producers).
    for nid, d in descriptors.items():
        if isinstance(d, CaseDescriptor):
            result = d.build(BuildContext(producers))
            nodes[nid] = result.node
            flow_wiring[nid] = result.wiring
    flow_inputs = {decl.name for decl in read_flow_inputs(f.inputs, {})}
    return nodes, flow_inputs, producers, f.outputs, flow_wiring


# --------------------------------------------------------------------------- #
# clean flow passes
# --------------------------------------------------------------------------- #


def test_clean_seed01_passes():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("01-structured-agent.yaml")
    validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)  # no raise


def test_clean_case_seed02_passes():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("02-case.yaml")
    validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)  # no raise


def test_clean_case_on_seed06_passes():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("06-case-on.yaml")
    validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)  # no raise


# --------------------------------------------------------------------------- #
# e01 — dangling flow-output reference
# --------------------------------------------------------------------------- #


def test_e01_dangling_flow_output_is_loud():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("errors/e01-undeclared-ref.yaml")
    with pytest.raises(LoadError) as exc:
        validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)
    msg = str(exc.value)
    assert "scor" in msg  # the typo'd target is named
    assert "flow output" in msg


def test_dangling_flow_output_uses_supplied_range():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load(
        "errors/e01-undeclared-ref.yaml"
    )
    expected = SourceRange(31, 4, 31, 24)

    with pytest.raises(LoadError) as exc:
        validate_references(
            nodes,
            flow_inputs,
            producers,
            outputs,
            flow_wiring,
            outputs_range=expected,
            diagnostics=_seed_diagnostics("errors/e01-undeclared-ref.yaml"),
        )

    assert exc.value.diagnostic.frames[0].range == expected


def test_aggregate_location_uses_first_available_range():
    node = AgentNode("broken", prompt="fixed")
    node.output_contract = ValueContract(Type.scalar(ValueKind.STRING))
    stamp_reads(node, {"value": "${missing_node.output}"})
    nodes = {"broken": node}
    flow_wiring = derive_wiring(nodes)
    later_output_range = SourceRange(20, 9, 20, 33)

    with pytest.raises(LoadError) as exc:
        validate_references(
            nodes,
            set(),
            {},
            "${missing_output.output}",
            flow_wiring,
            outputs_range=later_output_range,
            diagnostics=LoadDiagnostics(
                label="flow.yaml",
                text=("x" * 80 + "\n") * 30,
            ),
        )

    assert exc.value.diagnostic.frames[0].range == later_output_range


# --------------------------------------------------------------------------- #
# e03 — dotted-field miss on an anonymous producer record (anon records ARE checked)
# --------------------------------------------------------------------------- #


def test_e03_unknown_field_on_anon_record_is_loud():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("errors/e03-unknown-field.yaml")
    # the anonymous {rating, rationale} producer is a CHECKED record.
    assert producers["score"].type.fields is not None
    assert set(producers["score"].type.fields) == {"rating", "rationale"}
    with pytest.raises(LoadError) as exc:
        validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)
    msg = str(exc.value)
    assert "confidence" in msg  # the unknown field is named


def test_reference_walk_projects_type_without_replacing_contract_origins():
    """Reference checking reads semantic type without consuming provenance."""
    output_type = Type(
        kind=ValueKind.OBJECT,
        fields={"rating": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"rating"}),
    )
    contract = ValueContract(
        output_type,
        ContractOrigins(default=OriginRef("producer-output")),
    )
    score = AgentNode("score", prompt="fixed")
    sink = AgentNode("sink", prompt="fixed")
    stamp_reads(sink, {"value": "${score.output.rating}"})
    nodes = {"score": score, "sink": sink}

    validate_references(
        nodes,
        set(),
        {"score": contract},
        None,
        derive_wiring(nodes),
    )

    assert contract.type is output_type
    assert contract.origins.default == OriginRef("producer-output")


def test_reference_walk_rejects_non_contract_producer_map_values():
    """The Compose validator boundary accepts canonical carriers, not bare Types."""
    with pytest.raises(TypeError, match="producer contracts.*ValueContract"):
        validate_references(  # type: ignore[dict-item]
            {"score": AgentNode("score")},
            set(),
            {"score": Type.scalar(ValueKind.STRING)},
            None,
            {"score": {}},
        )


# --------------------------------------------------------------------------- #
# prompt — a prompt may only interpolate bare declared inputs
# --------------------------------------------------------------------------- #


def test_prompt_l1_pool_ref_is_loud():
    # an AGENT prompt that interpolates ${X.output} (a pool ref) -> loud.
    node = AgentNode("bad", prompt="Summarize ${upstream.output} for the user.")
    node.output_contract = ValueContract(Type.scalar(ValueKind.STRING))
    stamp_reads(node, {})  # no declared inputs
    nodes = {"bad": node}
    flow_wiring = derive_wiring(nodes)
    with pytest.raises(LoadError) as exc:
        validate_references(nodes, set(), {}, None, flow_wiring)
    msg = str(exc.value)
    # the head is the node id (`upstream`), not `outputs`. The error names the ref
    # and notes it isn't a declared input.
    assert "${upstream.output}" in msg or "upstream" in msg
    assert "bad" in msg  # the offending node is named


def test_prompt_l1_declared_input_passes():
    node = AgentNode("ok", prompt="Summarize ${topic} for the user.")
    node.output_contract = ValueContract(Type.scalar(ValueKind.STRING))
    stamp_reads(node, {"topic": "${input.topic}"})  # stamps params + the fake wiring
    nodes = {"ok": node}
    flow_wiring = derive_wiring(nodes)
    validate_references(nodes, {"topic"}, {}, None, flow_wiring)  # no raise


# --------------------------------------------------------------------------- #
# case data-refs validated from the DESUGARED CaseNode.inputs
# --------------------------------------------------------------------------- #


def _desugared_case_nodes(when_or_on: dict):
    """Build a tiny flow: a `gate` case whose lowered __rN/__on source is dangling.
    Returns (nodes, flow_wiring) — the gate's __rN sources live on flow.wiring now."""
    desc = CaseDescriptor(id="gate", **when_or_on)
    result = desc.build(BuildContext({}))
    return {"gate": result.node}, {"gate": result.wiring}


def test_case_dangling_data_ref_is_loud():
    # a searched case whose when: reads ${typo.output} (no such node) -> the desugared
    # __r0 source is the dangling ref; validation fires on the flow.wiring source.
    nodes, flow_wiring = _desugared_case_nodes(
        {"cases": [{"when": "${typo.output} >= 0.5", "then": "yes"}], "else_": "no"}
    )
    with pytest.raises(LoadError) as exc:
        validate_references(nodes, set(), {}, None, flow_wiring)
    msg = str(exc.value)
    assert "typo" in msg  # the dangling producer is named


def test_case_node_local_ref_does_not_error():
    # the rewritten node-local __r0 is EXCLUDED from _classify_path — a case whose
    # source IS a real node passes (the node-local rewrite never trips "unknown node").
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("02-case.yaml")
    # seed 02 gate: param __r0 wired to ${score.output} (score is real); when uses __r0.
    gate = nodes["gate"]
    assert [p.name for p in gate.params] == ["__r0"]
    assert flow_wiring["gate"] == {"__r0": "${score.output}"}
    assert gate.cases[0].when == "__r0 >= 0.5"  # node-local — must NOT be classified
    validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)  # no raise
