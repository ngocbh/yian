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

from agent_composer.nodes.agent import AgentNode
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.build import build_leaf_node, infer_data_edges
from agent_composer.compose.cases import desugar_case, reconcile_case_edges
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.parse.file import (
    AgentDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    parse_file,
)
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.helpers.types import read_flow_inputs
from agent_composer.compose.stages.references import validate_references
from tests.engine._fakes import derive_wiring, stamp_reads

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _load(seed: str):
    """(nodes, flow_inputs, producers, outputs, flow_wiring) for a seed — leaf + desugared cases."""
    f = parse_file((_SEEDS / seed).read_text())
    descriptors = parse_nodes(NodeReader.spanless(f.nodes))
    built = {
        nid: build_leaf_node(d, {})
        for nid, d in descriptors.items()
        if isinstance(d, (AgentDescriptor, CodeDescriptor))
    }
    leaf = {nid: n for nid, (n, _) in built.items()}
    flow_wiring = {nid: w for nid, (_, w) in built.items()}
    producers = {nid: n.output_type for nid, n in leaf.items() if n.output_type is not None}
    desugars = {
        nid: desugar_case(d, producers)
        for nid, d in descriptors.items()
        if isinstance(d, CaseDescriptor)
    }
    nodes = dict(leaf)
    for nid, d in desugars.items():
        nodes[nid] = d.node
        flow_wiring[nid] = d.wiring
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


# --------------------------------------------------------------------------- #
# e03 — dotted-field miss on an anonymous producer record (anon records ARE checked)
# --------------------------------------------------------------------------- #


def test_e03_unknown_field_on_anon_record_is_loud():
    nodes, flow_inputs, producers, outputs, flow_wiring = _load("errors/e03-unknown-field.yaml")
    # the anonymous {rating, rationale} producer is a CHECKED record.
    assert producers["score"].fields is not None
    assert set(producers["score"].fields) == {"rating", "rationale"}
    with pytest.raises(LoadError) as exc:
        validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)
    msg = str(exc.value)
    assert "confidence" in msg  # the unknown field is named


# --------------------------------------------------------------------------- #
# prompt — a prompt may only interpolate bare declared inputs
# --------------------------------------------------------------------------- #


def test_prompt_l1_pool_ref_is_loud():
    # an AGENT prompt that interpolates ${X.output} (a pool ref) -> loud.
    node = AgentNode("bad", prompt="Summarize ${upstream.output} for the user.")
    node.output_type = Type.scalar(ValueKind.STRING)
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
    node.output_type = Type.scalar(ValueKind.STRING)
    stamp_reads(node, {"topic": "${input.topic}"})  # stamps params + the fake wiring
    nodes = {"ok": node}
    flow_wiring = derive_wiring(nodes)
    validate_references(nodes, {"topic"}, {}, None, flow_wiring)  # no raise


# --------------------------------------------------------------------------- #
# case data-refs validated from the DESUGARED CaseNode.inputs
# --------------------------------------------------------------------------- #


def _desugared_case_nodes(when_or_on: dict):
    """Build a tiny flow: a `gate` case whose desugared __rN/__on source is dangling.
    Returns (nodes, flow_wiring) — the gate's __rN sources live on flow.wiring now."""
    desc = CaseDescriptor(id="gate", **when_or_on)
    desugar = desugar_case(desc, {})
    return {"gate": desugar.node}, {"gate": desugar.wiring}


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
