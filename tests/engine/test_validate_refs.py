"""Unit tests for reference-wiring validation.

`validate_references` runs the reused leaf checker (`compile.validation._classify_path`)
and the shared field-read walk over every binding site of a built flow — node input
bindings from authored node `input:`, flow `output:` bindings, and (via the desugared
`CaseNode.inputs`)
each `case`'s `on:`/`when:` data refs. It accumulates ALL located errors and raises a
single `LoadError`. Three loud mechanisms are pinned:

- **e01** — a dangling flow-output ref (`${scor.output}` typo) -> loud.
- **e03** — a dotted-field miss on an ANONYMOUS producer record (`{rating, rationale}`,
  read natively via `read_type`) -> loud (anonymous records ARE checked).
- **prompt** — an AGENT prompt interpolating `${X.output}`/`${input.X}` instead of a
  bare declared input -> loud.

A clean flow (seed 01) passes. Case data-refs are validated from the desugared
`CaseNode.inputs` (the `__rN`/`__on` sources, e.g. a dangling `${typo.output}`); the
rewritten node-local `${__rN}`/`__on` are EXCLUDED from `_classify_path`.
"""

from pathlib import Path

import pytest

from yian.compose import load_flow
from yian.diagnostics import OriginRef
from yian.nodes.agent import AgentNode
from yian.nodes.contracts import ContractOrigins, ValueContract
from yian.typesys.values import ValueKind, Type
from yian.compose.nodes.base import BuildContext, BuildResult
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.parse.file import (
    CaseDescriptor,
    parse_file,
)
from yian.compose.parse.nodes import parse_nodes
from yian.compose.helpers.types import read_flow_inputs
from yian.compose.source import SourceDocument
from yian.compose.validate.references import validate_references
from yian.diagnostics import SourceRange
from tests.engine._fakes import derive_wiring, stamp_reads

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _seed_diagnostics(seed: str) -> LoadDiagnostics:
    """Return the source authority for one seed file."""
    path = _SEEDS / seed
    return LoadDiagnostics(label=str(path), text=path.read_text())


def _load(seed: str):
    """Return build products and validation inputs for one seed flow.

    Every descriptor
    built kind-blind via `desc.build(...)`. A `case`'s lowering (its `__rN`/`__on` split) is
    part of its own build, so its rewritten `.node`/`.wiring` come out of the same call.
    """
    path = _SEEDS / seed
    document = SourceDocument.parse(path.read_text(), label=str(path))
    f = parse_file(document)
    descriptors = parse_nodes(document.root.child("nodes"))
    # First pass: build leaf nodes to learn the producer output contracts.
    build_results: dict[str, BuildResult] = {}
    flow_wiring: dict = {}
    for nid, d in descriptors.items():
        if not isinstance(d, CaseDescriptor):
            result = d.build(BuildContext({}))
            build_results[nid] = result
            flow_wiring[nid] = result.wiring
    producers = {
        nid: result.node.output_contract
        for nid, result in build_results.items()
        if result.node.output_contract is not None
    }
    # Second pass: build the cases (their on:/when: type-legality reads producers).
    for nid, d in descriptors.items():
        if isinstance(d, CaseDescriptor):
            result = d.build(BuildContext(producers))
            build_results[nid] = result
            flow_wiring[nid] = result.wiring
    flow_inputs = {decl.name for decl in read_flow_inputs(f.inputs, {})}
    return build_results, flow_inputs, producers, f.outputs, flow_wiring


def _source_free_results(nodes: dict[str, object]) -> dict[str, BuildResult]:
    """Wrap hand-built semantic nodes for a direct Compose validator call."""
    return {
        node_id: BuildResult(node=node, wiring={})  # type: ignore[arg-type]
        for node_id, node in nodes.items()
    }


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


_WAIT_REFERENCE_FLOW = """\
id: wait_refs
name: wait_refs
input:
  deadline: datetime
nodes:
  clock:
    kind: code
    output: datetime
    code: |
      return "2026-01-01T00:00:00Z"
  settle:
    kind: wait
    until: SOURCE
output: ${settle.output}
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("${missing.output}", "unknown namespace 'missing'"),
        ("${input.missing}", "unknown flow input 'missing'"),
        ("${system.tomorrow}", "unknown system ambient 'tomorrow'"),
        ("${item}", "unknown namespace 'item'"),
    ],
)
def test_wait_until_rejects_invalid_references_at_load(source: str, message: str):
    with pytest.raises(LoadError) as exc:
        load_flow(_WAIT_REFERENCE_FLOW.replace("SOURCE", source))

    assert message in str(exc.value)
    assert exc.value.diagnostic.frames[-1].range.start_line == 13


@pytest.mark.parametrize(
    "source",
    [
        "2026-01-01T00:00:00Z",
        "${input.deadline}",
        "${clock.output}",
        "${system.now}",
    ],
)
def test_wait_until_accepts_valid_reference_sources(source: str):
    load_flow(_WAIT_REFERENCE_FLOW.replace("SOURCE", source))


_SYSTEM_FIELD_FLOW = """\
id: system_fields
name: system_fields
nodes:
  read:
    kind: code
    input:
      t: SOURCE
    output: str
    code: |
      return str(inputs["t"])
output: ${read.output}
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("${system.today.year}", "reads field 'year' on type date, which has no fields"),
        ("${system.now.hour}", "reads field 'hour' on type datetime, which has no fields"),
        ("${system.run_id.x.y}", "reads field 'x' on type str, which has no fields"),
    ],
)
def test_system_ambient_field_read_is_a_load_error(source: str, message: str):
    with pytest.raises(LoadError) as exc:
        load_flow(_SYSTEM_FIELD_FLOW.replace("SOURCE", source))

    assert message in str(exc.value)


def test_system_ambient_field_read_in_case_when_is_a_load_error():
    flow = """\
id: system_case
name: system_case
nodes:
  route:
    kind: case
    cases:
      - when: system.today.year == "2026"
        then: done
    else: done
  done:
    kind: code
    output: int
    code: |
      return 1
output: ${done.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(flow)

    assert "reads field 'year' on type date, which has no fields" in str(exc.value)


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
            _source_free_results(nodes),
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
        _source_free_results(nodes),
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
            {"score": BuildResult(node=AgentNode("score"), wiring={})},
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
        validate_references(_source_free_results(nodes), set(), {}, None, flow_wiring)
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
    validate_references(
        _source_free_results(nodes),
        {"topic"},
        {},
        None,
        flow_wiring,
    )  # no raise


# --------------------------------------------------------------------------- #
# case data-refs validated from the DESUGARED CaseNode.inputs
# --------------------------------------------------------------------------- #


def _desugared_case_nodes(when_or_on: dict):
    """Build a tiny flow: a `gate` case whose lowered __rN/__on source is dangling.
    Returns (nodes, flow_wiring) — the gate's __rN sources live on flow.wiring now."""
    desc = CaseDescriptor(id="gate", **when_or_on)
    result = desc.build(BuildContext({}))
    return {"gate": result}, {"gate": result.wiring}


def test_case_dangling_data_ref_is_loud():
    # a searched case whose when: reads ${typo.output} (no such node) -> the desugared
    # __r0 source is the dangling ref; validation fires on the flow.wiring source.
    nodes, flow_wiring = _desugared_case_nodes(
        {"cases": [{"when": "typo.output >= 0.5", "then": "yes"}], "else_": "no"}
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
    gate = nodes["gate"].node
    assert [p.name for p in gate.params] == ["__r0"]
    assert flow_wiring["gate"] == {"__r0": "${score.output}"}
    assert gate.cases[0].when == "__r0 >= 0.5"  # node-local — must NOT be classified
    validate_references(nodes, flow_inputs, producers, outputs, flow_wiring)  # no raise
