"""Integration tests for the top-level loader.

`load_flow(text) -> ComposedFlow` wires every prior slice into one runnable
`CompiledFlow` + canonical flow parameters + the boundary/post-terminal assert split. The
load-test set is the seeds: flat (00/01), case (02/06), the
arithmetic asserts (10), anchors (11), MODEL (07), AGENT knobs (14), nested records +
record-typed inputs (13), the rich integration (18), and every-binding-stance (19).

EXCLUDED (recorded reasons):
  (15/16 + errors/e04/e05 — the dropped `kind: match`/payload-union design — were deleted;
  tagged data is modelled as a discriminant record + `case … on <field>`.)
- 04/05 — `call` nodes need a child resolver (asserted loud below; the resolver'd
  load+compile path lives in tests/engine/test_ref_map.py).
"""

from pathlib import Path

import pytest

from agent_composer.compile.model import END_ID, START_ID, CompiledFlow
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.contracts import Binding
from agent_composer.nodes.case import CaseNode
from agent_composer.nodes.model import ModelNode
from agent_composer.typesys.values import ValueKind
from agent_composer.compose import ComposedFlow, LoadError, load_flow

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"

# Each loadable seed must load to a ComposedFlow.
_LOADABLE = ["00", "01", "02", "06", "07", "09", "10", "11", "12", "13", "14", "17", "18", "19", "20", "21", "22", "23", "27", "28", "29", "30"]
_SEED_FILES = {
    "00": "00-hello-agent.yaml",
    "01": "01-structured-agent.yaml",
    "02": "02-case.yaml",
    "06": "06-case-on.yaml",
    "07": "07-model-rating.yaml",
    # 09 — ${...} operator forms incl. the ${system.run_id} ambient (now allow-listed).
    "09": "09-interpolation-ops.yaml",
    # 12 — depends_on/runs_after ordering edges (feature D); now wired, not parsed-only.
    "12": "12-depends-on.yaml",
    "10": "10-asserts-arithmetic.yaml",
    "11": "11-reuse-anchors.yaml",
    "13": "13-types-objects.yaml",
    "14": "14-agent-tools.yaml",
    # 17 loads resolver-free — effects (human_input/wait); promoted from _future (feature E).
    "17": "17-effects-human-wait.yaml",
    "18": "18-research-pipeline.yaml",
    "19": "19-binding-stances.yaml",
    # 20 loads resolver-free — its `call` targets an in-file `defs:` callable.
    "20": "20-call-defs.yaml",
    # 21 loads resolver-free — an INLINE call desugars to a synth call on the in-file def.
    "21": "21-inline-call.yaml",
    # 22 — the value-case: ${gate.output} desugars to the branch coalesce.
    "22": "22-case-value.yaml",
    # 23 — asserts at every scope (flow boundary/post + inline-call, def-child, node pre/post).
    "23": "23-asserts-scopes.yaml",
    # 27 — every unified-${...} expression form (arith/str/list/bool/builtins) in bindings/prompt/asserts.
    "27": "27-expr-ops.yaml",
    # 28 — the `loop` node (until: + max:) over an in-file `defs:` body; loads resolver-free.
    "28": "28-refine-loop.yaml",
    # 29 — inline `code:` source (a bare body read as `def main(inputs)`, run in-process).
    "29": "29-inline-code.yaml",
    # 30 — reference `code:` source (a `module:function` imported and called in-process).
    "30": "30-reference-code.yaml",
}


def _load(seed: str) -> ComposedFlow:
    return load_flow((_SEEDS / _SEED_FILES[seed]).read_text())


# --------------------------------------------------------------------------- #
# every loadable seed loads to a ComposedFlow (a CompiledFlow + inputs + asserts)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", _LOADABLE)
def test_seed_loads_to_loaded_flow(seed):
    loaded = _load(seed)
    assert isinstance(loaded, ComposedFlow)
    assert isinstance(loaded.compiled, CompiledFlow)
    assert loaded.compiled.start_id in loaded.compiled.nodes  # synthesized entry node present
    # an __end__ edge exists (the flow has a codomain).
    assert any(e.to == END_ID for e in loaded.compiled.edges)


# --------------------------------------------------------------------------- #
# seed 02 — case desugar: CaseNode gate + control edges + coalesce flow-output
# --------------------------------------------------------------------------- #


def test_seed02_gate_is_case_with_control_edges():
    loaded = _load("02")
    flow = loaded.compiled
    gate = flow.nodes["gate"]
    assert isinstance(gate, CaseNode)
    # gate -> positive / gate -> cautious control edges (the then/else targets).
    out = {(e.to, e.source_handle) for e in flow.outgoing("gate")}
    assert ("positive", "positive") in out
    assert ("cautious", "default") in out  # else: -> the internal default handle
    # coalesce flow-output: BOTH branches reach __end__.
    end_producers = {e.from_ for e in flow.edges if e.to == END_ID}
    assert end_producers == {"positive", "cautious"}
    # bare whole-string `outputs:` -> one FlowOutput (terminal_output returns bare value).
    assert len(flow.outputs) == 1
    assert flow.outputs[0].from_ == "${positive.output | cautious.output}"


def test_seed02_start_feeds_input_readers_branch_targets_control_gated():
    # every ${input.topic} reader (score/positive/cautious) gets a START_ID data edge
    # (input_group="topic"); the branch targets are ALSO control-gated by `gate` (the veto still
    # skip-floods the untaken branch). The single root is the synthesized START_ID node.
    flow = _load("02").compiled
    assert flow.start_id == START_ID
    start_data = {e.to for e in flow.edges if e.from_ == START_ID and e.input_group == "topic"}
    assert start_data == {"score", "positive", "cautious"}
    control = {e.to for e in flow.edges if e.from_ == "gate" and e.source_handle is not None}
    assert control == {"positive", "cautious"}


# --------------------------------------------------------------------------- #
# seed 13 — nested record output_type + a dict-valued (record-typed) flow input
# --------------------------------------------------------------------------- #


def test_seed13_nested_record_output_and_dict_input():
    loaded = _load("13")
    plan = loaded.compiled.nodes["plan"]
    assert isinstance(plan, CodeNode)
    typ = plan.output_type
    assert typ.kind == ValueKind.OBJECT
    # summary is a nested record; meta is nested again (two levels deep).
    summary = typ.fields["summary"]
    assert summary.kind == ValueKind.OBJECT
    meta = summary.fields["meta"]
    assert meta.kind == ValueKind.OBJECT
    assert set(meta.fields) == {"as_of", "dry_run"}
    # the record-typed `config:` flow input resolves to a record Type.
    config = next(d for d in loaded.input if d.name == "config")
    assert config.contract is not None
    assert config.contract.type.kind == ValueKind.OBJECT
    assert set(config.contract.type.fields) == {"regroup", "bands"}


# --------------------------------------------------------------------------- #
# seed 11 — anchored nodes all built (YAML <<: merge expanded before the schema)
# --------------------------------------------------------------------------- #


def test_seed11_anchored_nodes_built():
    flow = _load("11").compiled
    for nid in ("pro", "con", "judge"):
        node = flow.nodes[nid]
        assert isinstance(node, AgentNode)
        assert node.output_type.kind == ValueKind.STRING  # from the anchor
        assert node.llm_config is not None  # llm_config merged in from the anchor


# --------------------------------------------------------------------------- #
# seed 07 — MODEL fields (model_id / weights_uri / runtime)
# --------------------------------------------------------------------------- #


def test_seed07_model_fields():
    flow = _load("07").compiled
    predict = flow.nodes["predict"]
    assert isinstance(predict, ModelNode)
    assert predict.model_id == "topic-ranker-v1"
    assert predict.weights_uri == "manifold://calpha/models/topic-ranker-v1.pt"
    assert predict.runtime_name == "torchscript"
    # object output {score, rank}.
    assert set(predict.output_type.fields) == {"score", "rank"}


# --------------------------------------------------------------------------- #
# seed 14 — AGENT knobs (mode / tools / llm_config)
# --------------------------------------------------------------------------- #


def test_seed14_agent_knobs():
    flow = _load("14").compiled
    reviewer = flow.nodes["reviewer"]
    assert isinstance(reviewer, AgentNode)
    assert reviewer.mode == "tool_calling"
    assert reviewer.tools == ["get_web_data", "web_search", "ask_user"]
    assert reviewer.llm_config is not None
    assert reviewer.llm_config["provider"] == "anthropic"


# --------------------------------------------------------------------------- #
# seed 18 — the rich integration (the most likely to break)
# --------------------------------------------------------------------------- #


def test_seed18_synth_carries_view_record():
    flow = _load("18").compiled
    synth = flow.nodes["synth"]
    typ = synth.output_type
    assert typ.kind == ValueKind.OBJECT
    assert set(typ.fields) == {"stance", "claim", "confidence"}
    # stance is the Stance enum -> its Type carries the tags.
    assert typ.fields["stance"].tags == {"positive", "negative", "neutral"}
    assert typ.fields["confidence"].kind == ValueKind.NUMBER


def test_seed18_route_desugars_case_on_with_else():
    flow = _load("18").compiled
    route = flow.nodes["route"]
    assert isinstance(route, CaseNode)
    # on-form: one __on param wired to the stance ref (the source lives on flow.wiring).
    assert [p.name for p in route.params] == ["__on"]
    assert flow.wiring["route"] == {"__on": Binding("${synth.output.stance}")}
    out = {(e.to, e.source_handle) for e in flow.outgoing("route")}
    assert ("pro_note", "pro_note") in out
    assert ("con_note", "con_note") in out
    assert ("neutral_note", "default") in out  # else: -> default handle


def test_seed18_multi_output_with_coalesce():
    flow = _load("18").compiled
    # multi-output object: stance/confidence/note (each a named FlowOutput).
    names = {o.name for o in flow.outputs}
    assert names == {"stance", "confidence", "note"}
    # coalesce note -> all three note producers reach __end__.
    end_producers = {e.from_ for e in flow.edges if e.to == END_ID}
    assert end_producers == {"synth", "pro_note", "con_note", "neutral_note"}


def test_seed18_post_terminal_assert_against_view():
    loaded = _load("18")
    # the confidence-range assert reads ${synth.output.confidence} -> post-terminal, so it is
    # LOWERED onto the END node (`post_asserts`), not START; the START carries no boundary asserts.
    # Validated against the View record (load succeeds, not lenient-skipped).
    start = loaded.compiled.nodes[START_ID]
    end = loaded.compiled.nodes[END_ID]
    assert start.post_asserts == []
    assert end.post_asserts != []
    # `confidence` survives in executable and author-facing carrier text.
    assert any(
        "confidence" in assertion.expression or "confidence" in assertion.display
        for assertion in end.post_asserts
    )


# --------------------------------------------------------------------------- #
# seed 10 — boundary asserts split (inputs/system-only -> boundary)
# --------------------------------------------------------------------------- #


def test_seed10_asserts_are_boundary():
    loaded = _load("10")
    # all four asserts read only ${input.X} -> boundary, so they ride START's ordinary
    # `post_asserts` (rewritten to `${output.X}`); END carries no post asserts.
    start = loaded.compiled.nodes[START_ID]
    end = loaded.compiled.nodes[END_ID]
    assert start.post_asserts != []
    assert end.post_asserts == []


# --------------------------------------------------------------------------- #
# seed 19 — every binding stance loads clean (compile-time)
# --------------------------------------------------------------------------- #


def test_seed19_loads_clean():
    loaded = _load("19")
    # the assembler exercises every stance; all compile + name-check.
    sources = loaded.compiled.wiring["assemble"]
    assert sources["claim"] == Binding("${pro.output | con.output}")
    assert sources["detail"] == Binding("${pro_detail.output:-null}")
    assert sources["when"] == Binding("${input.as_of:-${system.today}}")
    assert sources["topic"] == Binding('${input.topic:?"a topic is required"}')


# --------------------------------------------------------------------------- #
# exclusions — `call` needs a child resolver; the call/map seeds not loaded
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", ["04-call.yaml", "05-call-map.yaml"])
def test_ref_map_without_resolver_is_loud(seed):
    # A `call` node is built given a child resolver; without one it is loud (the
    # resolver-required path). The resolver'd load+compile path is covered in
    # tests/engine/test_ref_map.py.
    with pytest.raises(LoadError) as exc:
        load_flow((_SEEDS / seed).read_text())
    assert "resolver" in str(exc.value).lower()


def test_depends_on_seed_emits_ordering_edge():
    # seed 12 is wired (feature D): `fetch depends_on [warm_cache]` -> an ordering edge.
    loaded = load_flow((_SEEDS / "12-depends-on.yaml").read_text())
    assert isinstance(loaded, ComposedFlow)
    ordering = [e for e in loaded.compiled.edges if e.ordering]
    assert len(ordering) == 1
    e = ordering[0]
    assert (e.from_, e.to) == ("warm_cache", "fetch")
    assert e.optional is False  # depends_on -> co-skip


def test_loaded_flow_carries_version():
    from agent_composer import load_flow
    lf = load_flow(
        'id: x\nname: x\nversion: v2\nnodes:\n  a: {kind: code, code: m:f}\n'
        'output: {r: "${a.output}"}\n'
    )
    assert lf.version == "v2"


def test_loaded_flow_version_none_when_absent():
    from agent_composer import load_flow
    lf = load_flow(
        'id: x\nname: x\nnodes:\n  a: {kind: code, code: m:f}\n'
        'output: {r: "${a.output}"}\n'
    )
    assert lf.version is None


def test_loaded_flow_carries_name_and_description():
    from agent_composer import load_flow
    lf = load_flow(
        'id: x\nname: my_flow\ndescription: Does a thing.\n'
        'nodes:\n  a: {kind: code, code: m:f}\n'
        'output: {r: "${a.output}"}\n'
    )
    assert lf.name == "my_flow"
    assert lf.description == "Does a thing."


def test_loaded_flow_description_none_when_absent():
    from agent_composer import load_flow
    lf = load_flow(
        'id: x\nname: x\nnodes:\n  a: {kind: code, code: m:f}\n'
        'output: {r: "${a.output}"}\n'
    )
    assert lf.description is None
