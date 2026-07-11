"""Step 7 — the `compose/stages/` graph-stage axis: cross-node assembly passes.

7.0 scaffold: the package exists with its charter. Later sub-tasks (7.1+) populate it with
one relocated assembly pass each, driven off the span tree's `.range`/`field_ranges`.

7.1 — `stages/parse.parse_nodes`: the `nodes:` mapping pass. Consumes a `NodeReader` over
the span tree (not `yaml.safe_load` + `node_lines`) and dispatches each body to its kind's
`parse` via `KINDS`, returning located descriptors that carry `field_ranges`."""

import pytest

import agent_composer.compose.stages as stages
from agent_composer.compile.model import Edge
from agent_composer.compose.build import build_leaf_node
from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds.agent import AgentDescriptor
from agent_composer.compose.kinds.code import CodeDescriptor
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.spanloader import load_with_spans
from agent_composer.compose.stages.adaptive_questions import desugar_adaptive_questions
from agent_composer.compose.stages.case_targets import desugar_case_call_targets
from agent_composer.compose.stages.cycles import reject_cycles
from agent_composer.compose.stages.inline_calls import (
    _synth_id_gen,
    desugar_inline_calls,
)
from agent_composer.compose.stages.ordering import infer_ordering_edges
from agent_composer.compose.stages.parse import parse_nodes
from agent_composer.compose.stages.references import validate_references
from agent_composer.events import SourceRange


def test_stages_package_imports():
    assert stages.__doc__ is not None


def _nodes_reader(text):
    """A span-aware NodeReader over the `nodes:` mapping of a flow document."""
    value, spans, root = load_with_spans(text, return_root=True)
    return NodeReader(value, spans, root).child("nodes")


_FLOW = """\
id: f
name: f
nodes:
  writer:
    kind: agent
    prompt: "Write about ${topic}"
    input:
      topic: ${input.topic}
    output: string
output: ${writer.output}
"""


def test_parse_nodes_uses_span_tree():
    descriptors = parse_nodes(_nodes_reader(_FLOW))

    desc = descriptors["writer"]
    # The descriptor is exactly today's AgentDescriptor (field_ranges excluded from ==);
    # the singular `input:`/`output:` source keys are back-mapped to the plural fields.
    assert desc == AgentDescriptor(
        id="writer",
        prompt="Write about ${topic}",
        inputs={"topic": "${input.topic}"},
        outputs="string",
    )
    # Located: the span tree stamps ranges onto the descriptor + its fields.
    assert desc.range is not None
    assert desc.field_ranges["prompt"].start_line == 6  # 1-based, the prompt line
    # The back-mapped plural fields resolve their source ranges (via the rewritten key).
    assert desc.field_ranges["inputs"] is not None
    assert desc.field_ranges["outputs"] is not None


def test_parse_nodes_unknown_kind_is_located():
    text = "id: f\nname: f\nnodes:\n  n:\n    kind: wizard\n    output: str\noutput: 1\n"
    with pytest.raises(LoadError) as exc:
        parse_nodes(_nodes_reader(text))
    assert "wizard" in str(exc.value)
    assert exc.value.range is not None  # located at the `kind:` value


def test_parse_nodes_reserved_node_id_is_loud():
    text = "id: f\nname: f\nnodes:\n  input:\n    kind: agent\n    prompt: hi\noutput: 1\n"
    with pytest.raises(LoadError, match="reserved"):
        parse_nodes(_nodes_reader(text))


def test_parse_nodes_spanless_reader():
    # A spanless reader (defs / compact / in-memory) parses the already-plural body with
    # no source ranges — descriptors still build, ranges are None.
    reader = NodeReader.spanless(
        {"w": {"kind": "agent", "prompt": "hi", "inputs": {"x": "${input.x}"}, "outputs": "str"}}
    )
    desc = parse_nodes(reader)["w"]
    assert desc == AgentDescriptor(id="w", prompt="hi", inputs={"x": "${input.x}"}, outputs="str")
    assert desc.range is None
    assert desc.field_ranges == {}


# ---------- 7.2 desugar_inline_calls: synth call nodes inherit host ranges ----------


def test_desugar_inline_calls_preserves_ranges():
    # A whole-value `call(...)` in a node's `input:` binding lifts to a synth `call` node.
    # The synth CallDescriptor must inherit the HOST node's range/field_ranges (so its
    # downstream errors locate at the host binding site) — no `node_lines` map needed.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  writer:\n    kind: code\n    code: m:f\n"
        "    input:\n      x: call(helper, n=3)\n    output: str\n"
        "output: ${writer.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    host = descriptors["writer"]
    new_descriptors, _outputs, _asserts = desugar_inline_calls(
        descriptors, None, next_id=_synth_id_gen()
    )
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert call_desc.range == host.range
    assert call_desc.field_ranges == host.field_ranges


# ---------- 7.3 desugar_adaptive_questions: synth composer inherits gate ranges ----------


def test_desugar_adaptive_questions_preserves_ranges():
    # An adaptive human_input gate lifts a synth composer agent (`<gate>__compose`).
    # The synth agent must inherit the gate node's range/field_ranges (no node_lines map).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  ask:\n    kind: human_input\n"
        "    adaptive_questions:\n      prompt: Ask about ${input.topic}\n"
        "    input:\n      topic: ${input.topic}\n"
        "output: ${ask.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    gate = descriptors["ask"]
    new_descriptors = desugar_adaptive_questions(descriptors)
    composer = new_descriptors["ask__compose"]
    assert composer.range == gate.range
    assert composer.field_ranges == gate.field_ranges


# ---------- 7.4 desugar_case_call_targets: synth branch-target call inherits case ranges ----------


def test_desugar_case_call_targets_preserves_ranges():
    # A `case` then:/else: that is a whole-value `call(...)` directive lifts a synth `call`
    # node. That synth node must inherit the HOST case descriptor's range/field_ranges (so its
    # downstream errors locate at the case node) — no `node_lines` map needed.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  route:\n    kind: case\n    on: ${score.output}\n"
        "    cases:\n      - when: high\n        then: call(helper, n=3)\n"
        "    else: cold\n"
        "output: ${route.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    host = descriptors["route"]
    new_descriptors = desugar_case_call_targets(descriptors, _synth_id_gen())
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert call_desc.range == host.range
    assert call_desc.field_ranges == host.field_ranges


# ---------- 7.5 infer_ordering_edges: unknown target located at the declaring node ----------


def test_infer_ordering_edges_unknown_target_is_located():
    # A `depends_on:`/`runs_after:` that names an unknown node is a LoadError located at the
    # DECLARING node descriptor's range (no `node_lines` map).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  a:\n    kind: code\n    code: m:f\n    depends_on: [ghost]\n    output: str\n"
        "output: ${a.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    with pytest.raises(LoadError) as exc:
        infer_ordering_edges(descriptors, set(descriptors))
    assert "ghost" in str(exc.value)
    assert exc.value.range == descriptors["a"].range


def test_infer_ordering_edges_builds_ordering_edges():
    # A valid `depends_on:` lowers to a zero-data ordering edge (optional=False).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  warm:\n    kind: code\n    code: m:f\n    output: str\n"
        "  a:\n    kind: code\n    code: m:f\n    depends_on: [warm]\n    output: str\n"
        "output: ${a.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    edges = infer_ordering_edges(descriptors, set(descriptors))
    assert len(edges) == 1
    assert edges[0].from_ == "warm"
    assert edges[0].to == "a"
    assert edges[0].ordering is True
    assert edges[0].optional is False


# ---------- 7.6 reject_cycles: the cycle error anchors on a stuck node's range ----------


def test_reject_cycles_diag_has_range():
    # A directed cycle (a -> b -> a) is loud; the LoadError anchors its range on the first
    # (sorted) stuck node that has one, taken from the passed node_ranges map.
    edges = [
        Edge(id="a->b", from_="a", to="b", input_group="x"),
        Edge(id="b->a", from_="b", to="a", input_group="y"),
    ]
    ra = SourceRange(start_line=2, start_col=3, end_line=2, end_col=8)
    rb = SourceRange(start_line=5, start_col=3, end_line=5, end_col=8)
    with pytest.raises(LoadError) as exc:
        reject_cycles(edges, {"a", "b"}, node_ranges={"a": ra, "b": rb})
    # `a` sorts first among the stuck nodes, so its range is the anchor.
    assert exc.value.range == ra
    # The "why" legend still spells out the loop's dependency edges.
    assert exc.value.notes is not None


def test_reject_cycles_no_ranges_leaves_range_none():
    # Without a node_ranges map the cycle is still rejected; the error simply carries no range.
    edges = [Edge(id="a->b", from_="a", to="b"), Edge(id="b->a", from_="b", to="a")]
    with pytest.raises(LoadError) as exc:
        reject_cycles(edges, {"a", "b"})
    assert exc.value.range is None


# ---------- 7.7 validate_references: a dangling ref carries the field's range ----------


def _built_flow(text):
    """Build leaf nodes + flow wiring + producers from a flow's `nodes:` (span-aware).

    Mirrors the loader's leaf-build slice: parse the `nodes:` mapping off the span tree,
    build each agent/code leaf, and split into `{nid: node}` / `{nid: wiring}` plus the
    `{nid: output_type}` producer map — enough to drive `validate_references` in isolation.
    """
    value, spans, root = load_with_spans(text, return_root=True)
    reader = NodeReader(value, spans, root).child("nodes")
    descriptors = parse_nodes(reader)
    built = {
        nid: build_leaf_node(d, {})
        for nid, d in descriptors.items()
        if isinstance(d, (AgentDescriptor, CodeDescriptor))
    }
    nodes = {nid: n for nid, (n, _) in built.items()}
    wiring = {nid: w for nid, (_, w) in built.items()}
    producers = {
        nid: n.output_type for nid, n in nodes.items() if n.output_type is not None
    }
    return nodes, wiring, producers


def test_validate_references_diag_has_range():
    # A node input binding that references an unknown node (`${ghost.output}`) is a dangling
    # ref. The aggregate LoadError anchors its `range` on the referencing field — the node's
    # `input:` mapping range (`field_ranges["inputs"]`), the coarsest available field span.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  a:\n    kind: agent\n    prompt: hi\n"
        "    input:\n      x: ${ghost.output}\n    output: str\n"
        "output: ${a.output}\n"
    )
    nodes, wiring, producers = _built_flow(text)
    with pytest.raises(LoadError) as exc:
        validate_references(nodes, set(), producers, "${a.output}", wiring)
    assert "ghost" in str(exc.value)
    assert exc.value.range is not None
    assert exc.value.range == nodes["a"].field_ranges["inputs"]


# ---------- 7.8 check_ref_map_types: a call type mismatch carries the binding range ----------


def test_check_ref_map_types_moved():
    # A `call` binding whose SOURCE Type mismatches the child input's declared Type is loud.
    # After the relocation the check lives in `stages.ref_map_types`; the LoadError still
    # carries `.line` (additive) AND anchors its `.range` on the call node's `input:` mapping
    # span (`field_ranges["inputs"]`, the coarsest available binding-site span).
    from agent_composer.compose.stages.ref_map_types import check_ref_map_types
    from agent_composer.compose.types import InputDecl
    from agent_composer.nodes.call import CallNode
    from agent_composer.typesys.values import Type, ValueKind

    node = CallNode(
        "c",
        flow_id="child",
        child_inputs=[InputDecl("n", "int", None, True, Type.scalar(ValueKind.INTEGER))],
    )
    rng = SourceRange(start_line=7, start_col=5, end_line=8, end_col=20)
    node.field_ranges = {"inputs": rng}
    producers = {"src": Type.scalar(ValueKind.STRING)}  # source is a string
    wiring = {"c": {"n": "${src.output}"}}  # bound to the int-expecting child input
    with pytest.raises(LoadError) as exc:
        check_ref_map_types({"c": node}, producers, {}, wiring, {"c": 7})
    assert "child expects" in str(exc.value)
    assert exc.value.line == 7  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng


# ---------- 7.9 check_loop_type_contract: a carried-field type mismatch carries the range ----------


def test_check_loop_type_contract_moved():
    # A `loop` whose carried field is SEEDED from a source Type that mismatches the body
    # output's same-named field Type is loud. After the relocation the check lives in
    # `stages.loop_contract`; the LoadError still carries `.line` (additive) AND anchors its
    # `.range` on the loop node's `input:` mapping span (`field_ranges["inputs"]`).
    from agent_composer.compose.stages.loop_contract import check_loop_type_contract
    from agent_composer.nodes.loop import LoopNode
    from agent_composer.typesys.values import Type, ValueKind

    node = LoopNode("lp", flow_id="body")
    # body returns a record {n: int}; the carried field `n` must be seeded from an int.
    node.output_type = Type(
        kind=ValueKind.OBJECT, fields={"n": Type.scalar(ValueKind.INTEGER)}
    )
    rng = SourceRange(start_line=4, start_col=5, end_line=5, end_col=20)
    node.field_ranges = {"inputs": rng}
    producers = {"src": Type.scalar(ValueKind.STRING)}  # seed source is a string
    wiring = {"lp": {"n": "${src.output}"}}  # carried `n` seeded from the string
    with pytest.raises(LoadError) as exc:
        check_loop_type_contract({"lp": node}, producers, {}, wiring, {"lp": 4})
    assert "carried field" in str(exc.value)
    assert exc.value.line == 4  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng


# ---------- 7.10 check_wiring_parity: a key-parity mismatch carries the node range ----------


def test_check_wiring_parity_moved():
    # A node whose `flow.wiring` keys don't match its declared params is loud. After the
    # relocation the check lives in `stages.wiring_parity`; the LoadError still carries `.line`
    # (additive) AND anchors its `.range` on the node's `input:` mapping span (else node.range).
    from agent_composer.compose.stages.wiring_parity import check_wiring_parity

    node, wiring = build_leaf_node(CodeDescriptor(id="n", code="m:f", inputs={"x": "${input.x}"}), {})
    rng = SourceRange(start_line=3, start_col=5, end_line=4, end_col=20)
    node.field_ranges = {"inputs": rng}
    with pytest.raises(LoadError) as exc:
        # orphan wiring key with no matching param -> parity failure
        check_wiring_parity({"n": node}, {"n": {**wiring, "orphan": "${input.y}"}}, {"n": 3})
    assert "key parity" in str(exc.value)
    assert exc.value.line == 3  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng

