"""Tests for the validate phase: cross-node checks over the assembled flow.

Covers cycle rejection, reference validation, the cross-flow call/loop type checks,
wiring parity, the human_input questions surface, and assert/input-binding location
metadata. Each check rejects an ill-formed flow with a `LoadError` that anchors its
`.range` on the offending field span.
"""

import pytest

from agent_composer.compile.model import Edge
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.loader import load_flow
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.compose.nodes.agent import AgentDescriptor
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.code import CodeDescriptor
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.validate.cycles import reject_cycles
from agent_composer.compose.validate.references import validate_references
from agent_composer.events import SourceRange


def _nodes_reader(text):
    """A span-aware NodeReader over the `nodes:` mapping of a flow document."""
    value, spans, root = load_with_spans(text, return_root=True)
    return NodeReader(value, spans, root).child("nodes")


def _node_wiring(desc):
    """Build a leaf descriptor -> the `(node, wiring)` pair (unwrapping the BuildResult)."""
    result = desc.build(BuildContext({}))
    return result.node, result.wiring


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
        nid: _node_wiring(d)
        for nid, d in descriptors.items()
        if isinstance(d, (AgentDescriptor, CodeDescriptor))
    }
    nodes = {nid: n for nid, (n, _) in built.items()}
    wiring = {nid: w for nid, (_, w) in built.items()}
    producers = {
        nid: n.output_type for nid, n in nodes.items() if n.output_type is not None
    }
    return nodes, wiring, producers


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


def test_validate_references_diag_has_range():
    # A node input binding that references an unknown node (`${ghost.output}`) is a dangling
    # ref. The aggregate LoadError anchors its `range` on the OFFENDING ENTRY — the per-binding
    # `("input", name)` span the parse loop stamps for every kind — not the enclosing `input:`
    # mapping, which would also mark every innocent sibling entry (#87).
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
    assert exc.value.range == nodes["a"].field_ranges[("input", "x")]
    # ...and it is strictly narrower than the block it used to report: `x:` is on line 8, while
    # the `input:` mapping runs to line 9 (`output: str` — a different field entirely).
    assert exc.value.range != nodes["a"].field_ranges[("field", "inputs")]
    assert exc.value.range.start_line == exc.value.range.end_line == 8


def test_validate_references_diag_falls_back_when_no_per_binding_range():
    # The fallback path. A desugared CASE binds SYNTHETIC wiring params (`__on`/`__rN` — the
    # reconciled `when:` data refs), which own no authored `input:` entry, so the per-binding
    # lookup finds nothing. Such an error must still locate — degrading to the node's own range
    # rather than vanishing. Driven through the whole loader because the desugar is what mints
    # `__rN` (the `_built_flow` slice above only builds agent/code leaves).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  a:\n    kind: case\n"
        "    cases:\n      - when: \"ghost.output >= 1\"\n        then: b\n"
        "    else: b\n"
        "  b:\n    kind: code\n    output: int\n    code: \"return {'output': 1}\"\n"
        "output: ${b.output}\n"
    )
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "ghost" in str(exc.value)
    assert "'__r0'" in str(exc.value)          # the synthetic param — no authored entry to stamp
    assert exc.value.range is not None         # still located...
    assert exc.value.range.start_line == 4     # ...on the `a:` node header (line 4)


def test_check_ref_map_types_moved():
    # A `call` binding whose SOURCE Type mismatches the child input's declared Type is loud.
    # The check lives in `validate.call_types`; the LoadError carries `.line` (additive) AND
    # anchors its `.range` on the call node's `input:` mapping span (`field_ranges["inputs"]`,
    # the coarsest available binding-site span).
    from agent_composer.compose.validate.call_types import check_ref_map_types
    from agent_composer.compose.helpers.types import InputDecl
    from agent_composer.nodes.call import CallNode
    from agent_composer.typesys.values import Type, ValueKind

    node = CallNode(
        "c",
        flow_id="child",
        child_inputs=[InputDecl("n", "int", None, True, Type.scalar(ValueKind.INTEGER))],
    )
    rng = SourceRange(start_line=7, start_col=5, end_line=8, end_col=20)
    node.field_ranges = {("field", "inputs"): rng}
    producers = {"src": Type.scalar(ValueKind.STRING)}  # source is a string
    wiring = {"c": {"n": "${src.output}"}}  # bound to the int-expecting child input
    with pytest.raises(LoadError) as exc:
        check_ref_map_types({"c": node}, producers, {}, wiring, {"c": 7})
    assert "child expects" in str(exc.value)
    assert exc.value.line == 7  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng


def test_check_loop_type_contract_moved():
    # A `loop` whose carried field is SEEDED from a source Type that mismatches the body
    # output's same-named field Type is loud. The check lives in `validate.loop_types`; the
    # LoadError carries `.line` (additive) AND anchors its `.range` on the loop node's
    # `input:` mapping span (`field_ranges["inputs"]`).
    from agent_composer.compose.validate.loop_types import check_loop_type_contract
    from agent_composer.nodes.loop import LoopNode
    from agent_composer.typesys.values import Type, ValueKind

    node = LoopNode("lp", flow_id="body")
    # body returns a record {n: int}; the carried field `n` must be seeded from an int.
    node.output_type = Type(
        kind=ValueKind.OBJECT, fields={"n": Type.scalar(ValueKind.INTEGER)}
    )
    rng = SourceRange(start_line=4, start_col=5, end_line=5, end_col=20)
    node.field_ranges = {("field", "inputs"): rng}
    producers = {"src": Type.scalar(ValueKind.STRING)}  # seed source is a string
    wiring = {"lp": {"n": "${src.output}"}}  # carried `n` seeded from the string
    with pytest.raises(LoadError) as exc:
        check_loop_type_contract({"lp": node}, producers, {}, wiring, {"lp": 4})
    assert "carried field" in str(exc.value)
    assert exc.value.line == 4  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng


def test_check_wiring_parity_moved():
    # A node whose `flow.wiring` keys don't match its declared params is loud. The check lives
    # in `validate.wiring_parity`; the LoadError carries `.line` (additive) AND anchors its
    # `.range` on the node's `input:` mapping span (else node.range).
    from agent_composer.compose.validate.wiring_parity import check_wiring_parity

    node, wiring = _node_wiring(CodeDescriptor(id="n", code="m:f", inputs={"x": "${input.x}"}))
    rng = SourceRange(start_line=3, start_col=5, end_line=4, end_col=20)
    node.field_ranges = {("field", "inputs"): rng}
    with pytest.raises(LoadError) as exc:
        # orphan wiring key with no matching param -> parity failure
        check_wiring_parity({"n": node}, {"n": {**wiring, "orphan": "${input.y}"}}, {"n": 3})
    assert "key parity" in str(exc.value)
    assert exc.value.line == 3  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng


def test_validate_human_questions_diag_has_range():
    # A human_input gate with a structurally-invalid literal `questions:` list is loud. The
    # check lives in `validate.human_questions`; the LoadError carries `.line` (additive) AND
    # anchors its `.range` on the gate's `questions:` field span.
    from agent_composer.compose.validate.human_questions import (
        validate_human_questions,
    )
    from agent_composer.nodes.human_input.node import HumanInputNode

    node = HumanInputNode("ask", questions=[])  # empty -> "expected 1..4 questions, got 0"
    rng = SourceRange(start_line=5, start_col=5, end_line=6, end_col=20)
    node.field_ranges = {("field", "questions"): rng}
    with pytest.raises(LoadError) as exc:
        validate_human_questions({"ask": node}, node_lines={"ask": 5})
    assert "questions" in str(exc.value)
    assert exc.value.line == 5  # additive: node_lines still feeds `.line`
    assert exc.value.range == rng


_ASSERT_FLOW = """\
id: f
name: f
nodes:
  score:
    kind: agent
    prompt: "Rate ${topic}"
    input:
      topic: ${input.topic}
    output: int
    asserts:
      - ${output} >= 0
      - ${output} <= 10
output: ${score.output}
"""


def test_node_local_asserts_carry_ranges():
    # Each authored assert maps to its OWN SourceRange in the descriptor's POSITIONAL
    # `assert_ranges` (aligned 1:1 with the authored `asserts:` list), so every assert gets its own
    # caret located by INDEX — never by its (non-unique) text. Two identical asserts would still get
    # distinct ranges. The coarse `asserts:` list span stays in field_ranges under `("field", ...)`.
    descriptors = parse_nodes(_nodes_reader(_ASSERT_FLOW))
    desc = descriptors["score"]
    assert [r.start_line for r in desc.assert_ranges] == [11, 12]
    assert ("field", "asserts") in desc.field_ranges  # coarse list range still present


_INPUT_FLOW = """\
id: f
name: f
nodes:
  report:
    kind: agent
    prompt: "About ${topic} at ${as_of}"
    input:
      topic: ${input.topic}
      as_of: ${input.as_of:?"as_of is required"}
    output: str
output: ${report.output}
"""


def test_per_input_param_carries_ranges():
    # Each input binding maps to its OWN SourceRange under its param NAME in the descriptor's
    # field_ranges — the key eval_node's binding-failure funnel resolves (`loc.key` == param
    # name). Recorded kind-blind in the parse choke point alongside the coarse `inputs:` span
    # (still present under the field name).
    descriptors = parse_nodes(_nodes_reader(_INPUT_FLOW))
    fr = descriptors["report"].field_ranges
    assert fr[("input", "topic")].start_line == 8
    assert fr[("input", "as_of")].start_line == 9
    assert ("field", "inputs") in fr  # coarse mapping range still present under the field name


_INPUT_LIKE_FIELD_FLOW = """\
id: f
name: f
nodes:
  n:
    kind: code
    env:
      retries: 2
    input:
      env: ${input.data}
    output: int
    code: tests.seeds.fns:const_one
input:
  data: str
output: ${n.output}
"""


def test_input_named_like_field_no_collision():
    # Case 2 (#88): a node with an input named `env` AND its own `env:` config field. Under the old
    # flat string key both stamped `"env"` and collided (last-write-wins, one range survived); now
    # they are tagged `("input", "env")` and `("field", "env")` — distinct keys, distinct ranges.
    fr = parse_nodes(_nodes_reader(_INPUT_LIKE_FIELD_FLOW))["n"].field_ranges
    assert ("input", "env") in fr and ("field", "env") in fr
    assert fr[("input", "env")].start_line == 9  # the input binding line
    assert fr[("field", "env")].start_line == 7  # the `env:` config value (retries: 2)
    assert fr[("input", "env")] != fr[("field", "env")]
