"""Walkers over a flow's nested CALL, MAP, and LOOP bodies don't recurse once per level.

Each test loads a chain far past the call nesting limit, then walks it with only a few
frames of headroom, which a walker that recursed per nested flow would overflow.
"""

import json
import re

from pydantic import SecretStr
import pytest

from tests.compose.test_call_nesting_limit import _defs_chain, _mixed_chain
from tests.deep_chains import deep_load, lifted_limit, tight_stack
from yian.compile.execution_layout import resolve_execution_layout
from yian.compile.llm_cascade import (
    _static_agent_paths,
    apply_effective_llm_map,
    bind_api_key,
    resolve_llm_cascade,
    static_agent_labels,
)
from yian.compile.manifest import expansion_digest, flow_fingerprint
from yian.compile.memory_layout import resolve_memory_layout
from yian.compile.model import CompiledFlow, Flow, copy_flow_graph
from yian.compile.origins import iter_flow_origins, rebase_flow_origins
from yian.compose import ComposedFlow, load_flow
from yian.diagnostics import OriginRef
from yian.events import NodeStarted
from yian.expr import ExpressionError
from yian.llm_clients.connection import HOST_KEY_MARKER, missing_model_message
from yian.memory import MemorySetupError
from yian.memory.contracts import _iter_memory_decl_origins
from yian.nodes.base import Node
from yian.nodes.call import CallNode
from yian.nodes.contracts import Binding, _iter_origin_carrier
from yian.run import _check_templates, _static_nodes, resume_flow, run_flow

_DEPTH = 300
_HEADROOM = 150
# Resolving and applying LLM configs copies each callsite's child, so their deep chain
# is shorter to keep the copies' quadratic cost down.
_CASCADE_DEPTH = 120
_CASCADE_HEADROOM = 80
# Running or resuming a loaded chain takes about 40 frames whatever its depth, so this
# headroom catches a run that recurses even once per nested flow over 60 levels.
_RUN_DEPTH = 60
_RUN_HEADROOM = 70


@pytest.fixture(scope="module")
def mixed() -> ComposedFlow:
    """A chain `_DEPTH` levels deep whose levels nest by CALL, MAP, and LOOP in turn."""
    return deep_load(_mixed_chain(_DEPTH))


def _static_nodes_reference(flow, path=()):
    """Return the static nodes `_static_nodes` returns, by recursing per child flow."""
    found = []
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        found.append(("/".join(node_path), node))
        for child in node.origin_children():
            found.extend(_static_nodes_reference(child, node_path))
    return found


def test_static_nodes_of_a_deep_chain_come_in_pre_order(mixed):
    expected = _static_nodes_reference(mixed.compiled)
    with tight_stack(_HEADROOM):
        found = _static_nodes(mixed.compiled)
    assert [(path, id(node)) for path, node in found] == [
        (path, id(node)) for path, node in expected
    ]
    assert len(found) == 3 * _DEPTH + 3


def test_templates_of_a_deep_chain_check_without_recursing(mixed):
    with tight_stack(_HEADROOM):
        _check_templates(mixed.compiled)


def test_first_bad_span_in_check_order_is_reported():
    """A node's child flows are checked before the next node of its own flow."""
    flow = load_flow(_defs_chain(5))
    first = flow.compiled.nodes["top"].child
    third = first.nodes["step"].child.nodes["step"].child
    third.wiring["step"]["x"] = Binding("${upper(input.x, 1)}")
    first.wiring["__end__"]["result"] = Binding("${upper(step.output, 1)}")
    with pytest.raises(
        ExpressionError,
        match=r"^node 'top/step/step/step' input 'x': expression function 'upper' takes "
        r"\(s\): too many positional arguments$",
    ):
        _check_templates(flow.compiled)


def test_fingerprint_and_expansion_digest_of_a_deep_chain(mixed):
    with tight_stack(_HEADROOM):
        fingerprint = flow_fingerprint(mixed.compiled)
        digest = expansion_digest(mixed.compiled)
    assert len(fingerprint) == len(digest) == 64


def test_fingerprint_of_a_nested_chain_is_unchanged():
    """The digests main computed for this chain, so walking it iteratively keeps identity."""
    flow = load_flow(_mixed_chain(30))
    assert flow_fingerprint(flow.compiled) == (
        "122a83b235f2eec4d186aabba535a83d648b2c55599f054136c34982bebc9e13"
    )
    assert expansion_digest(flow.compiled) == (
        "a956c8a55e2ae28b69abc49cc85b5d2f3c5cac7e1ce9a16fbcd56d7e4bc08b6c"
    )


def test_a_child_flow_cycle_deep_in_a_chain_is_not_manifestable():
    flow = load_flow(_defs_chain(4))
    first = flow.compiled.nodes["top"].child
    third = first.nodes["step"].child.nodes["step"].child
    third.nodes["step"].child = first
    cycle = r"nodes\.top\.child\.nodes\.step\.child\.nodes\.step\.child\.nodes\.step\.child"
    with pytest.raises(
        ValueError, match=rf"^flow\.{cycle}: recursive child-flow cycle is not manifestable$"
    ):
        flow_fingerprint(flow.compiled)
    with pytest.raises(
        ValueError, match=rf"^expansion\.{cycle}: recursive child-flow cycle is not manifestable$"
    ):
        expansion_digest(flow.compiled)


def test_a_bad_field_in_a_grandchild_names_its_manifest_path():
    flow = load_flow(_defs_chain(3))
    grandchild = flow.compiled.nodes["top"].child.nodes["step"].child
    object.__setattr__(grandchild.nodes["step"], "extra", 1)
    node = r"nodes\.top\.child\.nodes\.step\.child\.nodes\.step"
    unknown = r"unknown noncanonical execution field\(s\) for call: \['extra'\]$"
    with pytest.raises(ValueError, match=rf"^flow\.{node}: {unknown}"):
        flow_fingerprint(flow.compiled)
    with pytest.raises(ValueError, match=rf"^expansion\.{node}: {unknown}"):
        expansion_digest(flow.compiled)


def _graph_objects(flow):
    """Return the ids of `flow`, its nodes, and every flow and node nested under them."""
    found = set()
    pending = [flow]
    while pending:
        current = pending.pop()
        if id(current) in found:
            continue
        found.add(id(current))
        for node in current.nodes.values():
            found.add(id(node))
            pending.extend(node.origin_children())
    return found


def _leaf_flow(node_id="only"):
    """Return a one-node flow."""
    return Flow(
        nodes={node_id: Node(node_id)},
        edges=[],
        wiring={node_id: {}},
        start_id=node_id,
        end_id=node_id,
    )


def _calling_flow(**calls):
    """Return a flow of CALL nodes, each named by its keyword and holding the given child."""
    first = next(iter(calls))
    return Flow(
        nodes={
            node_id: CallNode(node_id, flow_id=node_id, child=child)
            for node_id, child in calls.items()
        },
        edges=[],
        wiring={node_id: {} for node_id in calls},
        start_id=first,
        end_id=first,
    )


def test_a_deep_chain_copies_without_recursing_per_level(mixed):
    with tight_stack(_HEADROOM):
        copied = copy_flow_graph(mixed.compiled)
    assert flow_fingerprint(copied) == flow_fingerprint(mixed.compiled)
    original = _graph_objects(mixed.compiled)
    assert len(_graph_objects(copied)) == len(original)
    assert not original & _graph_objects(copied)


def test_a_child_flow_shared_by_two_nodes_stays_shared_in_the_copy():
    shared = _leaf_flow()
    flow = _calling_flow(a=shared, b=shared)
    copied = copy_flow_graph(flow)
    assert copied.nodes["a"].child is copied.nodes["b"].child
    assert copied.nodes["a"].child is not shared


def test_a_child_flow_cycle_is_kept_in_the_copy():
    flow = _calling_flow(loop=None)
    flow.nodes["loop"].child = flow
    copied = copy_flow_graph(flow)
    assert copied is not flow
    assert copied.nodes["loop"].child is copied


def test_copying_a_node_copies_its_child_flows():
    node = CallNode("call", flow_id="child", child=_calling_flow(inner=_leaf_flow()))
    copied = copy_flow_graph(node)
    assert type(copied) is CallNode and copied is not node
    assert copied.child is not node.child
    assert copied.child.nodes["inner"].child is not node.child.nodes["inner"].child
    assert copied.child.nodes["inner"].child.nodes["only"].id == "only"


def test_objects_outside_the_child_flows_are_deep_copied():
    leaf = _leaf_flow()
    payload = {"items": [1, 2]}
    leaf.nodes["only"].extra = payload
    flow = _calling_flow(call=leaf)
    flow.extra = payload
    copied = copy_flow_graph(flow)
    copied_payload = copied.nodes["call"].child.nodes["only"].extra
    assert copied_payload == payload and copied_payload is not payload
    assert copied.extra is copied_payload


_CODE_LEAF = (
    "        kind: code\n"
    "        code: tests.engine._compose_codefns:echo\n"
    "        input:\n"
    "          topic: ${input.v}\n"
    "        output: str\n"
)


def _agent_chain(depth, *, leaf_model=True):
    """Return `_mixed_chain(depth)` with an AGENT leaf and a root-level provider.

    The leaf sets its own model unless `leaf_model` is false.
    """
    leaf = "        kind: agent\n        input:\n          topic: ${input.v}\n        prompt: ${topic}\n"
    if leaf_model:
        leaf += "        llm_config: {model: leaf-model}\n"
    text = _mixed_chain(depth).replace(_CODE_LEAF, leaf)
    return text.replace("\ndefs:\n", "\nllm_config: {provider: anthropic}\ndefs:\n", 1)


def _leaf_path(depth):
    """Return the static path of a mixed chain's leaf node."""
    return "/".join(["top", *["step"] * depth])


def _leaf_node(flow):
    """Return the node at the end of a chain's CALL, MAP, and LOOP children."""
    node = flow.nodes["top"]
    while getattr(node, "child", None) is not None:
        node = node.child.nodes["step"]
    return node


@pytest.fixture(scope="module")
def agents() -> ComposedFlow:
    """A `_DEPTH`-level mixed chain whose leaf is a labelled AGENT marked for a host key."""
    flow = deep_load(_agent_chain(_DEPTH))
    leaf = _leaf_node(flow.compiled)
    leaf.llm_config_label = "the leaf's llm_config"
    leaf.llm_config = {"provider": "anthropic", "model": "leaf-model", HOST_KEY_MARKER: True}
    return flow


def test_agent_paths_and_labels_of_a_deep_chain(agents):
    path = _leaf_path(_DEPTH)
    parent = path.rsplit("/", 1)[0]
    with tight_stack(_HEADROOM):
        paths = _static_agent_paths(agents.compiled)
        labels = static_agent_labels(agents.compiled)
    assert paths == [path]
    assert labels == {path: f"the leaf's llm_config in {parent!r}"}


def test_a_host_key_binds_to_a_deep_agent(agents):
    key = SecretStr("host-key")
    with tight_stack(_HEADROOM):
        bind_api_key(agents.compiled, key)
    assert _leaf_node(agents.compiled).llm_config["api_key"] is key


def test_llm_configs_of_a_deep_chain_resolve_and_apply_without_recursing():
    flow = deep_load(_agent_chain(_CASCADE_DEPTH))
    with tight_stack(_CASCADE_HEADROOM):
        effective = resolve_llm_cascade(flow.compiled, {}, defaults={})
    expected = {"provider": "anthropic", "model": "leaf-model"}
    assert effective == {_leaf_path(_CASCADE_DEPTH): expected}
    assert _leaf_node(flow.compiled).llm_config == expected

    replayed = deep_load(_agent_chain(_CASCADE_DEPTH))
    with tight_stack(_CASCADE_HEADROOM):
        apply_effective_llm_map(replayed.compiled, effective)
    leaf = _leaf_node(replayed.compiled)
    assert leaf.llm_config == expected
    assert leaf.llm_config is not effective[_leaf_path(_CASCADE_DEPTH)]


def test_a_deep_agent_without_a_model_is_named():
    flow = deep_load(_agent_chain(_CASCADE_DEPTH, leaf_model=False))
    message = missing_model_message(_leaf_path(_CASCADE_DEPTH), "anthropic")
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        with tight_stack(_CASCADE_HEADROOM):
            resolve_llm_cascade(flow.compiled, {}, defaults={})


def _origins_reference(flow, seen_flows, seen_nodes, seen_carriers):
    """Yield the origins `iter_flow_origins` yields, by recursing per child flow."""
    if id(flow) in seen_flows:
        return
    seen_flows.add(id(flow))
    if isinstance(flow, CompiledFlow):
        yield from _iter_memory_decl_origins(flow.memories, seen_carriers)
    for node_wiring in flow.wiring.values():
        for binding in node_wiring.values():
            yield from _iter_origin_carrier(binding, seen_carriers)
    if isinstance(flow, CompiledFlow):
        for output in flow.outputs:
            yield from _iter_origin_carrier(output.binding, seen_carriers)
    for node in flow.nodes.values():
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        yield from node.iter_local_origins(_seen_carriers=seen_carriers)
        for child in node.origin_children():
            yield from _origins_reference(child, seen_flows, seen_nodes, seen_carriers)


def test_origins_of_a_deep_chain_come_in_structural_order(mixed):
    expected = list(_origins_reference(mixed.compiled, set(), set(), set()))
    with tight_stack(_HEADROOM):
        found = list(iter_flow_origins(mixed.compiled))
    assert [id(origin) for origin in found] == [id(origin) for origin in expected]
    assert len(found) > 3 * _DEPTH


class _Resolver:
    """An origin resolver that resolves nothing."""

    def resolve(self, origin):
        return None


def test_origins_of_a_deep_chain_rebase_without_recursing(mixed):
    original = list(iter_flow_origins(mixed.compiled))
    with tight_stack(_HEADROOM):
        rebased, _table = rebase_flow_origins(
            mixed.compiled, lambda origin: OriginRef(f"rebased/{origin.key}"), resolver=_Resolver()
        )
        found = list(iter_flow_origins(rebased))
    assert [origin.key for origin in found] == [f"rebased/{origin.key}" for origin in original]
    assert not _graph_objects(mixed.compiled) & _graph_objects(rebased)


def _layout_reference(flow, path=()):
    """Return the (locator, node id) pairs a layout targets, by recursing per child flow.

    Layouts visit a flow's nodes in sorted id order, each before its child flow.
    """
    found = []
    for node_id in sorted(flow.nodes):
        node = flow.nodes[node_id]
        found.append(((*path, node_id), id(node)))
        for child in node.origin_children():
            found.extend(_layout_reference(child, (*path, node_id)))
    return found


def test_memory_layout_of_a_deep_chain(mixed):
    with tight_stack(_HEADROOM):
        layout = resolve_memory_layout(mixed.compiled)
    assert [(target.locator, id(target.node)) for target in layout.targets] == (
        _layout_reference(mixed.compiled)
    )


def test_execution_layout_of_a_deep_chain(mixed):
    with tight_stack(_HEADROOM):
        layout = resolve_execution_layout(mixed.compiled)
    assert [(target.locator, id(target.node)) for target in layout.targets] == (
        _layout_reference(mixed.compiled)
    )
    assert all(target.scope.flow_path == target.locator[:-1] for target in layout.targets)


def test_a_child_flow_cycle_deep_in_a_chain_has_no_layout():
    flow = load_flow(_defs_chain(4))
    first = flow.compiled.nodes["top"].child
    third = first.nodes["step"].child.nodes["step"].child
    third.nodes["step"].child = first
    with pytest.raises(
        MemorySetupError, match="^recursive child-flow cycle is not memory-layout resolvable$"
    ):
        resolve_memory_layout(flow.compiled)
    with pytest.raises(
        ValueError, match="^recursive child-flow cycle is not execution-layout resolvable$"
    ):
        resolve_execution_layout(flow.compiled)


@pytest.mark.parametrize("depth", [3, _RUN_DEPTH])
def test_a_deep_chain_runs_and_resumes_without_recursing(tmp_path, depth):
    """The 3-level case bounds the frames a run takes at any depth.

    A learnable run with a memory mapping also resolves its execution and memory
    layouts.
    """
    flow = deep_load(_mixed_chain(depth))
    path = tmp_path / "run.jsonl"
    with lifted_limit():
        with tight_stack(_RUN_HEADROOM):
            first = run_flow(
                flow, {"v": "go"}, log=path, num_workers=0, learnable=True, memory={}
            )
        assert first.status == "succeeded", first.failure
        # Cut the log after its last expansion, so resume replays every expansion
        # and then runs the leaf.
        lines = path.read_text().splitlines(keepends=True)
        expanded = [
            i for i, line in enumerate(lines) if json.loads(line)["kind"] == "node_expanded"
        ]
        assert len(expanded) == depth
        path.write_text("".join(lines[: expanded[-1] + 1]))
        with tight_stack(_RUN_HEADROOM):
            resumed = resume_flow(flow, log=path, num_workers=0, memory={})
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output
    assert any(isinstance(event, NodeStarted) for event in resumed.events)
