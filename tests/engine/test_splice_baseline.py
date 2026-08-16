"""Splice baseline (characterization, GREEN on first run).

Pins the child START_ID/END_ID shape the splice keys off, so the splice can assert its deltas
against recorded literals (not a remembered set). Loads a one-output REF parent + a CODE-only
child and a MAP parent WITHOUT running them, recording the boundary-node shape (the child
START_ID's params == declared input names; the child END_ID's params == declared output names; START_ID
is the single root; END_ID is the terminal) + the parent edge id set + the CALL/MAP wiring.

No OUTPUT_RESOLVER/COLLECTOR kind is reachable from a LOADED (un-run) flow — synthesis removed them;
they only appear at RUNTIME.
"""

from agent_composer.compose import load_flow
from agent_composer.nodes.base import NodeKind
from tests.engine.test_map import _ECHO_CHILD, _map_flow
from tests.engine.test_map import _resolver as _map_resolver
from tests.engine.test_ref_run import _CHILD, _REF_PARENT, _resolver


def test_ref_child_carries_one_start_and_one_end_boundary():
    parent = load_flow(_REF_PARENT, child_resolver=_resolver(**{"child-one": _CHILD}))
    child = parent.compiled.nodes["research"].child

    # (i) exactly one START_ID (single root) + one END_ID (terminal), with the reserved ids.
    assert child.start_id == "__start__" and child.end_id == "__end__"
    assert child.nodes[child.start_id].kind == NodeKind.START
    assert child.nodes[child.end_id].kind == NodeKind.END

    # (ii) the child START's params == the declared input names.
    assert [p.name for p in child.nodes[child.start_id].params] == ["topic", "suffix"]
    # (iii) the child END_ID's params == the declared output names.
    assert [p.name for p in child.nodes[child.end_id].params] == ["report", "n"]


def test_ref_parent_edge_ids_and_call_wiring():
    parent = load_flow(_REF_PARENT, child_resolver=_resolver(**{"child-one": _CHILD}))
    c = parent.compiled
    assert sorted(e.id for e in c.edges) == [
        "__start__->research#0",
        "research->__end__#0",
        "research->__end__#1",
    ]
    wiring = c.wiring["research"]
    assert {name: binding.source for name, binding in wiring.items()} == {
        "topic": "${input.topic}",
    }
    assert wiring["topic"].origin is not None


def test_map_child_boundary_and_over_wiring():
    parent = load_flow(_map_flow("echo-one", parallel=False),
                       child_resolver=_map_resolver(**{"echo-one": _ECHO_CHILD}))
    c = parent.compiled
    child = c.nodes["each"].child
    assert child.start_id == "__start__" and child.end_id == "__end__"
    assert child.nodes[child.start_id].kind == NodeKind.START
    assert child.nodes[child.end_id].kind == NodeKind.END
    assert [p.name for p in child.nodes[child.start_id].params] == ["topic"]
    # the MAP `over` source rides flow.wiring[id]["over"] (over-then-inputs).
    assert {name: binding.source for name, binding in c.wiring["each"].items()} == {
        "over": "${input.topics}",
        "topic": "${item}",
    }
    assert c.wiring["each"]["over"].origin is not None
    assert c.wiring["each"]["topic"].origin is not None
    assert sorted(e.id for e in c.edges) == ["__start__->each#0", "each->__end__#0"]


def test_no_output_resolver_or_collector_in_a_loaded_flow():
    # synthesis removed OUTPUT_RESOLVER/COLLECTOR; a LOADED (un-run) flow carries none.
    # (the kinds were deleted entirely; assert by VALUE so this holds across the deletion.)
    for parent in (
        load_flow(_REF_PARENT, child_resolver=_resolver(**{"child-one": _CHILD})),
        load_flow(_map_flow("echo-one", parallel=False),
                  child_resolver=_map_resolver(**{"echo-one": _ECHO_CHILD})),
    ):
        kinds = {n.kind for n in parent.compiled.nodes.values()}
        child = next(n.child for n in parent.compiled.nodes.values()
                     if n.kind in (NodeKind.CALL, NodeKind.MAP))
        kinds |= {n.kind for n in child.nodes.values()}
        values = {k.value for k in kinds}
        assert "output_resolver" not in values
        assert "collector" not in values
