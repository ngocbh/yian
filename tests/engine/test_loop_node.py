"""LoopNode direct-run unit tests — the self-respawning driver model.

Each iteration is a fresh `LoopNode` driver clone whose `run` owns the whole loop policy:
STOP → `Output(carried, commit_as=origin)`; CONTINUE → `Grow(seed=(carried, k), prune=…)`.
These test `run`/`respawn` and compile-owned reconstruction directly (no engine); e2e behavior is in
`test_loop_run.py`.
"""
import json

import pytest

from yian.compile.expand import build_grow
from yian.compile.manifest import expansion_digest
from yian.diagnostics import OriginRef
from yian.memory import MemoryScope
from yian.nodes.base import Grow, NodeKind, Output
from yian.nodes.contracts import (
    Assertion,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from yian.nodes.ids import map_callsite, ns
from yian.nodes.loop import LoopNode
from yian.typesys import Type, ValueKind

from tests.engine.test_expand import _child_flow


def test_node_origin_id_defaults_none():
    from tests.engine._fakes import FuncNode
    assert FuncNode("x", lambda p: {"output": 1}).origin_id is None


def test_loop_is_a_spawner_kind():
    assert LoopNode.is_spawner is True
    assert NodeKind.LOOP.value == "loop"


def test_compiled_loop_origin_is_self():
    node = LoopNode("lp", flow_id="f", child=object(), predicate_kind="while",
                    predicate="${x}", max_iters=10)
    assert node.origin_id == "lp" and node.iteration == 0


def test_run_grows_iteration_zero_with_next_driver_when_continue():
    child = _child_flow()
    node = LoopNode("chat_loop", flow_id="ct", child=child, predicate_kind="while",
                    predicate="not ${exited}", max_iters=1000)
    node.params = child.nodes[child.start_id].params
    seed = {"messages": [], "exited": False}
    result = node.run(dict(seed))
    assert isinstance(result, Grow)
    assert result.seed == (seed, 0)
    assert result.prune == frozenset()                # k==0: no previous body, origin never self-prunes
    assert not hasattr(result, "subgraph")


def test_run_commits_seed_when_while_predicate_false():
    node = LoopNode("chat_loop", flow_id="ct", child=object(), predicate_kind="while",
                    predicate="${exited}", max_iters=1000)
    seed = {"messages": [], "exited": False}
    result = node.run(dict(seed))
    assert isinstance(result, Output)
    assert result.value == seed and result.commit_as == "chat_loop"


def test_clone_driver_self_prunes_previous_body_and_self():
    child = _child_flow()
    node = LoopNode("chat_loop", flow_id="ct", child=child, predicate_kind="while",
                    predicate="not ${exited}", max_iters=1000)
    node.params = child.nodes[child.start_id].params
    clone = node.respawn(1)                            # L~1, iteration 1, origin chat_loop
    clone.params = node.params
    assert clone.id == "chat_loop~1" and clone.origin_id == "chat_loop" and clone.iteration == 1
    result = clone.run({"messages": ["hi"], "exited": False})
    assert isinstance(result, Grow)
    # prune = previous body (chat_loop#0/*) + self (chat_loop~1)
    assert "chat_loop~1" in result.prune
    assert any(p.startswith("chat_loop#0/") for p in result.prune)
    assert result.seed == ({"messages": ["hi"], "exited": False}, 1)
    assert not hasattr(result, "subgraph")


def test_respawn_preserves_assertion_carriers():
    """A fresh LOOP driver retains semantic assertion origins."""
    node = LoopNode(
        "loop",
        flow_id="body",
        child=object(),
        predicate_kind="while",
        predicate="${x}",
        max_iters=3,
    )
    origin = OriginRef("nodes/loop/asserts/0")
    node.pre_asserts = [Assertion("${x} > 0", "x must be positive", origin)]

    clone = node.respawn(1)

    assert clone.pre_asserts == node.pre_asserts
    assert clone.pre_asserts is not node.pre_asserts
    assert clone.pre_asserts[0].origin is origin


def test_respawn_preserves_the_full_output_contract_carrier():
    """A fresh LOOP driver keeps the immutable type and nested output origins."""
    node = LoopNode(
        "loop",
        flow_id="body",
        child=object(),
        predicate_kind="while",
        predicate="${x}",
        max_iters=3,
    )
    origins = ContractOrigins(
        default=OriginRef("nodes/loop/output"),
        entries=((("answer",), OriginRef("nodes/loop/output/answer")),),
    )
    contract = ValueContract(
        Type(
            kind=ValueKind.OBJECT,
            fields={"answer": Type.scalar(ValueKind.INTEGER)},
            required=frozenset({"answer"}),
        ),
        origins,
    )
    node.output_contract = contract

    clone = node.respawn(1)

    assert clone.output_contract is contract
    assert clone.output_contract == contract
    assert clone.output_contract.origins is origins
    assert clone.output_contract.type is contract.type


def test_respawn_preserves_the_derived_base_memory_scope() -> None:
    node = LoopNode(
        "loop",
        flow_id="body",
        child=object(),
        predicate_kind="while",
        predicate="${x}",
        max_iters=3,
    )
    node.memory_scope = MemoryScope(("loop",), memories=("outer",))

    clone = node.respawn(1)

    assert clone.memory_scope is node.memory_scope


def _partial_loop() -> LoopNode:
    node = LoopNode("lp", flow_id="f", child=_child_flow(), predicate_kind="times",
                    times=1, max_iters=1, returned_fields=("n",))
    node.params = [Parameter("n"), Parameter("person")]
    return node


def test_respawn_defaults_omitted_fields_from_the_carried_record():
    node = _partial_loop()
    origin_params = node.params
    origin_snapshot = list(node.params)

    clone = node.respawn(1, {"n": 1, "person": {"name": "A"}})

    by_name = {param.name: param for param in clone.params}
    assert by_name["person"].has_default is True
    assert by_name["person"].default == {"name": "A"}
    assert by_name["person"].required is False
    assert by_name["n"].has_default is False
    assert clone.returned_fields == ("n",)
    assert node.params is origin_params
    assert node.params == origin_snapshot
    assert all(param.has_default is False for param in node.params)


def test_respawned_driver_binds_merged_record():
    clone = _partial_loop().respawn(1, {"n": 1, "person": {"name": "A"}})

    merged = clone.run({"n": 2})
    assert isinstance(merged, Output)
    assert merged.value == {"n": 2, "person": {"name": "A"}}

    explicit_null = clone.run({"n": 2, "person": None})
    assert isinstance(explicit_null, Output)
    assert explicit_null.value == {"n": 2, "person": None}


def test_partial_respawn_requires_the_carried_record():
    with pytest.raises(ValueError):
        _partial_loop().respawn(1)


def test_total_respawn_keeps_params():
    node = LoopNode("lp", flow_id="f", child=_child_flow(), predicate_kind="times",
                    times=1, max_iters=1)
    node.params = [Parameter("n"), Parameter("person")]

    clone = node.respawn(1, {"n": 1, "person": {"name": "A"}})

    assert clone.params is node.params
    assert clone.returned_fields is None


def test_until_do_while_continues_on_iteration_zero():
    child = _child_flow()
    node = LoopNode("lp", flow_id="f", child=child, predicate_kind="until",
                    predicate="${x} <= 0", max_iters=10)
    node.params = child.nodes[child.start_id].params
    result = node.run({"x": 5})                        # until stops on TRUE post-check; k==0 continues
    assert isinstance(result, Grow)


def test_times_stops_at_count():
    child = _child_flow()
    node = LoopNode("lp", flow_id="f", child=child, predicate_kind="times",
                    times=3, max_iters=3)
    node.params = child.nodes[child.start_id].params
    node3 = node.respawn(3)                            # iteration 3 == times -> stop
    node3.params = node.params
    result = node3.run({"x": 1})
    assert isinstance(result, Output) and result.commit_as == "lp"


def test_build_grow_rebuilds_body_and_next_driver_on_origin():
    child = _child_flow()
    node = LoopNode("lp", flow_id="f", child=child, predicate_kind="while",
                    predicate="${x}", max_iters=10)
    node.params = child.nodes[child.start_id].params
    sg = build_grow(node, [{"x": 1}, 2])              # JSON-round-tripped seed: body_2 + lp~3
    assert ns(map_callsite("lp", 2), child.start_id) in sg.nodes
    assert "lp~3" in sg.nodes


def test_build_grow_matches_live_clone_and_origin_replay_byte_for_byte():
    """A live clone and compiled replay origin reconstruct the same origin-keyed window."""
    child = _child_flow()
    origin = LoopNode("lp", flow_id="f", child=child, predicate_kind="while",
                      predicate="${x}", max_iters=10)
    origin.params = child.nodes[child.start_id].params
    clone = origin.respawn(2)

    live_outcome = clone.run({"x": 1})
    assert isinstance(live_outcome, Grow)
    live = build_grow(clone, live_outcome.seed)
    replay = build_grow(origin, [{"x": 1}, 2])       # canonical log JSON turns the pair into a list

    assert live.start_id == replay.start_id
    assert live.end_id == replay.end_id
    assert set(live.nodes) == set(replay.nodes)
    assert live.edges == replay.edges
    assert live.wiring == replay.wiring
    assert live.nodes["lp~3"].origin_id == replay.nodes["lp~3"].origin_id == "lp"
    assert live.nodes["lp~3"].iteration == replay.nodes["lp~3"].iteration == 3


def test_later_loop_growth_does_not_duplicate_the_loop_origin():
    child = _child_flow()
    outer = OriginRef("root/nodes/call")
    loop_origin = OriginRef("child/nodes/loop")
    origin = LoopNode(
        "lp",
        flow_id="f",
        child=child,
        predicate_kind="while",
        predicate="${x}",
        max_iters=10,
    )
    origin.params = child.nodes[child.start_id].params
    origin.origin_context = (outer,)
    origin.origin = loop_origin

    first = build_grow(origin, [{"x": 1}, 0])
    driver = first.nodes["lp~1"]
    second = build_grow(driver, [{"x": 1}, 1])

    assert driver.failure_trace() == (outer, loop_origin)
    assert second.nodes["lp~2"].failure_trace() == (outer, loop_origin)
    body_prefix = ns(map_callsite("lp", 1), "")
    assert {
        node.origin_context
        for node_id, node in second.nodes.items()
        if node_id.startswith(body_prefix)
    } == {(outer, loop_origin)}


def test_runaway_guard_raises_loop_max_exceeded():
    from yian.nodes.base import Grow
    from yian.nodes.loop.node import LoopMaxExceeded
    child = _child_flow()
    origin = LoopNode("lp", flow_id="f", child=child, predicate_kind="while",
                      predicate="${x}", max_iters=1)     # max: 1 permits exactly ONE body run
    origin.params = child.nodes[child.start_id].params
    # driver@0 continues and grows body_0 (within budget: should_stop(0) is False) — NO raise.
    assert isinstance(origin.run({"x": 1}), Grow)
    # driver@1 would grow the 2nd body -> over budget (should_stop(1) True) -> guard trips.
    with pytest.raises(LoopMaxExceeded):
        origin.respawn(1).run({"x": 1})


def test_should_stop_is_the_max_iters_budget():
    node = LoopNode("l", flow_id="f", child=object(), predicate_kind="while",
                    predicate="${x}", max_iters=3)
    assert node.should_stop(0) is False
    assert node.should_stop(2) is False
    assert node.should_stop(3) is True
    assert node.should_stop(4) is True


def test_should_stop_without_max_never_stops():
    node = LoopNode("lp", flow_id="f", child=object(), predicate_kind="while",
                    predicate="${x}", max_iters=None)
    assert node.should_stop(0) is False
    assert node.should_stop(10**9) is False


@pytest.mark.parametrize(("kind", "predicate"), [
    ("while", "not ${exited}"),
    ("until", "${exited}"),
])
def test_unlimited_driver_far_past_any_cap_still_grows(kind, predicate):
    child = _child_flow()
    origin = LoopNode("chat_loop", flow_id="ct", child=child, predicate_kind=kind,
                      predicate=predicate, max_iters=None)
    origin.params = child.nodes[child.start_id].params
    driver = origin.respawn(100_000)
    assert driver.max_iters is None
    result = driver.run({"messages": [], "exited": False})
    assert isinstance(result, Grow) and result.seed[1] == 100_000


def test_build_grow_partial_loop_matches_live_and_replay_digest():
    """Live and replayed partial-loop growth build one fragment whose next driver keeps omitted fields."""
    child = _child_flow()
    origin = LoopNode("lp", flow_id="f", child=child, predicate_kind="while",
                      predicate="${x}", max_iters=10, returned_fields=("x",))
    origin.params = [Parameter("x"), Parameter("exited")]
    carried = {"x": 1, "exited": False}
    clone = origin.respawn(2, carried)

    live_outcome = clone.run(dict(carried))
    assert isinstance(live_outcome, Grow)
    live = build_grow(clone, live_outcome.seed)
    replay = build_grow(origin, json.loads(json.dumps([carried, 2])))

    assert expansion_digest(live) == expansion_digest(replay)
    exited = {param.name: param for param in live.nodes["lp~3"].params}["exited"]
    assert exited.has_default is True
    assert exited.default == carried["exited"]
