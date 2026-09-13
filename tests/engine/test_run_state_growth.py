"""Transactional flat graph-expansion tests for the run-state reducer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

import agent_composer.runtime.state as state_module
from agent_composer.compile.expand import build_grow
from agent_composer.compile.manifest import expansion_digest
from agent_composer.compile.model import (
    CompiledFlow,
    Edge,
    END_ID,
    Flow,
    GrowthRejected,
    START_ID,
)
from agent_composer.runtime.common import NodeState
from agent_composer.diagnostics import OriginRef
from agent_composer.events import NodeExpanded
from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.nodes.contracts import (
    Binding,
    ContractOrigins,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.map import MapNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.log import MemoryRunLog, NodeExpandedRecord
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_expanded,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
    prepare_run_started,
)
from agent_composer.typesys import Type, ValueKind, VariablePool


_FINGERPRINT = "d" * 64
_RUN_ID = "growth-run"


class _Spawner(Node):
    """Minimal kind-blind spawner used to isolate reducer behavior."""

    kind = NodeKind.CALL
    is_spawner = True
    grow_depth_delta = 1


class _ParameterSubclass(Parameter):
    """Near-canonical fixture proving growth requires the exact carrier."""


class _OverlayNode(Node):
    """Ordinary dynamic-overlay node used by prune safety tests."""

    kind = NodeKind.CODE


def _parent_flow(
    spawner: Node | None = None,
    *,
    sibling: bool = False,
) -> CompiledFlow:
    """Return a complete static graph whose one spawner can be put in flight."""
    spawner = _Spawner("spawn") if spawner is None else spawner
    nodes = {
        START_ID: StartNode(START_ID, params=[]),
        spawner.id: spawner,
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    if sibling:
        nodes["side"] = Node("side")
        nodes["side"].kind = NodeKind.CODE
    edges = [
        Edge(id="start-spawn", from_=START_ID, to=spawner.id),
        Edge(id="spawn-end", from_=spawner.id, to=END_ID),
    ]
    if sibling:
        edges.append(Edge(id="start-side", from_=START_ID, to="side"))
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=edges,
        wiring={node_id: {} for node_id in nodes},
    )


def _initialized_spawner(
    flow: CompiledFlow | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Return a replay-valid state/log with the static spawner in flight."""
    flow = _parent_flow() if flow is None else flow
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    state = RunState(flow)
    log = MemoryRunLog(_RUN_ID)
    commit_prepared(
        log,
        state,
        prepare_run_started(
            state,
            RunStartData(
                run_id=_RUN_ID,
                flow_fingerprint=_FINGERPRINT,
                initial_pool=pool,
                start_wiring_override={},
                effective_llm_by_static_path={},
                num_workers=0,
            ),
        ),
    )
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(state, START_ID, {}))
    commit_prepared(log, state, prepare_node_started(state, "spawn"))
    return state, log


def _canonical_child(*, optional_input: bool = False) -> CompiledFlow:
    """Return a real manifestable child accepted by ``build_grow``."""
    parameters = (
        [
            Parameter(
                name="optional",
                contract=ValueContract(Type.scalar(ValueKind.STRING)),
                has_default=True,
                default="fallback",
                coerce_strings=True,
            )
        ]
        if optional_input
        else []
    )
    start = StartNode(START_ID, params=parameters)
    end = EndNode.record(END_ID, output_names=[])
    return CompiledFlow.from_parts(
        nodes={START_ID: start, END_ID: end},
        edges=[
            Edge(
                id="child-boundary",
                from_=START_ID,
                to=END_ID,
                ordering=True,
            )
        ],
        wiring={
            START_ID: (
                {"optional": "${input.optional}"} if optional_input else {}
            ),
            END_ID: {},
        },
    )


def _fragment(spawner_id: str, *, suffix: str = "child") -> Flow:
    """Return a manifestable two-boundary fragment committing under ``spawner_id``."""
    start_id = f"{spawner_id}/{suffix}/start"
    end_id = f"{spawner_id}/{suffix}/end"
    start = StartNode(start_id, params=[])
    end = EndNode.record(end_id, output_names=[])
    end.commit_as = spawner_id
    return Flow(
        nodes={start_id: start, end_id: end},
        edges=[
            Edge(
                id=f"{start_id}->{end_id}",
                from_=start_id,
                to=end_id,
                ordering=True,
            )
        ],
        wiring={start_id: {}, end_id: {}},
        start_id=start_id,
        end_id=end_id,
    )


def _snapshot(state: RunState) -> dict[str, Any]:
    """Return every reducer and topology coordinate in detached comparable form."""
    return {
        "nodes": {
            node_id: (
                type(node),
                node.id,
                node.commit_as,
                node.origin_id,
            )
            for node_id, node in state.flow.nodes.items()
        },
        "edges": list(state.flow.edges),
        "wiring": deepcopy(state.flow.wiring),
        "incoming": {key: list(value) for key, value in state.flow._incoming.items()},
        "outgoing": {key: list(value) for key, value in state.flow._outgoing.items()},
        "pool": state.pool.model_dump(mode="json"),
        "node_state": dict(state.node_state),
        "edge_state": dict(state.edge_state),
        "ready": tuple(state.ready),
        "in_flight": tuple(state.in_flight),
        "result": state.result,
        "depth": dict(state.depth),
        "wiring_overrides": deepcopy(state.wiring_overrides),
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "next_sequence": state.next_sequence,
        "num_workers": state.num_workers,
        "input": deepcopy(state.input),
        "effective": deepcopy(state.effective_llm_by_static_path),
        "initial_static_ids": state.initial_static_ids,
    }


def _install_settled_overlay(
    state: RunState,
    *,
    prefix: str = "old",
    residue: bool = False,
) -> tuple[str, ...]:
    """Install settled dynamic nodes without changing ``initial_static_ids``."""
    fragment = _fragment(prefix)
    state.flow.add_subgraph(fragment.nodes, fragment.edges, fragment.wiring)
    for node_id in fragment.nodes:
        state.node_state[node_id] = NodeState.TAKEN
        state.pool.set(node_id, {"retained": node_id})
        state.depth[node_id] = 4
        state.wiring_overrides[node_id] = {"copy": [node_id]}
    for edge in fragment.edges:
        state.edge_state[edge.id] = NodeState.TAKEN
    if residue:
        node_id = f"{prefix}/child/residue"
        node = _OverlayNode(node_id)
        state.flow.add_subgraph({node_id: node}, [], {node_id: {}})
        state.node_state[node_id] = NodeState.TAKEN
    return tuple(fragment.nodes)


def test_growth_canonicalizes_seed_and_isolates_builder_record_and_observers():
    state, log = _initialized_spawner()
    original_seed = {
        "items": ({"label": "stable"},),
        "nested": {"values": [1, 2]},
    }
    builder_inputs: list[Any] = []

    def mutating_builder(spawner: Node, seed: Any) -> Flow:
        builder_inputs.append(deepcopy(seed))
        seed["nested"]["values"].append(99)
        return _fragment(spawner.id)

    before = _snapshot(state)
    prepared = prepare_expanded(
        state,
        "spawn",
        Grow(seed=original_seed),
        mutating_builder,
    )

    assert _snapshot(state) == before
    assert builder_inputs == [
        {"items": [{"label": "stable"}], "nested": {"values": [1, 2]}}
    ]
    assert prepared.record == NodeExpandedRecord(
        spawner_id="spawn",
        seed={"items": [{"label": "stable"}], "nested": {"values": [1, 2]}},
        prune_ids=(),
        topology_digest=expansion_digest(_fragment("spawn")),
    )
    prepared.observers[0].grow.seed["nested"]["values"].append(88)
    original_seed["nested"]["values"].append(77)
    assert prepared.record.seed["nested"]["values"] == [1, 2]

    observers = commit_prepared(log, state, prepared)

    assert observers == (
        NodeExpanded(
            node_id="spawn",
            grow=Grow(
                seed={
                    "items": [{"label": "stable"}],
                    "nested": {"values": [1, 2]},
                }
            ),
        ),
    )
    assert observers[0] is not prepared.observers[0]
    assert state.node_state["spawn"] is NodeState.EXPANDED
    assert "spawn" not in state.in_flight
    assert tuple(state.ready) == ("spawn/child/start",)
    assert state.depth["spawn/child/end"] == 1
    assert state.next_sequence == 5


def test_growth_seed_rejection_retains_the_original_exception(monkeypatch):
    state, _ = _initialized_spawner()
    original = ValueError("seed codec rejected node data")

    def reject_seed(_seed: Any) -> tuple[Any, Any]:
        raise original

    monkeypatch.setattr(state_module, "canonical_seed", reject_seed)

    with pytest.raises(GrowthRejected) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda spawner, _seed: _fragment(spawner.id),
        )

    assert caught.value.original is original


@pytest.mark.parametrize(
    "seed",
    [
        {True: "call"},
        [{True: "map"}],
        ({True: "loop"}, 0),
    ],
    ids=("call", "map", "loop"),
)
def test_growth_rejects_non_string_record_keys_before_builder(seed: Any):
    state, _ = _initialized_spawner()
    builder_called = False

    def builder(spawner: Node, canonical_seed: Any) -> Flow:
        nonlocal builder_called
        builder_called = True
        return _fragment(spawner.id)

    with pytest.raises(GrowthRejected, match="keys must be strings") as caught:
        prepare_expanded(state, "spawn", Grow(seed=seed), builder)

    assert type(caught.value.original) is ValueError
    assert builder_called is False


def test_growth_rejects_a_cyclic_seed_before_builder():
    state, _ = _initialized_spawner()
    seed: list[Any] = []
    seed.append(seed)

    with pytest.raises(GrowthRejected, match="cyclic container") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed=seed),
            lambda spawner, _seed: _fragment(spawner.id),
        )

    assert type(caught.value.original) is ValueError


def test_growth_rejects_an_excessively_nested_seed_before_builder():
    state, _ = _initialized_spawner()
    seed: Any = "leaf"
    for _ in range(1_200):
        seed = [seed]
    builder_called = False

    def builder(_spawner: Node, _seed: Any) -> Flow:
        nonlocal builder_called
        builder_called = True
        return _fragment("spawn")

    with pytest.raises(GrowthRejected, match="nesting exceeds") as caught:
        prepare_expanded(state, "spawn", Grow(seed=seed), builder)

    assert type(caught.value.original) is ValueError
    assert builder_called is False


def test_growth_builder_receives_a_detached_spawner_and_failure_is_pure():
    state, _ = _initialized_spawner()
    live_spawner = state.flow.nodes["spawn"]
    live_spawner.title = "live"
    before = _snapshot(state)

    original = RuntimeError("builder failed")

    def mutating_failing_builder(spawner: Node, seed: Any) -> Flow:
        assert spawner is not live_spawner
        spawner.title = "builder-mutated"
        spawner.env["builder"] = seed
        raise original

    with pytest.raises(RuntimeError, match="builder failed") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={"value": 1}),
            mutating_failing_builder,
        )

    assert caught.value is original
    assert _snapshot(state) == before
    assert live_spawner.title == "live"
    assert live_spawner.env == {}


@pytest.mark.parametrize(
    "helper_name",
    ["_validate_growth_fragment_data", "expansion_digest", "_growth_adjacency"],
)
def test_growth_host_helper_faults_propagate_with_identity(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
):
    state, _ = _initialized_spawner()
    original = ValueError(f"{helper_name} fault")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise original

    monkeypatch.setattr(state_module, helper_name, fail)

    with pytest.raises(ValueError) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda spawner, _seed: _fragment(spawner.id),
        )

    assert caught.value is original
    assert not isinstance(caught.value, GrowthRejected)


@pytest.mark.parametrize("stage", ["spawner", "fragment"])
def test_growth_detachment_faults_propagate_with_identity(stage: str):
    original = TypeError(f"{stage} detachment failed")

    class FailingSpawner(_Spawner):
        def __deepcopy__(self, _memo: dict[int, Any]) -> Node:
            raise original

    class FailingFlow(Flow):
        def __deepcopy__(self, _memo: dict[int, Any]) -> Flow:
            raise original

    spawner: Node = FailingSpawner("spawn") if stage == "spawner" else _Spawner("spawn")
    state, _ = _initialized_spawner(_parent_flow(spawner))
    base = _fragment("spawn")
    built: Flow = (
        FailingFlow(
            nodes=base.nodes,
            edges=base.edges,
            wiring=base.wiring,
            start_id=base.start_id,
            end_id=base.end_id,
        )
        if stage == "fragment"
        else base
    )

    with pytest.raises(TypeError) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda _spawner, _seed: built,
        )

    assert caught.value is original
    assert not isinstance(caught.value, GrowthRejected)


@pytest.mark.parametrize("attribute", ["commit_as", "origin_id"])
@pytest.mark.parametrize(
    ("target", "error", "match"),
    [
        ("absent-target", ValueError, "target.*does not exist"),
        ("", ValueError, "non-empty"),
        (7, TypeError, "string or None"),
    ],
)
def test_growth_rejects_invalid_node_owned_topology_targets(
    attribute: str,
    target: Any,
    error: type[Exception],
    match: str,
):
    state, _ = _initialized_spawner()

    def invalid_target_builder(spawner: Node, seed: Any) -> Flow:
        fragment = _fragment(spawner.id)
        setattr(fragment.nodes[fragment.end_id], attribute, target)
        return fragment

    with pytest.raises(GrowthRejected, match=match) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            invalid_target_builder,
        )

    assert type(caught.value.original) is error


def test_growth_rejects_invalid_skipped_output_hook_target() -> None:
    """A future hook override cannot hide a missing logical output target."""
    state, _ = _initialized_spawner()

    def invalid_target_builder(spawner: Node, seed: Any) -> Flow:
        fragment = _fragment(spawner.id)
        fragment.nodes[fragment.end_id].skipped_output_target = lambda: "missing"
        return fragment

    with pytest.raises(GrowthRejected, match="skipped output target.*does not exist"):
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            invalid_target_builder,
        )


def test_growth_rejects_duplicate_redirect_owner_in_final_topology() -> None:
    """A growth fragment cannot add a second completion for one logical output."""
    state, log = _initialized_spawner()
    existing = _OverlayNode("existing-owner")
    existing.commit_as = "spawn"
    state.flow.add_subgraph(
        {existing.id: existing},
        [],
        {existing.id: {}},
    )
    state.node_state[existing.id] = NodeState.UNKNOWN
    before = _snapshot(state)

    with pytest.raises(GrowthRejected, match="multiple skipped output owners"):
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda spawner, seed: _fragment(spawner.id),
        )

    assert _snapshot(state) == before
    assert log.next_sequence == 4


def test_growth_rejects_executable_redirect_target_in_final_topology() -> None:
    """A completion cannot redirect to a node that still has incoming work."""
    state, log = _initialized_spawner()
    before = _snapshot(state)

    def executable_target_builder(spawner: Node, seed: Any) -> Flow:
        fragment = _fragment(spawner.id)
        fragment.nodes[fragment.end_id].commit_as = END_ID
        return fragment

    with pytest.raises(GrowthRejected, match="also executable"):
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            executable_target_builder,
        )

    assert _snapshot(state) == before
    assert log.next_sequence == 4


def test_growth_accepts_disconnected_inert_redirect_placeholder() -> None:
    """A sparse no-edge placeholder remains a valid logical output anchor."""
    state, log = _initialized_spawner()
    anchor = _OverlayNode("anchor")
    state.flow.add_subgraph({anchor.id: anchor}, [], {anchor.id: {}})
    state.node_state[anchor.id] = NodeState.UNKNOWN
    before = _snapshot(state)

    def inert_target_builder(spawner: Node, seed: Any) -> Flow:
        fragment = _fragment(spawner.id)
        fragment.nodes[fragment.end_id].commit_as = anchor.id
        return fragment

    prepared = prepare_expanded(
        state,
        "spawn",
        Grow(seed={}),
        inert_target_builder,
    )

    assert isinstance(prepared.record, NodeExpandedRecord)
    assert _snapshot(state) == before
    assert log.next_sequence == 4


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("source_handle", 7, TypeError),
        ("source_handle", "", ValueError),
        ("input_group", 7, TypeError),
        ("input_group", "", ValueError),
        ("optional", 1, TypeError),
        ("ordering", 1, TypeError),
    ],
)
def test_growth_rejects_invalid_edge_field_types(
    attribute: str,
    value: Any,
    error: type[Exception],
):
    state, _ = _initialized_spawner()

    def invalid_edge_builder(spawner: Node, seed: Any) -> Flow:
        fragment = _fragment(spawner.id)
        edge = fragment.edges[0]
        fields = {
            "id": edge.id,
            "from_": edge.from_,
            "to": edge.to,
            "source_handle": edge.source_handle,
            "input_group": edge.input_group,
            "optional": edge.optional,
            "ordering": edge.ordering,
        }
        fields[attribute] = value
        fragment.edges[0] = Edge(**fields)
        return fragment

    with pytest.raises(GrowthRejected, match=attribute) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            invalid_edge_builder,
        )

    assert type(caught.value.original) is error


class _FailingLog(MemoryRunLog):
    """Run log whose append fails before retaining the supplied transition."""

    def append(self, record):
        if isinstance(record, NodeExpandedRecord):
            raise OSError("durability unavailable")
        return super().append(record)


def test_growth_append_failure_leaves_topology_and_every_state_coordinate_unchanged():
    state, prior = _initialized_spawner()
    log = _FailingLog(_RUN_ID)
    for envelope in prior:
        log.append(envelope.record)
    prepared = prepare_expanded(
        state,
        "spawn",
        Grow(seed={"value": "stable"}),
        lambda spawner, seed: _fragment(spawner.id),
    )
    before = _snapshot(state)

    with pytest.raises(OSError, match="durability unavailable"):
        commit_prepared(log, state, prepared)

    assert _snapshot(state) == before
    assert log.next_sequence == prior.next_sequence


def test_growth_replay_rejects_a_changed_but_valid_builder_digest():
    live, _ = _initialized_spawner()
    prepared = prepare_expanded(
        live,
        "spawn",
        Grow(seed={"value": "stable"}),
        lambda spawner, seed: _fragment(spawner.id),
    )
    replay, _ = _initialized_spawner()

    with pytest.raises(ValueError, match="digest"):
        prepare_record(
            replay,
            prepared.record,
            grow_builder=lambda spawner, seed: _fragment(
                spawner.id,
                suffix="changed",
            ),
        )

    assert "spawn/changed/start" not in replay.flow.nodes


def test_growth_replay_unwraps_shared_growth_rejection_identity(monkeypatch):
    live, _ = _initialized_spawner()
    prepared = prepare_expanded(
        live,
        "spawn",
        Grow(seed={}),
        lambda spawner, _seed: _fragment(spawner.id),
    )
    replay, _ = _initialized_spawner()
    original = ValueError("recorded fragment is inconsistent")

    def reject_fragment(*_args: Any, **_kwargs: Any) -> Any:
        raise GrowthRejected(original)

    monkeypatch.setattr(
        state_module,
        "_validate_growth_fragment_data",
        reject_fragment,
    )

    with pytest.raises(ValueError) as caught:
        prepare_record(
            replay,
            prepared.record,
            grow_builder=lambda spawner, _seed: _fragment(spawner.id),
        )

    assert caught.value is original
    assert not isinstance(caught.value, GrowthRejected)
    assert caught.value.__cause__ is None


def test_growth_live_and_record_replay_produce_equal_state():
    live, live_log = _initialized_spawner()
    replay, replay_log = _initialized_spawner()
    builder = lambda spawner, seed: _fragment(spawner.id)

    live_prepared = prepare_expanded(
        live,
        "spawn",
        Grow(seed={"labels": ("a", "b")}),
        builder,
    )
    live_observers = commit_prepared(live_log, live, live_prepared)
    replay_prepared = prepare_record(
        replay,
        live_prepared.record,
        grow_builder=builder,
    )
    replay_observers = commit_prepared(replay_log, replay, replay_prepared)

    assert replay_observers == live_observers
    assert _snapshot(replay) == _snapshot(live)


def test_growth_record_replay_requires_the_injected_builder():
    state, _ = _initialized_spawner()
    record = NodeExpandedRecord(
        spawner_id="spawn",
        seed={},
        prune_ids=(),
        topology_digest="a" * 64,
    )

    with pytest.raises((TypeError, ValueError), match="grow_builder"):
        prepare_record(state, record)


@pytest.mark.parametrize("kind", ["call", "map", "loop"])
def test_actual_build_grow_splices_call_map_and_loop_with_fifo_and_depth(kind: str):
    child = _canonical_child()
    if kind == "call":
        spawner: Node = CallNode("spawn", flow_id="child", child=child)
        spawner.params = []
        grow = Grow(seed={})
        expected_depth = {"spawn/__end__": 3}
    elif kind == "map":
        spawner = MapNode("spawn", flow_id="child", child=child)
        spawner.params = []
        grow = Grow(seed=[{}, {}])
        expected_depth = {"spawn/__end__": 3}
    else:
        spawner = LoopNode(
            "spawn",
            flow_id="child",
            child=child,
            child_inputs=[],
            predicate_kind="times",
            times=3,
            max_iters=3,
            origin_id="spawn",
        )
        spawner.params = []
        grow = Grow(seed=({}, 0))
        expected_depth = {"spawn~1": 2}

    flow = _parent_flow(spawner, sibling=True)
    state, log = _initialized_spawner(flow)
    state.depth["spawn"] = 2
    prior_pool = state.pool.model_dump(mode="json")

    prepared = prepare_expanded(state, "spawn", grow, build_grow)
    observers = commit_prepared(log, state, prepared)

    assert prepared.record.spawner_id == "spawn"
    assert prepared.record.topology_digest == expansion_digest(
        build_grow(spawner, prepared.record.seed)
    )
    assert observers == prepared.observers
    assert state.node_state["spawn"] is NodeState.EXPANDED
    assert "spawn" not in state.in_flight
    assert state.edge_state["spawn-end"] is NodeState.UNKNOWN
    assert tuple(state.ready)[0] == "side"
    assert tuple(state.ready)[-1] == prepared._plan.delta.fragment_start_id
    assert state.pool.model_dump(mode="json") == prior_pool
    assert state.result is None
    for node_id, depth in expected_depth.items():
        assert state.depth[node_id] == depth


def test_actual_call_growth_allows_an_omitted_optional_defaulted_child_input():
    child = _canonical_child(optional_input=True)
    spawner = CallNode("spawn", flow_id="child", child=child)
    spawner.params = []
    state, log = _initialized_spawner(_parent_flow(spawner))

    prepared = prepare_expanded(state, "spawn", Grow(seed={}), build_grow)
    commit_prepared(log, state, prepared)

    cloned_start = state.flow.nodes["spawn/__start__"]
    assert [param.name for param in cloned_start.params] == ["optional"]
    assert state.flow.wiring["spawn/__start__"] == {}


def test_zero_field_loop_growth_schedules_its_next_driver_after_the_body():
    child = _canonical_child()
    loop = LoopNode(
        "spawn",
        flow_id="child",
        child=child,
        child_inputs=[],
        predicate_kind="times",
        times=2,
        max_iters=2,
        origin_id="spawn",
    )
    loop.params = []
    state, log = _initialized_spawner(_parent_flow(loop))

    expansion = prepare_expanded(
        state,
        "spawn",
        Grow(seed=({}, 0)),
        build_grow,
    )
    commit_prepared(log, state, expansion)
    body_start = expansion._plan.delta.fragment_start_id
    body_end = expansion._plan.delta.fragment_end_id

    commit_prepared(log, state, prepare_node_started(state, body_start))
    commit_prepared(log, state, prepare_node_succeeded(state, body_start, {}))
    commit_prepared(log, state, prepare_node_started(state, body_end))
    commit_prepared(log, state, prepare_node_succeeded(state, body_end, None))

    assert tuple(state.ready) == ("spawn~1",)


@pytest.mark.parametrize("ambient", ["today", "now", "run_id"])
def test_growth_accepts_the_closed_system_ambients(ambient: str):
    state, _ = _initialized_spawner()

    def system_reference_builder(spawner: Node, seed: Any) -> Flow:
        fragment = _fragment(spawner.id)
        end = EndNode.record(fragment.end_id, output_names=["value"])
        end.commit_as = spawner.id
        fragment.nodes[fragment.end_id] = end
        fragment.wiring[fragment.end_id] = {
            "value": Binding(f"${{system.{ambient}}}")
        }
        return fragment

    prepare_expanded(
        state,
        "spawn",
        Grow(seed={}),
        system_reference_builder,
    )


def test_growth_accepts_typed_fields_and_nested_map_item_scope():
    source_type = Type(
        kind=ValueKind.OBJECT,
        fields={"known": Type.scalar(ValueKind.STRING)},
        required=frozenset({"known"}),
    )
    state, _ = _initialized_spawner(
        _parent_flow(
            _Spawner(
                "spawn",
                output_contract=ValueContract(
                    source_type,
                    ContractOrigins(
                        default=OriginRef("spawn/output"),
                        entries=((('known',), OriginRef("spawn/output/known")),),
                    ),
                ),
            )
        )
    )

    def typed_map_builder(spawner: Node, seed: Any) -> Flow:
        node_id = f"{spawner.id}/nested-map"
        node = MapNode(node_id, flow_id="child", child=_canonical_child())
        node.params = [Parameter(name="value")]
        return Flow(
            nodes={node_id: node},
            edges=[],
            wiring={
                node_id: {
                    "over": f"${{{spawner.id}.output.known}}",
                    "value": "${item.known}",
                }
            },
            start_id=node_id,
            end_id=node_id,
        )

    prepare_expanded(
        state,
        "spawn",
        Grow(seed={}),
        typed_map_builder,
    )


def _missing_required_fragment(spawner_id: str) -> Flow:
    """Return a fragment whose required START param has no wiring source."""
    start_id = f"{spawner_id}/required/start"
    end_id = f"{spawner_id}/required/end"
    start = StartNode(
        start_id,
        params=[
            Parameter(
                name="required",
                contract=ValueContract(Type.scalar(ValueKind.STRING)),
                required=True,
                coerce_strings=True,
            )
        ],
    )
    end = EndNode.record(end_id, output_names=[])
    return Flow(
        nodes={start_id: start, end_id: end},
        edges=[Edge("required-edge", start_id, end_id, ordering=True)],
        wiring={start_id: {}, end_id: {}},
        start_id=start_id,
        end_id=end_id,
    )


def _missing_reserved_fragment(spawner_id: str) -> Flow:
    """Return a one-node MAP fragment omitting its reserved ``over`` wiring."""
    node_id = f"{spawner_id}/nested-map"
    node = MapNode(node_id, flow_id="child", child=_canonical_child())
    node.params = []
    return Flow(
        nodes={node_id: node},
        edges=[],
        wiring={node_id: {}},
        start_id=node_id,
        end_id=node_id,
    )


def _unknown_reference_fragment(spawner_id: str) -> Flow:
    """Return a fragment with a valid param whose source head is absent."""
    start_id = f"{spawner_id}/ref/start"
    end_id = f"{spawner_id}/ref/end"
    start = StartNode(start_id, params=[])
    end = EndNode.record(end_id, output_names=["value"])
    return Flow(
        nodes={start_id: start, end_id: end},
        edges=[Edge("ref-edge", start_id, end_id, input_group="value")],
        wiring={start_id: {}, end_id: {"value": "${absent.output}"}},
        start_id=start_id,
        end_id=end_id,
    )


def _invalid_fragment(case: str) -> Any:
    """Build one isolated malformed builder result for the validation table."""
    if case == "not-flow":
        return object()
    if case == "missing-required":
        return _missing_required_fragment("spawn")
    if case == "missing-reserved":
        return _missing_reserved_fragment("spawn")
    if case == "unknown-reference":
        return _unknown_reference_fragment("spawn")

    fragment = _fragment("spawn")
    if case == "missing-start":
        fragment.start_id = "absent"
    elif case == "missing-end":
        fragment.end_id = "absent"
    elif case == "key-id-mismatch":
        start = fragment.nodes.pop(fragment.start_id)
        fragment.nodes["wrong-key"] = start
        fragment.wiring["wrong-key"] = fragment.wiring.pop(fragment.start_id)
    elif case == "empty-node-id":
        start = fragment.nodes.pop(fragment.start_id)
        fragment.wiring.pop(fragment.start_id)
        start.id = ""
        fragment.nodes[""] = start
        fragment.wiring[""] = {}
        fragment.start_id = ""
    elif case == "missing-wiring-entry":
        fragment.wiring.pop(fragment.end_id)
    elif case == "extra-wiring-key":
        fragment.wiring[fragment.start_id]["extra"] = Binding(1)
    elif case == "raw-param":
        fragment.nodes[fragment.start_id].params = ["raw"]  # type: ignore[list-item]
    elif case == "parameter-subclass":
        fragment.nodes[fragment.start_id].params = [_ParameterSubclass("x")]
    elif case == "raw-output-contract":
        fragment.nodes[fragment.start_id].output_contract = Type.scalar(  # type: ignore[assignment]
            ValueKind.STRING
        )
    elif case == "empty-param-tuple":
        fragment.nodes[fragment.start_id].params = ()  # type: ignore[assignment]
    elif case == "param-tuple":
        fragment.nodes[fragment.start_id].params = (Parameter("x"),)  # type: ignore[assignment]
    elif case == "duplicate-edge-id":
        edge = fragment.edges[0]
        fragment.edges.append(
            Edge(edge.id, edge.from_, edge.to, ordering=True)
        )
    elif case == "empty-edge-id":
        edge = fragment.edges[0]
        fragment.edges[0] = Edge("", edge.from_, edge.to, ordering=True)
    elif case == "node-collision":
        start = fragment.nodes.pop(fragment.start_id)
        fragment.wiring.pop(fragment.start_id)
        start.id = "spawn"
        fragment.nodes["spawn"] = start
        fragment.wiring["spawn"] = {}
        fragment.start_id = "spawn"
        edge = fragment.edges[0]
        fragment.edges[0] = Edge(edge.id, "spawn", edge.to, ordering=True)
    elif case == "edge-collision":
        edge = fragment.edges[0]
        fragment.edges[0] = Edge(
            "start-spawn", edge.from_, edge.to, ordering=True
        )
    elif case == "unknown-endpoint":
        edge = fragment.edges[0]
        fragment.edges[0] = Edge(edge.id, "absent", edge.to, ordering=True)
    elif case == "unreachable-node":
        orphan_id = "spawn/child/orphan"
        fragment.nodes[orphan_id] = StartNode(orphan_id, params=[])
        fragment.wiring[orphan_id] = {}
    elif case == "unknown-system-ambient":
        end = EndNode.record(fragment.end_id, output_names=["value"])
        end.commit_as = "spawn"
        fragment.nodes[fragment.end_id] = end
        fragment.wiring[fragment.end_id] = {
            "value": Binding("${system.not_declared}")
        }
    elif case == "unknown-typed-field":
        fragment.nodes[fragment.start_id].output_contract = ValueContract(
            Type(
                kind=ValueKind.OBJECT,
                fields={"known": Type.scalar(ValueKind.STRING)},
                required=frozenset({"known"}),
            )
        )
        end = EndNode.record(fragment.end_id, output_names=["value"])
        end.commit_as = "spawn"
        fragment.nodes[fragment.end_id] = end
        fragment.wiring[fragment.end_id] = {
            "value": Binding(f"${{{fragment.start_id}.output.absent}}")
        }
        fragment.edges[0] = Edge(
            fragment.edges[0].id,
            fragment.start_id,
            fragment.end_id,
            input_group="value",
        )
    elif case == "unknown-partial-field":
        fragment.nodes[fragment.start_id].output_contract = PartialRecordContract(
            fields=(
                ("known", ValueContract(Type.scalar(ValueKind.STRING))),
                ("opaque", None),
            )
        )
        end = EndNode.record(fragment.end_id, output_names=["value"])
        end.commit_as = "spawn"
        fragment.nodes[fragment.end_id] = end
        fragment.wiring[fragment.end_id] = {
            "value": Binding(f"${{{fragment.start_id}.output.absent}}")
        }
        fragment.edges[0] = Edge(
            fragment.edges[0].id,
            fragment.start_id,
            fragment.end_id,
            input_group="value",
        )
    elif case == "cycle":
        fragment.edges.append(
            Edge(
                "cycle-back",
                fragment.end_id,
                fragment.start_id,
                ordering=True,
            )
        )
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(case)
    return fragment


@pytest.mark.parametrize(
    "case",
    [
        "missing-start",
        "missing-end",
        "key-id-mismatch",
        "empty-node-id",
        "missing-wiring-entry",
        "extra-wiring-key",
        "raw-param",
        "parameter-subclass",
        "raw-output-contract",
        "empty-param-tuple",
        "param-tuple",
        "missing-required",
        "missing-reserved",
        "duplicate-edge-id",
        "empty-edge-id",
        "node-collision",
        "edge-collision",
        "unknown-endpoint",
        "unreachable-node",
        "unknown-system-ambient",
        "unknown-typed-field",
        "unknown-partial-field",
        "unknown-reference",
        "cycle",
    ],
)
def test_growth_rejects_invalid_fragments_atomically(case: str):
    state, log = _initialized_spawner()
    before = _snapshot(state)

    with pytest.raises(GrowthRejected) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda spawner, seed: _invalid_fragment(case),
        )

    assert type(caught.value.original) in (TypeError, ValueError)
    assert _snapshot(state) == before
    assert log.next_sequence == 4


def test_growth_builder_non_flow_return_is_a_raw_host_fault():
    state, _ = _initialized_spawner()

    with pytest.raises(TypeError, match="grow_builder must return a Flow") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda _spawner, _seed: object(),
        )

    assert not isinstance(caught.value, GrowthRejected)


def test_growth_reference_validation_reads_a_binding_source(
    monkeypatch: pytest.MonkeyPatch,
):
    """Valid wrapped references reach the same closed-reference scanner as raw text."""
    state, _log = _initialized_spawner()
    fragment = _unknown_reference_fragment("spawn")
    source = f"${{{fragment.start_id}.output}}"
    fragment.wiring[fragment.end_id]["value"] = Binding(
        source,
        OriginRef("growth-reference"),
    )
    scanned: list[str] = []
    scan_refs = state_module.expr_refs_of

    def record_scan(value: str) -> list[str]:
        """Record the unwrapped scanner input while delegating real classification."""
        scanned.append(value)
        return scan_refs(value)

    monkeypatch.setattr(state_module, "expr_refs_of", record_scan)

    node_ids, edge_ids = state_module._validate_growth_fragment_data(state, fragment)

    assert node_ids == (fragment.start_id, fragment.end_id)
    assert edge_ids == ("ref-edge",)
    assert scanned == [source]


def test_growth_rejects_a_binding_wrapped_unknown_reference_before_digesting():
    """Wrapped bad references fail as node data before manifest identity is needed."""
    state, log = _initialized_spawner()
    fragment = _unknown_reference_fragment("spawn")
    fragment.wiring[fragment.end_id]["value"] = Binding(
        "${absent.output}",
        OriginRef("unknown-growth-reference"),
    )
    before = _snapshot(state)

    with pytest.raises(GrowthRejected, match="unknown namespace") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda _spawner, _seed: fragment,
        )

    assert type(caught.value.original) is ValueError
    assert _snapshot(state) == before
    assert log.next_sequence == 4


@pytest.mark.parametrize(
    ("grow", "builder", "error", "original_type"),
    [
        (
            Grow(seed={}, prune={"old"}),
            lambda spawner, seed: _fragment(spawner.id),
            GrowthRejected,
            TypeError,
        ),
        (
            Grow(seed={}, prune=frozenset({""})),
            lambda spawner, seed: _fragment(spawner.id),
            GrowthRejected,
            ValueError,
        ),
        (
            Grow(seed={"bad": {1, 2}}),
            lambda spawner, seed: _fragment(spawner.id),
            GrowthRejected,
            ValueError,
        ),
        (Grow(seed={}), None, TypeError, None),
    ],
)
def test_growth_requires_exact_live_carriers(
    grow: Grow,
    builder: Any,
    error: type[Exception],
    original_type: type[Exception] | None,
):
    state, _ = _initialized_spawner()

    with pytest.raises(error) as caught:
        prepare_expanded(state, "spawn", grow, builder)

    if original_type is None:
        assert not isinstance(caught.value, GrowthRejected)
    else:
        assert type(caught.value.original) is original_type


def test_growth_rejects_malformed_static_contract_as_a_host_invariant():
    """Static IR corruption remains outside the builder-owned fragment boundary."""
    state, log = _initialized_spawner()
    state.flow.nodes["spawn"].output_contract = Type.scalar(  # type: ignore[assignment]
        ValueKind.STRING
    )
    before = _snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(TypeError, match="exact ValueContract") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda spawner, seed: _fragment(spawner.id),
        )

    assert not isinstance(caught.value, GrowthRejected)
    assert _snapshot(state) == before
    assert log.next_sequence == sequence


def test_growth_rejects_a_static_cycle_as_a_host_invariant():
    """A corrupt live graph must not become node-owned fragment rejection."""
    state, log = _initialized_spawner()
    state.flow.edges.append(
        Edge(
            id="corrupt-static-cycle",
            from_=state.flow.end_id,
            to=state.flow.start_id,
        )
    )
    before = _snapshot(state)

    with pytest.raises(ValueError, match="cycle") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda spawner, _seed: _fragment(spawner.id),
        )

    assert not isinstance(caught.value, GrowthRejected)
    assert _snapshot(state) == before
    assert log.next_sequence == 4


def test_growth_enforces_peak_node_budget_before_same_transition_pruning(monkeypatch):
    state, _ = _initialized_spawner()
    old_ids = _install_settled_overlay(state)
    new_count = len(_fragment("spawn").nodes)
    final_count = len(state.flow.nodes) - len(old_ids) + new_count
    peak_count = len(state.flow.nodes) + new_count
    assert final_count < peak_count
    monkeypatch.setattr(state_module, "MAX_TOTAL_NODES", peak_count - 1)
    before = _snapshot(state)

    with pytest.raises(GrowthRejected, match="node budget") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}, prune=frozenset(old_ids)),
            lambda spawner, seed: _fragment(spawner.id),
        )

    assert type(caught.value.original) is ValueError
    assert _snapshot(state) == before


def test_growth_accepts_peak_node_count_equal_to_budget(monkeypatch):
    state, log = _initialized_spawner()
    old_ids = _install_settled_overlay(state)
    new_count = len(_fragment("spawn").nodes)
    monkeypatch.setattr(
        state_module,
        "MAX_TOTAL_NODES",
        len(state.flow.nodes) + new_count,
    )

    commit_prepared(
        log,
        state,
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}, prune=frozenset(old_ids)),
            lambda spawner, seed: _fragment(spawner.id),
        ),
    )

    assert not (set(old_ids) & set(state.flow.nodes))
    assert "spawn/child/start" in state.flow.nodes


@pytest.mark.parametrize("delta", [True, -1, 1.5, "1"])
def test_growth_rejects_invalid_exact_depth_deltas(delta: Any):
    spawner = _Spawner("spawn")
    spawner.grow_depth_delta = delta
    state, _ = _initialized_spawner(_parent_flow(spawner))
    before = _snapshot(state)

    with pytest.raises(ValueError, match="grow_depth_delta") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda node, seed: _fragment(node.id),
        )

    assert not isinstance(caught.value, GrowthRejected)
    assert _snapshot(state) == before


def test_growth_treats_corrupt_stored_depth_as_a_host_invariant_error():
    state, _ = _initialized_spawner()
    state.depth["spawn"] = "corrupt"
    before = _snapshot(state)

    with pytest.raises(ValueError, match="invalid parent depth") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda node, seed: _fragment(node.id),
        )

    assert not isinstance(caught.value, GrowthRejected)
    assert _snapshot(state) == before


def test_growth_enforces_positive_reference_depth_and_stamps_the_boundary():
    rejected, _ = _initialized_spawner()
    rejected.depth["spawn"] = state_module.MAX_REF_DEPTH
    with pytest.raises(GrowthRejected, match="MAX_REF_DEPTH") as caught:
        prepare_expanded(
            rejected,
            "spawn",
            Grow(seed={}),
            lambda node, seed: _fragment(node.id),
        )
    assert type(caught.value.original) is ValueError

    accepted, log = _initialized_spawner()
    accepted.depth["spawn"] = state_module.MAX_REF_DEPTH - 1
    commit_prepared(
        log,
        accepted,
        prepare_expanded(
            accepted,
            "spawn",
            Grow(seed={}),
            lambda node, seed: _fragment(node.id),
        ),
    )

    assert accepted.depth["spawn/child/end"] == state_module.MAX_REF_DEPTH


def test_growth_prunes_only_named_settled_dynamic_overlay_coordinates():
    state, log = _initialized_spawner()
    old_ids = _install_settled_overlay(state, residue=True)
    old_edge_ids = {
        edge.id
        for edge in state.flow.edges
        if edge.from_ in old_ids or edge.to in old_ids
    }
    residue_id = "old/child/residue"

    commit_prepared(
        log,
        state,
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}, prune=frozenset(old_ids)),
            lambda node, seed: _fragment(node.id),
        ),
    )

    for node_id in old_ids:
        assert node_id not in state.flow.nodes
        assert node_id not in state.flow.wiring
        assert node_id not in state.node_state
        assert node_id not in state.pool.store
        assert node_id not in state.depth
        assert node_id not in state.wiring_overrides
        assert node_id not in state.ready
        assert node_id not in state.in_flight
        assert node_id not in state.flow._incoming
        assert node_id not in state.flow._outgoing
    assert not (old_edge_ids & set(state.edge_state))
    assert not any(
        edge.from_ in old_ids or edge.to in old_ids for edge in state.flow.edges
    )
    assert residue_id in state.flow.nodes
    assert state.node_state[residue_id] is NodeState.TAKEN


def test_growth_preserves_unrelated_in_flight_siblings_and_fifo_order():
    state, log = _initialized_spawner(_parent_flow(sibling=True))
    assert tuple(state.ready) == ("side",)
    state.ready.popleft()
    state.in_flight["side"] = None

    commit_prepared(
        log,
        state,
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}),
            lambda node, seed: _fragment(node.id),
        ),
    )

    assert tuple(state.in_flight) == ("side",)
    assert tuple(state.ready) == ("spawn/child/start",)
    assert state.node_state["side"] is NodeState.UNKNOWN


def test_growth_rejects_live_node_collision_even_when_the_id_is_pruned():
    state, _ = _initialized_spawner()
    colliding_id = "spawn/child/start"
    existing = _OverlayNode(colliding_id)
    state.flow.add_subgraph(
        {colliding_id: existing},
        [],
        {colliding_id: {}},
    )
    state.node_state[colliding_id] = NodeState.TAKEN
    before = _snapshot(state)

    with pytest.raises(GrowthRejected, match="collides") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}, prune=frozenset({colliding_id})),
            lambda node, seed: _fragment(node.id),
        )

    assert type(caught.value.original) is ValueError
    assert _snapshot(state) == before


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "top-start",
        "top-end",
        "initial-static",
        "ready",
        "unrelated-in-flight",
        "unsettled",
        "non-loop-self",
        "loop-missing-origin",
        "loop-pruned-origin",
        "edge-dependency",
        "wiring-dependency",
        "commit-dependency",
        "origin-dependency",
        "skipped-output-dependency",
    ],
)
def test_growth_rejects_unsafe_pruning_matrix_atomically(case: str):
    state, log = _initialized_spawner()
    old_ids = _install_settled_overlay(state)
    old_id = old_ids[0]
    prune_ids: frozenset[str]
    executing = "spawn"

    if case == "missing":
        prune_ids = frozenset({"absent"})
    elif case == "top-start":
        prune_ids = frozenset({START_ID})
    elif case == "top-end":
        prune_ids = frozenset({END_ID})
    elif case == "initial-static":
        prune_ids = frozenset({"spawn"})
    elif case == "ready":
        state.ready.append(old_id)
        prune_ids = frozenset({old_id})
    elif case == "unrelated-in-flight":
        state.node_state[old_id] = NodeState.UNKNOWN
        state.in_flight[old_id] = None
        prune_ids = frozenset({old_id})
    elif case == "unsettled":
        state.node_state[old_id] = NodeState.UNKNOWN
        prune_ids = frozenset({old_id})
    elif case == "non-loop-self":
        clone = _Spawner("clone")
        state.flow.add_subgraph({"clone": clone}, [], {"clone": {}})
        state.node_state["clone"] = NodeState.UNKNOWN
        state.in_flight = {"clone": None}
        executing = "clone"
        prune_ids = frozenset({"clone"})
    elif case == "loop-missing-origin":
        clone = LoopNode(
            "loop~1",
            flow_id="child",
            child=_canonical_child(),
            child_inputs=[],
            predicate_kind="times",
            times=3,
            max_iters=3,
            iteration=1,
            origin_id="missing-origin",
        )
        clone.params = []
        state.flow.add_subgraph({clone.id: clone}, [], {clone.id: {}})
        state.node_state[clone.id] = NodeState.UNKNOWN
        state.in_flight = {clone.id: None}
        executing = clone.id
        prune_ids = frozenset({clone.id})
    elif case == "loop-pruned-origin":
        origin = _OverlayNode("loop-origin")
        clone = LoopNode(
            "loop-origin~1",
            flow_id="child",
            child=_canonical_child(),
            child_inputs=[],
            predicate_kind="times",
            times=3,
            max_iters=3,
            iteration=1,
            origin_id=origin.id,
        )
        clone.params = []
        state.flow.add_subgraph(
            {origin.id: origin, clone.id: clone},
            [],
            {origin.id: {}, clone.id: {}},
        )
        state.node_state[origin.id] = NodeState.TAKEN
        state.node_state[clone.id] = NodeState.UNKNOWN
        state.in_flight = {clone.id: None}
        executing = clone.id
        prune_ids = frozenset({origin.id, clone.id})
    else:
        dependent = END_ID
        if case == "edge-dependency":
            edge = Edge("old-to-end", old_id, dependent, ordering=True)
            state.flow.add_subgraph({}, [edge], {})
            state.edge_state[edge.id] = NodeState.UNKNOWN
        elif case == "wiring-dependency":
            state.flow.wiring[dependent] = {
                "value": Binding(f"${{{old_id}.output}}")
            }
        elif case == "commit-dependency":
            state.flow.nodes[dependent].commit_as = old_id
        elif case == "origin-dependency":
            state.flow.nodes[dependent].origin_id = old_id
        elif case == "skipped-output-dependency":
            state.flow.nodes[dependent].skipped_output_target = lambda: old_id
        prune_ids = frozenset({old_id})

    before = _snapshot(state)
    with pytest.raises(GrowthRejected) as caught:
        prepare_expanded(
            state,
            executing,
            Grow(seed={}, prune=prune_ids),
            lambda node, seed: _fragment(node.id),
        )

    assert type(caught.value.original) is ValueError
    assert _snapshot(state) == before
    assert log.next_sequence == 4


def test_growth_rejects_a_binding_wrapped_dependency_on_a_pruned_node():
    """A wrapped reference cannot hide a live dependency on removed topology."""
    state, log = _initialized_spawner()
    old_id = _install_settled_overlay(state)[0]
    state.flow.wiring[END_ID] = {
        "value": Binding(
            f"${{{old_id}.output}}",
            OriginRef("pruned-dependency"),
        )
    }
    before = _snapshot(state)

    with pytest.raises(
        GrowthRejected,
        match="wiring depends on pruned node",
    ) as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}, prune=frozenset({old_id})),
            lambda node, _seed: _fragment(node.id),
        )

    assert type(caught.value.original) is ValueError
    assert _snapshot(state) == before
    assert log.next_sequence == 4


def test_growth_prune_scan_rejects_malformed_surviving_wiring_as_raw_invariant(
    monkeypatch: pytest.MonkeyPatch,
):
    state, _ = _initialized_spawner()
    old_id = _install_settled_overlay(state)[0]
    state.flow.wiring[END_ID] = {"value": Binding("${malformed}")}
    parse_error = state_module.ExpressionError("malformed surviving wiring")

    def fail_scan(_source: str) -> list[str]:
        raise parse_error

    monkeypatch.setattr(state_module, "expr_refs_of", fail_scan)

    with pytest.raises(ValueError, match="invalid wiring") as caught:
        prepare_expanded(
            state,
            "spawn",
            Grow(seed={}, prune=frozenset({old_id})),
            lambda node, _seed: _fragment(node.id),
        )

    assert not isinstance(caught.value, GrowthRejected)
    assert caught.value.__cause__ is parse_error


def test_loop_clone_record_uses_actual_id_self_prunes_and_inherits_depth_zero():
    child = _canonical_child()
    loop = LoopNode(
        "spawn",
        flow_id="child",
        child=child,
        child_inputs=[],
        predicate_kind="times",
        times=3,
        max_iters=3,
        origin_id="spawn",
    )
    loop.params = []
    state, log = _initialized_spawner(_parent_flow(loop))
    state.depth["spawn"] = 2

    commit_prepared(
        log,
        state,
        prepare_expanded(state, "spawn", Grow(seed=({}, 0)), build_grow),
    )
    first_dynamic = set(state.flow.nodes) - state.initial_static_ids
    assert "spawn~1" in first_dynamic
    assert state.depth["spawn~1"] == 2
    for node_id in first_dynamic:
        state.node_state[node_id] = (
            NodeState.UNKNOWN if node_id == "spawn~1" else NodeState.TAKEN
        )
    for edge_id in set(state.edge_state) - {"start-spawn", "spawn-end"}:
        state.edge_state[edge_id] = NodeState.TAKEN
    state.ready.clear()
    state.in_flight = {"spawn~1": None}

    clone = state.flow.nodes["spawn~1"]
    grow = clone.run({})
    assert type(grow) is Grow
    prepared = prepare_expanded(state, "spawn~1", grow, build_grow)

    assert prepared.record.spawner_id == "spawn~1"
    assert "spawn~1" in prepared.record.prune_ids
    observers = commit_prepared(log, state, prepared)

    assert observers[0].node_id == "spawn~1"
    assert "spawn~1" not in state.flow.nodes
    assert "spawn" in state.flow.nodes
    assert state.node_state["spawn"] is NodeState.EXPANDED
    assert "spawn~2" in state.flow.nodes
    assert state.depth["spawn~2"] == 2
    assert not any(node_id.startswith("spawn#0/") for node_id in state.flow.nodes)


def test_growth_commit_performs_no_builder_codec_graph_or_event_work_after_append(
    monkeypatch,
):
    state, prior = _initialized_spawner()
    built = _fragment("spawn")
    builder_seed: list[Any] = []
    builder_calls = 0

    def builder(spawner: Node, seed: Any) -> Flow:
        nonlocal builder_calls
        builder_calls += 1
        builder_seed.append(seed)
        return built

    raw_seed = {"nested": {"items": ["stable"]}}
    prepared = prepare_expanded(state, "spawn", Grow(seed=raw_seed), builder)
    expected_start = prepared._plan.delta.fragment_start_id

    # Every object still owned by the caller/builder may change after preparation.
    raw_seed["nested"]["items"].append("raw-mutated")
    builder_seed[0]["nested"]["items"].append("builder-mutated")
    built.nodes.clear()
    built.edges.clear()
    built.wiring.clear()
    prepared.observers[0].grow.seed["nested"]["items"].append("observer-mutated")

    guard = {"appended": False}

    class _GuardedLog(MemoryRunLog):
        def append(self, record):
            envelope = super().append(record)
            if isinstance(record, NodeExpandedRecord):
                guard["appended"] = True
            return envelope

    log = _GuardedLog(_RUN_ID)
    for envelope in prior:
        log.append(envelope.record)

    original_deepcopy = state_module.copy.deepcopy
    original_canonical_seed = state_module.canonical_seed
    original_digest = state_module.expansion_digest
    original_event = state_module.NodeExpanded

    def before_append(callable_):
        def guarded(*args, **kwargs):
            assert not guard["appended"]
            return callable_(*args, **kwargs)

        return guarded

    monkeypatch.setattr(state_module.copy, "deepcopy", before_append(original_deepcopy))
    monkeypatch.setattr(
        state_module,
        "canonical_seed",
        before_append(original_canonical_seed),
    )
    monkeypatch.setattr(
        state_module,
        "expansion_digest",
        before_append(original_digest),
    )
    monkeypatch.setattr(
        state_module,
        "NodeExpanded",
        before_append(original_event),
    )
    monkeypatch.setattr(
        CompiledFlow,
        "add_subgraph",
        lambda *args, **kwargs: pytest.fail("add_subgraph ran during commit"),
    )
    monkeypatch.setattr(
        CompiledFlow,
        "remove_subgraph",
        lambda *args, **kwargs: pytest.fail("remove_subgraph ran during commit"),
    )

    observers = commit_prepared(log, state, prepared)

    assert guard["appended"]
    assert builder_calls == 1
    assert expected_start in state.flow.nodes
    assert tuple(state.ready)[-1] == expected_start
    assert observers[0].grow.seed == {"nested": {"items": ["stable"]}}
