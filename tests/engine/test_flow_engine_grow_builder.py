"""The runtime receives growth construction as an injected plain callable."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from agent_composer.compile.model import END_ID, START_ID, Flow
from agent_composer.events import NodeExpanded, RunFailed
from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.runtime.engine import FlowEngine
from agent_composer.suspension.expansions import GrowRecord
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _graph


def test_runtime_import_does_not_load_kind_aware_growth_builder():
    """The runtime owns only the injected seam, not the compile-owned implementation."""
    package_dir = Path(__file__).resolve().parents[2] / "src" / "agent_composer"
    script = """
import importlib
import sys
import types

package = types.ModuleType("agent_composer")
package.__path__ = [sys.argv[1]]
package.__package__ = "agent_composer"
# Bypass the package-root public API, which intentionally imports run.py, so this process measures
# the runtime layer's own transitive imports.
sys.modules["agent_composer"] = package

importlib.import_module("agent_composer.runtime.engine")
assert "agent_composer.compile.expand" not in sys.modules, sorted(
    name for name in sys.modules if name.startswith("agent_composer.compile")
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(package_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class _SeedSpawner(Node):
    """A minimal dynamic node whose topology is supplied only by the test builder."""

    kind = NodeKind.CALL
    is_spawner = True

    def _run_impl(self, inputs):
        return Grow(seed={"value": "built"})


_CANONICAL_MUTABLE_SEED = {
    "value": "built",
    "nested": {"items": [{"label": "stable"}]},
}


class _MutableSeedSpawner(Node):
    """A spawner exposing mutable nested seed and node state to an event observer."""

    kind = NodeKind.CALL
    is_spawner = True

    def __init__(self, node_id):
        super().__init__(node_id)
        self.env["builder_marker"] = "stable"

    def _run_impl(self, inputs):
        return Grow(seed=copy.deepcopy(_CANONICAL_MUTABLE_SEED))


def _dynamic_flow():
    return _graph(
        [_SeedSpawner("spawn")],
        [(START_ID, "spawn"), ("spawn", END_ID)],
    )


def _mutable_seed_flow():
    return _graph(
        [_MutableSeedSpawner("spawn")],
        [(START_ID, "spawn"), ("spawn", END_ID)],
    )


def _fragment(spawner, seed):
    leaf = FuncNode(f"{spawner.id}/leaf", lambda inputs: seed["value"])
    leaf.commit_as = spawner.id
    return Flow(
        nodes={leaf.id: leaf},
        edges=[],
        wiring={leaf.id: {}},
        start_id=leaf.id,
        end_id=leaf.id,
    )


@pytest.mark.parametrize("num_workers", [0, 3])
def test_dynamic_direct_engine_requires_injected_builder(num_workers):
    events = list(FlowEngine(_dynamic_flow(), num_workers=num_workers).run())

    assert isinstance(events[-1], RunFailed)
    assert "grow_builder" in events[-1].error

    calls = []

    def builder(spawner, seed):
        calls.append((spawner.id, seed))
        return _fragment(spawner, seed)

    engine = FlowEngine(
        _dynamic_flow(), num_workers=num_workers, grow_builder=builder
    )
    list(engine.run())
    assert calls == [("spawn", {"value": "built"})]
    assert "spawn/leaf" in engine.flow.nodes


@pytest.mark.parametrize("num_workers", [0, 3])
def test_builder_failure_precedes_topology_and_ledger_mutation(num_workers):
    flow = _dynamic_flow()
    original_nodes = set(flow.nodes)
    original_edges = list(flow.edges)
    original_wiring = {node_id: dict(wiring) for node_id, wiring in flow.wiring.items()}

    def builder(spawner, seed):
        raise RuntimeError("builder exploded")

    engine = FlowEngine(flow, num_workers=num_workers, grow_builder=builder)
    events = list(engine.run())

    assert isinstance(events[-1], RunFailed)
    assert events[-1].error == "builder exploded"
    assert set(flow.nodes) == original_nodes
    assert flow.edges == original_edges
    assert flow.wiring == original_wiring
    assert engine.expansions == []
    assert not any(isinstance(event, NodeExpanded) for event in events)


@pytest.mark.parametrize("num_workers", [0, 3])
def test_expansion_is_applied_before_mutable_event_payload_is_exposed(num_workers):
    """Observers cannot alter the builder input or durable seed at the expansion event."""
    builder_inputs = []

    def builder(spawner, seed):
        builder_inputs.append((spawner.env["builder_marker"], copy.deepcopy(seed)))
        return _fragment(spawner, seed)

    engine = FlowEngine(
        _mutable_seed_flow(), num_workers=num_workers, grow_builder=builder
    )
    events = []
    for event in engine.run():
        events.append(event)
        if isinstance(event, NodeExpanded):
            assert "spawn/leaf" in engine.flow.nodes
            assert engine.expansions[0].seed == _CANONICAL_MUTABLE_SEED
            assert builder_inputs == [("stable", _CANONICAL_MUTABLE_SEED)]
            event.grow.seed["nested"]["items"][0]["label"] = "observer-mutated"
            event.grow.seed["nested"]["items"].append({"label": "observer-added"})
            engine.flow.nodes["spawn"].env["builder_marker"] = "observer-mutated"

    assert any(isinstance(event, NodeExpanded) for event in events)
    assert builder_inputs == [("stable", _CANONICAL_MUTABLE_SEED)]
    assert engine.expansions[0].seed == _CANONICAL_MUTABLE_SEED


@pytest.mark.parametrize("num_workers", [0, 3])
def test_builder_seed_mutation_does_not_change_event_or_replay_seed(num_workers):
    """The builder gets a disposable seed copy; replay receives the stable ledger value."""
    builder_inputs = []

    def mutating_builder(spawner, seed):
        builder_inputs.append(copy.deepcopy(seed))
        seed["nested"]["items"][0]["label"] = "builder-mutated"
        seed["nested"]["items"].append({"label": "builder-added"})
        return _fragment(spawner, seed)

    engine = FlowEngine(
        _mutable_seed_flow(), num_workers=num_workers, grow_builder=mutating_builder
    )
    events = list(engine.run())
    expanded = next(event for event in events if isinstance(event, NodeExpanded))
    checkpoint = engine.snapshot()

    restored = FlowEngine.restore(
        _mutable_seed_flow(), checkpoint, grow_builder=mutating_builder
    )

    assert builder_inputs == [_CANONICAL_MUTABLE_SEED, _CANONICAL_MUTABLE_SEED]
    assert expanded.grow.seed == _CANONICAL_MUTABLE_SEED
    assert engine.expansions[0].seed == _CANONICAL_MUTABLE_SEED
    assert restored.expansions[0].seed == _CANONICAL_MUTABLE_SEED
    assert set(restored.flow.nodes) == set(engine.flow.nodes)


def test_checkpoint_replay_uses_injected_builder():
    clean = _dynamic_flow()
    checkpoint = FlowEngine(clean).snapshot().model_copy(
        update={
            "expansions": [
                GrowRecord(
                    spawner_id="spawn",
                    seed={"value": "from-checkpoint"},
                    children=[],
                )
            ]
        }
    )
    calls = []

    def builder(spawner, seed):
        calls.append((spawner.id, seed))
        return _fragment(spawner, seed)

    restored = FlowEngine.restore(
        _dynamic_flow(), checkpoint, grow_builder=builder
    )

    assert calls == [("spawn", {"value": "from-checkpoint"})]
    assert "spawn/leaf" in restored.flow.nodes


def test_checkpoint_builder_failure_precedes_topology_mutation():
    checkpoint = FlowEngine(_dynamic_flow()).snapshot().model_copy(
        update={
            "expansions": [
                GrowRecord(
                    spawner_id="spawn",
                    seed={"value": "from-checkpoint"},
                    children=[],
                )
            ]
        }
    )
    flow = _dynamic_flow()
    original_nodes = set(flow.nodes)
    original_edges = list(flow.edges)
    original_wiring = {node_id: dict(wiring) for node_id, wiring in flow.wiring.items()}

    def builder(spawner, seed):
        raise RuntimeError("replay builder exploded")

    with pytest.raises(RuntimeError, match="replay builder exploded"):
        FlowEngine.restore(flow, checkpoint, grow_builder=builder)

    assert set(flow.nodes) == original_nodes
    assert flow.edges == original_edges
    assert flow.wiring == original_wiring
