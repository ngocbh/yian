"""The runtime receives growth construction as an injected plain callable."""

import subprocess
import sys
from pathlib import Path

import pytest

from agent_composer.compile.model import END_ID, START_ID, Flow
from agent_composer.events import RunFailed
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


def _dynamic_flow():
    return _graph(
        [_SeedSpawner("spawn")],
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
