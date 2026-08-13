"""The `eval_node` seam — the engine's read/dispatch boundary.

`eval_node(node, flow, pool)` binds a node's inputs, runs it, and normalizes the returned
`NodeResult` into the engine event stream: `NodeStarted` first, then exactly one terminal
(`NodeSucceeded`/`NodeFailed`), or a single `PauseRequested` for a returned `Pause`. A node
`raise` becomes `NodeFailed` (no Failure variant). The bind reads sources from `flow.wiring`;
these direct-drive tests supply them via the `_fakes` helpers (`stamp_reads`/`drive`). The MAP
`bind_item` path is covered end-to-end by `test_map.py`; the AGENT cap path (incl.
AgentLoopError -> NodeFailed) by `test_agent.py`.
"""

from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Edge
from agent_composer.events import NodeFailed, NodeRouted, NodeStarted, NodeSucceeded, PauseRequested, RunFailed
from agent_composer.nodes.base import Grow, Node, NodeKind, Output
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.map import MapNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import BranchNode, FailNode, FuncNode, PauseOnceNode, drive, stamp_reads
from tests.engine._graph_builder import _graph
from tests.engine.test_golden_baseline import MAP_OVER_NOT_LIST_FMT


def _drive(node, pool=None):
    # drive derives a stub flow.wiring + stamps params from the node's transitional inputs
    # (eval_node binds purely from flow.wiring).
    return list(drive(node, pool))


def test_yields_started_then_succeeded():
    evs = _drive(FuncNode("n", lambda i: {"output": "ok"}))
    assert isinstance(evs[0], NodeStarted)
    assert isinstance(evs[-1], NodeSucceeded)
    assert evs[-1].output == {"output": "ok"}


class _CommitAsNode(Node):
    """Returns an `Output` (optionally carrying a node-chosen `commit_as`), to exercise
    `eval_node` folding `result.commit_as or node.commit_as` onto `NodeSucceeded`."""

    kind = NodeKind.CODE

    def __init__(self, node_id, out):
        super().__init__(node_id)
        self._out = out

    def _run_impl(self, inputs, **caps):
        return self._out


def test_baked_commit_as_rides_onto_succeeded():
    # `eval_node` folds the engine-baked `node.commit_as` onto the terminal NodeSucceeded so
    # `_on_success` can commit the value under the spawner id (the alias-map replacement).
    node = _CommitAsNode("n", Output("v"))
    node.commit_as = "s"
    evs = _drive(node)
    assert isinstance(evs[-1], NodeSucceeded)
    assert evs[-1].output == "v"
    assert evs[-1].commit_as == "s"


def test_node_chosen_commit_as_wins_over_baked():
    # A node-CHOSEN `Output(commit_as=...)` overrides the engine-baked `node.commit_as`
    # (the roadmap seam: a node names its own commit target).
    node = _CommitAsNode("n", Output("v", commit_as="c"))
    node.commit_as = "s"
    evs = _drive(node)
    assert isinstance(evs[-1], NodeSucceeded)
    assert evs[-1].commit_as == "c"


def test_binding_failure_is_started_then_failed():
    # A required input with NO wiring edge (omitted) -> BindingError inside the bind seam ->
    # NodeFailed. (Param `required` is presence-based; a present-but-null source is the `:?` grammar.)
    node = FuncNode("n", lambda i: i)
    node.params = [ParamDecl(name="x", required=True)]
    node._wiring_src = {}  # x has no edge -> omitted
    evs = _drive(node)
    assert isinstance(evs[0], NodeStarted)
    assert isinstance(evs[-1], NodeFailed)
    assert evs[-1].error_type == "BindingError"


def test_node_raise_becomes_failed():
    evs = _drive(FailNode("n", "boom"))
    assert isinstance(evs[-1], NodeFailed)
    assert evs[-1].error == "boom"
    assert evs[-1].error_type == "RuntimeError"


def test_returned_pause_emits_pause_requested():
    evs = _drive(PauseOnceNode("n"))  # always pauses on its single run
    assert isinstance(evs[0], NodeStarted)
    assert isinstance(evs[-1], PauseRequested)
    assert evs[-1].reason == "needs-input"
    assert not any(isinstance(e, (NodeSucceeded, NodeFailed)) for e in evs)


def test_routing_emits_node_routed():
    evs = _drive(BranchNode("n", "case_a"))
    assert isinstance(evs[-1], NodeRouted)
    assert evs[-1].handle == "case_a"
    assert not hasattr(evs[-1], "output")  # routing-only: NodeRouted carries no value


# --- review lock-ins: every node-side failure path funnels to NodeFailed uniformly --------- #


class _NonSpawnerGrowNode(Node):
    """A non-spawner (is_spawner=False by default) whose run returns a Grow — a clear error:
    only spawner kinds may grow the graph."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        return Grow(seed={})


class _BadReturnNode(Node):
    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        return {"not": "a NodeResult"}  # neither a NodeResult nor a generator


class _MutatingNode(Node):
    """A CODE-style leaf that mutates the dict it is handed (a legal in-place transform)."""

    kind = NodeKind.CODE

    def __init__(self, node_id):
        super().__init__(node_id)
        stamp_reads(self, {"x": "${input.x}"})

    def _run_impl(self, inputs):
        inputs["x"] = 999  # must NOT leak into this node's own post-assert
        return Output(value="done")


def test_grow_from_non_spawner_kind_is_node_failed():
    # A non-spawner (CODE) returning Grow is a clear error, not a silent NodeExpanded.
    evs = _drive(_NonSpawnerGrowNode("n"))               # is_spawner is False (default)
    assert isinstance(evs[-1], NodeFailed)
    assert "Grow" in evs[-1].error and "spawner" in evs[-1].error
    assert evs[-1].error_type == "RuntimeError"


def test_non_spawner_grow_fails_run_in_both_engines():
    # The seam must normalize identically on the serial and pooled engines — no uncaught
    # escape on serial. A non-spawner kind returning Grow fails the run on both.
    def graph():
        return _graph([_NonSpawnerGrowNode("n")], [(START_ID, "n"), ("n", END_ID)])

    serial = list(FlowEngine(graph()).run())
    assert isinstance(serial[-1], RunFailed) and "Grow" in serial[-1].error

    par = list(FlowEngine(graph(), num_workers=4).run())   # ParallelFlowEngine retired
    assert isinstance(par[-1], RunFailed) and "Grow" in par[-1].error


def test_bad_return_type_is_clear_node_failed():
    evs = _drive(_BadReturnNode("n"))
    assert isinstance(evs[-1], NodeFailed)
    assert "not a NodeResult" in evs[-1].error
    assert evs[-1].error_type == "RuntimeError"


def test_post_assert_isolated_from_node_input_mutation():
    pool = VariablePool()
    pool.set(START_ID, {"x": 1})
    node = _MutatingNode("n")
    node.post_asserts = ["${x} == 1"]  # reads the input; the node sets it to 999 in place
    evs = _drive(node, pool)  # derives flow.wiring {"x": "${input.x}"} + params from inputs
    assert isinstance(evs[-1], NodeSucceeded)  # post-assert saw the pristine 1, not 999


def test_map_over_not_a_list_message_byte_identical():
    # The over-resolution + not-a-list guard live in eval_node now; the message must stay
    # byte-identical to the golden constant. (child=None is never reached — over fails first.)
    pool = VariablePool()
    pool.set(START_ID, {"bad": "notalist"})
    node = MapNode("m", flow_id="f")
    node._wiring_src = {"over": "${input.bad}"}  # the over source lives on flow.wiring
    evs = _drive(node, pool)
    assert isinstance(evs[-1], NodeFailed)
    assert evs[-1].error == MAP_OVER_NOT_LIST_FMT.format(id="m", src="${input.bad}")
