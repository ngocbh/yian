"""Lightweight fake nodes for engine/runtime tests.

Real node kinds (TOOL/CODE/AGENT/...) are tested separately; these exercise the
contract and the runtime scheduling in isolation from LLMs/tools.
"""

from typing import Any, Callable

from agent_composer.events import StreamChunk
from agent_composer.nodes.base import Node, NodeKind, Output, Pause, Route
from agent_composer.nodes.binding import ParamDecl
from agent_composer.suspension.pause import HumanInputRequired


def stamp_reads(node, reads):
    """Declare a fake's inputs the way the flow does (the node no longer holds sources). `reads` is a
    `{param: source}` map -> stamp `node.params` + a fake-local `node._wiring_src` that
    `derive_wiring` collects into the flow's `wiring`. Returns the node (chainable)."""
    node.params = [ParamDecl(name=k) for k in (reads or {})]
    node._wiring_src = dict(reads or {})
    return node


def derive_wiring(nodes):
    """Test helper: collect each node's fake-local `_wiring_src` into a `{node_id: {param:
    source}}` flow-wiring map (the engine binds purely from `params` + `flow.wiring`). A node
    without declared reads contributes `{}`."""
    return {nid: getattr(node, "_wiring_src", {}) for nid, node in nodes.items()}


def drive(node, pool=None, flow=None):
    """Drive a node through the engine's `eval_node` seam (binds purely from `params` +
    `flow.wiring`) and return its event generator. With no `flow`, derive a stub `flow.wiring`
    from the node's declared `_wiring_src`."""
    from agent_composer.runtime.eval_node import eval_node
    from agent_composer.typesys.pool import VariablePool
    from types import SimpleNamespace

    if flow is None:
        flow = SimpleNamespace(wiring=derive_wiring({node.id: node}))
    return eval_node(node, flow, pool if pool is not None else VariablePool())


class FuncNode(Node):
    """A deterministic node: `fn(inputs) -> Any` becomes the node's single value
    (often a `{"output": ...}` object so downstream `${n.output.output}` refs work
    via object-walk; a bare scalar/list works too). `inputs` is the bound record;
    `reads={param: source}` declares the node's inputs (stamps params + wiring)."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, fn: Callable[[dict], Any], *, reads=None, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.fn = fn
        stamp_reads(self, reads)

    def _run_impl(self, inputs: dict) -> Output:
        return Output(value=self.fn(inputs))


class RecordNode(FuncNode):
    """Appends its id to a shared list when it runs (to assert execution order)."""

    def __init__(self, node_id: str, log: list, output: Any = "", **kw: Any) -> None:
        def fn(inputs: dict) -> dict:
            log.append(node_id)
            return {"output": output}

        super().__init__(node_id, fn, **kw)


class StreamNode(Node):
    """Yields chunks then returns a result (the AGENT streaming shape)."""

    kind = NodeKind.AGENT

    def __init__(self, node_id: str, chunks: list[str], **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.chunks = chunks

    def _run_impl(self, inputs: dict):
        text = ""
        for c in self.chunks:
            text += c
            yield StreamChunk(self.id, "output", c)
        return Output(value={"output": text})


class FailNode(Node):
    kind = NodeKind.CODE

    def __init__(self, node_id: str, message: str = "boom", **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.message = message

    def _run_impl(self, inputs: dict) -> Output:
        raise RuntimeError(self.message)


class Policy(Exception):
    """Named so the boundary yields NodeFailed(error_type='Policy')."""


class ReturnsFailedNode(Node):
    kind = NodeKind.CODE

    def _run_impl(self, inputs: dict) -> Output:
        raise Policy("declined")


class BranchNode(Node):
    """CASE: returns a chosen handle (which downstream edge to take)."""

    kind = NodeKind.CASE

    def __init__(self, node_id: str, handle: str, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.handle = handle

    def _run_impl(self, inputs: dict) -> Route:
        return Route(self.handle)


class PauseNode(Node):
    """A generic test-only Pause leaf for dormant checkpoint compatibility.

    Its borrowed kind tag is incidental; authorable HUMAN_INPUT blocks instead.
    The engine delivers a compatibility value and the fake never re-runs.
    """

    kind = NodeKind.HUMAN_INPUT

    def __init__(self, node_id: str, *, reason: Any = "needs-input", **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.reason = reason

    def _run_impl(self, inputs: dict) -> Pause:
        return Pause(self.reason)


class ArtificialHumanPauseNode(PauseNode):
    """A test-only HUMAN_INPUT-shaped leaf for dormant checkpoint coverage.

    Authorable HUMAN_INPUT blocks in-process now. Tests that specifically exercise
    the temporary generic ``Pause`` compatibility API use this artificial leaf so
    they do not teach the old authoring behavior. The reason is built at run time so
    a cloned leaf reports its fully namespaced id.
    """

    def __init__(self, node_id: str, *, prompt: str = "", **kw: Any) -> None:
        super().__init__(node_id, reason=None, **kw)
        self.prompt = prompt

    def _run_impl(self, inputs: dict) -> Pause:
        return Pause(HumanInputRequired(
            prompt=self.prompt,
            node_title=self.title,
            node_id=self.id,
        ))


def replace_human_inputs_with_artificial_pauses(flow):
    """Replace authored HUMAN_INPUT leaves recursively for checkpoint-only tests.

    The loaded topology and all render/type metadata stay intact; only execution of
    each HUMAN_INPUT leaf is swapped for :class:`ArtificialHumanPauseNode`. Static
    child flows are visited before runtime growth clones them.
    """
    from agent_composer.nodes.human_input import HumanInputNode

    for node_id, node in list(flow.nodes.items()):
        child = getattr(node, "child", None)
        if child is not None:
            replace_human_inputs_with_artificial_pauses(child)
        if not isinstance(node, HumanInputNode):
            continue

        replacement = ArtificialHumanPauseNode(
            node.id,
            prompt=node.prompt or "",
            title=node.title,
        )
        replacement.__dict__.update(node.__dict__)
        replacement.prompt = node.prompt or ""
        flow.nodes[node_id] = replacement
    return flow


# Legacy alias: the re-run "pause once then complete" intent is gone (deliver-as-Output);
# the alias keeps the pause-only call sites (which never injected) unchanged.
PauseOnceNode = PauseNode


# --------------------------------------------------------------------------- #
# Golden-test helpers: simple module-level functions a CODE node can call
# via `code: tests.engine._fakes:<fn>`. Used by tests/engine/test_surface_vocab_golden.py.
#
# CodeNode.run calls `func(inputs)` with the bound input dict (not splatted), so
# helpers take one dict arg.
# --------------------------------------------------------------------------- #


def passthrough(inputs):
    """Return the single input value `v` unchanged. The simplest possible CODE body."""
    return inputs["v"]


def tagged_pos(inputs):
    return {"tag": "pos", "val": inputs["v"]}


def tagged_neg(inputs):
    return {"tag": "neg", "val": inputs["v"]}
