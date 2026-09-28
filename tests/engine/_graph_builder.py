"""Manual-graph helper: build a CompiledFlow from (from, to[, handle]) edge tuples,
injecting real StartNode/EndNode objects so the ~12 direct-construction test files keep working
once the __start__/__end__ sentinel guards retire (the boundary becomes ordinary NODES).

Mirrors the existing test_engine.py manual-graph convention (Edge per tuple + derive_wiring);
the additions are the StartNode/EndNode node objects under START_ID/END_ID. When the caller
declares `outputs`, the END_ID node is built RECORD-mode (one param per output) and its producer
edges are minted from the output bindings (the loader's emit_for shape) so the engine reads the
declared-output value as `store[END_ID]` — the manual `(x, END_ID)` tuples are then dropped
(superseded by the minted producer edges). With no `outputs`, END_ID is output-less (returns None)
and the manual `(x, END_ID)` tuples stand as bare adjacencies.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Edge, FlowOutput
from agent_composer.compile.expand import build_grow
from agent_composer.compile.execution_layout import apply_execution_layout, resolve_execution_layout
from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.compose.edges.producers import _binding_producers
from agent_composer.expr import binding_co_skips, expr_refs_of
from agent_composer.memory import MemoryFrameSpec, MemorySnapshotEnvelope
from agent_composer.nodes.contracts import Parameter
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.log import MemoryRunLog
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import derive_wiring  # the SAME auto-derive test_engine.py:24 imports


_TEST_RUN_ID = "direct-engine-test"
_TEST_FINGERPRINT = "f" * 64


def _forward_capture(node, inputs):
    """Give directly prepared test outcomes explicit canonical binding data."""
    from agent_composer.expr.lineage import ValueRoute
    from agent_composer.runtime.contracts import ForwardCapture

    return ForwardCapture(inputs, ValueRoute.record(
        (name, ValueRoute.literal(value)) for name, value in (inputs or {}).items()
    ), scope=node.execution_scope)


def _engine(
    flow: CompiledFlow,
    run_inputs: Optional[dict[str, Any]] = None,
    *,
    num_workers: int = 0,
    grow_builder=build_grow,
    on_event=None,
    on_stream=None,
    stop_event=None,
    memory_layout: tuple[MemoryFrameSpec, ...] | None = None,
    memory_snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None,
    learnable: bool = False,
    memory_context_opener=None,
) -> FlowEngine:
    """Prepare one in-memory authority and return the final direct engine surface."""
    inputs = dict(run_inputs or {})
    effective_llm = {}
    if learnable:
        effective_llm = resolve_llm_cascade(flow, {}, defaults={"provider": "test", "model": "test"})
        apply_execution_layout(flow, resolve_execution_layout(flow))
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, inputs)
    pool.add_system("run_id", _TEST_RUN_ID)
    log = MemoryRunLog(_TEST_RUN_ID)
    prepared = prepare_run_started(
        state,
        RunStartData(
            run_id=_TEST_RUN_ID,
            flow_fingerprint=_TEST_FINGERPRINT,
            initial_pool=pool,
            start_wiring_override={
                name: f"${{input.{name}}}" for name in inputs
            },
            effective_llm_by_static_path=effective_llm,
            num_workers=num_workers,
            memory_layout=memory_layout,
            memory_snapshots=memory_snapshots,
            learnable=learnable,
        ),
    )
    commit_prepared(log, state, prepared)
    return FlowEngine(
        state,
        log,
        grow_builder=grow_builder,
        num_workers=num_workers,
        on_event=on_event,
        on_stream=on_stream,
        stop_event=stop_event,
        memory_context_opener=memory_context_opener,
    )


def _synth_start_parameters(node_map: dict, wiring: Optional[dict]) -> list[Parameter]:
    """Collect the `${input.X}` names a manual graph reads as START parameters.

    Scans every node's wiring source (the passed `wiring`, else the auto-derived one) for
    `input.<name>` reference paths so the scheduled START can `bind_params` the run args
    against matching untyped params. Returns parameters in first-seen order.
    """
    sources = wiring if wiring is not None else derive_wiring(node_map)
    seen: dict[str, None] = {}
    for node_wiring in sources.values():
        if not isinstance(node_wiring, dict):
            continue
        for src in node_wiring.values():
            if not isinstance(src, str):
                continue
            for ref in expr_refs_of(src):
                if ref == "input" or ref.startswith("input."):
                    name = ref[len("input."):] if ref.startswith("input.") else ""
                    name = name.split(".")[0]
                    if name and name not in seen:
                        seen[name] = None
    return [Parameter(name=name) for name in seen]


def _graph(nodes: list, raw_edges: list[tuple], *,
           outputs: Optional[list[FlowOutput]] = None,
           wiring: Optional[dict[str, dict[str, Any]]] = None) -> CompiledFlow:
    # Signature mirrors test_engine.py's local `_graph(nodes, raw_edges, outputs=None)` EXACTLY
    # (nodes as a LIST; node_map computed internally) so migrating the ~12 files is a true
    # drop-in. derive_wiring is DEFAULTED here — `wiring is None`
    # -> `derive_wiring(node_map)`, the same auto-derive test_engine.py passes today.
    node_map = {n.id: n for n in nodes}
    # Synthesize START's parameters from the `${input.X}` refs the manual graph reads, so the
    # scheduled START can bind_params the run args (an empty-param START would drop every run arg —
    # its bind reads params, and no params means no binding). The caller may still pass its own
    # StartNode in `nodes` (respected by setdefault).
    if START_ID not in node_map:
        node_map[START_ID] = StartNode(
            START_ID, params=_synth_start_parameters(node_map, wiring))
    out_list = list(outputs or [])
    node_map.setdefault(END_ID, EndNode.record(END_ID, output_names=[o.name for o in out_list]))

    if out_list:
        # END_ID is record-mode: drop the manual `(x, END_ID)` terminal tuples and mint the producer
        # edges from the output bindings (input_group = output name; optional = not co-skips) so
        # the engine reads the declared value at store[END_ID] and the terminal disposition fires via END_ID.
        # Mint an edge only for a producer that EXISTS as a node (a missing ref binds leniently to
        # None — the loader's terminal-resolve semantics; real flows have all producers present).
        raw_edges = [t for t in raw_edges if t[1] != END_ID]
        end_edges: list[Edge] = []
        counts: dict[tuple[str, str], int] = {}
        for o in out_list:
            optional = not binding_co_skips(o.from_)
            for producer in _binding_producers(o.from_):
                if producer not in node_map:
                    continue
                i = counts.get((producer, END_ID), 0)
                counts[(producer, END_ID)] = i + 1
                end_edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                                      input_group=o.name, optional=optional))
    else:
        end_edges = []

    edges = [Edge(f"e{i}", t[0], t[1], t[2] if len(t) > 2 else None)
             for i, t in enumerate(raw_edges)]
    edges += end_edges
    # A 0-producer-edge END_ID (literal/default-only output, or output-less) is a root so it runs.
    if not any(e.to == END_ID for e in edges):
        edges.append(Edge(id=f"{START_ID}->{END_ID}", from_=START_ID, to=END_ID))
    if wiring is None:
        wiring = derive_wiring(node_map)
    # END_ID binds each declared output from its `from_` (so EndNode.run returns the declared value).
    if out_list:
        wiring = {**wiring, END_ID: {o.name: o.from_ for o in out_list}}
    return CompiledFlow.from_parts(node_map, edges, outputs=outputs, wiring=wiring)
