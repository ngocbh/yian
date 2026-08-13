"""Unit tests for durable suspend/resume.

The headline cycle: a HUMAN_INPUT node suspends; the run is serialized to JSON
(simulating cross-process persistence), reloaded, the answer is injected via a
command, and a FRESH engine resumes the run to completion.
"""

import json

import pytest

from agent_composer.events import RunPaused, RunResumed, RunSucceeded
from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow, NodeState, FlowOutput
from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.nodes.wait import WaitNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.typesys.pool import VariablePool
from agent_composer.suspension.checkpoint import RunCheckpoint
from agent_composer.suspension.commands import DeliverAnswerCommand
from agent_composer.suspension.pause import HumanInputRequired, EventAwaited
from tests.engine._fakes import FuncNode, derive_wiring, drive, stamp_reads
from tests.engine._graph_builder import _graph


class EchoNode(Node):
    """Reads an upstream node's whole value (via a declared input) and re-emits it
    (a stand-in 'after' step). `${<source>.output}` resolves to the same whole value
    `pool.get(<source>)` returned before the node lost pool access."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, source: str) -> None:
        super().__init__(node_id)
        stamp_reads(self, {"value": f"${{{source}.output}}"})

    def _run_impl(self, inputs: dict) -> Output:
        return Output(value=inputs["value"])


def _ask_flow():
    # start -> ask(HUMAN_INPUT) -> after(echo ask's value) -> END_ID
    # The flow's single declared output is the echoed answer.
    return _graph(
        [
            FuncNode("start", lambda p: {"output": "go"}),
            HumanInputNode("ask", prompt="Approve the action?"),
            EchoNode("after", source="ask"),
        ],
        [(START_ID, "start"), ("start", "ask"), ("ask", "after"), ("after", END_ID)],
        outputs=[FlowOutput(name="result", from_="${after.output}")],
    )


def _two_pause_flow():
    # start -> ask1(HUMAN_INPUT) -> ask2(HUMAN_INPUT) -> after(echo ask2) -> END_ID
    # Two suspension points in a linear chain; the terminal echoes ask2's answer.
    return _graph(
        [
            FuncNode("start", lambda p: {"output": "go"}),
            HumanInputNode("ask1", prompt="First?"),
            HumanInputNode("ask2", prompt="Second?"),
            EchoNode("after", source="ask2"),
        ],
        [(START_ID, "start"), ("start", "ask1"), ("ask1", "ask2"), ("ask2", "after"), ("after", END_ID)],
        outputs=[FlowOutput(name="result", from_="${after.output}")],
    )


def test_live_engine_resume_delivers_answer_as_output():
    """A live engine paused via run() resumes by DELIVERING the answer as the parked
    leaf's Output: RunResumed lead, the value flows downstream, no re-run."""
    g = _ask_flow()
    engine = FlowEngine(g)                                       # serial (num_workers=0)
    assert isinstance(list(engine.run())[-1], RunPaused)         # parked at "ask"
    evs2 = list(engine.resume(commands=[DeliverAnswerCommand(node_id="ask", value="approve")]))
    assert isinstance(evs2[0], RunResumed)                       # lead, not RunStarted
    assert isinstance(evs2[-1], RunSucceeded)
    assert evs2[-1].output == "approve"                          # ask's Output flowed through "after"


def test_live_engine_multi_pause_resume_delivers_each():
    g = _two_pause_flow()
    engine = FlowEngine(g)
    assert isinstance(list(engine.run())[-1], RunPaused)
    r1 = list(engine.resume(commands=[DeliverAnswerCommand(node_id="ask1", value="a")]))
    assert isinstance(r1[-1], RunPaused)                          # parked at ask2 (no re-run of ask1)
    r2 = list(engine.resume(commands=[DeliverAnswerCommand(node_id="ask2", value="b")]))
    assert isinstance(r2[-1], RunSucceeded)
    assert r2[-1].output == "b"


def test_paused_leaf_is_not_reset_to_unknown():
    g = _ask_flow()
    engine = FlowEngine(g)
    list(engine.run())
    # a pause PARKS the node WITHOUT a UNKNOWN reset (no re-run on resume).
    assert engine.sm.node_state["ask"] != NodeState.UNKNOWN


def _fork_two_pause_join():
    # start -> {ask1, ask2}(HUMAN_INPUT) -> join(records once) -> END_ID
    log = []

    def _join(i):
        log.append("j")
        return {"output": [i["a"], i["b"]]}

    join = FuncNode("join", _join)
    stamp_reads(join, {"a": "${ask1.output}", "b": "${ask2.output}"})
    g = _graph(
        [FuncNode("start", lambda p: {"output": "go"}),
         HumanInputNode("ask1", prompt="1?"), HumanInputNode("ask2", prompt="2?"), join],
        [(START_ID, "start"), ("start", "ask1"), ("start", "ask2"),
         ("ask1", "join"), ("ask2", "join"), ("join", END_ID)],
        outputs=[FlowOutput(name="r", from_="${join.output.output}")],
    )
    return g, log


def test_multi_command_single_resume_no_drop_no_double_run():
    g, log = _fork_two_pause_join()
    engine = FlowEngine(g)
    list(engine.run())                                           # parks at BOTH ask1 and ask2
    evs = list(engine.resume(commands=[
        DeliverAnswerCommand(node_id="ask1", value="a"),
        DeliverAnswerCommand(node_id="ask2", value="b"),
    ]))
    assert isinstance(evs[-1], RunSucceeded)
    assert evs[-1].output == ["a", "b"]
    assert log == ["j"]                                          # join ran EXACTLY once


def test_checkpoint_json_round_trip():
    g = _ask_flow()
    engine = FlowEngine(g)
    events = list(engine.run())
    assert isinstance(events[-1], RunPaused)

    ckpt = engine.snapshot()
    back = RunCheckpoint.loads(ckpt.dumps())

    assert back.paused_nodes == ["ask"]
    assert isinstance(back.pause_reasons[0], HumanInputRequired)
    assert back.pause_reasons[0].prompt == "Approve the action?"
    # node/edge state and the pool survived the round-trip. The parked node is now TAKEN
    # (_on_pause no longer resets to UNKNOWN; deliver-as-Output, no re-run).
    assert back.node_state["ask"] == NodeState.TAKEN
    assert back.pool.resolve("start", ["output", "output"]) == "go"


def test_checkpoint_v1_blob_rejected_by_loads():
    # breaking blob migration: a 1.0 checkpoint (the old node_id->key->TypedValue store)
    # is not loadable. loads() reads the raw-JSON version BEFORE model_validate, so the
    # error is the clear "incompatible version", not an opaque pydantic failure.
    blob = RunCheckpoint(pool=VariablePool()).dumps()
    tampered = json.dumps({**json.loads(blob), "version": "1.0"})
    with pytest.raises(ValueError, match="incompatible checkpoint version"):
        RunCheckpoint.loads(tampered)


def test_checkpoint_v1_object_rejected_by_restore():
    # Defense-in-depth: restore() ALSO gates — a RunCheckpoint object can reach it
    # without passing through loads().
    g = _ask_flow()
    ckpt = RunCheckpoint(pool=VariablePool()).model_copy(update={"version": "1.0"})
    with pytest.raises(ValueError, match="incompatible checkpoint version"):
        FlowEngine.restore(g, ckpt)


def test_checkpoint_current_version_round_trip_carries_store():
    # The single-value store survives dumps/loads transitively through the pool, at the "8.0"
    # checkpoint version (bumped when MAP grew a synthetic map#/__start__ fan-out node).
    # The version label here tracks the CURRENT default.
    pool = VariablePool()
    pool.set("n", "v")
    back = RunCheckpoint.loads(RunCheckpoint(pool=pool).dumps())
    assert back.version == "8.0"
    assert back.pool.get("n") == "v"


def test_checkpoint_v2_blob_rejected_by_loads():
    # A pre-4.0 (e.g. 2.0) blob is NOT loadable after the type-surface rename.
    blob = RunCheckpoint(pool=VariablePool()).dumps()
    tampered = json.dumps({**json.loads(blob), "version": "2.0"})
    with pytest.raises(ValueError, match="incompatible checkpoint version"):
        RunCheckpoint.loads(tampered)


def test_pause_persist_restore_inject_resume():
    g = _ask_flow()

    # --- process 1: run until it suspends, then persist ---
    e1 = FlowEngine(g)
    evs1 = list(e1.run())
    assert isinstance(evs1[-1], RunPaused)
    blob = e1.snapshot().dumps()

    # --- process 2: reload checkpoint, rebuild engine on the same topology ---
    ckpt = RunCheckpoint.loads(blob)
    e2 = FlowEngine.restore(g, ckpt)

    # the human (via an external command channel) supplies the answer
    inject = DeliverAnswerCommand(node_id="ask", value="approve")
    evs2 = list(e2.resume(commands=[inject]))

    assert isinstance(evs2[-1], RunSucceeded)
    # the answer flowed through ask -> after to the flow's single declared output
    assert evs2[-1].output == "approve"


def test_resume_without_answer_pauses_again():
    # DURABLE variant: restore() re-seeds self.paused, so the commandless
    # short-circuit fires on the restored engine — a no-injected-answer resume suspends again,
    # not a crash. The LIVE-engine equivalent is test_live_engine_commandless_resume_re_emits_paused.
    g = _ask_flow()
    e1 = FlowEngine(g)
    list(e1.run())
    ckpt = RunCheckpoint.loads(e1.snapshot().dumps())
    e2 = FlowEngine.restore(g, ckpt)
    # resume with no injected answer -> the gate suspends again, not a crash
    evs = list(e2.resume(commands=[]))
    assert isinstance(evs[-1], RunPaused)


def test_live_engine_commandless_resume_re_emits_paused():
    """A LIVE paused engine — resume(commands=[]) AND resume() both re-emit
    RunPaused WITHOUT clearing self.paused. A no-op poll / watcher tick / partial
    delivery must not destroy the pause or fall through to a state-destroying terminal."""
    g = _ask_flow()
    engine = FlowEngine(g)
    assert isinstance(list(engine.run())[-1], RunPaused)
    assert engine.paused

    # commands=[] short-circuits: exactly [RunResumed, RunPaused], paused preserved
    evs2 = list(engine.resume(commands=[]))
    assert isinstance(evs2[0], RunResumed)
    assert isinstance(evs2[-1], RunPaused)
    assert len(evs2) == 2
    assert engine.paused

    # commands=None (no kwarg) ALSO short-circuits — guards the `commands or []` path
    evs3 = list(engine.resume())
    assert isinstance(evs3[-1], RunPaused)
    assert engine.paused

    # the real answer still resolves afterward (the pause was never destroyed)
    evs4 = list(engine.resume(commands=[DeliverAnswerCommand(node_id="ask", value="approve")]))
    assert isinstance(evs4[-1], RunSucceeded)
    assert evs4[-1].output == "approve"


def test_snapshot_is_point_in_time_not_aliased_to_live_pool():
    """snapshot() captures the pool by value — advancing the LIVE engine after the
    snapshot must not retro-mutate the held checkpoint."""
    g = _two_pause_flow()
    engine = FlowEngine(g)
    assert isinstance(list(engine.run())[-1], RunPaused)            # parked at ask1
    held = engine.snapshot()
    keys_at_pause1 = set(held.pool.store.keys())
    assert "ask1" not in keys_at_pause1                             # ask1 parked, not committed

    # advance the LIVE engine past pause-1 -> commits store["ask1"], parks at ask2
    evs = list(engine.resume(commands=[DeliverAnswerCommand(node_id="ask1", value="yes")]))
    assert isinstance(evs[-1], RunPaused)
    assert "ask1" in engine.pool.store                             # live pool advanced

    # the held checkpoint is unchanged (would carry "ask1" if it aliased the live pool)
    assert set(held.pool.store.keys()) == keys_at_pause1
    assert "ask1" not in held.pool.store


def test_snapshot_deep_copies_expansion_descriptors():
    """the GrowRecord ledger is deep-copied into the checkpoint, so a
    later live append (e.g. a multi-pause AGENT chaining a segment under `children`) does not
    mutate a held checkpoint."""
    from agent_composer.suspension.expansions import GrowRecord

    engine = FlowEngine(_ask_flow())
    list(engine.run())                                             # park so snapshot() is valid
    desc = GrowRecord(spawner_id="agent", seed={"hi_desc": {}, "resume_desc": {}}, children=[])
    engine.expansions = [desc]
    held = engine.snapshot()
    assert len(held.expansions[0].children) == 0

    # a live second-pause append (nesting chain) must not touch the held checkpoint
    desc.children.append(GrowRecord(spawner_id="agent/__resume#0", seed={}, children=[]))
    assert len(engine.expansions[0].children) == 1
    assert len(held.expansions[0].children) == 0                   # deep-copied, unchanged


def test_wait_node_suspends_with_market_event():
    g = _graph(
        [FuncNode("start", lambda p: {}), WaitNode("watch", event_spec={"kind": "alert"})],
        [(START_ID, "watch"), ("watch", END_ID)],
        outputs=[FlowOutput(name="event", from_="${watch.output}")],
    )
    engine = FlowEngine(g)
    evs = list(engine.run())
    assert isinstance(evs[-1], RunPaused)
    assert isinstance(evs[-1].reasons[0], EventAwaited)
    assert evs[-1].reasons[0].event_spec == {"kind": "alert"}
    assert evs[-1].reasons[0].node_id == "watch"                 # self-addressing for resume
    # the watcher fires: deliver the payload as the WAIT node's Output
    evs2 = list(engine.resume(commands=[DeliverAnswerCommand(node_id="watch", value={"eps": 1.2})]))
    assert isinstance(evs2[-1], RunSucceeded)
    assert evs2[-1].output == {"eps": 1.2}                       # node's Output binds to ${watch.output}


def test_event_awaited_type_literal_round_trip():
    from agent_composer.suspension.pause import EventAwaited
    r = EventAwaited(event_spec={"kind": "x"})
    assert r.type == "event_awaited"
    import json
    assert json.loads(r.model_dump_json())["type"] == "event_awaited"


def test_wait_timed_until_pauses():
    from agent_composer.nodes.wait import WaitNode
    from agent_composer.typesys.pool import VariablePool
    from agent_composer.events import PauseRequested
    from agent_composer.suspension.pause import ScheduledPause
    pool = VariablePool()
    pool.set(START_ID, {"settle_at": "2026-07-01"})     # a date input (stored as ISO string)
    node = WaitNode("settle", is_timed=True)
    node._wiring_src = {"until": "${input.settle_at}"}  # the until source lives on flow.wiring
    # Deliver-as-Output: the node ALWAYS pauses on its single run; the engine delivers.
    pause = [e for e in drive(node, pool) if isinstance(e, PauseRequested)][0]
    assert isinstance(pause.reason, ScheduledPause)
    assert pause.reason.resume_at.startswith("2026-07-01")
    assert pause.reason.node_id == "settle"            # self-addressing for resume_command


def test_human_input_renders_prompt_from_inputs():
    from agent_composer.nodes.human_input import HumanInputNode
    from agent_composer.typesys.pool import VariablePool
    from agent_composer.events import PauseRequested
    pool = VariablePool()
    pool.set("propose", "order ACME")            # producer value
    node = HumanInputNode("approve", prompt="Approve? ${action}")
    stamp_reads(node, {"action": "${propose.output}"})
    evs = list(drive(node, pool))
    pause = [e for e in evs if isinstance(e, PauseRequested)][0]
    assert pause.reason.prompt == "Approve? order ACME"


def test_human_input_run_takes_no_scratch_cap():
    import inspect
    from agent_composer.nodes.human_input import HumanInputNode
    sig = inspect.signature(HumanInputNode.run)
    assert "scratch" not in sig.parameters     # HUMAN_INPUT.run takes no *, scratch cap


def test_checkpoint_v8_is_current_version():
    # the engine's blob version is now 8.0 (MAP synthetic map#/__start__ fan-out topology).
    # A pre-8.0 blob is not loadable.
    from agent_composer.suspension.checkpoint import CHECKPOINT_VERSION
    assert CHECKPOINT_VERSION == "8.0"


def test_checkpoint_v4_blob_rejected_by_loads():
    # breaking blob migration: a 4.0 checkpoint predates the `expansions` field (5.0),
    # the `num_workers` drive-mode field (6.0), and the unified GrowRecord ledger (7.0),
    # so it is not loadable.
    blob = RunCheckpoint(pool=VariablePool()).dumps()
    tampered = json.dumps({**json.loads(blob), "version": "4.0"})
    with pytest.raises(ValueError, match=r"incompatible checkpoint version '4\.0'.*unified GrowRecord expansion ledger"):
        RunCheckpoint.loads(tampered)


def test_checkpoint_carries_expansions_field():
    # RunCheckpoint carries the GrowRecord ledger for runtime-grown subgraphs
    # so a paused run that has already expanded a CALL/MAP/AGENT/LOOP spawner can be
    # restored top-down on resume.
    from agent_composer.suspension import GrowRecord
    cp = RunCheckpoint(
        pool=VariablePool(),
        expansions=[GrowRecord(spawner_id="x", seed={}, children=[])],
    )
    back = RunCheckpoint.loads(cp.dumps())
    assert back.expansions[0].spawner_id == "x"


def test_checkpoint_expansions_default_empty():
    # the field defaults to []. A pure-static flow that never expanded
    # serializes/deserializes with an empty descriptor tree — no API change at the
    # call sites that don't pass `expansions=`.
    cp = RunCheckpoint(pool=VariablePool())
    back = RunCheckpoint.loads(cp.dumps())
    assert back.expansions == []


def test_snapshot_carries_expansions_from_ledger():
    # snapshot() reads engine.expansions (the live ledger) into
    # RunCheckpoint.expansions (the checkpoint field), so a paused checkpoint blob
    # carries the descriptor tree end-to-end. CALL spawner -> one GrowRecord.
    from tests.engine.test_engine_expansions_ledger import call_with_inner_pause
    g = call_with_inner_pause()
    engine = FlowEngine(g, run_inputs={"payload": "go"}, grow_builder=build_grow)
    assert isinstance(list(engine.run())[-1], RunPaused)
    ckpt = engine.snapshot()
    assert ckpt.expansions  # non-empty: bridge expanded into a child clone
    assert ckpt.expansions[0].spawner_id == "bridge"


def test_snapshot_empty_expansions_for_static_pause():
    # a pure-static flow (no REF/CALL/MAP/AGENT spawners) pauses
    # without growing the ledger; snapshot() preserves the empty descriptor tree.
    g = _ask_flow()
    engine = FlowEngine(g)
    assert isinstance(list(engine.run())[-1], RunPaused)
    ckpt = engine.snapshot()
    assert ckpt.expansions == []
