"""Restore-side replay: a deterministic fold over the persisted `expansions` descriptor
tree that re-grows a paused, runtime-GROWN run on a FRESH process.

`snapshot()` is the write half; `restore()` + `_replay_expansions` are the read half. These
tests pin: the clone+register helpers are deterministic (re-key identically),
`_replay_expansions` reproduces the live overlay byte-for-byte on a freshly recompiled flow
(a NESTED oracle), and a true cross-process `dumps->loads->restore(fresh)->resume` of a CALL
/ MAP / AGENT / nested grown run reaches the SAME terminal as the live engine.
"""

from agent_composer.compile.model import END_ID, START_ID, FlowOutput, NodeState
from agent_composer.events import RunPaused, RunSucceeded
from agent_composer.nodes.base import Grow
from agent_composer.nodes.call.node import CallNode
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.nodes.map.node import MapNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.suspension.checkpoint import RunCheckpoint
from agent_composer.suspension.commands import DeliverAnswerCommand
from agent_composer.suspension.expansions import GrowRecord
from tests.engine._fakes import FuncNode, stamp_reads
from tests.engine._graph_builder import _graph
from tests.engine.test_engine_expansions_ledger import (
    _inner_pause_child,
    call_with_inner_pause,
    map_with_inner_pause,
)


# --- the whole-arm clone+register helpers are deterministic ------------------ #


def _replay_one(engine, spawner_id, seed):
    """Replay a SINGLE grow the way `_replay_expansions` does — `node.replay_grow(seed)` rebuilds
    the spawner's own subgraph and `_apply_grow(..., schedule=False, record=rec)` re-splices it with
    effects suppressed. Returns the reused GrowRecord."""
    engine.sm.add_executing(spawner_id)
    rec = GrowRecord(spawner_id=spawner_id, seed=seed, children=[])
    engine.expansions.append(rec)
    sg = engine.flow.nodes[spawner_id].replay_grow(seed)
    engine._apply_grow(spawner_id, Grow(sg, seed=seed), schedule=False, record=rec)
    return rec


def test_grow_call_re_registers_identical_node_ids():
    """`replay_grow(seed)` + `_apply_grow(schedule=False)` re-keys the SAME cloned node ids for the
    same `(spawner, child, record)` across two fresh engines (the clone is pure: `ns(callsite,
    child_id)` has no counter)."""
    e1 = FlowEngine(call_with_inner_pause())
    _replay_one(e1, "bridge", {"payload": "go"})
    ids1 = {n for n in e1.flow.nodes if n.startswith("bridge/")}

    e2 = FlowEngine(call_with_inner_pause())
    _replay_one(e2, "bridge", {"payload": "go"})
    ids2 = {n for n in e2.flow.nodes if n.startswith("bridge/")}

    assert ids1 == ids2 and ids1  # identical, non-empty
    assert e1.flow.nodes["bridge/__end__"].commit_as == "bridge"
    assert e1.sm.node_state["bridge"] == NodeState.EXPANDED


# --- nested fixtures: CALL-in-CALL — the recursion the flat fixtures can't probe -------- #


def call_in_call_with_inner_pause():
    """Parent -> outer(CALL deep_child) -> after; deep_child -> deep(CALL inner_pause_child)
    -> END_ID. A CALL whose child contains a CALL whose child pauses — two nesting levels, so a
    dropped recursion / mis-stamped depth shows up (a flat CALL would not)."""
    inner = _inner_pause_child()
    deep = CallNode("deep", flow_id="inner_pause_child", child=inner,
                    child_inputs=inner.nodes[inner.start_id].params)
    stamp_reads(deep, {"payload": "${input.payload}"})
    deep_child = _graph(
        [deep],
        [(START_ID, "deep"), ("deep", END_ID)],
        outputs=[FlowOutput(name="answer", from_="${deep.output.answer}")],
    )
    outer = CallNode("outer", flow_id="deep_child", child=deep_child,
                     child_inputs=deep_child.nodes[deep_child.start_id].params)
    stamp_reads(outer, {"payload": "${input.payload}"})
    after = FuncNode("after", lambda p: {"output": p["v"]})
    stamp_reads(after, {"v": "${outer.output.answer}"})
    return _graph(
        [outer, after],
        [(START_ID, "outer"), ("outer", "after"), ("after", END_ID)],
        outputs=[FlowOutput(name="r", from_="${after.output}")],
    )


def _capture_overlay(engine):
    """The overlay a replay must reproduce: the registered topology + the commit redirect
    (baked on the nodes as `commit_as`, the alias-map replacement) + depth + spawner keys +
    edge_state keys."""
    return {
        "nodes": set(engine.flow.nodes),
        "commit_as": {nid: n.commit_as for nid, n in engine.flow.nodes.items() if n.commit_as},
        "depth": dict(engine.depth),
        "spawner_keys": set(engine._spawner_expansion),
        "edge_state_keys": set(engine.sm.edge_state),
    }


def test_replay_reproduces_live_overlay_nested_oracle():
    """Replaying `ckpt.expansions` onto a FRESH recompiled flow rebuilds the SAME
    flow.nodes / commit_as / depth / _spawner_expansion keys / edge_state (set-equality) the
    live engine grew. Uses a NESTED (CALL-in-CALL) fixture — a flat fixture is blind to a
    dropped recursion."""
    live = FlowEngine(call_in_call_with_inner_pause(), run_inputs={"payload": "go"})
    assert isinstance(list(live.run())[-1], RunPaused)
    live_overlay = _capture_overlay(live)
    ckpt = RunCheckpoint.loads(live.snapshot().dumps())

    # Replay in isolation (before restore() wires it in): a bare fresh engine on a
    # recompiled flow + _replay_expansions reproduces the overlay.
    fresh = FlowEngine(call_in_call_with_inner_pause())
    fresh._replay_expansions(ckpt.expansions)
    replayed = _capture_overlay(fresh)

    assert replayed["nodes"] == live_overlay["nodes"]
    assert replayed["commit_as"] == live_overlay["commit_as"]
    assert replayed["depth"] == live_overlay["depth"]
    assert replayed["spawner_keys"] == live_overlay["spawner_keys"]
    assert replayed["edge_state_keys"] == live_overlay["edge_state_keys"]


# --- restore() re-seed + clean-flow guard (the pure-static deferred case) ---------------- #


def _fork_with_deferred():
    """start -> {ask(HUMAN_INPUT), b1 -> b2} -> join -> END_ID. While `ask` is parked, the b1->b2
    branch runs b1 and b2 becomes ready DURING suspension -> held in `deferred`. Pure-static
    (no expansion), so it isolates the restore re-seed half (paused + deferred + ready)."""
    def _join(i):
        return {"output": [i["a"], i["b"]]}

    join = FuncNode("join", _join)
    stamp_reads(join, {"a": "${ask.output}", "b": "${b2.output}"})
    b1 = FuncNode("b1", lambda p: {"output": "b1"})
    b2 = FuncNode("b2", lambda p: {"output": p["v"]})
    stamp_reads(b2, {"v": "${b1.output}"})
    return _graph(
        [FuncNode("start", lambda p: {"output": "go"}),
         HumanInputNode("ask", prompt="ok?"), b1, b2, join],
        [(START_ID, "start"), ("start", "ask"), ("start", "b1"),
         ("b1", "b2"), ("ask", "join"), ("b2", "join"), ("join", END_ID)],
        outputs=[FlowOutput(name="r", from_="${join.output.output}")],
    )


def test_restore_static_fork_with_deferred_matches_live():
    """A fork whose un-parked branch deferred a node. A durable
    dumps->loads->restore(fresh)->resume(Deliver ask) reaches the SAME terminal as the live
    resume (RunSucceeded, both branches present)."""
    live = FlowEngine(_fork_with_deferred())
    assert isinstance(list(live.run())[-1], RunPaused)
    ckpt = live.snapshot()
    assert ckpt.deferred_nodes == ["b2"]        # b2 became ready while suspending
    assert ckpt.ready == []                      # the live serial pause fully drains ready
    live_out = list(live.resume(
        commands=[DeliverAnswerCommand(node_id="ask", value="A")]))[-1]
    assert isinstance(live_out, RunSucceeded)
    assert live_out.output[0] == "A"            # the delivered ask answer (both branches present)

    back = RunCheckpoint.loads(ckpt.dumps())
    fresh = FlowEngine.restore(_fork_with_deferred(), back)
    assert [n for n, _ in fresh.paused] == ["ask"]
    assert fresh.deferred == ["b2"]
    dur_out = list(fresh.resume(
        commands=[DeliverAnswerCommand(node_id="ask", value="A")]))[-1]
    assert isinstance(dur_out, RunSucceeded)
    assert dur_out.output == live_out.output    # durable restore(fresh)+resume == live


def test_restore_on_grown_flow_raises():
    """restore() requires a CLEAN flow. A flow already carrying namespaced/cloned ids (a
    re-grown one) raises ValueError BEFORE replay (add_subgraph is non-idempotent)."""
    import pytest

    live = FlowEngine(call_with_inner_pause(), run_inputs={"payload": "go"})
    assert isinstance(list(live.run())[-1], RunPaused)
    ckpt = RunCheckpoint.loads(live.snapshot().dumps())
    with pytest.raises(ValueError, match="requires a clean flow"):
        FlowEngine.restore(live.flow, ckpt)     # live.flow is already grown (has bridge/... ids)


# --- durable e2e over GROWN graphs on FRESHLY recompiled flows -------------------------- #


def _durable_resume(make_flow, run_inputs, answer):
    """Run `make_flow()` to a pause, snapshot->dumps->loads->restore on a SECOND freshly
    built flow (true cross-process), deliver `answer` to the parked leaf, drive to terminal.
    Returns (live_terminal, durable_terminal, restored_engine)."""
    live = FlowEngine(make_flow(), run_inputs=run_inputs)
    live_evs = list(live.run())
    assert isinstance(live_evs[-1], RunPaused)
    parked = live.snapshot().paused_nodes
    live_term = list(live.resume(
        commands=[DeliverAnswerCommand(node_id=p, value=answer) for p in parked]))[-1]

    # restart fresh: a true second process re-runs to the SAME pause, persists, restores.
    proc1 = FlowEngine(make_flow(), run_inputs=run_inputs)
    assert isinstance(list(proc1.run())[-1], RunPaused)
    ckpt = RunCheckpoint.loads(proc1.snapshot().dumps())
    fresh = FlowEngine.restore(make_flow(), ckpt)
    parked2 = ckpt.paused_nodes
    dur_term = list(fresh.resume(
        commands=[DeliverAnswerCommand(node_id=p, value=answer) for p in parked2]))[-1]
    return live_term, dur_term, fresh


def test_durable_call_inner_pause_resumes_on_fresh_flow():
    """A CALL grown to an inner pause resumes cross-process on a freshly recompiled flow
    -> RunSucceeded matching live. The delivered answer flows through the cloned child and the
    CALL substitutes it under the spawner id (pool['bridge']=='approve')."""
    live_term, dur_term, fresh = _durable_resume(
        call_with_inner_pause, {"payload": "go"}, "approve")
    assert isinstance(dur_term, RunSucceeded)
    assert dur_term.output == live_term.output            # durable == live (same terminal)
    assert fresh.pool.get("bridge") == "approve"          # the answer propagated through the CALL


def test_durable_map_inner_pause_resumes_on_fresh_flow():
    """A MAP grown to per-element inner pauses resumes cross-process -> RunSucceeded list,
    element order preserved. Asserts the map_end fan-in node was rebuilt + redirect-baked."""
    live_term, dur_term, fresh = _durable_resume(
        map_with_inner_pause, {"items": ["a", "b"]}, "ok")
    assert isinstance(dur_term, RunSucceeded)
    assert dur_term.output == live_term.output
    assert dur_term.output == ["ok", "ok"]                  # one per element, order preserved
    assert "each/__end__" in fresh.flow.nodes               # the LIST fan-in node
    assert fresh.flow.nodes["each/__end__"].commit_as == "each"


def _map_n0_via_sibling():
    """start -> {empty(MAP over []), ask(HUMAN_INPUT)} -> join -> END_ID. The MAP fires with N=0
    (emits []) while `ask` parks -> at pause the map GrowRecord carries seed==[] (N=0)."""
    child = _inner_pause_child()
    empty = MapNode("empty", flow_id="inner_pause_child", child=child,
                    child_inputs=child.nodes[child.start_id].params)
    stamp_reads(empty, {"over": "${input.items}", "payload": "${item}"})
    join = FuncNode("join", lambda i: {"output": [i["m"], i["a"]]})
    stamp_reads(join, {"m": "${empty.output}", "a": "${ask.output}"})
    return _graph(
        [empty, HumanInputNode("ask", prompt="ok?"), join],
        [(START_ID, "empty"), (START_ID, "ask"), ("empty", "join"), ("ask", "join"), ("join", END_ID)],
        outputs=[FlowOutput(name="r", from_="${join.output.output}")],
    )


def test_durable_map_n0_via_sibling_resumes_on_fresh_flow():
    """A MAP over [] that fired before a sibling pause restores + resumes cross-process.
    The replay rebuilds EndNode.list_(n=0) (a 0-incoming root that emits [])."""
    proc1 = FlowEngine(_map_n0_via_sibling(), run_inputs={"items": []})
    assert isinstance(list(proc1.run())[-1], RunPaused)
    ckpt = RunCheckpoint.loads(proc1.snapshot().dumps())
    maps = [e for e in ckpt.expansions if e.spawner_id == "empty"]
    assert len(maps) == 1 and maps[0].seed == []          # N=0 seed (empty records list) persisted

    fresh = FlowEngine.restore(_map_n0_via_sibling(), ckpt)
    assert "empty/__end__" in fresh.flow.nodes              # N=0 fan-in node rebuilt
    term = list(fresh.resume(
        commands=[DeliverAnswerCommand(node_id=ckpt.paused_nodes[0], value="A")]))[-1]
    assert isinstance(term, RunSucceeded)
    assert term.output == [[], "A"]                         # MAP over [] -> [], sibling -> "A"


# --- AGENT durable resume + the 2-hop ledger regression --------------------------------- #


def _two_pause_agent_chat():
    """A 2-pause mock chat (the test_agent_continuation pattern): two ask_user tool calls,
    then a FINAL answer."""
    from langchain_core.messages import AIMessage
    from tests.engine.test_agent_continuation import _ask
    return [_ask({"question": "q1?"}, "q1"), _ask({"question": "q2?"}, "q2"),
            AIMessage(content="FINAL")]


def test_durable_two_pause_agent_resumes_on_fresh_flow(monkeypatch):
    """A 2-pause AGENT restored at pause 1 onto a freshly recompiled flow resumes past BOTH
    pauses -> RunSucceeded 'FINAL'. The segment-2 leaf is deeply namespaced under the
    segment-1 resume id, and exists in the restored flow after the live segment-2 growth on
    the restored engine.

    ONE shared mock chat across the simulated processes: the carried memo replays prior turns
    WITHOUT re-invoking the model, so the 3 replies [q1, q2, FINAL] are consumed exactly once
    each even across restore (mirrors test_agent_continuation's single-chat invariant)."""
    import agent_composer.llm_clients as llm
    from agent_composer import load_flow
    from agent_composer.run import resume_command, run_flow
    from tests.engine.test_agent_continuation import _chat, ASK

    chat = _chat(_two_pause_agent_chat())                   # ONE instance shared across processes
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    loaded = load_flow(ASK)
    rec = run_flow(loaded, {})                              # parks at pause 1 (invoke #1 -> q1)
    assert rec.status == "paused"
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())

    fresh = FlowEngine.restore(load_flow(ASK).compiled, ckpt)
    # resume past pause 1 -> the restored engine grows segment 2 and parks at pause 2 (invoke #2).
    evs1 = list(fresh.resume(commands=[resume_command(loaded, ckpt.pause_reasons[0], "a1")]))
    paused2 = [e for e in evs1 if isinstance(e, RunPaused)]
    assert paused2, "segment-2 pause must surface on the restored engine"
    seg2_leaf = paused2[-1].reasons[0].node_id
    assert seg2_leaf.count("/") >= 2 and seg2_leaf in fresh.flow.nodes  # deeply namespaced
    # deliver pause 2 -> FINAL (invoke #3)
    evs2 = list(fresh.resume(commands=[resume_command(loaded, paused2[-1].reasons[0], "a2")]))
    assert isinstance(evs2[-1], RunSucceeded) and evs2[-1].output == "FINAL"


def test_two_hop_agent_resnapshot_ledger_matches_live(monkeypatch):
    """run->snapshot->restore(fresh)->resume-to-2nd-pause->RE-snapshot. The re-snapshot's
    expansions tree must equal the live engine's: ONE agent GrowRecord whose `children` carries the
    segment-2 nesting record (NOT two top-level agent records / a truncated 0-child tree). Then a
    3rd process restore(fresh)->resume reaches RunSucceeded 'FINAL'."""
    import agent_composer.llm_clients as llm
    from agent_composer import load_flow
    from agent_composer.run import resume_command, run_flow
    from tests.engine.test_agent_continuation import _chat, ASK

    # --- live oracle: run to pause 2 on ONE engine; capture its ledger shape ---
    # Its OWN chat (consumes its own 3 replies independently of the durable sequence).
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: _chat(_two_pause_agent_chat()))
    loaded = load_flow(ASK)
    live = run_flow(loaded, {})                             # pause 1
    list(live.engine.resume(commands=[resume_command(loaded, live.pause_reasons[0], "a1")]))  # pause 2
    live_tree = live.engine.snapshot().expansions
    assert len(live_tree) == 1 and live_tree[0].spawner_id == "agent"
    assert len(live_tree[0].children) == 1                  # ONE agent record, segment-2 nested child

    # --- the durable sequence: ONE shared chat across the 3 simulated processes (the memo
    # replays prior turns without re-invoking, so [q1, q2, FINAL] is consumed once each) ---
    chat = _chat(_two_pause_agent_chat())
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    proc1 = run_flow(load_flow(ASK), {})                   # pause 1 (invoke #1)
    ckpt1 = RunCheckpoint.loads(proc1.checkpoint.dumps())
    hop1 = FlowEngine.restore(load_flow(ASK).compiled, ckpt1)
    list(hop1.resume(commands=[resume_command(loaded, ckpt1.pause_reasons[0], "a1")]))  # pause 2 (invoke #2)
    hop1_tree = hop1.snapshot().expansions
    # the re-snapshot after a durable hop is the FULL tree, not a truncated/duplicated one.
    assert len(hop1_tree) == 1 and hop1_tree[0].spawner_id == "agent"
    assert len(hop1_tree[0].children) == 1

    # --- hop 2: a 3rd process restores the re-snapshot and finishes ---
    ckpt2 = RunCheckpoint.loads(hop1.snapshot().dumps())
    hop2 = FlowEngine.restore(load_flow(ASK).compiled, ckpt2)
    evs = list(hop2.resume(commands=[resume_command(loaded, ckpt2.pause_reasons[0], "a2")]))  # invoke #3
    assert isinstance(evs[-1], RunSucceeded) and evs[-1].output == "FINAL"


# --- NESTED durable resume — CALL-in-CALL + MAP-of-CALL --------------------------------- #


def test_durable_call_in_call_resumes_on_fresh_flow():
    """A CALL-in-CALL grown to a deep inner pause restores + resumes cross-process on a
    freshly recompiled flow. Asserts the doubly-namespaced leaf, the depth tree, and the
    nested-spawner parent-pointer were all rebuilt by the recursion."""
    proc1 = FlowEngine(call_in_call_with_inner_pause(), run_inputs={"payload": "go"})
    assert isinstance(list(proc1.run())[-1], RunPaused)
    ckpt = RunCheckpoint.loads(proc1.snapshot().dumps())
    live_term = list(proc1.resume(
        commands=[DeliverAnswerCommand(node_id=ckpt.paused_nodes[0], value="A")]))[-1]

    fresh = FlowEngine.restore(call_in_call_with_inner_pause(), ckpt)
    assert "outer/deep/ask" in fresh.flow.nodes                 # doubly namespaced
    assert fresh.depth["outer/deep"] == 1
    assert fresh.depth["outer/__end__"] == 1
    assert fresh.depth["outer/deep/__end__"] == 2
    assert "outer/deep" in fresh._spawner_expansion            # nested spawner parent-pointer
    dur_term = list(fresh.resume(
        commands=[DeliverAnswerCommand(node_id=ckpt.paused_nodes[0], value="A")]))[-1]
    assert isinstance(dur_term, RunSucceeded)
    assert dur_term.output == live_term.output                 # durable == live


def _map_of_call_with_inner_pause():
    """A MAP whose child contains a CALL whose child pauses (the in-repo
    test_nested_call_inside_map shape): the inner call GrowRecord rides under the map record's flat
    `children`, namespaced `each#0/inner_bridge` — recursion through a MAP element."""
    inner_child = _inner_pause_child()
    nested_call = CallNode("inner_bridge", flow_id="inner_pause_child", child=inner_child,
                           child_inputs=inner_child.nodes[inner_child.start_id].params)
    stamp_reads(nested_call, {"payload": "${input.payload}"})
    parent_child = _graph(
        [nested_call],
        [(START_ID, "inner_bridge"), ("inner_bridge", END_ID)],
        outputs=[FlowOutput(name="result", from_="${inner_bridge.output.answer}")],
    )
    each = MapNode("each", flow_id="parent_child", child=parent_child,
                   child_inputs=parent_child.nodes[parent_child.start_id].params)
    stamp_reads(each, {"over": "${input.items}", "payload": "${item}"})
    return _graph(
        [each],
        [(START_ID, "each"), ("each", END_ID)],
        outputs=[FlowOutput(name="r", from_="${each.output}")],
    )


def test_durable_map_of_call_resumes_on_fresh_flow():
    """A MAP-of-CALL grown to a deep per-element pause restores + resumes cross-process. The
    nested call GrowRecord lives under the map record's flat `children` (namespaced under each#0);
    the replay recurses into it and rebuilds the doubly-namespaced clone."""
    proc1 = FlowEngine(_map_of_call_with_inner_pause(), run_inputs={"items": ["a"]})
    assert isinstance(list(proc1.run())[-1], RunPaused)
    ckpt = RunCheckpoint.loads(proc1.snapshot().dumps())
    top_maps = [e for e in ckpt.expansions if e.spawner_id == "each"]
    assert len(top_maps) == 1
    kids = top_maps[0].children                                     # flat, namespaced by spawner_id
    assert len(kids) == 1 and kids[0].spawner_id == "each#0/inner_bridge"  # nested under elem 0
    live_term = list(proc1.resume(
        commands=[DeliverAnswerCommand(node_id=ckpt.paused_nodes[0], value="A")]))[-1]

    fresh = FlowEngine.restore(_map_of_call_with_inner_pause(), ckpt)
    assert any(n.startswith("each#0/inner_bridge/") for n in fresh.flow.nodes)  # doubly namespaced
    dur_term = list(fresh.resume(
        commands=[DeliverAnswerCommand(node_id=ckpt.paused_nodes[0], value="A")]))[-1]
    assert isinstance(dur_term, RunSucceeded)
    assert dur_term.output == live_term.output                 # durable == live


# --- AGENT(ask_user) nested under a CALL — durable resume ------------------------------- #
# The AGENT durable tests cover a TOP-LEVEL agent, and the CALL/MAP durable tests use a
# HUMAN_INPUT child — but nothing else exercises an AGENT pause whose GrowRecord rides UNDER a
# CALL's GrowRecord (the nested-spawner arm: the agent spawner is stamped in `_spawner_expansion`
# at the enclosing call record, so its grow rides under that record's flat `children`). These
# drive that nesting cross-process.

_CALL_WRAPS_AGENT = """
id: cag
name: cag
defs:
  approver:
    nodes:
      agent: {kind: agent, prompt: go, controls: [ask_user], output: str}
    output: ${agent.output}
nodes:
  gate:
    kind: call
    call: approver
output: ${gate.output}
"""


def test_durable_agent_under_call_resumes_on_fresh_flow(monkeypatch):
    """An AGENT(ask_user) inside a CALL child pauses ONCE; its GrowRecord rides UNDER the gate
    call's GrowRecord (the nested-spawner arm). A cross-process
    dumps->loads->restore(fresh)->resume drives past the pause to RunSucceeded 'FINAL'. The
    parked leaf is deeply namespaced under BOTH the call AND the agent spawner."""
    import agent_composer.llm_clients as llm
    from langchain_core.messages import AIMessage

    from agent_composer import load_flow
    from agent_composer.run import resume_command, run_flow
    from tests.engine.test_agent_continuation import _ask, _chat

    chat = _chat([_ask({"question": "ok?"}, "q1"), AIMessage(content="FINAL")])  # ONE shared chat
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    loaded = load_flow(_CALL_WRAPS_AGENT)
    rec = run_flow(loaded, {})                                  # invoke #1 -> ask_user -> pause
    assert rec.status == "paused"
    reason = rec.pause_reasons[0]
    assert reason.node_id == "gate/agent/__ask#q1"             # deep: call ns + agent spawner ns
    assert reason.node_id in rec.engine.flow.nodes

    # the descriptor tree: ONE call GrowRecord whose child is the agent GrowRecord (nesting)
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())
    calls = [e for e in ckpt.expansions if e.spawner_id == "gate"]
    assert len(calls) == 1
    assert [c.spawner_id for c in calls[0].children] == ["gate/agent"]

    fresh = FlowEngine.restore(load_flow(_CALL_WRAPS_AGENT).compiled, ckpt)
    assert reason.node_id in fresh.flow.nodes                  # the deep leaf rebuilt by replay
    evs = list(fresh.resume(commands=[resume_command(loaded, ckpt.pause_reasons[0], "yes")]))
    assert isinstance(evs[-1], RunSucceeded) and evs[-1].output == "FINAL"


def test_durable_two_pause_agent_under_call_resumes_on_fresh_flow(monkeypatch):
    """The multi-pause variant — an AGENT inside a CALL pauses TWICE. Restored at pause 1 onto
    a freshly recompiled flow, the resume grows segment 2 (still under the gate call GrowRecord)
    and parks at pause 2; delivering it reaches RunSucceeded 'FINAL'. The segment-2 leaf chains
    under the segment-1 resume id AND the call namespace (triply deep)."""
    import agent_composer.llm_clients as llm
    from langchain_core.messages import AIMessage

    from agent_composer import load_flow
    from agent_composer.run import resume_command, run_flow
    from tests.engine.test_agent_continuation import _ask, _chat

    chat = _chat([_ask({"question": "q1?"}, "q1"), _ask({"question": "q2?"}, "q2"),
                  AIMessage(content="FINAL")])                  # ONE shared chat across processes
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    loaded = load_flow(_CALL_WRAPS_AGENT)
    rec = run_flow(loaded, {})                                  # pause 1 (invoke #1)
    assert rec.status == "paused"
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())

    fresh = FlowEngine.restore(load_flow(_CALL_WRAPS_AGENT).compiled, ckpt)
    # resume past pause 1 -> the restored engine grows segment 2 and parks at pause 2 (invoke #2).
    evs1 = list(fresh.resume(commands=[resume_command(loaded, ckpt.pause_reasons[0], "a1")]))
    paused2 = [e for e in evs1 if isinstance(e, RunPaused)]
    assert paused2, "segment-2 pause must surface on the restored engine"
    seg2_leaf = paused2[-1].reasons[0].node_id
    assert seg2_leaf == "gate/agent/__resume#q1/__ask#q2"      # call ns + agent continuation chain
    assert seg2_leaf in fresh.flow.nodes

    # the re-snapshot ledger is still ONE call GrowRecord -> ONE agent GrowRecord with a nested
    # segment-2 child (the K-pause nesting chain).
    tree = fresh.snapshot().expansions
    calls = [e for e in tree if e.spawner_id == "gate"]
    assert len(calls) == 1 and len(calls[0].children) == 1
    agent_desc = calls[0].children[0]
    assert agent_desc.spawner_id == "gate/agent" and len(agent_desc.children) == 1

    # deliver pause 2 -> FINAL (invoke #3)
    evs2 = list(fresh.resume(commands=[resume_command(loaded, paused2[-1].reasons[0], "a2")]))
    assert isinstance(evs2[-1], RunSucceeded) and evs2[-1].output == "FINAL"


# --- review fixes ----------------------------------------------------------- #


def _side_counter_flow(runs):
    # start -> ask(HUMAN_INPUT) -> END_ID ; plus a `side` counter rooted off START_ID (dead-end leaf)
    return _graph(
        [FuncNode("start", lambda p: {"output": "go"}),
         HumanInputNode("ask", prompt="?"),
         FuncNode("side", lambda p: runs.append(1) or {"output": "s"})],
        [(START_ID, "start"), ("start", "ask"), ("ask", END_ID), (START_ID, "side"), ("side", END_ID)],
        outputs=[FlowOutput(name="r", from_="${ask.output}"), FlowOutput(name="s", from_="${side.output}")],
    )


def test_resume_clears_ready_so_a_queued_node_runs_once():
    """restore() re-seeds `self.ready` from the checkpoint; resume() must clear it before
    re-enqueuing the seed, else a queued id runs TWICE. No in-system path produces a non-empty
    `checkpoint.ready` (serial drain empties it), so this FORGES one to pin the invariant the
    re-seed introduced."""
    runs: list = []
    e1 = FlowEngine(_side_counter_flow(runs))
    assert isinstance(list(e1.run())[-1], RunPaused)          # `side` runs once here, parks at ask
    ck = e1.snapshot()
    # forge: pretend `side` was queued-but-not-run at the pause (the dormant re-seed path)
    ck = ck.model_copy(update={
        "ready": ["side"],
        "node_state": {**ck.node_state, "side": NodeState.UNKNOWN},
    })
    runs.clear()
    e2 = FlowEngine.restore(_side_counter_flow(runs), ck)
    assert list(e2.ready) == ["side"]                          # re-seeded from the checkpoint
    list(e2.resume(commands=[DeliverAnswerCommand(node_id="ask", value="a")]))
    assert runs == [1]                                         # ran EXACTLY once (without the fix: [1, 1])


def test_restore_does_not_mutate_a_held_checkpoint():
    """restore() deep-copies the pool + expansion descriptors (symmetric with snapshot()'s
    write-side copy), so resuming a restored engine does not retro-mutate a checkpoint object
    the host still holds (reachable via the public resume_flow(checkpoint=))."""
    proc1 = FlowEngine(call_with_inner_pause(), run_inputs={"payload": "go"})
    assert isinstance(list(proc1.run())[-1], RunPaused)
    held = proc1.snapshot()                                    # a retained object (no loads())
    before_keys = set(held.pool.store.keys())
    e2 = FlowEngine.restore(call_with_inner_pause(), held)
    assert held.pool is not e2.pool                            # deep-copied, not aliased
    assert held.expansions[0] is not e2.expansions[0]
    list(e2.resume(commands=[DeliverAnswerCommand(node_id=e2.paused[0][0], value="approve")]))
    assert set(held.pool.store.keys()) == before_keys          # the held checkpoint is untouched


def test_replay_does_not_promote_a_nested_child(monkeypatch):
    """A nested child GrowRecord (a grow spliced UNDER a parent — e.g. an AGENT re-pause chained
    under its segment-1 record, or a CALL inside a MAP element) must NOT be promoted to a top-level
    ledger entry. `_replay_expansions` appends only when `is_top_level`; a parent's `children` are
    walked with `is_top_level=False` (they already ride under the parent by deserialization). This
    pins that gate against a reserved nested slot."""
    inner = GrowRecord(spawner_id="inner", seed={}, children=[])
    top = GrowRecord(spawner_id="agent", seed={}, children=[inner])
    e = FlowEngine(_side_counter_flow([]))

    class _FakeSpawner:
        # so the fold's `flow.nodes[...]/.replay_grow` succeed without cloning a real subgraph
        def replay_grow(self, seed):
            return None

    e.flow.nodes["agent"] = _FakeSpawner()
    e.flow.nodes["inner"] = _FakeSpawner()
    monkeypatch.setattr(e, "_apply_grow", lambda *a, **k: None)
    e.expansions = []
    e._replay_expansions([top])
    assert e.expansions == [top]                               # NOT [top, inner]


def test_grow_loop_schedule_false_registers_without_scheduling():
    """Replaying ONE loop iteration (`replay_grow(seed)` + `_apply_grow(schedule=False)`) rebuilds
    the iteration overlay (clones + registers the `#0/` body namespace AND the fresh `loop~1`
    driver, re-attaches the single record to `_origin_record`) but schedules NOTHING and does not
    trip the node-budget guard — the suppressed replay path, mirroring the CALL/MAP replay."""
    from agent_composer.compose.loader import load_flow
    from tests.engine.test_loop_run import COUNTER

    engine = FlowEngine(load_flow(COUNTER).compiled, run_inputs={})
    spawner_id = "loop"
    record = {"n": 0, "exited": False}
    rec = _replay_one(engine, spawner_id, (record, 0))

    # Registered: the iteration's namespaced body nodes + the fresh next driver are present and the
    # single record is re-attached under the origin.
    assert any(n.startswith(f"{spawner_id}#0/") for n in engine.flow.nodes)
    assert "loop~1" in engine.flow.nodes                  # fresh next-iteration driver rebuilt
    assert engine._origin_record[spawner_id] is rec

    # Scheduled: nothing — the serial ready queue is empty right after the suppressed grow.
    assert list(engine.ready) == []


def test_replay_reproduces_live_loop_overlay():
    """Replaying a paused-loop `ckpt.expansions` onto a FRESH recompiled flow rebuilds the SAME
    live iteration overlay the live engine grew: flow.nodes (the `loop#0/` body + the fresh `loop~1`
    driver) / the baked `commit_as` map / sm.node_state (set-equality), the spawner staying
    EXPANDED, and the single record re-attached to `_origin_record`.

    Pruning drops committed iterations before any pause, so exactly ONE iteration (the LIVE one)
    is resident — the replay re-grows that single last-recorded seed at its recorded index. The
    CONTINUE body-END carries NO `commit_as` (it feeds `loop~1` by plain wiring), so the loop
    contributes nothing to the `commit_as` map at this pause."""
    from agent_composer.compose.loader import load_flow
    from agent_composer.run import run_flow
    from tests.engine.test_loop_run import LOOP_CHAT

    # Drive LOOP_CHAT to its FIRST pause on a live engine: iteration #0 is grown (its body parks on
    # the human_input leaf BEFORE its END fires, so the #0 overlay is un-pruned and resident).
    loaded = load_flow(LOOP_CHAT)
    live = run_flow(loaded, {})
    assert live.status == "paused"
    live_engine = live.engine

    # Precondition: the live overlay has the `loop#0/` body namespace AND the fresh `loop~1` driver.
    assert any(n.startswith("loop#0/") for n in live_engine.flow.nodes)
    assert "loop~1" in live_engine.flow.nodes

    # Capture the loop-relevant overlay a replay must reproduce: flow.nodes, the baked `commit_as`
    # map (the CONTINUE body-END carries none), sm.node_state, and the spawner's EXPANDED state.
    live_nodes = set(live_engine.flow.nodes)
    live_commit_as = {nid: n.commit_as for nid, n in live_engine.flow.nodes.items() if n.commit_as}
    live_node_state = set(live_engine.sm.node_state)
    assert live_engine.sm.node_state["loop"] == NodeState.EXPANDED

    # Snapshot + round-trip: exactly ONE loop GrowRecord with a non-empty `(record, index)` seed.
    ckpt = RunCheckpoint.loads(live_engine.snapshot().dumps())
    loop_descs = [e for e in ckpt.expansions if e.spawner_id == "loop"]
    assert len(loop_descs) == 1
    assert loop_descs[0].seed                           # the live iteration (record, index) seed

    # Replay in isolation onto a FRESH recompiled flow (before restore() wires it in).
    fresh = FlowEngine(load_flow(LOOP_CHAT).compiled)
    fresh._replay_expansions(ckpt.expansions)

    # Set-for-set overlay equality against the live engine — the rebuilt window {body_0, loop~1}
    # matches live.
    assert set(fresh.flow.nodes) == live_nodes
    assert {nid: n.commit_as for nid, n in fresh.flow.nodes.items() if n.commit_as} == live_commit_as
    assert set(fresh.sm.node_state) == live_node_state
    assert fresh.sm.node_state["loop"] == NodeState.EXPANDED
    assert fresh._origin_record["loop"] is loop_descs[0]  # the single record was re-attached


def test_durable_loop_resumes_on_fresh_flow():
    """A `loop` paused mid-body resumes cross-process on a freshly recompiled flow, driven
    through MULTIPLE turns (with a durable hop across a mid-loop pause), reaching the SAME
    terminal as the live in-process resume.

    Pruning drops committed iterations before any pause, so at the turn-1 pause only the LIVE
    `loop#0/` iteration (plus the fresh `loop~1` driver) is resident. The checkpoint carries ONE
    loop GrowRecord with the live seed recorded; restore()+replay re-grows exactly that window on a
    clean flow. Delivering "hi" then drives the fresh `loop~1` driver's grow, which must grow turn 2
    and prune turn 1 (its previous body + itself) from the RESTORED overlay — the post-restore
    `_origin_record` must be consistent for the next driver to attribute its grow. Delivering "bye"
    sets exited=true -> predicate false -> the run succeeds identically to the live oracle."""
    from agent_composer.compose.loader import load_flow
    from agent_composer.run import resume_command, resume_flow, run_flow
    from tests.engine.test_loop_run import LOOP_CHAT

    # --- Live oracle (in-process): drive LOOP_CHAT to terminal exactly as the in-process test.
    loaded = load_flow(LOOP_CHAT)
    live1 = run_flow(loaded, {})
    assert live1.status == "paused"
    live2 = resume_flow(loaded, engine=live1.engine,
                        commands=[resume_command(loaded, live1.pause_reasons[0], "hi")])
    assert live2.status == "paused"
    live3 = resume_flow(loaded, engine=live2.engine,
                        commands=[resume_command(loaded, live2.pause_reasons[0], "bye")])
    assert live3.status == "succeeded", live3.error
    live_output = live3.output
    assert live_output == {"messages": ["hi", "bye"], "exited": True}

    # --- Durable sequence: a fresh process parks at the turn-1 pause, persists, round-trips.
    proc1 = run_flow(load_flow(LOOP_CHAT), {})
    assert proc1.status == "paused"
    ckpt = RunCheckpoint.loads(proc1.checkpoint.dumps())   # cross-process round-trip

    # Exactly ONE loop GrowRecord, with the live iteration's `(record, index)` seed (non-empty).
    loop_descs = [e for e in ckpt.expansions if e.spawner_id == "loop"]
    assert len(loop_descs) == 1
    assert loop_descs[0].seed                               # the live #0 (record, index) seed

    # Pruning leaves only the live iteration: node_state carries `loop#0/` keys but no `loop#1/`.
    assert any(k.startswith("loop#0/") for k in ckpt.node_state)
    assert not any(k.startswith("loop#1/") for k in ckpt.node_state)

    # --- Restore on a FRESH recompiled flow and drive the remaining turns to terminal.
    fresh = FlowEngine.restore(load_flow(LOOP_CHAT).compiled, ckpt)

    # The replay rebuilt the live iteration's namespaced human_input leaf on the fresh flow.
    parked_leaf = ckpt.pause_reasons[0].node_id
    assert parked_leaf in fresh.flow.nodes

    # Deliver "hi" to the restored turn-1 pause -> the fresh `loop~1` driver grows turn 2 -> a NEW pause.
    dur2 = resume_flow(loaded, engine=fresh,
                       commands=[resume_command(loaded, ckpt.pause_reasons[0], "hi")])
    assert dur2.status == "paused", dur2.error
    # Deliver "bye" -> fold sets exited=true -> predicate false -> succeeds.
    dur3 = resume_flow(loaded, engine=dur2.engine,
                       commands=[resume_command(loaded, dur2.pause_reasons[0], "bye")])
    assert dur3.status == "succeeded", dur3.error
    assert dur3.output == live_output                       # durable == live (same terminal)


def test_loop_multi_hop_resnapshot_ledger_matches_live():
    """run->snapshot->restore(fresh)->resume-past-a-mid-loop-pause->RE-snapshot. The re-snapshot's
    loop GrowRecord ledger must equal the live engine's at the SAME (2nd) pause: ONE loop GrowRecord
    whose `seed` reflects the CURRENT live iteration (grown+pruned once), NOT a stale record from
    hop 1, nor a duplicated one. Then a 3rd process restores the re-snapshot and finishes to
    `succeeded`.

    Mirrors `test_two_hop_agent_resnapshot_ledger_matches_live` for the loop: the durable hop must
    continue growing the ONE re-attached descriptor object (`_origin_record["loop"]` is the
    deserialized `expansions[0]`), so the next driver's CONTINUE grow supersedes it on the ledger the
    re-snapshot serializes."""
    from agent_composer.compose.loader import load_flow
    from agent_composer.run import resume_command, resume_flow, run_flow
    from tests.engine.test_loop_run import LOOP_CHAT

    # --- Live oracle: drive LOOP_CHAT to the SECOND pause (turn 2) in-process; capture its ledger
    # shape. Turn 1 grows iter #0; delivering "a" drives the fresh `loop~1` driver, which grows iter #1
    # (seed = ({messages:["a"], exited:false}, 1)) and prunes #0 — exactly ONE record remains.
    loaded = load_flow(LOOP_CHAT)
    live1 = run_flow(loaded, {})
    assert live1.status == "paused"
    live2 = resume_flow(loaded, engine=live1.engine,
                        commands=[resume_command(loaded, live1.pause_reasons[0], "a")])
    assert live2.status == "paused"
    live_tree = live2.engine.snapshot().expansions
    assert len(live_tree) == 1 and live_tree[0].spawner_id == "loop"
    live_record, live_index = live_tree[0].seed
    assert live_record == {"messages": ["a"], "exited": False}         # the live iteration seed

    # --- Durable sequence: 3 simulated processes. proc1 parks at pause 1, persists, round-trips.
    proc1 = run_flow(load_flow(LOOP_CHAT), {})
    assert proc1.status == "paused"
    ckpt1 = RunCheckpoint.loads(proc1.checkpoint.dumps())             # cross-process round-trip
    hop1_descs = [e for e in ckpt1.expansions if e.spawner_id == "loop"]
    assert len(hop1_descs) == 1 and hop1_descs[0].seed                # pause-1 ledger: one record

    # hop 1: restore on a FRESH recompiled flow, resume past pause 1 delivering "a" -> pause 2.
    hop1 = FlowEngine.restore(load_flow(LOOP_CHAT).compiled, ckpt1)
    dur2 = resume_flow(loaded, engine=hop1,
                       commands=[resume_command(loaded, ckpt1.pause_reasons[0], "a")])
    assert dur2.status == "paused", dur2.error

    # CORE regression: the re-snapshot after a durable hop is the CURRENT live ledger, not a
    # stale/duplicated one. ONE loop GrowRecord whose seed is the current live iteration.
    hop1_tree = hop1.snapshot().expansions
    assert len(hop1_tree) == 1 and hop1_tree[0].spawner_id == "loop"
    hop_record, hop_index = hop1_tree[0].seed
    assert hop_record == live_record and hop_index == live_index

    # Pruning invariant survives the hop: exactly ONE `loop#*/` iteration is resident.
    ckpt2 = RunCheckpoint.loads(hop1.snapshot().dumps())
    live_iters = {k.split("/", 1)[0] for k in ckpt2.node_state if k.startswith("loop#")}
    assert len(live_iters) == 1

    # --- hop 2: a 3rd process restores the re-snapshot and finishes. Deliver "bye" -> fold sets
    # exited=true -> predicate false -> succeeds. Two messages delivered total: "a" then "bye".
    hop2 = FlowEngine.restore(load_flow(LOOP_CHAT).compiled, ckpt2)
    dur3 = resume_flow(loaded, engine=hop2,
                       commands=[resume_command(loaded, ckpt2.pause_reasons[0], "bye")])
    assert dur3.status == "succeeded", dur3.error
    assert dur3.output == {"messages": ["a", "bye"], "exited": True}


# --- nested loop durability (review S2): a loop whose body/callsite rides UNDER a CALL --------- #
# Nothing else exercises a LOOP nested inside a CALL — the origin-keyed ledger must nest the loop's
# ONE GrowRecord under the enclosing CALL record (origin = the NAMESPACED `wrap/loop`, since the CALL
# clones the loop into its child callsite), stay bounded across iterations, and replay live==restored.

# A `times: 3` loop nested in a CALL — completes with no pause. Pins the nested single-record ledger.
_NESTED_LOOP_IN_CALL = """
id: nlc
name: nlc
defs:
  countdown:
    input:
      n: int
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:loop_countdown
        input:
          n: ${input.n}
        output:
          n: int
    output: ${step.output}
  looped:
    input:
      n: int
    nodes:
      loop:
        kind: loop
        call: countdown
        input:
          n: ${input.n}
        times: 3
    output: ${loop.output}
nodes:
  wrap:
    kind: call
    call: looped
    input:
      n: 10
output: ${wrap.output}
"""


def test_nested_loop_in_call_ledger_single_record_nested_under_call():
    """A `times: 3` loop nested inside a CALL: the ledger stays ONE top-level CALL GrowRecord whose
    `children` carries EXACTLY ONE loop record (origin = the namespaced `wrap/loop`), stable across
    the 3 iterations — not an unbounded chain (review S2). The run completes to `{n: 7}`. This also
    guards the origin-namespacing fix in `clone_child`: without it the cloned loop keeps the
    pre-clone origin `loop`, its STOP `commit_as` targets a non-existent `flow.nodes['loop']`, and
    the run crashes."""
    from agent_composer.compose.loader import load_flow

    engine = FlowEngine(load_flow(_NESTED_LOOP_IN_CALL).compiled, run_inputs={})
    terminal = list(engine.run())[-1]
    assert isinstance(terminal, RunSucceeded)
    assert terminal.output == {"n": 7}                      # 10 - 3 body runs = 7
    # The CALL record is the sole top-level entry; the loop rides under it as ONE nested record.
    assert len(engine.expansions) == 1
    call_rec = engine.expansions[0]
    assert call_rec.spawner_id == "wrap"
    loop_kids = [c for c in call_rec.children if c.spawner_id == "wrap/loop"]
    assert len(loop_kids) == 1                              # single record, not an unbounded chain
    assert engine._origin_record["wrap/loop"] is loop_kids[0]


# A PAUSING nested loop (chat body inside a CALL) — snapshot MID-loop, restore, resume to terminal.
_NESTED_CHAT_LOOP_IN_CALL = """
id: ncc
name: ncc
defs:
  chat_turn:
    input:
      messages: list[str]
    nodes:
      ask:
        kind: human_input
        prompt: "your message"
        output: str
      fold:
        kind: code
        code: tests.engine._compose_codefns:chat_fold
        input:
          messages: ${input.messages}
          msg: ${ask.output}
        output:
          messages: list[str]
          exited: bool
    output: ${fold.output}
  looped:
    input:
      messages: list[str]
    nodes:
      loop:
        kind: loop
        call: chat_turn
        input:
          messages: ${input.messages}
          exited: false
        while: not ${exited}
        max: 100
    output: ${loop.output}
nodes:
  wrap:
    kind: call
    call: looped
    input:
      messages: []
output: ${wrap.output}
"""


def test_nested_loop_in_call_durable_hop_bounded_and_matches_live():
    """A loop nested in a CALL, iterating across a DURABLE hop: run -> mid-loop pause -> snapshot ->
    restore(fresh) -> resume. The re-attached ledger stays ONE CALL record with ONE nested loop
    record (origin `wrap/loop`) at each pause — bounded, no chain — and the restored run reaches the
    SAME terminal as the live in-process run (review S2)."""
    from agent_composer.compose.loader import load_flow
    from agent_composer.run import resume_command, resume_flow, run_flow

    loaded = load_flow(_NESTED_CHAT_LOOP_IN_CALL)

    # --- Live oracle: drive the nested chat loop to terminal in-process ("hi" then "bye").
    live1 = run_flow(loaded, {})
    assert live1.status == "paused"
    assert live1.pause_reasons[0].node_id == "wrap/loop#0/ask"   # deep: CALL ns + loop body ns
    live_tree = [(r.spawner_id, [c.spawner_id for c in r.children]) for r in live1.engine.expansions]
    assert live_tree == [("wrap", ["wrap/loop"])]               # ONE call record, ONE nested loop
    live2 = resume_flow(loaded, engine=live1.engine,
                        commands=[resume_command(loaded, live1.pause_reasons[0], "hi")])
    assert live2.status == "paused"
    live3 = resume_flow(loaded, engine=live2.engine,
                        commands=[resume_command(loaded, live2.pause_reasons[0], "bye")])
    assert live3.status == "succeeded", live3.error
    assert live3.output == {"messages": ["hi", "bye"], "exited": True}

    # --- Durable hop: a fresh process parks at the turn-1 pause, persists, round-trips.
    proc1 = run_flow(load_flow(_NESTED_CHAT_LOOP_IN_CALL), {})
    assert proc1.status == "paused"
    ckpt = RunCheckpoint.loads(proc1.checkpoint.dumps())
    # The persisted ledger is bounded: ONE call record with ONE nested loop record.
    ckpt_tree = [(r.spawner_id, [c.spawner_id for c in r.children]) for r in ckpt.expansions]
    assert ckpt_tree == [("wrap", ["wrap/loop"])]

    fresh = FlowEngine.restore(load_flow(_NESTED_CHAT_LOOP_IN_CALL).compiled, ckpt)
    assert ckpt.pause_reasons[0].node_id in fresh.flow.nodes    # deep leaf rebuilt by replay
    dur2 = resume_flow(loaded, engine=fresh,
                       commands=[resume_command(loaded, ckpt.pause_reasons[0], "hi")])
    assert dur2.status == "paused", dur2.error
    # RE-snapshot mid-loop after the hop: STILL bounded — the next iteration superseded the prior.
    hop_tree = [(r.spawner_id, [c.spawner_id for c in r.children])
                for r in dur2.engine.snapshot().expansions]
    assert hop_tree == [("wrap", ["wrap/loop"])]               # single record survives the hop
    dur3 = resume_flow(loaded, engine=dur2.engine,
                       commands=[resume_command(loaded, dur2.pause_reasons[0], "bye")])
    assert dur3.status == "succeeded", dur3.error
    assert dur3.output == live3.output                          # durable == live (same terminal)


def test_snapshot_captures_num_workers():
    from agent_composer.runtime.engine import FlowEngine
    from tests.engine.test_engine_expansions_ledger import call_with_inner_pause
    eng = FlowEngine(call_with_inner_pause(), num_workers=3)
    assert eng.snapshot().num_workers == 3


def test_restore_defaults_to_checkpointed_count_and_override():
    """restore() rebuilds at the checkpoint's num_workers; an explicit kwarg overrides."""
    from agent_composer.runtime.engine import FlowEngine
    from tests.engine.test_engine_expansions_ledger import call_with_inner_pause
    src = FlowEngine(call_with_inner_pause(), num_workers=2)
    ckpt = src.snapshot()
    # fresh clean flow per restore (restore mutates flow in place; replay needs a clean graph)
    e_default = FlowEngine.restore(call_with_inner_pause(), ckpt)
    assert e_default.num_workers == 2
    e_override = FlowEngine.restore(call_with_inner_pause(), ckpt, num_workers=0)
    assert e_override.num_workers == 0


def test_durable_resume_pooled_matches_serial():
    """dumps -> loads -> restore(num_workers=N) -> resume reaches the same terminal as a
    serial durable resume. A run checkpointed serial is resumable pooled (override) and
    vice-versa — correctness is worker-count-independent."""
    from agent_composer.run import run_flow, resume_flow
    from agent_composer.compose.loader import load_flow
    from tests.engine.test_run_resume import _RESUME_FANOUT

    loaded = load_flow(_RESUME_FANOUT)
    paused = run_flow(loaded, {"settle_at": "2026-07-01"}, num_workers=0)
    blob = paused.checkpoint.dumps()
    assert paused.checkpoint.num_workers == 0

    # cross-process round-trip, resumed POOLED via the override (resume_flow passthrough)
    ckpt = RunCheckpoint.loads(blob)
    res = resume_flow(load_flow(_RESUME_FANOUT), checkpoint=ckpt, num_workers=4,
                      commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert res.status == "succeeded", res.error

    # serial durable resume for the oracle
    ser = resume_flow(load_flow(_RESUME_FANOUT), checkpoint=RunCheckpoint.loads(blob),
                      commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert res.output == ser.output
