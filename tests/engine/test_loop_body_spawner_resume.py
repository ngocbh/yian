"""F1 — a SPAWNER (CALL/MAP) nested inside a LOOP body durably resumes.

The existing durable-loop coverage parks on a `human_input` LEAF inside a loop body
(`test_durable_replay.test_durable_loop_resumes_on_fresh_flow`) and on a loop nested
inside a CALL (`test_nested_loop_in_call_*`). This file closes the missing direction:
a CALL / MAP spawner nested *inside* a dynamically-grown loop iteration, paused mid-body
and resumed both in-process and across a durable checkpoint hop. If the generic growth
splice stamps nested spawners uniformly, this "falls out for free"; these tests are the
proof.
"""

from agent_composer.compose.loader import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow
from agent_composer.runtime.engine import FlowEngine
from agent_composer.suspension.checkpoint import RunCheckpoint


# A loop whose BODY (`chat_turn`) contains a CALL (`getmsg` -> `turn_body`) whose child
# holds the pausing `human_input`. So each iteration grows: body clone -> CALL clone ->
# child clone with the parked `ask` leaf — a spawner nested inside a grown loop iteration.
_CALL_IN_LOOP_BODY = """
id: cil
name: cil
defs:
  turn_body:
    nodes:
      ask:
        kind: human_input
        prompt: "your message"
        output: str
    output: ${ask.output}
  chat_turn:
    input:
      messages: list[str]
    nodes:
      getmsg:
        kind: call
        call: turn_body
      fold:
        kind: code
        code: tests.engine._compose_codefns:chat_fold
        input:
          messages: ${input.messages}
          msg: ${getmsg.output}
        output:
          messages: list[str]
          exited: bool
    output: ${fold.output}
nodes:
  loop:
    kind: loop
    call: chat_turn
    input:
      messages: []
      exited: false
    while: not ${exited}
    max: 100
output: ${loop.output}
"""


def test_call_in_loop_body_pauses_and_resumes_each_turn():
    """In-process: the CALL's child `human_input` parks the run each iteration; delivering
    a message resumes into the next iteration; "bye" flips `exited` and the run succeeds."""
    loaded = load_flow(_CALL_IN_LOOP_BODY)
    r1 = run_flow(loaded, {})
    assert r1.status == "paused", r1.error
    assert len(r1.pause_reasons) == 1

    r2 = resume_flow(loaded, engine=r1.engine,
                     commands=[resume_command(loaded, r1.pause_reasons[0], "hi")])
    assert r2.status == "paused", r2.error

    r3 = resume_flow(loaded, engine=r2.engine,
                     commands=[resume_command(loaded, r2.pause_reasons[0], "bye")])
    assert r3.status == "succeeded", r3.error
    assert r3.output == {"messages": ["hi", "bye"], "exited": True}


def test_call_in_loop_body_durable_hop_matches_live():
    """Durable: park at the turn-1 pause, round-trip the checkpoint, restore on a freshly
    recompiled flow, and drive the remaining turns to the SAME terminal as the live run."""
    loaded = load_flow(_CALL_IN_LOOP_BODY)

    # Live oracle (in-process).
    live1 = run_flow(loaded, {})
    assert live1.status == "paused", live1.error
    live2 = resume_flow(loaded, engine=live1.engine,
                        commands=[resume_command(loaded, live1.pause_reasons[0], "hi")])
    assert live2.status == "paused", live2.error
    live3 = resume_flow(loaded, engine=live2.engine,
                        commands=[resume_command(loaded, live2.pause_reasons[0], "bye")])
    assert live3.status == "succeeded", live3.error
    live_output = live3.output

    # Durable sequence: a fresh process parks at turn 1, persists, round-trips the blob.
    proc1 = run_flow(load_flow(_CALL_IN_LOOP_BODY), {})
    assert proc1.status == "paused", proc1.error
    ckpt = RunCheckpoint.loads(proc1.checkpoint.dumps())

    # Restore on a FRESH recompiled flow; the parked (namespaced) leaf must be in the graph.
    fresh = FlowEngine.restore(load_flow(_CALL_IN_LOOP_BODY).compiled, ckpt)
    assert ckpt.pause_reasons[0].node_id in fresh.flow.nodes

    dur2 = resume_flow(loaded, engine=fresh,
                       commands=[resume_command(loaded, ckpt.pause_reasons[0], "hi")])
    assert dur2.status == "paused", dur2.error
    dur3 = resume_flow(loaded, engine=dur2.engine,
                       commands=[resume_command(loaded, dur2.pause_reasons[0], "bye")])
    assert dur3.status == "succeeded", dur3.error
    assert dur3.output == live_output


# A loop whose body maps a pausing child over a 2-element list — a MAP spawner nested inside
# a grown loop iteration. Each iteration parks on the FIRST unresolved element's `ask` leaf.
_MAP_IN_LOOP_BODY = """
id: mil
name: mil
defs:
  turn_body:
    input:
      seed: str
    nodes:
      ask:
        kind: human_input
        prompt: "your message"
        output: str
    output: ${ask.output}
  chat_turn:
    input:
      messages: list[str]
    nodes:
      getmsgs:
        kind: map
        call: turn_body
        over: ${["a", "b"]}
        input:
          seed: ${item}
      fold:
        kind: code
        code: tests.engine._compose_codefns:chat_fold_list
        input:
          messages: ${input.messages}
          msgs: ${getmsgs.output}
        output:
          messages: list[str]
          exited: bool
    output: ${fold.output}
nodes:
  loop:
    kind: loop
    call: chat_turn
    input:
      messages: []
      exited: false
    while: not ${exited}
    max: 100
output: ${loop.output}
"""


def _deliver_all(loaded, result, value):
    """Resume `result`'s engine by delivering `value` to EVERY current pause (a MAP fans out
    one `human_input` per element, so an iteration parks on N simultaneous pauses)."""
    cmds = [resume_command(loaded, pr, value) for pr in result.pause_reasons]
    return resume_flow(loaded, engine=result.engine, commands=cmds)


def test_map_in_loop_body_pauses_and_resumes_each_turn():
    """In-process: each loop iteration maps the pausing child over 2 seeds, parking on 2
    simultaneous `human_input` pauses; delivering "hi" to both advances a turn, "bye" exits."""
    loaded = load_flow(_MAP_IN_LOOP_BODY)
    r1 = run_flow(loaded, {})
    assert r1.status == "paused", r1.error
    assert len(r1.pause_reasons) == 2

    r2 = _deliver_all(loaded, r1, "hi")
    assert r2.status == "paused", r2.error
    assert len(r2.pause_reasons) == 2

    r3 = _deliver_all(loaded, r2, "bye")
    assert r3.status == "succeeded", r3.error
    assert r3.output == {"messages": ["hi", "hi", "bye", "bye"], "exited": True}


def test_map_in_loop_body_durable_hop_matches_live():
    """Durable: park mid-loop on the MAP's per-element pauses, round-trip the checkpoint,
    restore on a fresh flow, and drive to the SAME terminal as the live run."""
    loaded = load_flow(_MAP_IN_LOOP_BODY)

    live1 = run_flow(loaded, {})
    live2 = _deliver_all(loaded, live1, "hi")
    live3 = _deliver_all(loaded, live2, "bye")
    assert live3.status == "succeeded", live3.error
    live_output = live3.output

    proc1 = run_flow(load_flow(_MAP_IN_LOOP_BODY), {})
    assert proc1.status == "paused", proc1.error
    ckpt = RunCheckpoint.loads(proc1.checkpoint.dumps())

    fresh = FlowEngine.restore(load_flow(_MAP_IN_LOOP_BODY).compiled, ckpt)
    for pr in ckpt.pause_reasons:
        assert pr.node_id in fresh.flow.nodes

    dur2 = resume_flow(loaded, engine=fresh,
                       commands=[resume_command(loaded, pr, "hi") for pr in ckpt.pause_reasons])
    assert dur2.status == "paused", dur2.error
    dur3 = _deliver_all(loaded, dur2, "bye")
    assert dur3.status == "succeeded", dur3.error
    assert dur3.output == live_output

