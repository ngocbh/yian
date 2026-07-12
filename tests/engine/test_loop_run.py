"""End-to-end `while:` loop runs (Ollama-free, pure-CODE body).

The loop carries `{n, exited}`, runs the body `bump` (`{n, exited} -> {n: n+1, exited: n+1>=3}`)
while `not ${exited}`, and terminates when the predicate goes false. Covers the multi-iteration
happy path, the turn-0 (0-iteration) case (seed already satisfies exit), the max-exceeded
runaway guard, and the error boundary — a `while:` predicate that raises at runtime and a
node-budget blowup both become failed runs, never uncaught escapes.

Loop model (self-respawn, no engine `_loop_step`): each iteration is a fresh `LoopNode` driver
clone. `L` is the compiled loop id (iteration 0); `L~k` is the fresh driver clone for iteration
`k >= 1`; `L#k/…` is the cloned body namespace for iteration `k`. A driver's `run` decides
continue-vs-stop: STOP returns `Output(carried, commit_as=L)` (generic commit under `L`);
CONTINUE emits a `Grow` splicing `body_k` + a fresh `L~(k+1)` driver, self-pruning the previous
body and (except the origin) itself. The CONTINUE body-END carries NO `commit_as` — it feeds the
next driver by plain wiring.
"""

from agent_composer.compose.loader import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow


COUNTER = """
id: counter
name: counter
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not ${exited}
    max: 10
output: ${loop.output}
"""

# Seed already satisfies exited: true -> 0 body runs, seed committed unchanged.
TURN0 = COUNTER.replace("exited: false", "exited: true")
# max: 2 (< the 3 iterations the body needs) -> the runaway guard fires.
MAXED = COUNTER.replace("max: 10", "max: 2")


def test_while_loop_runs_until_predicate_false():
    result = run_flow(load_flow(COUNTER), {})
    assert result.status == "succeeded"
    assert result.output == {"n": 3, "exited": True}


def test_while_loop_zero_iterations_commits_seed():
    result = run_flow(load_flow(TURN0), {})
    assert result.status == "succeeded"
    assert result.output == {"n": 0, "exited": True}   # seed committed unchanged; 0 body runs


def test_while_loop_max_exceeded_fails_run():
    result = run_flow(load_flow(MAXED), {})
    assert result.status != "succeeded"
    # Assert the runaway guard fired specifically — not just any error mentioning "max"
    # (a `MAX_TOTAL_NODES` budget blowup also contains "max"). The guard message is
    # `loop 'loop' exceeded max (2)`; the budget error says "exceeded node budget".
    assert "exceeded max" in (result.error or "")
    # And confirm the located error type reaches the RunFailed event.
    error_types = {getattr(e, "error_type", None) for e in result.events}
    assert "LoopMaxExceeded" in error_types


def test_while_loop_max_permits_exactly_max_body_runs():
    """The `max:` budget permits EXACTLY `max` body runs, not `max - 1` (the runaway guard
    fires when the driver at index `k` would grow the (k+1)-th body past budget: raise on
    `k >= max_iters`). COUNTER needs 3 iterations to reach `exited`, so `max: 3` succeeds
    (3 bodies) while `max: 2` fails — pins the guard against an off-by-one either way."""
    at_boundary = COUNTER.replace("max: 10", "max: 3")
    ok = run_flow(load_flow(at_boundary), {})
    assert ok.status == "succeeded"
    assert ok.output == {"n": 3, "exited": True}   # exactly 3 bodies ran
    below = COUNTER.replace("max: 10", "max: 2")   # one short -> runaway guard fires
    assert run_flow(load_flow(below), {}).status != "succeeded"


# The chat-shaped slice: a body that PAUSES each turn on a `human_input` leaf, folds the
# delivered message into the carried {messages, exited} record, and loops until the human
# types "bye". Drives run() -> paused -> resume -> paused -> ... -> succeeded in-process.
LOOP_CHAT = """
id: chat
name: chat
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


def test_loop_body_pauses_and_resumes_each_turn():
    loaded = load_flow(LOOP_CHAT)
    # Turn 1: the body's human_input leaf parks the run.
    r1 = run_flow(loaded, {})
    assert r1.status == "paused"
    assert len(r1.pause_reasons) == 1
    # Deliver "hi" -> body END fires -> the fresh driver `loop~1` grows the next iteration -> pauses.
    r2 = resume_flow(loaded, engine=r1.engine,
                     commands=[resume_command(loaded, r1.pause_reasons[0], "hi")])
    assert r2.status == "paused"
    # Deliver "bye" -> fold sets exited=true -> predicate false -> the run succeeds.
    r3 = resume_flow(loaded, engine=r2.engine,
                     commands=[resume_command(loaded, r2.pause_reasons[0], "bye")])
    assert r3.status == "succeeded", r3.error
    assert r3.output == {"messages": ["hi", "bye"], "exited": True}


# A loop that is NOT the terminal node: its committed record feeds a downstream code node.
# Guards that the STOP `Output(commit_as=loop)` commits under the spawner id AND fires the
# spawner's out-edges (the terminate -> commit -> advance tail), not just that a terminal loop
# returns a value.
DOWNSTREAM = COUNTER.replace(
    "output: ${loop.output}",
    """  after:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${loop.output.n}
    output: int
output: ${after.output}""",
)


def test_loop_feeding_downstream_node_commits_and_advances():
    result = run_flow(load_flow(DOWNSTREAM), {})
    assert result.status == "succeeded", result.error
    assert result.output == 6          # loop ends at n=3; double(3) = 6


# The `while:` predicate is evaluated INSIDE the driver's `run` (in `_should_stop_now`), which
# `eval_node` wraps in its try/except. A predicate that raises at runtime must become a FAILED
# run, never an uncaught escape from run(). Here the predicate divides by the carried `n`
# (`10 / ${n} > 0`) and the body counts n down: the seed pre-check (n=2, on the compiled `L`) and
# iteration-0 check (n=1, on `loop~1`) pass, then iteration-1's check divides by 0 -> the predicate
# raises inside the driver's `run`.
PREDICATE_RAISES = """
id: pred-raise
name: pred_raise
defs:
  down:
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
nodes:
  loop:
    kind: loop
    call: down
    input:
      n: 2
    while: 10 / ${n} > 0
    max: 10
output: ${loop.output}
"""


def test_while_loop_predicate_runtime_error_fails_run():
    # Must NOT raise out of run_flow; the predicate's division-by-zero on iteration 1 is
    # converted to a located run failure. The predicate is evaluated inside the driver's `run`
    # (via `_should_stop_now`), which `eval_node`'s boundary turns into a NodeFailed.
    result = run_flow(load_flow(PREDICATE_RAISES), {})
    assert result.status != "succeeded"
    assert "division by zero" in (result.error or "")


# `until:` is a DO-WHILE: the body must run AT LEAST once before the predicate is ever
# consulted. Here the seed n=1 does NOT yet satisfy `until: ${n} <= 0`, so a `while`-style
# turn-0 pre-check would commit the seed unchanged with 0 runs ({n: 1}). Correct `until` grows
# #0 unconditionally: the body counts down 1 -> 0, then the post-check `${n} <= 0` on {n:0} is
# true and terminates at {n:0} after exactly one run.
UNTIL_SEED_TRUE = """
id: ust
name: ust
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
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 1
    until: ${n} <= 0
    max: 5
output: ${loop.output}
"""


def test_until_runs_body_at_least_once():
    # `until` is do-while: turn-0 grows #0 unconditionally even though the seed n=1 does NOT yet
    # satisfy `until: ${n} <= 0`. Body counts down 1 -> 0, post-check `${n} <= 0` on {n:0} is
    # true -> stop after exactly one run. (A `while`-style pre-check would have committed the seed
    # with 0 runs, giving {n:1}.)
    result = run_flow(load_flow(UNTIL_SEED_TRUE), {})
    assert result.output == {"n": 0}


def test_loop_budget_exceeded_in_step_fails_run(monkeypatch):
    # The node-budget guard inside `_apply_grow` raises a RuntimeError; when a fresh driver's
    # grow (iteration >= 1, driven by `loop~1` off the body-END wiring) trips it, it must become a
    # failed run, not an uncaught escape. The COUNTER flow adds ~3 body nodes + 1 driver node per
    # iteration off a base of 3, so a budget of 6 lets iteration 0 grow and trips a later iteration
    # inside `_apply_grow` — exercising the grow boundary specifically.
    import agent_composer.runtime.engine as engine_mod

    monkeypatch.setattr(engine_mod, "MAX_TOTAL_NODES", 6)
    result = run_flow(load_flow(COUNTER), {})
    assert result.status != "succeeded"
    assert "node budget" in (result.error or "")


# `times: N` is a fixed count: exactly N body runs, no predicate. Seed n=10, times 3 ->
# 10 -> 9 -> 8 -> 7 (three body runs), then the driver at k=3 stops (k >= N).
TIMES_3 = """
id: t3
name: t3
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
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 10
    times: 3
output: ${loop.output}
"""


# `until:` do-while over a countdown: seed n=3, stop once `${n} <= 0` is true. The body runs
# 3 -> 2 -> 1 -> 0, then the post-check `${n} <= 0` on {n:0} is true and terminates at {n:0}.
UNTIL_COUNTDOWN = """
id: uc
name: uc
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
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 3
    until: ${n} <= 0
    max: 10
output: ${loop.output}
"""


def test_times_runs_exactly_n():
    result = run_flow(load_flow(TIMES_3), {})   # {n: 10}, times: 3
    assert result.output == {"n": 7}            # 10 -> 9 -> 8 -> 7 (exactly 3 body runs)


def test_until_stops_when_predicate_becomes_true():
    result = run_flow(load_flow(UNTIL_COUNTDOWN), {})   # {n: 3}, until: ${n} <= 0, max: 10
    assert result.output == {"n": 0}            # 3 -> 2 -> 1 -> 0, then n<=0 true -> stop


def test_prune_drops_all_live_overlay_traces():
    """The generic `_prune(ids)` clears EVERY live-overlay trace of the named id-set — an iteration
    `#k`'s nodes/edges, sm state (node_state/edge_state/executing), pool entries — while leaving the
    durable loop `GrowRecord` ledger untouched and the spawner id (no `#k` prefix) alone.

    Deterministic grown-#0 setup: drive `LOOP_CHAT` to its first pause. The body's `human_input`
    leaf parks the run with iteration #0 fully grown (its `#0/` nodes registered, the START seed
    committed to `pool.store`). In the self-respawn model the body-END filler `loop#0/__end__`
    carries NO `commit_as` — it feeds the fresh driver `loop~1` by plain wiring, not a loop-back
    redirect. The `#0/` overlay is live and un-pruned, the exact state prune must remove.
    """
    loaded = load_flow(LOOP_CHAT)
    r = run_flow(loaded, {})
    assert r.status == "paused"
    engine = r.engine
    spawner = "loop"
    prefix = f"{spawner}#0/"

    # Precondition: the grown #0 overlay is actually present in every registry prune touches,
    # so a clean assert below is meaningful (not vacuously true on an empty overlay).
    assert any(n.startswith(prefix) for n in engine.flow.nodes)
    assert any(n.startswith(prefix) for n in engine.pool.store)
    assert any(n.startswith(prefix) for n in engine.sm.node_state)
    # The CONTINUE body-END filler carries NO commit_as (it feeds the next driver `loop~1` by
    # plain wiring; only the STOP arm commits, under the origin `loop`).
    assert engine.flow.nodes[f"{spawner}#0/__end__"].commit_as is None
    # And the fresh next-iteration driver `loop~1` was spliced (bounded live overlay {L, body_0, L~1}).
    assert "loop~1" in engine.flow.nodes

    # Snapshot the durable ledger BEFORE pruning — it must survive untouched.
    ledger_len = len(engine.expansions)
    rec = engine._origin_record[spawner]
    # The loop GrowRecord's seed is the LIVE iteration's `(record, index)` pair.
    record_before, index_before = rec.seed

    ids = frozenset(n for n in engine.flow.nodes if n.startswith(prefix))
    engine._prune(ids)

    # Every per-id registry is clean for the `#0/` prefix.
    assert not any(n.startswith(prefix) for n in engine.flow.nodes)
    assert not any(n.startswith(prefix) for n in engine.pool.store)
    assert not any(k.startswith(prefix) for k in engine.depth)
    assert not any(k.startswith(prefix) for k in engine._spawner_expansion)
    assert not any(n.startswith(prefix) for n in engine.sm.node_state)
    assert not any(n.startswith(prefix) for n in engine.sm.executing)
    # No surviving edge references a pruned id (from OR to), and no edge_state keys one.
    assert not any(e.from_.startswith(prefix) or e.to.startswith(prefix)
                   for e in engine.flow.edges)

    # The durable GrowRecord ledger is UNTOUCHED — replay needs it intact.
    assert len(engine.expansions) == ledger_len
    assert engine._origin_record[spawner] is rec
    assert rec.seed == (dict(record_before), index_before)

    # The spawner id itself (no `#k` prefix) is NOT pruned — it is the durable replay spawner.
    assert spawner in engine.flow.nodes


# A `times: 20` countdown far exceeds a lowered 40-node budget when NOTHING is pruned
# (each iteration adds ~3 body nodes + 1 driver node off a small base). With the self-respawn
# model's per-iteration self-prune (each driver retires the previous body + itself), only ~one
# iteration is resident at a time, so the run stays bounded and reaches its terminal. Drives
# `FlowEngine` directly because `run_flow(...).engine` is None on a succeeded result (mirrors
# `test_durable_replay.py`).
TIMES_20 = """
id: t20
name: t20
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
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 30
    times: 20
output: ${loop.output}
"""


def test_long_loop_stays_within_node_budget(monkeypatch):
    import agent_composer.runtime.engine as eng
    from agent_composer.runtime.engine import FlowEngine

    monkeypatch.setattr(eng, "MAX_TOTAL_NODES", 40)   # a 20-iteration loop blows this un-pruned
    engine = FlowEngine(load_flow(TIMES_20).compiled, run_inputs={})
    terminal = list(engine.run())[-1]
    from agent_composer.events import RunSucceeded

    assert isinstance(terminal, RunSucceeded)
    # 20 body runs from n=30 -> n=10, and only ~one iteration resident at the end:
    assert engine.flow.nodes                          # sanity
    assert len(engine.flow.nodes) < 40                # bounded; un-pruned this would exceed 40
    assert terminal.output == {"n": 10}               # 30 - 20 body runs = 10


def test_terminate_leaves_exactly_one_dead_final_body_and_driver():
    """The STOP arm commits under `L` and does NOT prune the final body `L#(k-1)/` NOR the STOP
    driver clone `L~k` — a bounded one-time residue (design risk #5 / D4). Self-prune happens ONLY
    in the CONTINUE `Grow` arm; the STOP driver returns `Output` and grows nothing, so it and the
    last body it read are left resident.

    For `times: 3` on `n=10` the bodies run at k=0,1,2 and STOP fires on the fresh driver `loop~3`
    reading `loop#2/__end__`. So the residue is EXACTLY: one dead body namespace `loop#2/` AND one
    dead driver clone `loop~3`. The committed origin `loop` stays (legitimate result node)."""
    from agent_composer.events import RunSucceeded
    from agent_composer.runtime.engine import FlowEngine

    engine = FlowEngine(load_flow(TIMES_3).compiled, run_inputs={})   # n=10, times 3
    terminal = list(engine.run())[-1]
    assert isinstance(terminal, RunSucceeded)
    assert terminal.output == {"n": 7}
    # (a) residual body namespaces: exactly one, the final iteration's `loop#2/`.
    body_ns = {n.split("/", 1)[0] for n in engine.flow.nodes if n.startswith("loop#")}
    assert body_ns == {"loop#2"}                          # exactly the final body namespace
    # (b) residual driver clones: exactly one, the STOP driver `loop~3` (the origin `loop` is NOT a
    #     `loop~<k>` clone — it is the compiled result node and is excluded by the `~` filter).
    driver_clones = {n for n in engine.flow.nodes if n.startswith("loop~")}
    assert driver_clones == {"loop~3"}                    # exactly the STOP driver clone
    assert "loop" in engine.flow.nodes                    # committed origin stays (not residue)

