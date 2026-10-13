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

from collections.abc import Callable

import pytest

from yian.compile.expand import build_grow
from yian.compose.loader import load_flow
from yian.run import resume_flow, run_flow
from yian.nodes.contracts import PartialRecordContract
from yian.typesys import ValueKind
from tests.engine._graph_builder import _engine


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
    assert "exceeded max" in result.failure.message
    # And confirm the located cause type reaches the RunFailed event.
    cause_types = {
        getattr(getattr(e, "failure", None), "cause_type", None)
        for e in result.events
    }
    assert "LoopMaxExceeded" in cause_types


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


NESTED_CALL_LOOP_MAXED = """
id: nested-call-loop-maxed
name: nested_call_loop_maxed
defs:
  keep:
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
  runaway:
    nodes:
      repeat:
        kind: loop
        call: keep
        input:
          n: 0
        while: "true"
        max: 1
    output: ${repeat.output}
nodes:
  outer:
    kind: call
    call: runaway
output: ${outer.output}
"""


def test_nested_call_loop_respawn_failure_keeps_each_authored_origin_once():
    """A respawned LOOP driver retains CALL ancestry without appending itself twice."""
    loaded = load_flow(NESTED_CALL_LOOP_MAXED)

    result = run_flow(loaded, {})

    outer = loaded.compiled.nodes["outer"]
    repeat = outer.child.nodes["repeat"]
    assert result.status != "succeeded"
    assert result.failure.node_id == "outer/repeat~1"
    assert result.failure.cause_type == "LoopMaxExceeded"
    assert result.failure.trace == (outer.origin, repeat.origin)


# The chat scenario: a body blocks each turn on a `human_input` leaf, folds the
# returned message into the carried {messages, exited} record, and loops until "bye".
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


def test_loop_body_blocks_for_each_human_turn():
    loaded = load_flow(LOOP_CHAT)
    answers = iter(["hi", "bye"])
    result = run_flow(loaded, {}, tools={"ask_user": lambda **request: next(answers)})

    assert result.status == "succeeded", result.failure
    assert result.output == {"messages": ["hi", "bye"], "exited": True}


DIRECT_EXIT_LOOP = """
id: direct-exit-chat
name: direct_exit_chat
input:
  initial_exited: bool
defs:
  chat_turn:
    input:
      messages: list[str]
      exited: bool
    nodes:
      ask:
        kind: human_input
        prompt: "your message"
        output: str
    output:
      messages: ${input.messages}
      exited: ${ask.output == "/exit"}
nodes:
  loop:
    kind: loop
    call: chat_turn
    input:
      messages: []
      exited: ${input.initial_exited}
    while: not ${exited}
    max: 10
output: ${loop.output}
"""


def test_loop_body_can_compute_carried_boolean_in_its_output_binding():
    """A comparison result is statically bool and satisfies LOOP's type contract."""
    loaded = load_flow(DIRECT_EXIT_LOOP)
    prompts: list[str] = []

    def answer(**request):
        """Record the single authored prompt and return the exit command."""
        prompts.append(request["question"])
        return "/exit"

    result = run_flow(loaded, {"initial_exited": False}, tools={"ask_user": answer})

    body_type = loaded.compiled.nodes["loop"].output_contract.type
    assert body_type.fields["exited"].kind is ValueKind.BOOLEAN
    assert result.status == "succeeded", result.failure
    assert result.output == {"messages": [], "exited": True}
    assert prompts == ["your message"]


OPAQUE_ARITHMETIC_LOOP = """
id: opaque-arithmetic-loop
name: opaque_arithmetic_loop
input:
  initial_n: int
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes: {}
    output:
      n: ${input.n + 1}
      exited: ${input.exited}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: ${input.initial_n}
      exited: false
    times: 1
output: ${loop.output}
"""


def test_loop_keeps_declared_record_shape_when_one_field_type_is_opaque():
    """Unknown computation stays lenient without becoming an actual None field."""
    loaded = load_flow(OPAQUE_ARITHMETIC_LOOP)

    loop = loaded.compiled.nodes["loop"]
    assert type(loop.output_contract) is PartialRecordContract
    assert loop.output_contract.names == ("n", "exited")
    assert loop.output_contract.fields[0] == ("n", None)
    assert loop.output_contract.fields[1][1] is not None
    assert loop.respawn(1).output_contract is loop.output_contract
    assert not hasattr(loop, "output_field_contracts")

    result = run_flow(loaded, {"initial_n": 1})

    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2, "exited": False}


OPAQUE_TURN_ZERO_LOOP = """
id: opaque-turn-zero
name: opaque_turn_zero
input:
  initial_n: str
defs:
  body:
    input:
      n: int
      note: str
    nodes: {}
    output:
      n: ${input.n}
      note: ${input.note + "!"}
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: ${input.initial_n + ""}
      note: seed
    while: "false"
    max: 1
output: ${loop.output}
"""


def test_partial_loop_contract_rejects_invalid_turn_zero_seed():
    """Known body fields remain enforced when an opaque sibling hides the aggregate."""
    result = run_flow(load_flow(OPAQUE_TURN_ZERO_LOOP), {"initial_n": "not-an-int"})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert "int" in result.failure.message


NESTED_OPAQUE_TURN_ZERO_LOOP = """
id: nested-opaque-turn-zero
name: nested_opaque_turn_zero
input:
  initial_n: str
defs:
  inner_body:
    input:
      n: int
      note: str
    nodes: {}
    output:
      n: ${input.n}
      note: ${input.note + "!"}
  outer_body:
    input:
      n: int
      note: str
    nodes:
      inner:
        kind: loop
        call: inner_body
        input:
          n: ${input.n}
          note: ${input.note}
        while: "false"
        max: 1
    output:
      n: ${inner.output.n}
      note: ${inner.output.note}
nodes:
  outer:
    kind: loop
    call: outer_body
    input:
      n: ${input.initial_n + ""}
      note: seed
    while: "false"
    max: 1
output: ${outer.output}
"""


def test_partial_loop_contract_survives_nested_child_projection():
    """A nested LOOP cannot drop a proven sibling before turn-zero validation."""
    loaded = load_flow(NESTED_OPAQUE_TURN_ZERO_LOOP)

    outer = loaded.compiled.nodes["outer"]
    assert type(outer.output_contract) is PartialRecordContract
    assert outer.output_contract.fields[0][0] == "n"
    assert outer.output_contract.fields[0][1] is not None

    result = run_flow(loaded, {"initial_n": "not-an-int"})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert "int" in result.failure.message


PASSTHROUGH_OPAQUE_TURN_ZERO_LOOP = """
id: passthrough-opaque-turn-zero
name: passthrough_opaque_turn_zero
input:
  initial_n: str
defs:
  inner_body:
    input:
      n: int
      note: str
    nodes: {}
    output:
      n: ${input.n}
      note: ${input.note + "!"}
  middle:
    input:
      n: int
      note: str
    nodes:
      inner:
        kind: loop
        call: inner_body
        input:
          n: ${input.n}
          note: ${input.note}
        while: "false"
        max: 1
    output: ${inner.output}
  outer_body:
    input:
      n: int
      note: str
    nodes:
      middle_call:
        kind: call
        call: middle
        input:
          n: ${input.n}
          note: ${input.note}
    output:
      n: ${middle_call.output.n}
      note: ${middle_call.output.note}
nodes:
  outer:
    kind: loop
    call: outer_body
    input:
      n: ${input.initial_n + ""}
      note: seed
    while: "false"
    max: 1
output: ${outer.output}
"""


def test_partial_loop_contract_survives_transparent_call_passthrough():
    """Whole-value child and CALL boundaries preserve the partial contract arm."""
    loaded = load_flow(PASSTHROUGH_OPAQUE_TURN_ZERO_LOOP)

    outer = loaded.compiled.nodes["outer"]
    assert type(outer.output_contract) is PartialRecordContract
    assert outer.output_contract.fields[0][1] is not None

    result = run_flow(loaded, {"initial_n": "not-an-int"})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert "int" in result.failure.message


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
    assert result.status == "succeeded", result.failure
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
    assert "division by zero" in result.failure.message


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


# The seed `seen: null` is a placeholder the first body fills in; `'b' in seen` cannot be
# evaluated on it. `until:` must not consult the predicate before the first body.
UNTIL_SEED_UNEVALUABLE = """
id: usu
name: usu
defs:
  step:
    input:
      seen: Optional[list[str]]
      n: int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          seen: Optional[list[str]]
          n: int
        code: |
          return {"seen": ["a", "b"][: inputs["n"] + 1], "n": inputs["n"] + 1}
    output: ${bump.output}
nodes:
  repeat:
    kind: loop
    call: step
    until: "'b' in seen"
    max: 5
    input:
      seen: ${null}
      n: 0
output: ${repeat.output}
"""


def test_until_does_not_evaluate_predicate_on_seed():
    result = run_flow(load_flow(UNTIL_SEED_UNEVALUABLE), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"seen": ["a", "b"], "n": 2}


def test_loop_budget_exceeded_in_step_fails_run(monkeypatch):
    # The node-budget guard inside `_apply_grow` raises a RuntimeError; when a fresh driver's
    # grow (iteration >= 1, driven by `loop~1` off the body-END wiring) trips it, it must become a
    # failed run, not an uncaught escape. The COUNTER flow adds ~3 body nodes + 1 driver node per
    # iteration off a base of 3, so a budget of 6 lets iteration 0 grow and trips a later iteration
    # inside `_apply_grow` — exercising the grow boundary specifically.
    import yian.runtime.state as state_module

    monkeypatch.setattr(state_module, "MAX_TOTAL_NODES", 6)
    result = run_flow(load_flow(COUNTER), {})
    assert result.status != "succeeded"
    assert "node budget" in result.failure.message


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
    import yian.runtime.state as state_module

    monkeypatch.setattr(state_module, "MAX_TOTAL_NODES", 40)
    engine = _engine(
        load_flow(TIMES_20).compiled,
        run_inputs={},
        grow_builder=build_grow,
    )
    terminal = list(engine.run())[-1]
    from yian.events import RunSucceeded

    assert isinstance(terminal, RunSucceeded)
    # 20 body runs from n=30 -> n=10, and only ~one iteration resident at the end:
    assert engine.state.flow.nodes
    assert len(engine.state.flow.nodes) < 40
    assert terminal.output == {"n": 10}               # 30 - 20 body runs = 10


def test_terminate_leaves_exactly_one_dead_final_body_and_driver():
    """The STOP arm commits under `L` and does NOT prune the final body `L#(k-1)/` NOR the STOP
    driver clone `L~k` — a bounded one-time termination residue. Self-prune happens ONLY
    in the CONTINUE `Grow` arm; the STOP driver returns `Output` and grows nothing, so it and the
    last body it read are left resident.

    For `times: 3` on `n=10` the bodies run at k=0,1,2 and STOP fires on the fresh driver `loop~3`
    reading `loop#2/__end__`. So the residue is EXACTLY: one dead body namespace `loop#2/` AND one
    dead driver clone `loop~3`. The committed origin `loop` stays (legitimate result node)."""
    from yian.events import RunSucceeded
    engine = _engine(
        load_flow(TIMES_3).compiled,
        run_inputs={},
        grow_builder=build_grow,
    )   # n=10, times 3
    terminal = list(engine.run())[-1]
    assert isinstance(terminal, RunSucceeded)
    assert terminal.output == {"n": 7}
    # (a) residual body namespaces: exactly one, the final iteration's `loop#2/`.
    body_ns = {
        n.split("/", 1)[0]
        for n in engine.state.flow.nodes
        if n.startswith("loop#")
    }
    assert body_ns == {"loop#2"}                          # exactly the final body namespace
    # (b) residual driver clones: exactly one, the STOP driver `loop~3` (the origin `loop` is NOT a
    #     `loop~<k>` clone — it is the compiled result node and is excluded by the `~` filter).
    driver_clones = {n for n in engine.state.flow.nodes if n.startswith("loop~")}
    assert driver_clones == {"loop~3"}                    # exactly the STOP driver clone
    assert "loop" in engine.state.flow.nodes              # committed origin stays


def test_while_loop_without_max_runs_until_predicate_ends_it():
    result = run_flow(load_flow(COUNTER.replace("    max: 10\n", "")), {})
    assert result.status == "succeeded"
    assert result.output == {"n": 3, "exited": True}


def test_until_loop_without_max_runs_until_predicate_ends_it():
    result = run_flow(load_flow(UNTIL_COUNTDOWN.replace("    max: 10\n", "")), {})
    assert result.status == "succeeded"
    assert result.output == {"n": 0}


def test_loop_without_max_resumes_from_its_log(tmp_path):
    src = COUNTER.replace("    max: 10\n", "")
    log = tmp_path / "run.jsonl"
    first = run_flow(load_flow(src), {}, log=log)
    assert first.status == "succeeded"
    # Cut the log mid-run so resume replays the recorded drivers and runs the rest live.
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join(lines[: len(lines) // 2]))
    resumed = resume_flow(load_flow(src), log=log)
    assert resumed.status == "succeeded"
    assert resumed.output == first.output


# A partial LOOP body returns a subset of the carried record; omitted fields keep the
# previous iteration's value. This chat-shaped body reads the whole record but returns
# only `{n, done}`, so `person` and `talking_to` ride through every turn unchanged.
PARTIAL_CHAT = """
id: partial-chat
name: partial_chat
typedefs:
  Person:
    name: str
input:
  person: Person
  talking_to: str
defs:
  turn:
    input:
      person: Person
      talking_to: str
      n: int
    nodes:
      step:
        kind: code
        code: |
          assert inputs["person"]["name"] == "Ada"
          assert inputs["talking_to"] == "Bob"
          n = inputs["n"] + 1
          return {"n": n, "done": n >= 3}
        input:
          person: ${input.person}
          talking_to: ${input.talking_to}
          n: ${input.n}
        output:
          n: int
          done: bool
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      person: ${input.person}
      talking_to: ${input.talking_to}
      n: 0
      done: false
    while: not done
output: ${loop.output}
"""

PARTIAL_CHAT_INPUTS = {"person": {"name": "Ada"}, "talking_to": "Bob"}


def test_partial_body_keeps_omitted_fields_across_turns():
    result = run_flow(load_flow(PARTIAL_CHAT), PARTIAL_CHAT_INPUTS)
    assert result.status == "succeeded", result.failure
    assert result.output == {
        "person": {"name": "Ada"},
        "talking_to": "Bob",
        "n": 3,
        "done": True,
    }


# A one-field body `output: {count: ...}` returns only `count`; `label` keeps its seed.
ONE_FIELD_BODY = """
id: one-field-body
name: one_field_body
nodes:
  loop:
    kind: loop
    times: 3
    input:
      count: 0
      label: x
    nodes:
      bump:
        kind: code
        input:
          n: ${input.count}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      count: ${bump.output}
output: ${loop.output}
"""


def test_one_field_body_keeps_the_other_carried_fields():
    loaded = load_flow(ONE_FIELD_BODY)
    assert loaded.compiled.nodes["loop"].returned_fields == ("count",)
    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"count": 3, "label": "x"}


PARTIAL_NOTE = """
id: partial-note
name: partial_note
input:
  note0: Optional[str]
defs:
  clear_note:
    input:
      n: int
    nodes:
      step:
        kind: code
        code: |
          return {"n": inputs["n"] + 1, "note": None}
        input:
          n: ${input.n}
        output:
          n: int
          note: Optional[str]
    output: ${step.output}
  bump_only:
    input:
      n: int
    nodes:
      step:
        kind: code
        code: |
          return {"n": inputs["n"] + 1}
        input:
          n: ${input.n}
        output:
          n: int
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: BODY
    input:
      n: 0
      note: ${input.note0}
    times: 2
output: ${loop.output}
"""


def test_partial_body_explicit_null_sets_nullable_field():
    """An explicit null clears a nullable carried field; omitting it keeps the seed."""
    cleared = run_flow(load_flow(PARTIAL_NOTE.replace("BODY", "clear_note")), {"note0": "seed"})
    assert cleared.status == "succeeded", cleared.failure
    assert cleared.output == {"n": 2, "note": None}

    kept = run_flow(load_flow(PARTIAL_NOTE.replace("BODY", "bump_only")), {"note0": "seed"})
    assert kept.status == "succeeded", kept.failure
    assert kept.output == {"n": 2, "note": "seed"}


def _with_loop_asserts(*asserts: str) -> str:
    lines = "".join(f'      - "{a}"\n' for a in asserts)
    return PARTIAL_CHAT.replace(
        "    while: not done\n", f"    while: not done\n    asserts:\n{lines}"
    )


def test_partial_loop_asserts_see_the_merged_record():
    post_ok = run_flow(
        load_flow(_with_loop_asserts("${output.person.name} == 'Ada'")), PARTIAL_CHAT_INPUTS
    )
    assert post_ok.status == "succeeded", post_ok.failure
    assert post_ok.output["person"] == {"name": "Ada"}

    post_bad = run_flow(
        load_flow(_with_loop_asserts("${output.person.name} == 'Bob'")), PARTIAL_CHAT_INPUTS
    )
    assert post_bad.status == "failed"
    assert post_bad.failure.cause_type == "NodeAssertFailed"
    assert "post-assert failed" in post_bad.failure.message

    # A PRE assert reads the carried inputs; each respawned driver re-checks it against the
    # merged record, where the omitted `person` still holds its seed.
    pre_ok = run_flow(
        load_flow(_with_loop_asserts("${person.name} == 'Ada'")), PARTIAL_CHAT_INPUTS
    )
    assert pre_ok.status == "succeeded", pre_ok.failure
    assert pre_ok.output["n"] == 3
    # Respawned drivers do run PRE asserts: one on a returned field fails mid-loop.
    pre_bad = run_flow(load_flow(_with_loop_asserts("${n} < 2")), PARTIAL_CHAT_INPUTS)
    assert pre_bad.status == "failed"
    assert pre_bad.failure.node_id == "loop~2"
    assert "pre-assert failed" in pre_bad.failure.message


def test_partial_loop_zero_iterations_commits_seed():
    src = PARTIAL_CHAT.replace("      done: false\n", "      done: true\n")
    result = run_flow(load_flow(src), PARTIAL_CHAT_INPUTS)
    assert result.status == "succeeded", result.failure
    assert result.output == {
        "person": {"name": "Ada"},
        "talking_to": "Bob",
        "n": 0,
        "done": True,
    }


def test_partial_until_and_times_loops():
    until = run_flow(
        load_flow(PARTIAL_CHAT.replace("while: not done", "until: done")), PARTIAL_CHAT_INPUTS
    )
    assert until.status == "succeeded", until.failure
    assert until.output == {
        "person": {"name": "Ada"},
        "talking_to": "Bob",
        "n": 3,
        "done": True,
    }

    times = run_flow(
        load_flow(PARTIAL_CHAT.replace("while: not done", "times: 2")), PARTIAL_CHAT_INPUTS
    )
    assert times.status == "succeeded", times.failure
    assert times.output == {
        "person": {"name": "Ada"},
        "talking_to": "Bob",
        "n": 2,
        "done": False,
    }


PARTIAL_IN_CALL = """
id: partial-in-call
name: partial_in_call
defs:
  bump:
    input:
      n: int
    nodes:
      step:
        kind: code
        code: |
          return {"n": inputs["n"] + 1}
        input:
          n: ${input.n}
        output:
          n: int
    output: ${step.output}
  counter:
    input:
      tag: str
    nodes:
      loop:
        kind: loop
        call: bump
        input:
          n: 0
          tag: ${input.tag}
        times: 2
    output: ${loop.output}
nodes:
  outer:
    kind: call
    call: counter
    input:
      tag: kept
  mapped:
    kind: map
    call: counter
    over: ${["a", "b"]}
    input:
      tag: ${item}
output:
  outer: ${outer.output}
  mapped: ${mapped.output}
"""


def test_partial_loop_inside_call_body():
    result = run_flow(load_flow(PARTIAL_IN_CALL), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {
        "outer": {"n": 2, "tag": "kept"},
        "mapped": [{"n": 2, "tag": "a"}, {"n": 2, "tag": "b"}],
    }


def test_partial_loop_resumes_from_its_log(tmp_path):
    log = tmp_path / "run.jsonl"
    first = run_flow(load_flow(PARTIAL_CHAT), PARTIAL_CHAT_INPUTS, log=log)
    assert first.status == "succeeded", first.failure
    # Cut the log mid-run so resume replays the recorded drivers and runs the rest live.
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join(lines[: len(lines) // 2]))
    resumed = resume_flow(load_flow(PARTIAL_CHAT), log=log)
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output


_ONE = """
  one:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output: ${step.output}
"""

_LOOP = """
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 0
    times: TIMES
output: ${loop.output}
"""

_CALL_BODY = """
id: nested_call
name: nested_call
defs:""" + _ONE + """
  body:
    input:
      n: int
    nodes:
      c:
        kind: call
        call: one
        input:
          n: ${input.n}
    output:
      n: ${c.output}
""" + _LOOP

_MAP_BODY = """
id: nested_map
name: nested_map
defs:""" + _ONE + """
  body:
    input:
      n: int
    nodes:
      m:
        kind: map
        call: one
        over: ${[input.n, input.n]}
        input:
          n: ${item}
      pick:
        kind: code
        input:
          xs: ${m.output}
        output: int
        code: |
          return inputs["xs"][0]
    output:
      n: ${pick.output}
""" + _LOOP

_LOOP_BODY = """
id: nested_loop
name: nested_loop
defs:""" + _ONE + """
  inc:
    input:
      n: int
    nodes:
      c:
        kind: call
        call: one
        input:
          n: ${input.n}
    output:
      n: ${c.output}
  body:
    input:
      n: int
    nodes:
      inner:
        kind: loop
        call: inc
        input:
          n: ${input.n}
        times: 3
    output:
      n: ${inner.output.n}
""" + _LOOP

_CASE_BODY = """
id: nested_case
name: nested_case
defs:
  up:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output: ${step.output}
  up2:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 2
    output: ${step.output}
  body:
    input:
      n: int
    nodes:
      pick:
        kind: case
        cases:
          - when: "input.n % 2 == 0"
            then: call(up, n=${input.n})
        else: call(up2, n=${input.n})
    output:
      n: ${pick.output}
""" + _LOOP


def _case_total(times: int) -> int:
    """Return the CASE body's carried value after ``times`` iterations from zero."""
    n = 0
    for _ in range(times):
        n += 1 if n % 2 == 0 else 2
    return n


# Each body grows nodes inside its iteration, so pruning only the direct body ids leaks them.
_NESTED_BODIES: dict[str, tuple[str, Callable[[int], int]]] = {
    "call": (_CALL_BODY, lambda times: times),
    "map": (_MAP_BODY, lambda times: times),
    "nested": (_LOOP_BODY, lambda times: 3 * times),
    "case": (_CASE_BODY, _case_total),
}


def _finish_nested(shape, times, **engine_kwargs):
    """Run one nested-growth body ``times`` iterations and return the finished engine."""
    from yian.events import RunSucceeded
    source, expected = _NESTED_BODIES[shape]
    engine = _engine(load_flow(source.replace("TIMES", str(times))).compiled,
                     run_inputs={}, grow_builder=build_grow, **engine_kwargs)
    terminal = list(engine.run())[-1]
    assert isinstance(terminal, RunSucceeded), terminal
    assert terminal.output == {"n": expected(times)}
    return engine


@pytest.mark.parametrize("num_workers", [0, 2], ids=["serial", "pooled"])
@pytest.mark.parametrize("shape", sorted(_NESTED_BODIES))
def test_loop_with_nested_growth_keeps_one_iteration_resident(shape, num_workers, monkeypatch):
    """Pruning an iteration also retires the nodes its nested CALL/MAP/LOOP grew."""
    import yian.runtime.state as state_module

    resident = len(_finish_nested(shape, 3, num_workers=num_workers).state.flow.nodes)
    # A tight budget fails mid-run if any earlier iteration's nested nodes leak.
    monkeypatch.setattr(state_module, "MAX_TOTAL_NODES", resident + 20)
    engine = _finish_nested(shape, 20, num_workers=num_workers)
    assert len(engine.state.flow.nodes) == resident


@pytest.mark.parametrize("shape", sorted(_NESTED_BODIES))
def test_loop_with_nested_growth_keeps_trace_indexes_bounded(shape):
    """A learnable run's current-name trace indexes shrink with the live graph."""
    small = _finish_nested(shape, 3, learnable=True).state.trace
    large = _finish_nested(shape, 20, learnable=True).state.trace
    assert len(large.current_writers) == len(small.current_writers)
    assert len(large.pending_connections) == len(small.pending_connections)


def test_nested_growth_loop_resumes_from_a_cut_log(tmp_path):
    """A resumed nested-growth LOOP finishes, and its log replays to one resident iteration."""
    import yian.run as run_module
    from yian.runtime.log import JsonlRunLog
    src = _NESTED_BODIES["nested"][0].replace("TIMES", "6")
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(src), {}, log=path)
    assert first.status == "succeeded", first.failure
    # Cut the log mid-run so resume replays the recorded growth and runs the rest live.
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[: len(lines) // 2]))
    resumed = resume_flow(load_flow(src), log=path)
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output
    with JsonlRunLog.resume(path) as log:
        state = run_module._replay_run_log(load_flow(src), log)
    assert len(state.flow.nodes) == len(_finish_nested("nested", 2).state.flow.nodes)


def test_loop_growth_record_keeps_only_the_requested_prune_ids():
    """The log records the body's direct ids and the driver; nested ids are derived."""
    from yian.runtime.log import NodeExpandedRecord
    engine = _finish_nested("call", 3)
    pruned = [e.record.prune_ids for e in engine.log
              if type(e.record) is NodeExpandedRecord and e.record.spawner_id == "loop~2"]
    assert pruned == [("loop#1/__end__", "loop#1/__start__", "loop#1/c", "loop~2")]


def test_self_pruning_spawner_hands_its_fragment_to_its_owner():
    """A driver that prunes itself passes the nodes it grew to its own owner."""
    owners = _finish_nested("call", 3).state.growth_owner
    assert set(owners.values()) == {"loop", "loop#2/c"}
    assert owners["loop~3"] == "loop"


# The child's unread side chain can still be waiting when the next driver prunes its iteration.
_DANGLING_SIDE = """
id: dangling_side
name: dangling_side
defs:
  one:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      side:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] * 2
      side2:
        kind: code
        input:
          n: ${side.output}
        output: int
        code: |
          return inputs["n"] * 2
      side3:
        kind: code
        input:
          n: ${side2.output}
        output: int
        code: |
          return inputs["n"] * 2
      side4:
        kind: code
        input:
          n: ${side3.output}
        output: int
        code: |
          return inputs["n"] * 2
    output: ${step.output}
  body:
    input:
      n: int
    nodes:
      c:
        kind: call
        call: one
        input:
          n: ${input.n}
    output:
      n: ${c.output}
""" + _LOOP.replace("TIMES", "4")


@pytest.mark.parametrize("num_workers", [0, 3], ids=["serial", "pooled"])
def test_loop_keeps_a_nested_subtree_with_unfinished_work(num_workers):
    """A nested node still waiting when its iteration is pruned keeps its subtree alive
    until a later growth finds it settled."""
    from yian.events import RunSucceeded
    engine = _engine(load_flow(_DANGLING_SIDE).compiled, run_inputs={},
                     grow_builder=build_grow, num_workers=num_workers)
    terminal = list(engine.run())[-1]
    assert isinstance(terminal, RunSucceeded), terminal
    assert terminal.output == {"n": 4}
    # Every owner entry names live nodes.
    owners = engine.state.growth_owner
    assert all(k in engine.state.flow.nodes and v in engine.state.flow.nodes
               for k, v in owners.items())
    # Serially, each iteration's side chain is still waiting when the next driver prunes
    # it; later drivers retire it, so only the last iteration stays. Pooled order varies.
    if num_workers == 0:
        assert {n.split("/")[0] for n in engine.state.flow.nodes} == {
            "__start__", "__end__", "loop", "loop#3", "loop~4"}
        assert engine.state.deferred_prune == frozenset()


# The body output reads only `bump`, so the next driver can start while the side chain
# still runs; `side2` also reads `bump`, which must then stay live with it.
_DIRECT_SIDE = """
id: direct_side
name: direct_side
defs:
  body:
    input:
      n: int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      side:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          import time
          time.sleep(SLEEP)
          return inputs["n"]
      side2:
        kind: code
        input:
          n: ${side.output}
          m: ${bump.output}
        output: int
        code: |
          return inputs["n"] + inputs["m"]
    output:
      n: ${bump.output}
""" + _LOOP.replace("TIMES", "4")


@pytest.mark.parametrize("sleep", ["0", "0.3"])
@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_loop_keeps_an_unfinished_body_node_live(num_workers, sleep):
    """A body node the output doesn't read may finish after the next driver starts."""
    result = run_flow(load_flow(_DIRECT_SIDE.replace("SLEEP", sleep)), {},
                      num_workers=num_workers)
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 4}


def test_loop_retires_a_kept_body_node_once_settled(monkeypatch):
    """A body node kept while unfinished is retired by a later prune once it has settled."""
    from yian.events import RunSucceeded
    import yian.runtime.state as state_module

    def finish(times):
        source = _DIRECT_SIDE.replace("SLEEP", "0.05").replace("times: 4", f"times: {times}")
        engine = _engine(load_flow(source).compiled, run_inputs={},
                         grow_builder=build_grow, num_workers=4)
        terminal = list(engine.run())[-1]
        assert isinstance(terminal, RunSucceeded), terminal
        assert terminal.output == {"n": times}
        return engine

    resident = len(finish(3).state.flow.nodes)
    # A few iterations' side chains may linger at once; leaking every one would add ~80.
    monkeypatch.setattr(state_module, "MAX_TOTAL_NODES", resident + 40)
    engine = finish(20)
    assert len(engine.state.flow.nodes) <= resident + 40


def test_loop_with_a_kept_body_node_resumes_from_a_cut_log(tmp_path):
    """Replay re-derives which body nodes were kept, so a resumed run finishes."""
    import yian.run as run_module
    from yian.runtime.log import JsonlRunLog
    flow = load_flow(_DIRECT_SIDE.replace("SLEEP", "0.2"))
    path = tmp_path / "run.jsonl"
    first = run_flow(flow, {}, log=path, num_workers=4)
    assert first.status == "succeeded", first.failure
    with JsonlRunLog.resume(path) as log:
        assert run_module._replay_run_log(flow, log).deferred_prune
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[: len(lines) // 2]))
    resumed = resume_flow(flow, log=path, num_workers=4)
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output


# The unread `note` CALL is still waiting on `slow` when the next driver prunes its
# iteration, then grows its own child while deferred.
_DEFERRED_CALL = """
id: deferred_call
name: deferred_call
defs:
  echo:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"]
    output: ${step.output}
  body:
    input:
      n: int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      slow:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          import time
          time.sleep(0.2)
          return inputs["n"]
      note:
        kind: call
        call: echo
        input:
          n: ${slow.output}
    output:
      n: ${bump.output}
""" + _LOOP.replace("TIMES", "3")


def test_loop_body_call_grows_after_its_iteration_was_pruned():
    """A kept body CALL can still expand its child after the next driver started."""
    result = run_flow(load_flow(_DEFERRED_CALL), {}, num_workers=4)
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 3}
