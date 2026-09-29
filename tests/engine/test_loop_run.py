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

from agent_composer.compile.expand import build_grow
from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow
from agent_composer.nodes.contracts import PartialRecordContract
from agent_composer.typesys import ValueKind
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


def test_loop_budget_exceeded_in_step_fails_run(monkeypatch):
    # The node-budget guard inside `_apply_grow` raises a RuntimeError; when a fresh driver's
    # grow (iteration >= 1, driven by `loop~1` off the body-END wiring) trips it, it must become a
    # failed run, not an uncaught escape. The COUNTER flow adds ~3 body nodes + 1 driver node per
    # iteration off a base of 3, so a budget of 6 lets iteration 0 grow and trips a later iteration
    # inside `_apply_grow` — exercising the grow boundary specifically.
    import agent_composer.runtime.state as state_module

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
    import agent_composer.runtime.state as state_module

    monkeypatch.setattr(state_module, "MAX_TOTAL_NODES", 40)
    engine = _engine(
        load_flow(TIMES_20).compiled,
        run_inputs={},
        grow_builder=build_grow,
    )
    terminal = list(engine.run())[-1]
    from agent_composer.events import RunSucceeded

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
    from agent_composer.events import RunSucceeded
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
