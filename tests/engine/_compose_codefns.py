"""Test-local CODE-node functions for the `run_flow` tests (Ollama-free).

A CODE node references one of these as `tests.engine._compose_codefns:fn`; it receives its
bound typed input record (a dict of its declared inputs, resolved from their
sources) and returns the node's one output value (a dict for a record `output:`,
a scalar otherwise). No Ollama, no agent runtime — these keep the run tests pure.
"""

import threading


def make_plan(inputs: dict) -> dict:
    # returns a multi-field object matching a declared {rating, score} record.
    return {"rating": f"plan for {inputs['topic']}", "score": 0.9}


def score(inputs: dict) -> float:
    # a single numeric value (the case `on:`/`when:` producer in the coalesce flow).
    return inputs["seed"]


def positive(inputs: dict) -> str:
    return f"pro case for {inputs['topic']}"


def cautious(inputs: dict) -> str:
    return f"cautious note for {inputs['topic']}"


def join_two(inputs: dict) -> str:
    # fan-in: concatenate two upstream string outputs (parallel-branch merge tests).
    return f"{inputs['pro']} | {inputs['con']}"


# --- CALL/MAP child-flow CODE fns (Ollama-free subflows) -------------------- #
def make_report(inputs: dict) -> dict:
    # A {report, n} child codomain (CALL re-exports it; MAP collects a list of it).
    t = inputs["topic"]
    return {"report": f"report for {t}", "n": len(t)}


def echo(inputs: dict):
    # returns the bound `topic` unchanged (MAP order/identity checks).
    return inputs["topic"]


def echo_x(inputs: dict):
    # returns the bound `x` unchanged (the boundary-node _FLOW fixture).
    return inputs["x"]


def boom(inputs: dict):
    # A child CODE node that fails makes the CALL/MAP run fail.
    raise ValueError("boom")


# --- co-skip fns (Ollama-free branch+join) ---------------------------------- #
def pick_pro(inputs: dict) -> str:
    return f"pro: {inputs['topic']}"


def pick_con(inputs: dict) -> str:
    return f"con: {inputs['topic']}"


def detail_of(inputs: dict) -> str:
    return f"detail: {inputs['base']}"


def assemble_join(inputs: dict) -> str:
    # claim = the taken branch (ref-coalesce join); detail = pro_detail or null (`:-null`).
    return f"{inputs['claim']}|detail={inputs['detail']}"


def took(inputs: dict) -> str:
    # the `take(stance=...)` def body for then:/else: ${call} tests.
    return f"took:{inputs['stance']}"


def double(inputs: dict) -> int:
    # the `checked` def body for child-boundary-assert tests.
    return inputs["n"] * 2


def echo_rid(inputs: dict) -> str:
    # surfaces the bound ${system.run_id} so the run test can assert on it.
    return inputs["rid"]


# --- LOOP body CODE fns (Ollama-free while-loop e2e) ------------------------ #
def loop_bump(inputs: dict) -> dict:
    # a loop body 'a -> 'a over {n, exited}: increment n, set exited once n reaches 3.
    # Drives the while-loop e2e (3 iterations from a {n:0, exited:false} seed).
    n = inputs["n"] + 1
    return {"n": n, "exited": n >= 3}


def chat_fold(inputs: dict) -> dict:
    # the chat-shaped loop body 'a -> 'a over {messages, exited}: append this turn's human
    # message to the carried list and set exited once the human types "bye". Drives the
    # blocking-turn e2e (a human_input leaf feeds `msg` each iteration).
    messages = inputs["messages"] + [inputs["msg"]]
    return {"messages": messages, "exited": inputs["msg"] == "bye"}


def chat_fold_list(inputs: dict) -> dict:
    # like chat_fold but folds a LIST of this turn's messages (a MAP over the turn's elements
    # feeds `msgs`): extend the carried list and exit once any message is "bye". Drives the
    # MAP-spawner-inside-a-loop-body artificial-pause compatibility e2e.
    messages = inputs["messages"] + list(inputs["msgs"])
    return {"messages": messages, "exited": "bye" in inputs["msgs"]}


def loop_countdown(inputs: dict) -> dict:
    # decrements the carried n. Drives a loop whose `while:` predicate divides by n
    # (`10 / n > 0`), so the predicate RAISES (division by zero) once n reaches 0 on
    # a later iteration — the engine must convert that predicate error into a failed run,
    # not let it escape run() uncaught.
    return {"n": inputs["n"] - 1}


def reply_fold(inputs: dict) -> dict:
    # loop body 'a -> 'a over {transcript, exited}: append this turn's agent reply.
    return {"transcript": inputs["transcript"] + "|" + inputs["reply"],
            "exited": bool(inputs["exited"])}


def huge_float_list(inputs: dict) -> list:
    # a mixed int/float list whose int widens past float's range (no `output:` declared).
    return [0.5, 10**400]


# Tags `record` saw, in call order; a test clears it before running a flow.
RECORDED: list = []


def record(inputs: dict) -> str:
    # a side-effect probe: notes that its node ran, then echoes its tag.
    RECORDED.append(inputs["tag"])
    return inputs["tag"]


# Two `meet` nodes pass the barrier only when both are running, so a run that finishes proves
# they ran at the same time; one waiting on the other times out instead. Reset before each use.
MEET = threading.Barrier(2, timeout=5)


def meet(inputs: dict) -> str:
    # waits for the other `meet` node, then records its tag like `record`.
    MEET.wait()
    return record(inputs)


def deep_tree(inputs: dict) -> dict:
    # a {tree} record whose tree is a chain of `n` nested dicts (value nesting tests).
    tree: object = "leaf"
    for _ in range(inputs["n"]):
        tree = {"a": tree}
    return {"tree": tree}
