"""Importable CODE-node functions for the engine end-to-end tests.

A CODE node references one of these as `module:function`; it receives its bound
typed input record (a dict of its declared inputs, resolved from their `from:`
sources), NOT the pool. Each returns a dict of outputs.
"""


def analyze(inputs: dict) -> dict:
    return {"output": f"analyzed {inputs['topic']}", "score": 0.8}


def approve(inputs: dict) -> dict:
    return {"output": "APPROVE"}


def defer(inputs: dict) -> dict:
    return {"output": "DEFER"}


def finalize(inputs: dict) -> dict:
    return {"output": inputs.get("buy_out") or inputs.get("hold_out")}


def child_step(inputs: dict) -> dict:
    # Echoes the bound `topics` argument (CALL binds it explicitly — no
    # ambient inheritance, no parent hyperparameter).
    return {"output": inputs.get("topics")}


def echo(inputs: dict) -> dict:
    # echoes its single declared input back as the output (for run-boundary tests)
    return {"output": next(iter(inputs.values()), None)}


def echo_window(inputs: dict) -> dict:
    # Echoes the bound `window` to prove CALL child declared-default seeding.
    return {"output": inputs.get("window")}


def echo_as_of(inputs: dict) -> dict:
    # Echoes the bound `as_of_date` to prove CALL child date seeding.
    return {"output": inputs.get("as_of_date")}


def child_report(inputs: dict) -> dict:
    # A string-typed child output used for typed CALL output re-export tests.
    return {"output": f"REPORT:{inputs.get('topics')}"}


def echo_topic(inputs: dict) -> dict:
    # echoes the bound `topic` (the MAP per-element child output)
    return {"output": inputs["topic"]}


def make_rating(inputs: dict) -> dict:
    # returns a well-formed Rating record {category, score} — the node's single value
    # (used for the write-boundary record-enforcement test).
    return {"category": "pro", "score": 0.8}


def bad_rating(inputs: dict) -> dict:
    # returns a Rating record MISSING the required `score` — must fail the typed
    # write boundary (record values are enforced once the registry is threaded).
    return {"category": "pro"}


def boom(inputs: dict) -> dict:
    # Always raises, proving a CALL/MAP child failure fails the parent run.
    raise RuntimeError("boom")


def passthrough(inputs: dict) -> dict:
    # echoes the bound `items` list back (used to prove a MAP's typed List[U] binds
    # downstream via the whole-value ref ${<map>.output}).
    return {"output": inputs["items"]}


def sleepy_echo(inputs: dict) -> dict:
    # sleeps ~0.1s then echoes the bound `topic` — proves MAP parallel:true
    # overlaps element runs (4 in parallel finish well under 4x0.1s sequential).
    import time

    time.sleep(0.1)
    return {"output": inputs["topic"]}
