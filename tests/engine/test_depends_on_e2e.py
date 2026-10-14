"""End-to-end `depends_on` co-skip versus `runs_after` pure ordering (Ollama-free).

A CODE-bodied flow where a side-effect `warm` node sits behind a `case`, so it can be
skipped. Two downstream nodes carry NO data from `warm`, only an ordering edge:

- `dep_fetch` — `depends_on: [warm]`  -> co-skips when `warm` is skipped.
- `after_fetch` — `runs_after: [warm]` -> still runs when `warm` is skipped.

Both run when `warm` runs (the ordering edge is just satisfied). The flow output
coalesces the two, so whichever survived is the terminal value.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.events import NodeSucceeded

_FLOW = """
id: ordering-e2e
name: ordering_e2e
input:
  score: !float
  topic: !str
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: warm
  warm:
    kind: code
    code: tests.engine._compose_codefns:pick_pro
    input:
      topic: ${input.topic}
    output: str
  dep_fetch:
    kind: code
    depends_on: [warm]
    code: tests.engine._compose_codefns:positive
    input:
      topic: ${input.topic}
    output: str
  after_fetch:
    kind: code
    runs_after: [warm]
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.topic}
    output: str
output: ${dep_fetch.output | after_fetch.output}
"""


def _succeeded(result) -> set[str]:
    return {e.node_id for e in result.events if isinstance(e, NodeSucceeded)}


def test_warm_runs_both_dependents_run():
    # score >= 0.5 -> warm runs -> both ordering edges satisfied -> both dependents run.
    result = run_flow(load_flow(_FLOW), {"score": 0.9, "topic": "ACME"})
    assert result.status == "succeeded", result.failure
    ran = _succeeded(result)
    assert {"warm", "dep_fetch", "after_fetch"} <= ran
    # coalesce picks the first non-None producer (dep_fetch ran).
    assert result.output == "pro case for ACME"


def test_warm_skipped_depends_on_co_skips_runs_after_survives():
    # score < 0.5 -> warm is skipped. dep_fetch co-skips (depends_on); after_fetch runs.
    result = run_flow(load_flow(_FLOW), {"score": 0.2, "topic": "ACME"})
    assert result.status == "succeeded", result.failure
    ran = _succeeded(result)
    assert "warm" not in ran
    assert "dep_fetch" not in ran        # depends_on co-skipped it
    assert "after_fetch" in ran          # runs_after kept it alive
    assert result.output == "cautious note for ACME"


# --- ordering edges FROM a case: the routed case settled, so they hold ---------- #


def _case_source_flow(*, field: str, on_branch: bool, with_else: bool = True) -> str:
    """A `gate` case routing to `pro`/`con`, with an ordering edge `field: [gate]` on a
    third node `note` (or, with ``on_branch``, on the `pro` branch target itself)."""
    branch_dep = f"\n    {field}: [gate]" if on_branch else ""
    note_dep = "" if on_branch else f"\n    {field}: [gate]"
    else_line = "\n    else: con" if with_else else ""
    con_output = "${pro.output | con.output}" if with_else else "${pro.output}"
    return f"""
id: case-source
name: case_source
input:
  score: !float
  topic: !str
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: pro{else_line}
  pro:
    kind: code{branch_dep}
    code: tests.engine._compose_codefns:pick_pro
    input:
      topic: ${{input.topic}}
    output: str
  con:
    kind: code
    code: tests.engine._compose_codefns:pick_con
    input:
      topic: ${{input.topic}}
    output: str
  note:
    kind: code{note_dep}
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${{input.topic}}
    output: str
output:
  pick: "{con_output}"
  note: ${{note.output}}
"""


@pytest.mark.parametrize("field", ["depends_on", "runs_after"])
@pytest.mark.parametrize("score", [0.9, 0.2])
def test_ordering_on_a_case_runs_on_every_branch(field, score):
    # The case routes either way, so it settles; an ordering edge from it is satisfied.
    result = run_flow(
        load_flow(_case_source_flow(field=field, on_branch=False)),
        {"score": score, "topic": "ACME"},
    )
    assert result.status == "succeeded", result.failure
    assert "note" in _succeeded(result)
    assert result.output["note"] == "cautious note for ACME"


@pytest.mark.parametrize("field", ["depends_on", "runs_after"])
def test_branch_target_ordered_on_its_case_runs_when_taken(field):
    flow = load_flow(_case_source_flow(field=field, on_branch=True))
    taken = run_flow(flow, {"score": 0.9, "topic": "ACME"})
    assert taken.status == "succeeded", taken.failure
    assert taken.output["pick"] == "pro: ACME"
    assert "con" not in _succeeded(taken)
    lost = run_flow(flow, {"score": 0.2, "topic": "ACME"})
    assert lost.status == "succeeded", lost.failure
    assert lost.output["pick"] == "con: ACME"
    assert "pro" not in _succeeded(lost)


def test_depends_on_a_case_with_no_match_and_no_else_still_runs():
    # No case matches and there is no `else:`: every branch skips, but the case settled.
    result = run_flow(
        load_flow(_case_source_flow(field="depends_on", on_branch=False, with_else=False)),
        {"score": 0.2, "topic": "ACME"},
    )
    assert result.status == "succeeded", result.failure
    ran = _succeeded(result)
    assert "note" in ran
    assert "pro" not in ran


# --- negatives: located unknown target + ordering cycle --------------------- #

_UNKNOWN_DEP = """
id: bad-dep
name: bad_dep
input:
  topic: !str
nodes:
  fetch:
    kind: code
    depends_on: [nope]
    code: tests.engine._compose_codefns:pick_pro
    input:
      topic: ${input.topic}
    output: str
output: ${fetch.output}
"""

_ORDERING_CYCLE = """
id: dep-cycle
name: dep_cycle
input:
  topic: !str
nodes:
  a:
    kind: code
    depends_on: [b]
    code: tests.engine._compose_codefns:pick_pro
    input:
      topic: ${input.topic}
    output: str
  b:
    kind: code
    runs_after: [a]
    code: tests.engine._compose_codefns:pick_con
    input:
      topic: ${input.topic}
    output: str
output: ${a.output}
"""


def test_unknown_depends_on_target_is_located_load_error():
    with pytest.raises(LoadError) as exc:
        load_flow(_UNKNOWN_DEP)
    msg = str(exc.value)
    assert "depends_on" in msg and "nope" in msg
    assert exc.value.diagnostic.frames[-1].range.start_line is not None


def test_ordering_cycle_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_ORDERING_CYCLE)
    assert "cycle" in str(exc.value)
