"""End-to-end (Ollama-free): a searched case reading a skipped branch under `:-`, `|`, `:?`.

`gate` routes on `input.score` to `pro` or `con` (only one runs). `gate2` is a searched case
whose `when:` reads `pro.output`, routing to `hot` or `cold`. With `score: 0.2`, `pro` is
skipped: a plain read skips `gate2` with it, while a read under an escape runs `gate2` and
lets the escape supply the value, exactly as a binding of that escape would. The `when:`
stays lazy: a later case's `:?` never runs when an earlier case matches.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow

_FLOW = """
id: when_escape
name: when_escape
input:
  score: float
  topic: str
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: pro
    else: con
  pro:
    kind: code
    code: tests.engine._compose_codefns:pick_pro
    input:
      topic: ${input.topic}
    output: str
  con:
    kind: code
    code: tests.engine._compose_codefns:pick_con
    input:
      topic: ${input.topic}
    output: str
  gate2:
    kind: case
    cases:
<CASES>
    else: cold
  hot:
    kind: code
    code: tests.engine._compose_codefns:positive
    input:
      topic: ${input.topic}
    output: str
  cold:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.topic}
    output: str
<EXTRA>output: ${gate2.output}
"""

_HOT = "pro case for ACME"
_COLD = "cautious note for ACME"

# A node with a record output, for a field read inside an escape.
_PLAN = """  plan:
    kind: code
    code: tests.engine._compose_codefns:make_plan
    input:
      topic: ${input.topic}
    output:
      rating: str
      score: float
"""


def _flow(*cases: tuple[str, str], extra: str = "") -> str:
    """`_FLOW` with `gate2`'s `(when, then)` cases and `extra` nodes spliced in."""
    body = "".join(f"      - when: {when}\n        then: {then}\n" for when, then in cases)
    return _FLOW.replace("<CASES>\n", body).replace("<EXTRA>", extra)


def _run(*cases: tuple[str, str], score: float = 0.2):
    """Load and run the flow with `cases`, on a score that skips `pro` unless raised."""
    return run_flow(load_flow(_flow(*cases)), {"score": score, "topic": "ACME"})


def test_a_default_runs_the_gate_on_a_skipped_branch():
    out = _run(('\'(pro.output :- "none") == "none"\'', "hot"))
    assert out.status == "succeeded"
    assert out.output == _HOT


def test_a_plain_read_skips_the_gate_with_the_branch():
    out = _run(('\'pro.output == "pro: ACME"\'', "hot"))
    assert out.status == "succeeded"
    assert out.output is None


def test_a_literal_coalesce_runs_the_gate_on_a_skipped_branch():
    out = _run(('\'(pro.output | "none") == "none"\'', "hot"))
    assert out.status == "succeeded"
    assert out.output == _HOT


def test_a_required_read_fails_the_gate_on_a_skipped_branch():
    out = _run(('\'(pro.output :? "pro must run") == "x"\'', "hot"))
    assert out.status == "failed"
    assert out.failure.code == "node.execute"
    assert out.failure.message == "pro must run"
    assert out.failure.node_id == "gate2"
    assert out.failure.cause_type == "RequiredError"
    assert out.failure.trace[0].key.endswith("/s:case/i:0/s:when")


def test_a_later_case_required_read_is_not_evaluated_when_an_earlier_case_matches():
    out = _run(
        ('"input.score < 0.5"', "hot"),
        ('\'(pro.output :? "m") == "x"\'', "cold"),
    )
    assert out.status == "succeeded"
    assert out.output == _HOT


def test_a_coalesce_of_two_branches_reads_the_one_that_ran():
    out = _run(('\'(pro.output | con.output) == "con: ACME"\'', "hot"))
    assert out.status == "succeeded"
    assert out.output == _HOT


def test_a_call_on_a_skipped_branch_under_a_default_fails_in_the_predicate():
    out = _run(('\'(upper(pro.output) :- con.output) == "con: ACME"\'', "hot"))
    # The same failure a binding `${upper(pro.output) :- con.output}` raises.
    assert out.status == "failed"
    assert out.failure.code == "node.execute"
    assert out.failure.node_id == "gate2"
    assert out.failure.message == (
        "expression function 'upper' failed: upper() got null; give a default, "
        'e.g. `upper(x :- "")`'
    )


def test_a_call_on_a_skipped_branch_in_a_later_case_is_not_evaluated():
    out = _run(
        ('"input.score < 0.5"', "hot"),
        ('\'(upper(pro.output) :- con.output) == "con: ACME"\'', "cold"),
    )
    assert out.status == "succeeded"
    assert out.output == _HOT


def test_a_co_skipping_default_skips_the_gate_when_its_producers_skip():
    # A binding `${pro.output :- input.topic}` skips with `pro`; so does the gate, through
    # its required join input `${pro.output}`.
    out = _run(('\'(pro.output :- input.topic) == "ACME"\'', "hot"))
    assert out.status == "succeeded"
    assert out.output is None


def test_a_default_reads_the_branch_when_it_ran():
    out = _run(('\'(pro.output :- "none") == "none"\'', "hot"), score=0.9)
    assert out.status == "succeeded"
    assert out.output == _COLD


def test_an_unknown_field_inside_an_escape_names_the_authored_reference():
    with pytest.raises(LoadError) as caught:
        load_flow(_flow(('"(plan.output.nope :- 0) > 1"', "hot"), extra=_PLAN))
    message = str(caught.value)
    assert message == (
        "flow has unresolved references:\n"
        "  node 'gate2' cases[0].when from: reference ${plan.output.nope} reads unknown "
        "field 'nope' on record"
    )
    assert "__r" not in message
    assert ":- null" not in message


@pytest.mark.parametrize(
    "when, reference",
    [
        ('"(nope.output | pro.output) == 1"', "${nope.output}"),
        ('"(pro.output :- nope.output.x) == 1"', "${nope.output.x}"),
    ],
)
def test_an_unknown_namespace_inside_a_co_skipping_escape_is_reported_once(when, reference):
    with pytest.raises(LoadError) as caught:
        load_flow(_flow((when, "hot")))
    assert str(caught.value) == (
        "flow has unresolved references:\n"
        f"  node 'gate2' cases[0].when from: reference {reference} uses unknown namespace "
        "'nope'"
    )
