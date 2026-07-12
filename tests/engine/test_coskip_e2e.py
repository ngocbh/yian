"""End-to-end (Ollama-free): co-skip + `:-null` escape + ref-coalesce join.

A CODE-bodied flow that exercises all three data stances through a real `case` route, so
the veto + co-skip run via the full loader -> engine path (not just hand-built edges):

- `pro` / `con` are the two branches (only one runs).
- `pro_detail` reads a PLAIN `${pro.output}` -> co-skips when `pro` is skipped.
- `assemble` joins `${pro.output | con.output}` (survives, binds the taken branch) and
  `${pro_detail.output:-null}` (optional escape -> binds null when `pro_detail` co-skips).
"""

from agent_composer.compose import load_flow
from agent_composer.run import run_flow

_FLOW = """
id: coskip
name: coskip
input:
  score: float
  topic: str
nodes:
  gate:
    kind: case
    cases:
      - when: "${input.score} >= 0.5"
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
  pro_detail:
    kind: code
    code: tests.engine._compose_codefns:detail_of
    input:
      base: ${pro.output}
    output: str
  assemble:
    kind: code
    code: tests.engine._compose_codefns:assemble_join
    input:
      claim: ${pro.output | con.output}
      detail: ${pro_detail.output:-null}
    output: str
output: ${assemble.output}
"""


def test_con_route_co_skips_and_escape_binds_null():
    out = run_flow(load_flow(_FLOW), {"score": 0.2, "topic": "ACME"})
    assert out.status == "succeeded"
    # claim = con's value (the join survives the skipped pro); detail = null (pro_detail
    # co-skipped with pro, the `:-null` escape kept assemble alive).
    assert out.output == "con: ACME|detail=None"


def test_pro_route_runs_detail():
    out = run_flow(load_flow(_FLOW), {"score": 0.9, "topic": "ACME"})
    assert out.status == "succeeded"
    assert out.output == "pro: ACME|detail=detail: pro: ACME"
