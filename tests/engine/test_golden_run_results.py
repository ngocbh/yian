from pathlib import Path

from agent_composer.compose import (
    LoadError,
    load_flow,
)
from agent_composer.run import run_flow
from agent_composer.tools import install_ask_user

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _text(name: str) -> str:
    return (_SEEDS / name).read_text()


def _resolver(**children):
    # Mirrors tests/engine/test_ref_run.py:_resolver —
    # resolve(flow_id, version=None) returns a pre-loaded child; unknown id is a loud LoadError.
    loaded = {fid: load_flow(text) for fid, text in children.items()}

    def resolve(flow_id, version=None):
        try:
            return loaded[flow_id]
        except KeyError as exc:
            raise LoadError(f"unknown child {flow_id!r}") from exc

    return resolve


# --- (i) CODE-only REF run + CODE-only MAP run — the run-result behavior must be preserved.
#     LLM-free via run_flow + a child_resolver fake. NOT seeds 04/05.
_CHILD = """
id: child-one
name: child_one
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${emit.output.report}
  n: ${emit.output.n}
"""

_REF = """
id: ref-parent
name: ref_parent
input:
  topic: str
uses:
  child-one: child-one
nodes:
  research:
    kind: call
    call: child-one
    input:
      topic: ${input.topic}
output: ${research.output}
"""

_MAP = """
id: map-parent
name: map_parent
input:
  topics: list[str]
uses:
  child-one: child-one
nodes:
  research_each:
    kind: map
    call: child-one
    over: ${input.topics}
    input:
      topic: ${item}
output: ${research_each.output}
"""


def test_golden_ref_run_result():
    parent = load_flow(_REF, child_resolver=_resolver(**{"child-one": _CHILD}))
    res = run_flow(parent, {"topic": "ACME"})
    assert res.status == "succeeded"
    assert res.output == {"report": "report for ACME", "n": 4}


def test_golden_map_run_result():
    parent = load_flow(_MAP, child_resolver=_resolver(**{"child-one": _CHILD}))
    res = run_flow(parent, {"topics": ["ACME", "BETA"]})
    assert res.status == "succeeded"
    # join is in OVER ORDER (element index) — the invariant the MAP -> END_ID(list) fan-in must keep.
    assert res.output == [
        {"report": "report for ACME", "n": 4},
        {"report": "report for BETA", "n": 4},
    ]


# --- (ii) blocking HUMAN_INPUT followed by an absolute UTC WAIT (CODE-only, LLM-free).
#     Mirrors seed 17's shape, but agent-free so it runs without an LLM.
_EFFECTS = """
id: e
name: e
typedefs:
  Approval: Literal[approve, reject]
input:
  settle_at: date
nodes:
  approve:
    kind: human_input
    prompt: "approve? (approve/reject)"
    output: Approval
  gate:
    kind: case
    on: ${approve.output}
    cases:
      - when: approve
        then: settle
    else: abort
  settle:
    kind: wait
    until: ${input.settle_at}
  confirm:
    kind: code
    depends_on: [settle]
    input:
      answer: ${approve.output}
    output: str
    code: tests.seeds.fns:confirm_action
  abort:
    kind: code
    input:
      answer: ${approve.output}
    output: str
    code: tests.seeds.fns:confirm_action
output: ${confirm.output | abort.output}
"""


def test_golden_human_input_then_past_wait_completes_without_checkpoint():
    # HUMAN_INPUT completes through its trusted host; the past WAIT returns in the same run.
    loaded = load_flow(_EFFECTS)
    with install_ask_user(lambda **request: "approve"):
        result = run_flow(loaded, {"settle_at": "2000-01-01"})
    assert result.status == "succeeded"
    assert result.checkpoint is None
    assert result.output == "approve"


# --- (iii) static edge-id sets for a few flows — LOAD-pinned against the
#     __start__/__end__ boundary nodes (START_ID/END_ID are ordinary nodes whose ids ARE the
#     old sentinel strings). test_golden_run_results's test_boundary_nodes re-pins these edge ids.
def _edge_ids(name: str, **kw) -> list[str]:
    return sorted(e.id for e in load_flow(_text(name), **kw).compiled.edges)


def test_golden_static_edge_ids_case_and_call_and_map():
    # seed 02 (case + coalesce). `score`/`positive`/`cautious` read ${input.topic} -> START_ID data edges.
    assert _edge_ids("02-case.yaml") == [
        "__start__->cautious#0",
        "__start__->positive#0",
        "__start__->score#0",
        "cautious->__end__#0",
        "gate->cautious#0",
        "gate->positive#0",
        "positive->__end__#0",
        "score->gate#0",
    ]
    # seed 04 (REF): `take` AGENT reads TWO refs from `research`; `research` reads ${input.topic}.
    assert _edge_ids("04-call.yaml", search_paths=[_SEEDS]) == [
        "__start__->research#0",
        "research->__end__#0",
        "research->take#0",
        "research->take#1",
        "take->__end__#0",
    ]
    # seed 05 (MAP): `research_each` reads ${input.topics} (over) + ${input.as_of} (opt).
    assert _edge_ids("05-call-map.yaml", search_paths=[_SEEDS]) == [
        "__start__->research_each#0",
        "__start__->research_each#1",
        "compare->__end__#0",
        "research_each->__end__#0",
        "research_each->compare#0",
    ]


# --- (iii.b) a single ${input.x} reader with a bare output. Pins the input-producer DATA edge
#     + the END_ID producer edge — the canonical `START_ID -> body -> END_ID` shape with NO
#     bare-root edge for an input-reader.
_P25_FLOW = """
id: f
name: f
input:
  x: str
nodes:
  n:
    kind: code
    input:
      x: ${input.x}
    output: str
    code: tests.engine._compose_codefns:echo_x
output: ${n.output}
"""


def test_golden_static_edge_ids_start_end():
    flow = load_flow(_P25_FLOW).compiled
    assert sorted(e.id for e in flow.edges) == [
        "__start__->n#0",   # the ${input.x} producer DATA edge (input_group=x); NO bare root edge
        "n->__end__#0",     # the outputs: ${n.output} producer edge (input_group=result)
    ]


# --- (iv) the inputs/outputs author surface: a flow with input: defaults + a multi-output
#     output:. Pins coerce + apply_defaults + the >=2-output record arity that START_ID/END_ID
#     must reproduce (coerce/default; EndNode record-mode >=2 -> {name: value}).
_SURFACE = """
id: surface
name: surface
input:
  topic: str
  window: int = 30
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
      window: ${input.window}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${emit.output.report}
  n: ${emit.output.n}
"""


def test_golden_inputs_defaults_and_multi_output_surface():
    loaded = load_flow(_SURFACE)
    # The declared default fills when OMITTED, coerced to the declared type, INSIDE START's wrapper
    # (window omitted -> 30) — proven by the terminal output below (n=4 uses window=30). This is the
    # coerce+apply_defaults lifted onto StartNode.run. `RunResult.input` is the caller-passed args
    # coerced (defaulting lives in START, not the run boundary), so window is NOT re-surfaced here.
    res = run_flow(loaded, {"topic": "ACME"})
    assert res.status == "succeeded"
    assert res.input == {"topic": "ACME"}
    # >=2 declared outputs -> a record keyed by name (the END_ID record-mode arity).
    assert res.output == {"report": "report for ACME", "n": 4}


# --- (v) the declared-output arity END_ID(record) must keep byte-identical: 0 -> None,
#     1 -> bare, >=2 -> record keyed by name. Author-visible outputs/status only — never edge ids.
_ARITY_0 = """
id: arity0
name: arity0
input:
  x: str
nodes:
  n:
    kind: code
    input:
      topic: ${input.x}
    output: str
    code: tests.engine._compose_codefns:echo
"""

_ARITY_1 = """
id: arity1
name: arity1
input:
  x: str
nodes:
  n:
    kind: code
    input:
      topic: ${input.x}
    output: str
    code: tests.engine._compose_codefns:echo
output: ${n.output}
"""

_ARITY_2 = """
id: arity2
name: arity2
input:
  x: str
nodes:
  n:
    kind: code
    input:
      topic: ${input.x}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${n.output.report}
  count: ${n.output.n}
"""


def test_golden_output_arity_0():
    # no `outputs:` section -> END_ID(record, 0 names) -> None.
    res = run_flow(load_flow(_ARITY_0), {"x": "ACME"})
    assert res.status == "succeeded"
    assert res.output is None


def test_golden_output_arity_1():
    # bare `outputs: ${n.output}` -> END_ID(record, 1 name) -> the bare value.
    res = run_flow(load_flow(_ARITY_1), {"x": "ACME"})
    assert res.status == "succeeded"
    assert res.output == "ACME"


def test_golden_output_arity_2():
    # name-map `outputs:` -> END_ID(record, >=2 names) -> a record keyed by output name.
    res = run_flow(load_flow(_ARITY_2), {"x": "ACME"})
    assert res.status == "succeeded"
    assert res.output == {"report": "report for ACME", "count": 4}
