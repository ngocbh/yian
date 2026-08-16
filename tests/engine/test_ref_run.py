"""REF child-run seam — end-to-end, Ollama-free.

A plain `call` (`kind: call` without `over:`) applies a callable by explicit application:
bind the declared args against the parent pool, seed a child pool (+ the child's own
declared defaults for omitted args), drive the baked child engine, and re-export the
child's single value. These drive real CODE-only children through `run_flow` (no LLM).
"""

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow

# A CODE-only child: topic -> {report, n}.
_CHILD = """
id: child-one
name: child_one
input:
  topic: str
  suffix: str = "!"
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

_BOOM_CHILD = """
id: boom-child
name: boom_child
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.engine._compose_codefns:boom
output: ${emit.output}
"""

_REF_PARENT = """
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
output:
  report: ${research.output.report}
  n: ${research.output.n}
"""

_REF_BOOM = """
id: ref-boom
name: ref_boom
input:
  topic: str
uses:
  boom-child: boom-child
nodes:
  research:
    kind: call
    call: boom-child
    input:
      topic: ${input.topic}
output: ${research.output}
"""


def _resolver(**children):
    loaded = {fid: load_flow(text) for fid, text in children.items()}

    def resolve(flow_id, version=None):
        try:
            return loaded[flow_id]
        except KeyError as exc:
            raise LoadError(f"unknown child {flow_id!r}") from exc

    return resolve


def test_ref_runs_child_and_reexports_value():
    parent = load_flow(_REF_PARENT, child_resolver=_resolver(**{"child-one": _CHILD}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status == "succeeded"
    # the REF node re-exports the child's {report, n}; the parent re-keys it.
    assert result.output == {"report": "report for ACME", "n": 4}


def test_ref_child_failure_fails_the_run():
    parent = load_flow(_REF_BOOM, child_resolver=_resolver(**{"boom-child": _BOOM_CHILD}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status != "succeeded"
    assert "boom" in result.failure.message
    call = parent.compiled.nodes["research"]
    assert result.failure.node_id == "research/emit"
    assert result.failure.trace == (
        call.origin,
        call.child.nodes["emit"].origin,
    )


def test_ref_child_applies_its_own_default():
    # the child declares `suffix: str = "!"`, which the parent does not bind; the child's
    # own default fills it (the default-seeding path runs without the parent supplying it).
    parent = load_flow(_REF_PARENT, child_resolver=_resolver(**{"child-one": _CHILD}))
    result = run_flow(parent, {"topic": "X"})
    assert result.status == "succeeded"
    assert result.output["n"] == 1


_CHILD_BOUNDARY_DEFAULT = """
id: child-bd
name: child_bd
input:
  topic: str
  window: int = 30
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.engine._compose_codefns:echo
output: ${emit.output}
asserts:
  - ${input.window} >= 1
"""

_REF_PARENT_BD = """
id: ref-parent-bd
name: ref_parent_bd
input:
  topic: str
uses:
  child-bd: child-bd
nodes:
  research:
    kind: call
    call: child-bd
    input:
      topic: ${input.topic}
output: ${research.output}
"""


def test_ref_boundary_assert_sees_omitted_default():
    # `window`'s default (30) is omitted by the parent, yet the child's BOUNDARY
    # assert `${input.window} >= 1` must see the EFFECTIVE input (30), not None — the eager
    # pre-run check mirrors START_ID's omitted-input defaulting (regression guard: a naive
    # driver-apply_defaults drop left the assert reading the raw, un-defaulted record).
    parent = load_flow(_REF_PARENT_BD, child_resolver=_resolver(**{"child-bd": _CHILD_BOUNDARY_DEFAULT}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status == "succeeded", result.failure
    assert result.output == "ACME"


_REF_PARENT_BD_COERCE = """
id: ref-parent-bdc
name: ref_parent_bdc
input:
  topic: str
uses:
  child-bd: child-bd
nodes:
  research:
    kind: call
    call: child-bd
    input:
      topic: ${input.topic}
      window: "30"
output: ${research.output}
"""


def test_ref_boundary_assert_sees_coerced_present_value():
    # The eager boundary check must mirror START_ID's FULL transform (coerce THEN default), not just
    # defaulting: a PRESENT call-arg that is a coercible literal ("30" -> int 30) must be seen
    # COERCED by the boundary assert `${input.window} >= 1` — else `"30" >= 1` raises a
    # str-vs-int TypeError and spuriously fails a child the body would run fine.
    parent = load_flow(_REF_PARENT_BD_COERCE, child_resolver=_resolver(**{"child-bd": _CHILD_BOUNDARY_DEFAULT}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status == "succeeded", result.failure
    assert result.output == "ACME"


def test_ref_run_expands_not_child_engine():
    # The CallNode returns only a seed; compile reconstructs the child fragment for the engine.
    from agent_composer.compile.expand import build_grow
    from agent_composer.nodes.base import Grow
    from agent_composer.nodes.call import CallNode
    from agent_composer.nodes.ids import ns
    from tests.engine.test_expand import _child_flow

    child = _child_flow()
    n = CallNode("r", flow_id="c", child=child, child_inputs=[])   # REF — call once
    out = n.run({"topic": "ACME"})          # no *, system cap anymore
    assert isinstance(out, Grow)
    assert out.seed == {"topic": "ACME"}
    assert build_grow(n, out.seed).start_id == ns("r", child.start_id)
