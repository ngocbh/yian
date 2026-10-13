"""CALL child-run seam — end-to-end, Ollama-free.

A plain `call` (`kind: call` without `over:`) applies a callable by explicit application:
bind the declared args against the parent pool, seed a child pool (+ the child's own
declared defaults for omitted args), drive the baked child engine, and re-export the
child's single value. These drive real CODE-only children through `run_flow` (no LLM).
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.run import run_flow

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

_CALL_PARENT = """
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

_CALL_BOOM = """
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
            raise LoadDiagnostics().error(f"unknown child {flow_id!r}") from exc

    return resolve


def test_call_runs_child_and_reexports_value():
    parent = load_flow(_CALL_PARENT, child_resolver=_resolver(**{"child-one": _CHILD}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status == "succeeded"
    # The CALL node re-exports the child's {report, n}; the parent re-keys it.
    assert result.output == {"report": "report for ACME", "n": 4}


def test_call_output_projection_preserves_nullable_parent_path():
    """A dotted read through a nullable record may itself evaluate to None."""
    parent = load_flow(
        """id: nullable-projection
name: nullable_projection
typedefs:
  Payload:
    value: int
input:
  payload: Optional[Payload]
defs:
  pick:
    input:
      payload: Optional[Payload]
    nodes: {}
    output: ${input.payload.value}
nodes:
  invoke:
    kind: call
    call: pick
    input:
      payload: ${input.payload}
output: ${invoke.output}
"""
    )

    contract = parent.compiled.nodes["invoke"].output_contract
    assert contract is not None
    assert contract.type.nullable is True

    result = run_flow(parent, {"payload": None})
    assert result.status == "succeeded"
    assert result.output is None


def test_call_accepts_none_elements_for_nullable_list_sink():
    """Static list compatibility recurses into the nullable element contract."""
    parent = load_flow(
        """id: list-none-call
name: list_none_call
defs:
  echo:
    input:
      values: list[Optional[str]]
    nodes: {}
    output: ${input.values}
nodes:
  source:
    kind: code
    code: |
      return [None]
    output: list[None]
  invoke:
    kind: call
    call: echo
    input:
      values: ${source.output}
output: ${invoke.output}
"""
    )

    result = run_flow(parent, {})
    assert result.status == "succeeded"
    assert result.output == [None]


def test_call_child_failure_fails_the_run():
    parent = load_flow(_CALL_BOOM, child_resolver=_resolver(**{"boom-child": _BOOM_CHILD}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status != "succeeded"
    assert "boom" in result.failure.message
    call = parent.compiled.nodes["research"]
    assert result.failure.node_id == "research/emit"
    assert result.failure.trace == (
        call.origin,
        call.child.nodes["emit"].code_origin,
    )


def test_call_child_applies_its_own_default():
    # the child declares `suffix: str = "!"`, which the parent does not bind; the child's
    # own default fills it (the default-seeding path runs without the parent supplying it).
    parent = load_flow(_CALL_PARENT, child_resolver=_resolver(**{"child-one": _CHILD}))
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

_CALL_PARENT_BD = """
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


def test_call_boundary_assert_sees_omitted_default():
    # `window`'s default (30) is omitted by the parent, yet the child's BOUNDARY
    # assert `${input.window} >= 1` must see the EFFECTIVE input (30), not None — the eager
    # pre-run check mirrors START_ID's omitted-input defaulting (regression guard: a naive
    # driver-apply_defaults drop left the assert reading the raw, un-defaulted record).
    parent = load_flow(_CALL_PARENT_BD, child_resolver=_resolver(**{"child-bd": _CHILD_BOUNDARY_DEFAULT}))
    result = run_flow(parent, {"topic": "ACME"})
    assert result.status == "succeeded", result.failure
    assert result.output == "ACME"


_CALL_PARENT_BD_TEXT = """
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


def test_call_arg_text_is_not_converted_for_a_typed_child_input():
    # A CALL edge is not a flow boundary: the literal "30" into the child's `int` input is
    # not converted, so the flow fails to load instead of the boundary assert ever
    # comparing `"30" >= 1`.
    with pytest.raises(LoadError) as exc:
        load_flow(_CALL_PARENT_BD_TEXT, child_resolver=_resolver(**{"child-bd": _CHILD_BOUNDARY_DEFAULT}))
    assert "binding 'window' — child expects 'int', but the value \"30\" does not fit" in str(exc.value)


def test_call_run_expands_not_child_engine():
    # The CallNode returns only a seed; compile reconstructs the child fragment for the engine.
    from yian.compile.expand import build_grow
    from yian.nodes.base import Grow
    from yian.nodes.call import CallNode
    from yian.nodes.ids import ns
    from tests.engine.test_expand import _child_flow

    child = _child_flow()
    n = CallNode("r", flow_id="c", child=child, child_inputs=[])   # call once
    out = n.run({"topic": "ACME"})          # no *, system cap anymore
    assert isinstance(out, Grow)
    assert out.seed == {"topic": "ACME"}
    assert build_grow(n, out.seed).start_id == ns("r", child.start_id)
