"""A named `call` node whose `.output` a flow `asserts:` expression reads.

A flow-level post assert can reference any node's output, including a named `call`
node's: `dbl_call` runs the `dbl` def, and `${dbl_call.output} >= 0` is classified
post and checked after the run. No synth node is minted — the call is an ordinary,
author-named node in the flow.
"""

from yian.compose import load_flow
from yian.run import run_flow

# An in-file `dbl` def (the assert callee): doubles its int input via the `double` CODE fn.
_DBL_DEF = """
defs:
  dbl:
    input:
      n: int
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${input.n}
        output: int
    output: ${x.output}
"""

_FLOW = f"""
id: ai
name: ai
input:
  v: int
{_DBL_DEF}
nodes:
  main:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${{input.v}}
    output: int
  dbl_call:
    kind: call
    call: dbl
    input:
      n: ${{input.v}}
output: ${{main.output}}
asserts:
  - "${{dbl_call.output}} >= 0"
"""


def test_named_call_node_assert_loads():
    loaded = load_flow(_FLOW)
    assert "dbl_call" in loaded.compiled.nodes  # the assert reads an author-named call node
    assert not any(nid.startswith("__call_") for nid in loaded.compiled.nodes)  # no synth node minted


def test_inline_call_assert_passes():
    out = run_flow(load_flow(_FLOW), {"v": 5})
    assert out.status == "succeeded"
    assert out.output == 10


def test_inline_call_assert_fails_post():
    out = run_flow(load_flow(_FLOW), {"v": -3})  # dbl(-3) = -6, violates `>= 0`
    assert out.status != "succeeded"
    assert out.failure is not None
    assert "assert" in out.failure.message.lower()
