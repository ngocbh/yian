"""`NodeFailed` carries a `SourceSpan` locator with the node id stamped in.

`bind_params` produces a node-less `input` locator; eval_node's funnel fills the
node id. The three direct node-assert yields produce `assert` locators directly.
"""

from pathlib import Path

from agent_composer.compose import load_flow, run_flow
from agent_composer.events import NodeFailed

_ERRORS = Path(__file__).resolve().parents[1] / "seeds" / "errors"


def _node_failed(result):
    fs = [e for e in result.events if isinstance(e, NodeFailed)]
    return fs[-1] if fs else None


def test_binding_failure_locator_reaches_node_failed():
    text = (_ERRORS / "e07-required-missing.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X"})  # as_of omitted -> :? fires
    nf = _node_failed(result)
    assert nf is not None and nf.locator is not None
    assert nf.locator.node == "report" and nf.locator.kind == "input"
    assert nf.locator.key == "as_of"


def test_binding_failure_carries_concrete_range():
    # ALONGSIDE the symbolic `input`/`as_of` locator, the funnel resolves a concrete
    # SourceRange from the node's recorded per-input-param `field_ranges["as_of"]` — the
    # `as_of:` binding value on line 23 of e07. The parse choke point records it kind-blind.
    text = (_ERRORS / "e07-required-missing.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X"})
    nf = _node_failed(result)
    assert nf is not None and nf.range is not None
    assert nf.range.start_line == 23


_NODE_POST_ASSERT = """
id: npa
name: npa
input:
  topic: str
nodes:
  emit:
    kind: code
    code: tests.seeds.fns:const_one
    input:
      topic: ${input.topic}
    output: int
    asserts:
      - "${output} > 100"
output: ${emit.output}
"""


def test_node_post_assert_locator():
    # const_one returns 1 -> the post-assert `${output} > 100` fails on node `emit`.
    result = run_flow(load_flow(_NODE_POST_ASSERT), {"topic": "X"})
    nf = _node_failed(result)
    assert nf is not None and nf.locator is not None
    assert nf.locator.node == "emit" and nf.locator.kind == "assert"
    assert nf.locator.key == "${output} > 100"
