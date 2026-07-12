"""Flow-level failures carry a `SourceSpan` on `RunResult.locator`.

A post-terminal assert (`${id.output} ...`) sets an `assert` locator; an input-coercion
failure at the boundary sets an `input_decl` locator (Step 8). Both flow-level paths have
no node behind them, so the locator rides the `RunResult`, not a `NodeFailed`.
"""

from pathlib import Path

from agent_composer.compose import load_flow
from agent_composer.run import run_flow
from agent_composer.run import resume_flow
from agent_composer.suspension.commands import DeliverAnswerCommand

_ERRORS = Path(__file__).resolve().parents[1] / "seeds" / "errors"


def test_post_assert_sets_run_result_locator():
    text = (_ERRORS / "e19-false-post-assert.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X"})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "assert"
    assert result.locator.node is None
    assert result.locator.key == "calc.output > 100"


def test_input_coercion_sets_input_decl_locator():
    text = (_ERRORS / "e08-input-type-mismatch.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X", "window": "soon"})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "input_decl"
    assert result.locator.node is None
    assert result.locator.key == "window"


def test_boundary_assert_sets_assert_locator():
    text = (_ERRORS / "e18-false-boundary-assert.yaml").read_text()
    result = run_flow(load_flow(text), {"window": -5})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "assert"
    assert result.locator.node is None
    assert result.locator.key == "input.window > 0"


def test_code_wrong_type_output_sets_field_locator():
    # A CODE node whose returned value fails its declared `output:` Type is rejected at the
    # typed write boundary. The failure surfaces as a node-less RunFailed; its locator points
    # at the node's `output:` field so the CLI boxes the declaration, not a plain message.
    text = (_ERRORS / "e21-code-wrong-type.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X"})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "field"
    assert result.locator.node == "calc"
    assert result.locator.key == "output"


_RESUME_WRONG_TYPE = """
id: f
name: f
nodes:
  ask:
    kind: human_input
    prompt: "how many?"
    output: int
output: ${ask.output}
"""


def test_resume_wrong_type_answer_sets_field_locator():
    # A delivered HUMAN_INPUT answer that fails the parked node's declared `output:` Type is
    # rejected at the resume write boundary. The RunFailed must still carry the `field` locator
    # (the resume() path forwards it identically to the run()/pooled paths).
    loaded = load_flow(_RESUME_WRONG_TYPE)
    res = run_flow(loaded, {})
    assert res.status == "paused"
    bad = DeliverAnswerCommand(node_id="ask", value="not-an-int")
    res2 = resume_flow(loaded, engine=res.engine, commands=[bad])
    assert res2.status == "failed"
    assert res2.locator is not None and res2.locator.kind == "field"
    assert res2.locator.node == "ask"
    assert res2.locator.key == "output"

