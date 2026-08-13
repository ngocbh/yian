"""Flow-level failures carry a `SourceSpan` on `RunResult.locator`.

A post-terminal assert (`${id.output} ...`) is lowered onto the END node and fired through
`eval_node`, so its locator names `__end__` (the boundary node it rides) with the author's
original assert text as the key. A boundary assert (`${input.X} ...`) is lowered onto START's
ordinary `post_asserts` (rewritten to `${output.X}`) and fired through `eval_node` when START
runs, so its locator names `__start__`. An input-coercion failure at the boundary sets an
`input_decl` locator, likewise stamped with `__start__` by the eval_node funnel.
"""

from pathlib import Path

from agent_composer.compose import load_flow
from agent_composer.run import run_flow
from agent_composer.tools import install_ask_user

_ERRORS = Path(__file__).resolve().parents[1] / "seeds" / "errors"


def test_post_assert_sets_run_result_locator():
    text = (_ERRORS / "e19-false-post-assert.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X"})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "assert"
    # The post assert now rides the END node (lowered onto `end.post_asserts`), so its locator
    # names `__end__` — the key stays the author's ORIGINAL text (not the lowered synth form).
    assert result.locator.node == "__end__"
    assert result.locator.key == "calc.output > 100"


def test_input_coercion_sets_input_decl_locator():
    text = (_ERRORS / "e08-input-type-mismatch.yaml").read_text()
    result = run_flow(load_flow(text), {"topic": "X", "window": "soon"})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "input_decl"
    assert result.locator.node == "__start__"          # eval_node funnel fills START's id
    assert result.locator.key == "window"


def test_boundary_assert_sets_assert_locator():
    text = (_ERRORS / "e18-false-boundary-assert.yaml").read_text()
    result = run_flow(load_flow(text), {"window": -5})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "assert"
    assert result.locator.node == "__start__"          # boundary assert rides START's post_asserts
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


def test_human_input_wrong_type_answer_sets_field_locator():
    # A host answer that fails the HUMAN_INPUT node's declared `output:` Type is
    # rejected at the ordinary write boundary with the field locator preserved.
    loaded = load_flow(_RESUME_WRONG_TYPE)
    with install_ask_user(lambda **request: "not-an-int"):
        result = run_flow(loaded, {})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "field"
    assert result.locator.node == "ask"
    assert result.locator.key == "output"
