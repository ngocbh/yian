"""Unit tests for inline CODE — the in-process path of `CodeNode`.

Inline mode: the author writes a **bare body** that reads the `inputs` dict and `return`s
a value; `CodeNode` wraps it as `def main(inputs):`, compiles it once, and calls
`main(inputs)` **in-process**. These tests pin execution plus identity-owned traceback
attribution; the error *seeds* (`e27`-`e32`) pin loading and running end-to-end.
"""

import textwrap

import pytest

from yian.compose import LoadError, load_flow
from yian.diagnostics import OriginRef
from yian.run import run_flow
from yian.nodes.base import Output
from yian.nodes.code.node import (
    CodeNode,
    classify_code_source,
    select_inline_code_origin,
)


def _run(code: str, inputs: dict, **kw):
    return CodeNode("n", code=code, **kw).run(inputs)


def _failure_origin(node: CodeNode, inputs: dict) -> OriginRef | None:
    """Run one CODE node and ask it to attribute the original exception."""
    try:
        node.run(inputs)
    except Exception as exc:
        return node.exception_origin(exc)
    raise AssertionError("CODE node unexpectedly succeeded")


class _FalseyOriginRef(OriginRef):
    """Valid origin whose truth value must not affect attribution."""

    def __bool__(self):
        return False


class _RaisingTruthOriginRef(OriginRef):
    """Valid origin that exposes accidental truth-value inspection."""

    def __bool__(self):
        raise AssertionError("origin truthiness must not be inspected")


# --- classification --------------------------------------------------------------------- #


def test_classify_reference_inline_reject():
    assert classify_code_source("pkg.mod:fn") == "reference"
    assert classify_code_source("mymod:myfunc") == "reference"
    assert classify_code_source("return inputs['x']") == "inline"
    assert classify_code_source("x = 1\nreturn x") == "inline"
    with pytest.raises(ValueError, match="did you mean 'pkg.mod:helper'"):
        classify_code_source("pkg.mod.helper")  # dotted, no colon -> reject


# --- the in-process entrypoint (one-dict `main(inputs)`) -------------------------------- #


def test_inline_reads_inputs_dict_and_returns():
    out = _run("return f\"{inputs['label']}: {inputs['rating']:.1f}\"", {"label": "BUY", "rating": 4.25})
    assert isinstance(out, Output) and out.value == "BUY: 4.2"


def test_inline_calling_convention_matches_reference():
    # inline is called with ONE dict (like reference `func(inputs)`), NOT unpacked kwargs.
    out = _run("return sorted(inputs.keys())", {"b": 1, "a": 2})
    assert out.value == ["a", "b"]


def test_inline_multi_statement_body_with_helpers():
    body = "def label(r):\n    return 'pos' if r >= 0 else 'neg'\nreturn label(inputs['r'])"
    assert _run(body, {"r": -1}).value == "neg"


def test_inline_stdlib_import_in_body_works_in_process():
    assert _run("import math\nreturn math.sqrt(inputs['x'])", {"x": 9}).value == 3.0


def test_inline_body_that_prints_still_returns_value(capsys):
    out = _run("print('side effect')\nreturn inputs['x'] * 2", {"x": 21})
    assert out.value == 42
    assert "side effect" in capsys.readouterr().out


# --- load-time gates -------------------------------------------------------------------- #


def test_no_return_body_rejected_at_construction():
    with pytest.raises(ValueError, match="has no `return`"):
        CodeNode("n", code="y = inputs['a'] + 1")


def test_return_only_in_nested_def_still_rejected():
    # a helper's return is not main's — the gate must not descend into nested scopes.
    with pytest.raises(ValueError, match="has no `return`"):
        CodeNode("n", code="def helper():\n    return 1")


def test_syntax_error_rejected_at_construction():
    with pytest.raises(SyntaxError):
        CodeNode("n", code="return (")


# --- source-neutral exception attribution --------------------------------------------- #


def test_wrapped_line_selector_uses_exact_whole_then_node_fallback():
    whole = OriginRef("code")
    lines = (OriginRef("line:0"), OriginRef("line:1"))
    node = OriginRef("node")

    assert select_inline_code_origin(
        3,
        code_origin=whole,
        code_line_origins=lines,
        node_origin=node,
    ) is lines[1]
    assert select_inline_code_origin(
        1,
        code_origin=whole,
        code_line_origins=lines,
        node_origin=node,
    ) is whole
    assert select_inline_code_origin(
        4,
        code_origin=whole,
        code_line_origins=lines[:1],
        node_origin=node,
    ) is whole
    assert select_inline_code_origin(
        2,
        code_origin=None,
        code_line_origins=(),
        node_origin=node,
    ) is node
    assert (
        select_inline_code_origin(
            True,
            code_origin=None,
            code_line_origins=(),
            node_origin=None,
        )
        is None
    )


@pytest.mark.parametrize(
    ("code", "expected_line"),
    [
        (
            "def helper():\n    raise RuntimeError('boom')\nhelper()\nreturn 0",
            1,
        ),
        ("fail = lambda: 1 / 0\nreturn fail()", 0),
        ("return [1 / value for value in inputs['values']]", 0),
    ],
    ids=["nested-helper", "lambda", "comprehension"],
)
def test_inline_exception_selects_deepest_owned_code_line(code, expected_line):
    line_origins = tuple(
        OriginRef(f"line:{index}") for index in range(len(code.splitlines()))
    )
    node = CodeNode(
        "n",
        code=code,
        code_origin=OriginRef("code"),
        code_line_origins=line_origins,
    )

    origin = _failure_origin(node, {"values": [0]})

    assert origin is line_origins[expected_line]


def test_structurally_equal_foreign_code_cannot_impersonate_owned_code():
    code = (
        "if inputs['foreign']:\n"
        "    raise RuntimeError('boom')\n"
        "return inputs['call']({'foreign': True})"
    )
    foreign_namespace: dict = {}
    foreign_compiled = compile(
        "def main(inputs):\n" + textwrap.indent(code, "    "),
        "same.py",
        "exec",
    )
    exec(foreign_compiled, foreign_namespace)
    foreign_main = foreign_namespace["main"]
    lines = tuple(OriginRef(f"line:{index}") for index in range(3))
    node = CodeNode(
        "n",
        code=code,
        source_file="same.py",
        code_origin=OriginRef("code"),
        code_line_origins=lines,
    )
    owned_main = next(
        owned for owned in node._owned_code_objects if owned.co_name == "main"
    )
    assert foreign_main.__code__ is not owned_main
    assert foreign_main.__code__ == owned_main
    assert hash(foreign_main.__code__) == hash(owned_main)

    origin = _failure_origin(
        node,
        {"foreign": False, "call": foreign_main},
    )

    assert origin is lines[2]


def test_dynamic_same_filename_code_falls_back_to_owned_call_site():
    foreign = compile("raise RuntimeError('boom')", "same.py", "exec")
    code = "exec(inputs['foreign'], {})\nreturn 0"
    lines = (OriginRef("line:0"), OriginRef("line:1"))
    node = CodeNode(
        "n",
        code=code,
        source_file="same.py",
        code_origin=OriginRef("code"),
        code_line_origins=lines,
    )

    assert _failure_origin(node, {"foreign": foreign}) is lines[0]


def test_inline_non_owned_failure_uses_whole_code_origin():
    whole = OriginRef("code")
    node = CodeNode(
        "n",
        code="return 0",
        code_origin=whole,
        code_line_origins=(OriginRef("line:0"),),
    )

    # Raised outside the node: no traceback, so no frame of the authored code.
    assert node.exception_origin(RuntimeError("outside")) is whole


def test_reference_failure_uses_code_then_node_fallback():
    code_origin = OriginRef("code:reference")
    reference = CodeNode(
        "reference",
        code="tests.engine._compose_codefns:boom",
        code_origin=code_origin,
    )
    reference.origin = OriginRef("node:reference")
    without_code_origin = CodeNode(
        "reference-fallback",
        code="tests.engine._compose_codefns:boom",
    )
    without_code_origin.origin = OriginRef("node:reference-fallback")
    source_free = CodeNode("inline", code="raise RuntimeError('boom')\nreturn 0")

    assert _failure_origin(reference, {}) is code_origin
    assert _failure_origin(without_code_origin, {}) is without_code_origin.origin
    assert _failure_origin(source_free, {}) is None


@pytest.mark.parametrize(
    "code_origin",
    (_FalseyOriginRef("falsey"), _RaisingTruthOriginRef("raising")),
)
def test_reference_failure_does_not_inspect_origin_truthiness(code_origin):
    node = CodeNode(
        "reference",
        code="tests.engine._compose_codefns:boom",
        code_origin=code_origin,
    )
    node.origin = OriginRef("node:reference")

    assert _failure_origin(node, {}) is code_origin


# --- the output boundary, not the node, checks the returned value ---------------------- #


def test_inline_return_comes_back_unchecked():
    value = _run("return {'ok': 1, 'bad': object()}", {}).value
    assert value.keys() == {"ok", "bad"}
    assert value["ok"] == 1
    assert type(value["bad"]) is object


# --- reference mode is untouched -------------------------------------------------------- #


def test_reference_mode_still_dispatches_in_process():
    n = CodeNode("r", code="tests.engine._compose_codefns:echo")
    assert n._mode == "reference"
    assert n.run({"topic": "hi"}).value == "hi"  # echo returns inputs["topic"]


# --- integration through the loader (framing + run) ------------------------------------- #


_INLINE_FLOW = """\
id: t
name: t
input:
  rating: !float
nodes:
  verdict:
    kind: code
    input:
      rating: ${input.rating}
    output: str
    code: |
      return f"r={inputs['rating']:.1f}"
output: ${verdict.output}
"""


def test_inline_flow_loads_and_runs():
    result = run_flow(load_flow(_INLINE_FLOW), {"rating": 3.0})
    assert result.status == "succeeded"
    assert result.output == "r=3.0"


def test_inline_raise_becomes_failed_run():
    flow = _INLINE_FLOW.replace(
        'return f"r={inputs[\'rating\']:.1f}"', 'raise RuntimeError("boom")\n      return ""'
    )
    result = run_flow(load_flow(flow), {"rating": 3.0})
    assert result.status != "succeeded"
    assert "boom" in result.failure.message
