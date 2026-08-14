"""Unit tests for inline CODE — the in-process path of `CodeNode`.

Inline mode: the author writes a **bare body** that reads the `inputs` dict and `return`s
a value; `CodeNode` wraps it as `def main(inputs):`, compiles it (padded for absolute-YAML
tracebacks), and calls `main(inputs)` **in-process**. These tests pin the mechanism at the
node level (construction gates + `run`); the error *seeds* (`e27`-`e32`) pin the same
behaviors end-to-end through `load_flow`/`run_flow`.
"""

import logging
from datetime import datetime, timezone

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow
from agent_composer.nodes.base import Output
from agent_composer.nodes.code.node import CodeNode, classify_code_source


def _run(code: str, inputs: dict, **kw):
    return CodeNode("n", code=code, code_line=2, **kw).run(inputs)


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
        CodeNode("n", code="y = inputs['a'] + 1", code_line=2)


def test_return_only_in_nested_def_still_rejected():
    # a helper's return is not main's — the gate must not descend into nested scopes.
    with pytest.raises(ValueError, match="has no `return`"):
        CodeNode("n", code="def helper():\n    return 1", code_line=2)


def test_syntax_error_rejected_at_construction():
    with pytest.raises(SyntaxError):
        CodeNode("n", code="return (", code_line=2)


# --- deep serialize-once check (fails AT the node) -------------------------------------- #


def test_nested_non_serializable_return_fails_at_node():
    with pytest.raises(TypeError, match="non-serializable"):
        _run("return {'ok': 1, 'bad': object()}", {})


def test_top_level_non_serializable_return_fails_at_node():
    with pytest.raises(TypeError, match="non-serializable"):
        _run("return object()", {})


def test_serializable_returns_pass_including_coerced_set():
    # a nested set is coerced to a list by the canonical typed-value serializer — OK.
    assert _run("return {'xs': {1, 2, 3}}", {}).value == {"xs": {1, 2, 3}}
    assert _run("return [1, 'a', {'k': 2.5}]", {}).value == [1, "a", {"k": 2.5}]


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
  rating: float
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


def test_inline_raise_is_framed_at_the_yaml_line():
    flow = _INLINE_FLOW.replace(
        'return f"r={inputs[\'rating\']:.1f}"', 'raise RuntimeError("boom")\n      return ""'
    )
    result = run_flow(load_flow(flow), {"rating": 3.0})
    assert result.status != "succeeded"
    assert "boom" in (result.error or "")


# --- CODE-owned watchdog (soft-budget logger) ------------------------------------------- #


def test_watchdog_logs_a_slow_node(caplog):
    from agent_composer.nodes.code.watchdog import node_watchdog
    import time

    with caplog.at_level(logging.WARNING, logger="agent_composer.nodes.code.watchdog"):
        with node_watchdog("slowpoke", budget=0.02):
            time.sleep(0.08)
    assert any("slowpoke" in r.getMessage() for r in caplog.records)


def test_watchdog_silent_on_a_fast_node(caplog):
    from agent_composer.nodes.code.watchdog import node_watchdog
    import time

    with caplog.at_level(logging.WARNING, logger="agent_composer.nodes.code.watchdog"):
        with node_watchdog("speedy", budget=0.5):
            pass
        time.sleep(0.02)
    assert not any("speedy" in r.getMessage() for r in caplog.records)


def test_watchdog_wraps_both_code_modes_but_not_other_blocking_kinds(monkeypatch):
    """The timer belongs to CODE execution, not the kind-blind eval seam."""
    import threading
    from types import SimpleNamespace

    from agent_composer.nodes.agent import AgentNode
    from agent_composer.nodes.agent.modes import MODES
    from agent_composer.nodes.code import CodeNode
    from agent_composer.nodes.human_input import HumanInputNode
    from agent_composer.nodes.wait import WaitNode
    from agent_composer.nodes.code import watchdog
    from agent_composer.runtime.eval_node import eval_node
    from agent_composer.typesys.pool import VariablePool

    timers = []

    class _Timer:
        def __init__(self, interval, function):
            self.interval = interval
            self.function = function
            self.daemon = False
            self.started = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(watchdog.threading, "Timer", _Timer)
    monkeypatch.setattr(
        "agent_composer.nodes.wait.node._utc_now",
        lambda: datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    monkeypatch.setitem(MODES, "plain", lambda ctx: Output("agent answer"))
    monkeypatch.setattr(
        "agent_composer.nodes.human_input.node.invoke_tool_trusted",
        lambda tool_id, metadata: "human answer",
    )

    empty_flow = SimpleNamespace(wiring={"agent": {}, "human": {}})
    assert eval_node(AgentNode("agent"), empty_flow, VariablePool()) == Output(
        "agent answer"
    )
    assert eval_node(
        HumanInputNode("human", prompt="?"),
        empty_flow,
        VariablePool(),
    ) == Output("human answer")
    wait_result = eval_node(
        WaitNode("wait"),
        SimpleNamespace(wiring={"wait": {"until": "2026-01-01T00:00:00+00:00"}}),
        VariablePool(),
        stop_event=threading.Event(),
    )
    assert timers == []
    inline_result = CodeNode("inline", code="return inputs['v']").run({"v": 1})
    reference_result = CodeNode(
        "reference", code="tests.engine._fakes:passthrough"
    ).run({"v": 2})

    assert wait_result == Output(None)
    assert inline_result.value == 1
    assert reference_result.value == 2
    assert [timer.interval for timer in timers] == [30.0, 30.0]
    assert all(timer.started and timer.cancelled for timer in timers)
