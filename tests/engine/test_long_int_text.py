"""An expression that writes an int over 4300 digits as text fails with yian's digit limit message."""

import re
import sys

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian import load_flow, run_flow
from yian.run import resume_flow


LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"


@pytest.fixture
def python_limit_off():
    """Turn off Python's own int/str digit limit for one test."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(before)


@pytest.fixture
def python_limit_lowered():
    """Lower Python's own int/str digit limit to 1000 digits for one test."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(1000)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(before)


def _assert_clean(message: str) -> None:
    """Check a message shows no digits of the long int and no Python limit hint."""
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None


def _run(text: str, tmp_path, *, learnable: bool = False, tools=None):
    """Run the flow `text` with `n = 4999`, check its log resumes, and return the result."""
    flow = load_flow(text)
    path = tmp_path / "run.jsonl"
    result = run_flow(flow, {"n": 4999}, log=path, learnable=learnable, tools=tools)
    resumed = resume_flow(flow, log=path, tools=tools)
    assert resumed.status == result.status
    if result.failure is not None:
        assert resumed.failure.message == result.failure.message
    return result


def _assert_fails(result, code: str, node_id: str, message: str) -> None:
    """Assert the run failed `node_id` with `code` and exactly `message`, without digits."""
    assert result.status == "failed"
    assert result.failure.code == code
    assert result.failure.node_id == node_id
    assert result.failure.message == message
    _assert_clean(result.failure.message)


def _binding_flow(binding: str) -> str:
    """Return a flow whose `c` CODE node binds its input `x` to `binding`."""
    return f"""
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: {binding}
    code: |
      return 1
output: ${{c.output}}
"""


def test_a_long_int_in_a_text_binding_fails_at_the_binding(tmp_path):
    result = _run(_binding_flow("'v=${10 ** input.n}'"), tmp_path)

    _assert_fails(result, "binding.evaluate", "c", LIMIT_MSG)
    assert result.failure.trace[-1].key.endswith("/s:c/s:input/s:x/s:binding")


def test_a_long_int_in_a_text_binding_fails_with_pythons_limit_off(python_limit_off, tmp_path):
    result = _run(_binding_flow("'v=${10 ** input.n}'"), tmp_path)

    _assert_fails(result, "binding.evaluate", "c", LIMIT_MSG)


class _Model:
    """A chat model that answers every call with `ok`."""

    def invoke(self, messages):
        return AIMessage(content="ok")


def test_a_long_int_in_an_agent_prompt_fails_the_node(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model())
    result = _run("""
id: long
name: long
llm_config: {provider: openai, model: test-model}
input:
  n: !int
nodes:
  a:
    kind: agent
    prompt: "v=${10 ** n}"
    input:
      n: ${input.n}
    output: str
output: ${a.output}
""", tmp_path)

    _assert_fails(result, "node.execute", "a", LIMIT_MSG)


def test_a_long_int_in_a_human_input_question_fails_the_node(tmp_path):
    result = _run("""
id: long
name: long
input:
  n: !int
nodes:
  ask:
    kind: human_input
    prompt: "v=${10 ** n}"
    input:
      n: ${input.n}
    output: str
output: ${ask.output}
""", tmp_path, tools={"ask_user": lambda **request: "yes"})

    _assert_fails(result, "node.execute", "ask", LIMIT_MSG)


@pytest.mark.parametrize("binding, message", [
    ("""'${"%d" % (10 ** input.n)}'""", "cannot compute '%d' % <int with 5000 digits>: " + LIMIT_MSG),
    (
        """'${"%s" % [10 ** input.n]}'""",
        "cannot compute '%s' % <list holding an int with 5000 digits>: " + LIMIT_MSG,
    ),
], ids=["int", "list"])
def test_formatting_a_long_int_with_percent_fails(binding, message, tmp_path):
    result = _run(_binding_flow(binding), tmp_path)

    _assert_fails(result, "binding.evaluate", "c", message)


@pytest.mark.parametrize("binding, callee", [
    ("""'${join([10 ** input.n, 1], ",")}'""", "join"),
    ("'${render_as_json([1, 10 ** input.n])}'", "render_as_json"),
], ids=["join", "render_as_json"])
def test_a_builtin_writing_a_long_int_as_text_fails(binding, callee, tmp_path):
    result = _run(_binding_flow(binding), tmp_path)

    _assert_fails(
        result, "binding.evaluate", "c", f"expression function {callee!r} failed: " + LIMIT_MSG
    )


@pytest.mark.parametrize("binding", [
    """'${"%d" % (10 ** input.n)}'""",
    """'${join([10 ** input.n, 1], ",")}'""",
    "'${render_as_json([1, 10 ** input.n])}'",
], ids=["percent", "join", "render_as_json"])
def test_writing_a_long_int_as_text_fails_with_pythons_limit_off(
    binding, python_limit_off, tmp_path
):
    result = _run(_binding_flow(binding), tmp_path)

    assert result.status == "failed"
    assert result.failure.message.endswith(LIMIT_MSG)
    _assert_clean(result.failure.message)


_LEARNABLE_FLOW = """
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: ${(10 ** input.n) % 7}
    code: |
      return inputs["x"]
output: ${c.output}
"""


@pytest.mark.parametrize("limit", ["default", "off"])
def test_a_learnable_run_with_a_long_intermediate_succeeds_and_resumes(limit, request, tmp_path):
    if limit == "off":
        request.getfixturevalue("python_limit_off")

    result = _run(_LEARNABLE_FLOW, tmp_path, learnable=True)

    assert result.status == "succeeded", result.failure
    assert result.output == 10**4999 % 7


_PYTHON_LIMIT_MSG = "cannot render the {} value as text: it holds an int longer than Python's digit limit"


@pytest.mark.parametrize("binding, message", [
    ("'v=${10 ** input.m}'", _PYTHON_LIMIT_MSG.format("int")),
    (
        """'${join([10 ** input.m, 1], ",")}'""",
        "expression function 'join' failed: " + _PYTHON_LIMIT_MSG.format("list"),
    ),
    (
        "'${render_as_json([1, 10 ** input.m])}'",
        "expression function 'render_as_json' failed: " + _PYTHON_LIMIT_MSG.format("list"),
    ),
    (
        """'${"%d" % (10 ** input.m)}'""",
        "cannot compute '%d' % <int too large to display>: " + _PYTHON_LIMIT_MSG.format("int"),
    ),
], ids=["text", "join", "render_as_json", "percent"])
def test_an_int_under_yians_limit_that_python_refuses_fails_the_run(
    binding, message, python_limit_lowered, tmp_path
):
    # A host lowered Python's limit below yian's: a 2000-digit int is a yian value
    # Python can't write as text.
    flow = load_flow(f"""
id: long
name: long
input:
  m: !int
nodes:
  c:
    kind: code
    input:
      x: {binding}
    code: |
      return 1
output: ${{c.output}}
""")
    result = run_flow(flow, {"m": 1999}, log=tmp_path / "run.jsonl")

    _assert_fails(result, "binding.evaluate", "c", message)


def test_an_int_over_yians_limit_fails_with_yians_message_under_a_lowered_limit(
    python_limit_lowered, tmp_path
):
    result = _run(_binding_flow("'v=${10 ** input.n}'"), tmp_path)

    _assert_fails(result, "binding.evaluate", "c", LIMIT_MSG)
