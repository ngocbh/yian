"""CASE value branches: a `then:`/`else:` may be a typed value instead of a node.

An unquoted bare word names a sibling node; an unquoted word that names no node is a
located load error that suggests the nearest node and the quoted form. A quoted scalar
(`"..."`, `'...'`, or a block `|`/`>`) is text, and a non-string scalar, mapping, or list
is a value. Each value lifts at load to a synthetic CODE node `__value_<n>` that becomes
the branch target; `${...}` spans in the value bind as its inputs. These tests cover the
classification errors (unknown word, reserved prefixes, a value without `output:`, an
unquoted typed binding, a quoted or nested call with spans, a value reading its own CASE,
a bad ref inside a value) and runs that return the taken value, skip the other branch,
and work inside a `defs:` body and an inline LOOP body.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.events import NodeSkipped, NodeStarted
from yian.run import run_flow


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _started(result) -> set[str]:
    """Return the ids of every node that started in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeStarted)}


def _skipped(result) -> set[str]:
    """Return the ids of every node skipped in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeSkipped)}


def _flow(case_body: str, *, inputs: str = "  cmd: str\n", extra_nodes: str = "") -> str:
    """Return a flow with one `route` case (indented body), a `reply` node, and an output."""
    return (
        "id: f\n"
        "name: f\n"
        "input:\n"
        f"{inputs}"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        f"{case_body}"
        "  reply:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n"
        "      topic: replied\n"
        "    output: str\n"
        f"{extra_nodes}"
        "output: ${route.output}\n"
    )


# --------------------------------------------------------------------------- #
# Classification errors
# --------------------------------------------------------------------------- #


def test_unknown_then_word_suggests_nearest_node_and_quoting():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: replyy\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'route' then: `replyy` names no node. Did you mean `reply`?" in message
    assert 'quote it: `then: "replyy"`' in message
    assert _error_line(caught.value) == _line_of(text, "then: replyy")


def test_unknown_else_word_suggests_nearest_node_and_quoting():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "    else: replyy\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'route' else: `replyy` names no node. Did you mean `reply`?" in message
    assert 'quote it: `else: "replyy"`' in message
    assert _error_line(caught.value) == _line_of(text, "else: replyy")


def test_unknown_sentence_gets_quote_hint_without_suggestion():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: The user declined\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "`The user declined` names no node." in message
    assert "Did you mean" not in message
    assert 'If you meant the text "The user declined", quote it: ' in message
    assert '`then: "The user declined"`' in message


@pytest.mark.parametrize(
    ("node_id", "prefix"), [("__value_0", "__value_"), ("__case_x", "__case_")]
)
def test_authored_node_with_reserved_prefix_is_rejected(node_id, prefix):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n",
        extra_nodes=(
            f"  {node_id}:\n"
            "    kind: code\n"
            "    code: tests.engine._compose_codefns:echo\n"
            "    input:\n"
            "      topic: x\n"
            "    output: str\n"
        ),
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert f"node id {node_id!r} uses the reserved prefix {prefix!r}" in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, f"{node_id}:")


@pytest.mark.parametrize("else_value", ['"approve"', "null"])
def test_value_branch_without_output_type_is_rejected(else_value):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        f"    else: {else_value}\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' else: a value branch needs the case's type; "
        "add `output: <Type>` to 'route'"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, f"else: {else_value}")


def test_unquoted_typed_binding_branch_is_rejected_with_quote_hint():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: int\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: int = ${input.n}\n"
        "    else: reply\n",
        inputs="  cmd: str\n  n: int\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' then: `int = ${input.n}` is not a branch; the case's "
        "`output:` gives the type. If you meant the text, quote it: "
        '`then: "int = ${input.n}"`'
    ) in str(caught.value)


def test_quoted_call_with_span_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "call(child, x=${input.cmd})"\n'
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' then: a quoted `call(...)` is text, so it can't hold `${...}` "
        "here; drop the quotes to call the flow, or reword the text"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "then: \"call(")


def test_nested_call_with_span_in_value_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      a: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        '          a: "call(child, x=${input.cmd})"\n'
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'route' then: a `call(...)` nested in a value" in message
    assert "then: call(...)" in message


def test_value_with_a_non_string_key_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: object\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        "          1: one\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "case node 'route' then: record keys must be strings (got 1)" in str(caught.value)


def test_value_reading_its_own_case_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      a: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        "          a: ${route.output.a}\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "case node 'route' then: the value reads 'route' itself" in str(caught.value)


def test_bad_ref_inside_value_is_rejected_at_the_branch():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      a: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        '          a: "${nope.output}"\n'
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    # Reference validation names the synth node; the frame points at the branch value,
    # which for a block mapping starts on the line after `then:`.
    assert "reference ${nope.output} uses unknown namespace 'nope'" in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "a: \"${nope.output}\"")


# --------------------------------------------------------------------------- #
# Quoted text is a value, never a node or a call
# --------------------------------------------------------------------------- #


def test_quoted_node_name_is_text():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "reply"\n'
        "    else: reply\n"
    )
    result = run_flow(load_flow(text), {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == "reply"
    assert "reply" not in _started(result)


@pytest.mark.parametrize(
    "then_text",
    [
        '"call(child, x=1)"',
        "'call(child, x=1)'",
        "|-\n          call(child, x=1)",
    ],
)
def test_quoted_call_without_span_is_text(then_text):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        "    else: reply\n"
    )
    result = run_flow(load_flow(text), {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == "call(child, x=1)"
    assert not any(nid.startswith("__call_") for nid in _started(result))


def test_quoted_lone_span_keeps_its_type():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: int\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "${input.n}"\n'
        "    else: reply\n",
        inputs="  cmd: str\n  n: int\n",
    )
    result = run_flow(load_flow(text), {"cmd": "a", "n": 7})
    assert result.status == "succeeded", result.failure
    assert result.output == 7


@pytest.mark.parametrize(
    ("then_text", "expected"),
    [
        ('"costs $$5"', "costs $5"),
        ('\n          a: "costs $$5"', {"a": "costs $5"}),
        ('\n          a: "costs $$5 ${input.cmd}"', {"a": "costs $5 a"}),
    ],
)
def test_double_dollar_renders_one_dollar(then_text, expected):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: object\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        "    else: reply\n"
    )
    result = run_flow(load_flow(text), {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_unquoted_date_is_its_iso_text():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: date\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: 2026-01-02\n"
        "    else: reply\n"
    )
    result = run_flow(load_flow(text), {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == "2026-01-02"


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def test_text_then_and_node_else_route_by_on():
    loaded = load_flow(
        _flow(
            "    on: ${input.cmd}\n"
            "    output: str\n"
            "    cases:\n"
            "      - when: a\n"
            '        then: "picked a"\n'
            "    else: reply\n"
        )
    )
    taken = run_flow(loaded, {"cmd": "a"})
    assert taken.status == "succeeded", taken.failure
    assert taken.output == "picked a"
    assert "reply" in _skipped(taken)
    other = run_flow(loaded, {"cmd": "b"})
    assert other.status == "succeeded", other.failure
    assert other.output == "replied"
    assert "__value_0" in _skipped(other)
    assert "__value_0" not in _started(other)


def test_record_value_with_span():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      reply: str\n"
        "      exited: bool\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        "          reply: ${input.text}\n"
        "          exited: true\n"
        "    else: reply\n",
        inputs="  cmd: str\n  text: str\n",
    )
    result = run_flow(load_flow(text), {"cmd": "a", "text": "hi"})
    assert result.status == "succeeded", result.failure
    assert result.output == {"reply": "hi", "exited": True}


def test_nested_list_with_span():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      items: list[str]\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        "          items:\n"
        '            - "${input.a}"\n'
        "            - x\n"
        "    else: reply\n",
        inputs="  cmd: str\n  a: str\n",
    )
    result = run_flow(load_flow(text), {"cmd": "a", "a": "first"})
    assert result.status == "succeeded", result.failure
    assert result.output == {"items": ["first", "x"]}


def test_else_null_with_optional_output():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: Optional[str]\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "    else: null\n"
    )
    loaded = load_flow(text)
    result = run_flow(loaded, {"cmd": "b"})
    assert result.status == "succeeded", result.failure
    assert result.output is None
    assert "__value_0" in _started(result)
    taken = run_flow(loaded, {"cmd": "a"})
    assert taken.output == "replied"


@pytest.mark.parametrize(
    ("output_type", "then_text", "expected"),
    [
        ("int", "3", 3),
        ("list[str]", "[]", []),
        ("int", "${input.n}", 5),
        ("bool", "false", False),
    ],
)
def test_scalar_and_list_values(output_type, then_text, expected):
    text = _flow(
        "    on: ${input.cmd}\n"
        f"    output: {output_type}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        "    else: reply\n",
        inputs="  cmd: str\n  n: int\n",
    )
    result = run_flow(load_flow(text), {"cmd": "a", "n": 5})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_value_branch_inside_defs_body():
    text = """\
id: f
name: f
input:
  cmd: str
defs:
  pick:
    input:
      cmd: str
    nodes:
      route:
        kind: case
        on: ${input.cmd}
        output: str
        cases:
          - when: a
            then: "picked a"
        else: "picked other"
    output: ${route.output}
nodes:
  run_pick:
    kind: call
    call: pick
    input:
      cmd: ${input.cmd}
output: ${run_pick.output}
"""
    loaded = load_flow(text)
    assert run_flow(loaded, {"cmd": "a"}).output == "picked a"
    assert run_flow(loaded, {"cmd": "z"}).output == "picked other"


def test_value_branch_inside_inline_loop_body():
    text = """\
id: f
name: f
nodes:
  loop:
    kind: loop
    input:
      n: 0
      exited: false
    while: not ${exited}
    max: 10
    nodes:
      bump:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
        output:
          n: int
          exited: bool
      route:
        kind: case
        output:
          n: int
          exited: bool
        cases:
          - when: "${bump.output.n} >= 2"
            then:
              n: ${bump.output.n}
              exited: true
        else:
          n: ${bump.output.n}
          exited: false
    output:
      n: ${route.output.n}
      exited: ${route.output.exited}
output: ${loop.output}
"""
    result = run_flow(load_flow(text), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2, "exited": True}
