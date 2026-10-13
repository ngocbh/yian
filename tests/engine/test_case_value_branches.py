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

A CASE that declares `output: T` gets one synthetic result node `__case_<id>` that
`${case.output}` reads: every taken branch is checked against T at runtime
(`binding.type`, located at the CASE's `output:`), read or not, and readers see T.

At load, a branch that can never produce T is rejected at the branch: a literal value
that does not fit, a node, call, or inner case whose type is disjoint from T, a span in
a value whose inferred type is disjoint from its position's type, and a span that reads
another branch of the same CASE without a `:-` default.
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


def test_value_branch_without_output_type_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        '    else: "approve"\n'
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' else: a value branch needs the case's type; "
        "add `output: <Type>` to 'route'"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, 'else: "approve"')


@pytest.mark.parametrize(
    ("then_text", "else_text", "label"),
    [("reply", "null", "else:"), ("null", "reply", "then:"), ("", "reply", "then:")],
)
def test_empty_branch_without_output_type_is_rejected(then_text, else_text, label):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        f"    else: {else_text}\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        f"case node 'route' {label} is empty; name a node, or add "
        "`output: Optional[...]` for a null value"
    ) in str(caught.value)


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


def test_nested_call_without_span_in_value_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      a: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        "          a: call(child, x=1)\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "case node 'route' then: a `call(...)` nested in a value" in str(caught.value)


@pytest.mark.parametrize(
    ("then_text", "else_text", "label", "needle"),
    [
        ("call(child, x=1) | x", "reply", "then:", "then: call("),
        ("reply", "call(child, x=${input.cmd}) | y", "else:", "else: call("),
    ],
)
def test_unquoted_call_that_is_not_one_directive_is_rejected(
    then_text, else_text, label, needle
):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        f"    else: {else_text}\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert f"case node 'route' {label} a branch target must be a node id or a single call" in (
        message
    )
    assert "no longer supported" not in message
    assert _error_line(caught.value) == _line_of(text, needle)


def test_malformed_span_in_value_names_the_branch():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "hello ${input.cmd"\n'
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert message.startswith("case node 'route' then: ")
    assert "__value_" not in message
    assert _error_line(caught.value) == _line_of(text, "then: \"hello")


@pytest.mark.parametrize("number", [".inf", "-.inf", ".nan"])
def test_non_finite_number_in_value_is_rejected(number):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: object\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        f"          a: {number}\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "case node 'route' then: numbers must be finite" in str(caught.value)


def test_self_containing_alias_in_value_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: object\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: &R [*R]\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "case node 'route' then: a branch value can't contain itself" in str(caught.value)


def test_unknown_word_that_reads_as_a_number_suggests_yaml_number():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: float\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: 1e3\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "`1e3` names no node." in message
    assert "or write a YAML number such as 1.0e+3" in message


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
        "    else:\n"
        '      a: "other"\n'
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
        "    else: 0\n",
        inputs="  cmd: str\n  n: int\n",
    )
    result = run_flow(load_flow(text), {"cmd": "a", "n": 7})
    assert result.status == "succeeded", result.failure
    assert result.output == 7


@pytest.mark.parametrize(
    ("output_type", "then_text", "else_text", "expected"),
    [
        ("str", '"costs $$5"', "reply", "costs $5"),
        ("object", '\n          a: "costs $$5"', "{}", {"a": "costs $5"}),
        ("object", '\n          a: "costs $$5 ${input.cmd}"', "{}", {"a": "costs $5 a"}),
    ],
)
def test_double_dollar_renders_one_dollar(output_type, then_text, else_text, expected):
    text = _flow(
        "    on: ${input.cmd}\n"
        f"    output: {output_type}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        f"    else: {else_text}\n"
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
        "    else:\n"
        '      reply: "no"\n'
        "      exited: false\n",
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
        "    else:\n"
        "      items: []\n",
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
    ("output_type", "then_text", "else_text", "expected"),
    [
        ("int", "3", "0", 3),
        ("list[str]", "[]", "[x]", []),
        ("int", "${input.n}", "0", 5),
        ("bool", "false", "true", False),
    ],
)
def test_scalar_and_list_values(output_type, then_text, else_text, expected):
    text = _flow(
        "    on: ${input.cmd}\n"
        f"    output: {output_type}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        f"    else: {else_text}\n",
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
      note: start
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
          note: str
        cases:
          - when: "${bump.output.n} >= 2"
            then:
              n: ${bump.output.n}
              exited: true
              note: "done"
        else:
          n: ${bump.output.n}
          exited: false
          note: 'more'
    output: ${route.output}
output: ${loop.output}
"""
    result = run_flow(load_flow(text), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2, "exited": True, "note": "done"}


# --------------------------------------------------------------------------- #
# A typed CASE's result node
# --------------------------------------------------------------------------- #


def _literal_flow(output: str = "${route.output}", extra_nodes: str = "") -> str:
    """Return a flow whose `Literal[yes, no]` case routes `else:` to a CODE node that
    returns `"maybe"`."""
    return (
        "id: f\n"
        "name: f\n"
        "input:\n"
        "  cmd: str\n"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${input.cmd}\n"
        "    output: Literal[yes, no]\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "yes"\n'
        "    else: other\n"
        "  other:\n"
        "    kind: code\n"
        "    code: |\n"
        '      return "maybe"\n'
        "    output: str\n"
        f"{extra_nodes}"
        f"output: {output}\n"
    )


def _assert_bad_branch_fails(loaded, text: str) -> None:
    """Run the bad branch and check it fails `binding.type` at the result node, located
    at the case's `output:` line."""
    result = run_flow(loaded, {"cmd": "b"})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "__case_route"
    assert "binding 'route'" in result.failure.message
    diagnostic = loaded.diagnose(result.failure)
    assert diagnostic.frames, diagnostic.notes
    frame = diagnostic.frames[-1]
    assert frame.text == text
    assert frame.range.covered_lines(text) == [_line_of(text, "output: Literal[yes, no]")]


def test_typed_case_checks_a_narrower_branch_at_runtime():
    text = _literal_flow()
    loaded = load_flow(text)
    _assert_bad_branch_fails(loaded, text)
    good = run_flow(loaded, {"cmd": "a"})
    assert good.status == "succeeded", good.failure
    assert good.output == "yes"


def test_typed_case_checks_its_branch_even_when_unread():
    extra = (
        "  side:\n"
        "    kind: code\n"
        "    code: |\n"
        '      return "side"\n'
        "    output: str\n"
    )
    text = _literal_flow(output="${side.output}", extra_nodes=extra)
    loaded = load_flow(text)
    _assert_bad_branch_fails(loaded, text)
    assert run_flow(loaded, {"cmd": "a"}).output == "side"


def test_typed_case_read_inside_text():
    text = _literal_flow(output="\n  out: 'got ${route.output}'")
    loaded = load_flow(text)
    _assert_bad_branch_fails(loaded, text)
    good = run_flow(loaded, {"cmd": "a"})
    assert good.status == "succeeded", good.failure
    assert good.output == {"out": "got yes"}


def test_untyped_case_gets_no_result_node():
    loaded = load_flow(
        _flow(
            "    on: ${input.cmd}\n"
            "    cases:\n"
            "      - when: a\n"
            "        then: reply\n"
            "    else: reply\n"
        )
    )
    assert not any(nid.startswith("__case_") for nid in loaded.compiled.nodes)
    assert loaded.compiled.outputs[0].from_ == "${reply.output}"


def test_typed_case_reader_sees_the_case_type_in_a_call():
    text = """\
id: f
name: f
input:
  cmd: str
defs:
  child:
    input:
      n: str
    nodes:
      c:
        kind: code
        code: |
          return inputs["n"]
        input:
          n: ${input.n}
        output: str
    output: ${c.output}
nodes:
  route:
    kind: case
    on: ${input.cmd}
    output: int
    cases:
      - when: a
        then: 1
    else: 2
  use:
    kind: call
    call: child
    input:
      n: ${route.output}
output: ${use.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "binding 'n' — child expects 'str', source is 'int'" in str(exc.value)


def test_typed_case_disjoint_typed_reader_is_a_load_error():
    text = """\
id: f
name: f
input:
  cmd: str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    output: str
    cases:
      - when: a
        then: "x"
    else: "y"
  use:
    kind: code
    input:
      x: int = ${route.output}
    code: |
      return inputs["x"]
output: ${use.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "declared int, but the value is str" in str(exc.value)
    assert _error_line(exc.value) == _line_of(text, "x: int = ")


def test_typed_case_wider_reader_still_checks_the_case_type():
    text = """\
id: f
name: f
typedefs:
  Reply:
    reply: str
    n: int
input:
  cmd: str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    output: Reply
    cases:
      - when: a
        then: partial
    else:
      reply: "y"
      n: 1
  partial:
    kind: code
    code: |
      return {"reply": "x"}
    output: object
  use:
    kind: code
    input:
      x: object = ${route.output}
    code: |
      return inputs["x"]
output: ${use.output}
"""
    loaded = load_flow(text)
    result = run_flow(loaded, {"cmd": "a"})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "__case_route"
    assert "missing required" in result.failure.message
    assert "use" not in _started(result)
    good = run_flow(loaded, {"cmd": "b"})
    assert good.status == "succeeded", good.failure
    assert good.output == {"reply": "y", "n": 1}


def _record_case_flow(read: str) -> str:
    """Return a flow with a record-typed case whose output reads `read`."""
    return (
        "id: f\n"
        "name: f\n"
        "input:\n"
        "  cmd: str\n"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      reply: str\n"
        "      exited: bool\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        '          reply: "hi"\n'
        "          exited: true\n"
        "    else:\n"
        '      reply: "bye"\n'
        "      exited: false\n"
        f"output: {read}\n"
    )


def test_typed_case_field_read_runs():
    result = run_flow(load_flow(_record_case_flow("${route.output.reply}")), {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == "hi"


def test_typed_case_unknown_field_read_is_a_load_error():
    with pytest.raises(LoadError) as exc:
        load_flow(_record_case_flow("${route.output.nope}"))
    assert "reads unknown field 'nope' on record" in str(exc.value)


def test_typed_case_feeds_loop_seed_inline_call_and_map():
    text = """\
id: f
name: f
input:
  cmd: str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    output: list[str]
    cases:
      - when: a
        then:
          - x
          - y
    else: []
  seeded:
    kind: loop
    times: 1
    input:
      items: ${route.output}
    nodes:
      keep:
        kind: code
        code: |
          return inputs["items"]
        input:
          items: ${input.items}
        output: list[str]
    output:
      items: ${keep.output}
  wrapped:
    kind: call
    input:
      items: ${route.output}
    nodes:
      count:
        kind: code
        code: |
          return len(inputs["items"])
        input:
          items: ${input.items}
        output: int
    output: ${count.output}
  each:
    kind: map
    over: ${route.output}
    input:
      value: ${item}
    nodes:
      upper:
        kind: code
        code: |
          return inputs["value"].upper()
        input:
          value: ${input.value}
        output: str
    output: ${upper.output}
output:
  seeded: ${seeded.output}
  wrapped: ${wrapped.output}
  each: ${each.output}
"""
    result = run_flow(load_flow(text), {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == {
        "seeded": {"items": ["x", "y"]},
        "wrapped": 2,
        "each": ["X", "Y"],
    }


def test_typed_case_on_union_tag_still_narrows():
    text = """\
id: f
name: f
typedefs:
  Buy:
    action: Literal[buy]
    qty: int
  Hold:
    action: Literal[hold]
    note: str
  Decision: Union[Buy, Hold]
input:
  kind: str
nodes:
  d:
    kind: code
    input:
      kind: ${input.kind}
    code: |
      if inputs["kind"] == "buy":
          return {"action": "buy", "qty": 3}
      return {"action": "hold", "note": "wait"}
    output: Decision
  route:
    kind: case
    on: ${d.output.action}
    output: int
    cases:
      - when: buy
        then: qty
      - when: hold
        then: 0
  qty:
    kind: code
    input:
      q: ${d.output.qty}
    code: |
      return inputs["q"]
    output: int
output: ${route.output}
"""
    loaded = load_flow(text)
    assert run_flow(loaded, {"kind": "buy"}).output == 3
    assert run_flow(loaded, {"kind": "hold"}).output == 0


def test_bad_case_output_type_is_located():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: NotAType\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "x"\n'
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "NotAType" in str(exc.value)
    assert _error_line(exc.value) == _line_of(text, "output: NotAType")


def test_expand_case_outputs_adds_the_result_node_for_a_typed_case():
    from yian.compose import expand_case_outputs
    from yian.compose.parse.file import CaseDescriptor, CodeDescriptor
    from yian.nodes.contracts import ValueContract
    from yian.typesys.values import Type, ValueKind

    contract = ValueContract(Type.scalar(ValueKind.STRING))
    descriptors = {
        "gate": CaseDescriptor(
            id="gate",
            cases=[{"when": "${s.output} >= 0.5", "then": "pro"}],
            else_="caut",
            outputs="str",
        ),
        "c": CodeDescriptor(
            id="c",
            code="m:f",
            inputs={
                "whole": "${gate.output}",
                "dotted": "${gate.output.report}",
                "text": "pre ${gate.output} post",
            },
        ),
    }
    new, outputs = expand_case_outputs(
        descriptors, "${gate.output}", [], case_types={"gate": contract}
    )
    assert new["c"].inputs == {
        "whole": "${__case_gate.output}",
        "dotted": "${__case_gate.output.report}",
        "text": "pre ${__case_gate.output} post",
    }
    assert outputs == "${__case_gate.output}"
    result = new["__case_gate"]
    assert isinstance(result, CodeDescriptor)
    assert result.inputs == {"gate": "${pro.output | caut.output}"}
    assert result.input_types == {"gate": contract}
    assert result.outputs == "str"
    assert result.code == "return inputs['gate']"
    assert result.node_name == "gate result"


def test_typed_case_result_node_skips_when_its_case_is_skipped():
    text = """\
id: f
name: f
input:
  cmd: str
nodes:
  outer:
    kind: case
    on: ${input.cmd}
    cases:
      - when: a
        then: inner
    else: other
  inner:
    kind: case
    output: str
    cases:
      - when: "${input.cmd} == 'a'"
        then: "in-a"
    else: "in-else"
  other:
    kind: code
    code: |
      return "other"
    output: str
output: ${outer.output}
"""
    result = run_flow(load_flow(text), {"cmd": "b"})
    assert result.status == "succeeded", result.failure
    assert result.output == "other"
    assert "__case_inner" in _skipped(result)


def test_typed_case_without_a_resolvable_branch_is_rejected_even_unread():
    from yian.compose import expand_case_outputs
    from yian.compose.parse.file import CaseDescriptor
    from yian.nodes.contracts import ValueContract
    from yian.typesys.values import Type, ValueKind

    descriptors = {
        "a": CaseDescriptor(id="a", cases=[{"when": "${x}", "then": "b"}], outputs="str"),
        "b": CaseDescriptor(id="b", cases=[{"when": "${x}", "then": "a"}]),
    }
    contract = ValueContract(Type.scalar(ValueKind.STRING))
    with pytest.raises(LoadError, match="has no resolvable branch value"):
        expand_case_outputs(descriptors, None, [], case_types={"a": contract})


# --------------------------------------------------------------------------- #
# Branches must fit the CASE's `output:`
# --------------------------------------------------------------------------- #


def _int_node(node_id: str = "num") -> str:
    """Return a CODE node (indented under `nodes:`) whose output is `int`."""
    return (
        f"  {node_id}:\n"
        "    kind: code\n"
        "    code: |\n"
        "      return 3\n"
        "    output: int\n"
    )


def test_literal_number_into_str_output_is_rejected_with_quote_hint():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: 3\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'route' then: the value 3 is not a valid str" in message
    assert 'quote it to keep it as text: `then: "3"`' in message
    assert _error_line(caught.value) == _line_of(text, "then: 3")


def test_null_into_non_optional_output_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "    else: null\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' else: null needs an Optional output; the case declares str "
        "(write `output: Optional[str]`)"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "else: null")


def test_unquoted_yes_into_str_output_gets_the_quote_hint():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "    else: yes\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'route' else: the value true is not a valid str" in message
    assert 'quote it to keep it as text: `else: "yes"`' in message
    assert "YAML reads unquoted yes/no/on/off/true/false as booleans" in message
    assert _error_line(caught.value) == _line_of(text, "else: yes")


def test_record_literal_with_an_unknown_key_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      reply: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        '          reply: "x"\n'
        "          extra: 1\n"
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert (
        'case node \'route\' then: the value {"reply": "x", "extra": 1} is not a valid '
        "{\"reply\":str} (record has unknown fields: ['extra'])"
    ) in message
    assert _error_line(caught.value) == _line_of(text, 'reply: "x"')


def test_record_literal_missing_a_required_field_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      reply: str\n"
        "      n: int\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        '          reply: "y"\n'
        "          n: 1\n"
        "    else:\n"
        '      reply: "x"\n'
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "case node 'route' else: the value {\"reply\": \"x\"} is not a valid" in message
    assert "(record missing required fields: ['n'])" in message
    assert _error_line(caught.value) == _line_of(text, 'reply: "x"')


def test_disjoint_node_branch_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "    else: num\n",
        extra_nodes=_int_node(),
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' else: `num` produces int, but the case declares output: str"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "else: num")


def test_disjoint_call_branch_is_rejected():
    text = """\
id: f
name: f
input:
  cmd: str
defs:
  child:
    nodes:
      c:
        kind: code
        code: |
          return 3
        output: int
    output: ${c.output}
nodes:
  route:
    kind: case
    on: ${input.cmd}
    output: str
    cases:
      - when: a
        then: call(child)
    else: "other"
output: ${route.output}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' then: the call to `child` produces int, but the case declares "
        "output: str"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "then: call(child)")


def test_narrower_node_branch_loads():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: Literal[a, b]\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "a"\n'
        "    else: reply\n"
    )
    loaded = load_flow(text)
    assert run_flow(loaded, {"cmd": "a"}).output == "a"


def _case_of_case_flow(inner_output: str, inner_then: str) -> str:
    """Return a flow whose `outer` (`output: str`) routes `else:` to the case `inner`."""
    return (
        "id: f\n"
        "name: f\n"
        "input:\n"
        "  cmd: str\n"
        "nodes:\n"
        "  outer:\n"
        "    kind: case\n"
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "picked a"\n'
        "    else: inner\n"
        "  inner:\n"
        "    kind: case\n"
        f"{inner_output}"
        "    cases:\n"
        "      - when: \"${input.cmd} == 'b'\"\n"
        f"        then: {inner_then}\n"
        "    else: num\n"
        f"{_int_node()}"
        "output: ${outer.output}\n"
    )


def test_typed_inner_case_disjoint_from_outer_is_rejected():
    text = _case_of_case_flow("    output: int\n", "4")
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'outer' else: case 'inner' produces int, but the case declares output: str"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "else: inner")


def test_untyped_inner_case_branch_is_checked_against_the_outer_output():
    text = _case_of_case_flow("", "num")
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'inner' then: `num` produces int, but case 'outer' declares output: str"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "then: num")


# --------------------------------------------------------------------------- #
# Values with spans must fit the CASE's `output:`
# --------------------------------------------------------------------------- #


def test_span_value_disjoint_from_output_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: ${input.n}\n"
        "    else: reply\n",
        inputs="  cmd: str\n  n: int\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "case node 'route' then: declared str, but the value is int" in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "then: ${input.n}")


def test_narrower_span_value_loads_and_is_checked_at_runtime():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: Literal[a, b]\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: ${input.s}\n"
        '    else: "b"\n',
        inputs="  cmd: str\n  s: str\n",
    )
    loaded = load_flow(text)
    bad = run_flow(loaded, {"cmd": "a", "s": "c"})
    assert bad.status == "failed"
    assert bad.failure.code == "binding.type"
    assert bad.failure.node_id == "__case_route"
    good = run_flow(loaded, {"cmd": "a", "s": "a"})
    assert good.status == "succeeded", good.failure
    assert good.output == "a"


def test_record_value_field_disjoint_from_output_names_the_field():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output:\n"
        "      reply: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then:\n"
        "          reply: ${input.n}\n"
        "    else:\n"
        '      reply: "no"\n',
        inputs="  cmd: str\n  n: int\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' then: declared str, but the value at .reply is int"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "reply: ${input.n}")


@pytest.mark.parametrize(
    ("output", "then_text", "message"),
    [
        (
            "\n      reply: str\n      n: int",
            "\n          reply: ${input.s}",
            "declared {\"n\":int,\"reply\":str}, but the value is missing required field 'n'",
        ),
        (
            "\n      reply: str",
            "\n          reply: ${input.s}\n          extra: 1",
            "declared {\"reply\":str}, but the value has unknown field 'extra'",
        ),
        ("int", "\n          reply: ${input.s}", "declared int, but the value is a record"),
        (
            "\n      items: list[int]",
            "\n          items:\n            - ${input.s}",
            "declared int, but the value at .items[0] is str",
        ),
        (
            "\n      items: list[int]",
            "\n          items:\n            - ${input.n}\n            - x",
            'the value "x" at .items[1] is not a valid int',
        ),
        ("str", "\n          - ${input.s}", "declared str, but the value is a list"),
    ],
)
def test_span_value_shapes_that_cannot_fit_are_rejected(output, then_text, message):
    text = _flow(
        "    on: ${input.cmd}\n"
        f"    output: {output}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        "    else: null\n",
        inputs="  cmd: str\n  s: str\n  n: int\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert f"case node 'route' then: {message}" in str(caught.value)


@pytest.mark.parametrize(
    ("output", "then_text"),
    [
        ("Optional[object]", "\n          a: ${input.s}"),
        ("Optional[Reply]", "\n          text: ${input.s}"),
        ("Optional[list[str]]", "\n          - ${input.s}"),
    ],
)
def test_span_value_shapes_that_fit_load(output, then_text):
    text = (
        "typedefs:\n"
        "  Text:\n    text: str\n"
        "  Num:\n    num: int\n"
        "  Reply: Union[Text, Num]\n"
    ) + _flow(
        "    on: ${input.cmd}\n"
        f"    output: {output}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then_text}\n"
        "    else: null\n",
        inputs="  cmd: str\n  s: str\n",
    )
    result = run_flow(load_flow(text), {"cmd": "a", "s": "hi"})
    assert result.status == "succeeded", result.failure


def _union_flow(*, then_value: str, else_value: str) -> str:
    """Return a flow whose case on a `Union[Buy, Sell]` tag has the given branch values."""
    return (
        "id: f\n"
        "name: f\n"
        "typedefs:\n"
        "  Buy:\n"
        "    action: Literal[buy]\n"
        "    qty: int\n"
        "  Sell:\n"
        "    action: Literal[sell]\n"
        "    price: float\n"
        "input:\n"
        "  kind: str\n"
        "nodes:\n"
        "  d:\n"
        "    kind: code\n"
        "    input:\n"
        "      kind: ${input.kind}\n"
        "    code: |\n"
        '      if inputs["kind"] == "buy":\n'
        '          return {"action": "buy", "qty": 3}\n'
        '      return {"action": "sell", "price": 1.5}\n'
        "    output: Union[Buy, Sell]\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${d.output.action}\n"
        "    output:\n"
        "      qty: int\n"
        "    cases:\n"
        "      - when: buy\n"
        "        then:\n"
        f"          qty: {then_value}\n"
        "    else:\n"
        f"      qty: {else_value}\n"
        "output: ${route.output}\n"
    )


def test_value_reads_a_union_member_field_its_branch_selects():
    loaded = load_flow(_union_flow(then_value='"${d.output.qty}"', else_value="0"))
    assert run_flow(loaded, {"kind": "buy"}).output == {"qty": 3}
    assert run_flow(loaded, {"kind": "sell"}).output == {"qty": 0}


def test_value_reading_a_field_its_branch_excludes_is_rejected():
    text = _union_flow(then_value="0", else_value='"${d.output.qty}"')
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    # Reference validation reports it, reading the value node against the narrowed union.
    assert (
        "node '__value_1' input 'v0' from: reference ${d.output.qty} reads unknown field "
        "'qty' on record"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, 'qty: "${d.output.qty}"')


def test_value_reading_a_sibling_branch_is_rejected():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        '        then: "got ${reply.output}"\n'
        "    else: reply\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "case node 'route' then: the value reads `reply`, another branch of 'route', "
        "which never runs when this branch does"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "then: \"got ${reply.output}\"")


def test_value_reading_a_sibling_branch_with_a_default_loads():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    output: str\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: ${reply.output:-'none'}\n"
        "    else: reply\n"
    )
    loaded = load_flow(text)
    result = run_flow(loaded, {"cmd": "a"})
    assert result.status == "succeeded", result.failure
    assert result.output == "none"
