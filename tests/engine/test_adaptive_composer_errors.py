"""A load error in an adaptive gate's composer names the gate and points at the authored text."""

from __future__ import annotations

import pytest

from yian.compose import LoadError, load_flow

# The gate's `llm_config:` value starts on line 10.
_FLOW = """\
id: f
name: f
input: {{proj: !str }}
nodes:
  ask:
    kind: human_input
    input: {{proj: '${{input.proj}}'}}
    adaptive_questions:
      prompt: "Q ${{proj}}"
{llm_config}output: ${{ask.output}}
"""

_LLM_SITE = "node 'ask' adaptive_questions.llm_config"


def _flow(*, llm_config: str = "") -> str:
    return _FLOW.format(llm_config=llm_config)


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


@pytest.mark.parametrize(
    ("config", "fragments"),
    [
        (
            "{inherit: 'false'}",
            ["adaptive_questions.llm_config `inherit:` must be true or false, got 'false'"],
        ),
        ("{model: 5}", ["adaptive_questions.llm_config: 1 validation error for LLMConfig"]),
        ("{temparature: 1}", ["temparature", "extra_forbidden"]),
        ("5", ["adaptive_questions.llm_config must be a mapping"]),
        ("{api_key: x}", ["adaptive_questions.llm_config.api_key is a host credential"]),
    ],
    ids=["inherit", "model-type", "unknown-key", "not-mapping", "api-key"],
)
def test_bad_llm_config_is_named_and_located_at_the_llm_config(config, fragments):
    error = _load_error(_flow(llm_config=f"      llm_config: {config}\n"))
    message = str(error)
    assert message.startswith(_LLM_SITE)
    for fragment in fragments:
        assert fragment in message
    assert "ask__compose" not in message
    assert _line(error) == 10


def test_block_llm_config_error_is_located_at_its_first_key():
    error = _load_error(
        _flow(llm_config="      llm_config:\n        model: m\n        bogus: 1\n")
    )
    message = str(error)
    assert message.startswith(_LLM_SITE)
    assert "bogus" in message
    assert "ask__compose" not in message
    assert _line(error) == 11


def test_compact_flow_llm_config_error_names_the_flow_node():
    error = _load_error(
        """\
id: f
name: f
input: {proj: !str }
kind: human_input
adaptive_questions:
  prompt: "Q ${proj}"
  llm_config: {temparature: 1}
"""
    )
    message = str(error)
    assert message.startswith("node 'f' adaptive_questions.llm_config")
    assert "f__compose" not in message
    assert _line(error) == 7


def test_valid_llm_config_loads():
    load_flow(_flow(llm_config="      llm_config: {model: m}\n"))


# The gate's first input is on line 12.
_INPUT_FLOW = """\
id: f
name: f
input: {{proj: !str }}
nodes:
  up:
    kind: code
    output: int
    code: "return 1"
  ask:
    kind: human_input
    input:
{inputs}{gate_prompt}    adaptive_questions:
      prompt: "{brief}"
output: ${{ask.output}}
"""


def _input_flow(inputs: str, *, brief: str = "Q ${proj}", gate_prompt: str = "") -> str:
    return _INPUT_FLOW.format(inputs=inputs, brief=brief, gate_prompt=gate_prompt)


def _node_lines(error: LoadError) -> list[str]:
    return [
        ln.strip() for ln in str(error).splitlines() if ln.strip().startswith("node ")
    ]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("${input.nope}", "points to unknown flow input 'nope'"),
        ("${nope.output}", "uses unknown namespace 'nope'"),
        ("${up.output.x}", "reads field 'x' on type int"),
        ("${input.nope :- 'x'}", "points to unknown flow input 'nope'"),
    ],
    ids=["flow-input", "namespace", "field", "default"],
)
def test_bad_input_is_reported_once_on_the_gate(source, fragment):
    error = _load_error(_input_flow(f'      proj: "{source}"\n'))
    message = str(error)
    lines = [ln.strip() for ln in message.splitlines() if "input 'proj'" in ln]
    assert len(lines) == 1
    assert lines[0].startswith("node 'ask' input 'proj' from: ")
    assert fragment in lines[0]
    assert "ask__compose" not in message
    assert _line(error) == 12


def test_two_inputs_with_the_same_bad_source_are_both_reported():
    error = _load_error(
        _input_flow(
            "      x: ${input.nope}\n      y: ${input.nope}\n", brief="Q ${x} ${y}"
        )
    )
    assert [ln.split(": ", 1)[0] for ln in _node_lines(error)] == [
        "node 'ask' input 'x' from",
        "node 'ask' input 'y' from",
    ]
    assert "ask__compose" not in str(error)


def test_bad_brief_and_bad_input_are_each_reported_once():
    error = _load_error(_input_flow("      proj: ${input.nope}\n", brief="Q ${nope}"))
    assert [ln.split(": ", 1)[0] for ln in _node_lines(error)] == [
        "node 'ask' adaptive_questions.prompt",
        "node 'ask' input 'proj' from",
    ]
    assert "ask__compose" not in str(error)


def test_bad_input_is_listed_before_a_bad_gate_prompt():
    error = _load_error(
        _input_flow("      proj: ${input.nope}\n", gate_prompt='    prompt: "${nope}"\n')
    )
    assert [ln.split(": ", 1)[0] for ln in _node_lines(error)] == [
        "node 'ask' input 'proj' from",
        "node 'ask' prompt",
    ]
    assert "ask__compose" not in str(error)


def test_isinstance_unknown_type_names_the_gate():
    error = _load_error(_input_flow("      proj: ${isinstance(input.proj, Nope)}\n"))
    message = str(error)
    assert message.startswith("node 'ask' input 'proj': unknown type 'Nope'")
    assert "ask__compose" not in message
    assert _line(error) == 12


def test_literal_inputs_still_load():
    inputs = "      cfg: {a: 1}\n      ls: [1, 2]\n      n: !list(int) [1]\n"
    load_flow(_input_flow(inputs, brief="Q ${cfg} ${ls} ${n}"))
    load_flow(
        f"""\
id: f
name: f
nodes:
  a:
    kind: agent
    input:
{inputs}    prompt: "Q ${{cfg}} ${{ls}} ${{n}}"
    output: str
output: ${{a.output}}
"""
    )


def test_typed_input_mismatch_names_the_gate():
    error = _load_error(_input_flow("      proj: !list(int) ${input.proj}\n"))
    message = str(error)
    assert "node 'ask' input 'proj': declared list(int), but the value is str" in message
    assert "ask__compose" not in message
    assert _line(error) == 12


def test_skipped_operand_arithmetic_names_the_gate():
    error = _load_error(
        """\
id: f
name: f
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: ${input.approve}
        then: order
  order:
    kind: code
    output: {qty: int}
    code: |
      return {"qty": 1}
  ask:
    kind: human_input
    input:
      q: ${order.output.qty * 2}
    adaptive_questions:
      prompt: "Q ${q}"
output: ${ask.output}
"""
    )
    message = str(error)
    assert (
        "node 'ask' input 'q': ${order.output.qty * 2} does arithmetic on order.output.qty"
        in message
    )
    assert "ask__compose" not in message
    assert _line(error) == 19


def test_defs_body_input_error_names_the_gate():
    error = _load_error(
        """\
id: f
name: f
input:
  topic: !str
defs:
  pick:
    input:
      topic: !str
    nodes:
      ask:
        kind: human_input
        input:
          proj: ${input.nope}
        adaptive_questions:
          prompt: "Q ${proj}"
    output: ${ask.output}
nodes:
  run:
    kind: call
    call: pick
    input:
      topic: ${input.topic}
output: ${run.output}
"""
    )
    message = str(error)
    assert message.startswith("defs entry 'pick': ")
    lines = [ln for ln in message.splitlines() if "input 'proj'" in ln]
    assert len(lines) == 1
    assert "node 'ask' input 'proj' from: " in lines[0]
    assert "ask__compose" not in message
    assert _line(error) == 13


# --------------------------------------------------------------------------- #
# cycles: the gate is named, not its composer, unless an author reads the composer.
# --------------------------------------------------------------------------- #

_CYCLE_MESSAGE = "flow has a cycle involving {}; flows must be acyclic"
_GENERATED = "ask__compose is generated from node 'ask'"

_SELF_READING = """\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '${ask.output}'}
    adaptive_questions: {prompt: "Q ${proj}"}
"""


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line of the first line of `text` that starts with `needle`."""
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith(needle):
            return number
    raise AssertionError(f"{needle!r} not in flow")


def _frame_lines(error: LoadError) -> list[int]:
    return [frame.range.start_line for frame in error.diagnostic.frames]


def test_self_reading_gate_cycle_names_only_the_gate():
    text = _SELF_READING + "output: ${ask.output}\n"
    error = _load_error(text)
    message = str(error)
    assert message.startswith(_CYCLE_MESSAGE.format(["ask"]))
    assert "ask__compose" not in message
    assert _frame_lines(error) == [_line_of(text, "  ask:")]
    assert error.diagnostic.notes == ("ask depends on ask (ask.input.proj)",)


_THROUGH_B = """\
id: f
name: f
input: {{proj: !str }}
nodes:
  b:
    kind: agent
    input: {{a: '${{{b_reads}}}'}}
    prompt: "x ${{a}}"
  ask:
    kind: human_input
{gate_extra}    input: {{proj: '${{{gate_reads}}}'}}
    adaptive_questions: {{prompt: "Q ${{proj}}"}}
output: ${{ask.output}}
"""


def test_cycle_through_a_gate_names_only_authored_nodes():
    text = _THROUGH_B.format(b_reads="ask.output", gate_extra="", gate_reads="b.output")
    error = _load_error(text)
    message = str(error)
    assert message.startswith(_CYCLE_MESSAGE.format(["ask", "b"]))
    assert _frame_lines(error) == [_line_of(text, "  b:"), _line_of(text, "  ask:")]
    assert error.diagnostic.notes == (
        "ask depends on b (ask.input.proj)",
        "b depends on ask (b.input.a)",
    )
    assert "ask__compose" not in message
    assert all("ask__compose" not in note for note in error.diagnostic.notes)


def test_ordering_cycle_through_a_gate_names_only_authored_nodes():
    text = _THROUGH_B.format(
        b_reads="ask.output", gate_extra="    depends_on: [b]\n", gate_reads="input.proj"
    )
    error = _load_error(text)
    assert str(error).startswith(_CYCLE_MESSAGE.format(["ask", "b"]))
    assert error.diagnostic.notes == (
        "ask depends on b (ordering)",
        "b depends on ask (b.input.a)",
    )


def test_cycle_through_a_direct_composer_reader_keeps_the_id():
    text = _THROUGH_B.format(
        b_reads="ask__compose.output", gate_extra="", gate_reads="b.output"
    )
    error = _load_error(text)
    assert str(error).startswith(_CYCLE_MESSAGE.format(["ask__compose", "b"]))
    assert "b depends on ask__compose (b.input.a)" in error.diagnostic.notes
    assert _GENERATED in error.diagnostic.notes


def test_composer_read_outside_the_cycle_still_names_only_the_gate():
    text = (
        _SELF_READING
        + """\
  log:
    kind: code
    input:
      q: ${ask__compose.output}
    output: int
    code: return len(inputs["q"])
output: {a: '${ask.output}', q: '${ask__compose.output}'}
"""
    )
    error = _load_error(text)
    assert str(error).startswith(_CYCLE_MESSAGE.format(["ask"]))
    assert _frame_lines(error) == [_line_of(text, "  ask:")]
    assert error.diagnostic.notes == ("ask depends on ask (ask.input.proj)",)


def test_two_self_reading_gates_each_name_only_the_gate():
    text = (
        _SELF_READING
        + """\
  ask2:
    kind: human_input
    input: {proj: '${ask2.output}'}
    adaptive_questions: {prompt: "Q ${proj}"}
output: {a: '${ask.output}', b: '${ask2.output}'}
"""
    )
    error = _load_error(text)
    message = str(error)
    assert message.startswith(_CYCLE_MESSAGE.format(["ask", "ask2"]))
    assert "__compose" not in message
    assert _frame_lines(error) == [_line_of(text, "  ask:"), _line_of(text, "  ask2:")]
    assert error.diagnostic.notes == (
        "ask depends on ask (ask.input.proj)",
        "ask2 depends on ask2 (ask2.input.proj)",
    )


def test_gate_reading_its_own_composer_keeps_the_id():
    # The composer copies the gate's `x`, so it reads itself: the cycle is its self-edge.
    error = _load_error(
        """\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '${input.proj}', x: '${ask__compose.output}'}
    adaptive_questions: {prompt: "Q ${proj} ${x}"}
output: ${ask.output}
"""
    )
    assert str(error).startswith(_CYCLE_MESSAGE.format(["ask__compose"]))
    assert error.diagnostic.notes == (
        "ask__compose depends on ask__compose (ask__compose.input.x)",
        _GENERATED,
    )


def test_defs_child_cycle_names_only_the_gate():
    text = """\
id: f
name: f
input:
  topic: !str
defs:
  pick:
    input:
      topic: !str
    nodes:
      ask:
        kind: human_input
        input:
          proj: ${ask.output}
        adaptive_questions:
          prompt: "Q ${proj}"
    output: ${ask.output}
nodes:
  run:
    kind: call
    call: pick
    input:
      topic: ${input.topic}
output: ${run.output}
"""
    error = _load_error(text)
    message = str(error)
    assert message.startswith(
        "defs entry 'pick': " + _CYCLE_MESSAGE.format(["ask"])
    )
    assert "ask__compose" not in message
    # The defs entry's frame is its body, which starts on the line after `pick:`.
    assert _frame_lines(error) == [
        _line_of(text, "  pick:") + 1,
        _line_of(text, "      ask:"),
    ]
