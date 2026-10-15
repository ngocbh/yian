"""Load-time field-read checks in prompts, node asserts, flow asserts and builtin results."""

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError

_PREAMBLE = (
    "id: p\nname: p\n"
    "typedefs:\n"
    "  Rec: {a: str}\n"
    "  Rec3: {a: str, b: Optional(str)}\n"
    "  Buy: {action: 'Literal(buy)', qty: int}\n"
    "  Sell: {action: 'Literal(sell)'}\n"
    "  Order: Union(Buy, Sell)\n"
)

_INPUTS = "input:\n  s: !str\n  r: !Rec\n"

_STEP = (
    "  step:\n"
    "    kind: code\n"
    "    input:\n      x: ${input.s}\n"
    "    output: Rec\n"
    "    code: return dict(a='x')\n"
)

_QTY_HINT = (
    "reads field 'qty', which only union members with action buy have; "
    "read it inside a case on ${d.output.action}"
)


def _flow(nodes: str, output: str, *, defs: str = "") -> str:
    """The shared preamble, `defs`, the inputs, node `step`, then `nodes` and `output`."""
    return _PREAMBLE + defs + _INPUTS + "nodes:\n" + _STEP + nodes + f"output: {output}\n"


def _agent(
    inputs: dict[str, str], prompt: str, *, node_id: str = "ag", extra: str = ""
) -> str:
    """An AGENT node with `inputs`, a quoted `prompt`, and extra YAML lines."""
    text = f"  {node_id}:\n    kind: agent\n"
    if inputs:
        text += "    input:\n"
        for name, source in inputs.items():
            text += f"      {name}: {source}\n"
    return text + f'    prompt: "{prompt}"\n' + extra


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return str(caught.value)


def _producer(node_id: str, output: str) -> str:
    """A CODE node typed `output` reading the flow input `s`."""
    return (
        f"  {node_id}:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        f"    output: {output}\n"
        "    code: return None\n"
    )


def _routed(agent: str) -> str:
    """Producer `d` (Order), a case on its action routing buy to `ag`, sell to `other`."""
    return (
        _producer("d", "Order")
        + "  route:\n"
        "    kind: case\n"
        "    on: ${d.output.action}\n"
        "    cases:\n"
        "      - when: buy\n        then: ag\n"
        "      - when: sell\n        then: other\n"
        + agent
        + "  other:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: str\n"
        "    code: return 'x'\n"
    )


def test_agent_prompt_reads_unknown_field():
    message = _load_error(
        _flow(_agent({"v": "${step.output}"}, "hi ${v.b}"), "${ag.output}")
    )
    assert "node 'ag' prompt: reference ${v.b} reads unknown field 'b' on record" in message


def test_agent_system_prompt_reads_unknown_field():
    agent = _agent(
        {"v": "${step.output}"}, "hi", extra='    system_prompt: "sys ${v.b}"\n'
    )
    message = _load_error(_flow(agent, "${ag.output}"))
    assert (
        "node 'ag' system_prompt: reference ${v.b} reads unknown field 'b' on record"
        in message
    )


def test_typed_input_field_read_errors_only_when_every_alternative_rejects():
    message = _load_error(
        _flow(_agent({"w": "!Rec ${step.output}"}, "${w.b}"), "${ag.output}")
    )
    assert "node 'ag' prompt: reference ${w.b} reads unknown field 'b' on record" in message
    load_flow(_flow(_agent({"w": "!Rec3 ${step.output}"}, "${w.b}"), "${ag.output}"))


def test_typed_union_input_reads_the_narrowed_member_inside_a_branch():
    load_flow(
        _flow(
            _routed(_agent({"v": "!Order ${d.output}"}, "${v.qty}")),
            "${ag.output | other.output}",
        )
    )
    outside = _producer("d", "Order") + _agent({"v": "!Order ${d.output}"}, "${v.qty}")
    message = _load_error(_flow(outside, "${ag.output}"))
    assert f"node 'ag' prompt: reference ${{v.qty}} {_QTY_HINT}" in message


def test_optional_binding_field_reads():
    message = _load_error(
        _flow(_agent({"v": "${step.output :- null}"}, "${v.b}"), "${ag.output}")
    )
    assert "node 'ag' prompt: reference ${v.b} reads unknown field 'b' on record" in message
    load_flow(_flow(_agent({"v": "${step.output :- null}"}, "${v.a}"), "${ag.output}"))


def test_adaptive_brief_field_read_is_reported_on_the_gate():
    gate = (
        "  h:\n"
        "    kind: human_input\n"
        "    input:\n      v: ${step.output}\n"
        '    prompt: "Pick for ${v.a}"\n'
        "    adaptive_questions:\n"
        '      prompt: "${v.b}"\n'
    )
    message = _load_error(_flow(gate, "${h.output}"))
    assert (
        "node 'h' adaptive_questions.prompt: reference ${v.b} reads unknown field 'b' "
        "on record" in message
    )


def test_prompt_inside_a_def_reads_unknown_field():
    defs = (
        "defs:\n"
        "  child:\n"
        "    input:\n      x: !Rec\n"
        "    nodes:\n"
        "      inner:\n"
        "        kind: agent\n"
        "        input:\n          x: ${input.x}\n"
        '        prompt: "${x.b}"\n'
        "    output: ${inner.output}\n"
    )
    nodes = "  c:\n    kind: call\n    call: child\n    input:\n      x: ${step.output}\n"
    message = _load_error(_flow(nodes, "${c.output}", defs=defs))
    assert "reference ${x.b} reads unknown field 'b' on record" in message


def test_literal_input_field_read():
    message = _load_error(_flow(_agent({"mode": "fast"}, "${mode.x}"), "${ag.output}"))
    assert (
        "node 'ag' prompt: reference ${mode.x} reads field 'x' on type str, "
        "which has no fields" in message
    )


def test_null_literal_input_field_read():
    message = _load_error(_flow(_agent({"v": "null"}, "${v.x}"), "${ag.output}"))
    assert (
        "node 'ag' prompt: reference ${v.x} reads field 'x' on type None, "
        "which has no fields" in message
    )


def test_field_read_inside_a_builtin_argument():
    message = _load_error(
        _flow(_agent({"v": "${step.output}"}, "${render_as_json(v.b)}"), "${ag.output}")
    )
    assert "node 'ag' prompt: reference ${v.b} reads unknown field 'b' on record" in message


def test_agent_variable_field_read():
    agent = _agent({}, "${cfg.b}", extra="    variables:\n      cfg: !Rec {a: x}\n")
    message = _load_error(_flow(agent, "${ag.output}"))
    assert (
        "node 'ag' prompt: reference ${cfg.b} reads unknown field 'b' on record" in message
    )


def test_human_input_prompt_field_read():
    gate = (
        "  h:\n"
        "    kind: human_input\n"
        "    input:\n      v: ${step.output}\n"
        '    prompt: "${v.b}"\n'
    )
    message = _load_error(_flow(gate, "${h.output}"))
    assert "node 'h' prompt: reference ${v.b} reads unknown field 'b' on record" in message


def test_human_input_question_field_read():
    gate = (
        "  h:\n"
        "    kind: human_input\n"
        "    input:\n      v: ${step.output}\n"
        '    prompt: "Pick for ${v.a}"\n'
        "    questions:\n"
        '      - question: "${v.b}"\n'
        "        header: Pick\n"
        "        options:\n"
        "          - label: one\n"
        "            description: the first\n"
    )
    message = _load_error(_flow(gate, "${h.output}"))
    assert (
        "node 'h' questions[0].question: reference ${v.b} reads unknown field 'b' "
        "on record" in message
    )


def test_prompt_field_error_is_reported_once_per_ref():
    message = _load_error(
        _flow(_agent({"v": "${step.output}"}, "${v.b} and ${v.b}"), "${ag.output}")
    )
    assert message.count("reference ${v.b} reads unknown field 'b'") == 1


@pytest.mark.parametrize(
    "nodes",
    [
        _agent({"v": "${step.output}"}, "${v.a}"),
        _agent({"v": "${step.output}"}, "${v.a :- 'n'}"),
        _producer("o", "object") + _agent({"v": "${o.output}"}, "${v.anything}"),
        _agent({"v": "!Any ${step.output}"}, "${v.anything}"),
        "  u:\n    kind: code\n    input:\n      s: ${input.s}\n    code: return 1\n"
        + _agent({"v": "${u.output}"}, "${v.x}"),
        _producer("m", "Optional(Rec)") + _agent({"v": "${m.output}"}, "${v.a}"),
    ],
    ids=["field", "default", "object", "any", "untyped", "optional-record"],
)
def test_readable_fields_still_load(nodes):
    load_flow(_flow(nodes, "${ag.output}"))


def test_narrowed_input_reads_the_branch_member():
    load_flow(
        _flow(
            _routed(_agent({"v": "${d.output}"}, "${v.qty}")),
            "${ag.output | other.output}",
        )
    )
    outside = _producer("d", "Order") + _agent({"v": "${d.output}"}, "${v.qty}")
    message = _load_error(_flow(outside, "${ag.output}"))
    assert f"node 'ag' prompt: reference ${{v.qty}} {_QTY_HINT}" in message
