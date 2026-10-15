"""Load-time field-read checks in prompts, node asserts, flow asserts and builtin results."""

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.expr.builtins import TEMPLATE_FNS
from yian.run import run_flow

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


def _flow(
    nodes: str, output: str, *, defs: str = "", inputs: str = "", tail: str = ""
) -> str:
    """Return a flow built from the shared preamble and the given sections.

    The flow holds `defs`, the inputs (plus `inputs` lines), node `step`, then `nodes`,
    `output` and `tail`."""
    return (
        _PREAMBLE
        + defs
        + _INPUTS
        + inputs
        + "nodes:\n"
        + _STEP
        + nodes
        + f"output: {output}\n"
        + tail
    )


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
    """Return the message of the `LoadError` loading `text` raises."""
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


@pytest.mark.parametrize(
    "source",
    ["!Order {action: buy, qty: '${n.output}'}", "!Order {action: buy, qty: 1}"],
    ids=["template", "literal"],
)
def test_typed_record_binding_reads_its_tagged_member(source):
    nodes = _producer("n", "int") + _agent({"v": source}, "${v.qty}")
    load_flow(_flow(nodes, "${ag.output}"))


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


def _code(
    inputs: dict[str, str], asserts: list[str], *, node_id: str = "c"
) -> str:
    """A CODE node returning `'x'` (typed `str`) with `inputs` and `asserts`."""
    text = f"  {node_id}:\n    kind: code\n    input:\n"
    for name, source in inputs.items():
        text += f"      {name}: {source}\n"
    text += "    output: str\n    code: return 'x'\n    asserts:\n"
    return text + "".join(f'      - "{a}"\n' for a in asserts)


def _asserts(*asserts: str) -> str:
    """A flow-level `asserts:` section."""
    return "asserts:\n" + "".join(f'  - "{a}"\n' for a in asserts)


@pytest.mark.parametrize(
    "assertion", ["v.b == 'x'", "output == 'x' and v.b == 'x'"], ids=["pre", "post"]
)
def test_node_assert_reads_unknown_input_field(assertion):
    message = _load_error(_flow(_code({"v": "${step.output}"}, [assertion]), "${c.output}"))
    assert "node 'c' assert: reference ${v.b} reads unknown field 'b' on record" in message


def test_node_assert_field_error_is_reported_once_per_ref():
    message = _load_error(
        _flow(_code({"v": "${step.output}"}, ["v.b == null or v.b == 'y'"]), "${c.output}")
    )
    assert message.count("reference ${v.b} reads unknown field 'b'") == 1


@pytest.mark.parametrize("child_type", ["Rec", "Rec3"])
def test_call_node_assert_reads_a_field_its_binding_lacks(child_type):
    defs = (
        "defs:\n"
        "  child:\n"
        f"    input:\n      v: !{child_type}\n"
        "    nodes:\n"
        "      inner:\n"
        "        kind: code\n"
        "        input:\n          v: ${input.v}\n"
        "        output: str\n"
        "        code: return 'x'\n"
        "    output: ${inner.output}\n"
    )
    nodes = (
        "  c:\n    kind: call\n    call: child\n"
        "    input:\n      v: ${step.output}\n"
        "    asserts:\n      - \"v.b == 'x'\"\n"
    )
    message = _load_error(_flow(nodes, "${c.output}", defs=defs))
    assert "node 'c' assert: reference ${v.b} reads unknown field 'b' on record" in message


_LOOP_NAMED_BODY = (
    "defs:\n"
    "  body:\n"
    "    input:\n      st: !Rec3\n"
    "    nodes:\n"
    "      bump:\n"
    "        kind: code\n"
    "        input:\n          st: ${input.st}\n"
    "        output:\n          st: Rec3\n"
    "        code: return dict(st=dict(a='x', b='y'))\n"
    "    output: ${bump.output}\n"
)


def _named_loop(assertion: str) -> str:
    """A LOOP over the named `body` (carries `Rec3`), seeded with `step`'s `Rec`."""
    nodes = (
        "  drive:\n    kind: loop\n    call: body\n    times: 3\n"
        "    input:\n      st: ${step.output}\n"
        f'    asserts:\n      - "{assertion}"\n'
    )
    return _flow(nodes, "${drive.output}", defs=_LOOP_NAMED_BODY)


def _inline_loop(assertion: str) -> str:
    """A one-turn inline LOOP carrying `Rec3`, seeded with `step`'s `Rec`."""
    nodes = (
        "  drive:\n    kind: loop\n    times: 1\n"
        "    input:\n      st: ${step.output}\n"
        f'    asserts:\n      - "{assertion}"\n'
        "    nodes:\n"
        "      bump:\n"
        "        kind: code\n"
        "        input:\n          st: ${input.st}\n"
        "        output:\n          st: Rec3\n"
        "        code: return dict(st=dict(a='x', b='y'))\n"
        "    output: ${bump.output}\n"
    )
    return _flow(nodes, "${drive.output}")


@pytest.mark.parametrize("build", [_named_loop, _inline_loop], ids=["named", "inline"])
def test_loop_assert_reads_the_carried_field(build):
    flow = load_flow(build("st.b == null or st.b == 'y'"))
    result = run_flow(flow, {"s": "x", "r": {"a": "x"}})
    assert result.status == "succeeded", result.failure


@pytest.mark.parametrize("build", [_named_loop, _inline_loop], ids=["named", "inline"])
def test_loop_assert_reads_a_field_neither_seed_nor_carried_has(build):
    message = _load_error(build("st.zz == 'x'"))
    assert "node 'drive' assert: reference ${st.zz} reads unknown field 'zz'" in message


def test_node_assert_literal_input_field_read():
    message = _load_error(_flow(_code({"n": "3"}, ["n.x == 1"]), "${c.output}"))
    assert (
        "node 'c' assert: reference ${n.x} reads field 'x' on type int, "
        "which has no fields" in message
    )


@pytest.mark.parametrize(
    "nodes",
    [
        _code({"v": "!Any ${step.output}"}, ["v.k == 1"]),
        _producer("o", "object") + _code({"v": "${o.output}"}, ["v.k == 1"]),
    ],
    ids=["any", "object"],
)
def test_node_assert_open_input_field_reads_load(nodes):
    load_flow(_flow(nodes, "${c.output}"))


def test_node_assert_known_input_field_loads_and_runs():
    flow = load_flow(_flow(_code({"v": "${step.output}"}, ["v.a == 'x'"]), "${c.output}"))
    result = run_flow(flow, {"s": "x", "r": {"a": "x"}})
    assert result.status == "succeeded", result.failure


def test_node_assert_reads_the_narrowed_member_inside_a_branch():
    load_flow(
        _flow(
            _routed(_code({"v": "${d.output}"}, ["v.qty > 0"], node_id="ag")),
            "${ag.output | other.output}",
        )
    )
    outside = _producer("d", "Order") + _code({"v": "${d.output}"}, ["v.qty > 0"])
    message = _load_error(_flow(outside, "${c.output}"))
    assert f"node 'c' assert: reference ${{v.qty}} {_QTY_HINT}" in message


@pytest.mark.parametrize(
    ("assertion", "expected"),
    [
        ("input.r.b == 'x'", "reference ${input.r.b} reads unknown field 'b' on record"),
        ("input.s.x == 'y'", "reference ${input.s.x} reads field 'x' on type str"),
    ],
    ids=["record", "scalar"],
)
def test_flow_assert_reads_unknown_input_field(assertion, expected):
    message = _load_error(_flow("", "${step.output}", tail=_asserts(assertion)))
    assert f"assert {assertion!r}: {expected}" in message


def test_flow_assert_known_input_field_loads_and_runs():
    flow = load_flow(_flow("", "${step.output}", tail=_asserts("input.r.a == 'x'")))
    assert run_flow(flow, {"s": "x", "r": {"a": "x"}}).status == "succeeded"
    assert run_flow(flow, {"s": "x", "r": {"a": "y"}}).status == "failed"


@pytest.mark.parametrize(
    ("inputs", "assertion"),
    [
        ("  o: !object\n", "input.o.k == 1"),
        ("", "step.output.a == 'x' and input.r.a == 'x'"),
    ],
    ids=["object", "post"],
)
def test_flow_assert_readable_input_fields_load(inputs, assertion):
    load_flow(_flow("", "${step.output}", inputs=inputs, tail=_asserts(assertion)))


_TEXT_FIELD = "reference ${{{name}(...).f}} reads field 'f' on type str, which has no fields"


def _case_targets() -> str:
    """CODE nodes `a` and `b`, the targets of the CASE helpers below."""
    return (
        "  a: {kind: code, code: \"return 'a'\", output: str}\n"
        "  b: {kind: code, code: \"return 'b'\", output: str}\n"
    )


def test_binding_reads_a_field_on_a_builtin_result():
    nodes = (
        "  c:\n    kind: code\n"
        "    input:\n      v: ${upper(input.s).f}\n"
        "    output: str\n    code: return 'x'\n"
    )
    message = _load_error(_flow(nodes, "${c.output}"))
    assert f"node 'c' input 'v' from: {_TEXT_FIELD.format(name='upper')}" in message


def test_flow_output_reads_a_field_on_a_builtin_result():
    message = _load_error(_flow("", "\n  o: ${lower(input.s).f}"))
    assert f"flow output from: {_TEXT_FIELD.format(name='lower')}" in message


def test_prompt_reads_a_field_on_a_builtin_result():
    message = _load_error(
        _flow(_agent({"v": "${step.output}"}, "${upper(v.a).x}"), "${ag.output}")
    )
    assert (
        "node 'ag' prompt: reference ${upper(...).x} reads field 'x' on type str, "
        "which has no fields" in message
    )


def test_case_when_reads_a_field_on_a_builtin_result():
    nodes = (
        "  g:\n    kind: case\n    cases:\n"
        "      - when: \"upper(input.s).f == 'x'\"\n        then: a\n"
        "    else: b\n" + _case_targets()
    )
    message = _load_error(_flow(nodes, "${g.output}"))
    assert f"case node 'g' cases[0].when: {_TEXT_FIELD.format(name='upper')}" in message


def test_node_assert_reads_a_field_on_a_builtin_result():
    message = _load_error(
        _flow(_code({"v": "${step.output}"}, ["upper(v.a).f == 'x'"]), "${c.output}")
    )
    assert (
        f"node 'c' assert \"upper(v.a).f == 'x'\": {_TEXT_FIELD.format(name='upper')}"
        in message
    )


def test_flow_assert_reads_a_field_on_a_builtin_result():
    message = _load_error(
        _flow("", "${step.output}", tail=_asserts("upper(input.s).f == 'x'"))
    )
    assert (
        f"flow assert \"upper(input.s).f == 'x'\": {_TEXT_FIELD.format(name='upper')}"
        in message
    )


def test_loop_while_reads_a_field_on_a_builtin_result():
    nodes = (
        "  drive:\n    kind: loop\n    call: body\n"
        "    while: ${upper(st.a).f == 'x'}\n    max: 3\n"
        "    input:\n      st: ${step.output}\n"
    )
    message = _load_error(_flow(nodes, "${drive.output}", defs=_LOOP_NAMED_BODY))
    assert f"loop node 'drive' while: {_TEXT_FIELD.format(name='upper')}" in message


_MAP_CHILD = (
    "defs:\n"
    "  child:\n"
    "    input:\n      x: !str\n"
    "    nodes:\n"
    "      inner:\n"
    "        kind: code\n"
    "        input:\n          x: ${input.x}\n"
    "        output: str\n"
    "        code: return 'x'\n"
    "    output: ${inner.output}\n"
)


@pytest.mark.parametrize(
    ("nodes", "output", "site"),
    [
        (
            "  m:\n    kind: map\n    call: child\n    over: ${upper(input.s).f}\n"
            "    input:\n      x: ${item}\n",
            "${m.output}",
            "node 'm' map over",
        ),
        (
            "  g:\n    kind: case\n    on: ${upper(input.s).f}\n"
            "    cases:\n      - when: x\n        then: a\n    else: b\n"
            + _case_targets(),
            "${g.output}",
            "node 'g' on",
        ),
        (
            "  c:\n    kind: code\n"
            "    input:\n      v: call(child, x=${upper(input.s).f})\n"
            "    output: str\n    code: return 'x'\n",
            "${c.output}",
            "node 'c' input 'v' call(child) argument 'x' from",
        ),
    ],
    ids=["map-over", "case-on", "inline-call-arg"],
)
def test_scanned_sites_read_a_field_on_a_builtin_result_once(nodes, output, site):
    message = _load_error(_flow(nodes, output, defs=_MAP_CHILD))
    assert site in message
    assert message.count(_TEXT_FIELD.format(name="upper")) == 1


@pytest.mark.parametrize(
    ("nodes", "output"),
    [
        (_agent({"v": "${upper(input.s)}"}, "${v}"), "${ag.output}"),
        (_agent({"v": "!list(str) [a, b]"}, "${join(v, ',')}"), "${ag.output}"),
        (_code({"v": "${input.s}"}, ["upper(v) == 'X'"]), "${c.output}"),
    ],
    ids=["upper", "join", "assert"],
)
def test_builtin_results_without_a_field_read_load(nodes, output):
    load_flow(_flow(nodes, output))


def test_host_builtin_field_reads_load(monkeypatch):
    monkeypatch.setitem(TEMPLATE_FNS, "mkrec", lambda s: {"f": s})
    monkeypatch.setitem(TEMPLATE_FNS, "upper", lambda s: {"f": s})
    for name in ("mkrec", "upper"):
        nodes = (
            "  c:\n    kind: code\n"
            f"    input:\n      v: ${{{name}(input.s).f}}\n"
            "    output: str\n    code: return 'x'\n"
            f"    asserts:\n      - \"{name}(v).f == v\"\n"
        )
        load_flow(_flow(nodes, "${c.output}"))


def test_unparsable_condition_keeps_its_located_parse_error():
    message = _load_error(
        _flow(_code({"v": "${step.output}"}, ["upper(v.a).f =="]), "${c.output}")
    )
    assert "node 'c' assert 'upper(v.a).f ==': not a valid boolean expression" in message
