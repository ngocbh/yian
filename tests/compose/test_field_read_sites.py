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


_ASSERT_QTY_HINT = (
    "reads field 'qty', which only union members with action buy have; "
    "check the member first with `isinstance(v, <Type>) and ...`"
)


def _flow(
    nodes: str,
    output: str,
    *,
    typedefs: str = "",
    defs: str = "",
    inputs: str = "",
    tail: str = "",
) -> str:
    """Return a flow built from the shared preamble and the given sections.

    The flow holds the shared typedefs plus `typedefs` lines, `defs`, the inputs (plus
    `inputs` lines), node `step`, then `nodes`, `output` and `tail`."""
    return (
        _PREAMBLE
        + typedefs
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


def test_defaulted_union_binding_hint_names_the_bound_source():
    nodes = _producer("d", "Order") + _agent({"v": '"${d.output :- null}"'}, "${v.qty}")
    message = _load_error(_flow(nodes, "${ag.output}"))
    assert f"node 'ag' prompt: reference ${{v.qty}} {_QTY_HINT}" in message


def test_transformed_union_binding_hint_names_no_bound_source():
    """A constructor wraps the value, so the hint can't name a path below its source."""
    nodes = _producer("d", "Order") + _agent({"v": "${Wrap(o=d.output)}"}, "${v.o.qty}")
    message = _load_error(_flow(nodes, "${ag.output}", typedefs="  Wrap: {o: Order}\n"))
    assert "read it inside a case on ${v.o.action}" in message
    assert "d.output.o" not in message


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
    assert f"node 'c' assert: reference ${{v.qty}} {_ASSERT_QTY_HINT}" in message


_UNION_TYPES = (
    "  Short: {action: 'Literal(short)', qty: int}\n"
    "  Hold: {action: 'Literal(hold)'}\n"
    "  Trade: Union(Buy, Short, Hold)\n"
    "  Sized: Union(Buy, Short)\n"
    "  A: {x: int}\n"
    "  B: {y: str}\n"
    "  AB: Union(A, B)\n"
)


def _isinstance_routed(produced: str, member: str, node: str) -> str:
    """Producer `d` typed `produced`, a case routing `isinstance(d.output, member)` to
    `ag` and anything else to `other`, then `node` (id `ag`) and `other`."""
    return (
        _producer("d", produced)
        + "  route:\n"
        "    kind: case\n"
        "    cases:\n"
        f"      - when: isinstance(d.output, {member})\n        then: ag\n"
        "    else: other\n"
        + node
        + "  other:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: str\n"
        "    code: return 'x'\n"
    )


@pytest.mark.parametrize(
    ("produced", "member", "node"),
    [
        ("AB", "A", _agent({"v": "${d.output}"}, "${v.x}")),
        ("AB", "A", _code({"v": "${d.output}"}, ["v.x > 0"], node_id="ag")),
        ("Order", "Buy", _agent({"v": "${d.output}"}, "${v.qty}")),
        ("Trade", "Sized", _agent({"v": "${d.output}"}, "${v.qty}")),
        ("Trade", "Sized", _code({"v": "${d.output}"}, ["v.qty > 0"], node_id="ag")),
    ],
    ids=[
        "untagged-prompt",
        "untagged-assert",
        "tagged-prompt",
        "shared-prompt",
        "shared-assert",
    ],
)
def test_isinstance_branch_reads_the_narrowed_member(produced, member, node):
    flow = _flow(
        _isinstance_routed(produced, member, node),
        "${ag.output | other.output}",
        typedefs=_UNION_TYPES,
    )
    load_flow(flow)


_ORDER_INPUT = "  o: !Order\n"


def _local_assert(assertion: str, *, kind: str = "code") -> str:
    """A flow whose node `c` asserts `assertion` on its local `v`, bound from `${d.output}`
    with `d` typed `Order`; `c` is an AGENT node when `kind` is `agent`."""
    if kind == "agent":
        node = _agent(
            {"v": "${d.output}"},
            "hi",
            node_id="c",
            extra=f'    asserts:\n      - "{assertion}"\n',
        )
    else:
        node = _code({"v": "${d.output}"}, [assertion])
    return _flow(_producer("d", "Order") + node, "${c.output}")


def _line_of(text: str, needle: str) -> int:
    """The 1-based line of `text` holding `needle`."""
    return next(i for i, line in enumerate(text.splitlines(), 1) if needle in line)


@pytest.mark.parametrize(
    "assertion",
    [
        "isinstance(v, Buy) and v.qty > 0",
        "not isinstance(v, Buy) or v.qty > 0",
        "isinstance(v, Sell) or v.qty > 0",
        "v.action == 'buy' and v.qty > 0",
        "v.action != 'buy' or v.qty > 0",
        "'sell' == v.action or v.qty > 0",
        "not (isinstance(v, Buy) and v.qty < 0)",
        "(isinstance(v, Buy) and v.qty > 0) or v.action == 'sell'",
        "isinstance(v, Buy) and (v.qty > 0 or v.qty < -10)",
    ],
)
def test_node_assert_reads_are_narrowed_by_the_checks_before_them(assertion):
    load_flow(_local_assert(assertion))


@pytest.mark.parametrize(
    ("assertion", "expected"),
    [
        ("v.qty > 0 and isinstance(v, Buy)", f"reference ${{v.qty}} {_ASSERT_QTY_HINT}"),
        ("v.qty > 0", f"reference ${{v.qty}} {_ASSERT_QTY_HINT}"),
        (
            "isinstance(v, Buy) or v.qty > 0",
            "reference ${v.qty} reads field 'qty', which Sell doesn't have; "
            "v is Sell here, by the checks before this read",
        ),
    ],
    ids=["check-after-read", "unguarded", "or-sees-the-other-member"],
)
def test_node_assert_reads_the_checks_do_not_cover_are_rejected(assertion, expected):
    message = _load_error(_local_assert(assertion))
    assert f"node 'c' assert: {expected}" in message


def test_node_assert_read_under_two_guards_is_checked_under_each():
    message = _load_error(_local_assert("v.qty > 0 or (isinstance(v, Buy) and v.qty > 1)"))
    assert message.count(f"reference ${{v.qty}} {_ASSERT_QTY_HINT}") == 1


@pytest.mark.parametrize(
    ("order", "status"),
    [
        ({"action": "sell"}, "succeeded"),
        ({"action": "buy", "qty": 2}, "succeeded"),
        ({"action": "buy", "qty": 0}, "failed"),
    ],
    ids=["sell", "buy-positive", "buy-zero"],
)
def test_guarded_node_assert_runs(order, status):
    node = _code({"v": "${input.o}"}, ["not isinstance(v, Buy) or v.qty > 0"])
    flow = load_flow(_flow(node, "${c.output}", inputs=_ORDER_INPUT))
    result = run_flow(flow, {"s": "x", "r": {"a": "x"}, "o": order})
    assert result.status == status, result.failure
    if status == "failed":
        assert result.failure.code == "assertion.failed"


def test_agent_node_assert_reads_are_narrowed():
    load_flow(_local_assert("isinstance(v, Sell) or v.qty > 0", kind="agent"))


def _output_assert(assertion: str) -> str:
    """A flow whose CODE node `c` returns an `Order` and asserts `assertion`."""
    node = (
        "  c:\n    kind: code\n    input:\n      s: ${input.s}\n"
        "    output: Order\n    code: return dict(action='sell')\n"
        f'    asserts:\n      - "{assertion}"\n'
    )
    return _flow(node, "${c.output}")


def test_node_assert_output_reads_are_narrowed():
    load_flow(_output_assert("isinstance(output, Buy) and output.qty > 0"))
    message = _load_error(_output_assert("output.qty > 0"))
    assert (
        "node 'c' assert: reference ${output.qty} reads field 'qty', which only union "
        "members with action buy have; check the member first with "
        "`isinstance(output, <Type>) and ...`" in message
    )


def _flow_assert(assertion: str, nodes: str = "", output: str = "${step.output}") -> str:
    """A flow with inputs `o: !Order` and `ab: !AB` asserting `assertion`."""
    return _flow(
        nodes,
        output,
        typedefs=_UNION_TYPES,
        inputs=_ORDER_INPUT + "  ab: !AB\n",
        tail=_asserts(assertion),
    )


@pytest.mark.parametrize(
    "assertion",
    [
        "isinstance(input.o, Buy) and input.o.qty > 0",
        "not isinstance(input.o, Buy) or input.o.qty > 0",
        "input.o.action == 'sell' or input.o.qty > 0",
        "not isinstance(input.ab, A) or input.ab.x > 0",
        "isinstance(o, Buy) and input.o.qty > 0",
    ],
    ids=["and", "not-or", "tag-or", "untagged", "shorthand-check"],
)
def test_flow_assert_reads_are_narrowed_by_the_checks_before_them(assertion):
    load_flow(_flow_assert(assertion))


def test_flow_post_assert_producer_reads_are_narrowed():
    load_flow(
        _flow_assert(
            "isinstance(d.output, Buy) and d.output.qty > 0",
            _producer("d", "Order"),
            "${d.output}",
        )
    )


def test_flow_assert_read_before_its_check_is_rejected():
    message = _load_error(_flow_assert("input.o.qty > 0 and isinstance(input.o, Buy)"))
    assert (
        "reference ${input.o.qty} reads field 'qty', which only union members with "
        "action buy have; check the member first with "
        "`isinstance(input.o, <Type>) and ...`" in message
    )


@pytest.mark.parametrize(
    ("order", "status"),
    [({"action": "sell"}, "succeeded"), ({"action": "buy", "qty": 0}, "failed")],
    ids=["sell", "buy-zero"],
)
def test_guarded_flow_assert_runs(order, status):
    flow = load_flow(_flow_assert("not isinstance(input.o, Buy) or input.o.qty > 0"))
    result = run_flow(flow, {"s": "x", "r": {"a": "x"}, "o": order, "ab": {"x": 1}})
    assert result.status == status, result.failure


def test_partial_union_node_assert_reads_the_narrowed_member():
    """On a cycle a member-only field is readable, below a narrowed member too; the
    cycle is what the load reports."""
    typedefs = (
        "  Limit: {type: 'Literal(limit)', price: int}\n"
        "  Market: {type: 'Literal(market)'}\n"
        "  LBuy: {action: 'Literal(buy)', leg: 'Union(Limit, Market)'}\n"
        "  LOrder: Union(LBuy, Sell)\n"
    )
    nodes = (
        _producer("d", "LOrder")
        + "  b:\n    kind: code\n    input:\n      y: ${c.output}\n"
        "    output: str\n    code: return 'x'\n"
        + _code(
            {"v": "${d.output}", "y": "${b.output}"},
            ["isinstance(v, LBuy) and v.leg.price > 0"],
        )
    )
    message = _load_error(_flow(nodes, "${c.output}", typedefs=typedefs))
    assert "flow has a cycle involving ['b', 'c']" in message
    assert "reads field" not in message


def test_impossible_node_assert_check_is_located_at_the_assert():
    """The read after the impossible check is reached by no value, so it is not checked."""
    text = _local_assert("isinstance(v, Buy) and v.action == 'sell' and v.zz > 0")
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "node 'c' assert: `v.action == \"sell\"` can never be true here: v is Buy, "
        "by the checks before it" in str(caught.value)
    )
    assert "${v.zz}" not in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == _line_of(
        text, "v.action == 'sell'"
    )


def test_impossible_flow_assert_check_is_located_at_the_assert():
    text = _flow_assert("input.o.action == 'hold'")
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "assert \"input.o.action == 'hold'\": `input.o.action == \"hold\"` compares "
        "input.o.action with 'hold', which is not one of its tags (buy, sell)"
        in str(caught.value)
    )
    assert caught.value.diagnostic.frames[-1].range.start_line == _line_of(
        text, "input.o.action == 'hold'"
    )


_ORDER_CHILD = (
    "defs:\n"
    "  child:\n"
    "    input:\n      v: !Order\n"
    "    nodes:\n"
    "      inner:\n"
    "        kind: code\n"
    "        input:\n          v: ${input.v}\n"
    "        output: str\n"
    "        code: return 'x'\n"
    "    output: ${inner.output}\n"
)


def test_call_node_assert_tag_checks_narrow_its_local():
    nodes = _producer("d", "Order") + (
        "  c:\n    kind: call\n    call: child\n"
        "    input:\n      v: ${d.output}\n"
        "    asserts:\n      - \"v.action == 'buy' and v.qty > 0\"\n"
    )
    load_flow(_flow(nodes, "${c.output}", defs=_ORDER_CHILD))


def test_assert_reading_many_typed_inputs_loads():
    """Equal alternatives of a typed binding count once, so the checks stay linear."""
    names = [f"a{i}" for i in range(14)]
    assertion = " and ".join(f"{name}.a == 'x'" for name in names)
    node = _code({name: "!Rec ${step.output}" for name in names}, [assertion])
    load_flow(_flow(node, "${c.output}"))


def _locals_assert(assertion: str, producers: dict[str, str], inputs: dict[str, str]) -> str:
    """A flow with CODE producers typed by `producers` and node `c` asserting `assertion`
    on its `inputs`."""
    nodes = "".join(_producer(node_id, typ) for node_id, typ in producers.items())
    return _flow(nodes + _code(inputs, [assertion]), "${c.output}")


@pytest.mark.parametrize(
    ("assertion", "expected"),
    [
        (
            "isinstance(v, Rec) and v.zz > 0",
            "`isinstance(v, Rec)` can never be true: v is declared Order, and no Order value "
            "fits Rec",
        ),
        (
            "v.action == 'byu' and v.zz > 0",
            "`v.action == \"byu\"` compares v.action with 'byu', which is not one of its tags "
            "(buy, sell); did you mean 'buy'?",
        ),
    ],
    ids=["no-overlap", "not-a-tag"],
)
def test_node_assert_read_after_a_reported_check_is_not_checked(assertion, expected):
    message = _load_error(_local_assert(assertion))
    assert f"node 'c' assert: {expected}" in message
    assert "${v.zz}" not in message


def test_node_assert_read_after_an_always_true_negation_is_not_checked():
    text = _locals_assert("not isinstance(v, Buy) and v.zz > 0", {"d": "Buy"}, {"v": "${d.output}"})
    message = _load_error(text)
    assert (
        "node 'c' assert: `not isinstance(v, Buy)` can never be true: v is declared Buy, so "
        "`isinstance(v, Buy)` is always true" in message
    )
    assert "${v.zz}" not in message


def test_node_assert_read_of_one_local_after_a_reported_check_on_another_is_not_checked():
    text = _locals_assert(
        "isinstance(v, Buy) and v.action == 'sell' and w.zz > 0",
        {"d": "Order"},
        {"v": "${d.output}", "w": "${step.output}"},
    )
    message = _load_error(text)
    assert "`v.action == \"sell\"` can never be true here" in message
    assert "${w.zz}" not in message


# `v` is declared Order but bound from a Buy, so it has two alternatives: Order and Buy.
_TWO_ALTERNATIVES = {"v": "!Order ${d.output}", "w": "!Order ${e.output}"}


def test_node_assert_check_impossible_under_every_alternative_is_reported_once():
    text = _locals_assert(
        "isinstance(v, Rec) or v.zz > 0", {"d": "Buy", "e": "Order"}, _TWO_ALTERNATIVES
    )
    message = _load_error(text)
    assert message.count("`isinstance(v, Rec)` can never be true") == 1
    assert message.count("reference ${v.zz} reads unknown field 'zz'") == 1


def test_node_assert_read_dead_under_only_some_alternatives_is_checked():
    """Under Order the tag check rules `v` out, but under Buy it is no check at all, so
    neither the check is reported nor the read after it skipped."""
    text = _locals_assert(
        "isinstance(v, Buy) and v.action == 'sell' and w.zz > 0",
        {"d": "Buy", "e": "Order"},
        _TWO_ALTERNATIVES,
    )
    message = _load_error(text)
    assert "node 'c' assert: reference ${w.zz} reads unknown field 'zz' on record" in message
    assert "can never be true" not in message


@pytest.mark.parametrize(
    ("assertion", "expected"),
    [
        (
            "isinstance(output, Rec)",
            "`isinstance(output, Rec)` can never be true: output is declared Order, and no "
            "Order value fits Rec",
        ),
        (
            "not isinstance(output, Order)",
            "`not isinstance(output, Order)` can never be true: output is declared Order, so "
            "`isinstance(output, Order)` is always true",
        ),
        (
            "isinstance(output, Buy) and isinstance(output, Sell)",
            "`isinstance(output, Sell)` can never be true here: output is Buy, by the checks "
            "before it",
        ),
    ],
    ids=["no-overlap", "always", "context"],
)
def test_node_assert_impossible_output_check_is_reported(assertion, expected):
    message = _load_error(_output_assert(assertion))
    assert f"node 'c' assert: {expected}" in message


def test_node_assert_output_read_after_a_reported_check_is_not_checked():
    message = _load_error(_output_assert("isinstance(output, Rec) and output.action == 'buy'"))
    assert "`isinstance(output, Rec)` can never be true" in message
    assert "${output.action}" not in message


def test_node_assert_over_many_locals_with_two_alternatives_loads():
    """Each local is checked against its own alternatives, not every combination."""
    names = [f"v{i}" for i in range(12)]
    assertion = " and ".join(f"(isinstance({x}, Buy) or {x}.action == 'sell')" for x in names)
    text = _locals_assert(assertion, {"d": "Buy"}, {x: "!Order ${d.output}" for x in names})
    load_flow(text)


def _skippable_assert(assertion: str) -> str:
    """A flow whose `case` runs `d` (typed `Order`) only for `s: go`, asserting `assertion`."""
    nodes = (
        "  route:\n    kind: case\n    cases:\n"
        "      - when: \"input.s == 'go'\"\n        then: d\n"
        "    else: e\n"
        + _producer("d", "Order")
        + "  e:\n    kind: code\n    input:\n      s: ${input.s}\n    code: return 1\n"
    )
    return _flow(nodes, "${step.output}", tail=_asserts(assertion))


@pytest.mark.parametrize(
    "assertion",
    [
        "not isinstance(d.output, Order) or d.output.action == 'sell'",
        "not isinstance(d.output, Buy) and not isinstance(d.output, Sell)",
    ],
    ids=["always-on-the-declared-type", "every-member-ruled-out"],
)
def test_flow_assert_on_a_skippable_producer_reads_null(assertion):
    """END binds a producer it reads with `:- null`, so a skipped one reads null."""
    flow = load_flow(_skippable_assert(assertion))
    result = run_flow(flow, {"s": "stop", "r": {"a": "x"}})
    assert result.status == "succeeded", result.failure


@pytest.mark.parametrize(
    ("assertion", "expected"),
    [
        (
            "isinstance(d.output, Rec)",
            "`isinstance(d.output, Rec)` can never be true: d.output is declared Order, and "
            "no Order value fits Rec",
        ),
        (
            "isinstance(d.output, Buy) and isinstance(d.output, Sell)",
            "`isinstance(d.output, Sell)` can never be true here: d.output is Buy, by the "
            "checks before it",
        ),
    ],
    ids=["no-overlap", "context"],
)
def test_flow_assert_check_impossible_even_on_null_is_reported(assertion, expected):
    message = _load_error(_skippable_assert(assertion))
    assert expected in message


def test_flow_assert_shorthand_check_is_reported_as_written():
    message = _load_error(_flow_assert("isinstance(o, Rec)"))
    assert (
        "`isinstance(o, Rec)` can never be true: o is declared Order, and no Order value "
        "fits Rec" in message
    )
    assert "input.o" not in message


def _child_of(typ: str) -> str:
    """A `defs:` child flow taking `v: !<typ>`."""
    return _ORDER_CHILD.replace("v: !Order", f"v: !{typ}")


def test_call_node_assert_member_only_hint_compares_the_tag():
    """`isinstance` is not allowed in a CALL node's asserts, so the hint compares the tag."""
    nodes = _producer("d", "Order") + (
        "  c:\n    kind: call\n    call: child\n"
        "    input:\n      v: ${d.output}\n"
        "    asserts:\n      - \"v.qty > 0\"\n"
    )
    message = _load_error(_flow(nodes, "${c.output}", defs=_ORDER_CHILD))
    assert (
        "node 'c' assert: reference ${v.qty} reads field 'qty', which only union members "
        "with action buy have; check the member first with `v.action == \"buy\" and ...`"
    ) in message
    assert "isinstance" not in message


def test_call_node_assert_member_only_hint_on_an_untagged_union_names_a_case():
    nodes = _producer("d", "AB") + (
        "  c:\n    kind: call\n    call: child\n"
        "    input:\n      v: ${d.output}\n"
        "    asserts:\n      - \"v.x > 0\"\n"
    )
    message = _load_error(_flow(nodes, "${c.output}", typedefs=_UNION_TYPES, defs=_child_of("AB")))
    assert (
        "node 'c' assert: reference ${v.x} reads field 'x', which only 1 of 2 union members "
        "have; check the member with isinstance(d.output, <Type>) in a case first"
    ) in message


def _order_loop(assertion: str) -> str:
    """A one-turn inline LOOP carrying `o: Order`, seeded from the flow input `o`."""
    nodes = (
        "  drive:\n    kind: loop\n    times: 1\n"
        "    input:\n      o: ${input.o}\n"
        f'    asserts:\n      - "{assertion}"\n'
        "    nodes:\n"
        "      bump:\n"
        "        kind: code\n"
        "        input:\n          o: ${input.o}\n"
        "        output:\n          o: Order\n"
        "        code: return dict(o=inputs['o'])\n"
        "    output: ${bump.output}\n"
    )
    return _flow(nodes, "${drive.output}", inputs=_ORDER_INPUT)


def test_loop_node_assert_tag_checks_narrow_its_local():
    flow = load_flow(_order_loop("o.action != 'buy' or o.qty > 0"))
    order = {"action": "buy", "qty": 2}
    result = run_flow(flow, {"s": "x", "r": {"a": "x"}, "o": order})
    assert result.status == "succeeded", result.failure
    message = _load_error(_order_loop("o.qty > 0"))
    assert (
        "node 'drive' assert: reference ${o.qty} reads field 'qty', which only union members "
        "with action buy have; check the member first with `o.action == \"buy\" and ...`"
    ) in message
    message = _load_error(_order_loop("o.action == 'sell' and o.action == 'buy' and o.zz > 0"))
    assert "`o.action == \"buy\"` can never be true here: o is Sell" in message
    assert "${o.zz}" not in message


_PARTIAL_CHILD = (
    "defs:\n"
    "  child:\n"
    "    input:\n      x: !str\n"
    "    nodes:\n"
    "      typed:\n"
    "        kind: code\n"
    "        input:\n          x: ${input.x}\n"
    "        output: str\n"
    "        code: return 'a'\n"
    "      untyped:\n"
    "        kind: code\n"
    "        input:\n          x: ${input.x}\n"
    "        code: return dict(deep=dict(k=1))\n"
    "    output:\n"
    "      a: ${typed.output}\n"
    "      b: ${untyped.output}\n"
)


def _partial_read(assertion: str) -> str:
    """A CALL to `child` (a partial record: typed `a`, opaque `b`), and a CODE node
    asserting `assertion` on it through the local name `v`."""
    nodes = (
        "  call_child:\n    kind: call\n    call: child\n"
        "    input:\n      x: ${input.s}\n"
        + _code({"v": "${call_child.output}"}, [assertion])
    )
    return _flow(nodes, "${c.output}", defs=_PARTIAL_CHILD)


def test_partial_record_local_root_reads():
    message = _load_error(_partial_read("v.zz == 1"))
    assert "node 'c' assert: reference ${v.zz} reads unknown field 'zz' on record" in message
    flow = load_flow(_partial_read("v.a == 'a' and v.b.deep.k == 1"))
    assert run_flow(flow, {"s": "x", "r": {"a": "x"}}).status == "succeeded"


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
