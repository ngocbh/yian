"""Reading fields of a union of records: shared fields anywhere, member-only fields rejected."""

import pytest

from yian.compile.model import GrowthRejected
from yian.compose import load_flow
from yian.compose.build.inline_inputs import canonicalize_inline_type
from yian.compose.helpers.output_contracts import project_output_contract
from yian.compose.helpers.errors import LoadError
from yian.compose.helpers.types import read_type
from yian.compose.nodes.human_input import _answer_schema
from yian.compose.nodes.resolver import child_signature
from yian.nodes.code import CodeNode
from yian.nodes.contracts import (
    PartialRecordContract,
    ValueContract,
    output_contract_field_error,
)
from yian.runtime.state import _validate_growth_wiring_refs
from yian.typesys import Type, ValueKind
from yian.typesys.types import read_typedefs

_TYPEDEFS = (
    "typedefs:\n"
    "  Buy: {action: 'Literal(buy)', qty: int, note: str}\n"
    "  Hold: {action: 'Literal(hold)', note: str}\n"
    "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n"
    "  Decision: Union(Buy, Hold, Sell)\n"
    "  Text: {text: str}\n"
    "  Num: {num: int}\n"
    "  Reply: Union(Text, Num)\n"
)

_TAGGED_QTY = (
    "reference ${d.output.qty} reads field 'qty', which only union members with action "
    "buy, sell have; read it inside a case on ${d.output.action}"
)


def _flow(rest: str, *, output: str = "Decision", typedefs: str = _TYPEDEFS) -> str:
    """A flow with one CODE node `d` typed `output`, plus `rest` (YAML tail)."""
    return (
        "id: p\nname: p\n"
        + typedefs
        + "input:\n  s: !str\n"
        "nodes:\n"
        "  d:\n"
        "    kind: code\n"
        "    input:\n"
        "      x: ${input.s}\n"
        f"    output: {output}\n"
        "    code: return None\n"
        + rest
    )


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    return str(exc.value)


def _output_field(text: str, name: str):
    """Return the flow output contract of field `name` (`None` when untyped)."""
    output = child_signature(load_flow(text)).output
    if type(output) is PartialRecordContract:
        return dict(output.fields)[name]
    assert type(output) is ValueContract
    return ValueContract(output.type.fields[name])


def test_tag_field_reads_as_the_merged_literal():
    contract = _output_field(_flow("output:\n  a: ${d.output.action}\n"), "a")
    assert contract.type == Type(
        kind=ValueKind.STRING, tags=frozenset({"buy", "hold", "sell"})
    )


def test_non_exhaustive_on_over_the_tag_names_the_missing_member():
    rest = (
        "  route:\n"
        "    kind: case\n"
        "    on: ${d.output.action}\n"
        "    cases:\n"
        "      - when: buy\n"
        "        then: sink\n"
        "      - when: sell\n"
        "        then: sink\n"
        "  sink:\n"
        "    kind: code\n"
        "    input:\n      x: ${input.s}\n"
        "    output: str\n"
        "    code: return 'x'\n"
        "output: ${sink.output}\n"
    )
    assert "enum tag(s) ['hold'] not covered" in _load_error(_flow(rest))


def test_shared_field_of_one_type_reads_as_that_type():
    contract = _output_field(_flow("output:\n  n: ${d.output.note}\n"), "n")
    assert contract.type == Type.scalar(ValueKind.STRING)


def test_shared_field_optional_in_one_member_reads_as_optional():
    typedefs = _TYPEDEFS.replace(
        "{action: 'Literal(hold)', note: str}",
        "{action: 'Literal(hold)', note: 'Optional(str)'}",
    )
    contract = _output_field(
        _flow("output:\n  n: ${d.output.note}\n", typedefs=typedefs), "n"
    )
    assert contract.type == Type(kind=ValueKind.STRING, nullable=True)


def test_shared_field_through_an_optional_union_reads_as_optional():
    contract = _output_field(
        _flow("output:\n  n: ${d.output.note}\n", output="Optional(Decision)"), "n"
    )
    assert contract.type == Type(kind=ValueKind.STRING, nullable=True)


def test_shared_field_one_member_does_not_require_reads_as_optional():
    text = Type.scalar(ValueKind.STRING)
    tag = lambda value: Type(kind=ValueKind.STRING, tags=frozenset({value}))  # noqa: E731
    union = Type(
        kind=ValueKind.OBJECT,
        variants=(
            Type(
                kind=ValueKind.OBJECT,
                fields={"kind": tag("a"), "note": text},
                required=frozenset({"kind", "note"}),
            ),
            Type(
                kind=ValueKind.OBJECT,
                fields={"kind": tag("b"), "note": text},
                required=frozenset({"kind"}),
            ),
        ),
    )
    projected = project_output_contract(ValueContract(union), ["note"])
    assert projected.type == Type(kind=ValueKind.STRING, nullable=True)


def test_shared_field_of_differing_types_is_untyped_and_loads():
    typedefs = (
        "typedefs:\n"
        "  A: {kind: 'Literal(a)', v: int}\n"
        "  B: {kind: 'Literal(b)', v: str}\n"
        "  AB: Union(A, B)\n"
    )
    text = _flow("output:\n  v: ${d.output.v}\n", output="AB", typedefs=typedefs)
    assert _output_field(text, "v") is None


def test_member_only_field_of_a_tagged_union_is_a_load_error():
    msg = _load_error(_flow("output:\n  q: ${d.output.qty}\n"))
    assert f"flow output from: {_TAGGED_QTY}" in msg


def test_member_only_field_of_an_untagged_union_is_a_load_error():
    msg = _load_error(_flow("output:\n  t: ${d.output.text}\n", output="Reply"))
    assert (
        "reference ${d.output.text} reads field 'text', which only 1 of 2 union members "
        "have; check the member with isinstance(d.output, <Type>) in a case first"
    ) in msg


def test_member_only_field_in_a_flow_assert_is_a_load_error():
    text = _flow("asserts:\n  - d.output.qty > 0\noutput: ${d.output}\n")
    assert f"assert 'd.output.qty > 0': {_TAGGED_QTY}" in _load_error(text)


def test_member_only_field_in_a_searched_case_when_is_a_load_error():
    rest = (
        "  route:\n"
        "    kind: case\n"
        "    cases:\n"
        "      - when: ${d.output.qty > 5}\n"
        "        then: sink\n"
        "    else: sink\n"
        "  sink:\n"
        "    kind: code\n"
        "    input:\n      x: ${input.s}\n"
        "    output: str\n"
        "    code: return 'x'\n"
        "output: ${sink.output}\n"
    )
    assert f"node 'route' cases[0].when from: {_TAGGED_QTY}" in _load_error(_flow(rest))


def test_field_no_member_has_keeps_the_unknown_field_message():
    msg = _load_error(_flow("output:\n  z: ${d.output.zzz}\n"))
    assert "reference ${d.output.zzz} reads unknown field 'zzz' on record" in msg


def test_member_only_field_through_a_record_field_suggests_the_union_path():
    text = _flow("output:\n  q: ${d.output.decision.qty}\n", output="{decision: Decision}")
    assert (
        "reference ${d.output.decision.qty} reads field 'qty', which only union members "
        "with action buy, sell have; read it inside a case on ${d.output.decision.action}"
    ) in _load_error(text)


def test_member_only_field_of_a_typed_flow_input_is_a_load_error():
    text = (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n  d: !Decision\n"
        "nodes:\n"
        "  step:\n"
        "    kind: code\n"
        "    input:\n      v: ${input.d.qty}\n"
        "    output: str\n"
        "    code: return 'x'\n"
        "output: ${step.output}\n"
    )
    assert (
        "node 'step' input 'v' from: reference ${input.d.qty} reads field 'qty', which only "
        "union members with action buy, sell have; read it inside a case on ${input.d.action}"
    ) in _load_error(text)


_EACH_DEFS = (
    "defs:\n"
    "  each:\n"
    "    input:\n      v: !int\n"
    "    nodes:\n"
    "      inner:\n"
    "        kind: code\n"
    "        input:\n          v: ${input.v}\n"
    "        output: str\n"
    "        code: return 'x'\n"
    "    output: ${inner.output}\n"
)


def test_member_only_field_of_a_map_item_is_a_load_error():
    text = (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n  xs: !list(Decision)\n"
        + _EACH_DEFS
        + "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        "    call: each\n"
        "    over: ${input.xs}\n"
        "    input:\n      v: ${item.qty}\n"
        "output: ${m.output}\n"
    )
    assert (
        "node 'm' map input 'v' from: reference ${item.qty} reads field 'qty', which only "
        "union members with action buy, sell have; read it inside a case on ${item.action}"
    ) in _load_error(text)


def test_member_only_field_in_a_node_assert_is_a_load_error():
    text = _flow("output: ${d.output}\n").replace(
        "    code: return None\n",
        "    code: return None\n    asserts:\n      - output.qty > 0\n",
    )
    assert (
        "node 'd' assert: reference ${output.qty} reads field 'qty', which only union "
        "members with action buy, sell have; read it inside a case on ${output.action}"
    ) in _load_error(text)


def test_shared_field_in_a_node_assert_loads():
    text = _flow("output: ${d.output}\n").replace(
        "    code: return None\n",
        "    code: return None\n    asserts:\n      - output.note != ''\n",
    )
    load_flow(text)


def test_member_only_field_in_an_inline_child_body_is_a_load_error():
    rest = (
        "  c:\n"
        "    kind: call\n"
        "    input:\n      v: ${d.output}\n"
        "    nodes:\n"
        "      inner:\n"
        "        kind: code\n"
        "        input:\n          w: ${input.v.qty}\n"
        "        output: str\n"
        "        code: return 'x'\n"
        "    output: ${inner.output}\n"
        "output: ${c.output}\n"
    )
    assert (
        "reference ${input.v.qty} reads field 'qty', which only union members with action "
        "buy, sell have; read it inside a case on ${input.v.action}"
    ) in _load_error(_flow(rest))


def test_shared_field_in_an_inline_child_body_loads():
    rest = (
        "  c:\n"
        "    kind: call\n"
        "    input:\n      v: ${d.output}\n"
        "    nodes:\n"
        "      inner:\n"
        "        kind: code\n"
        "        input:\n          w: ${input.v.note}\n"
        "        output: str\n"
        "        code: return 'x'\n"
        "    output: ${inner.output}\n"
        "output: ${c.output}\n"
    )
    load_flow(_flow(rest))


def _decision() -> Type:
    registry = read_typedefs(
        {
            "Buy": {"action": "Literal(buy)", "qty": "int"},
            "Sell": {"action": "Literal(sell)", "price": "float"},
            "Decision": "Union(Buy, Sell)",
        }
    )
    return read_type("Decision", registry)


def test_shared_walk_allows_member_only_fields_when_asked():
    contract = ValueContract(_decision())
    assert output_contract_field_error(contract, ["qty"], "d.output.qty") == (
        "reference ${d.output.qty} reads field 'qty', which only union members with "
        "action buy have; read it inside a case on ${d.output.action}"
    )
    assert (
        output_contract_field_error(
            contract, ["qty"], "d.output.qty", partial_union_fields=True
        )
        is None
    )
    assert output_contract_field_error(
        contract, ["zzz"], "d.output.zzz", partial_union_fields=True
    ) == ("reference ${d.output.zzz} reads unknown field 'zzz' on record")


def test_member_only_hint_names_the_union_root():
    contract = ValueContract(_decision())
    assert output_contract_field_error(contract, ["qty"], "v.qty", union_root="d.output") == (
        "reference ${v.qty} reads field 'qty', which only union members with "
        "action buy have; read it inside a case on ${d.output.action}"
    )
    nested = PartialRecordContract((("order", contract), ("note", None)))
    assert output_contract_field_error(
        nested, ["order", "qty"], "v.order.qty", union_root="d.output"
    ) == (
        "reference ${v.order.qty} reads field 'qty', which only union members with "
        "action buy have; read it inside a case on ${d.output.order.action}"
    )
    untagged = ValueContract(
        read_type("Reply", read_typedefs({"Text": {"text": "str"}, "Num": {"num": "int"},
                                          "Reply": "Union(Text, Num)"}))
    )
    assert output_contract_field_error(
        untagged, ["text"], "v.text", union_root="s.output"
    ) == (
        "reference ${v.text} reads field 'text', which only 1 of 2 union members have; "
        "check the member with isinstance(s.output, <Type>) in a case first"
    )


def test_runtime_growth_allows_member_only_fields():
    node = CodeNode("g", code="return None")
    producers = {"d": ValueContract(_decision())}
    _validate_growth_wiring_refs(
        "g", node, {"v": "${d.output.qty}"}, {"d", "g"}, producers.get, set()
    )
    with pytest.raises(GrowthRejected, match="reads unknown field 'zzz' on record"):
        _validate_growth_wiring_refs(
            "g", node, {"v": "${d.output.zzz}"}, {"d", "g"}, producers.get, set()
        )


def test_canonicalize_inline_type_keeps_variants():
    decision = _decision()
    canonical = canonicalize_inline_type(decision)
    assert canonical == decision
    assert canonical.variants is not None


def test_map_over_a_union_output_gives_a_list_of_objects():
    text = (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n  xs: !list(str)\n"
        "defs:\n"
        "  each:\n"
        "    input:\n      v: !str\n"
        "    nodes:\n"
        "      inner:\n"
        "        kind: code\n"
        "        input:\n          v: ${input.v}\n"
        "        output: Decision\n"
        "        code: return None\n"
        "    output: ${inner.output}\n"
        "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        "    call: each\n"
        "    over: ${input.xs}\n"
        "    input:\n      v: ${item}\n"
        "output: ${m.output}\n"
    )
    contract = load_flow(text).compiled.nodes["m"].output_contract
    assert contract.type.kind is ValueKind.LIST_OBJECT
    assert contract.type.element.variants is not None


def test_human_input_answer_schema_for_a_union_is_one_object_entry():
    assert _answer_schema(_decision()) == [
        {"name": "answer", "type": "object", "required": True}
    ]


def test_loop_carrying_a_union_keeps_the_non_record_error():
    text = (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n  s: !str\n"
        "nodes:\n"
        "  drive:\n"
        "    kind: loop\n"
        "    times: 2\n"
        "    input:\n      action: buy\n"
        "    nodes:\n"
        "      step:\n"
        "        kind: code\n"
        "        input:\n          action: ${input.action}\n"
        "        output: Decision\n"
        "        code: return None\n"
        "    output: ${step.output}\n"
        "output: ${drive.output}\n"
    )
    assert "must return a record with at least one carried field" in _load_error(text)
