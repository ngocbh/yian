"""Tests for reading slot types and values through YAML tags."""

import re

import pytest

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.tags import (
    TaggedSlot,
    read_tag_type,
    read_tagged_value,
    suggest_tag,
)
from yian.compose.source import SourceDocument
from yian.compose.source.catalog import _SourceCatalogBuilder
from yian.nodes.contracts import Binding, ValueTemplate
from yian.typesys.types import read_typedefs, type_for

REG = read_typedefs(
    {
        "Buy": {"action": "Literal(buy)", "conviction": "float"},
        "Sell": {"action": "Literal(sell)", "conviction": "float"},
        "Decision": "Union(Buy, Sell)",
        "Order": {"decision": "Buy", "qty": "int"},
    }
)


def rt(tag):
    return read_tag_type(tag, REG, label="x", diagnostics=LoadDiagnostics())


def test_simple_and_one_argument_tags():
    assert rt("int").kind.value == "int"
    assert rt("Optional(str)").nullable
    assert rt("int(1..12)").bounds == (1, 12)
    assert rt("Decision").variants
    assert rt("list(Buy)").element.fields


@pytest.mark.parametrize(
    "tag, fragment",
    [
        ("Union(Buy,Sell)", "name it in `typedefs:`"),
        ("int,b:", "a tag can't contain `,`"),
        ("Optional[str]", "`!Optional(str)`"),
        ("list[Buy]", "(strict YAML 1.2); write `!list(Buy)`"),
        ("Union[Buy, Sell]", "name `Union(Buy, Sell)` in `typedefs:`"),
        ("Union(Buy,", "ends at the first space"),
        ("Literal(a", "ends at the first space"),
        ("Strr", "Did you mean `str`?"),
        ("Bye", "Did you mean `Buy`?"),
    ],
)
def test_tag_errors(tag, fragment):
    with pytest.raises(LoadError, match=re.escape(fragment)):
        rt(tag)


def test_incomplete_tag_message_names_the_tag_and_the_fix():
    with pytest.raises(LoadError) as info:
        rt("Union(Buy,")
    message = str(info.value)
    assert "`!Union(Buy,` is not a complete type" in message
    assert "(e.g. `Decision: Union(Buy, Sell)`)" in message
    assert "(`!Decision`)" in message


def test_unknown_inner_name_gets_no_misleading_head_hint():
    with pytest.raises(LoadError) as info:
        rt("list(Strr)")
    assert "Did you mean" not in str(info.value)


def test_suggest_tag():
    assert suggest_tag("qty", type_for("int", REG), "${x}") == "`qty: !int ${x}`"
    assert suggest_tag("qty", type_for("Optional(int(1..12))", REG), None) == (
        "`qty: !Optional(int(1..12))`"
    )
    assert suggest_tag("a", type_for("Literal(buy)", REG), "buy") == "`a: !Literal(buy) buy`"
    assert "typedefs:" in suggest_tag("d", type_for("Union(Buy, Sell)", REG), "${x}")
    assert suggest_tag("d", type_for("Literal(a, b)", REG), None) == (
        'name `Literal(a, b)` in `typedefs:` and tag with its name'
    )
    assert "typedefs:" in suggest_tag("d", type_for("Buy", REG), None)
    assert "typedefs:" in suggest_tag("d", type_for('Literal("a b")', REG), None)



@pytest.mark.parametrize(
    "tag, spelled",
    [("dict[str, int]", "`!object`"), ("list[dict[str, int]]", "`!list(object)`")],
)
def test_bracket_dict_suggests_object(tag, spelled):
    with pytest.raises(LoadError, match=re.escape(f"write {spelled}")):
        rt(tag)


def test_bracket_hint_is_offered_only_for_a_type_that_resolves():
    with pytest.raises(LoadError) as info:
        rt("list[Nope]")
    assert "write" not in str(info.value)
    assert "typedefs:" not in str(info.value)


def test_suggest_tag_prefers_a_typedef_name():
    assert suggest_tag("d", type_for("Buy", REG), None, registry=REG) == "`d: !Buy`"
    assert suggest_tag("d", type_for("Optional(Buy)", REG), "${x}", registry=REG) == (
        "`d: !Optional(Buy) ${x}`"
    )
    assert suggest_tag("d", type_for("list(Decision)", REG), None, registry=REG) == (
        "`d: !list(Decision)`"
    )
    assert suggest_tag("d", type_for("int", REG), None, registry=REG) == "`d: !int`"
    assert suggest_tag("d", type_for("Buy", REG), None, type_text="Buy") == "`d: !Buy`"
    assert "typedefs:" in suggest_tag("d", type_for("Buy", REG), None, type_text="{a: int}")


def test_suggest_tag_spells_a_type_directly_before_naming_a_scalar_alias():
    registry = read_typedefs({"Price": "float", "Qty": "int(1..12)", "Buy": {"action": "Literal(buy)"}})
    assert suggest_tag("p", type_for("float", registry), None, registry=registry) == (
        "`p: !float`"
    )
    assert suggest_tag("p", type_for("list(float)", registry), None, registry=registry) == (
        "`p: !list(float)`"
    )
    assert suggest_tag("q", type_for("int(1..12)", registry), None, registry=registry) == (
        "`q: !int(1..12)`"
    )
    assert suggest_tag("b", type_for("list(Buy)", registry), None, registry=registry) == (
        "`b: !list(Buy)`"
    )


# ---------- read_tagged_value ----------


def read(text, key="x", *, ledger=None, document=None, **options):
    document = document or SourceDocument.parse(text)
    return read_tagged_value(
        document.root.child(key),
        REG,
        label=key,
        site_path=("node", "n", "input", key),
        catalog_builder=_SourceCatalogBuilder(),
        diagnostics=LoadDiagnostics.from_document(document),
        ledger=set() if ledger is None else ledger,
        **options,
    )


def fails(text, fragment, **options):
    with pytest.raises(LoadError, match=re.escape(fragment)):
        read(text, **options)


def test_top_level_tag_gives_the_contract():
    slot = read("x: !int ${a.output}\n")
    assert slot == TaggedSlot(value="${a.output}", contract=slot.contract, has_value=True)
    assert slot.contract.type == type_for("int", REG)
    assert slot.contract.origins.default is not None


def test_untagged_value_has_no_contract_and_is_unchanged():
    assert read("x: ${a.output}\n") == TaggedSlot("${a.output}", None, True)
    assert read("x: {a: 1, b: [y, 2.5], c: null}\n") == TaggedSlot(
        {"a": 1, "b": ["y", 2.5], "c": None}, None, True
    )
    assert read("x: '$$5'\n").value == "$$5"


def test_every_tag_read_is_recorded_in_the_ledger():
    document = SourceDocument.parse(
        "x: !Order {decision: !Buy {action: buy, conviction: !float 1}, qty: !int 1}\n"
        "y: !int 2\n"
    )
    ledger = set()
    read("", document=document, ledger=ledger)
    unread = document.root.child("y").tag_definition_id
    assert ledger == {n.definition_id for n in document.tagged_nodes()} - {unread}
    assert len(ledger) == 4


def test_empty_tagged_value_has_no_value():
    slot = read("x: !Optional(int)\n")
    assert slot.value is None and not slot.has_value
    assert slot.contract.type.nullable


@pytest.mark.parametrize(
    "text", ["x:\n  - !int\n", "x:\n  d: !Buy\n", "x: !list(int)\n  - 1\n  - !int\n"]
)
def test_nested_empty_tag_is_an_error(text):
    fails(text, "a nested tag needs a value")


@pytest.mark.parametrize(
    "text, spelled",
    [
        ("x: !int (1..12) 5\n", "`!int(1..12)`"),
        ("x: !Optional (str)\n", "`!Optional(str)`"),
        ("x: [!int (1..12) 5]\n", "`!int(1..12)`"),
    ],
)
def test_space_after_a_tag_is_an_error(text, spelled):
    fails(text, f"a YAML tag ends at the first space; write {spelled} without spaces")


def test_quoted_text_starting_with_a_parenthesis_is_text():
    assert read("x: !str '(a) b'\n").value == "(a) b"


def test_top_level_strings_are_kept_as_written():
    assert read('x: !str "Status: ${a.output}"\n').value == "Status: ${a.output}"
    assert read("x: !int call(f, a=1)\n").value == "call(f, a=1)"
    assert read("x: !str '$$5'\n").value == "$$5"


def test_nested_tagged_template_becomes_a_nested_binding():
    slot = read('x: {d: !Buy "${d.output}", n: 1}\n')
    template = slot.value
    assert isinstance(template, ValueTemplate) and slot.contract is None
    nested = template.value["d"]
    assert isinstance(nested, Binding) and nested.source == "${d.output}"
    assert nested.contract.type == type_for("Buy", REG)
    assert nested.origin is not None and nested.contract.origins.default == nested.origin
    assert template.value["n"] == 1
    assert template.origin_at(("d",)) not in (None, nested.origin)


def test_nested_tagged_literal_is_checked_and_kept_as_a_binding():
    template = read("x: {q: !int 5, s: !str '$$x'}\n").value
    assert isinstance(template, ValueTemplate)
    assert template.value["q"].source == 5
    assert template.value["q"].contract.type == type_for("int", REG)
    # The binder unescapes `$$` in a template, so the raw text is kept.
    assert template.value["s"].source == "$$x"
    fails("x: {q: !int '5'}\n", '\'x.q\': `"5"` is not a valid int')


def test_list_of_tagged_items():
    template = read('x: !list(int) [!int "${a.output}", !int 2]\n').value
    first, second = template.value
    assert (first.source, second.source) == ("${a.output}", 2)
    assert {first.contract.type, second.contract.type} == {type_for("int", REG)}


@pytest.mark.parametrize("text", ["x: {a: call(f)}\n", "x: {a: 'call(f, b=1)'}\n", "x: [' call(f)']\n"])
def test_nested_call_is_an_error(text):
    fails(
        text,
        "a `call(...)` inside a value isn't supported; bind it at the top level, "
        "or reword the text",
    )


def test_nested_spans_make_a_template_with_raw_leaf_strings():
    template = read('x: {a: "${n.output}", b: [1, "${m.output}"], c: "$$x"}\n').value
    assert isinstance(template, ValueTemplate)
    assert template.value == {"a": "${n.output}", "b": [1, "${m.output}"], "c": "$$x"}
    assert template.origin_at(("a",)) is not None
    assert template.origin_at(("b", 1)) is not None
    assert template.origin_at(("c",)) is not None


def test_span_free_container_is_a_literal_unescaped_once():
    slot = read('x: {c: "$$x", d: ["$$$${y}"]}\n')
    assert slot.value == {"c": "$x", "d": ["$${y}"]}


def test_top_level_tagged_container_with_spans_is_checked_later():
    slot = read('x: !Buy {action: buy, conviction: "${c.output}"}\n')
    assert isinstance(slot.value, ValueTemplate)


def test_wrong_literal_is_an_error():
    fails("x: !int '5'\n", '\'x\': `"5"` is not a valid int')
    fails("x: !int(1..12) 13\n", "`13` is not a valid int(1..12)")
    fails(
        "x: !date 2026/01/01\n",
        "'x': `2026/01/01` is not a valid date (date must look like 2026-05-21)",
    )
    fails("x: !Buy {action: sell, conviction: 1}\n", "not a valid Buy")
    fails(
        'x: !Order {decision: !Buy {action: buy, conviction: 1}, qty: "1"}\n',
        "(at `x.qty`)",
    )


def test_str_literal_read_as_a_number_gets_the_quote_hint():
    fails(
        "x: !str 30\n",
        '\'x\': `30` is not a valid str; quote it to keep it as text: `x: !str "30"`',
    )
    with pytest.raises(LoadError) as info:
        read("x: !str true\n")
    assert '`x: !str "true"`' in str(info.value)
    assert "(YAML reads unquoted true/false as booleans)" in str(info.value)
    assert read("x: !str yes\n").value == "yes"
    fails("x: [!str 2026-01-01]\n", 'quote it to keep it as text: `!str "2026-01-01"`')


@pytest.mark.parametrize(
    "text, at",
    [
        ("x: !Buy {conviction: 1}\n", "x.action"),
        ("x: !list(Buy) [{conviction: 1}]\n", "x[0].action"),
        ("x: !Order {decision: {conviction: 1}, qty: 1}\n", "x.decision.action"),
        ("x: {d: !Buy {conviction: 1}}\n", "x.d.action"),
    ],
)
def test_a_tag_never_fills_in_a_missing_field(text, at):
    fails(text, f"record missing required fields: ['action'] (at `{at}`)")


def test_valid_union_member_loads():
    slot = read("x: !Decision {action: sell, conviction: 0.5}\n")
    assert slot.value == {"action": "sell", "conviction": 0.5}


def test_literal_only_folds_nested_tags_into_a_plain_value():
    slot = read(
        "x: !Order {decision: !Buy {action: buy, conviction: 1}, qty: 1}\n",
        literal_only=True,
    )
    assert slot.value == {"decision": {"action": "buy", "conviction": 1}, "qty": 1}
    assert read("x: !str ${y}\n", literal_only=True).value == "${y}"
    assert read('x: {s: "$$y", t: call(f)}\n', literal_only=True).value == {
        "s": "$$y",
        "t": "call(f)",
    }
    assert not read("x: !int\n", literal_only=True).has_value
    with pytest.raises(LoadError, match=re.escape("record missing required fields")):
        read("x: {d: !Buy {conviction: 1}}\n", literal_only=True)


def test_text_starting_with_a_parenthesis_after_a_tag_must_be_quoted():
    fails(
        "x: !str (555) 123-4567\n",
        '\'x\': a YAML tag ends at the first space; quote the text: '
        '`x: !str "(555) 123-4567"`',
    )
    fails("x: [!str (555) 1]\n", 'quote the text: `!str "(555) 1"`')
    fails("x: !Strr (a)\n", "Did you mean `str`?")


def test_mismatch_at_the_root_does_not_repeat_the_path():
    with pytest.raises(LoadError) as info:
        read("x: !int [1]\n")
    assert "'x': the value is not a valid int:" in str(info.value)
    assert "(at" not in str(info.value)


def test_block_scalar_text_is_quoted_in_messages():
    with pytest.raises(LoadError) as info:
        read("x: !int |\n  5\n")
    assert '`"5\\n"` is not a valid int' in str(info.value)


def test_recursive_value_error_names_the_slot():
    fails("x: &a {b: *a}\n", "'x': cannot descend through a recursive alias")


def test_text_starting_with_a_call_skips_the_load_check():
    # The inline-call desugar knows the YAML style: unquoted, the text must be one
    # well-formed directive; quoted, it is text.
    assert read("x: !str call(f) and more\n").value == "call(f) and more"
    assert read("x: !int call(f) and more\n").value == "call(f) and more"
    assert read("x: !int call(f, a=1)\n").value == "call(f, a=1)"
    fails("x: !int calls(f)\n", "is not a valid int")


def test_any_other_value_error_from_the_check_is_a_located_load_error(monkeypatch):
    import yian.compose.helpers.tags as tags

    def reject(typ, value):
        raise ValueError("rejected by the value model")

    monkeypatch.setattr(tags, "build_value_as", reject)
    with pytest.raises(LoadError) as info:
        read("x: !object {a: 1}\n")
    assert "'x': the value is not a valid object: rejected by the value model" in str(
        info.value
    )
    assert info.value.diagnostic.frames[-1].range.start_line == 1
