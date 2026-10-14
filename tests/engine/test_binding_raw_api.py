"""The raw-string binding API on the unified engine — `eval_binding(source, resolve, item)`
and `expr_refs_of(source)`.

The binding functions take a raw template string and scan it internally through the
unified grammar. These tests cover:
- binding semantics (typed whole-span, embedded stringify, `$$`,
  coalesce `|`, `:-` default, `:?` required, a default ref `${x :- y}`, `${item}` scope);
- full expressions inside a binding span, including arithmetic and string/list operations.
"""

import pytest

from yian.expr import RequiredError, eval_binding, expr_refs_of


def _from(mapping):
    """A resolver over a flat dict (a path it lacks -> None, the BINDING_NONE miss)."""
    return lambda path: mapping.get(path)


# --- binding value semantics ------------------------------------------------- #


def test_whole_span_is_typed():
    # a value that is EXACTLY one ${...} -> the typed value (not the string).
    assert eval_binding("${a}", _from({"a": 0.7})) == 0.7
    assert eval_binding("${a}", _from({"a": ["x", "y"]})) == ["x", "y"]


def test_embedded_span_is_stringified():
    assert eval_binding("pe=${a}", _from({"a": 0.7})) == "pe=0.7"


def test_dollar_escape_is_literal_dollar():
    assert eval_binding("cost is $$5", _from({})) == "cost is $5"


def test_coalesce_first_non_none():
    assert eval_binding("${a | b}", _from({"b": "second"})) == "second"
    # a present falsy value wins (only None falls through).
    assert eval_binding("${a | b}", _from({"a": 0, "b": "second"})) == 0


def test_default_and_required():
    assert eval_binding('${a :- "d"}', _from({})) == "d"
    assert eval_binding("${a :- 30}", _from({})) == 30
    # the `:?` message is a full expression -> quote a plain-text message.
    with pytest.raises(RequiredError, match="need a topic"):
        eval_binding('${a :? "need a topic"}', _from({}))


def test_default_ref():
    # `${a :- b}`: the default ref resolves when the head misses.
    assert eval_binding("${a :- b}", _from({"b": "fb"})) == "fb"


def test_item_scope():
    assert eval_binding("${item.title}", _from({}), item={"title": "T"}) == "T"
    assert eval_binding("${item}", _from({}), item="whole") == "whole"


# --- full expressions inside a binding span ---------------------------------- #


def test_arithmetic_in_binding():
    # A binding span evaluates arithmetic over resolved values.
    assert eval_binding("${x + 1}", _from({"x": 2})) == 3


def test_string_and_list_ops_in_binding():
    assert eval_binding("${a + b}", _from({"a": "foo", "b": "bar"})) == "foobar"
    assert eval_binding("${xs + [item]}", _from({"xs": [1, 2]}), item=3) == [1, 2, 3]


# --- expr_refs_of: the raw-string ref-walk ----------------------------------- #


def test_expr_refs_of_collects_span_refs_in_order():
    assert expr_refs_of("${a.output | b.output}") == ["a.output", "b.output"]


def test_expr_refs_of_literal_text_contributes_nothing():
    assert expr_refs_of("just text, no spans") == []


def test_expr_refs_of_embedded_and_default_and_arith():
    # embedded span + default ref + arithmetic operands all collected.
    assert expr_refs_of("n=${a :- b} m=${x + y}") == ["a", "b", "x", "y"]


def test_expr_refs_of_item_is_collected():
    # item-headed refs ARE collected (caller skips them for edge minting).
    assert expr_refs_of("${item.x}") == ["item.x"]


def test_capture_marks_actual_template_evaluation_in_materialized_seed(monkeypatch):
    import yian.expr.template as template
    from yian.expr.lineage import ResolvedValue
    from yian.nodes.binding import BindingCollector, bind_params
    from yian.nodes.contracts import Parameter
    from yian.typesys.pool import VariablePool

    scanned = []
    parsed = template.grammar.parse_expr

    def parse_once(text):
        scanned.append(text)
        return parsed(text)

    monkeypatch.setattr(template.grammar, "parse_expr", parse_once)
    collector = BindingCollector()
    result = bind_params(
        [Parameter("evaluated"), Parameter("escaped"), Parameter("literal")],
        {
            "evaluated": "${missing | absent}",
            "escaped": "$${missing | absent}",
            "literal": None,
        },
        VariablePool(), collector=collector,
        source_resolver=lambda path: ResolvedValue(None, False),
    )
    captured = collector.freeze()
    assert result == {
        "evaluated": None, "escaped": "${missing | absent}", "literal": None,
    }
    assert captured.evaluated_keys == frozenset({"evaluated"})
    assert all(route.kind == "literal" for route in captured.route.children)
    assert scanned == ["missing | absent"]
