"""Messages that show a value or key stay within `YIAN_MAX_SHOWN_VALUE_CHARS`."""

import pytest

from yian.compose import load_flow
from yian.nodes.binding import check_input_names
from yian.nodes.contracts import Parameter, output_contract_origin
from yian.run import run_flow
from yian.typesys import Type, TypeCheckError, ValueKind, build_value_as
from yian.typesys.values import _sort_repr, value_repr

from tests.engine._bounded_codefns import LONG_FIELD, BrokenRepr

CAP = 200
MESSAGE_BOUND = 6 * CAP


def _flow(fn: str, output: str | None = None) -> str:
    """Return a flow whose one CODE node runs `fn` with an optional declared output."""
    output_line = f"    output: {output}\n" if output is not None else ""
    return (
        "id: bounded\nname: bounded\nnodes:\n  b:\n    kind: code\n"
        f"    code: tests.engine._bounded_codefns:{fn}\n"
        "    input:\n      topic: x\n"
        f"{output_line}"
        "output: ${b.output}\n"
    )


def _failure(fn: str, output: str | None = None):
    """Run the flow for `fn` and return its failure, which must be a `contract.output`."""
    result = run_flow(load_flow(_flow(fn, output)), {}, num_workers=0)
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "contract.output"
    return failure


def test_value_repr_cuts_a_long_set_to_the_cap():
    shown = value_repr(set(range(200_000)))
    assert len(shown) == CAP
    assert shown.endswith("…")
    assert shown.startswith("{0, 1, 2")


def test_value_repr_cuts_a_long_string_to_the_cap():
    shown = value_repr("x" * 500)
    assert shown == "'" + "x" * (CAP - 2) + "…"


def test_value_repr_keeps_a_repr_exactly_at_the_cap():
    value = "x" * (CAP - 2)
    assert value_repr(value) == repr(value)
    assert len(value_repr(value)) == CAP


def test_value_repr_keeps_placeholders():
    class Broken:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    assert value_repr(10**5000) == "<int with 5001 digits>"
    assert value_repr([10**5000]) == "<list holding an int with 5001 digits>"
    assert value_repr(Broken()) == "<Broken object>"


def test_value_repr_follows_the_configured_cap(set_defaults):
    set_defaults(MAX_SHOWN_VALUE_CHARS=10)
    assert value_repr("abcdefghijklmnop") == "'abcdefgh…"


def test_record_lists_many_unknown_fields_within_the_cap():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"a": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"a"}),
    )
    value = {f"key{i}": i for i in range(200_000)}
    value["a"] = 1
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(record, value)
    message = str(caught.value)
    assert message.startswith("record has unknown fields: [")
    assert len(message) <= MESSAGE_BOUND
    assert caught.value.path == ("key0",)


def test_sort_repr_orders_past_the_cap_and_keeps_placeholders():
    assert _sort_repr("x" * 300 + "a") < _sort_repr("x" * 300 + "b")
    assert _sort_repr(BrokenRepr()) == "<BrokenRepr object>"


def test_record_sorts_mixed_unknown_keys_by_type_without_crashing():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"a": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"a"}),
    )
    # Equal up to the cap, so a sort on the cut text could not order them.
    later, earlier = "x" * 300 + "b", "x" * 300 + "a"
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(record, {"a": 1, later: 1, earlier: 2, 1: 3, BrokenRepr(): 4})
    message = str(caught.value)
    assert message.startswith("record has unknown fields: [<BrokenRepr object>, 1, 'xxx")
    assert len(message) <= MESSAGE_BOUND


def test_unknown_inputs_list_within_the_cap():
    raw = {f"name{i}": i for i in range(200_000)}
    with pytest.raises(ValueError) as caught:
        check_input_names([Parameter("topic")], raw)
    message = str(caught.value)
    assert message.startswith("unknown inputs 'name0', 'name1', ")
    assert message.endswith("; declared inputs: topic")
    assert len(message) <= MESSAGE_BOUND


@pytest.mark.parametrize("output", [None, "Any"])
def test_output_holding_a_big_set_shows_a_cut_repr(output):
    failure = _failure("big_set", output)
    assert "…" in failure.message
    assert len(failure.message) < 300


@pytest.mark.parametrize("output", [None, "Any", "{s: Any}"])
def test_too_deep_value_under_long_keys_shows_a_bounded_path(output):
    failure = _failure("long_str_keys", output)
    assert "value nests deeper" in failure.message
    assert "…" in failure.message
    assert len(failure.message) <= MESSAGE_BOUND


@pytest.mark.parametrize("output", [None, "Any"])
@pytest.mark.parametrize("fn", ["long_tuple_key", "long_int_key", "broken_repr_key"])
def test_too_deep_value_under_a_non_text_key_shows_a_bounded_message(fn, output):
    failure = _failure(fn, output)
    assert "value nests deeper" in failure.message
    assert len(failure.message) <= MESSAGE_BOUND
    if fn == "long_int_key":
        assert "<int with 5001 digits>" in failure.message
    if fn == "broken_repr_key":
        assert "<BrokenRepr object>" in failure.message


def test_output_record_with_many_unknown_fields_shows_a_bounded_message():
    failure = _failure("many_unknown_fields", "{a: int}")
    assert "unknown fields" in failure.message
    assert len(failure.message) <= MESSAGE_BOUND


def test_long_declared_field_keeps_its_origin():
    flow = load_flow(_flow("wrong_long_field", f"{{{LONG_FIELD}: int, g: int}}"))
    contract = flow.compiled.nodes["b"].output_contract
    field_origin = output_contract_origin(contract, (LONG_FIELD,))
    assert field_origin is not None
    assert field_origin != output_contract_origin(contract, ())

    result = run_flow(flow, {}, num_workers=0)
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.trace == (field_origin,)


@pytest.mark.parametrize("output", [None, "Any"])
@pytest.mark.parametrize("fn", ["deep_list", "deep_tuple", "dict_of_deep_list"])
def test_deep_output_is_located(fn, output):
    failure = _failure(fn, output)
    assert "value nests deeper than 100 levels" in failure.message
    assert " at " in failure.message
    assert "canonical data nesting" not in failure.message


@pytest.mark.parametrize("output", [None, "Any", "list(Any)"])
def test_deep_frozenset_output_is_not_a_json_value(output):
    failure = _failure("deep_frozenset", output)
    assert (
        "is not a JSON value" in failure.message
        or "is not a list for declared" in failure.message
    )
    assert "canonical data nesting" not in failure.message


def test_list_hiding_a_deep_list_fails_the_output_contract():
    result = run_flow(load_flow(_flow("hidden_deep_list")), {}, num_workers=0)
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "b"
