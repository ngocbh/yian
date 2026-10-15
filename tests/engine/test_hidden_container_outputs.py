"""A node output is checked and stored as one plain copy, read through Python iteration."""

import json

import pytest

from yian.compose import load_flow
from yian.run import run_flow

from tests.engine import _hidden_container_codefns as codefns


def _run(fn: str, output: str | None = None):
    """Run a flow whose one CODE node runs `fn` with an optional declared output."""
    output_line = f"    output: {output}\n" if output is not None else ""
    text = (
        "id: f\nname: f\nnodes:\n  b:\n    kind: code\n"
        f"    code: tests.engine._hidden_container_codefns:{fn}\n"
        "    input:\n      topic: x\n"
        f"{output_line}"
        "output: ${b.output}\n"
    )
    return run_flow(load_flow(text), {}, num_workers=0)


def _failure(fn: str, output: str | None = None):
    """Run the flow for `fn` and return its failure, which must be a `contract.output`."""
    result = _run(fn, output)
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "b"
    return result.failure


@pytest.mark.parametrize(
    ("fn", "output", "expected"),
    [
        ("hidden_list", None, {"s": []}),
        ("hidden_list", "Any", {"s": []}),
        ("hidden_dict_nan", None, {"s": {}}),
        ("hidden_dict_nan", "object", {"s": {}}),
        ("hidden_dict_int_key", None, {"s": {}}),
        ("shows_a_float", None, {"s": [1.5]}),
        ("typed_hidden_list", None, {"s": []}),
        ("typed_hidden_list", "Any", {"s": []}),
    ],
)
def test_stored_output_is_what_iteration_yields(fn, output, expected):
    result = _run(fn, output)
    assert result.status == "succeeded"
    assert result.output == expected


def test_a_cycle_through_a_subclass_reads_it_once():
    codefns.COUNTED.clear()
    _failure("cycle_through_subclass")
    assert [sub.iterations for sub in codefns.COUNTED] == [1]


def test_a_yielded_nan_is_located():
    message = _failure("shows_a_nan").message
    assert "non-finite" in message
    assert "s[0]" in message


def test_a_yielded_set_is_located():
    message = _failure("shows_a_set").message
    assert "is not a JSON value" in message
    assert "s[0]" in message


def test_a_yielded_text_key_is_required():
    assert "object keys must be text, got 1" in _failure("items_int_key").message


@pytest.mark.parametrize("output", [None, "object"])
def test_a_yielded_unhashable_key_is_located(output):
    message = _failure("items_unhashable_key", output).message
    assert message == "output at s: object keys must be text, got [1]"


@pytest.mark.parametrize(
    ("fn", "output"),
    [
        ("ordered_dict", None),
        ("ordered_dict", "{s: {b: int, a: int}}"),
        ("default_dict", None),
        ("default_dict", "{s: {k: list(int)}}"),
        ("named_tuple", None),
        ("named_tuple", "{s: list(int)}"),
        ("plain_sub", None),
        ("plain_sub", "{s: list(int)}"),
    ],
)
def test_ordinary_subclasses_store_like_their_plain_equivalents(fn, output):
    result = _run(fn, output)
    plain = _run(f"{fn}_plain", output)
    assert result.status == plain.status == "succeeded"
    assert json.dumps(result.output) == json.dumps(plain.output)
