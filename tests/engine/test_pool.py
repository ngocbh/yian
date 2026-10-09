"""Unit tests for VariablePool.

Feature -> contract:
- set / get_value / get                : typed read/write of a node's single value
- resolve(head, rest)                   : backs ${...}; structured traversal
- set with declared type                : write-time type-drift detection
- dumps / loads                         : lossless full-pool round-trip
"""

import pytest

from yian.compile.model import START_ID
from yian.typesys.pool import VariablePool
from yian.typesys.values import (
    ListObjectValue,
    Type,
    TypeCheckError,
    ValueKind,
)


def test_set_and_get():
    pool = VariablePool()
    pool.set("n1", "hello")
    assert pool.get("n1") == "hello"
    assert pool.get_value("n1").kind == ValueKind.STRING
    # a missing node resolves to the default
    assert pool.get("missing", default="d") == "d"


def test_get_whole_object_value():
    # "multiple outputs" are fields of one object value, read back whole.
    pool = VariablePool()
    pool.set("n1", {"answer": "txt", "score": 0.8})
    assert pool.get("n1") == {"answer": "txt", "score": 0.8}
    assert pool.get("nope") is None


def test_resolve_node_first_and_structured():
    # `.output` is a SYNTACTIC discriminator the resolver SKIPS — store the
    # node's value directly (do NOT wrap with `{"output": ...}`, which would only test
    # the dict-key arm by accident).
    pool = VariablePool()
    pool.set("reviewer", {"ratio": 21.5, "nested": {"k": "v"}})
    # ${reviewer.output} (2-segment) returns the whole stored value
    assert pool.resolve("reviewer", ["output"]) == {"ratio": 21.5, "nested": {"k": "v"}}
    # ${reviewer.output.ratio} (3-segment) walks one level in
    assert pool.resolve("reviewer", ["output", "ratio"]) == 21.5
    assert pool.resolve("reviewer", ["output", "nested", "k"]) == "v"
    # missing path -> None, not an error
    assert pool.resolve("reviewer", ["output", "absent"]) is None
    assert pool.resolve("ghost", ["output"]) is None


def test_retired_heads_resolve_none():
    # `node` and `ref_outputs` were unified into `<id>.output` — they
    # are now just unknown heads and resolve to None.
    pool = VariablePool()
    pool.set("reviewer", "txt")
    assert pool.resolve("node", ["reviewer", "output"]) is None
    assert pool.resolve("ref_outputs", ["reviewer", "output"]) is None


def test_resolve_inputs_and_system():
    pool = VariablePool()
    pool.set(START_ID, {"topic": "ACME"})  # ${input.X} ≡ store[START_ID].X
    assert pool.resolve("input", ["topic"]) == "ACME"  # the flow's run args
    assert pool.resolve("input", ["missing"]) is None
    assert pool.resolve("system", ["anything"]) is None  # ambient namespace, unseeded
    # `parent` is retired — it is now just an unknown head -> None
    assert pool.resolve("parent", ["threshold"]) is None
    assert not hasattr(pool, "parent")
    assert pool.resolve("unknown_ns", ["x"]) is None


def test_subflow_outputs_resolve_via_node_first_head():
    # A CALL/subflow node's value lives under its own id in `store` and resolves via
    # the node-first `<id>.output` head (`.output` is a syntactic SKIP).
    pool = VariablePool()
    pool.set("child_ref", "child-result")
    assert pool.resolve("child_ref", ["output"]) == "child-result"


def test_declared_type_drift_raises():
    pool = VariablePool()
    # a node declaring an integer output but emitting a string fails at the write
    with pytest.raises(TypeCheckError):
        pool.set("n1", "twelve", declared=ValueKind.INTEGER)
    # the good case writes through
    pool.set("n1", 12, declared=ValueKind.INTEGER)
    assert pool.get("n1") == 12


def test_pool_set_accepts_type_and_enforces():
    from yian.typesys.values import Type

    pool = VariablePool()
    action = Type(kind=ValueKind.STRING, tags=frozenset({"Approve", "Reject"}))
    pool.set("n", "Approve", declared=action)
    assert pool.get("n") == "Approve"
    with pytest.raises(TypeCheckError):
        pool.set("n", "approve", declared=action)


def test_inputs_resolve_via_start_store():
    # the standalone `inputs` namespace (+ add_inputs e08) is retired. ${input.X}
    # resolves to store[START_ID].X — the START_ID node's committed bound-input record (e08 now
    # lives on StartNode.run, covered by the start-node / run_flow tests).
    pool = VariablePool()
    pool.set(START_ID, {"window": 30, "note": "anything"})
    assert pool.resolve("input", ["window"]) == 30
    assert pool.resolve("input", ["note"]) == "anything"
    assert not hasattr(pool, "add_inputs")
    assert "inputs" not in VariablePool.model_fields


def test_full_pool_round_trip():
    pool = VariablePool()
    pool.set("n1", {"a": 1, "b": [1.5, 2.5]})
    pool.set("flag_node", True)
    pool.set("n2", ["ACME", "BETA"])
    pool.set(START_ID, {"as_of_date": "2026-06-05"})  # the flow's run args (the START_ID record)

    back = VariablePool.loads(pool.dumps())

    assert back.get("n1") == {"a": 1, "b": [1.5, 2.5]}
    assert back.get_value("flag_node").kind == ValueKind.BOOLEAN
    assert back.get_value("n2").kind == ValueKind.LIST_STRING
    assert back.resolve("input", ["as_of_date"]) == "2026-06-05"  # store[START_ID] survives dumps/loads
    # types survived: flag is still a real bool, not int
    assert back.get("flag_node") is True


def test_pool_round_trip_preserves_nullable_specialized_list_elements():
    row = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"score"}),
        nullable=True,
    )
    rows = Type(kind=ValueKind.LIST_OBJECT, element=row)
    pool = VariablePool()
    pool.set("rows", [{"score": 0.9}, None], declared=rows)

    restored = VariablePool.loads(pool.dumps())

    stored = restored.get_value("rows")
    assert type(stored) is ListObjectValue
    assert stored.kind is ValueKind.LIST_OBJECT
    assert restored.get("rows") == [{"score": 0.9}, None]
