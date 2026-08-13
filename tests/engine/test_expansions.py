"""Round-trip tests for the uniform `GrowRecord` durability ledger.

A `GrowRecord` is pure durability metadata appended to `RunCheckpoint.expansions`; the restore-side
replay (`_replay_expansions`) is exercised elsewhere. These verify the one uniform record type
round-trips cleanly across the wire — the per-kind `seed` shapes ride an opaque `Any`, and nested
grows ride the flat `children`.
"""

from pydantic import TypeAdapter

from agent_composer.suspension.expansions import GrowRecord


def test_call_grow_record_round_trip() -> None:
    # CALL: seed is the call-arg record dict; one record per call, no children (leaf child).
    rec = GrowRecord(spawner_id="analyze", seed={"x": 1}, children=[])
    restored = GrowRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_map_grow_record_round_trip() -> None:
    # MAP: seed is the per-element records list; one record per map.
    rec = GrowRecord(spawner_id="each", seed=[{"i": 0}, {"i": 1}], children=[])
    restored = GrowRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_loop_grow_record_round_trip_seed_is_pair() -> None:
    # LOOP: seed is the LIVE iteration's (record, index); the tuple round-trips through JSON as a
    # 2-element list, so the restored seed is [record, index].
    rec = GrowRecord(spawner_id="loop", seed=[{"acc": 3}, 2], children=[])
    restored = GrowRecord.model_validate_json(rec.model_dump_json())
    assert restored.spawner_id == "loop"
    assert restored.seed == [{"acc": 3}, 2]


def test_nested_call_grow_record_chain_round_trip() -> None:
    inner = GrowRecord(spawner_id="outer/inner", seed={"x": 2}, children=[])
    outer = GrowRecord(spawner_id="outer", seed={"x": 1}, children=[inner])

    restored = GrowRecord.model_validate_json(outer.model_dump_json())

    assert restored == outer
    assert restored.children[0].spawner_id == "outer/inner"


def test_grow_record_list_round_trip_preserves_order_and_primitives() -> None:
    # A ledger is an ordered list of top-level GrowRecords; the per-kind opaque seeds preserve
    # nested primitives through the wire.
    records: list[GrowRecord] = [
        GrowRecord(spawner_id="a", seed={"k": "v"}, children=[]),
        GrowRecord(
            spawner_id="b",
            seed={"l": [1, 2, "three"], "d": {"nested": {"deep": [None, "x"]}}},
            children=[],
        ),
    ]
    adapter = TypeAdapter(list[GrowRecord])
    raw = adapter.dump_python(records)
    restored = adapter.validate_python(raw)
    assert restored == records


def test_top_level_reexports() -> None:
    """`GrowRecord` is re-exported from the suspension package."""
    from agent_composer.suspension import GrowRecord as GR

    assert GR is GrowRecord
