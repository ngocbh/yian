"""Round-trip tests for the uniform `GrowRecord` durability ledger.

A `GrowRecord` is pure durability metadata appended to `RunCheckpoint.expansions`; the restore-side
replay (`_replay_expansions`) is exercised elsewhere. These verify the one uniform record type
round-trips cleanly across the wire — the per-kind `seed` shapes ride an opaque `Any`, and nested
grows ride the flat `children`. `AgentSegment` is the read-side validation schema for one AGENT
pause's `{hi_desc, resume_desc}` seed.
"""

from pydantic import TypeAdapter

from agent_composer.suspension.expansions import AgentSegment, GrowRecord


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


def test_agent_grow_record_nesting_chain_round_trip() -> None:
    # AGENT: a K-pause agent is a NESTING CHAIN — segment i+1 rides under segment i's `children`,
    # keyed by the resume-terminal id. Each seed is one pause's {hi_desc, resume_desc}.
    hi0 = {"kind": "human_input", "node_id": "agent/hi#0", "prompt": "What next?", "slot": "0"}
    resume0 = {"kind": "resume_agent", "memo": {"turns": 2}, "iterations": 2, "pending": {}}
    hi1 = {"kind": "human_input", "node_id": "agent/__resume#0/hi#1", "prompt": "And?", "slot": "1"}
    resume1 = {"kind": "resume_agent", "memo": {"turns": 4}, "iterations": 4, "pending": {}}
    seg1 = GrowRecord(
        spawner_id="agent/__resume#0", seed={"hi_desc": hi1, "resume_desc": resume1}, children=[]
    )
    seg0 = GrowRecord(
        spawner_id="agent", seed={"hi_desc": hi0, "resume_desc": resume0}, children=[seg1]
    )
    restored = GrowRecord.model_validate_json(seg0.model_dump_json())
    assert restored == seg0
    # The nested chain survives: segment 1 is the sole child of segment 0.
    assert restored.children[0].spawner_id == "agent/__resume#0"


def test_agent_segment_validates_a_pause_seed() -> None:
    # The public model validates the opaque agent seed's descriptor-pair shape.
    seed = {
        "hi_desc": {"kind": "human_input", "node_id": "a/hi#0", "prompt": "?", "slot": "0"},
        "resume_desc": {"kind": "resume_agent", "memo": {}, "iterations": 1, "pending": {}},
    }
    seg = AgentSegment.model_validate(seed)
    assert seg.hi_desc["slot"] == "0"
    assert seg.resume_desc["iterations"] == 1


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
    """`GrowRecord`/`AgentSegment` re-exported from the suspension package."""
    from agent_composer.suspension import AgentSegment as AS
    from agent_composer.suspension import GrowRecord as GR

    assert GR is GrowRecord
    assert AS is AgentSegment
