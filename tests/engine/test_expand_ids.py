from yian.nodes.ids import map_callsite, ns


def test_ns_joins_callsite_and_child_id():
    assert ns("each", "leaf") == "each/leaf"          # both node AND edge ids
    assert ns("each#0/inner#0", "leaf") == "each#0/inner#0/leaf"   # nested composes


def test_map_callsite_is_spawner_hash_index():
    assert map_callsite("each", 0) == "each#0"
    assert map_callsite("each", 12) == "each#12"


def test_minting_is_pure_re_callable():
    # No emission counter — a re-clone on recovery re-keys identically.
    assert ns("each#0", "x") == ns("each#0", "x")
    assert map_callsite("each", 3) == map_callsite("each", 3)
