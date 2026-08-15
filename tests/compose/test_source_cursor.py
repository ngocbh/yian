"""Tests for immutable, occurrence-aware YAML source cursors."""

from collections.abc import Mapping, Sequence, Set
from datetime import datetime

import pytest

from agent_composer.compose.source import (
    SourceAccessError,
    SourceCursor,
    SourceDocument,
)
from agent_composer.events import SourceRange


def test_document_preserves_constructed_keys_and_effective_collision_ranges():
    text = """\
values:
  0x10: hexadecimal
  1.50: floating
  2001-12-15T02:59:43.1Z: timestamp
  !!binary YQ==: binary
collision:
  true: first
  1:
    nested: second
"""
    root = SourceDocument.parse(text).root
    values = root.child("values")
    timestamp_key = next(
        key for key in values.value if isinstance(key, datetime)
    )

    assert values.get(16) == "hexadecimal"
    assert values.get(1.5) == "floating"
    assert values.get(timestamp_key) == "timestamp"
    assert values.get(b"a") == "binary"
    assert values.range_of(16) == SourceRange(2, 9, 2, 20)

    collision = root.child("collision").child(True)
    assert collision.source_key_value == 1
    assert type(collision.source_key_value) is int
    assert collision.range == SourceRange(9, 5, 10, 1)
    assert collision.str("nested") == "second"

    copied_values = values.value
    copied_values[16] = "mutated"
    assert values.get(16) == "hexadecimal"


def test_root_value_is_a_defensive_copy():
    document = SourceDocument.parse(
        "root:\n"
        "  items:\n"
        "    - name: first\n"
    )

    first = document.root.value
    first["added"] = True
    first["root"]["items"][0]["name"] = "changed"

    assert document.root.value == {
        "root": {"items": [{"name": "first"}]},
    }


def test_value_copy_preserves_alias_identity_and_recursive_alias():
    document = SourceDocument.parse(
        "shared: &shared\n"
        "  value: 1\n"
        "left: *shared\n"
        "right: *shared\n"
        "root: &root\n"
        "  self: *root\n"
    )

    copied = document.root.value

    assert copied["shared"] is copied["left"]
    assert copied["left"] is copied["right"]
    assert copied["root"]["self"] is copied["root"]


def test_value_view_is_recursive_and_read_only():
    document = SourceDocument.parse(
        "shared: &shared\n"
        "  values: [1, 2]\n"
        "left: *shared\n"
        "right: *shared\n"
        "root: &root\n"
        "  self: *root\n"
        "names: &names !!set {alice: null, bob: null}\n"
        "names_left: *names\n"
        "names_right: *names\n"
        "ordered: !!omap [{first: [1, {nested: true}]}, {second: 2}]\n"
        "pairs: !!pairs [{same: {nested: [1]}}, {same: 2}]\n"
    )

    view = document.root._value_view()

    assert isinstance(view, Mapping)
    assert view["shared"] is view["left"]
    assert view["left"] is view["right"]
    assert view["root"]["self"] is view["root"]
    assert isinstance(view["shared"]["values"], Sequence)
    assert isinstance(view["names"], Set)
    assert view["names"] is view["names_left"]
    assert view["names_left"] is view["names_right"]
    assert set(view["names"]) == {"alice", "bob"}
    assert [item[0] for item in view["ordered"]] == ["first", "second"]
    assert view["ordered"][0][1][0] == 1
    assert view["ordered"][0][1][1]["nested"] is True
    assert [item[0] for item in view["pairs"]] == ["same", "same"]
    assert view["pairs"][0][1]["nested"][0] == 1
    assert view["pairs"][1][1] == 2
    with pytest.raises(TypeError):
        view["left"]["new"] = 2
    with pytest.raises(AttributeError):
        view["shared"]["values"].append(3)
    with pytest.raises(AttributeError):
        view["names"].add("carol")
    with pytest.raises(AttributeError):
        view["ordered"][0][1].append(3)
    with pytest.raises(TypeError):
        view["pairs"][0][1]["new"] = False


def test_value_view_set_supports_hashable_container_members_and_operations():
    original = {("tuple", 1), frozenset({"nested"})}
    view = SourceCursor.virtual(original)._value_view()

    assert ("tuple", 1) in view
    assert frozenset({"nested"}) in view
    assert set(view) == original
    assert view & {("tuple", 1)} == {("tuple", 1)}
    assert view | {("added", 2)} == original | {("added", 2)}
    assert type(view & original) is set
    with pytest.raises(TypeError, match="unhashable"):
        ["tuple", 1] in view
    with pytest.raises(TypeError, match="unhashable"):
        hash(view)

    frozen = frozenset(original)
    frozen_view = SourceCursor.virtual(frozen)._value_view()
    assert hash(frozen_view) == hash(frozen)
    assert frozen_view & {("tuple", 1)} == frozenset({("tuple", 1)})
    assert frozen_view | {("added", 2)} == frozen | {("added", 2)}
    assert type(frozen_view & frozen) is frozenset


def test_value_view_set_preserves_nan_container_membership():
    nan = float("nan")
    original = {(nan,)}
    view = SourceCursor.virtual(original)._value_view()

    assert (nan,) in view
    assert next(iter(view)) == (nan,)
    assert set(view) == original


def test_virtual_rejects_custom_objects_and_container_subclasses():
    class Custom:
        pass

    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    class TupleSubclass(tuple):
        pass

    class SetSubclass(set):
        pass

    class FrozenSetSubclass(frozenset):
        pass

    class StrSubclass(str):
        pass

    invalid = [
        Custom(),
        DictSubclass(),
        ListSubclass(),
        TupleSubclass(),
        SetSubclass(),
        FrozenSetSubclass(),
        StrSubclass("text"),
        {"nested": ListSubclass()},
    ]

    base = SourceCursor.virtual({})
    for value in invalid:
        with pytest.raises(TypeError, match="SafeLoader value"):
            SourceCursor.virtual(value)
        with pytest.raises(TypeError, match="SafeLoader value"):
            base.with_value(value)
        with pytest.raises(TypeError, match="SafeLoader value"):
            base.project_mapping({"value": (value, None)})


def _binary_alias_document(levels: int = 24) -> SourceDocument:
    lines = ["level_0: &level_0 [leaf]"]
    for level in range(1, levels + 1):
        lines.append(
            f"level_{level}: &level_{level} "
            f"[*level_{level - 1}, *level_{level - 1}]"
        )
    lines.append(f"root: *level_{levels}")
    return SourceDocument.parse("\n".join(lines) + "\n")


def test_definition_dag_is_linear_for_binary_alias_graph():
    document = _binary_alias_document()

    assert len(document._definitions) < 200
    assert len({id(definition) for definition in document._definitions}) == len(
        document._definitions
    )


def _binary_merge_document(levels: int) -> SourceDocument:
    """Return a short YAML document with exponentially many logical merge pairs."""

    lines = ["level_0: &level_0", "  value: leaf"]
    for level in range(1, levels + 1):
        lines.extend(
            (
                f"level_{level}: &level_{level}",
                f"  <<: [*level_{level - 1}, *level_{level - 1}]",
            )
        )
    lines.append(f"root: *level_{levels}")
    return SourceDocument.parse("\n".join(lines) + "\n")


def test_merge_definition_dag_retains_only_unique_raw_pairs():
    levels = 16
    document = _binary_merge_document(levels)
    pending = [
        definition.pair_sequence
        for definition in document._definitions
        if hasattr(definition, "pair_sequence")
    ]
    sequences = {}
    while pending:
        sequence = pending.pop()
        if id(sequence) in sequences:
            continue
        sequences[id(sequence)] = sequence
        pending.extend(sequence.parts)
    retained_arcs = sum(len(sequence.arcs) for sequence in sequences.values())
    retained_effective = sum(
        len(definition.effective_index)
        for definition in document._definitions
        if hasattr(definition, "effective_index")
    )
    root_node = document._root_node
    assert root_node is not None
    deepest_node = next(
        value_node
        for key_node, value_node in root_node.value
        if key_node.value == f"level_{levels}"
    )

    assert document.root.child("root").get("value") == "leaf"
    assert retained_arcs < levels * 8
    assert retained_effective < levels * 8
    assert len(sequences) < levels * 8
    assert max(sequence.flat_pair_count for sequence in sequences.values()) >= 2**levels
    assert len(deepest_node.value) == 1


def test_repeated_merge_occurrences_keep_logical_ordinals_and_replay():
    document = _binary_merge_document(5)
    target = document.root.child("root")

    occurrences = target.mapping_occurrences("value")

    assert len(occurrences) == 32
    assert [item.path[-1].pair_ordinal for item in occurrences] == list(range(32))
    assert {item.range for item in occurrences} == {SourceRange(2, 10, 2, 14)}
    for item in (occurrences[0], occurrences[-1]):
        assert document._resolve_occurrence(item.path).path == item.path
        assert document._resolve_occurrence(item.path).value == "leaf"


def test_merge_range_summary_folds_shared_rope_without_enumerating(monkeypatch):
    document = _binary_merge_document(24)

    def reject_enumeration(*_args, **_kwargs):
        raise AssertionError("effective range lookup must not flatten occurrences")

    monkeypatch.setattr(
        SourceCursor,
        "_iter_flat_occurrences",
        reject_enumeration,
    )

    assert document.root.mapping_value_ranges("root") == {
        "value": SourceRange(2, 10, 2, 14),
    }


def test_recursive_sequence_merge_preserves_order_occurrences_and_replay():
    document = SourceDocument.parse(
        "base: &base {base_only: B, clash: base}\n"
        "root: &root\n"
        "  <<: [*base, *root]\n"
        "  root_only: R\n"
        "  clash: root\n"
    )
    root = document.root.child("root")
    occurrences = root.mapping_occurrences()

    assert list(root.value) == ["root_only", "clash", "base_only"]
    assert [item.source_key_value for item in occurrences] == [
        "root_only",
        "clash",
        "base_only",
        "clash",
        "root_only",
        "clash",
    ]
    assert [item.path[-1].pair_ordinal for item in occurrences] == list(range(6))
    for item in occurrences:
        replayed = document._resolve_occurrence(item.path)
        assert replayed.path == item.path
        assert replayed.value == item.value


def test_double_recursive_merge_retains_four_logical_local_occurrences():
    root = SourceDocument.parse(
        "root: &root\n"
        "  <<: *root\n"
        "  <<: *root\n"
        "  value: local\n"
    ).root.child("root")

    occurrences = root.mapping_occurrences("value")

    assert len(occurrences) == 4
    assert [item.path[-1].pair_ordinal for item in occurrences] == list(range(4))


def test_repeated_self_source_in_one_merge_sequence_retains_three_occurrences():
    root = SourceDocument.parse(
        "root: &root\n"
        "  <<: [*root, *root]\n"
        "  value: local\n"
    ).root.child("root")

    occurrences = root.mapping_occurrences("value")

    assert len(occurrences) == 3
    assert [item.path[-1].pair_ordinal for item in occurrences] == list(range(3))


def test_many_recursive_merges_keep_metadata_linear_and_conversion_iterative():
    width = 1_200
    lines = ["root: &root"]
    lines.extend("  <<: *root" for _ in range(width))
    lines.extend(f"  key_{index}: {index}" for index in range(width))

    document = SourceDocument.parse("\n".join(lines) + "\n")
    root = document.root.child("root")
    pending = [
        definition.pair_sequence
        for definition in document._definitions
        if hasattr(definition, "pair_sequence")
    ]
    sequences = {}
    while pending:
        sequence = pending.pop()
        if id(sequence) in sequences:
            continue
        sequences[id(sequence)] = sequence
        pending.extend(sequence.parts)

    assert root.get("key_0") == 0
    assert root.get(f"key_{width - 1}") == width - 1
    assert root._mapping_definition().pair_sequence.flat_pair_count == width * 2**width
    assert sum(len(sequence.arcs) for sequence in sequences.values()) < width * 3
    assert sum(
        len(definition.effective_index)
        for definition in document._definitions
        if hasattr(definition, "effective_index")
    ) < width * 3


def test_nested_mapping_can_merge_its_recursive_value_ancestor():
    document = SourceDocument.parse(
        "root: &root\n"
        "  child: &child\n"
        "    <<: *root\n"
        "    leaf: child\n"
        "  own: root\n"
    )
    copied = document.root.value
    child = copied["root"]["child"]

    assert child["child"] is child
    assert child["own"] == "root"
    assert child["leaf"] == "child"


def test_deep_alias_path_is_lazy():
    cursor = _binary_alias_document().root.child("root")

    for _ in range(24):
        cursor = cursor.child(0)

    assert not isinstance(cursor._path_link, tuple)
    assert len(cursor.path) == 25


def test_mapping_child_and_children_use_effective_order():
    root = SourceDocument.parse(
        "first: original\n"
        "second:\n"
        "  nested: true\n"
        "first: final\n"
    ).root

    first = root.child("first")
    children = root.children()

    assert first.value == "final"
    assert first.range == SourceRange(4, 8, 4, 13)
    assert first.key_range == SourceRange(4, 1, 4, 6)
    assert [child.value for child in children] == [
        "final",
        {"nested": True},
    ]


def test_physical_and_projected_wide_mapping_lookup_is_indexed(monkeypatch):
    width = 200
    root = SourceDocument.parse(
        "".join(f"key_{index}: {index}\n" for index in range(width))
    ).root
    projected = root.project_mapping(
        {
            f"key_{index}": (index, root.child(f"key_{index}"))
            for index in range(width)
        }
    )
    comparisons = 0
    original = SourceCursor._keys_equal

    def counted(left, right):
        nonlocal comparisons
        comparisons += 1
        return original(left, right)

    monkeypatch.setattr(SourceCursor, "_keys_equal", staticmethod(counted))

    assert [child.value for child in root.children()] == list(range(width))
    for index in range(width):
        key = f"key_{index}"
        assert projected.has(key)
        assert projected.child(key).value == index

    assert comparisons <= width * 4


def test_many_one_key_merge_sources_use_root_effective_index(monkeypatch):
    width = 200
    lines = [
        f"source_{index}: &source_{index} {{key_{index}: {index}}}"
        for index in range(width)
    ]
    sources = ", ".join(f"*source_{index}" for index in range(width))
    lines.extend(("root:", f"  <<: [{sources}]"))
    root = SourceDocument.parse("\n".join(lines) + "\n").root.child("root")
    definition = root._mapping_definition()
    sequence = definition.pair_sequence
    index_type = type(definition.effective_index)
    original_get = index_type.get
    lookups = 0

    def counted(index, key):
        nonlocal lookups
        lookups += 1
        return original_get(index, key)

    monkeypatch.setattr(index_type, "get", counted)

    for index in range(width):
        assert root.child(f"key_{index}").value == index

    assert len(definition.effective_index) == width
    assert lookups <= width * 2


def test_sequence_child_and_children_follow_indices():
    sequence = SourceDocument.parse(
        "items:\n"
        "  - first\n"
        "  - name: second\n"
    ).root.child("items")

    assert sequence.child(0).value == "first"
    assert sequence.child(0).range == SourceRange(2, 5, 2, 10)
    assert sequence.child(0).key_range is None
    assert [child.value for child in sequence.children()] == [
        "first",
        {"name": "second"},
    ]
    for invalid in (True, -1, 2, "0"):
        with pytest.raises(SourceAccessError):
            sequence.child(invalid)


def test_duplicate_lookup_is_last_but_occurrences_keep_both():
    root = SourceDocument.parse(
        "name: first\n"
        "other: middle\n"
        "name: final\n"
    ).root

    occurrences = root.mapping_occurrences("name")

    assert root.child("name").value == "final"
    assert [cursor.value for cursor in occurrences] == ["first", "final"]
    assert [cursor.range.start_line for cursor in occurrences] == [1, 3]
    assert [cursor.path[-1].pair_ordinal for cursor in occurrences] == [0, 2]
    assert [cursor.value for cursor in root.mapping_occurrences()] == [
        "first",
        "middle",
        "final",
    ]


def test_merge_local_value_wins_but_inherited_occurrence_remains():
    root = SourceDocument.parse(
        "base: &base\n"
        "  value: inherited\n"
        "target:\n"
        "  <<: *base\n"
        "  value: local\n"
    ).root.child("target")

    occurrences = root.mapping_occurrences("value")

    assert root.child("value").value == "local"
    assert [cursor.value for cursor in occurrences] == ["inherited", "local"]
    assert [cursor.range.start_line for cursor in occurrences] == [2, 5]


def test_null_key_filter_is_distinct_from_no_filter():
    root = SourceDocument.parse(
        "~: null-key\n"
        "named: named-value\n"
    ).root

    assert [cursor.value for cursor in root.mapping_occurrences(None)] == [
        "null-key"
    ]
    assert [cursor.value for cursor in root.mapping_occurrences()] == [
        "null-key",
        "named-value",
    ]


def test_occurrence_paths_distinguish_constructed_key_types():
    root = SourceDocument.parse(
        "bool_map: {on: bool-value}\n"
        "int_map: {1: int-value}\n"
        'str_map: {"1": str-value}\n'
        "null_map: {~: null-value}\n"
    ).root

    steps = [
        root.child(name).mapping_occurrences()[0].path[-1]
        for name in ("bool_map", "int_map", "str_map", "null_map")
    ]

    assert [step.pair_ordinal for step in steps] == [0, 0, 0, 0]
    assert len(set(steps)) == 4


def test_alias_occurrences_have_distinct_paths_and_anchor_ranges():
    document = SourceDocument.parse(
        "shared: &shared {value: 1}\n"
        "left: *shared\n"
        "right: *shared\n"
    )

    left = document.root.child("left")
    right = document.root.child("right")
    saved_path = left.path
    saved_range = left.range

    assert left.path != right.path
    assert left.range == right.range == SourceRange(1, 9, 1, 27)
    assert left.key_range == SourceRange(2, 1, 2, 5)
    assert right.key_range == SourceRange(3, 1, 3, 6)

    del left
    replayed = document._resolve_occurrence(saved_path)
    assert replayed.path == saved_path
    assert replayed.range == saved_range


def test_cursors_use_identity_equality_and_hashing():
    document = SourceDocument.parse(
        "shared: &shared {value: 1}\n"
        "left: *shared\n"
        "right: *shared\n"
    )
    left = document.root.child("left")
    right = document.root.child("right")
    virtual = SourceCursor.virtual(left.value)

    assert left is not right
    assert left != right
    assert left.path != right.path
    assert left != virtual
    assert len({left, right, virtual}) == 3
    assert len({left: "left", right: "right", virtual: "virtual"}) == 3


def test_projected_entries_use_identity_equality_and_hashing():
    projected = SourceCursor.virtual({}).project_mapping(
        {"first": (1, None), "second": (2, None)}
    )
    first, second = projected._projected_entries

    assert first is not second
    assert first != second
    assert len({first, second}) == 2


def test_recursive_alias_path_stops_at_ancestor():
    root = SourceDocument.parse(
        "root: &root\n"
        "  name: recursive\n"
        "  self: *root\n"
    ).root.child("root")

    recursive = root.child("self")
    copied = recursive.value

    assert copied["self"] is copied
    assert recursive.path[-1].constructed_key == "self"
    assert recursive.range == root.range
    assert recursive.key_range == SourceRange(3, 3, 3, 7)
    with pytest.raises(SourceAccessError, match="recursive alias") as caught:
        recursive.child("name")
    assert caught.value.range == recursive.range


def test_typed_accessors():
    root = SourceDocument.parse(
        "name: compose\n"
        "count: 3\n"
        "config:\n"
        "  enabled: true\n"
        "items: [one, two]\n"
    ).root

    assert root.has("name")
    assert not root.has("missing")
    assert root.get("name") == "compose"
    assert root.get("missing", {"fallback": []}) == {"fallback": []}
    assert root.str("name") == "compose"
    assert root.int("count") == 3
    assert root.mapping("config") == {"enabled": True}
    assert root.sequence("items") == ["one", "two"]
    assert root.range_of("name") == SourceRange(1, 7, 1, 14)
    assert root.key_range_of("name") == SourceRange(1, 1, 1, 5)
    assert root.key_range_of("missing") is None

    mapping = root.mapping("config")
    sequence = root.sequence("items")
    mapping["enabled"] = False
    sequence.append("three")
    assert root.mapping("config") == {"enabled": True}
    assert root.sequence("items") == ["one", "two"]

    with pytest.raises(SourceAccessError, match="missing required field"):
        root.str("missing")
    with pytest.raises(SourceAccessError, match="must be a string") as caught:
        root.str("count")
    assert caught.value.range == root.range_of("count")
    with pytest.raises(SourceAccessError, match="must be a mapping"):
        root.mapping("items")
    with pytest.raises(SourceAccessError, match="must be a sequence"):
        root.sequence("config")


def test_int_rejects_bool():
    root = SourceDocument.parse("flag: true\n").root

    with pytest.raises(SourceAccessError, match="must be an integer") as caught:
        root.int("flag")

    assert caught.value.range == SourceRange(1, 7, 1, 11)


def test_item_range_list_keeps_duplicate_positions():
    root = SourceDocument.parse(
        "asserts:\n"
        "  - ${output} > 0\n"
        "  - ${output} > 0\n"
    ).root

    ranges = root.item_range_list("asserts")

    assert ranges == [
        SourceRange(2, 5, 2, 18),
        SourceRange(3, 5, 3, 18),
    ]
    assert root.item_ranges("asserts") == {
        "${output} > 0": SourceRange(3, 5, 3, 18),
    }


def test_sequence_field_ranges_follow_item_order():
    root = SourceDocument.parse(
        "cases:\n"
        '  - when: "a >= 1"\n'
        "    then: x\n"
        '  - when: "b >= 2"\n'
        "    then: y\n"
        "  - scalar\n"
    ).root

    assert root.sequence_field_ranges("cases", "when") == [
        SourceRange(2, 11, 2, 19),
        SourceRange(4, 11, 4, 19),
        None,
    ]
    assert root.sequence_field_ranges("cases", "then") == [
        SourceRange(3, 11, 3, 12),
        SourceRange(5, 11, 5, 12),
        None,
    ]
    assert root.mapping_value_ranges("missing") == {}


def test_virtual_value_is_snapshotted_and_copy_on_read():
    recursive: dict[str, object] = {"items": [1]}
    recursive["self"] = recursive
    original = {"root": recursive}

    cursor = SourceCursor.virtual(original)
    original["root"]["items"].append(2)
    first = cursor.value
    first["root"]["items"].append(3)
    second = cursor.value

    assert second["root"]["items"] == [1]
    assert second["root"]["self"] is second["root"]
    assert cursor.document is None
    assert cursor.path == ()
    assert cursor.range is None
    assert cursor.key_range is None


def test_renamed_cursor_does_not_mutate_base():
    base = SourceDocument.parse(
        "input: source\n"
        "keep: unchanged\n"
    ).root

    renamed = base.renamed({"input": "inputs"})

    assert renamed.value == {"inputs": "source", "keep": "unchanged"}
    assert base.value == {"input": "source", "keep": "unchanged"}
    assert not base.has("inputs")
    assert renamed.child("inputs").path == base.child("input").path
    assert renamed.child("inputs").range == base.child("input").range
    assert renamed.child("inputs").key_range == base.child("input").key_range


def test_renamed_range_helpers_delegate_to_child_sources():
    base = SourceDocument.parse(
        "input:\n"
        "  value: ${start.output}\n"
        "asserts:\n"
        "  - ${output} > 0\n"
        "cases:\n"
        "  - when: ready\n"
        "    then: finish\n"
    ).root

    renamed = base.renamed({"input": "inputs"})

    assert renamed.mapping_value_ranges("inputs") == base.mapping_value_ranges(
        "input"
    )
    assert renamed.item_ranges("asserts") == base.item_ranges("asserts")
    assert renamed.item_range_list("asserts") == base.item_range_list("asserts")
    assert renamed.sequence_field_ranges(
        "cases", "when"
    ) == base.sequence_field_ranges("cases", "when")
    assert renamed.sequence_field_ranges(
        "cases", "then"
    ) == base.sequence_field_ranges("cases", "then")


def test_with_value_aligns_nested_mapping_and_sequence_sources():
    source = SourceDocument.parse(
        "body:\n"
        "  mapping:\n"
        "    field: old\n"
        "  sequence:\n"
        "    - field: first\n"
    ).root.child("body")

    aligned = source.with_value(
        {
            "mapping": {"field": "new", "generated": True},
            "sequence": [
                {"field": "changed"},
                {"field": "generated"},
            ],
        }
    )

    assert aligned.child("mapping").child("field").range == source.child(
        "mapping"
    ).child("field").range
    assert aligned.child("mapping").child("generated").range is None
    assert aligned.child("sequence").child(0).child("field").range == source.child(
        "sequence"
    ).child(0).child("field").range
    assert aligned.child("sequence").child(1).document is None
    assert aligned.child("sequence").child(1).child("field").range is None


def test_projected_child_retains_source_path_and_range():
    base = SourceDocument.parse(
        "original:\n"
        "  nested: old\n"
    ).root
    source = base.child("original")

    projected = base.project_mapping(
        {
            "renamed": ({"nested": ["new"]}, source),
            "generated": ({"virtual": True}, None),
        }
    )
    retained = projected.child("renamed")
    generated = projected.child("generated")

    assert projected.document is base.document
    assert projected.path == base.path
    assert projected.range == base.range
    assert retained.value == {"nested": ["new"]}
    assert retained.document is source.document
    assert retained.path == source.path
    assert retained.range == source.range
    assert retained.key_range == source.key_range
    assert generated.document is None
    assert generated.path == ()
    assert generated.range is None
    assert generated.key_range is None


def test_projected_mutation_cannot_change_overlay():
    base = SourceDocument.parse("source: original\n").root
    original = {"nested": [1]}
    projected = base.project_mapping(
        {"source": (original, base.child("source"))}
    )

    original["nested"].append(2)
    first = projected.value
    first["source"]["nested"].append(3)
    child_value = projected.child("source").value
    child_value["nested"].append(4)

    assert projected.value == {"source": {"nested": [1]}}
    assert base.value == {"source": "original"}
