"""Tests for one-pass, privately retained YAML source documents."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pytest
import yaml

from yian.compose.source import document as document_module
from yian.compose.source import SourceDocument, SourceSyntaxError
from yian.compose.source.scalars import _FlowLoader
from yian.diagnostics import SourceRange


def test_document_has_private_retained_slots():
    text = "count: 30\nflag: yes\n"
    document = SourceDocument.parse(text, label="flow.yaml")

    assert document._value == {"count": 30, "flag": "yes"}
    assert document._root_node is not None
    assert document._definitions
    assert document._root_definition == 0
    with pytest.raises(FrozenInstanceError):
        document._definitions[0].range = None
    assert document.text == text
    assert document.label == "flow.yaml"
    assert not hasattr(document, "value")
    assert not hasattr(document, "root_node")
    assert not hasattr(document, "definitions")
    assert not hasattr(document, "root_definition")
    assert not hasattr(document, "index")
    assert not hasattr(document, "_index")
    assert not hasattr(document, "__dict__")
    with pytest.raises(FrozenInstanceError):
        document.label = "other.yaml"


def test_document_uses_one_loader_instance(monkeypatch):
    calls = {
        "init": 0,
        "get_single_node": 0,
        "construct_document": 0,
        "dispose": 0,
    }
    original_init = document_module._DocumentLoader.__init__
    original_get_single_node = document_module._DocumentLoader.get_single_node
    original_construct_document = document_module._DocumentLoader.construct_document
    original_dispose = document_module._DocumentLoader.dispose

    def counting_init(loader, stream):
        calls["init"] += 1
        original_init(loader, stream)

    def counting_get_single_node(loader):
        calls["get_single_node"] += 1
        return original_get_single_node(loader)

    def counting_construct_document(loader, node):
        calls["construct_document"] += 1
        return original_construct_document(loader, node)

    def counting_dispose(loader):
        calls["dispose"] += 1
        assert loader._source_values == {}
        assert loader._effective_mappings == {}
        assert loader._mapping_states == {}
        assert loader._sequence_values == {}
        assert loader._forced_string_nodes == set()
        return original_dispose(loader)

    monkeypatch.setattr(document_module._DocumentLoader, "__init__", counting_init)
    monkeypatch.setattr(
        document_module._DocumentLoader,
        "get_single_node",
        counting_get_single_node,
    )
    monkeypatch.setattr(
        document_module._DocumentLoader,
        "construct_document",
        counting_construct_document,
    )
    monkeypatch.setattr(document_module._DocumentLoader, "dispose", counting_dispose)

    SourceDocument.parse("count: 30\n")

    assert calls == {
        "init": 1,
        "get_single_node": 1,
        "construct_document": 1,
        "dispose": 1,
    }


def test_document_raises_source_syntax_error_with_problem_range():
    with pytest.raises(SourceSyntaxError) as caught:
        SourceDocument.parse("nodes:\n  broken: ]\n")

    assert caught.value.range == SourceRange(2, 11, 2, 12)


@pytest.mark.parametrize(
    "line_break",
    ["\n", "\r", "\r\n", "\x85", "\u2028", "\u2029"],
    ids=["lf", "cr", "crlf", "nel", "ls", "ps"],
)
def test_document_translates_reader_error_with_problem_range(line_break):
    with pytest.raises(SourceSyntaxError) as caught:
        SourceDocument.parse(f"first: ok{line_break}bad:\x00 value\n")

    assert caught.value.range == SourceRange(2, 5, 2, 6)


def test_document_rejects_multiple_documents():
    with pytest.raises(SourceSyntaxError, match="another document") as caught:
        SourceDocument.parse("first: 1\n---\nsecond: 2\n")

    assert caught.value.range == SourceRange(2, 1, 2, 2)


def test_document_accepts_an_empty_document():
    document = SourceDocument.parse("")

    assert document._value is None
    assert document._root_node is None
    assert document._root_definition == 0
    assert len(document._definitions) == 1


@pytest.mark.parametrize(
    "text",
    [
        """\
first: &first {shared: first, first_only: 1}
second: &second {shared: second, second_only: 2}
target:
  <<: [*first, *second]
  local: 3
""",
        """\
first: &first {shared: first, first_only: 1}
second: &second {shared: second, second_only: 2}
target:
  <<: *first
  <<: *second
  shared: local
""",
        """\
base: &base
  true: boolean
  when: 2001-12-15T02:59:43.1Z
  blob: !!binary YQ==
target:
  <<: *base
  1: integer-wins
  !!value =: explicit-value-key
""",
        """\
target:
  <<: &inline
    values: !!set {one: null, two: null}
    ordered: !!omap [{first: 1}, {second: 2}]
    pairs: !!pairs [{same: 1}, {same: 2}]
""",
    ],
    ids=["sequence", "multiple-merge-keys", "keys", "inline-anchor"],
)
def test_merge_construction_matches_safe_loader_without_mutating_nodes(text):
    expected = yaml.load(text, Loader=_FlowLoader)
    document = SourceDocument.parse(text)

    assert document.root.value == expected


def test_inline_merge_source_keeps_recursive_value_alias_identity():
    document = SourceDocument.parse(
        "target:\n"
        "  <<: &base\n"
        "    self: *base\n"
    )

    target = document.root.value["target"]

    assert target["self"]["self"] is target["self"]
    assert target["self"] is not target


def test_recursive_self_merge_matches_safe_loader():
    text = (
        "root: &root\n"
        "  <<: *root\n"
        "  value: local\n"
    )

    document = SourceDocument.parse(text)

    assert document.root.value == yaml.load(text, Loader=_FlowLoader)
    assert [item.value for item in document.root.child("root").mapping_occurrences()] == [
        "local",
        "local",
    ]


def test_value_tag_key_alias_is_constructed_as_a_string_everywhere():
    text = (
        "first: &equals !!value =\n"
        "? *equals\n"
        ": aliased-key\n"
    )

    assert SourceDocument.parse(text).root.value == yaml.load(text, Loader=_FlowLoader)


def test_value_tag_alias_in_a_later_nested_mapping_keeps_safe_loader_error():
    text = (
        "first: &equals !!value =\n"
        "nested: {*equals: aliased-key}\n"
    )

    with pytest.raises(SourceSyntaxError, match="could not determine a constructor"):
        SourceDocument.parse(text)


def test_local_tag_on_a_merge_source_is_an_error():
    text = "target: {<<: !mystery {value: inherited}}\n"

    with pytest.raises(SourceSyntaxError, match="checks nothing") as info:
        SourceDocument.parse(text)
    assert info.value.range == SourceRange(1, 14, 1, 41)


def test_merge_source_can_force_a_value_tag_alias():
    text = (
        "target:\n"
        "  first: &equals !!value =\n"
        "  <<: {*equals: inherited}\n"
    )

    assert SourceDocument.parse(text).root.value == yaml.load(text, Loader=_FlowLoader)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "root:\n  <<:\n    - {<<: 7}\n    - 8\n",
            "expected a mapping or list of mappings for merging",
        ),
        (
            "root:\n  <<:\n    - 8\n    - {<<: 7}\n",
            "expected a mapping for merging",
        ),
    ],
    ids=["nested-first", "outer-first"],
)
def test_merge_sequence_reports_the_first_authored_source_error(text, message):
    with pytest.raises(SourceSyntaxError, match=message):
        SourceDocument.parse(text)


def _non_text_key(text):
    """Return the parsed document's first non-text key."""

    return SourceDocument.parse(text).non_text_key


def test_non_text_key_is_the_constructed_key_its_text_and_range():
    found = _non_text_key("a: 1\nTrue: x\n")

    assert found == (True, "True", SourceRange(2, 1, 2, 5))
    assert type(found[0]) is bool


@pytest.mark.parametrize(
    ("raw", "key", "text"),
    [
        ("true", True, "true"),
        ("FALSE", False, "FALSE"),
        ("1", 1, "1"),
        ("-3", -3, "-3"),
        ("0x1F", 31, "0x1F"),
        ("1.5", 1.5, "1.5"),
        ("1e3", 1000.0, "1e3"),
        ("null", None, "null"),
        ("~", None, "~"),
        ("2026-10-13", date(2026, 10, 13), "2026-10-13"),
        ("2026-10-13 10:00:00", datetime(2026, 10, 13, 10, 0), "2026-10-13 10:00:00"),
        ("!!binary aGVsbG8=", b"hello", "aGVsbG8="),
        ('!!int "7"', 7, "7"),
    ],
)
def test_non_text_key_reports_each_kind_of_scalar(raw, key, text):
    found = _non_text_key(f"{raw}: x\n")

    assert found is not None
    assert found[0] == key and type(found[0]) is type(key)
    assert found[1] == text
    assert (found[2].start_line, found[2].start_col) == (1, 1)


@pytest.mark.parametrize(
    "raw",
    ["yes", "on", "off", "no", "1:30", "1_000", '"True"', "'1'", "!!str 1", "=", "!!value 1"],
)
def test_text_keys_are_not_reported(raw):
    assert _non_text_key(f"{raw}: x\n") is None


@pytest.mark.parametrize(
    ("text", "key", "key_range"),
    [
        ("a:\n  - {1: x}\n", 1, SourceRange(2, 6, 2, 7)),
        ("a: {b: {2.5: x}}\n", 2.5, SourceRange(1, 9, 1, 12)),
        ("a:\n  b:\n    c:\n      null: x\n", None, SourceRange(4, 7, 4, 11)),
        ("a: {2: x}\n3: y\n", 2, SourceRange(1, 5, 1, 6)),
        ("1: x\nb: {2: y}\n", 1, SourceRange(1, 1, 1, 2)),
        ("base: &b {1: x}\nc:\n  <<: *b\n", 1, SourceRange(1, 11, 1, 12)),
        ("&k 1: a\nb: {*k : c}\n", 1, SourceRange(1, 1, 1, 5)),
        ("a: &k 1\n*k : b\n", 1, SourceRange(1, 4, 1, 8)),
    ],
    ids=[
        "sequence-item",
        "flow-mapping-value",
        "deep-value",
        "earlier-value-beats-later-key",
        "earlier-key-beats-later-value",
        "merged-base-at-its-written-key",
        "alias-of-a-key",
        "alias-of-a-value",
    ],
)
def test_non_text_key_is_found_anywhere_in_document_order(text, key, key_range):
    found = _non_text_key(text)

    assert found is not None
    assert found[0] == key and type(found[0]) is type(key)
    assert found[2] == key_range


def test_merge_key_itself_is_not_reported():
    assert _non_text_key("base: &b {a: x}\nc:\n  <<: *b\n  d: y\n") is None


@pytest.mark.parametrize(
    ("text", "key", "key_range"),
    [
        ('!!binary "": x\n', b"", SourceRange(1, 1, 1, 12)),
        ("? \n: x\n", None, SourceRange(1, 2, 1, 2)),
        ("{? : x}\n", None, SourceRange(1, 3, 1, 3)),
    ],
    ids=["tagged-empty", "explicit-empty", "flow-empty"],
)
def test_empty_key_is_reported_with_empty_text(text, key, key_range):
    found = _non_text_key(text)

    assert found is not None
    assert found[0] == key and type(found[0]) is type(key)
    assert found[1] == ""
    assert found[2] == key_range


def test_a_key_repeated_through_an_alias_is_located_where_each_key_is_written():
    assert SourceDocument.parse("x: &k a\ny: {*k : 1, *k : 2}").repeated_key == (
        "a",
        SourceRange(2, 5, 2, 7),
        SourceRange(2, 13, 2, 15),
    )
    assert SourceDocument.parse("&k a: 1\n*k : 2\n").repeated_key == (
        "a",
        SourceRange(1, 1, 1, 5),
        SourceRange(2, 1, 2, 3),
    )
    # Aliases written as values before the repeat don't shift its range.
    assert SourceDocument.parse(
        "b: &v 1\n&k a: *v\nc: [*v, *k]\n*k : 3\n"
    ).repeated_key == ("a", SourceRange(2, 1, 2, 5), SourceRange(4, 1, 4, 3))


def test_non_text_key_and_repeated_key_are_both_found():
    document = SourceDocument.parse("017: a\n17: b\n")

    assert document.non_text_key == (17, "017", SourceRange(1, 1, 1, 4))
    assert document.repeated_key is not None
