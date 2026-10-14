"""Tests for one-pass, privately retained YAML source documents."""

from dataclasses import FrozenInstanceError

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


def test_syntactic_only_custom_tagged_merge_source_is_not_constructed():
    text = "target: {<<: !mystery {value: inherited}}\n"

    assert SourceDocument.parse(text).root.value == yaml.load(text, Loader=_FlowLoader)


def test_custom_tagged_merge_source_can_force_a_value_tag_alias():
    text = (
        "target:\n"
        "  first: &equals !!value =\n"
        "  <<: !mystery {*equals: inherited}\n"
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
