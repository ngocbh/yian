"""Tests for one-pass, privately retained YAML source documents."""

from dataclasses import FrozenInstanceError

import pytest

from agent_composer.compose.source import document as document_module
from agent_composer.compose.source import SourceDocument, SourceSyntaxError
from agent_composer.events import SourceRange


def test_document_has_private_retained_slots():
    text = "count: 30\nflag: yes\n"
    document = SourceDocument.parse(text, label="flow.yaml")

    assert document._value == {"count": 30, "flag": True}
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
