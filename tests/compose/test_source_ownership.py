"""Source-document ownership tests for the Compose loading boundary."""

import pytest

from agent_composer.compose import loader as compose_loader
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers import types as compose_types
from agent_composer.compose.helpers import spanloader as legacy_spanloader
from agent_composer.compose.parse.file import (
    parse_file,
    parse_source_document,
)
from agent_composer.compose.source import SourceDocument, parse_yaml_literal
from agent_composer.compose.source import document as source_document


_FLOW = """\
id: owned
name: owned
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      value: hello
    output: str
output: ${echo.output}
"""

_INLINE_FLOW = """\
id: owned-inline
name: owned_inline
nodes:
  wrapper:
    kind: call
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
    output: ${echo.output}
output: ${wrapper.output}
"""

_DEFS_FLOW = """\
id: owned-defs
name: owned_defs
defs:
  echo_child:
    input:
      value: str
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          value: ${input.value}
        output: str
    output: ${echo.output}
nodes:
  invoke:
    kind: call
    call: echo_child
    input:
      value: hello
output: ${invoke.output}
"""


def test_parse_file_reuses_supplied_document(monkeypatch):
    document = SourceDocument.parse(_FLOW)
    loader_calls = 0

    def count_loader(*args, **kwargs):
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("parse_file must reuse a supplied source document")

    monkeypatch.setattr(
        "agent_composer.compose.source.document._DocumentLoader.__init__",
        count_loader,
    )

    parsed = parse_file(document)

    assert loader_calls == 0
    assert parsed.id == "owned"
    assert parsed.nodes["echo"]["kind"] == "code"


def test_parse_source_document_translates_syntax_error():
    with pytest.raises(LoadError, match="invalid YAML") as caught:
        parse_source_document("id: [", "broken.yaml")

    assert caught.value.range is not None
    assert caught.value.range.start_line == 1


@pytest.mark.parametrize(
    "source",
    [_FLOW, _INLINE_FLOW, _DEFS_FLOW],
    ids=["plain", "inline", "defs"],
)
def test_root_variants_use_one_document_and_no_legacy_loader(
    monkeypatch,
    source,
):
    document_calls = 0
    legacy_calls = 0
    original_init = source_document._DocumentLoader.__init__
    original_legacy_init = legacy_spanloader.SpanLoader.__init__

    def count_loader(self, *args, **kwargs):
        nonlocal document_calls
        document_calls += 1
        original_init(self, *args, **kwargs)

    def count_legacy_loader(self, *args, **kwargs):
        nonlocal legacy_calls
        legacy_calls += 1
        original_legacy_init(self, *args, **kwargs)

    monkeypatch.setattr(
        source_document._DocumentLoader,
        "__init__",
        count_loader,
    )
    monkeypatch.setattr(
        legacy_spanloader.SpanLoader,
        "__init__",
        count_legacy_loader,
    )

    compose_loader.load_flow(source)

    assert document_calls == 1
    assert legacy_calls == 0


def test_root_load_does_not_mutate_captured_document(monkeypatch):
    captured = []
    original_parse = compose_loader.parse_source_document

    def capture(text, label):
        document = original_parse(text, label)
        captured.append((document, document.root.value))
        return document

    monkeypatch.setattr(compose_loader, "parse_source_document", capture)

    compose_loader.load_flow(_INLINE_FLOW)

    assert len(captured) == 1
    document, original_value = captured[0]
    assert document.root.value == original_value


def test_load_flow_malformed_root_is_located_load_error():
    with pytest.raises(LoadError, match="invalid YAML") as caught:
        compose_loader.load_flow("id: [")

    assert caught.value.range is not None
    assert caught.value.range.start_line == 1


def test_recursive_boundary_alias_does_not_escape_as_source_access_error():
    source = """&root
id: recursive-boundary
name: recursive_boundary
input: *root
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: {value: hello}
    output: str
output: ${echo.output}
"""

    with pytest.raises(LoadError):
        compose_loader.load_flow(source)


def test_input_default_uses_literal_parser_not_document_loader(monkeypatch):
    literal_calls = []

    def track_literal(text):
        literal_calls.append(text)
        return parse_yaml_literal(text)

    def reject_document_loader(*args, **kwargs):
        raise AssertionError("input defaults must not create a source document")

    monkeypatch.setattr(compose_types, "parse_yaml_literal", track_literal)
    monkeypatch.setattr(
        source_document._DocumentLoader,
        "__init__",
        reject_document_loader,
    )

    decl = compose_types.read_flow_inputs(
        {"items": 'list[str] = ["one", "two"]'},
        {},
    )[0]

    assert literal_calls == ['["one", "two"]']
    assert decl.default == ["one", "two"]
