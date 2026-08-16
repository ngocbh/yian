"""Source-document ownership tests for the Compose loading boundary."""

import ast
from pathlib import Path

import pytest

from agent_composer.compose import loader as compose_loader
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers import types as compose_types
from agent_composer.compose.parse.file import (
    parse_file,
    parse_source_document,
)
from agent_composer.compose.source import SourceDocument, parse_yaml_literal
from agent_composer.compose.source import document as source_document


_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "agent_composer"
_COMPOSE_ROOT = _PACKAGE_ROOT / "compose"
_SOURCE_ROOT = _COMPOSE_ROOT / "source"


def _production_modules(root=_COMPOSE_ROOT):
    """Yield `(relative path, parsed AST)` for each production module below `root`."""

    for path in sorted(root.rglob("*.py")):
        yield path.relative_to(root), ast.parse(path.read_text())


def test_only_source_package_imports_pyyaml():
    offenders = []
    for relative, tree in _production_modules(_PACKAGE_ROOT):
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            if any(name == "yaml" or name.startswith("yaml.") for name in imported):
                if relative.parts[:2] != ("compose", "source"):
                    offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


def test_legacy_source_modules_are_absent():
    assert not (_COMPOSE_ROOT / "helpers" / "reader.py").exists()
    assert not (_COMPOSE_ROOT / "helpers" / "spanloader.py").exists()


def test_source_package_has_no_upward_imports():
    offenders = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                line = node.lineno
            elif isinstance(node, ast.ImportFrom):
                if node.level >= 2:
                    offenders.append(
                        f"{path.relative_to(_COMPOSE_ROOT)}:{node.lineno}:relative-upward"
                    )
                    continue
                modules = [node.module] if node.module is not None else []
                line = node.lineno
            else:
                continue
            for module in modules:
                if (
                    module == "agent_composer.compose"
                    or module.startswith("agent_composer.compose.")
                ) and not module.startswith("agent_composer.compose.source"):
                    offenders.append(f"{path.relative_to(_COMPOSE_ROOT)}:{line}:{module}")
    assert offenders == []


def test_document_graph_slots_are_private_to_source_package():
    private_slots = {
        "_value",
        "_root_node",
        "_definitions",
        "_root_definition",
        "_index",
    }
    offenders = []
    for relative, tree in _production_modules():
        if relative.parts[0] == "source":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in private_slots:
                offenders.append(f"{relative}:{node.lineno}:{node.attr}")
    assert offenders == []


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

    assert caught.value.diagnostic.frames[-1].range.start_line == 1


@pytest.mark.parametrize(
    "source",
    [_FLOW, _INLINE_FLOW, _DEFS_FLOW],
    ids=["plain", "inline", "defs"],
)
def test_root_variants_use_one_document(
    monkeypatch,
    source,
):
    document_calls = 0
    original_init = source_document._DocumentLoader.__init__

    def count_loader(self, *args, **kwargs):
        nonlocal document_calls
        document_calls += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(
        source_document._DocumentLoader,
        "__init__",
        count_loader,
    )
    compose_loader.load_flow(source)

    assert document_calls == 1


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

    assert caught.value.diagnostic.frames[-1].range.start_line == 1


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
