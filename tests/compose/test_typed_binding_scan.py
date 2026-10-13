"""Bundled flows never hold unquoted ` = ` text in a binding position.

A plain (unquoted) YAML string containing ` = ` in a node `input:` or a flow
`output:` is read as a typed binding `Type = value`, so any such text that is
meant literally must be quoted. This test walks every bundled YAML file and
fails on the first plain binding value that would change meaning.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from yian.compose.source import SourceCursor, SourceDocument, SourceSyntaxError

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_GLOBS = (
    "tests/seeds/**/*.yaml",
    "examples/**/*.yaml",
    ".claude/skills/**/templates/**/*.yaml",
    "src/yian/**/*.yaml",
)


def _bundled_yaml_files() -> list[Path]:
    """Return every bundled YAML file, sorted and without duplicates."""

    found = {path for pattern in BUNDLED_GLOBS for path in REPO_ROOT.glob(pattern)}
    return sorted(found)


def _mapping_items(cursor: SourceCursor, key: str) -> Iterator[tuple[object, SourceCursor]]:
    """Yield `(key, child)` pairs of the mapping at `cursor[key]`, if any."""

    if not isinstance(cursor.value, dict) or not cursor.has(key):
        return
    child = cursor.child(key)
    if isinstance(child.value, dict):
        yield from zip(child.value, child.children())


def _is_flow_body(cursor: SourceCursor) -> bool:
    """Return whether `cursor` is a mapping with a `nodes:` graph."""

    return isinstance(cursor.value, dict) and cursor.has("nodes")


def _binding_values(flow: SourceCursor) -> Iterator[tuple[str, SourceCursor]]:
    """Yield `(where, value)` for each binding position of a flow body.

    Covers each node's `input`/`inputs` values, the flow's `output`/`outputs`
    (a bare value or each field), every `defs:` entry with a `nodes:` graph,
    and every node with an inline `nodes:` body.
    """

    for node_id, node in _mapping_items(flow, "nodes"):
        for key in ("input", "inputs"):
            for name, value in _mapping_items(node, key):
                yield f"nodes.{node_id}.{key}.{name}", value
        if _is_flow_body(node):
            for where, value in _binding_values(node):
                yield f"nodes.{node_id}.{where}", value
    for key in ("output", "outputs"):
        if not flow.has(key):
            continue
        output = flow.child(key)
        if isinstance(output.value, dict):
            for name, value in _mapping_items(flow, key):
                yield f"{key}.{name}", value
        else:
            yield key, output
    # A single-node def (a `kind:` without `nodes:`) declares parameters in
    # `input:` and a type in `output:`, so only graph defs hold bindings.
    for def_id, definition in _mapping_items(flow, "defs"):
        if _is_flow_body(definition):
            for where, value in _binding_values(definition):
                yield f"defs.{def_id}.{where}", value


def _looks_typed(value: SourceCursor) -> bool:
    """Return whether `value` is plain string text that reads as `Type = value`."""

    text = value.value
    return (
        isinstance(text, str)
        and value.scalar_style in ("plain", None)
        and (" = " in text or text.endswith(" ="))
    )


def _unquoted_separators(text: str, label: str) -> list[str]:
    """Return `label:line: where: value` for each plain binding with ` = `.

    Compact flows (a root `kind:` without `nodes:`) hold parameter
    declarations and a type rather than bindings, so they are skipped.
    """

    root = SourceDocument.parse(text, label).root
    if not isinstance(root.value, dict):
        return []
    if root.has("kind") and not root.has("nodes"):
        return []
    flagged = []
    for where, value in _binding_values(root):
        if _looks_typed(value):
            line = value.range.start_line if value.range is not None else "?"
            flagged.append(f"{label}:{line}: {where}: {value.value!r}")
    return flagged


def test_walker_flags_plain_separators_only() -> None:
    text = (
        "nodes:\n"
        "  ask:\n"
        "    kind: agent\n"
        "    input:\n"
        "      q: a = b\n"
        "      quoted: \"a = b\"\n"
        "      block: |\n"
        "        a = b\n"
        "    output: str\n"
        "output: Size = ${x}\n"
    )

    assert _unquoted_separators(text, "inline.yaml") == [
        "inline.yaml:5: nodes.ask.input.q: 'a = b'",
        "inline.yaml:10: output: 'Size = ${x}'",
    ]


def test_walker_reaches_defs_and_inline_bodies() -> None:
    text = (
        "defs:\n"
        "  body:\n"
        "    nodes:\n"
        "      step:\n"
        "        kind: call\n"
        "        input: {v: x =}\n"
        "        nodes: {}\n"
        "        output: {r: a = b}\n"
        "    output: ${step.output}\n"
        "  single:\n"
        "    kind: code\n"
        "    input: {v: int = 0}\n"
        "    output: str\n"
    )

    assert _unquoted_separators(text, "f.yaml") == [
        "f.yaml:6: defs.body.nodes.step.input.v: 'x ='",
        "f.yaml:8: defs.body.nodes.step.output.r: 'a = b'",
    ]


def test_bundled_flows_quote_separator_text_in_bindings() -> None:
    files = _bundled_yaml_files()
    assert files, f"no bundled YAML found under {REPO_ROOT}"

    flagged: list[str] = []
    unparsed: list[str] = []
    for path in files:
        label = str(path.relative_to(REPO_ROOT))
        try:
            flagged += _unquoted_separators(path.read_text(encoding="utf-8"), label)
        except SourceSyntaxError:
            unparsed.append(label)

    assert not unparsed, "bundled YAML failed to parse:\n" + "\n".join(unparsed)
    assert not flagged, (
        "plain binding values containing ' = ' now load as typed bindings; "
        "quote them to keep them as text:\n" + "\n".join(flagged)
    )
