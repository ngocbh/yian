"""Kind-dispatch census guarding the kind-agnostic engine core.

The engine core (`runtime/engine.py` + `runtime/eval_node.py`) knows only an abstract node
contract and a closed `Outcome` sum — it must NOT branch on a node's `NodeKind` nor on a
concrete `*Expansion` type. This test counts kind-dispatch sites and asserts the count never
rises above the configured ceiling.

The count is derived from the source rather than a hand-maintained line list, so it survives
line-number drift. The zero ceiling makes any kind-specific dispatch a test failure.

A "kind-dispatch site" is a distinct source line that does any of:
  - reference a `NodeKind` member  (`NodeKind.AGENT`, `NodeKind.CASE`, ...)   -> branches on kind
  - reference `_SPAWNER_KINDS`      (the spawner-kind set used for membership) -> branches on kind
  - call `isinstance(x, <...>Expansion)`                                       -> branches on concrete type

Import lines are excluded (importing `NodeKind` / `_SPAWNER_KINDS` is not dispatch), and a
diagnostic read like `node.kind.value` (for an error message) is not counted — it touches the
instance attribute, not a `NodeKind` member or an `*Expansion` type.

Known blind spots (do NOT route dispatch through these to dodge the ratchet; these shapes also
violate the kind-agnostic engine contract):
indirect enum access (`getattr(NodeKind, name)`, `NodeKind[name]`, `NodeKind(value)`), a
`match node.kind` / `case NodeKind.X` block, kind-keyed dispatch dicts built off-file, and
type-name string compares (`type(desc).__name__ == "CallExpansion"`). The `*Expansion` detection
is a name-suffix heuristic tied to the codebase convention that every expansion type ends in
`Expansion`; keep that convention so the census stays accurate.
"""

import ast
from pathlib import Path

from yian.runtime import engine as engine_mod
from yian.runtime import eval_node as eval_node_mod

# The two modules that make up the engine core the census guards.
CORE_MODULES = {
    "runtime/engine.py": engine_mod,
    "runtime/eval_node.py": eval_node_mod,
}

# The engine core dispatches only on node traits/hooks and the closed Outcome sum, never on
# `NodeKind`, `_SPAWNER_KINDS`, or a concrete `*Expansion` type.
KIND_DISPATCH_CEILING = 0


def _import_lines(tree: ast.Module) -> set[int]:
    """Line numbers spanned by `import` / `from ... import` statements (excluded from the census)."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = node.end_lineno or node.lineno
            lines.update(range(node.lineno, end + 1))
    return lines


def _is_expansion_isinstance(node: ast.AST) -> bool:
    """True for `isinstance(x, <Name ending in 'Expansion'>)` — dispatch on a concrete expansion type."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
        return False
    if len(node.args) < 2:
        return False
    cls = node.args[1]
    names = cls.elts if isinstance(cls, ast.Tuple) else [cls]
    return any(isinstance(n, ast.Name) and n.id.endswith("Expansion") for n in names)


def _dispatch_lines(source: str) -> set[int]:
    """Distinct source line numbers carrying a kind-dispatch construct, imports excluded."""
    tree = ast.parse(source)
    skip = _import_lines(tree)
    hits: set[int] = set()
    for node in ast.walk(tree):
        # NodeKind.<member> — an attribute access on the NodeKind enum.
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "NodeKind":
            hits.add(node.lineno)
        # _SPAWNER_KINDS — the spawner-kind set, used only for kind membership tests.
        elif isinstance(node, ast.Name) and node.id == "_SPAWNER_KINDS":
            hits.add(node.lineno)
        # isinstance(x, *Expansion) — dispatch on a concrete expansion type.
        elif _is_expansion_isinstance(node):
            hits.add(node.lineno)
    return hits - skip


def _census() -> dict[str, set[int]]:
    """Map each core module's relative path -> the set of its kind-dispatch line numbers."""
    return {
        rel: _dispatch_lines(Path(mod.__file__).read_text())
        for rel, mod in CORE_MODULES.items()
    }


def test_kind_dispatch_stays_at_or_below_ceiling():
    census = _census()
    total = sum(len(lines) for lines in census.values())
    breakdown = {rel: sorted(lines) for rel, lines in census.items()}
    assert total <= KIND_DISPATCH_CEILING, (
        f"kind-dispatch census rose to {total} (ceiling {KIND_DISPATCH_CEILING}). "
        f"The engine core must not gain kind dispatch. Sites: {breakdown}"
    )


def test_kind_dispatch_ceiling_matches_observed_count():
    """The configured ceiling must equal the observed number of dispatch sites."""
    total = sum(len(lines) for lines in _census().values())
    assert total == KIND_DISPATCH_CEILING, (
        f"kind-dispatch census is {total} but KIND_DISPATCH_CEILING is "
        f"{KIND_DISPATCH_CEILING}. Set the ceiling to {total} to keep the ratchet tight."
    )
