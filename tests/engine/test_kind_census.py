"""Kind-dispatch census — a ratchet that guards the kind-agnostic refactor.

The engine is being migrated so that its core (`runtime/engine.py` + `runtime/eval_node.py`)
knows only an abstract node contract and a closed `Outcome` sum — it must NOT branch on a node's
`NodeKind` nor on a concrete `*Expansion` type. This test *counts* the remaining kind-dispatch
sites and asserts the count never rises above a recorded ceiling.

Each refactor phase lowers `BASELINE`; the final phase drives it to `0`. The count is derived
from the source (self-calibrating) rather than a hand-maintained line list, so it survives
line-number drift — only the ceiling constant moves.

A "kind-dispatch site" is a distinct source line that does any of:
  - reference a `NodeKind` member  (`NodeKind.AGENT`, `NodeKind.CASE`, ...)   -> branches on kind
  - reference `_SPAWNER_KINDS`      (the spawner-kind set used for membership) -> branches on kind
  - call `isinstance(x, <...>Expansion)`                                       -> branches on concrete type

Import lines are excluded (importing `NodeKind` / `_SPAWNER_KINDS` is not dispatch), and a
diagnostic read like `node.kind.value` (for an error message) is not counted — it touches the
instance attribute, not a `NodeKind` member or an `*Expansion` type.

Known blind spots (do NOT route dispatch through these to dodge the ratchet — none exist in the
core today, and reintroducing kind dispatch in one of these shapes defeats the whole refactor):
indirect enum access (`getattr(NodeKind, name)`, `NodeKind[name]`, `NodeKind(value)`), a
`match node.kind` / `case NodeKind.X` block, kind-keyed dispatch dicts built off-file, and
type-name string compares (`type(desc).__name__ == "CallExpansion"`). The `*Expansion` detection
is a name-suffix heuristic tied to the codebase convention that every expansion type ends in
`Expansion`; keep that convention so the census stays accurate.
"""

import ast
from pathlib import Path

from agent_composer.runtime import engine as engine_mod
from agent_composer.runtime import eval_node as eval_node_mod

# The two modules that make up the engine core the census guards.
CORE_MODULES = {
    "runtime/engine.py": engine_mod,
    "runtime/eval_node.py": eval_node_mod,
}

# Ratchet ceiling: the number of kind-dispatch lines allowed in the engine core.
# LOWER this as each refactor phase removes dispatch; the final phase drives it to 0.
# (Measured at P0 baseline 20; dropped to 19 when the `eval_node` grow guard moved from a
# `node.kind not in _SPAWNER_KINDS` membership test to the kind-blind `not node.is_spawner`.
# Rose through the former spawner migrations to 27 while the live `_grow_residual` arms
# COEXISTED with the legacy `_apply_enqueue`/`_grow_*` replay path + the `*Expansion` union
# isinstance dispatch. P3.5 unified the durability ledger to a single `GrowRecord` and made
# replay kind-blind: the `*Expansion` union + its isinstance dispatch and the legacy
# per-kind replay bodies are gone, dropping the ceiling to 13.
# P3.6 deleted `Enqueue`/`_apply_enqueue` (LOOP turn-0 now returns `Grow`) and swapped the 3
# `_SPAWNER_KINDS` residual stamps + the `eval_node` def for the kind-blind `node.is_spawner`,
# dropping the ceiling to 8.
# P5.1 moved the two `eval_node` read-boundary reads (MAP over-mode, WAIT until) behind the
# node-owned `bind_reserved`/`binds_per_item` hooks, dropping the ceiling to 6.
# P5.2 bound assert-refs at the read boundary (record-first, pool-fallback) so END post-asserts
# stop being special: the `node.kind == NodeKind.END` post-assert branch in `eval_node` is gone,
# dropping the ceiling to 5.
# P5.3 dropped the `target_node.kind == NodeKind.CALL` conjunct on the commit-site post-assert
# gate (engine.py) — a redirect-commit post-check is now kind-blind, dropping the ceiling to 4.
# P5.4 dismantles the `_grow_residual` 4-arm switch concern by concern (finish/mark, boundary
# asserts, depth, `_spawner_expansion` stamping, and loop bookkeeping all lifted into
# the kind-blind growth core behind node traits/hooks). The final
# P5.4 commit gated the LOOP bookkeeping on the `is_loop` trait and deleted the switch entirely.
# The census is now EMPTY: the engine core dispatches only on node traits/hooks + the closed
# Outcome sum, never on `NodeKind` / `_SPAWNER_KINDS` / a concrete `*Expansion` type.
BASELINE = 0


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


def test_kind_dispatch_stays_at_or_below_baseline():
    census = _census()
    total = sum(len(lines) for lines in census.values())
    breakdown = {rel: sorted(lines) for rel, lines in census.items()}
    assert total <= BASELINE, (
        f"kind-dispatch census rose to {total} (ceiling {BASELINE}). "
        f"The engine core must not gain kind dispatch. Sites: {breakdown}"
    )


def test_baseline_is_tight():
    """The ceiling must track reality: if dispatch was removed, LOWER BASELINE (don't leave slack)."""
    total = sum(len(lines) for lines in _census().values())
    assert total == BASELINE, (
        f"kind-dispatch census is {total} but BASELINE is {BASELINE}. "
        f"A phase removed dispatch — lower BASELINE to {total} to keep the ratchet tight."
    )
