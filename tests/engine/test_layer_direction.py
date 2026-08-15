"""Import-direction ratchets for seed-first graph growth.

Nodes are the lower layer: they may describe a growth seed but cannot import the
compile package that interprets it or the runtime that drives it. Runtime receives
the growth interpreter through an injected callable and therefore cannot import
``compile.expand`` directly.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "agent_composer"


def _package_parts(path: Path) -> tuple[str, ...]:
    """Return the ``__package__`` path relative imports resolve against."""
    return path.relative_to(SOURCE_ROOT).parent.parts


def _parse_import_targets(
    source: str,
    *,
    package: tuple[str, ...],
    filename: str = "<import-ratchet>",
) -> list[tuple[int, str, str]]:
    """Return ``(line, target, source)`` for imports in one module body.

    ``ImportFrom`` statements include both their resolved base and each imported
    name.  That catches equivalent spellings such as ``from agent_composer import
    compile`` and ``from .. import compile`` as well as direct module imports.
    """
    lines = source.splitlines()
    tree = ast.parse(source, filename=filename)
    targets: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        line = lines[node.lineno - 1].strip() if hasattr(node, "lineno") else ""
        if isinstance(node, ast.Import):
            targets.extend((node.lineno, alias.name, line) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            keep = len(package) - (node.level - 1)
            base_parts = (*package[:keep], *((node.module or "").split(".")))
            base = ".".join(part for part in base_parts if part)
        else:
            base = node.module or ""

        if base:
            targets.append((node.lineno, base, line))
        targets.extend(
            (node.lineno, ".".join(part for part in (base, alias.name) if part), line)
            for alias in node.names
            if alias.name != "*"
        )

    return targets


def _import_targets(path: Path) -> list[tuple[int, str, str]]:
    """Parse imports from ``path`` using that module's real package context."""
    return _parse_import_targets(
        path.read_text(), package=_package_parts(path), filename=str(path)
    )


def _forbidden_imports(root: Path, forbidden: str) -> list[str]:
    """Format imports below ``root`` that target ``forbidden`` or a child module."""
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line, target, statement in _import_targets(path):
            if target == forbidden or target.startswith(f"{forbidden}."):
                relative = path.relative_to(SOURCE_ROOT)
                violations.append(f"{relative}:{line}: {statement} -> {target}")
    return violations


def test_nodes_never_import_compile() -> None:
    """Node implementations return seeds and never depend on their compiler."""
    violations = _forbidden_imports(PACKAGE_ROOT / "nodes", "agent_composer.compile")
    assert not violations, (
        "nodes must not import compile; return a canonical Grow seed and let the "
        "injected compile-layer builder interpret it:\n" + "\n".join(violations)
    )


def test_nodes_never_import_runtime() -> None:
    """Node implementations depend only on lower layers, never their driver."""
    violations = _forbidden_imports(PACKAGE_ROOT / "nodes", "agent_composer.runtime")
    assert not violations, (
        "nodes must not import runtime; direct-driver fallbacks belong on a lower seam:\n"
        + "\n".join(violations)
    )


def test_semantic_contracts_import_only_their_declared_project_leaves() -> None:
    """Keep the shared semantic vocabulary independent of engine and front-end layers."""
    path = PACKAGE_ROOT / "nodes" / "contracts.py"
    allowed = (
        "__future__",
        "dataclasses",
        "typing",
        "agent_composer.diagnostics",
        "agent_composer.typesys",
    )
    violations = [
        f"{path.relative_to(SOURCE_ROOT)}:{line}: {statement} -> {target}"
        for line, target, statement in _import_targets(path)
        if not any(target == leaf or target.startswith(f"{leaf}.") for leaf in allowed)
    ]
    assert not violations, (
        "nodes/contracts.py may import only stdlib plus diagnostics and typesys:\n"
        + "\n".join(violations)
    )


def test_checkpoint_era_suspension_package_is_deleted() -> None:
    """Keep durable lifecycle authority in runtime logs, not a parallel package."""
    assert not (PACKAGE_ROOT / "suspension").exists()


def test_scanner_normalizes_absolute_and_relative_import_spellings() -> None:
    """Equivalent import syntax resolves to the same forbidden module targets."""
    assert _package_parts(PACKAGE_ROOT / "nodes" / "call" / "node.py") == (
        "agent_composer",
        "nodes",
        "call",
    )
    assert _package_parts(PACKAGE_ROOT / "nodes" / "call" / "__init__.py") == (
        "agent_composer",
        "nodes",
        "call",
    )

    cases = (
        (
            "import agent_composer.compile.expand\n",
            ("agent_composer", "nodes", "call"),
            "agent_composer.compile.expand",
        ),
        (
            "from agent_composer import compile\n",
            ("agent_composer", "nodes", "call"),
            "agent_composer.compile",
        ),
        (
            "from agent_composer.compile import expand\n",
            ("agent_composer", "nodes", "call"),
            "agent_composer.compile.expand",
        ),
        (
            "from ... import compile\n",
            ("agent_composer", "nodes", "call"),
            "agent_composer.compile",
        ),
        (
            "from ...compile import expand\n",
            ("agent_composer", "nodes", "call"),
            "agent_composer.compile.expand",
        ),
        (
            "from ..compile import expand\n",
            ("agent_composer", "runtime"),
            "agent_composer.compile.expand",
        ),
    )

    for source, package, expected in cases:
        targets = {
            target
            for _, target, _ in _parse_import_targets(source, package=package)
        }
        assert expected in targets, f"{source.strip()} resolved to {sorted(targets)}"


def test_runtime_never_imports_compile_expand() -> None:
    """Runtime accepts growth construction as a seam instead of importing it."""
    violations = _forbidden_imports(
        PACKAGE_ROOT / "runtime", "agent_composer.compile.expand"
    )
    assert not violations, (
        "runtime must not import compile.expand; the run boundary injects build_grow:\n"
        + "\n".join(violations)
    )
