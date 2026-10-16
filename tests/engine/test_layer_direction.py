"""Import-direction ratchets for seed-first graph growth.

Nodes are the lower layer: they may describe a growth seed but cannot import the
compile package that interprets it or the runtime that drives it. Runtime receives
the growth interpreter through an injected callable and therefore cannot import
``compile.expand`` directly.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).parents[2] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "yian"


def _package_parts(path: Path) -> tuple[str, ...]:
    """Return the ``__package__`` path relative imports resolve against."""
    return path.relative_to(SOURCE_ROOT).parent.parts


def _parse_import_targets(
    source: str,
    *,
    package: tuple[str, ...],
    filename: str = "<import-ratchet>",
    include_from_bases: bool = True,
) -> list[tuple[int, str, str]]:
    """Return ``(line, target, source)`` for imports in one module body.

    ``ImportFrom`` statements include both their resolved base and each imported
    name.  That catches equivalent spellings such as ``from yian import
    compile`` and ``from .. import compile`` as well as direct module imports.
    With ``include_from_bases=False``, omit only the synthetic base of a named
    ``ImportFrom``; direct imports and wildcard bases remain actual targets.
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

        if base and (
            include_from_bases or any(alias.name == "*" for alias in node.names)
        ):
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
    violations = _forbidden_imports(PACKAGE_ROOT / "nodes", "yian.compile")
    assert not violations, (
        "nodes must not import compile; return a canonical Grow seed and let the "
        "injected compile-layer builder interpret it:\n" + "\n".join(violations)
    )


def test_nodes_never_import_runtime() -> None:
    """Node implementations depend only on lower layers, never their driver."""
    violations = _forbidden_imports(PACKAGE_ROOT / "nodes", "yian.runtime")
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
        "yian.diagnostics",
        "yian.typesys",
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


def test_compile_never_imports_front_end_or_runtime() -> None:
    """Compile owns source-neutral graph transforms, not loading or execution."""
    violations = []
    for forbidden in ("yian.compose", "yian.runtime"):
        violations.extend(_forbidden_imports(PACKAGE_ROOT / "compile", forbidden))
    assert not violations, (
        "compile must stay representation-neutral and execution-free:\n"
        + "\n".join(violations)
    )


def test_checkpoint_era_suspension_package_is_deleted() -> None:
    """Keep durable lifecycle authority in runtime logs, not a parallel package."""
    assert not (PACKAGE_ROOT / "suspension").exists()


def test_scanner_normalizes_absolute_and_relative_import_spellings() -> None:
    """Equivalent import syntax resolves to the same forbidden module targets."""
    assert _package_parts(PACKAGE_ROOT / "nodes" / "call" / "node.py") == (
        "yian",
        "nodes",
        "call",
    )
    assert _package_parts(PACKAGE_ROOT / "nodes" / "call" / "__init__.py") == (
        "yian",
        "nodes",
        "call",
    )

    cases = (
        (
            "import yian.compile.expand\n",
            ("yian", "nodes", "call"),
            "yian.compile.expand",
        ),
        (
            "from yian import compile\n",
            ("yian", "nodes", "call"),
            "yian.compile",
        ),
        (
            "from yian.compile import expand\n",
            ("yian", "nodes", "call"),
            "yian.compile.expand",
        ),
        (
            "from ... import compile\n",
            ("yian", "nodes", "call"),
            "yian.compile",
        ),
        (
            "from ...compile import expand\n",
            ("yian", "nodes", "call"),
            "yian.compile.expand",
        ),
        (
            "from ..compile import expand\n",
            ("yian", "runtime"),
            "yian.compile.expand",
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
        PACKAGE_ROOT / "runtime", "yian.compile.expand"
    )
    assert not violations, (
        "runtime must not import compile.expand; the run boundary injects build_grow:\n"
        + "\n".join(violations)
    )


def _memory_leaf_violations(module: str, source: str) -> list[str]:
    """Check one declared memory leaf through the shared import target scanner."""
    leaves = {
        "contracts": ("yian.diagnostics", "yian.defaults"),
        "conversation": ("yian.memory.contracts",),
    }[module]
    targets = _parse_import_targets(
        source,
        package=("yian", "memory"),
        include_from_bases=False,
    )
    violations = []
    for line, target, statement in targets:
        if target.split(".")[0] in sys.stdlib_module_names:
            continue
        if any(target == leaf or target.startswith(f"{leaf}.") for leaf in leaves):
            continue
        violations.append(f"memory/{module}.py:{line}: {statement} -> {target}")
    return violations




def test_memory_leaf_ratchet_rejects_absolute_and_relative_imports() -> None:
    """Prove the lower-memory boundary using snippets, not production mutations."""
    for module in ("contracts", "conversation"):
        for forbidden in (
            "nodes",
            "compile",
            "compose",
            "runtime",
            "run",
            "llm_clients",
        ):
            for source in (
                f"import yian.{forbidden}\n",
                f"from yian import {forbidden}\n",
                f"from .. import {forbidden}\n",
                f"from ..{forbidden} import Something\n",
            ):
                assert _memory_leaf_violations(module, source), source
        for backend in ("episodic", "reflective", "session"):
            for source in (
                f"import yian.memory.{backend}\n",
                f"from . import {backend}\n",
                f"from .{backend} import Something\n",
            ):
                assert _memory_leaf_violations(module, source), source
        for sdk in ("langchain", "openai", "anthropic", "codex"):
            assert _memory_leaf_violations(module, f"import {sdk}\n")
    assert _memory_leaf_violations(
        "contracts", "from .conversation import advance_conversation\n"
    )
    assert _memory_leaf_violations(
        "conversation", "from ..diagnostics import Failure\n"
    )


def test_memory_leaf_ratchet_accepts_only_declared_leaves_and_stdlib() -> None:
    """Normalize import bases without accidentally allowing a package facade."""
    for module, sources in (
        (
            "contracts",
            (
                "from yian.diagnostics import OriginRef\n",
                "from ..diagnostics import OriginRef\n",
                "from yian import diagnostics\n",
                "from yian.defaults import current\n",
            ),
        ),
        (
            "conversation",
            (
                "import yian.memory.contracts as c\n",
                "from .contracts import ConversationView\n",
                "from . import contracts\n",
                "from yian.memory import contracts\n",
            ),
        ),
    ):
        for source in sources:
            assert not _memory_leaf_violations(module, source), source
        assert not _memory_leaf_violations(
            module, "import json\nfrom typing import Callable\n"
        )
        assert _memory_leaf_violations(module, "import yian\n")
        assert _memory_leaf_violations(module, "import yian.memory\n")
    assert _memory_leaf_violations("conversation", "from . import contracts, session\n")


def test_memory_leaf_ratchet_never_exempts_direct_imports_on_a_shared_line() -> None:
    """An allowed leaf cannot authorize a facade in another import or alias."""
    cases = (
        (
            "conversation",
            "import yian.memory, yian.memory.contracts",
        ),
        ("conversation", "import yian.memory; from . import contracts"),
        ("contracts", "import yian, yian.diagnostics"),
    )
    missed = [
        source
        for module, source in cases
        if not _memory_leaf_violations(module, source)
    ]
    assert not missed, f"direct facade imports escaped the memory ratchet: {missed}"


def test_memory_leaf_ratchet_checks_wildcard_import_bases() -> None:
    """Wildcard imports have no named targets, so their own base remains checked."""
    assert not _memory_leaf_violations("conversation", "from .contracts import *")
    assert not _memory_leaf_violations("contracts", "from ..diagnostics import *")
    for module in ("contracts", "conversation"):
        for source in (
            "from . import *",
            "from yian import *",
            "from .session import *",
            "from ..runtime import *",
            "from openai import *",
        ):
            assert _memory_leaf_violations(module, source), source


def test_memory_conversation_leaves_import_only_declared_leaves() -> None:
    """Keep contracts below the pure conversation fold and both below backends."""
    violations = []
    for module in ("contracts", "conversation"):
        path = PACKAGE_ROOT / "memory" / f"{module}.py"
        violations.extend(_memory_leaf_violations(module, path.read_text()))
    assert (
        not violations
    ), "lower memory imports undeclared dependencies:\n" + "\n".join(violations)


def test_skills_leaf_imports_no_engine_or_front_end_layer() -> None:
    """Nodes import the skills leaf, so it must never import back up the stack."""
    violations: list[str] = []
    for forbidden in (
        "yian.nodes",
        "yian.compile",
        "yian.compose",
        "yian.runtime",
        "yian.run",
        "yian.cli",
    ):
        violations.extend(_forbidden_imports(PACKAGE_ROOT / "skills", forbidden))
    assert not violations, "skills must stay a leaf:\n" + "\n".join(violations)


def _package_root_imports(root: Path) -> list[str]:
    """Format imports below ``root`` of the ``yian`` package facade itself."""
    return [
        f"{path.relative_to(SOURCE_ROOT)}:{line}: {statement} -> {target}"
        for path in sorted(root.rglob("*.py"))
        for line, target, statement in _import_targets(path)
        if target == "yian"
    ]


def test_memory_never_imports_engine_host_or_cli_layers() -> None:
    """Backends sit below the engine; hosts build them, they never build hosts."""
    violations = []
    for forbidden in ("runtime", "compose", "compile", "run", "memory_host", "cli"):
        violations.extend(
            _forbidden_imports(PACKAGE_ROOT / "memory", f"yian.{forbidden}")
        )
    violations.extend(_package_root_imports(PACKAGE_ROOT / "memory"))
    assert not violations, (
        "memory must not import runtime, compose, compile, run, memory_host, cli, "
        "or the package root:\n" + "\n".join(violations)
    )


def test_runtime_never_imports_memory_host_or_cli() -> None:
    """Runtime receives built memory; construction and flags stay with hosts."""
    violations = []
    for forbidden in ("yian.memory_host", "yian.cli"):
        violations.extend(_forbidden_imports(PACKAGE_ROOT / "runtime", forbidden))
    assert not violations, (
        "runtime must not import memory_host or cli:\n" + "\n".join(violations)
    )


def test_package_root_scan_catches_facade_spellings() -> None:
    """The facade check sees absolute and relative root imports, not submodules."""
    for source in (
        "import yian\n",
        "from yian import run_flow\n",
        "from .. import build_memories\n",
    ):
        targets = {
            target
            for _, target, _ in _parse_import_targets(
                source, package=("yian", "memory")
            )
        }
        assert "yian" in targets, source
    targets = {
        target
        for _, target, _ in _parse_import_targets(
            "from ..diagnostics import Failure\n", package=("yian", "memory")
        )
    }
    assert "yian" not in targets


def test_defaults_leaf_imports_only_stdlib() -> None:
    """Every layer reads its defaults and limits here, so it imports no yian module."""
    path = PACKAGE_ROOT / "defaults.py"
    violations = [
        f"{path.relative_to(SOURCE_ROOT)}:{line}: {statement} -> {target}"
        for line, target, statement in _import_targets(path)
        if target.split(".")[0] not in sys.stdlib_module_names
    ]
    assert not violations, "defaults must stay a stdlib leaf:\n" + "\n".join(
        violations
    )


def _llm_leaf_violations(
    source: str, package: tuple[str, ...] = ("yian", "llm")
) -> list[str]:
    """Check one ``yian.llm`` module through the shared import target scanner.

    Args:
        source (`str`): The module source to scan.
        package (`tuple[str, ...]`): The module's ``__package__`` parts, which
            relative imports resolve against.

    Returns:
        `list[str]`: One ``line: statement -> target`` entry per undeclared import.
    """
    leaves = ("httpx2", "yian.defaults", "yian.llm")
    targets = _parse_import_targets(
        source, package=package, include_from_bases=False
    )
    violations = []
    for line, target, statement in targets:
        if target.split(".")[0] in sys.stdlib_module_names:
            continue
        if any(target == leaf or target.startswith(f"{leaf}.") for leaf in leaves):
            continue
        violations.append(f"{line}: {statement} -> {target}")
    return violations


def test_llm_leaf_imports_only_declared_leaves() -> None:
    """The provider-neutral llm core imports only stdlib, httpx2, and defaults."""
    root = PACKAGE_ROOT / "llm"
    assert root.is_dir(), "src/yian/llm is missing, so the ratchet would pass vacuously"
    violations = [
        f"{path.relative_to(SOURCE_ROOT)}:{violation}"
        for path in sorted(root.rglob("*.py"))
        for violation in _llm_leaf_violations(
            path.read_text(), package=_package_parts(path)
        )
    ]
    assert not violations, (
        "llm may import only the standard library, httpx2, and yian.defaults:\n"
        + "\n".join(violations)
    )


def test_llm_leaf_ratchet_rejects_other_imports() -> None:
    """SDKs, the package facade, and sibling yian packages are all rejected."""
    for source in (
        "import langchain\n",
        "import openai\n",
        "import yian\n",
        "from yian import *\n",
        "from .. import *\n",
        "import yian.llm_clients\n",
        "from yian import llm_clients\n",
        "from ..nodes import x\n",
    ):
        assert _llm_leaf_violations(source), source


def test_llm_leaf_ratchet_accepts_declared_leaves() -> None:
    """Absolute and relative spellings of the declared leaves pass."""
    for source in (
        "from yian.defaults import current\n",
        "from .. import defaults\n",
        "from . import messages\n",
        "from yian.llm.errors import ProviderError\n",
        "import httpx2\n",
        "import json\n",
    ):
        assert not _llm_leaf_violations(source), source
