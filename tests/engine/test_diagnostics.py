"""Contracts for source-neutral diagnostic values and origin resolution."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import subprocess
import sys

import pytest

from agent_composer.diagnostics import (
    DiagnosticFrame,
    OriginRef,
    OriginResolver,
    SourceRange,
)


def test_origin_ref_is_a_validated_frozen_single_key() -> None:
    """An origin is an opaque, non-empty string value and nothing more."""
    origin = OriginRef("root:node")

    assert origin.key == "root:node"
    assert [field.name for field in fields(OriginRef)] == ["key"]
    assert hash(origin) == hash(OriginRef("root:node"))
    with pytest.raises(FrozenInstanceError):
        origin.key = "changed"  # type: ignore[misc]

    for invalid in ("", None, 0, False):
        with pytest.raises(ValueError, match="non-empty string"):
            OriginRef(invalid)  # type: ignore[arg-type]


def test_diagnostic_frame_is_frozen_presentation_data() -> None:
    """A resolved frame carries only display text and one concrete range."""
    source_range = SourceRange(2, 3, 2, 8)
    frame = DiagnosticFrame(
        label="flow.yaml",
        text="nodes:\n  ask: {}\n",
        range=source_range,
        note="inside call 'review'",
    )

    assert frame == DiagnosticFrame(
        label="flow.yaml",
        text="nodes:\n  ask: {}\n",
        range=source_range,
        note="inside call 'review'",
    )
    with pytest.raises(FrozenInstanceError):
        frame.note = None  # type: ignore[misc]


def test_origin_resolver_is_structural_and_may_return_none() -> None:
    """Any source front end can implement resolution without a compose type."""

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            if origin.key != "known":
                return None
            return DiagnosticFrame(
                label="memory",
                text="value\n",
                range=SourceRange(1, 1, 1, 6),
            )

    resolver: OriginResolver = Resolver()

    assert resolver.resolve(OriginRef("known")) is not None
    assert resolver.resolve(OriginRef("missing")) is None


def test_diagnostics_module_imports_only_the_standard_library() -> None:
    """The diagnostic value spine must remain below every project package."""
    path = Path(__file__).parents[2] / "src" / "agent_composer" / "diagnostics.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    relative_imports = [
        node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level
    ]
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not relative_imports
    assert imports <= {"dataclasses", "typing"}


def test_public_diagnostics_import_does_not_load_engine_or_yaml_layers() -> None:
    """The package facade must not defeat the diagnostic leaf's isolation."""
    source_root = Path(__file__).parents[2] / "src"
    script = """
import sys
from agent_composer.diagnostics import DiagnosticFrame, OriginRef, OriginResolver, SourceRange

loaded_project_modules = sorted(
    name
    for name in sys.modules
    if name.startswith("agent_composer.") and name != "agent_composer.diagnostics"
)
loaded_yaml_modules = sorted(
    name for name in sys.modules if name == "yaml" or name.startswith("yaml.")
)
assert not loaded_project_modules, loaded_project_modules
assert not loaded_yaml_modules, loaded_yaml_modules
"""

    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_lazy_package_facade_preserves_every_declared_root_export() -> None:
    """Leaf isolation must not remove or rename the established root API."""
    source_root = Path(__file__).parents[2] / "src"
    script = """
import agent_composer

expected = [
    "ExpressionError",
    "FlowValidationError",
    "LoadError",
    "ComposedFlow",
    "CompiledFlow",
    "VariablePool",
    "evaluate_when",
    "load_flow",
    "run_flow",
]
assert agent_composer.__all__ == expected
assert set(expected) <= set(dir(agent_composer))
for name in expected:
    assert getattr(agent_composer, name) is not None
try:
    agent_composer.not_a_public_export
except AttributeError as error:
    assert error.__cause__ is None
    assert error.__context__ is None
else:
    raise AssertionError("unknown package attributes must raise AttributeError")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
