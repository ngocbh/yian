"""Contracts for source-neutral diagnostic values and origin resolution."""

from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
import subprocess
import sys
from typing import NamedTuple

import pytest

from agent_composer.diagnostics import (
    DiagnosticFrame,
    OriginRef,
    OriginResolver,
    ResolverTable,
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


def test_resolver_table_routes_each_new_origin_to_one_exact_delegate_origin() -> None:
    """One frozen table can route colliding delegate-local keys without precedence."""
    first_frame = DiagnosticFrame(
        label="first",
        text="first\n",
        range=SourceRange(1, 1, 1, 6),
    )
    second_frame = DiagnosticFrame(
        label="second",
        text="second\n",
        range=SourceRange(1, 1, 1, 7),
    )
    calls: list[tuple[str, OriginRef]] = []

    class Resolver:
        def __init__(self, name: str, frame: DiagnosticFrame) -> None:
            self.name = name
            self.frame = frame

        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            calls.append((self.name, origin))
            return self.frame if origin == OriginRef("local") else None

    first = Resolver("first", first_frame)
    second = Resolver("second", second_frame)
    first_new = OriginRef("call/a")
    second_new = OriginRef("call/b")

    class Entry(NamedTuple):
        new: OriginRef
        delegate: OriginResolver
        old: OriginRef

    table = ResolverTable.from_entries(
        [
            Entry(first_new, first, OriginRef("local")),
            (second_new, second, OriginRef("local")),
        ]
    )

    assert len(table) == 2
    assert table.origins == frozenset({first_new, second_new})
    assert first_new in table
    assert second_new in table
    assert OriginRef("missing") not in table
    assert table.resolve(first_new) == first_frame
    assert table.resolve(second_new) == second_frame
    assert table.resolve(OriginRef("missing")) is None
    assert calls == [
        ("first", OriginRef("local")),
        ("second", OriginRef("local")),
    ]


def test_resolver_table_merge_is_strict_and_preserves_delegate_behavior() -> None:
    """Merging uses exact disjoint routes and never adds fallback semantics."""

    class Resolver:
        def __init__(self, label: str) -> None:
            self.label = label

        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            if origin != OriginRef("old"):
                return None
            return DiagnosticFrame(
                label=self.label,
                text="value\n",
                range=SourceRange(1, 1, 1, 6),
            )

    first = ResolverTable.from_entries(
        [(OriginRef("first"), Resolver("first"), OriginRef("old"))]
    )
    second = ResolverTable.from_entries(
        [(OriginRef("second"), Resolver("second"), OriginRef("old"))]
    )

    merged = ResolverTable.merge(first, second)

    assert len(merged) == 2
    assert merged.resolve(OriginRef("first")).label == "first"  # type: ignore[union-attr]
    assert merged.resolve(OriginRef("second")).label == "second"  # type: ignore[union-attr]
    assert len(ResolverTable.merge()) == 0


def test_resolver_table_rejects_duplicate_routes_before_information_is_lost() -> None:
    """Duplicate public keys fail even when their resolution targets are identical."""

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return None

    resolver = Resolver()
    new = OriginRef("same")
    old = OriginRef("old")

    class DerivedOrigin(OriginRef):
        """Distinct Python class that still names the same opaque identity."""

    with pytest.raises(ValueError, match="duplicate resolver table origin 'same'"):
        ResolverTable.from_entries(
            [(new, resolver, old), (DerivedOrigin("same"), resolver, old)]
        )

    first = ResolverTable.from_entries([(new, resolver, old)])
    second = ResolverTable.from_entries([(new, resolver, old)])
    with pytest.raises(ValueError, match="duplicate resolver table origin 'same'"):
        ResolverTable.merge(first, second)


def test_resolver_table_revalidates_subclass_keys_before_routing() -> None:
    """A subclass cannot bypass the canonical origin key invariant."""

    class InvalidOrigin(OriginRef):
        def __post_init__(self) -> None:
            pass

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return None

    invalid = InvalidOrigin("")
    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        ResolverTable.from_entries(
            [(invalid, Resolver(), OriginRef("old"))]
        )
    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        ResolverTable.from_entries(
            [(OriginRef("new"), Resolver(), invalid)]
        )


def test_resolver_table_is_defensive_immutable_and_propagates_delegate_failures() -> None:
    """Routes are snapshotted, deepcopy is free, and delegates retain authority."""

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            raise RuntimeError(f"cannot resolve {origin.key}")

    entries = [(OriginRef("new"), Resolver(), OriginRef("old"))]
    table = ResolverTable.from_entries(entries)
    entries.clear()

    assert len(table) == 1
    assert copy.copy(table) is table
    assert copy.deepcopy(table) is table
    with pytest.raises(AttributeError, match="immutable"):
        table._targets = {}  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="immutable"):
        del table._targets  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="dataclass"):
        replace(table)
    with pytest.raises(RuntimeError, match="cannot resolve old"):
        table.resolve(OriginRef("new"))


def test_resolver_table_rejects_delegate_results_outside_the_protocol() -> None:
    """A structural delegate must still return one diagnostic frame or `None`."""

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return "not-a-frame"  # type: ignore[return-value]

    table = ResolverTable.from_entries(
        [(OriginRef("new"), Resolver(), OriginRef("old"))]
    )

    with pytest.raises(
        TypeError,
        match="delegate must return a DiagnosticFrame or None",
    ):
        table.resolve(OriginRef("new"))


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: ResolverTable.from_entries(
                [("new", object(), OriginRef("old"))]  # type: ignore[list-item]
            ),
            "new origin must be an OriginRef",
        ),
        (
            lambda: ResolverTable.from_entries(
                [(OriginRef("new"), object(), OriginRef("old"))]
            ),
            "delegate must implement resolve",
        ),
        (
            lambda: ResolverTable.from_entries(
                [(OriginRef("new"), ResolverTable(), "old")]  # type: ignore[list-item]
            ),
            "old origin must be an OriginRef",
        ),
        (
            lambda: ResolverTable.from_entries(
                [(OriginRef("new"), ResolverTable())]  # type: ignore[list-item]
            ),
            "entry must be a three-item tuple",
        ),
        (
            lambda: ResolverTable({}),  # type: ignore[call-arg]
            "positional argument",
        ),
        (
            lambda: ResolverTable.merge(ResolverTable(), object()),  # type: ignore[arg-type]
            "merge requires ResolverTable values",
        ),
        (
            lambda: ResolverTable().resolve("new"),  # type: ignore[arg-type]
            "resolve requires an OriginRef",
        ),
    ],
)
def test_resolver_table_rejects_malformed_routes(operation, message: str) -> None:
    """Malformed routing values fail at the public boundary with stable errors."""
    with pytest.raises(TypeError, match=message):
        operation()


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
    assert imports <= {"dataclasses", "types", "typing"}


def test_public_diagnostics_import_does_not_load_engine_or_yaml_layers() -> None:
    """The package facade must not defeat the diagnostic leaf's isolation."""
    source_root = Path(__file__).parents[2] / "src"
    script = """
import sys
from agent_composer.diagnostics import DiagnosticFrame, OriginRef, OriginResolver, ResolverTable, SourceRange

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
