"""Source-aware parsing of authored ``long_term_memory:`` declarations."""

from __future__ import annotations

from typing import Any

from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.diagnostics import SourceRange
from agent_composer.memory.contracts import (
    DeclSite,
    MemoryDecl,
    MemoryPolicySpec,
    check_memory_declaration,
)

_LONG_FORM_KEYS = frozenset({"type", "config"})
# Reserved so the retired ``{name:, config:}`` form and a stray top-level
# ``type:`` can never parse as memories.
_RESERVED_NAMES = frozenset({"config", "name", "type"})


def parse_memory_declarations(
    value: Any,
    *,
    diagnostics: LoadDiagnostics,
    source_range: SourceRange | None,
    site: DeclSite,
) -> tuple[MemoryDecl, ...]:
    """Parse one present, non-null ``long_term_memory:`` value into declarations.

    ``type`` is sugar for ``{type: type}``. A mapping declares one memory per
    entry, as ``name: type`` or ``name: {type: ..., config: {...}}``, in
    authored order. Origins are attached later by the caller.

    Args:
        value (`Any`): Authored value. The caller treats absence and ``null`` as
            no declaration and never passes them here.
        diagnostics (`LoadDiagnostics`): Source context for errors.
        source_range (`SourceRange | None`): Range of the whole value.
        site (`DeclSite`): Definition site stamped on every declaration.

    Returns:
        `tuple[MemoryDecl, ...]`: One declaration per authored entry.

    Raises:
        `LoadError`: The value has the wrong shape or a declaration is invalid.
    """
    if type(value) is str:
        return (_declaration(value, value, {}, site, diagnostics, source_range),)
    if type(value) is not dict:
        raise diagnostics.error(
            "long_term_memory must be a type name or a mapping of memory names "
            "to types",
            range=source_range,
        )
    if not value:
        raise diagnostics.error(
            "long_term_memory: {} declares no memory; remove the key or name one",
            range=source_range,
        )
    declarations: list[MemoryDecl] = []
    for name, entry in value.items():
        if type(name) is not str:
            raise diagnostics.error(
                f"long_term_memory name {name!r} must be a string",
                range=source_range,
            )
        if name in _RESERVED_NAMES:
            raise diagnostics.error(
                f"long_term_memory name {name!r} is reserved "
                f"(reserved: {', '.join(sorted(_RESERVED_NAMES))})",
                range=source_range,
            )
        if type(entry) is str:
            declarations.append(
                _declaration(name, entry, {}, site, diagnostics, source_range)
            )
            continue
        if type(entry) is not dict:
            raise diagnostics.error(
                f"long_term_memory {name!r} must be a type name or a mapping "
                "with 'type' and optional 'config'",
                range=source_range,
            )
        extras = sorted(repr(key) for key in entry if key not in _LONG_FORM_KEYS)
        if extras:
            raise diagnostics.error(
                f"long_term_memory {name!r} has unknown key(s) {extras} "
                "(allowed: type, config)",
                range=source_range,
            )
        if "type" not in entry:
            raise diagnostics.error(
                f"long_term_memory {name!r} requires 'type'", range=source_range
            )
        if type(entry["type"]) is not str:
            raise diagnostics.error(
                f"long_term_memory {name!r} type must be a string",
                range=source_range,
            )
        config = entry.get("config", {})
        if type(config) is not dict:
            raise diagnostics.error(
                f"long_term_memory {name!r} config must be a mapping",
                range=source_range,
            )
        declarations.append(
            _declaration(name, entry["type"], config, site, diagnostics, source_range)
        )
    return tuple(declarations)


def _declaration(
    name: str,
    type_name: str,
    config: dict[str, Any],
    site: DeclSite,
    diagnostics: LoadDiagnostics,
    source_range: SourceRange | None,
) -> MemoryDecl:
    """Build one declaration, reporting a shape failure at the authored value."""
    try:
        return MemoryDecl(name, type_name, config, site)
    except (TypeError, ValueError) as exc:
        raise diagnostics.error(
            f"long_term_memory {name!r}: {exc}", range=source_range
        ) from exc


def parse_memory_policy(
    value: Any,
    *,
    diagnostics: LoadDiagnostics,
    source_range: SourceRange | None,
) -> MemoryPolicySpec:
    """Parse one present policy occurrence into its source-neutral carrier."""
    if type(value) is str:
        if not value:
            raise diagnostics.error(
                "long_term_memory name must be non-empty",
                range=source_range,
            )
        try:
            check_memory_declaration(value, value, {})
        except (TypeError, ValueError) as exc:
            raise diagnostics.error(f"long_term_memory: {exc}", range=source_range) from exc
        return MemoryPolicySpec(value, {})

    if type(value) is not dict:
        raise diagnostics.error(
            "long_term_memory must be a non-empty string or a mapping",
            range=source_range,
        )

    extras = sorted(
        (repr(key) for key in value if type(key) is not str or key not in {"name", "config"})
    )
    if extras:
        raise diagnostics.error(
            f"long_term_memory has unknown key(s) {extras} (allowed: name, config)",
            range=source_range,
        )
    if "name" not in value:
        raise diagnostics.error(
            "long_term_memory mapping requires 'name'",
            range=source_range,
        )
    name = value["name"]
    if type(name) is not str or not name:
        raise diagnostics.error(
            "long_term_memory name must be a non-empty string",
            range=source_range,
        )
    config = value.get("config", {})
    if type(config) is not dict:
        raise diagnostics.error(
            "long_term_memory config must be a mapping",
            range=source_range,
        )
    try:
        spec = MemoryPolicySpec(name, config)
        check_memory_declaration(name, name, config)
        return spec
    except (TypeError, ValueError) as exc:
        raise diagnostics.error(
            f"long_term_memory config: {exc}",
            range=source_range,
        ) from exc


def memory_policy_declarations(
    spec: MemoryPolicySpec | None,
    site: DeclSite,
) -> tuple[MemoryDecl, ...]:
    """Return the old-form policy as one declaration at ``site``, or no declaration."""
    if spec is None:
        return ()
    if type(spec) is not MemoryPolicySpec:
        raise TypeError("memory policy must be an exact MemoryPolicySpec")
    return (MemoryDecl(spec.name, spec.name, spec.config, site, spec.origin),)


__all__ = ["memory_policy_declarations", "parse_memory_declarations", "parse_memory_policy"]
