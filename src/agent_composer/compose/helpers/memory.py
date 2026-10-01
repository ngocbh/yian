"""Source-aware validation for authored long-term-memory declarations."""

from __future__ import annotations

from typing import Any

from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.diagnostics import SourceRange
from agent_composer.memory.contracts import DeclSite, MemoryDecl, MemoryPolicySpec


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
        return MemoryPolicySpec(name, config)
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


__all__ = ["memory_policy_declarations", "parse_memory_policy"]
