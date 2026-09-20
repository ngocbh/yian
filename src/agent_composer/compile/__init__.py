"""Compiled flow IR, validation, origin traversal/rebasing, growth, and stable identity.

This package turns pure node-owned growth seeds into deterministic ``Flow``
fragments through ``expand.build_grow``. It also owns canonical manifests and
fingerprints, identity-safe opaque-origin traversal and source-neutral rebasing,
static execution scopes independent of memory policies, static memory-layout
derivation, and the complete static-path LLM configuration map used for replay.
Execution layouts derive root-relative template locations and initial LOOP drivers,
validating every target before stamping. Growth builders add structured occurrence
positions, child boundary associations, and source-neutral materialized seed recipes.
The shared model validates those recipes; expansion digests cover their topology
without adding derived state to authored fingerprints.

Knows about:   diagnostics, nodes, expr, typesys, and LLM configuration validation/merge.
Never imports: compose, runtime, the package-root run boundary, CLI, or provider SDKs.
"""

from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow
from agent_composer.compile.execution_layout import (
    apply_execution_layout,
    resolve_execution_layout,
)
from agent_composer.compile.origins import (
    iter_flow_origins,
    rebase_flow_origins,
    rewrite_flow_origins,
)

__all__ = [
    "END_ID",
    "START_ID",
    "Edge",
    "CompiledFlow",
    "apply_execution_layout",
    "resolve_execution_layout",
    "iter_flow_origins",
    "rebase_flow_origins",
    "rewrite_flow_origins",
]
