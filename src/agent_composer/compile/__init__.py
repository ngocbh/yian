"""Compiled flow IR, validation, origin traversal/rebasing, growth, and stable identity.

This package turns pure node-owned growth seeds into deterministic ``Flow``
fragments through ``expand.build_grow``. It also owns canonical manifests and
fingerprints, identity-safe opaque-origin traversal and source-neutral rebasing,
static execution scopes independent of memory policies, static memory-layout
derivation, and the complete static-path LLM configuration map used for replay.
Execution layouts derive root-relative template locations and validate all targets
before stamping; runtime occurrence data belongs to growth preparation.

Knows about:   diagnostics, nodes, expr, typesys, and the LLM configuration merge primitive.
Never imports: compose, runtime, the package-root run boundary, CLI, or provider SDKs.
"""

from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow, NodeState
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
    "NodeState",
    "apply_execution_layout",
    "resolve_execution_layout",
    "iter_flow_origins",
    "rebase_flow_origins",
    "rewrite_flow_origins",
]
