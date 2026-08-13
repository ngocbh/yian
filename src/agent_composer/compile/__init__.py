"""Compiled flow IR, validation, growth construction, and stable identity.

This package turns pure node-owned growth seeds into deterministic ``Flow``
fragments through ``expand.build_grow``.  It also owns canonical manifests and
fingerprints plus the complete static-path LLM configuration map used for replay.

Knows about:   nodes, expr, typesys, and the LLM configuration merge primitive.
Never imports: compose, runtime, the package-root run boundary, CLI, or provider SDKs.
"""

from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow, NodeState

__all__ = ["END_ID", "START_ID", "Edge", "CompiledFlow", "NodeState"]
