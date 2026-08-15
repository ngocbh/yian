"""Pure node contract and per-kind implementations.

Nodes consume bound input records and return the closed ``Output | Route | Grow``
result sum. Spawners return only a canonical ``Grow`` seed (plus ids to
prune); they never construct compiled graph fragments themselves. A node may
carry an optional opaque authored-context origin; Python-authored and
programmatically generated nodes default to it being absent, while clones
preserve an existing value. Runtime does not interpret this provenance.

Knows about:   diagnostics, events, expr, typesys, tools, llm_clients, and peer
               node modules.
Never imports: compile, compose, runtime, the package-root run boundary, or CLI.
"""

from agent_composer.nodes.base import Grow, Node, NodeKind, NodeResult, Output, Route
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
    ValuePath,
    as_binding,
    binding_source,
)

__all__ = [
    "Node",
    "NodeKind",
    "NodeResult",
    "Output",
    "Route",
    "Grow",
    "Assertion",
    "Binding",
    "ContractOrigins",
    "Parameter",
    "ValueContract",
    "ValuePath",
    "as_binding",
    "binding_source",
]
