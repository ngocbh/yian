"""Pure node contract and per-kind implementations.

Nodes consume bound input records and return the closed ``Output | Route | Grow``
result sum. Spawners return only a canonical ``Grow`` seed (plus ids to
prune); they never construct compiled graph fragments themselves. Frozen semantic
carriers enumerate and map only their declared origin slots. A node owns hooks
for its local origins and static children; higher layers may traverse them without
kind dispatch, while runtime does not interpret this provenance.

Knows about:   diagnostics, events, expr, typesys, tools, llm_clients, and peer
               node modules.
Never imports: compile, compose, runtime, the package-root run boundary, or CLI.
"""

from agent_composer.nodes.base import Grow, Node, NodeKind, NodeResult, Output, Route
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    OutputContract,
    Parameter,
    PartialRecordContract,
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
    "OutputContract",
    "Parameter",
    "PartialRecordContract",
    "ValueContract",
    "ValuePath",
    "as_binding",
    "binding_source",
]
