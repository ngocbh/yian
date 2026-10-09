"""Pure node contract and per-kind implementations.

Nodes consume bound input records and return the closed ``Output | Route | Grow``
result sum. Spawners return only a canonical ``Grow`` seed (plus ids to
prune); they never construct compiled graph fragments themselves. Frozen semantic
carriers enumerate and map only their declared origin slots. A node owns hooks
for its local origins and static children; higher layers may traverse them without
kind dispatch, while runtime does not interpret this provenance.
The base contract also owns detached backward context and feedback results;
execution scopes carry authored locations independently of memory declarations.

Knows about:   diagnostics, events, expr, memory contracts, typesys, tools,
               llm_clients, and peer node modules.
Never imports: compile, compose, runtime, the package-root run boundary, or CLI.
"""

from yian.nodes.base import (
    BackwardContext,
    BackwardResult,
    Grow,
    Node,
    NodeKind,
    NodeResult,
    Output,
    Route,
    validate_backward_result,
)
from yian.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    ExecutionPosition,
    ExecutionScope,
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
    "BackwardContext",
    "BackwardResult",
    "Assertion",
    "Binding",
    "ContractOrigins",
    "ExecutionPosition",
    "ExecutionScope",
    "OutputContract",
    "Parameter",
    "PartialRecordContract",
    "ValueContract",
    "ValuePath",
    "as_binding",
    "binding_source",
    "validate_backward_result",
]
