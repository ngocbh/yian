"""The Compose-YAML loader.

Reads Compose-shaped YAML flows into the engine's runtime model. The source side
of a node's `outputs:` / a flow's `inputs:` is read directly into one
`ValueContract` whose recursive `Type` (see `types.read_type`) replaces the old flat
`list[IOField]`; nested records are native.

Organized along two axes: the per-node-kind axis lives in `nodes/<kind>.py`
(each owns its descriptor + `parse` + `build(ctx) -> BuildResult` methods; `nodes/__init__.py`
derives the shared `KINDS` table from those classes), and the cross-node graph-stage axis lives
in the ordered phases `parse/ -> desugar/ -> edges/ -> validate/ -> build/`. The shared choke
points (`parse/nodes.py`, the loader's conductor loop, `nodes/bindings.py`) dispatch through
`KINDS` kind-blind — there is no per-kind isinstance ladder. `source/` owns the one-pass
YAML document, located cursors, and immutable occurrence-backed origin catalogs;
assembly stamps authored node headers, ordinary node-input keys and values, and
source-backed reserved MAP/WAIT wiring values on their canonical carriers. It creates one
`Assertion` per authored flow/node list position, deriving the carrier origin and temporary
range from the same retained item cursor; source-order indexing keeps duplicate text
distinct. Full roots and multi-node definitions retain flow assertion cursors and reuse the
lowered carriers on START/END. Compact roots and flat single-node definitions project their
assertions onto the synthesized node. Assembly also stamps source-backed bare/mapped root,
multi-node-definition, and inline-child output values on the final `FlowOutput.binding`
after rewrites; END reuses that binding, while output mapping keys stay origin-free.
Generated compact-root and single-node-definition reexports have no authored binding
occurrence and remain origin-free; their output declaration is reserved for contract-origin
work. Without a retained occurrence, Compose mints no origin and preserves any explicit
programmatic origin. Private assembly keeps the resolver unset while the shared catalog
grows. At the physical-file boundary the loader rebases external alias views, filters exact
reachable routes, and installs `ComposedFlow.origin_resolver`; source-free graphs use
`None`. `parse/file.py` consumes the document (`parse_file`).
The loader also owns each file's callable scope:
authored `defs:`/`uses:` plus a resolver-private, lazy cache for inline CALL/MAP/LOOP
children lowered by `desugar/` and assembled by `build/`.

Errors carry a concrete `SourceRange` (owned by `diagnostics` and temporarily
re-exported from `events`): a `SourceCursor` records each field's range during
`parse`, the runtime funnel resolves a failure's range from `node.field_ranges`,
and the CLI boxes it as a `Diagnostic`.

Imports flow DOWN only: this package imports `typesys` / `expr` / `nodes`; nothing
in the engine imports `compose`.
"""

from agent_composer.compose.build.boundary import synthesize_boundary_graph
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.nodes.resolver import (
    ChildResolver,
    ChildSignature,
    child_signature,
)
from agent_composer.compose.desugar.inline_calls import desugar_inline_calls
from agent_composer.compose.desugar.case_outputs import expand_case_outputs
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.loader import ComposedFlow, load_flow
from agent_composer.compose.parse.file import (
    AgentDescriptor,
    CallDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    ModelDescriptor,
    NodeDescriptor,
    ToolDescriptor,
    ComposeFile,
    parse_file,
)
from agent_composer.compose.validate.cycles import reject_cycles
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.validate.references import validate_references
from agent_composer.compose.helpers.types import read_flow_inputs, read_type
from agent_composer.compose.validate.handles import (
    check_case_handles,
)

__all__ = [
    "LoadError",
    "ComposeFile",
    "parse_file",
    "parse_nodes",
    "NodeDescriptor",
    "AgentDescriptor",
    "CodeDescriptor",
    "ModelDescriptor",
    "ToolDescriptor",
    "CaseDescriptor",
    "CallDescriptor",
    "read_flow_inputs",
    "read_type",
    "child_signature",
    "ChildSignature",
    "ChildResolver",
    "infer_data_edges",
    "synthesize_boundary_graph",
    "expand_case_outputs",
    "check_case_handles",
    "reject_cycles",
    "validate_references",
    "desugar_inline_calls",
    "ComposedFlow",
    "load_flow",
]
