"""The Compose-YAML loader.

Reads Compose-shaped YAML flows into the engine's runtime model. The source side
of a node's `outputs:` / a flow's `inputs:` is read directly into one recursive
`Type` (see `types.read_type`) rather than a flat `list[IOField]`; nested records
are native.

Organized along two axes: the per-node-kind axis lives in `nodes/<kind>.py`
(each owns its descriptor + `parse` + `build(ctx) -> BuildResult` methods; `nodes/__init__.py`
derives the shared `KINDS` table from those classes), and the cross-node graph-stage axis lives
in the ordered phases `parse/ -> desugar/ -> edges/ -> validate/ -> build/`. The shared choke
points (`parse/nodes.py`, the loader's conductor loop, `nodes/bindings.py`) dispatch through
`KINDS` kind-blind — there is no per-kind isinstance ladder. `parse/file.py` is the YAML
text/span front door (`parse_file`).

Errors carry a concrete `SourceRange` (from `events`): a located `NodeReader` records
each field's range during `parse`, the runtime funnel resolves a failure's range from
`node.field_ranges`, and the CLI boxes it as a `Diagnostic`.

Imports flow DOWN only: this package imports `typesys` / `expr` / `nodes`; nothing
in the engine imports `compose`.
"""

from agent_composer.compose.validate.asserts import AssertSet, classify_asserts
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
from agent_composer.compose.loader import LoadedFlow, load_flow
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
from agent_composer.compose.helpers.types import InputDecl, read_flow_inputs, read_type
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
    "InputDecl",
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
    "AssertSet",
    "classify_asserts",
    "LoadedFlow",
    "load_flow",
]
