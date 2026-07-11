"""The Compose-YAML loader.

Reads Compose-shaped YAML flows into the engine's runtime model. The source side
of a node's `outputs:` / a flow's `inputs:` is read directly into one recursive
`Type` (see `types.read_type`) rather than a flat `list[IOField]`; nested records
are native.

Imports flow DOWN only: this package imports `typesys` / `expr` / `nodes`; nothing
in the engine imports `compose`.
"""

from agent_composer.compose.asserts import AssertSet, classify_asserts
from agent_composer.compose.build import (
    ChildResolver,
    ChildSignature,
    build_call_node,
    build_leaf_node,
    child_signature,
    infer_data_edges,
    synthesize_boundary_graph,
    synthesize_roots,
)
from agent_composer.compose.stages.inline_calls import desugar_inline_calls
from agent_composer.compose.cases import (
    CaseDesugar,
    desugar_case,
    expand_case_outputs,
    reconcile_case_edges,
)
from agent_composer.compose.errors import LoadError
from agent_composer.compose.loader import LoadedFlow, load_flow
from agent_composer.compose.run import resume_command, resume_flow, run_flow
from agent_composer.compose.parser import (
    AgentDescriptor,
    CallDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    ModelDescriptor,
    NodeDescriptor,
    ToolDescriptor,
    ComposeFile,
    node_lines,
    parse_file,
    section_lines,
)
from agent_composer.compose.stages.cycles import reject_cycles
from agent_composer.compose.stages.parse import parse_nodes
from agent_composer.compose.types import InputDecl, read_flow_inputs, read_type
from agent_composer.compose.validate import (
    check_case_handles,
    validate_references,
)

__all__ = [
    "LoadError",
    "ComposeFile",
    "parse_file",
    "parse_nodes",
    "node_lines",
    "section_lines",
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
    "build_leaf_node",
    "build_call_node",
    "child_signature",
    "ChildSignature",
    "ChildResolver",
    "infer_data_edges",
    "synthesize_roots",
    "synthesize_boundary_graph",
    "desugar_case",
    "CaseDesugar",
    "expand_case_outputs",
    "reconcile_case_edges",
    "check_case_handles",
    "reject_cycles",
    "validate_references",
    "desugar_inline_calls",
    "AssertSet",
    "classify_asserts",
    "LoadedFlow",
    "load_flow",
    "run_flow",
    "resume_flow",
    "resume_command",
]
