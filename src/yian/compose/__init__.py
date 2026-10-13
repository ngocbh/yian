"""The Compose-YAML loader.

Reads Compose-shaped YAML flows into the engine's runtime model. One recursive
type reader turns a node's `output:` / a flow's `input:` into a `ValueContract`
whose `Type` represents scalars, containers, and nested records natively.
For retained root/definition flow signatures, the same traversal compiles the
parameter key, contract root/nested values, and packed default into opaque origins.
For authored node outputs, assembly injects an optional Compose-owned reader through
`BuildContext`; each authored-output leaf invokes it only at its existing contract
build step, after any earlier kind-owned validation. The contract default owns the
physical declaration root, nested record paths own their exact values, and
unrepresented scalar alias/list interiors fall back through the nearest authored
ancestor to that root. Compact roots project top-level `output:` onto the
synthesized node, and flat definitions use the equivalent source-aligned projection.
Source-free callers and generated contracts without an authorable type occurrence keep
empty origin carriers; external alias views use ordinary lexical graph rebasing.

Organized along two axes: the per-node-kind axis lives in `nodes/<kind>.py`
(each owns its descriptor + `parse` + immutable source-origin attachment +
`build(ctx) -> BuildResult` methods; `nodes/__init__.py`
derives the shared `KINDS` table from those classes), and the cross-node graph-stage axis lives
in the ordered phases `parse/ -> desugar/ -> edges/ -> validate/ -> build/`. The shared choke
points (`parse/nodes.py`, the loader's conductor loop, `nodes/bindings.py`) dispatch through
`KINDS` kind-blind — there is no per-kind isinstance ladder. `source/` owns the one-pass
YAML document, located cursors, and immutable occurrence-backed origin catalogs;
assembly stamps authored flow-input parameter/contract/default sites, node headers,
ordinary node-input keys and values, and
source-backed reserved MAP/WAIT wiring values on their canonical carriers. After each
common header stamp, assembly makes one kind-blind descriptor-hook call. CODE uses it to
carry the whole `code:` value plus ordered exact literal-line origins; CASE uses it for
indexed predicates and optional `on:` provenance; other descriptors use the no-op default.
Separately, its `BuildContext` contract reader selects
the normalized physical node `output:` cursor and delegates the same recursive contract
compiler from the leaf's established build position; a virtual/missing site falls back to
the source-free reader without moving validation order. It creates one
`Assertion` per authored flow/node list position, deriving the carrier origin and temporary
range from the same retained item cursor; source-order indexing keeps duplicate text
distinct. Full roots and multi-node definitions retain flow assertion cursors and reuse the
lowered carriers on START/END. Compact roots and flat single-node definitions project their
assertions onto the synthesized node. Assembly also stamps source-backed bare/mapped root,
multi-node-definition, and inline-child output values on the final `FlowOutput.binding`
after rewrites; END reuses that binding, while output mapping keys stay origin-free.
Generated compact-root and single-node-definition reexports have no authored binding
occurrence and remain origin-free. That binding rule is distinct from the authored
`output:` contract site: its synthesized leaf retains root/nested contract origins through
the compact or flat-definition projection.
Without a retained occurrence, Compose mints no origin and preserves any explicit
programmatic origin. Private assembly keeps the resolver unset while the shared catalog
grows. At the physical-file boundary the loader rebases external alias views, filters exact
reachable routes, and installs `ComposedFlow.origin_resolver`; source-free graphs use
`None`. `parse/file.py` consumes the document (`parse_file`).
`ComposedFlow.diagnose(failure)` is this front end's presentation boundary: it
copies `code`, `message`, `node_id`, `cause_type`, and `traceback` into a
canonical `diagnostics.Diagnostic` and resolves every opaque trace origin in
order through that installed authority.
Resolution is all-or-nothing, so unavailable source presentation never replaces
or partially decorates the original failure.
The loader also owns each file's callable scope:
authored `defs:`/`uses:` plus a resolver-private, lazy cache for inline CALL/MAP/LOOP
children lowered by `desugar/` and assembled by `build/`.

Load errors cross the package boundary only as `LoadError(Diagnostic)`: Compose
resolves each retained cursor/range against its owning document before raising,
and definition/import context becomes ordered outer frames. The CLI receives no
separate YAML text or coordinates for load failures. Runtime diagnosis reads only
the source-neutral origin trace and the installed resolver.

`skills.py` reads Codex/Claude `SKILL.md` folders into `yian.skills.Skill`
values for a run's host-supplied ``skills=`` mapping (`load_skills`,
`default_skill_roots`); frontmatter goes through the safe `parse_yaml_literal`.

Imports flow DOWN only: this package imports `typesys` / `expr` / `nodes` / `skills`;
nothing in the engine imports `compose`.
"""

from yian.compose.build.boundary import synthesize_boundary_graph
from yian.compose.edges.data import infer_data_edges
from yian.compose.nodes.resolver import (
    ChildResolver,
    ChildSignature,
    child_signature,
)
from yian.compose.desugar.inline_calls import desugar_inline_calls
from yian.compose.desugar.case_outputs import expand_case_outputs
from yian.compose.desugar.case_values import desugar_case_values
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import ComposedFlow, load_flow
from yian.compose.skills import default_skill_roots, load_skills
from yian.compose.parse.file import (
    AgentDescriptor,
    CallDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    NodeDescriptor,
    ToolDescriptor,
    ComposeFile,
    parse_file,
)
from yian.compose.validate.cycles import reject_cycles
from yian.compose.parse.nodes import parse_nodes
from yian.compose.validate.references import validate_references
from yian.compose.helpers.types import read_flow_inputs, read_type
from yian.compose.validate.handles import (
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
    "desugar_case_values",
    "check_case_handles",
    "reject_cycles",
    "validate_references",
    "desugar_inline_calls",
    "ComposedFlow",
    "load_flow",
    "load_skills",
    "default_skill_roots",
]
