# compose — how YAML becomes a `CompiledFlow`

> The front end. `compose/` is the **only** package that knows YAML exists. It takes
> flow text and lowers it to a `CompiledFlow` — the typed, frozen graph the
> [engine](engine.md) runs. Everything domain- or syntax-specific lives here so the
> layers below stay generic. For the authoring surface (what you *write*), see
> [Flow syntax](syntax.md); this page is how that surface is *compiled*.

## The shape of the phase

One function, `load_flow()`, drives a fixed pipeline. Read it like a compiler:
**parse → desugar → build → wire → check → freeze.**

```
 text
  │
  ├─ SourceDocument.parse  → one safe construction + a private YAML definition graph
  ├─ root SourceCursor     → occurrence-aware values, paths, ranges, and scalar provenance
  ├─ parse_file            → the strict top-level ComposeFile
  ├─ read_typedefs         → the type registry (typedefs:)
  ├─ read_flow_inputs      → canonical Parameter values + recursive authored
  │                          contract origins for root/def input:
  ├─ parse_nodes           → one frozen DESCRIPTOR per node (per-kind, source-ordered)
  ├─ stamp authored headers → optional Node origins in one document-local catalog
  ├─ stamp kind-owned sites → descriptor hook; CODE scalar/lines + CASE predicates/`on:`
  │
  ├─ desugar_inline_calls        ┐  rewrite author sugar into core nodes and
  │                               │  carrierize flow assertions by source order.
  ├─ desugar_adaptive_questions  │  Mint synthetic nodes where needed.
  ├─ desugar_case_call_targets   │  After this, only core kinds remain.
  ├─ expand_case_outputs         ┘
  │
  ├─ build (per kind, via KINDS) → a runtime NODE + its flow-owned WIRING/EDGES;
  │                                Compose-only node/field ranges stay on BuildResult
  │                                (physical output contracts compile origins here)
  ├─ stamp node wiring       → ordinary Parameter key origins + ordinary/reserved
  │                             final Binding value origins
  │
  ├─ infer_data_edges       → read every ${...} → draw the edges (nobody draws them by hand);
  │                            append each case's own data+control edges from its build
  ├─ validate_references    → every ${...} resolves and type-checks
  ├─ stamp node assertions → validate/classify one originated carrier per list position
  ├─ stamp flow outputs     → source-backed bare/mapped values on final
  │                            FlowOutput bindings (mapping keys stay origin-free)
  ├─ synthesize boundary    → START/END nodes, edges, and binding-preserving wiring
  ├─ reject_cycles          → the graph is a DAG
  ├─ check_case_handles     → every branch has a home
  ├─ discard build locations → no SourceRange enters Node or CompiledFlow
  ▼
 CompiledFlow   (inside a ComposedFlow: + parameters, exact origin resolver)
```

Everything below just explains three of those words: **descriptor**, **build**, and
**edge inference**.

---

## One document, many source views

`load_flow()` constructs each flow source document as one `SourceDocument`. That single
safe construction keeps the semantic value and a compact private definition graph for
YAML anchors, aliases, duplicate keys, and `<<:` merges. Source-aware parsing and inline
extraction navigate `SourceCursor`s into that document; later Compose passes use descriptors
and `BuildResult` values carrying derived ranges. Those ranges are discarded before
`CompiledFlow`. No phase parses the flow text again.

Merge topology stays compact as identity-shared pair/arc ropes. Parent mappings reuse
completed child summaries instead of rescanning inherited merge DAGs; explicit raw
occurrence enumeration alone is output-sensitive.

A cursor separates *what was defined* from *where it was used*. Its `range` points to the
value's definition, while its occurrence path records the route from the document root.
Two aliases therefore share a definition range but have different paths. A mapping alias
also has its own selecting-key range; a sequence alias has no key range. Flattened merge
entries likewise keep distinct logical occurrences while effective mapping lookup follows
PyYAML's constructed-value semantics.

There are three cursor forms:

- A **physical cursor** retains its `SourceDocument`, definition range, and occurrence
  path; mapping children additionally retain their selecting-key range.
- A **virtual cursor** carries a generated value with no source location.
- A **projected cursor** presents a rewritten mapping or sequence while explicitly
  retaining physical sources for the authored children. Generated children stay virtual.

An unchanged source-backed string scalar also exposes an immutable `ScalarSource`: its whole
range plus an optional zero-based tuple of constructed-line ranges. The tuple is exact for
literal `|` blocks, including chomping, explicit indentation, blank lines, and normalized
line breaks. Folded, quoted, and plain scalars explicitly provide no exact line tuple, so a
consumer can use the whole range instead. Virtual and value-changing projected cursors
provide no scalar provenance. The ranges come from the real block token during the same
construction pass; anchors or tags before `|` therefore cannot shift the body mapping.

The public source model is immutable: documents keep parser objects private, virtual and
projected values are snapshotted, and cursor value reads use a stack-safe iterative copy
while preserving alias and recursive-alias identity within each copy. Inline extraction
takes one whole-nodes snapshot before projection so unchanged alias siblings stay shared.

`SourceCatalog` is Compose's immutable implementation of `OriginResolver`. Its private
builder maps deterministic typed semantic paths to opaque origins and retains each physical
document occurrence plus a key/value anchor or semantic constructed-line index. Resolution
replays that occurrence and selects its current range; coordinates never enter the opaque
key. When exact scalar-line provenance is unavailable, line registration returns no origin
so the consumer can register the whole scalar instead. No YAML is reparsed and no parallel
range map is created. The loader shares one builder across a physical
document's root, `defs:`, and inline bodies. Assembly registers authored node headers,
root/def flow-input keys and recursive type values, physical node-output contract roots and
nested fields, ordinary node-input keys and values, source-backed reserved MAP/WAIT wiring
values, CODE source fields, indexed CASE predicates and `on:`, and source-backed
flow-output values on the canonical carriers themselves. A packed
flow-input default receives its own semantic key, even though it resolves to the same scalar
range as the contract root. After the common header is stamped, assembly also calls each
descriptor's immutable
`with_source_origins(...)` hook. The default is a no-op; CODE registers its whole `code:`
scalar and every available exact literal constructed line, then carries only those opaque
origins into `CodeNode`. CASE registers each positional `cases[N].when` and optional `on:`
value, retaining only opaque origins through its lowering. It also indexes flow and node
assertions by authored list order,
deriving each `Assertion.origin` from its retained item cursor. Equal text at two positions
is therefore two distinct source sites. Flow-output
registration happens after desugaring, so an authored occurrence locates the final
`FlowOutput.binding` source. Output mapping keys stay origin-free. Without a retained source
occurrence, Compose mints no origin and preserves any explicit programmatic origin.
Private assembly products leave their resolver unset while the shared physical-document
catalog is still growing. At the file boundary, the loader freezes that catalog once,
selects exactly one authority for every reachable origin, and installs an immutable
`ResolverTable` as `ComposedFlow.origin_resolver`. Source-free graphs use `None`; unused
defs and eagerly resolved but uncalled aliases contribute no public routes.

External `uses:` resolution adds one reuse boundary. During a load graph, successfully
compiled external flows are cached by resolved absolute path. A repeated alias or diamond
dependency therefore reuses the same canonical `ComposedFlow`; it does not reread,
reparse, or recompile that physical file. The file-local callable scope validates that
canonical child's resolver and creates one immutable, lexically rebased view per `uses:`
alias before CALL/MAP/LOOP bake the child graph. Repeated calls through one alias reuse its
view; distinct aliases and diamond branches receive disjoint origin keys. Only successfully
compiled, complete flows enter the physical cache.

---

## Descriptors vs. nodes

The pipeline has two representations of a node, and the split matters:

| | **Descriptor** | **Node** |
|---|---|---|
| what | a frozen record of *what you wrote* | the runtime object the engine calls |
| where | `compose/nodes/<kind>.py` | `nodes/<kind>/` |
| carries | raw fields + ranges + optional header/kind origins | `run()`, contracts, optional header/kind origins; no YAML ranges |
| lifetime | load-time only | lives through the whole run |

`parse_nodes` produces descriptors; the desugar passes rewrite descriptors; `build`
turns each surviving descriptor into a node. Keeping them separate means the messy,
YAML-shaped, source-located data stays in `compose/`, and the node the engine sees is
clean and generic.

---

## The two-axis layout

`compose/` is split along two axes so a change stays local:

- **Per node kind** → `compose/nodes/<kind>.py`. Each module owns a frozen **descriptor**
  dataclass (`CodeDescriptor`) that subclasses `NodeDescriptor`. The base holds the common
  contract; each kind carries:
  - **`parse(node_id, cursor)`** — a classmethod (the base template) that reads one node body,
    recording each field's `SourceRange`,
  - **`with_source_origins(cursor, ...)`** — an immutable hook for kind-owned semantic
    source sites; the default returns the descriptor unchanged,
  - **`build(ctx: BuildContext) -> BuildResult`** — a method that constructs the runtime
    `Node` + its flow-owned wiring (and, for a `case`, its own data+control edges). `ctx`
    carries the shared build inputs (`registry`, `resolver`, the optional physical
    node-output contract reader, and inline-CODE source hints).

  `compose/nodes/__init__.py` derives the `KINDS` table from a single `_DESCRIPTORS` tuple —
  each kind maps its `kind` string to its descriptor class:
  ```python
  _DESCRIPTORS = (agent.AgentDescriptor, code.CodeDescriptor, ...)
  KINDS = {d.kind: d for d in _DESCRIPTORS}   # kind string -> descriptor class
  ```
  Adding or changing a kind is one file plus one entry here. The shared dispatch sites
  (`parse/`, `build/pipeline.py::_assemble`) look the class up in `KINDS`
  and call `cls.parse(...)` / `desc.build(...)` — they never grow an `if kind == …` ladder.

- **Cross-node graph passes** → the ordered phase packages `compose/parse/ →
  compose/desugar/ → compose/edges/ → compose/validate/ → compose/build/`. These operate on
  the *whole* set of descriptors: `parse_nodes`, the desugars, `infer_data_edges`/
  `infer_ordering_edges`, `validate_references`, `reject_cycles`, wiring-parity, boundary
  synthesis, and the rest — each phase may import only the earlier ones.

A sibling table, `BINDING_FIELDS`, records which descriptor fields carry `${...}`
strings (a code node's `inputs`, a tool's `inputs`, a map's `over`). One kind-blind walk
in `bindings.py` reads it, so the desugar passes rewrite bindings without an isinstance
ladder.

---

## Build: descriptor → node + wiring

`build()` produces a `BuildResult` with three semantic products plus Compose-only
location metadata:

```python
result = desc.build(BuildContext(registry=registry, resolver=resolver))
node, wiring, edges = result.node, result.wiring, result.edges
node_range, field_ranges = result.node_range, result.field_ranges
```

- the **node** carries only what it needs to run: its canonical `output_contract`,
  its sink `params` (formal names plus optional input contracts/default policy), and
  an optional source-neutral authored-context `origin`. It has no `range`,
  `field_ranges`, `pre_assert_ranges`, or `post_assert_ranges`. An output-authoring leaf
  descriptor derives an exact `ValueContract`; composite CALL/LOOP builders may preserve
  a child's `PartialRecordContract`. Consumers retain the output-contract sum and project
  an exact type only inside semantic type algorithms. CODE additionally carries the diagnostic-only whole-source
  `code_origin` and ordered exact `code_line_origins`. Inline CODE compiles one
  unpadded wrapper and uses retained code-object identity to select the deepest
  owned traceback line; filenames and structural equality never establish ownership.
  CASE additionally carries one diagnostic-only origin on each ordered `Case`.
  `CaseNode.exception_origin()` selects the active predicate by exact `forward`,
  node, and `Case` identity, then falls back to the node origin.
- the **wiring** is *flow-owned*: `{param → "${source}"}`. The node knows its parameter
  *names*; the flow knows where each one is *sourced from*. This is why a node is
  reusable and why the flow, not the node, owns the graph.
- the **edges** are empty (`()`) for every kind EXCEPT `case`, which lowers to a
  `CaseNode` in its own `build()` and returns its data+control branch edges here.
- **`node_range` / `field_ranges`** remain in Compose so post-build validators can
  locate errors after a descriptor has produced its semantic node. Assembly drops
  both before `CompiledFlow.from_parts`; compile and runtime never receive them.

Leaf kinds (`agent`/`code`/`model`/`tool`) ignore `ctx.resolver`; `call`/`map`/`loop` use
it to resolve the child flow they expand into. CODE alone also reads
`ctx.source_file` as an opaque Python traceback label and the optional
`ctx.resolve_origin` callable when presenting a load-time syntax failure. Assembly
injects the current file catalog builder's registered-site lookup after every CODE
origin is attached. That lookup replays the already-parsed document without freezing;
the loader remains the single successful-load catalog freeze.

Each output-authoring leaf calls the descriptor's common output compiler at its established
position inside `build()`, after any earlier kind-owned validation. For a physical
declaration, the injected `BuildContext.output_contract_reader` selects the retained,
normalized `output:` cursor and the shared recursive type reader returns one
`ValueContract` carrying its root and nested field origins. Full and compact roots,
multi-node and flat `defs:`, and inline children all reach that path through their retained
or projected cursors. External `uses:` views rebase the resulting carrier with the rest of
the child graph. A virtual/generated declaration or a direct Python descriptor falls back
to the source-free reader. Scalar aliases and scalar `list[...]` expressions expose only
the contract root.

After all descriptor desugars and builds, the conductor pairs each source-backed ordinary
`input:` entry with those two carriers: its selecting key becomes `Parameter.origin`, and
its value becomes `Binding.origin`. It also stamps source-backed reserved MAP `over:` and
WAIT `until:` values on their canonical `Binding`; those fields are not formal parameters,
so they have no `Parameter.origin`. The final (possibly rewritten) payload stays in
`Binding.source`, so lowering does not detach provenance from the authored value. Entries
without a retained source occurrence receive no new Compose origin; an explicit
programmatic origin remains intact, and source-backed projections inherit their backing
occurrence. CASE lowering assigns each searched generated `Binding` the first predicate
origin that introduced its reference; the `on:` origin belongs to the generated `__on`
binding, while synthetic `Parameter` values stay source-free. Ordered predicate origins
and these bindings enter ordinary clone and external-rebase paths through existing carrier
hooks. CODE's extra origins enter the same paths through its node-owned traversal/rewrite hooks.
The same CODE-owned wrapped-line selector serves runtime tracebacks and Compose
`SyntaxError.lineno`: exact literal line, whole scalar, then node header. Quoted and
folded code has no exact line sites and therefore selects the whole scalar.

The source-aware flow-input reader separately compiles each root/def input declaration.
The selecting key becomes `Parameter.origin`; the type value becomes the contract-root
origin; and recursively authored record fields become nested
contract origins. A packed default receives a distinct `default_origin` key at that same scalar
range. Boundary failures select an exact contract path, then an authored ancestor, the
contract root, and finally the parameter key. Scalar aliases and scalar `list[...]`
spellings expose only a root site. Generated or Python-built contracts and inferred inline
signatures stay source-free, while external aliases use the existing origin-rebasing path.
Physical node `output:` declarations use that same recursive compiler at their existing
per-kind build position. Their canonical `Node.output_contract` owns the declaration root
and authored nested record sites; output validation resolves `TypeCheckError.path` through
those origins before falling back to the executing node. Compact roots, both definition
forms, inline children, and external aliases preserve the same contract provenance, while
generated and Python-built contracts stay source-free.

Authored assertions cross into one ordered `Assertion` carrier per source occurrence.
Flow assertions are carrierized during inline-call desugaring; node assertions are
carrierized during validation. For physical YAML entries, assembly uses the retained
list-item cursor as the single source of the opaque `OriginRef`; flow carrierization
preserves an explicit programmatic origin rather than replacing it. Inline-call desugaring
and flow lowering change only the executable expression, while
child cloning, CALL relocation, and LOOP respawn retain the carrier's display and origin.
When an assertion fails, `Node.run` diagnoses through that exact `Assertion.origin`;
there is no positional range list on the node.
Lowered flow carriers are reused on START/END. Full root flows and multi-node `defs:` retain a flow
assertion cursor; compact roots and flat single-node defs project `asserts:` onto their
synthesized node, so those assertions are node-local.

Source-backed bare and mapped output values on root flows, multi-node `defs:`, and inline
children stamp the final `FlowOutput.binding` after output rewrites. The synthesized END
boundary reuses that exact binding. Output mapping keys do not carry origins. Compact root
flows and single-node defs generate a reexport rather than retaining an authored binding
occurrence, so that generated binding stays origin-free; their authored `output:` value is
still the physical node-output contract and carries its root/nested origins. These are
different carriers: `FlowOutput.binding` locates the reexport expression, while
`Node.output_contract` locates the type declaration.

---

## Edges are inferred, never drawn

You never write an edge. You write a reference:

```yaml
verdict:
  input:
    for_case: ${for_case.output}      # <- this line IS the edge for_case -> verdict
    against_case: ${against_case.output}
```

`infer_data_edges` (in `edges/data.py`) ref-walks every binding, reads out the node ids each
`${...}` depends on, and adds exactly those edges. The graph is a
*consequence* of the data references — it cannot drift out of sync with them, and
parallelism is automatic: two nodes that don't reference each other simply have no edge
between them, so the engine runs them concurrently. (`depends_on`/`runs_after` add
ordering-only edges for the rare case with no data dependency.)

The same `${...}` grammar is *evaluated* at run time and *ref-walked* here at load
time — one grammar, two readers. See [Expressions](expressions.md).

---

## Desugaring: sugar in, core out

Several authoring conveniences are lowered to core nodes before `build`, so the rest of
the pipeline only ever sees a small set of kinds:

| You write | Lowered to |
|---|---|
| inline `${ f(x) }` in a binding | a synthetic `call` node; the binding rewritten to `${synth.output}` |
| `human_input` with `adaptive_questions:` | a synthetic composer `agent` + the rewritten gate |
| `case` branch `then: ${child}` | a synthetic `call` node as the branch target |
| `${some_case.output}` | a coalesce over the case's branch targets |

Because desugaring is a descriptor-level rewrite, the synthetic nodes flow through
`build`, edge inference, validation, and the DAG check exactly like hand-written ones —
no special path downstream. They remain origin-free because no authored node header
corresponds to them; authored nodes inside retained inline bodies are stamped normally.

---

## What you get back

`load_flow()` returns a **`ComposedFlow`**:

- `compiled` — the `CompiledFlow` (immutable topology: nodes, inferred edges, wiring,
  synthesized `start`/`end` boundaries, and ordered node/boundary assertion carriers),
- `input` — the canonical typed `Parameter` values. Authored defaults are
  validated at load; supplied values are checked when the boundary binds them,
- `origin_resolver` — an immutable source-neutral resolver whose exact domain is the
  distinct origins reachable from `compiled`; `None` only for a source-free graph.
- `diagnose(failure)` — the presentation-only boundary that resolves the complete
  ordered failure trace through that authority into a canonical `Diagnostic`.

Diagnosis is atomic: an unresolved non-empty trace yields no partial frames and
retains the failure's code, message, node, and debug fields in an unlocated
diagnostic. `ComposedFlow` retains no parallel source-text or range snapshot.

External physical files remain canonical in the resolved-path cache. Their alias views
rewrite origin keys on copies and retain exact resolver routes back through the canonical
child, so the returned parent owns one complete namespace without mutating or reparsing a
cached graph.

Hand it to [`run_flow()`](engine.md) and the interpreter takes over.

---

## Errors: one canonical diagnostic

Every load failure is one `LoadError(Diagnostic)`. The diagnostic owns its message,
notes, and ordered `DiagnosticFrame` values; each frame contains the source label,
complete text, and exact `SourceRange` it indexes. A bad imported child therefore keeps
the child's YAML frame instead of applying child coordinates to its parent file.
Definition and import sites prepend outer context without mutating the inner failure,
and a cycle uses authored-order peer frames. Source-free Python composition remains a
valid unlocated diagnostic. No load-error renderer reparses YAML or receives source text
separately, and no parallel line-map or section-line fallback is threaded through later
Compose phases.

---

## Where to look in the code

| You want to change… | Go to |
|---|---|
| a node kind's fields or how it builds | `compose/nodes/<kind>.py` |
| a whole-graph pass (edges, validation, cycles) | the phase packages `compose/{parse,desugar,edges,validate,build}/` |
| the one-pass YAML source document | `compose/source/document.py` (`SourceDocument`) |
| physical, virtual, or projected source navigation | `compose/source/cursor.py` (`SourceCursor`) |
| opaque Compose origin resolution | `compose/source/catalog.py` (`SourceCatalog`) |
| strict top-level section parsing | `parse/file.py` (`parse_file`) |
| the pipeline wiring itself | `build/pipeline.py` (`_assemble`), driven by `loader.py` |
| running the compiled flow | [engine.md](engine.md) |
