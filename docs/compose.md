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
  ├─ root SourceCursor     → occurrence-aware values, paths, and source ranges
  ├─ parse_file            → the strict top-level ComposeFile
  ├─ read_typedefs         → the type registry (typedefs:)
  ├─ read_flow_inputs      → canonical Parameter values for input:
  ├─ parse_nodes           → one frozen DESCRIPTOR per node (per-kind, source-ordered)
  ├─ stamp authored headers → optional Node origins in one document-local catalog
  │
  ├─ desugar_inline_calls        ┐  rewrite author sugar into core nodes,
  ├─ desugar_adaptive_questions  │  minting synthetic nodes where needed.
  ├─ desugar_case_call_targets   │  After this, only core kinds remain.
  ├─ expand_case_outputs         ┘
  │
  ├─ build (per kind, via KINDS) → a runtime NODE + its flow-owned WIRING
  │                                (a case lowers in its own build, returning its edges)
  ├─ stamp node wiring       → ordinary Parameter key origins + ordinary/reserved
  │                             final Binding value origins
  │
  ├─ infer_data_edges       → read every ${...} → draw the edges (nobody draws them by hand);
  │                            append each case's own data+control edges from its build
  ├─ validate_references    → every ${...} resolves & type-checks; classify asserts
  ├─ stamp flow outputs     → source-backed bare/mapped values on final
  │                            FlowOutput bindings (mapping keys stay origin-free)
  ├─ synthesize boundary    → START/END nodes, edges, and binding-preserving wiring
  ├─ reject_cycles          → the graph is a DAG
  ├─ check_case_handles     → every branch has a home
  ▼
 CompiledFlow   (returned inside a ComposedFlow: + parameters, asserts, source frame)
```

Everything below just explains three of those words: **descriptor**, **build**, and
**edge inference**.

---

## One document, many source views

`load_flow()` constructs each flow source document as one `SourceDocument`. That single
safe construction keeps the semantic value and a compact private definition graph for
YAML anchors, aliases, duplicate keys, and `<<:` merges. Source-aware parsing and inline
extraction navigate `SourceCursor`s into that document; later passes use descriptors and
nodes carrying derived ranges. No phase parses the flow text again.

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

The public source model is immutable: documents keep parser objects private, virtual and
projected values are snapshotted, and cursor value reads use a stack-safe iterative copy
while preserving alias and recursive-alias identity within each copy. Inline extraction
takes one whole-nodes snapshot before projection so unchanged alias siblings stay shared.

`SourceCatalog` is Compose's immutable implementation of `OriginResolver`. Its private
builder maps deterministic typed semantic paths to opaque origins and retains each physical
document occurrence plus an explicit key/value anchor. Resolution replays that occurrence
into a `DiagnosticFrame`; virtual or unanchored cursors produce no origin, and no YAML is
reparsed or parallel range map created. The loader shares one builder across a physical
document's root, `defs:`, and inline bodies. Assembly registers authored node headers,
ordinary node-input keys and values, source-backed reserved MAP/WAIT wiring values, and
source-backed flow-output values on the canonical carriers themselves. Flow-output
registration happens after desugaring, so an authored occurrence locates the final
`FlowOutput.binding` source. Output mapping keys stay origin-free. Without a retained
source occurrence, Compose mints no origin and preserves any explicit programmatic origin.
Each assembled `ComposedFlow` receives a frozen snapshot as private
`_local_catalog` staging metadata. It is deliberately not a public, complete resolver yet:
cached external-child origins still need lexical rebasing.

External `uses:` resolution adds one reuse boundary. During a load graph, successfully
compiled external flows are cached by resolved absolute path. A repeated alias or diamond
dependency therefore reuses the same `ComposedFlow`; it does not reread, reparse, or
recompile that physical file. Only successfully compiled flows enter the cache.

---

## Descriptors vs. nodes

The pipeline has two representations of a node, and the split matters:

| | **Descriptor** | **Node** |
|---|---|---|
| what | a frozen record of *what you wrote* | the runtime object the engine calls |
| where | `compose/nodes/<kind>.py` | `nodes/<kind>/` |
| carries | raw fields + ranges + optional header origin | `run()`, contracts, optional origin |
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
  - **`build(ctx: BuildContext) -> BuildResult`** — a method that constructs the runtime
    `Node` + its flow-owned wiring (and, for a `case`, its own data+control edges). `ctx`
    carries the shared build inputs (`registry`, `resolver`, inline-CODE source hints).

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

`build()` produces a `BuildResult` bundling **three** things, and the split is deliberate:

```python
result = desc.build(BuildContext(registry=registry, resolver=resolver))
node, wiring, edges = result.node, result.wiring, result.edges
```

- the **node** carries only what it needs to run: its canonical `output_contract`,
  its sink `params` (formal names plus optional input contracts/default policy), and
  an optional source-neutral authored-context `origin`. It also retains the legacy
  render-only `range`/`field_ranges` traceback bridge. Descriptors derive a
  `ValueContract` directly; consumers retain it and project `.type` only inside
  semantic type algorithms.
- the **wiring** is *flow-owned*: `{param → "${source}"}`. The node knows its parameter
  *names*; the flow knows where each one is *sourced from*. This is why a node is
  reusable and why the flow, not the node, owns the graph.
- the **edges** are empty (`()`) for every kind EXCEPT `case`, which lowers to a
  `CaseNode` in its own `build()` and returns its data+control branch edges here.

Leaf kinds (`agent`/`code`/`model`/`tool`) ignore `ctx.resolver`; `call`/`map`/`loop` use
it to resolve the child flow they expand into.

After all descriptor desugars and builds, the conductor pairs each source-backed ordinary
`input:` entry with those two carriers: its selecting key becomes `Parameter.origin`, and
its value becomes `Binding.origin`. It also stamps source-backed reserved MAP `over:` and
WAIT `until:` values on their canonical `Binding`; those fields are not formal parameters,
so they have no `Parameter.origin`. The final (possibly rewritten) payload stays in
`Binding.source`, so lowering does not detach provenance from the authored value. Entries
without a retained source occurrence receive no new Compose origin; an explicit
programmatic origin remains intact, and source-backed projections inherit their backing
occurrence. CASE wiring, synthesized START/END boundary parameters, assertions, contracts,
external rebasing, and runtime diagnostic consumption remain
separate migration steps.

Source-backed bare and mapped output values on root flows, multi-node `defs:`, and inline
children stamp the final `FlowOutput.binding` after output rewrites. The synthesized END
boundary reuses that exact binding. Output mapping keys do not carry origins. Compact root
flows and single-node defs generate a reexport rather than retaining an authored binding
occurrence, so that generated binding stays origin-free; their authored `output:` value is
the output contract and is reserved for separate contract-origin work.

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
  synthesized `start`/`end` boundaries),
- `input` — the canonical typed `Parameter` values. Authored defaults are
  validated at load; supplied values are checked when the boundary binds them,
- `asserts` — flow-level pre/post checks,
- `source` — a render-only `SourceFrame` (the text + per-node ranges) that lets a
  runtime failure draw a caret back at your YAML, even across nested flow calls.

Internally, `_local_catalog` retains the immutable document-local origin snapshot used
while the source-neutral IR migration proceeds. The underscore is intentional: it cannot
resolve origins embedded from external files until lexical rebasing is implemented.

Hand it to [`run_flow()`](engine.md) and the interpreter takes over.

---

## Errors: one type, always located

Every load failure is a single `LoadError` carrying a `SourceRange`. Each pass raises it
at the most precise anchor it has — a bad field at that field's range, an unresolved
`${...}` at the binding, a cycle at the node header. That is why the CLI can box the
offending line instead of dumping a stack trace. When a pass genuinely lacks an interior
line (e.g. a type error from the line-less `typesys` layer), it anchors at the enclosing
section header rather than guessing.

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
