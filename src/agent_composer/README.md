# agent_composer — the engine

The implementation of **Agent Composer**: a from-scratch flow engine. A flow is
authored as **Docker-Compose-shaped YAML**, read by the `compose/` loader into a
compiled IR, and executed by the engine's own scheduler — there is no compilation
down to a third-party graph runtime.

The engine is, at heart, a **small functional programming language** — one whose
leaf computations may be LLM agents. If you know OCaml, you already know the
shape of it:

- A **flow is a function**: typed inputs in, typed outputs out, and nothing
  hidden. An **agent is just a flow** whose leaf computation happens to be an LLM
  loop — same contract, same composability.
- Flows **compose**. A node can be another flow, so a large transformation is
  built from small ones nested to any depth — function composition, not a
  monolith.
- Computation is **pure at the boundary**: a node returns an outcome and the
  engine binds it; a node never receives or mutates the shared pool. Each kind
  implements `forward`; the base `Node.run` template applies defaults and fires
  record-only asserts around that body. A flow's own asserts are lowered onto its
  synthesized START/END boundary nodes, so top-level and nested-flow asserts use
  the same ordinary node path. `Node.backward` is a dormant no-op reserved for a
  future lineage-aware reverse pass; current execution never calls it.
- The **program structure is fixed by the author**, as a function's call graph is
  fixed by its source. The LLM fills leaf boxes; it does not rewrite the graph at
  runtime. That structural referential transparency makes execution reproducible
  and makes an interrupted run recoverable from its authoritative record prefix.

## Layout

```
diagnostics.py semantic failures, resolved presentation, origins/resolver routing [leaf]
typesys/      typed value system (values) + variable pool                 [leaf]
events.py     transient run/node observer vocabulary; imports diagnostics
_state.py     shared scheduler-state vocabulary                          [leaf]
expr/         one ${...} grammar: parse -> AST -> evaluate / infer type / ref-walk
nodes/        pure Node contract + per-kind implementations; spawners return seeds
              (start, end, agent, code, model, tool, case, call, map,
               loop, human_input, wait)
compile/      compiled IR + validation; identity-safe origin traversal/rebasing;
              growth builder; stable identity; static-path LLM config maps
compose/      Compose-YAML front end: text -> CompiledFlow; source/ owns YAML
              construction/navigation and opaque-origin resolution; later phases
              lower file-private children
              (source/helpers/nodes/parse/desugar/edges/validate/build/loader)
runtime/      run records/logs, reducer-owned RunState, kind-blind dispatcher,
              and worker pool
run.py        public run boundary: ComposedFlow + inputs/log -> RunResult
llm_clients/  provider clients (lazy-imported) + LLMConfig + model_from_config
tools/        tool registry + process-owned ask_user interaction seam
cli/          `ac` commands + main-thread terminal broker
_settings.py  environment-based default provider/model
```

The main import direction is acyclic: `diagnostics` is the lowest
source-neutral leaf; `events` imports it, and nodes may combine it with the
independent `typesys` and `expr` leaves before `compile <- compose`. Runtime
consumes the event/state leaves plus compile's topology, identity, and validation
contracts, but never imports `compile.expand` or `compose`.
Compose imports `SourceRange` directly from `diagnostics`; `events` does not
re-export that front-end presentation coordinate.
The package facade lazy-loads its declared root exports, so importing a leaf
submodule does not bootstrap those higher layers.
Package-root `run.py` sits above `compose`, `compile.expand`, and `runtime`: it
consumes a `ComposedFlow`, injects the kind-aware growth builder, and owns run-log
creation and recovery. None of those packages imports it back.

`diagnostics.Failure` is the canonical source-neutral semantic failure value. It
is a frozen product of `code`, `message`, optional execution `node_id`, an
outermost-to-innermost `tuple[OriginRef, ...]` trace, optional Python
`cause_type`, and optional debugging `traceback`. The exception class is retained
only as diagnostic data; consumers must branch on the closed semantic code,
never on `cause_type` or traceback text. `Failure` is imported from the leaf
module inside lower layers and is lazy-exported from the package root for public
result consumers.

`diagnostics.Diagnostic` is the frozen, resolved presentation product:
`code`, `message`, ordered `DiagnosticFrame` values, free-standing `notes`, and
the optional `node_id`, `cause_type`, and `traceback` copied from a failure. It
has no execution meaning and owns no source-resolution behavior. For a
Compose-authored graph, `ComposedFlow.diagnose(failure)` is the sole front-end
seam that turns a `Failure` into this value. It reads only the failure and the
flow's `OriginResolver`, resolves the complete trace in its existing
outermost-to-innermost order, and preserves duplicate origins. Resolution is
atomic: an empty trace yields an ordinary unlocated diagnostic, while any
missing, incomplete, malformed, or raising resolver for a non-empty trace yields
no frames and exactly the generic note `Source presentation was unavailable.`;
semantic and debug fields remain unchanged.

Compose load failures use the same presentation product directly. A
`LoadError` contains exactly one `Diagnostic`; a Compose-owned factory resolves
retained document/cursor ranges into source-owning frames before the exception
crosses the front-end boundary. Definition and import wrappers create a new
diagnostic with outer context prepended. They never mutate an inner error or ask
the CLI to combine coordinates with separately supplied source text.

Every node builds failures through one source-neutral rule:
`Node.failure_trace(operation_origin=None)` returns its generated, outermost-to-
innermost `origin_context` followed by the supplied operation origin, or by the
node's own origin when no operation origin is supplied. Only an absent local
suffix is omitted; order and duplicates are preserved. Direct and static nodes
start with an empty context. The context is diagnostic-only generated ancestry,
not authored provenance: origin traversal and rewriting ignore it, and manifests,
fingerprints, and expansion digests exclude it.

Compile-owned CALL/MAP/LOOP expansion prefixes cloned children with the
spawner's complete `failure_trace()`. MAP's generated boundaries receive the
same context. A LOOP respawn instead copies only the driver's outer
`origin_context`, so the loop's own origin is appended exactly once when the
next driver builds a failure. Evaluator, dispatcher, and terminal-unreachable
failure producers remain kind-blind and call the same node-owned rule. Runtime
results expose only the canonical `Failure`; runtime and load failures both reach the same
`render_diagnostic()` through `ComposedFlow.diagnose()`. Missing or broken source
resolution therefore stays unlocated instead of falling through to runtime-id or
graph inference.

The closed codes are `binding.evaluate`, `contract.input`, `assertion.failed`,
`node.execute`, `contract.node_result`, `contract.output`, `contract.route`,
`contract.growth`, and `graph.terminal_unreachable`. They cover expected
flow-owned rejection. `EvaluationFailure` owns exactly one `Failure`. The evaluator
assigns its first five codes by
operation stage: expression/default/required binding, supplied-input contracts,
base-wrapper assertions, node execution, and result validation. Each producer
passes its best operation origin, if any, through `Node.failure_trace()`; the
node origin is the local fallback after any generated ancestry. Host control,
stream callbacks, unexpected binder faults, and malformed compiled/setup state
escape this expected-failure sum.

The dispatcher assigns the next three codes structurally from the closed result
sum: rejected `Output`, `Route`, and `Grow` data become `contract.output`,
`contract.route`, and `contract.growth`. Output rejection uses the effective
target contract origin selected by the rejected value's structural
`TypeCheckError.path` (exact site, nearest ancestor, then contract default), then
the executing node origin. Non-type canonicalization failures use the contract
default. Route and growth use the executing node origin. The trace is empty only
when generated ancestry is empty and neither the operation nor node fallback
supplies a local origin. Unsupported growth kinds and arbitrary builder, compiler,
spawner/fragment-copy, digest, topology-materialization, reducer, append, or log
faults escape rather than becoming flow failures.

At the schema-4 boundary, every trace item becomes an exact base
`OriginRef(key)`: front-end-specific subclasses are valid in memory, but class
identity is intentionally absent from the single-key wire. The reducer carries
that canonical immutable `Failure` through `NodeFailed`, terminal `RunFailed`,
and `RunResult.failure` without flattened aliases; live carriers share the same
object, while recovery reconstructs an equal canonical value. A route that first
skip-floods END latches
`graph.terminal_unreachable` with the causal router and its `failure_trace()`. That
`NodeRoutedRecord` persists the derived failure; replay structurally revalidates
the route and restores the recorded value, including its complete trace, even if
the supplied graph now carries rebased origins. No additional terminal record
is added, and an explicit node failure still supersedes the latch.

## Semantic IR carriers

`nodes/contracts.py` is the source-neutral vocabulary that joins opaque
provenance from `diagnostics` with semantic `typesys.Type` values. Each class has
one responsibility:

| Carrier | Meaning |
| --- | --- |
| `Binding` | one flow-owned input or output expression/literal |
| `Parameter` | one formal input, its optional contract, and default/coercion policy |
| `Assertion` | one executable expression plus stable author-facing display text |
| `ValueContract` | one recursive semantic `Type` plus nested origin selection |
| `PartialRecordContract` | one closed record shape with optional field-local value contracts |
| `ContractOrigins` | immutable root/nested origin lookup for a value contract |

For a refined specialized list, `Type.element.nullable` permits `None` alongside
the element's ordinary values; a bare or non-nullable specialized list still
rejects it. Existing `LIST_ANY` and explicit `None`-element semantics remain
unchanged. The nullable specialized list retains its `LIST_*` discriminator and
carrier through canonical value, pool, and run-record round-trips. Cross-flow
compatibility recursively applies the same nullable-source-to-strict-sink rule
to list elements. The singleton `None` source is compatible with every nullable
sink, matching runtime coercion, but remains incompatible with a strict non-`None`
sink.

All origins are optional, so Python-authored values default to source-free and
may supply an opaque origin explicitly. The carriers are frozen only at their
fields: arbitrary binding/default payloads are retained by identity.
`iter_origins()` visits only declared provenance slots, while `map_origins()`
transforms those slots without inspecting or copying semantic payloads.
`Flow` construction is the single binding boundary:
every direct `flow.wiring[node_id][parameter]` value and every
`FlowOutput.binding` is canonical `Binding`, while raw Python-authored values are
accepted and wrapped origin-free. Expansion rewrites only `Binding.source` and
retains `Binding.origin`; pre-IR Compose dictionaries and runtime-only overrides
remain raw. `FlowOutput.from_` is a temporary read-only source view during the
remaining migration.

`Node.origin` is the same source-neutral provenance shape for an authored node
context. Compose stamps the authored node-header occurrence for root, `defs:`,
and inline-body nodes; direct Python construction defaults to origin-free, and
loader/desugar-synthesized nodes remain so. Clones preserve an existing origin.
`Node.origin_context` separately carries generated outer-to-inner spawner
ancestry for nested failure traces; it is empty on direct/static nodes and is
neither enumerated nor rewritten as local authored provenance. Both fields are
diagnostic-only and excluded from manifests, fingerprints, and expansion
digests. A semantic `Node` carries no YAML `range`, `field_ranges`, or aligned
assertion-range lists. Compose retains concrete locations on cursors,
descriptors, and `BuildResult` only until its validators finish, then discards
them before constructing `CompiledFlow`.

CODE also owns two source-neutral diagnostic slots for its authored executable
field. `code_origin` is the whole-`code:` fallback, while
`code_line_origins` is an immutable tuple aligned with the constructed lines of
an exact literal block. Quoted and folded scalars carry only the whole fallback;
source-free Python construction defaults to `None` and `()`. Compose transfers
these origins from `CodeDescriptor` to `CodeNode`, and clones preserve them.
Inline CODE compiles the normal unpadded `def main(inputs):` wrapper once and
retains its recursive module/function/lambda/comprehension code-object tree.
`CodeNode.exception_origin()` walks an exception traceback outer-to-inner and
selects the deepest frame whose code object is owned by identity (`is`), never by
filename, equality, or hash membership. Wrapper line `n` selects authored slot
`n - 2`; an unavailable slot, wrapper-only frame, or post-body serialization
failure falls back to `code_origin`, then `Node.origin`. Reference mode uses the
whole `code_origin` as its invocation site, then the node fallback. Compose applies
the same line selector to load-time
`SyntaxError.lineno` and reports `SyntaxError.msg`.

CASE carries one optional origin on each ordered `Case`, identifying that exact
`cases[N].when` occurrence even when predicate text is duplicated. Compose
registers those indexed sites before lowering and preserves the origin on the
rewritten predicate; searched CASE also assigns each generated input `Binding`
the origin of the first predicate that introduced its reference, while `on:`
owns the generated `__on` binding origin. `CaseNode.exception_origin()` locates
its own active `forward` frame by code-object and node identity, then selects
the active `Case` by identity. Failures outside predicate evaluation fall back
to `Node.origin`. These origins participate in ordinary traversal and rebasing,
but remain diagnostic-only and manifest-neutral.

AGENT adds no provenance carrier for structured output: it reuses the existing
`ValueContract.origins` on `Node.output_contract`. After structured-generation
retries are exhausted, `AgentNode.exception_origin()` accepts a Pydantic error
or its directly wrapped native-parser cause only when tracebacks prove both the
exact native/fallback generator and the exact generated schema validation. It
does not traverse deeper cause chains. It normalizes the first error location by
leaving record fields unchanged and removing only the generated scalar `value`
or list `items` wrapper, validates that path against the expected `Type`, and
selects `ContractOrigins.at()` (exact, nearest ancestor, then contract root).
Unrelated tool/provider validation, non-validation errors, and source-free
contracts keep the node fallback. Retry behavior and the original exception are
unchanged.

`Node.pre_asserts` and `Node.post_asserts` are ordered `Assertion` lists. Parsed
YAML remains raw text until the inline-call pass crosses each flow assertion into
one carrier; node-local validation creates one fresh carrier per authored
occurrence. Compose indexes those occurrences by source order and derives the
carrier's opaque `OriginRef` from the retained item cursor, so equal assertion
text at two positions remains two distinct sites. Flow carrierization preserves
an explicit programmatic origin
instead of replacing it. Inline-call desugaring and flow/CALL lowering replace only
`.expression`; child cloning and LOOP respawn retain every carrier field. Thus
`.display` and `.origin` survive every rewrite. Synthesized START/END nodes reuse
the lowered flow carriers.
`Node.run` evaluates the expression, reports the display text, and diagnoses a
failure through that exact `Assertion.origin`. Manifests include expression,
display, order, and duplicate count while excluding origin.

`Node.params` has one compiled shape too: an ordered list of exact `Parameter`
carriers (or `None` for a directly constructed no-input node). The flow still owns each
parameter's `Binding`; the node owns the formal name, optional `ValueContract`,
required/default presence, and string-coercion policy. Binding validates stored
defaults without replacing their authored payload, and coerces supplied strings
only when the parameter opts in. A bound null shadows a default but still passes
through contract validation, so it is accepted only when the parameter is untyped
or its contract accepts null (a nullable or `None`-kind type).
`TypeCheckError.path` contains only contract-relative record names and actual list
indices. A supplied-value rejection resolves that path through `ContractOrigins`
(exact, nearest ancestor, then default) before falling back to the parameter
origin. An invalid authored default remains `binding.evaluate` and uses its
default origin rather than the value-contract site.
Parameter/default/contract origins do not affect manifest identity. Compose
`input:` lowers directly into this same exact carrier shape; `ComposedFlow.input`
and START reuse the same carriers without a parallel front-end declaration type.
`compose/helpers/types.py` is the one source-aware recursive type compiler. For
each authored root or `defs:` flow input, its key owns `Parameter.origin`, its type
value owns the `ContractOrigins` root and authored nested record fields,
and a packed `TYPE = default` owns `default_origin`. Scalar alias and list
type expressions have no interior YAML occurrence, so nested failures fall back to
the contract root. Python/source-free declarations and inferred inline-child
parameters keep empty origins; external aliases use ordinary lexical graph rebasing.

That declaration ownership is distinct from ordinary node-input wiring. Final
assembly puts the selecting-key origin on the sink `Parameter` and the bound-value
origin on its flow-owned `Binding`. Source-backed reserved wiring carries only the
value origin on its canonical `Binding`: MAP `over:` and WAIT `until:` have no
corresponding `Parameter`. A desugar may change `Binding.source` without moving that
value origin. When no source occurrence is retained, Compose mints no origin and
preserves any explicit programmatic origin; source-backed projections inherit their
backing site.
Source-backed bare and
mapped flow-output values on root flows, multi-node definitions, and inline children stamp
the final `FlowOutput.binding` after desugaring rewrites; the synthesized END boundary
reuses that exact binding. Output mapping keys carry no origin. Compact-root and
single-node-definition reexports are generated without an authored binding occurrence and
remain origin-free. That binding provenance is separate from node-output contract
provenance: a physical `output:` type declaration owns the root and authored nested sites
on `Node.output_contract`, including when compact/flat lowering generates the reexport.

`Node.output_contract` is the canonical compiled output declaration: an exact
`ValueContract`, a `PartialRecordContract`, or `None` when the output is unenforced.
The partial arm is a closed record: every field name is enforced, proven fields retain
their recursive contracts, and opaque fields retain presence without pretending their
type is `None`. Runtime success and replay validate the effective commit target through
this same sum. Exact contracts also drive node-owned structured generation. For AGENT,
only an untagged `str` opts out to free text; a tagged string (`Literal[...]`) uses the
generated scalar `value` wrapper and is unwrapped back to its bare tag. A live type
rejection resolves its contract-relative structural path through the matching exact or
field-local origins. The path is consumed before `Failure` construction and is not part
of the run-record wire. Manifests include the semantic exact/partial contract but omit
origins, so changing or rebasing provenance does not change flow or expansion identity.
Compose and runtime growth reference validation share the source-neutral closed-field
walk in `nodes/contracts.py`, so a partial record cannot expose a provably absent field.

Physical Compose declarations derive exact `ValueContract`s. Output binding derivation,
builders, child signatures, producer maps, and validators retain the complete
`OutputContract` sum; algorithms project an exact type only where one is required. The
shared `compose/helpers/output_contracts.py` preserves exact or partial whole-value
references, projects independently proven fields, assigns source-free contracts to
results proven by `expr.infer_expr_type`, and keeps unsupported computations opaque.
Multi-output assembly produces an exact record when every field is exact, otherwise a
closed `PartialRecordContract` with the proven siblings. `ChildSignature.output` follows
the terminal arity rule and CALL/LOOP preserve it. MAP derives `list[U]` only from an exact
child contract because the output-contract sum has no partial-list arm. Strict inline
input inference remains separate: it accepts literals, direct exact references, and
proven fields projected from partial records, while rejecting whole or opaque partial
reads. LOOP uses the same canonical carrier for proven seed checks at load and for every
zero- or later-iteration commit, replay, origin lookup, and expansion identity; no
transient field-evidence authority exists.
There is no parallel output-Type API: Python-authored nodes pass `output_contract=` explicitly.
For a physical node `output:` declaration, Compose injects a source-aware reader into the
descriptor's existing per-kind build position. The shared recursive compiler returns the
canonical contract with its declaration root and each authored nested record field; this
preserves kind-owned validation order without a parallel source map or second type walk.
Full and compact roots, multi-node and flat definitions, and inline children retain or
project the needed cursor. External aliases use ordinary lexical graph rebasing. Scalar
aliases and scalar `list[...]` expressions have no interior declaration, so a nested
failure falls back to the contract root. Generated declarations and Python-authored nodes
remain source-free.

### Compiled origin traversal and rebasing

Nodes own their provenance layout. The base `Node` enumerates and rewrites its
node origin, output contract, parameters, and assertions. CODE extends those
hooks with its whole-source fallback and ordered exact-line origins; CASE
extends them with its ordered predicate origins;
CALL/MAP/LOOP also own their cached `child_inputs` origins and expose the baked
child through `origin_children()`. These hooks remain source-neutral and never
import `compile`.

`compile.origins` joins those hooks with flow-owned wiring and declared outputs.
`iter_flow_origins()` walks wiring, outputs, nodes, and static children in stable
order, guarding flow, node, and frozen-carrier identity so shared or cyclic
graphs terminate. `rewrite_flow_origins()` rewrites an isolated working copy by
identity, then copies that finalized graph for return so payload dict/set hashes
are rebuilt after frozen carrier values change. Aliases, payload back-references,
and deliberate cycles survive while the authored graph remains unchanged. Binding sources,
parameter defaults, node environments, and render metadata are opaque even when
they contain an `OriginRef`. Origins remain excluded from manifests,
fingerprints, and expansion digests.

`diagnostics.ResolverTable` composes opaque namespaces without owning any source
format. Each public origin key routes to exactly one delegate resolver and one origin
in that delegate's namespace. Duplicate public keys are rejected; unknown
origins return `None`; delegate exceptions propagate; and malformed delegate
results are rejected. The table never parses keys or introduces resolver precedence.

`rebase_flow_origins(flow, mapper, resolver=...)` pairs a pure graph rewrite with
those exact routes. It preserves first structural-visit order while deduplicating
reachable old origins by their opaque key, calls `mapper` once per distinct old key,
and rejects collisions between distinct mapped keys before rewriting through
the precomputed map. The result is an independent graph of the same concrete type
plus a table containing only its reachable routes. An originful graph requires a
resolver; a source-free graph requires `None` and returns an empty table. The
operation defines no lexical key policy and knows nothing about Compose or YAML.

## One-pass source ownership

`compose/source/` is the only production Compose package that imports PyYAML.
Root text and each uncached external file are constructed once as
`SourceDocument`s, whose public cursor interface is immutable and defensive. They
retain the original text and label while keeping PyYAML's composed/constructed
graph private. YAML aliases and merge mappings are interpreted without mutating
that graph. Persistent pair/arc ropes keep merge topology compact, and completed
mapping summaries bound repeated merge-DAG traversal. Public value snapshots use
a stack-safe iterative copier that preserves shared and recursive aliases. Within
one successful file-import graph, the loader's resolved-path cache also ensures
that each external physical file is read, parsed, and compiled once; aliases and
diamond imports reuse that canonical `ComposedFlow` as the source for their
lexically rebased views.

Source-aware parsing and inline extraction navigate stable `SourceCursor` values.
A physical cursor combines a definition range with an occurrence path (and a
mapping-key range when applicable); a virtual cursor is a defensive, source-free
value snapshot; a projected cursor replaces or renames values while explicitly
retaining the physical sources of selected children. While the cursor is still
available, a descriptor may attach opaque origins for kind-owned semantic sites.
Cursors and descriptors retain concrete ranges through parsing and desugaring;
`NodeDescriptor._finish()` copies the descriptor's header and field locations
for the built node to Compose-only `BuildResult.node_range` / `field_ranges`.
Post-build validators
consume that metadata, and assembly drops it before `CompiledFlow`; the semantic
node receives only the opaque origins. No phase reparses text or touches parser
objects.
For an unchanged source-backed string, `SourceCursor.scalar_source` returns immutable
`ScalarSource` metadata: the whole scalar range and exact zero-based constructed-line
ranges when the scalar is a literal `|` block. The document derives those ranges from
the real block token during its existing parse. Other scalar styles expose the whole
range but no exact line tuple; virtual and value-changing cursors expose no scalar
provenance.

`SourceCatalog` is Compose's immutable document-local `OriginResolver`. Its private
builder maps typed semantic paths to opaque origins while retaining the physical
document occurrence and a key/value anchor or semantic constructed-line index.
Resolution replays that occurrence and reselects the current range; line coordinates
are never stored in the site identity. Unavailable exact line registration leaves the
site free for a whole-scalar fallback. Resolution neither reparses YAML nor creates a
parallel range map. Virtual or unanchored cursors produce no origin. One builder is shared by
the root flow, its `defs:`, and its inline bodies. Private assembly leaves the public
resolver unset; the loader freezes the builder once when the physical-file root is
complete.
After the common node-header origin is stamped, assembly invokes
`NodeDescriptor.with_source_origins(...)` on every normalized descriptor. The
default is an immutable no-op. `CodeDescriptor` uses the hook to register the
whole `code:` scalar and, when `ScalarSource` exposes an exact literal tuple,
each constructed line under the node's lexical site path. It stores only the
returned `OriginRef` values, never a cursor or coordinate map. The same path runs
for roots, definitions, and inline children; virtual source produces no origins.
`CaseDescriptor` uses the same hook for each indexed `cases[N].when` and optional
`on:` value, then transfers those origins to the ordered `Case` values and
lowering-generated `Binding` values.
`compose/helpers/types.py` owns the sole source-aware recursive type compiler.
Final assembly uses it for retained root/definition flow inputs: declaration keys
stamp `Parameter.origin`, type values and authored nested record fields
populate `ContractOrigins`, and packed defaults stamp `default_origin`. START reuses
those carriers. Scalar alias/list type expressions fall back to the contract root;
Python/source-free declarations and inferred inline parameters remain source-free.
At each output-authoring leaf's established build position, an injected Compose reader
uses the retained normalized node cursor to compile a physical `output:` declaration into
the same canonical `ValueContract` shape. Compact roots and flat definitions project their
top-level declaration onto the node; multi-node definitions and inline children retain
their own node cursors. External child views rebase the resulting root/nested origins.
Virtual/generated declarations and direct Python descriptor builds remain source-free.
Final assembly separately registers authored node headers, ordinary node-input keys
and bound values, source-backed reserved MAP/WAIT wiring values, CODE source fields,
CASE predicate/`on:` sites, assertions, and source-backed flow-output values directly
on their canonical semantic carriers; it does not build a parallel location map. An
ordinary node input's key
belongs to its sink `Parameter`, while its value belongs to `Binding`. An assertion's semantic
path includes its authored list index, and that item's cursor supplies
`Assertion.origin`. Runtime assertion failure diagnosis uses that carrier origin;
no positional assertion range crosses into the semantic node. Full root flows and
multi-node definitions retain a flow assertion cursor; their lowered carriers
are reused by START/END. Compact roots and flat single-node definitions instead
project their `asserts:` into the synthesized node, so those sites are
node-local. Flow outputs are stamped after rewrites on `FlowOutput.binding`, and
END reuses that binding; mapping keys are not origin sites. For sites without
retained occurrences, Compose mints no origin and preserves any explicit
programmatic origin. Generated compact-root and single-node-definition reexports
are such source-free binding sites even though their physical type declaration originates
the node's output contract. Runtime output validation selects the effective target
contract's exact/nearest/default origin from `TypeCheckError.path`, then falls back to the
executing node.
At each `uses:` alias, the loader validates the canonical child's complete resolver,
copies and rebases its compiled origin graph under a collision-free lexical key, and
caches that alias view. The physical resolved-path cache remains canonical and is never
mutated. Finalization filters the document catalog and alias tables to the exact reachable
origin set and installs one immutable `ResolverTable` as
`ComposedFlow.origin_resolver`; a source-free graph carries `None`. Repeated calls through
one alias reuse its view, while distinct aliases and diamond branches remain disjoint.

`ComposedFlow` stores no source-text or range snapshot. Its optional
`origin_resolver` is the sole presentation authority for runtime failures, while
load diagnostics are resolved before they leave Compose.

## Inline composite lowering

An authored CALL, MAP, or LOOP may contain a non-empty `nodes:` plus a required
`output:` instead of naming a child with `call:`. This is entirely a `compose/`
front-end feature. `desugar.inline_bodies` retains the located child body and
masks that occurrence as an ordinary private `call:` before the normal
parse/build/validate pipeline runs. YAML anchors, aliases, and merged mappings
are not mutated: extraction works on occurrence-local cursor projections while
keeping the original child cursors and source ranges read-only.

Each source file owns one callable scope. Alongside its authored `defs:` and
`uses:` namespace, that scope has a resolver-private table for retained inline
children. A child's stable identity is `__inline_flow_` plus a digest of its
complete lexical owner path (root or def, enclosing node, and nested inline
occurrences), not a process counter or object identity. Authored defs, aliases,
and call targets cannot use that reserved prefix, and author-facing diagnostics
show the lexical child label rather than the opaque digest.

Private child builders are registered during assembly and invoked lazily. The
file scope builds each callable at most once on successful resolution, shares
the same cache across private children/defs/uses, and guards private/def
construction in progress. The default file resolver's load context guards
cross-file recursion; an injected resolver is responsible for any equivalent
protection.
Generated child parameters come from the enclosing composite's `input:`
bindings. Their types are proved structurally; parent producer descriptors are
themselves built on demand through one success cache, and a MAP's `over:` element
is proved and cached only if a lone whole-value `${item...}` binding requests it.
Nested literal container strings are deliberately not scanned as references.

The result is an ordinary compiled child baked onto the existing CallNode,
MapNode, or LoopNode. No private resolver state reaches execution, no runtime
kind or graph-rewrite rule is added, and live execution/recovery continue through
the same kind-blind `Grow`/record/reducer path as named children.

## Node contract

A node is a pure function of its bound input record. It returns exactly one member
of the closed result sum:

| Result | Meaning |
|--------|---------|
| `Output(value, commit_as=None)` | one value for the engine to type-check and bind under the resolved success target |
| `Route(handle)` | one named outgoing branch, with redirect-aware sibling skip-flooding |
| `Grow(seed, prune)` | one canonical deterministic expansion description |

Success and absence resolve their logical output boundaries through distinct
carriers. Success resolves an explicit `Output.commit_as`, then the baked
`Node.commit_as`, then the physical node id. A skip cannot observe result-carried
`Output.commit_as`, so the reducer instead asks the pure, deterministic
`Node.skipped_output_target()` hook, which may consult only manifest-backed node
state. The base hook returns baked `commit_as` or the node id; LOOP returns its
manifest-backed `origin_id` to mirror its dynamic success redirect. A redirected
target has exactly one unfinished physical owner and must be an inert logical
anchor: an expanded spawner or an unknown placeholder with no incoming work, never a
ready, in-flight, or already-committed node; all of its outgoing edges must still
be unknown. The reducer marks the physical node skipped, deduplicates logical
targets, and floods the target's outgoing edges.

Failure is a `raise`, not another result variant. `eval_node` structurally turns
expected binding, assertion, body, and result rejection into
`EvaluationFailure(Failure)`. Result preparation constructs the same canonical
value for expected output, route, and growth rejection. Schema 4 carries that
value unchanged through the durable record, reducer, observer events, and public
result; the durable outcome may also retain optional source-neutral evaluation
evidence. Host control, stream callbacks, unexpected binder faults, malformed
setup, and arbitrary host/compiler faults escape. A streaming kind may yield
transient `StreamChunk` values and then return one of the same three results.

The runtime core is **kind-blind**. It dispatches on this result sum and consults
node-owned execution-semantic traits/hooks (`is_spawner`, `binds_per_item`,
`bind_reserved`, `validate_execution_setup`, `grow_depth_delta`, `is_loop`,
`needs_stop_event`, and `skipped_output_target`) plus the diagnostic-only
projections `exception_origin` and `failure_trace`, never on `NodeKind`. The
diagnostic projections do not affect program identity. The closed kind-aware
growth match lives in `compile.expand.build_grow`, outside runtime.
`compile.model.GrowthRejected` crosses that boundary only when a legal
CALL/MAP/LOOP kind supplies a malformed seed. Unsupported kinds and builder or
compiler helper faults remain raw; runtime marks only explicit seed, fragment,
budget, depth, and prune-policy rejection as expected growth data. The kind
census test keeps runtime's `NodeKind`/expansion dispatch count at zero.

## One append-only authority

Every run has exactly one `RunLog` and one matching reducer-owned `RunState`.
`MemoryRunLog` is the process-local implementation; `JsonlRunLog` is the durable
implementation, with a nonblocking exclusive advisory lock, canonical JSONL, and
an fsync per append. Both accept the same closed record vocabulary:

```
RunStartedRecord | RunResumedRecord | NodeStartedRecord |
NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord |
MemoryFrameLearningStartedRecord | MemoryFrameLearnedRecord |
MemoryFrameLearningFailedRecord | MemoryCommitRequestedRecord |
MemoryCommitSucceededRecord | MemoryCommitFailedRecord
```

The exact wire version is schema 4. Accepted node outcomes may retain optional
source-neutral evaluation evidence; `NodeFailedRecord` still owns one canonical
nested `Failure`. Earlier schemas are rejected rather than migrated. Terminal
success remains derived from the END value. The first
route whose skip closure reaches END persists its causal `Failure` on that
`NodeRoutedRecord`; replay revalidates the route before restoring it, and no
additional terminal record is written.

Runtime handles every transition in one order:

```
prepare and validate -> append -> apply reducer -> publish transient observers
```

Preparation cannot mutate live state. If append fails, its state transition is
not applied and its outcome observer is not published. `FlowEngine` is the sole
ready-frontier consumer and state writer; pooled workers only evaluate durably
started nodes and return raw results to the dispatcher. Stream chunks remain
callback-only and never enter the authoritative log.

The package-root `run_flow` boundary commits the complete initialization record
before lending its state/log pair to `FlowEngine`. A direct runtime caller has the
same obligation: apply `RunStartedRecord` to a matching `RunState` and `RunLog`
before driving the engine. The engine borrows this authority and never closes it;
the boundary or direct caller owns its lifetime.

## Durable recovery

`run_flow(..., log=path)` exclusively creates a new JSONL authority. Its first
record captures the run ID, authored-flow fingerprint, complete initial typed
pool, START wiring, worker count, and fully resolved LLM configuration. Expansion
records retain canonical growth seeds plus topology digests, so replay rebuilds
the same nested CALL/MAP/LOOP graph without re-running the spawner.

`resume_flow(flow, log=path)` locks an existing log, durably removes bytes after
the final complete newline, validates its canonical contiguous prefix, and checks
the supplied flow fingerprint. It folds historical records silently through the
same reducer and injected growth builder used live. A terminal prefix is returned
without a write. A nonterminal prefix gets exactly one `RunResumedRecord`, which
puts durably started nodes with no durable outcome back ahead of the ready
frontier before dispatch continues. Recovery uses the recorded input, run ID, and
complete LLM map rather than current ambient defaults.

This provides deterministic engine-state recovery, not exactly-once external
effects: an interrupted node with a durable start and no outcome is retried.

## Blocking effects and structured drain

Authorable blocking effects stay inside their node calls. An AGENT may invoke the
explicitly granted `ask_user` tool, HUMAN_INPUT always invokes the same host-owned
interaction seam, and WAIT blocks on its absolute UTC deadline. AGENT checks the
run-owned stop signal between model/tool effects; WAIT blocks on that signal. Each
returns `Output` from the same invocation.

CODE's soft-budget watchdog is package-local under `nodes/code/` and times CODE
only. Pooled drives own their executors through a full join: an abnormal dispatch
exit signals cooperative stop before joining, so the boundary never releases its
run-log lock while a worker can still produce an outcome.

## Dynamic growth and stable identity

A spawner node returns `Grow(seed, prune)` without importing `compile`. The
kind-aware `compile.expand.build_grow(spawner, seed)` converts the canonical seed
into a deterministic `Flow`; package-root `run.py` injects that callable into both
live execution and replay. Runtime validates and splices the fragment generically,
remaining kind-blind. During that compile-owned construction, CALL/MAP/LOOP
prefix child clones with `spawner.failure_trace()`; MAP also stamps its generated
boundaries, while a LOOP's next driver keeps only the enclosing
`origin_context`. Expansion digests cover entry/end IDs and topology but exclude
that diagnostic context, while loop pruning keeps only one dynamic iteration
resident at a time.

Compile also owns stable identity for the static program. `flow_manifest`
canonicalizes execution semantics, `flow_fingerprint` hashes that manifest, and
assertion identity includes executable and display text in list order while
remaining origin-neutral. `expansion_digest` applies the same projection to
dynamic fragments. CODE identity includes its executable `code` text but excludes
`code_origin`, `code_line_origins`, and the retained compiled/owned-code caches,
while CASE identity includes predicate handle/expression semantics but excludes
each predicate origin. Thus source formatting, origin rebasing, and Python object
identity cannot change program or expansion identity. `resolve_llm_cascade` returns a
complete map keyed by slash-separated static agent paths. The run boundary reads
process provider/model defaults once at fresh-run initialization;
`apply_effective_llm_map` restores the
recorded map during recovery without consulting the environment.
