# Reference — package charter & layout

## Charter template (copy into a new `__init__.py`)

```python
"""<one line: what this package IS>.

<2–4 sentences: the single responsibility it owns, and why it's its own package.>

Knows about:   <lower-level/peer packages it may import, by name>
Never imports: <higher-level packages that import IT; plus forbidden deps>
"""

# re-exports only — no domain logic, no inline classes
from <pkg>.<submodule> import <PublicName>

__all__ = ["<PublicName>"]
```

## Real example (from this repo)

`src/yian/typesys/__init__.py` — the typed value system + variable pool,
an independent engine leaf (it imports nothing else in the engine; `nodes`,
`compile`, `compose`, and `runtime` import *it*, never the reverse):

```python
"""Typed runtime state: the typed value system + the variable pool."""

from yian.typesys.pool import VariablePool
from yian.typesys.values import TypedValue, build_value
# … (the full module re-exports the typed value + type surface)

__all__ = ["TypedValue", "VariablePool", "build_value"]
```

The worked top-level charter — with the full leaf-to-root layer diagram and the
"nothing here imports a DB or a server" boundary — is
`src/yian/__init__.py`. Read it to see the dependency direction written
down for the whole engine. It is also the deliberate facade exception: a
standard-library-only lazy export table prevents importing the lower
`diagnostics` submodule from bootstrapping compile, compose, or runtime while
preserving the declared package-root API.

## Where does it go? (one-glance table)

| The thing you're adding                          | Goes in                       |
|--------------------------------------------------|-------------------------------|
| Public behavior of one entity/kind               | that entity's own module      |
| Shared dataclass / base type / enum with no stronger domain name | `common.py` |
| Cohesive public contract vocabulary | a named module such as `contracts.py` |
| Pure stateless helper function                   | `utils.py`                    |
| A whole new distinct responsibility (>1 file)    | new package (write charter)   |
| Something an existing package already owns        | that package (don't make a new one) |

Kind-specific support follows the entity row: for example, API conversation
encoding belongs in `nodes/agent/memory.py`:
it adapts provider messages to lower memory contracts without storing history.
`memory/session.py` owns history, FIFO admission, and derived summary publication;
`memory/summary.py` owns pure whole-turn selection and cache-prefix proofs. The
API adapter injects projection and current-model callbacks, never stores summaries.
Session admission is per memory key: one conversation per session key, owned by
its single declaring AGENT.
Likewise `nodes/agent/live.py`, not `runtime/`, owns AGENT streaming and in-loop observers;
runtime only forwards opaque sinks by trait.
Runtime only binds authored scope and accepted invocation identity and manages
the lazy capability lifetime; it neither encodes provider messages nor
reconstructs history from the run log.

Optional persistence belongs below concrete memory backends in
`memory/storage/`: `SQLiteStorage` owns one local SQLite file and finite-JSON transactions,
not semantic snapshots, run handles, clients, or queues. Backends validate and
save candidates before publishing live state. Host-only storage paths never enter
YAML or runtime records, and the memory facade remains contracts-only.

Host construction belongs in package-root `memory_host.py` (`build_memories`,
`MemoryRegistry`), above `compile`'s memory table and `memory/` backends.
`cli/memory.py` only parses `--memory-root`/`--memory-storage KEY=PATH` and calls
it; `cli/run.py` owns the registry's lifetime. Runtime does not select or
construct implementations, and terminal inspection acquires no memory resources.

## Import-direction smell test

- Importing *upward* (leaf reaches into a root package) → wrong package, or needs a seam.
- Two packages import each other → extract shared contract to `common.py`/a leaf, or merge.
- `from x import y` written *inside a function* to dodge an error → hidden cycle; fix the structure.

## Dependency-inversion example: graph growth

Spawner nodes own the schema of a canonical `Grow(seed, prune)` but do not know how
to compile it. `compile.expand.build_grow` is the only kind-aware constructor, while
runtime knows only the injected `(spawner, seed) -> Flow` callable. Package-root
`run.py` wires them together. This keeps both forbidden edges absent:

- no module under `nodes/` imports `yian.compile`;
- no module under `runtime/` imports `yian.compile.expand` (imports of
  compile's execution contracts remain allowed).

`tests/engine/test_layer_direction.py` parses imports with `ast` and enforces these
rules, including equivalent relative-import spellings. Spawner seed schemas live
with their owning nodes; `runtime/common.py` owns the scheduling-state vocabulary.
Runtime owns the run-record/log codecs, append-before-apply reducer, and dispatcher,
while the package-root `run.py` owns log lifecycle and recovery.

## Ownership example: compiled origin traversal

Origin-bearing semantic carriers live in `nodes/contracts.py`; each enumerates
and maps only its declared provenance slots. The base `Node` owns common local
origins, CODE additionally owns its whole-source and exact-line origins, and
CALL/MAP/LOOP additionally own cached child-signature origins and expose their
baked child through `origin_children()`.

`diagnostics.ResolverTable` owns source-neutral namespace composition at the
lowest layer. Every public origin key has one exact delegate/old-origin route;
duplicate keys fail instead of gaining precedence, and keys remain opaque.

`compile/origins.py` owns the graph-wide operation because `compile` already owns
`Flow` wiring and `CompiledFlow.outputs`. It combines the node hooks with
identity guards, an isolated working copy whose provenance fields are rewritten
without replacing frozen graph objects, and a finalized return copy that rebuilds
payload hash containers. This preserves aliases, payload back-references, and
cycles without a `nodes -> compile` back-edge,
reflection, or `NodeKind` dispatch. Semantic payloads remain opaque, and origin
changes do not affect manifests or fingerprints.

Its `rebase_flow_origins` operation ordered-deduplicates reachable old origins by
opaque key, precomputes one collision-free mapped origin per old key, and returns the
same-type graph copy plus a `ResolverTable` containing exactly those routes. An
origin-free graph must carry no resolver. Neither layer defines lexical key
encoding; installing a complete returned-flow resolver remains higher front-end
work.

## Compose front-end ownership: source once, project thereafter

The Compose front end is an adapter above the representation-neutral compiler:

```
compose/source  <-  helpers  <-  nodes  <-  parse  <-  desugar  <-  edges/validate  <-  build  <-  loader
```

| Responsibility | Owner | Must not leak to |
|---|---|---|
| PyYAML construction and the private YAML node/occurrence graph | `compose/source` | every other compose phase, `compile`, `runtime` |
| Plain-scalar rules (YAML 1.2 core plus timestamps and merge keys), one loader class for documents, isolated literals (`parse_yaml_literal`), and one bare plain scalar (`read_plain_scalar`, the inline `call(...)` argument) | `compose/source/scalars.py` | PyYAML's global resolver/constructor tables, or a second resolver table elsewhere |
| Stable location-aware navigation, immutable projections, and whole/exact-literal provenance | public `SourceDocument` / `SourceCursor` / `ScalarSource` contracts from `compose/source` | private document slots, PyYAML nodes/tokens, or another parse |
| Immutable document-local origin resolution, semantic constructed-line selection, exact ownership, pre-finalization registered-site lookup, and Compose-private typed key/scoping policy | public `SourceCatalog` plus its package-private builder/encoder in `compose/source` | stored coordinates, parallel range maps, another YAML parse, parsing an opaque child key, per-node snapshots, or a premature cross-file resolver |
| Source-free and cursor-aware recursive type/contract compilation | `compose/helpers/types.py` | cursor/catalog/YAML concepts below Compose or a duplicated type-shape traversal |
| Template-aware exact/partial output-contract projection and closed-record assembly | `compose/helpers/output_contracts.py` | child resolution, graph-stage ownership, YAML/source documents, or runtime |
| `isinstance` type-name collection and resolution against one flow's `typedefs:` | `compose/helpers/isinstance_types.py` | the evaluator (which reads only `Node.types`), another flow's registry, or runtime |
| Union narrowing facts and per-node contract views from CASE branches | `compose/validate/narrowing.py` | runtime routing, node kinds, or a second field-read walk (rule checks stay in `nodes/contracts.py`) |
| Recognizing a binding's `Type = value` text by its YAML scalar style (plain is typed; quoted/block is plain text) | `compose/helpers/typed_bindings.py` | YAML parsing beyond the literal reader, value-against-type checks, or graph-stage policy |
| The fix text for a YAML 1.1 boolean spelling (`yes`, `off`) that reaches a `bool` slot, offered only when the boolean fits the slot's type (found inside a list or record by the type check's path); naming what YAML read a non-text value as (`yaml_read_as`, shared by the non-text key, prompt, and question-text errors) | `compose/helpers/yaml_words.py` | YAML parsing, or the error messages around the fix |
| Rejecting at load a typed binding that can never fit its declared type (literals checked exactly; references only when disjoint, narrower left to runtime) | `compose/validate/typed_bindings.py` | YAML/source navigation, runtime value checks, or recognition of `Type = value` text |
| Finding a literal value under a secret name (walker over nested mappings/lists) and wording its load error without the value; the `env:` check; used by helpers, nodes, desugar, and the loader | `compose/helpers/secret_literals.py`, with the one name list in the stdlib-only package-root leaf `secret_names.py` | a second secret-name list, the value in any message, or manifest redaction |
| The pre-pass that rejects a literal secret in node bindings, case values, and flow outputs while values are still as authored | `compose/validate/secret_literals.py`, imported only by `compose/build/pipeline.py` | the typed-binding split or desugar passes (which may echo a value), or runtime |
| Which nodes CASE routing may skip (`SkipModel`, mirroring the runtime join rule over compiled edges), null defaults for flow outputs whose producers may all be skipped, and rejecting arithmetic over a maybe-skipped value | `compose/validate/skips.py` (exhaustive CASEs from `compose/validate/case_exhaustive.py`) | runtime imports, a second join rule in `runtime/`, or template/comparison semantics |
| Semantic descriptors and flow sections | `compose/nodes` and `compose/parse` consuming a document/cursor | another YAML construction pass or retained parser object |
| Source-preserving inline extraction | `compose/desugar` projecting cursors | PyYAML and loader state |
| Compiled-flow assembly, per-occurrence carrier stamping, and temporary post-build locations | `compose/build` consuming descriptor `BuildResult`s and retained cursor sources | PyYAML, concrete loader implementation, or leaking temporary ranges into semantic nodes / `CompiledFlow` |
| Resolved-file reuse, cycles, and one file's callable namespace | `compose/loader` | `compile` and `runtime` |

Only modules under `compose/source` may import PyYAML or traverse the retained
parser graph. A text-accepting convenience in `compose/parse` must delegate the
single construction to `SourceDocument`. Source-aware parsing and inline extraction
navigate its cursors; descriptors and Compose-only `BuildResult.node_range` /
`field_ranges` carry derived locations, while semantic nodes carry only
source-neutral origins. Post-build validators consume the build-result metadata and
assembly drops it before `CompiledFlow`. No phase
copies, renders, or reparses YAML to reconstruct source.

The loader creates one package-private catalog builder per physical document and
threads it through root, definition, and inline-body assembly.
`compose/helpers/types.py` owns the sole source-aware recursive type compiler. For
retained root/definition flow inputs, declaration keys stamp `Parameter.origin`, type
values and authored nested record fields populate `ContractOrigins`, and
packed defaults stamp `default_origin`; START reuses those carriers. Scalar alias/list
type expressions fall back to the contract root. Python/source-free declarations and
inferred inline parameters remain source-free. For authored node outputs, the conductor
injects an optional Compose-owned reader through `BuildContext`. The base descriptor calls
it only from the output-authoring leaf's established contract step, preserving earlier
kind validation and error order. It selects the normalized physical `output:` cursor and
uses the same recursive compiler: the complete declaration owns the root/default, physical
record fields own exact nested paths, and scalar alias/list interiors use the nearest
authored ancestor or root fallback. Compact roots and flat definitions retain top-level
`output:` through source-aligned synthesized-node projections. Direct/virtual builds and
generated contracts without an authorable type occurrence stay source-free; external
aliases use existing lexical graph rebasing. The conductor separately stamps authored node-header origins, ordinary
node-input keys and bound values, and source-backed reserved MAP/WAIT wiring values directly
on their canonical carriers. It invokes one
kind-blind descriptor hook for extra semantic sites; CODE registers its whole scalar and
exact literal constructed lines without retaining a cursor below Compose. A CODE syntax
failure may resolve one already-registered site through the builder's read-only lookup; this
replays the retained document without copying, freezing, reparsing, or installing a resolver.
The conductor registers flow and node assertion sites by authored source-order index,
deriving each carrier origin from the retained item cursor. Duplicate text therefore
remains occurrence-distinct; full roots and multi-node definitions retain flow assertion
cursors and reuse their lowered carriers on START/END, while compact roots and flat
single-node definitions project assertions onto their synthesized node. Runtime assertion
diagnosis uses the failed `Assertion.origin`; semantic nodes retain no positional range
lists. The conductor also stamps
source-backed bare/mapped root, multi-node-definition, and inline-child output values on the
final `FlowOutput.binding` after rewrites; END reuses that binding, and output mapping keys
stay origin-free. Flow-input declaration keys and ordinary node-input keys belong to
`Parameter`; ordinary node-input, reserved, and flow-output bound values belong to `Binding`.
Generated compact-root and single-node-definition
reexports remain origin-free because no authored binding occurrence backs those generated
references. This `FlowOutput.binding` rule is distinct from their physical `output:`
node-contract site retained by the compact/flat projection. Without a retained
occurrence, Compose mints no origin and preserves any explicit programmatic origin. Private
assembly leaves resolver installation to the loader.
At the physical-file boundary it performs the only successful-load freeze of the shared local
catalog, combines exact reachable
routes with lexically rebased alias tables, and sets `ComposedFlow.origin_resolver`; source-free
graphs use `None`.
The returned flow stores no parallel source text, label, or range snapshot;
runtime presentation reaches authored documents only through that resolver.

The loader owns two intentionally separate caches:

- `_LoadCtx.flows` is keyed by resolved absolute path, shared across one external
  load graph, and populated only after a child has compiled successfully. It
  makes aliases and diamond imports reuse the same canonical, fully resolved
  physical flow.
- `_CallableScope._built` is keyed by file-local callable name and memoizes every
  successful compiler-private child, authored `defs:` flow, and resolved `uses:`
  alias view. A `uses:` view is a rebased copy; repeated calls through that alias
  reuse it without mutating the path cache. Defs load eagerly before the parent,
  and uses resolve eagerly during assembly; only compiler-private builders stay
  lazy. `_in_progress` rejects
  recursive private/def construction, while `_LoadCtx.loading` guards recursive
  external-file resolution. A failed callable attempt removes every private builder,
  nested cache entry, and catalog site added by that attempt before it can be retried.

`compile` owns representation-neutral flow contracts and origin traversal, while
`runtime` executes compiled flows. Neither layer imports Compose source
contracts, knows that YAML was the authoring format, or interprets loader-private
names.

## Front-end ownership example: inline composite children

Inline CALL/MAP/LOOP bodies do not earn a runtime package or a new compile
abstraction. Their responsibilities follow the existing compose phases:

| Responsibility | Owner | Must not know |
|---|---|---|
| Located, occurrence-local `SourceCursor` projection and stable lexical private identity | `compose/desugar/inline_bodies.py` | PyYAML/private source graph, later `build`, concrete loader scope, runtime |
| Strict generated input-type proof | `compose/build/inline_inputs.py` | loader state, runtime |
| Association, lazy parent-descriptor dependencies, child assembly | `compose/build/pipeline.py` | concrete loader implementation |
| File-private callable table, one-build cache, recursion guard | `compose/loader.py` | runtime |

`build.pipeline` accepts a narrow registrar/resolver protocol; the higher loader
supplies its concrete file scope. That seam prevents the tempting `build ->
loader` back-edge. Once lowered, the child is an ordinary compiled CALL/MAP/LOOP
child, so neither `compile` nor `runtime` needs to know the private prefix or
authored inline shape.
