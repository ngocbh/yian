# Unified Failures, Source Origins, and Architecture Maps

**Date:** 2026-08-15

**Issue:** [#110](https://github.com/ngocbh/agent-composer/issues/110)

**Related issues:** [#81](https://github.com/ngocbh/agent-composer/issues/81),
[#86](https://github.com/ngocbh/agent-composer/issues/86),
[#91](https://github.com/ngocbh/agent-composer/issues/91),
[#100](https://github.com/ngocbh/agent-composer/issues/100), and
[#103](https://github.com/ngocbh/agent-composer/issues/103)

**Status:** Final design after two adversarial review rounds

## Changes from the previous version

This revision keeps the approved single-key `OriginRef`, compose-only YAML
ownership, and canonical `Failure`, and closes the gaps found by the first
adversarial review:

- defines deterministic origin-key encoding plus pure catalog/IR rebasing for
  cached external children, repeated aliases, diamond imports, and custom
  resolvers;
- rebases native and third-party origins as opaque atomic segments behind one
  source-neutral `OriginResolver` protocol;
- replaces proposed parallel origin maps with canonical `Binding`, `Assertion`,
  `Parameter`, and `ValueContract` semantic carriers, deleting the overlapping
  `InputDecl`/`ParamDecl` pair;
- defines occurrence-path semantics for an immutable one-parse YAML document,
  including duplicate keys, merges, aliases, normalized names, and overlays;
- defines `LoadError(Diagnostic)` as the final load-time exception contract;
- treats terminal-unreachable as a deterministic failure derived from its causal
  success/route record, rather than inventing a second run-log record;
- closes the initial failure-code vocabulary and separates expected flow-data
  rejection from host/compiler faults;
- specifies inline CODE line-origin behavior;
- makes the clean 0.0.1 public API and run-record schema cutovers explicit;
- selects Mermaid, with pinned rendering and CI validation, for maintained OOD
  and end-to-end run-flow diagrams.

## Goal

Create one understandable path from authored source to a successful or failed
run:

```text
YAML text -> one SourceDocument -> semantic IR -> records/reducer
          -> RunResult(output | Failure) -> Diagnostic -> renderer
```

Every class has one role and every fact has one authority:

1. `SourceDocument` is the immutable, single parse of one physical YAML file.
2. Semantic IR values own optional opaque `OriginRef`s at the operation that can
   fail.
3. `Failure` is the durable semantic value for an uncaught flow failure.
4. `Diagnostic` is the resolved, presentation-only value rendered for a human.

The engine remains usable without YAML. A future Python front end can build the
same semantic IR with no origins, or provide its own opaque origins and resolver.

## Non-goals

- Author-visible retry, fallback, catch, default-on-error, or failure branches.
- Changing the node-result sum `Output | Route | Grow`.
- Giving `compile/` or `runtime/` YAML paths, YAML parsers, source text, or
  rendering behavior.
- Preserving schema-1 JSONL failure records. There are no existing run logs.
- Preserving the current flattened Python failure fields. Version 0.0.1 makes a
  clean cutover to `.failure`; no delegated compatibility aliases remain.
- Designing the complete future Python graph-authoring API.

## Ownership and dependency direction

The final class ownership is deliberately narrow:

| Owner | Classes/contracts | Meaning |
| --- | --- | --- |
| `diagnostics.py` | `SourceRange`, `OriginRef`, `Failure`, `DiagnosticFrame`, `Diagnostic`, `OriginResolver` | Dependency-free semantic and presentation values; no YAML and no engine imports. |
| `compose/source/` | `SourceDocument`, `SourceCursor`, `SourceSite`, `SourceCatalog` | Parse and navigate YAML once; register and resolve opaque origins. |
| `nodes/` | `Binding`, `Parameter`, `Assertion`, `ValueContract`, `ContractOrigins`, `Node` | Source-neutral executable operations and their optional provenance. |
| `compile/` | `Flow`, `CompiledFlow`, generic origin traversal/rewrite | Static topology and representation-neutral graph transformations. |
| `runtime/` | worker outcomes, run records, reducer plans/state, engine | Record-authoritative execution; carries `Failure`, never resolves it. |
| `compose.ComposedFlow` | compiled graph plus an `OriginResolver` | Front-end wrapper and `diagnose(failure)` seam. |
| package `run.py` | `RunResult` and run/resume composition | Converts terminal runtime state to one public result. |
| `cli/` | diagnostic renderer | Terminal formatting only; no graph/source inference. |

The dependency ladder becomes:

```text
diagnostics <- events <- typesys <- nodes <- compile <- compose
                                     ^          ^
                                     |          |
                                  runtime    package run.py -> CLI
```

`runtime/` may consume compile topology contracts but never `compile.expand` or
`compose`. Only `compose/source/` imports PyYAML. Package charters and the layer
ratchet must encode those boundaries.

## Dependency-free diagnostics contracts

### SourceRange and OriginRef

`SourceRange` moves out of `events.py`; events are events, not source ownership.
`OriginRef` stays in the same leaf module:

```python
@dataclass(frozen=True)
class OriginRef:
    key: str

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key:
            raise ValueError("origin key must be a non-empty string")
```

The key is opaque outside the producing front end. It is stable and unique in
one returned `ComposedFlow`, is serialized unchanged, and is excluded from the
executable flow fingerprint and expansion digest.

Compose internally builds a semantic path of string and integer segments. Its
private encoder distinguishes strings from indices and escapes `~` and `/` using
JSON-Pointer escaping:

```text
string segment x/y -> s:x~1y
integer segment 0  -> i:0
path                -> s:root/s:nodes/s:score/s:input/s:x
```

This prevents collisions between a key named `"0"` and index `0`, or between
adversarial node/def/alias names. The spelling is tested but is not a public path
API.

### Failure

```python
@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    node_id: str | None
    trace: tuple[OriginRef, ...] = ()
    cause_type: str | None = None
    traceback: str | None = None
```

`node_id` is execution identity. `trace` is authored identity, ordered outermost
to innermost, with the last reference naming the precise operation. Neither is a
substitute for the other.

The initial closed code vocabulary is:

| Code | Meaning |
| --- | --- |
| `binding.evaluate` | A binding expression/default/required operation could not be evaluated. |
| `contract.input` | A supplied node or flow input violates its declared contract. |
| `assertion.failed` | An authored assertion evaluated false or could not be evaluated under the current assertion semantics. |
| `node.execute` | A node implementation raised while executing its body. |
| `contract.node_result` | A node returned a value outside `Output | Route | Grow`. |
| `contract.output` | A committed node value violates the effective target contract. |
| `contract.route` | Node-produced route data is invalid. |
| `contract.growth` | Node-produced growth data or an expected growth constraint is invalid. |
| `graph.terminal_unreachable` | Quiescence was reached without a required terminal value. |

`cause_type` keeps the Python exception class without making it the semantic
category. `traceback` is optional debugging data, never parsed for behavior.

### Diagnostic and LoadError

```python
@dataclass(frozen=True)
class DiagnosticFrame:
    label: str
    text: str
    range: SourceRange
    note: str | None = None

@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    frames: tuple[DiagnosticFrame, ...] = ()
    notes: tuple[str, ...] = ()
    node_id: str | None = None
    cause_type: str | None = None
    traceback: str | None = None
```

`Diagnostic` is resolved presentation data. It is neither stored in run records
nor used by the reducer.

The leaf also defines the authoring-format-neutral resolution seam:

```python
class OriginResolver(Protocol):
    def resolve(self, origin: OriginRef) -> DiagnosticFrame | None: ...
```

`SourceCatalog` implements this protocol for YAML. A future Python authoring
front end may resolve an opaque key to a Python file or notebook cell without a
runtime change. `ComposedFlow` stores an optional `OriginResolver`, not a YAML
catalog type. With no resolver, diagnosis returns the unlocated semantic fields.

Loading can fail before a `ComposedFlow` exists, so its final public exception is:

```python
class LoadError(ValueError):
    diagnostic: Diagnostic
```

`LoadError` accepts exactly one `Diagnostic`; `str(error)` returns its message.
It has no `line`, `lines`, `range`, `locator`, or parallel notes channel.
Cursor/type/reference errors construct a located diagnostic immediately.
Malformed YAML constructs a frame directly from PyYAML's problem mark. A nested
external load prepends its call/import frame to the existing diagnostic. Cycles
use ordered frames plus notes. Pydantic `loc` values are translated through a
cursor before the exception crosses the compose boundary.

The same renderer consumes load-time `error.diagnostic` and runtime
`loaded.diagnose(result.failure)`.

## One immutable parse per physical YAML file

### SourceDocument

```python
SourceDocument.parse(text, label)
    -> constructed Python value
    -> one retained YAML node tree
    -> occurrence-path index
```

One `SpanLoader` invocation first composes the node tree and then constructs the
value against those same nodes. PyYAML may flatten merge keys while constructing;
after `SourceDocument` is created, its value, tree, and indexes are treated as
read-only. No later pass mutates key nodes, mapping values, or shared aliases.

"Parsed once" means one full-document parse of each physical source stream.
Parsing an already-authored scalar default substring as a literal is not a
second physical-document load, but it is separately instrumented so a future
change cannot accidentally feed the full source through PyYAML again.

`_LoadCtx` separates two caches:

- resolved physical path -> `SourceDocument`;
- resolved physical path -> canonical locally-scoped `ComposedFlow`.

The root caller-provided text creates one document. Every uncached external file
is read once and creates one document. Defs and inline bodies receive cursors into
their owning document and never parse text.

### Occurrence paths and SourceCursor

An occurrence path is private compose data. Mapping steps retain both the
constructed key and pair ordinal; sequence steps retain their index. This gives
two appearances of one aliased YAML node distinct occurrence identities even
when both display the anchor definition's range.

`SourceCursor` is an immutable view containing a document, occurrence path, and
optional virtual name/value overlay. Its rules are:

- duplicate mapping keys select the last effective pair, matching current
  behavior; ignored earlier pairs remain indexable for load diagnostics;
- merge keys use PyYAML's existing flattened precedence; a local/last value wins;
- aliases have occurrence-local paths but intentionally resolve to their anchor
  definition range;
- surface `input`/`output` can be exposed virtually as canonical
  `inputs`/`outputs` without changing the document;
- compact flows, single-node defs, CASE's YAML-1.1 `on` normalization, and inline
  child bodies use virtual cursors/overlays;
- traversal has an ancestry guard, so recursive aliases cannot recurse forever.

Range lookup is occurrence-path based, not `id(yaml_node)` based. Multiple
occurrence paths may intentionally point to the same definition-site range.

## SourceCatalog and deterministic child composition

`SourceCatalog` is an immutable mapping from one `OriginRef` to one `SourceSite`.
A site retains the document reference, occurrence path, and display label needed
to create a `DiagnosticFrame`. The catalog—not the compiled graph—retains source
text.

Origins are initially compiled in a local scope rooted at `root`:

- top-level sites: `root / ...`;
- defs: `root / defs / <name> / ...`;
- lexical inline bodies: their authored cursor path under the owning node;
- synthetic operations: the one authored site that caused the operation, never
  a generated private node ID.

External composition is a pure copy-and-rebase operation:

1. The file cache retains a canonical child with local `root/...` origins.
2. Resolving `uses: reviewer:` obtains that cached value without reparsing.
3. Compose copies its compiled IR and maps each old key as one **opaque atomic
   segment** under the lexical prefix. It never parses the old key:

   ```text
   scope(prefix, OriginRef(old_key))
     = OriginRef(encode(prefix + ("origin", old_key)))
   ```

   The ordinary typed-segment escaping therefore handles both compose-generated
   and arbitrary third-party keys without collision.
4. A rebased resolver maps each new key back to its exact old `OriginRef` and
   delegates to the child's `OriginResolver`. A compose `SourceCatalog` can share
   its documents; a custom resolver remains opaque.
5. The cached child is never mutated.

The operation recurses over already-rebased transitive children, so diamond
imports remain distinct by lexical import path. Two aliases of the same file get
two namespaces and may resolve to the same underlying `SourceDocument` and
definition-site ranges. Repeated calls through one alias reuse the same origins;
their execution IDs and outer callsite trace frames distinguish invocations.

A custom `child_resolver` must return either:

- a source-free `ComposedFlow` whose reachable IR contains no origins; or
- a `ComposedFlow` with an `OriginResolver` that resolves every reachable origin.

`compose/source/catalog.py` owns the small composite/rebased resolver wrappers;
they know only the source-neutral protocol. Compose applies the same atomic-key
rebasing to native and custom children. A dangling or multiply-defined origin is
a load error, not a renderer fallback. Before returning a compose-produced flow,
finalization checks:

```text
set(all non-None origins reachable from compiled IR)
    is resolved exactly once by the composed OriginResolver
```

Presentation remains defensive: a catalog that later becomes unavailable or a
third-party diagnostic resolver that raises yields an unlocated `Diagnostic`
without replacing the original `Failure`.

## Canonical provenance-bearing semantic IR

Source identity belongs to the operation that can fail. It does not live in a
parallel node/field/range map.

### Binding

```python
@dataclass(frozen=True)
class Binding:
    source: Any
    origin: OriginRef | None = None
```

Every `Flow.wiring[node_id][param]` value and every `FlowOutput` source is a
`Binding`. Expression walkers, edge inference, and parameter binding unwrap
`source`. Programmatic constructors normalize raw values to
`Binding(raw, None)` at their boundary.

### Parameter

`InputDecl` and `ParamDecl` are deleted. One source-neutral formal-parameter
value owns the signature for ordinary nodes, START, child signatures, and future
Python-authored graphs:

```python
@dataclass(frozen=True)
class Parameter:
    name: str
    contract: ValueContract | None = None
    required: bool = False
    has_default: bool = False
    default: Any = None
    coerce_strings: bool = False
    origin: OriginRef | None = None
    default_origin: OriginRef | None = None
```

`has_default` distinguishes an authored/defaulted `None` from no default.
`coerce_strings` is true only at a flow input boundary; coercion is derived from
`contract.type`, so there is no second `type_str` authority. Ordinary node
parameters usually contain only a name. Compose's YAML input parser lowers
directly into `Parameter`; no compose declaration object escapes the front end.
`StartNode` stores and binds that same tuple of parameters rather than deriving a
second list.

Origin selection is deterministic:

- an omitted required input uses `Parameter.origin`;
- an invalid authored default is a load diagnostic at `default_origin`, falling
  back to `origin`;
- a runtime boundary type failure uses
  `contract.origins.at(error.path)`, then `Parameter.origin`;
- a supplied START binding retains the compiled parameter/binding origin when the
  run boundary narrows wiring to the provided keys.

Compose validates defaults against the contract during loading, so a bad
authored default never waits until runtime. Invariants reject
`required and has_default` and a default without a contract-compatible value.

### Assertion

```python
@dataclass(frozen=True)
class Assertion:
    expression: str
    display: str
    origin: OriginRef | None = None
```

`Node.pre_asserts` and `Node.post_asserts` contain `Assertion`. Lowering rewrites
`expression` while preserving the author-facing `display` and `origin`. This
replaces `assert_texts`, positional range lists, and text-keyed lookup.

### ValueContract

Nested value failures need a source-neutral structured path without putting
provenance into `Type`:

```python
ValuePath = tuple[str | int, ...]

@dataclass(frozen=True)
class ContractOrigins:
    default: OriginRef | None = None
    entries: tuple[tuple[ValuePath, OriginRef], ...] = ()

    def at(self, path: ValuePath) -> OriginRef | None: ...

@dataclass(frozen=True)
class ValueContract:
    type: Type
    origins: ContractOrigins = ContractOrigins()
```

`ContractOrigins.at` selects an exact entry, then the longest existing ancestor,
then `default`. Entries are unique and sorted canonically. A `TypeCheckError`
carries the failing `ValuePath`; recursive record/list validation prefixes field
names and indices while unwinding.

`Node.output_contract` replaces `output_type`. `Parameter.contract` is the single
input contract. Runtime type checking reads only `contract.type` and chooses a
failure origin with `contract.origins.at(error.path)`. The type system stays
semantic and provenance-free.

### Remaining carriers

- `FlowOutput(name, binding)` owns its boundary binding.
- `Node.origin` is the node's default authored operation.
- `Node.origin_context` is the immutable outer-to-inner context inherited during
  graph growth.
- each CASE entry owns `when_origin` and `then_origin` beside its expressions and
  handle;
- inline CODE owns an immutable executable-line -> `OriginRef` table;
- synthesized START/END operations use the real boundary binding, assertion, and
  contract origins.

Every carrier implements source-neutral origin traversal/rewrite. `Node` handles
its common fields and exposes a node-owned hook for kind-specific carriers such
as CASE and CODE. `compile` can therefore clone, prefix context, rebase, and
validate origins without a `NodeKind` branch. Manifest serialization emits only
semantic fields (`Binding.source`, assertion expression/display, contract type,
CASE semantics) and explicitly omits origins, contexts, catalogs, and line maps.

This covers START run-input overrides, child START wiring replacement, MAP item
bindings, MAP/WAIT reserved bindings, cloned END contracts, inline children, and
LOOP `respawn`. A respawned LOOP driver preserves its outer context and own
origin; it does not append itself to its own context.

## Runtime failure authority

### Evaluator and dispatcher paths

The evaluator returns `EvaluationFailure(failure: Failure)` for expected uncaught
flow failure. It assigns codes by the stage that failed, never by parsing an
exception string:

- binding resolution/required expression -> `binding.evaluate`;
- input contract -> `contract.input`;
- assertion wrapper -> `assertion.failed`;
- node body -> `node.execute`;
- unsupported returned result -> `contract.node_result`.

The trace is `node.origin_context + (operation_origin or node.origin,)`, omitting
`None` values.

Dispatcher-owned validation maps expected node-produced data errors to
`contract.output`, `contract.route`, or `contract.growth`. The compile-owned
growth builder raises one source-neutral `GrowthRejected` only for expected
seed/budget/depth/fragment constraints. Runtime catches that type as a flow
failure; an arbitrary injected-builder exception remains a host/compiler error.

The following always remain ordinary Python exceptions outside `Failure`:

- invalid/corrupt log access or schema;
- fingerprint/digest mismatch;
- malformed canonical record bytes;
- reducer or topology invariant violation;
- an arbitrary growth-builder programming fault;
- observer, resolver, renderer, or host-control failure.

### Record, reducer, event, and result path

For an actual node failure the same immutable object is carried, not flattened:

```text
EvaluationFailure.failure
  -> NodeFailedRecord.failure
  -> failure reducer plan/state
  -> NodeFailed.failure + RunFailed.failure
  -> RunResult.failure
```

Preparation still canonicalizes before append. The order remains:

```text
prepare and validate -> append record -> apply reducer -> publish observers
```

Run-log schema becomes version 2. Schema-1 failure readers and compatibility
payload aliases are deleted. Live execution and replay produce equal `Failure`
and `RunResult.failure` values; replay still emits no invocation-local events.

Public `RunResult` contains `input`, `status`, `run_id`, `log`, `output`,
`failure`, and `events`. `NodeFailed` and `RunFailed` contain `.failure`. The old
flattened `error`, `error_type`, `locator`, `range`, and `traceback` fields are
removed rather than delegated.

### Terminal-unreachable is derived, not separately recorded

A successful or routed node record can leave END unreachable. Its causal
`NodeSucceededRecord` or `NodeRoutedRecord` is already the authoritative appended
transition, so runtime does not append an after-the-fact `NodeFailedRecord`.

During live apply and replay, terminal derivation deterministically constructs:

```python
Failure(
    code="graph.terminal_unreachable",
    message=...,
    node_id=None,
    trace=<origin of the first required unreachable END binding, if any>,
)
```

It produces `RunFailed(failure)` but no `NodeFailed` observer because no node
failed. Equal graph plus equal causal record therefore derives equal live/replay
results without weakening append-before-apply authority.

## Nested execution provenance

Generic graph construction prefixes each child clone with:

```python
child.origin_context = (
    spawner.origin_context
    + ((spawner.origin,) if spawner.origin is not None else ())
    + child.origin_context
)
```

The child's precise binding/assertion/contract origin remains the final trace
element. MAP iterations share authored origins while execution IDs distinguish
`review#0/check` from `review#1/check`. Runtime never strips clone suffixes,
walks baked child graphs, or resolves source frames.

This directly fixes nested precision: a child binding failure resolves the outer
CALL/MAP/LOOP callsite and the exact child binding, never a competing node header
or coarse `code:` fallback.

## Inline CODE origin protocol

CODE remains YAML-blind. Compose supplies an immutable mapping from executable
Python line numbers to `OriginRef`s:

- literal block scalars (`|`) receive one origin per physical content line,
  respecting indentation and chomping indicators;
- folded (`>`), quoted, or plain multiline scalars cannot promise a one-to-one
  physical mapping, so every constructed executable line maps to the whole
  scalar origin;
- wrapper lines map to the node's CODE-field origin;
- `SyntaxError.lineno` uses the same table;
- runtime exceptions select the deepest traceback frame whose code object belongs
  to that CODE node's retained compiled code-object tree, not merely a shared
  filename.

The selected opaque origin enters the ordinary `Failure.trace`. CODE never sees
a YAML line number, scalar style, `SourceRange`, or catalog.

## Migration strategy

Small green PRs require a temporary compatibility bridge inside the implementation
branches. The bridge is explicit and time-bounded:

1. The single-parse source model first feeds existing concrete-range consumers so
   behavior stays green.
2. Origin-bearing IR lands beside the old `SourceFrame`/range fields but becomes
   complete and catalog-validated before runtime consumes it.
3. Canonical Failure, diagnostic resolution, CLI cutover, and deletion of all old
   carriers land as one coordinated cutover. No final public release contains two
   authorities.

The final deletion includes `SourceSpan`, `SourceFrame`, `FieldRanges`,
`assert_texts`, positional assert/CASE range lists, `child_source`, top-frame
reparsing, singular/plural renderer aliases, and CLI graph-walking inference.

## Maintained OOD and run-flow diagrams

Mermaid is selected over PlantUML, D2, Graphviz, and generated `pyreverse` output:

- GitHub renders Mermaid source directly in review;
- MkDocs Material supports Mermaid fences;
- class, flow, and sequence syntax cover the required views;
- text diagrams evolve in the same diff as the classes they explain; and
- no Java runtime or committed generated SVGs are needed.

The implementation will:

1. configure the `mermaid` custom fence under `pymdownx.superfences`;
2. load the browser runtime from the immutable versioned URL
   `https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js` through
   `mkdocs.yml:extra_javascript`, rather than Material's floating `@11` URL (the
   exact artifact was verified available before finalizing this design);
3. add a docs-only `package.json` plus lockfile pinning `mermaid@11.12.1` and
   `@mermaid-js/mermaid-cli@11.12.0`; CI runs `npm ci`, validates every Mermaid
   block with the locked local `mmdc`, then runs `mkdocs build --strict`;
4. restrict diagrams to syntax shared by that pinned version and GitHub; and
5. keep diagrams curated around public/architectural contracts rather than emit
   one unreadable graph of every implementation helper.

The browser asset is not copied into the repository: pinning the immutable CDN
path avoids a roughly multi-megabyte generated vendor file, while the committed
lockfile makes parser validation reproducible. Mermaid is documentation tooling
only and never enters the Python package's runtime dependencies.

`docs/architecture.md` will contain:

- a package/layer ownership flowchart;
- a class diagram for the contract spine (`SourceDocument`, `SourceCursor`,
  `SourceCatalog`, `Binding`, `Parameter`, `Assertion`, `ValueContract`,
  `CompiledFlow`, `Failure`, records/events,
  `RunResult`, and `Diagnostic`);
- a sequence diagram from YAML load through successful node commits, recorded
  node failure, derived terminal-unreachable failure, replay, diagnosis, and CLI
  rendering; and
- a responsibility table linking every architectural class to its owning package
  and deeper reference page.

Per-kind and private helper classes remain documented by their package charter,
module docstring, API docs, and focused package pages. A 150-plus-class automatic
diagram would obscure ownership rather than clarify it.

## Delivery slices

1. **Single-parse source foundation:** immutable document/cursors, root/defs/
   external/inline migration, and parse-count tests.
2. **Origin-bearing semantic IR:** diagnostics leaf, catalog/rebasing, Binding,
   Parameter, Assertion, ValueContract, CASE/boundary provenance, and fingerprint
   neutrality.
3. **Canonical Failure cutover:** evaluator/dispatcher, schema-2 record, reducer,
   events, public result, deterministic terminal failure, and host boundaries.
4. **One diagnostic path:** nested context propagation, compose resolution,
   `LoadError(Diagnostic)`, CLI cutover, and deletion of legacy carriers.
5. **Precision completion:** structured validation paths, inline CODE lines,
   aliases/duplicates, boundaries, MAP/WAIT, CASE, and formatting-only replay.
6. **Architecture contract:** Mermaid configuration/CI, diagrams, docs, package
   charters, and engine/structure skills updated to the implemented design.

#100 is a separate WAIT reference-validation behavior bug; it should not be
hidden in provenance plumbing. #86 is an acceptance case for binding origins.
#91 and #103 close when the old resolution paths are deleted. Retry/error strategy
#63 may build on `Failure` later but is outside this work.

## Acceptance tests

### Parse and source ownership

- A plain flow, multiple defs, multiple aliases, and diamond external imports each
  invoke one full-document loader per physical file.
- No parse/build pass mutates a completed `SourceDocument`.
- Duplicate keys, merge precedence, recursive anchors, aliases, compact flows,
  defs, and inline overlays preserve declared semantics.
- Only `compose/source/` imports PyYAML after the cutover.

### Catalog and identity

- Adversarial path segments encode without collision.
- Cached children rebase independently under multiple aliases without reparsing
  or mutating the cache.
- Every compose-produced reachable origin resolves exactly once.
- Origins, catalogs, ranges, and CODE line tables do not affect fingerprints or
  growth digests; semantic binding/assert/type changes still do.
- Formatting-only edits retain origin keys and resolve to current ranges.

### Failure and durability

- All nine semantic codes are exercised at their real boundary.
- Expected node data becomes `Failure`; host/compiler/invariant faults remain
  Python exceptions.
- Memory and JSONL schema-2 logs round-trip equal `Failure` values.
- Fresh and terminal replay return equal `RunResult.failure`, including
  terminal-unreachable with no `NodeFailedRecord`.
- Append failure prevents reducer apply and matching observers.

### Precision and nesting

- Root, defs, external uses, CALL, MAP, and LOOP produce correct outer-to-inner
  traces.
- Input bindings, nested contracts, duplicate assertions, CASE items, boundary
  operations, MAP/WAIT reserved bindings, and literal inline CODE lines select
  exact authored sites.
- Folded/quoted multiline CODE reports the documented scalar-level fallback.
- A missing/broken presentation resolver never changes run status or masks the
  original `Failure`.

### Architecture and invariants

- Python-built source-free graphs compile, run, fail, and replay.
- `runtime/` stays kind-blind and imports neither YAML nor compose.
- Mermaid validation and `mkdocs build --strict` pass.
- The architecture diagrams and class responsibility table match the actual
  package charters and public contracts.
