# Implementation Plan: Unified Failure Origins — Delivery Slices 2–6

**Goal**: Replace parallel range/error carriers with one source-neutral semantic
IR, one durable `Failure`, one compose-owned diagnostic resolver, and maintained
Mermaid architecture maps, without teaching runtime about YAML or node kinds.

**Architecture**: `diagnostics.py` is the dependency-free value spine. Issue
#131 initially adds only `SourceRange`, `OriginRef`, `DiagnosticFrame`, and
`OriginResolver`; the coordinated #132/#133 branch later adds `Failure` and
`Diagnostic` before atomically migrating their consumers.
`compose/source/` owns YAML documents, source sites, catalogs, and lexical
rebasing. `nodes/` owns provenance-bearing `Binding`, `Parameter`, `Assertion`,
and `ValueContract` operations. `compile/` rewrites opaque origins generically.
Runtime carries one immutable `Failure` through evaluator, schema-2 records,
reducer, observers, and `RunResult`; only `ComposedFlow.diagnose()` resolves it.

**Tech stack**: Python 3.10+, frozen dataclasses, PyYAML (compose only), Pydantic
v2 run-record validation, pytest, MkDocs Material, Mermaid browser runtime
11.12.1, Mermaid CLI 11.12.0, Node 20.

**Design source**:
`docs/plans/2026-08-15-unified-failure-origins-design-v2.md`.

**Tracking issues**: [#131](https://github.com/ngocbh/agent-composer/issues/131)
owns the origin-bearing IR and resolver rebase; [#132](https://github.com/ngocbh/agent-composer/issues/132)
owns canonical durable failures; [#133](https://github.com/ngocbh/agent-composer/issues/133)
owns the diagnostic/`LoadError`/CLI cutover; [#134](https://github.com/ngocbh/agent-composer/issues/134)
owns origin precision; and [#135](https://github.com/ngocbh/agent-composer/issues/135)
owns Mermaid, architecture documentation, and CI validation.

**Prerequisite**: Delivery slice 1 has landed with `compose/source/document.py`,
`compose/source/cursor.py`, and `compose/source/literals.py` owning only
`SourceDocument`, occurrence-aware `SourceCursor`, and the separately measured
literal parser. The loader passes cursors into root, def, inline, and external
compilation, parses each physical file exactly once, and has deleted
`compose/helpers/reader.py` and `NodeReader`. Slice 1 deliberately does **not**
provide `OriginRef`, `SourceSite`, `SourceCatalog`, catalog rebasing, or origin
carriers; those start in #131 below. Never restore a reader compatibility layer.

## Final contract

```text
YAML text -> SourceDocument -> Binding/Parameter/Assertion/ValueContract
          -> CompiledFlow
          -> EvaluationFailure(Failure) -> NodeFailedRecord(Failure)
          -> reducer/events -> RunResult.failure
          -> ComposedFlow.diagnose -> Diagnostic -> CLI renderer
```

- `OriginRef.key` is opaque below compose and stable within one returned
  `ComposedFlow`.
- `Failure.trace` is outermost to innermost; the final origin is the precise
  failing operation.
- Catalogs, origins, ranges, and CODE line tables never affect fingerprints or
  expansion digests.
- Runtime never imports compose, PyYAML, or `compile.expand`, and never dispatches
  on `NodeKind`.
- Host/log/reducer/compiler faults remain Python exceptions. Only expected flow
  evaluation and node-produced-data rejection become `Failure`.
- The final tree has no `SourceSpan`, `SourceFrame`, `FieldRanges`,
  `assert_texts`, positional range lists, `child_source`, graph-walking CLI
  resolver, `InputDecl`, `ParamDecl`, `type_str`, or flattened public failure
  fields.

## Explicit temporary bridge

The bridge exists only between the commits named below and is removed by the
coordinated cutover commit.

1. `events.py` temporarily re-exports `SourceRange` from `diagnostics.py` so the
   slice-1 source work and existing tests keep importing while callers move.
2. `Node.range`, `Node.field_ranges`, descriptor range fields, and
   `ComposedFlow.source` stay beside complete origins until the catalog coverage
   test passes. Runtime still reads ranges during this interval.
3. `Node.output_type` is a temporary read/write property over its new
   `ValueContract`; it is not a second stored value. Existing `ParamDecl` and
   compose `InputDecl` survive only until Step 2F, where all callers migrate to
   canonical `Parameter` and both old classes are deleted. No compatibility
   aliases are committed at the Group C boundary.
4. Raw programmatic wiring is accepted only at `Flow`/`CompiledFlow.from_parts`
   construction and normalized immediately to `Binding(raw, None)`. Internal
   flow wiring is never mixed after that boundary.
5. There is no committed compatibility alias for `RunResult.error`,
   `NodeFailed.error`, schema 1, or the old locator/range payload. The failure
   record, reducer, result, diagnosis, and runtime CLI move in one coordinated
   commit after origin coverage is complete.

## Dependency and commit table

| Group | Steps | Depends on | Main files | Green commit |
| --- | --- | --- | --- | --- |
| A | 2A–2B | slice 1 | `diagnostics.py`, `nodes/contracts.py` | #131 · `feat(engine): add origin and semantic carrier contracts` |
| B | 2C–2D | A | `compile/model.py`, wiring consumers, assertions | #131 · `refactor(engine): make bindings and assertions semantic carriers` |
| C | 2F | B | parameters, top-level value contracts, manifest | #131 · `refactor(engine): carry parameters and value contracts through compiled IR` |
| D | 2G–2I | C | compose stamping, `compile/origins.py`, rebasing/growth | #131 · `feat(compose): compile complete opaque source origins` |
| E | 3F + 4A–4C | merged #131 | `Failure`, `Diagnostic`, `LoadError`, compose error sites | #132/#133 coordinated branch · `refactor(compose): establish canonical failure diagnostics` |
| F | 3A–3E + 4D | E | evaluator, engine, log, reducer, result, every consumer | same #132/#133 branch · `feat(engine): cut over to canonical durable failures` |
| G | 5A–5D | F | nested contracts, CASE/reserved sites, CODE, replay | #134 · `fix(compose): complete precise nested failure origins` |
| H | 6A–6D | G | MkDocs/CI, architecture/docs/skills/charters | #135 · `docs: add maintained architecture and run-flow maps` |
| I | 7A–7C | all | tests/review/PR | no code until review fixes are green |

Groups are sequential because each changes shared IR. Within a group, test
authoring and documentation can be delegated, but two implementers must not edit
the same production files concurrently.

Every numbered sub-action below is intended to take 2–5 minutes: add one narrow
test, run that exact test to see the intended failure, implement the named seam,
and rerun it. At every explicitly named green commit boundary, run the focused
tests **and the complete pytest suite** before committing.

## Mandatory issue, branch, PR, and merge sequence

Do not stack these branches. For each item: switch to freshly merged `main`, run
`git pull --ff-only`, assign and mark the issue in progress, create the named
branch, and post the relevant group plan before editing.

1. `dev/engine/origin-semantic-ir`: execute Groups A–D only; open one PR with
   `Fixes #131`, merge it, and verify merged `main`.
2. `dev/engine/unified-failure-diagnostics`: branch from that new `main`, execute
   Groups E–F, and open **one coordinated PR** whose body contains both
   `Fixes #132` and `Fixes #133`. Group E's load-path commit may precede Group F
   only inside this unmerged branch; no intermediate public API ships.
3. `dev/compose/failure-origin-precision`: branch from the newly merged
   #132/#133 `main`, execute Group G, open `Fixes #134`, merge, and verify main.
4. `dev/docs/architecture-maps`: branch from the newly merged #134 `main`,
   execute Group H, open `Fixes #135`, merge, and verify main.

Never claim a slice complete before its PR is merged. If a focused test passes
but the full suite fails, there is no green boundary and no commit.

---

## Slice 2 — origin-bearing semantic IR and catalog rebasing

### Step 2A: Add only the dependency-free origin-resolution leaf contracts

**Files**:

- Create `src/agent_composer/diagnostics.py`.
- Modify `src/agent_composer/events.py` (temporary `SourceRange` re-export only).
- Create `tests/engine/test_diagnostics.py`.
- Modify `tests/engine/test_source_range.py`.

#### Test first

Add tests for:

- `OriginRef("")` and non-strings reject; a valid ref is frozen and hashable;
- `SourceRange.covered_lines()` retains all current half-open behavior after the
  move;
- `DiagnosticFrame` is a frozen presentation value with label/text/range/note;
- a structural `OriginResolver` can resolve an origin to a frame or return
  `None`, without importing compose;
- importing `diagnostics.py` does not load `yaml`, `compose`, `nodes`, `compile`,
  or `runtime`.

Use this exact public vocabulary:

```python
from agent_composer.diagnostics import (
    DiagnosticFrame,
    OriginRef,
    OriginResolver,
    SourceRange,
)
```

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_diagnostics.py tests/engine/test_source_range.py -q
```

#### Implement

Move `SourceRange` unchanged. Add `OriginRef` and `DiagnosticFrame` exactly as
designed, plus the source-neutral
`OriginResolver(Protocol)` with `resolve(OriginRef) -> DiagnosticFrame | None`.
Do not add `Failure`, `FailureCode`, `Diagnostic`, rendering, or YAML knowledge
on #131; those belong to the coordinated #132/#133 branch.

Temporarily import/re-export `SourceRange` from `events.py`; do not keep the old
class definition. Keep the new vocabulary module-internal for now; the atomic
#132/#133 cutover decides package-root exports.

### Step 2B: Add canonical node semantic carriers

**Files**:

- Create `src/agent_composer/nodes/contracts.py`.
- Modify `src/agent_composer/nodes/__init__.py`.
- Create `tests/engine/test_semantic_contracts.py`.

#### Test first

Test:

- `Binding(source, origin)` and `Assertion(expression, display, origin)` are
  frozen and preserve arbitrary semantic values;
- `as_binding(raw)` returns `Binding(raw, None)` and returns an existing binding
  unchanged;
- `ContractOrigins` rejects duplicate/unsorted entries and non-string/non-int
  path segments;
- `ContractOrigins.at(path)` selects exact, then longest ancestor, then default;
- `ValueContract(type, origins)` is frozen and defaults to empty origins;
- `Parameter` preserves `name`, contract, required/default/coercion semantics,
  `origin`, and `default_origin`; rejects `required and has_default`;
- `has_default=True, default=None` remains distinct from no default, and
  `coerce_strings` does not introduce a second textual type authority;
- `map_origins(mapper)` rewrites only non-`None` origins and retains semantic
  fields.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_semantic_contracts.py -q
```

#### Implement and commit green

Implement `ValuePath`, `Binding`, `Parameter`, `Assertion`, `ContractOrigins`,
`ValueContract`, `as_binding`, and `binding_source` in `nodes/contracts.py`.
Use exactly the final `Parameter` fields from the design:
`name`, `contract`, `required`, `has_default`, `default`, `coerce_strings`,
`origin`, and `default_origin`. Do not retain `type_str` or infer default
presence from `default is None`.
Keep this module source-neutral: it may import only `diagnostics` and `typesys`.
Export the carriers from `nodes/__init__.py`.

Run Group A plus the layer ratchet, then commit:

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_diagnostics.py tests/engine/test_source_range.py \
  tests/engine/test_semantic_contracts.py tests/engine/test_layer_direction.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer/diagnostics.py src/agent_composer/events.py \
  src/agent_composer/nodes \
  tests/engine/test_diagnostics.py tests/engine/test_source_range.py \
  tests/engine/test_semantic_contracts.py
git commit -m "feat(engine): add origin and semantic carrier contracts"
```

### Step 2C: Normalize all flow-owned bindings once

**Files**:

- Modify `src/agent_composer/compile/model.py`.
- Modify `src/agent_composer/nodes/binding.py`.
- Modify `src/agent_composer/nodes/map/node.py`.
- Modify `src/agent_composer/nodes/wait/node.py`.
- Modify `src/agent_composer/compile/expand.py`.
- Modify `src/agent_composer/compile/validation.py`.
- Modify `src/agent_composer/compile/manifest.py`.
- Modify `src/agent_composer/compose/build/boundary.py`.
- Modify `src/agent_composer/compose/edges/data.py`.
- Modify `src/agent_composer/compose/validate/references.py`.
- Modify `src/agent_composer/compose/validate/wiring_parity.py`.
- Modify `src/agent_composer/runtime/state.py`.
- Create `tests/engine/test_semantic_ir.py`.
- Modify binding/graph/manifest/growth tests that directly inspect raw wiring.

#### Test first, in four micro-cycles

1. Test that `Flow` and `CompiledFlow.from_parts` normalize every raw wiring
   value to `Binding(raw, None)` and reject non-string node/parameter keys.
2. Test that `add_subgraph` normalizes raw programmatic values, while an existing
   binding preserves its origin.
3. Test that binding evaluation, reference walking, edge inference, MAP `over`,
   WAIT `until`, child clone renaming, and growth validation read
   `Binding.source` without losing `Binding.origin`.
4. Test fingerprints/digests are identical for equal semantic wiring with
   different origins, while changing `.source` still changes identity.

Run after each micro-cycle:

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_semantic_ir.py tests/engine/test_binding.py \
  tests/engine/test_infer.py tests/engine/test_expand.py \
  tests/engine/test_flow_identity.py -q
```

#### Implement

- Change `Flow.wiring` to `dict[str, dict[str, Binding]]`; normalize only in
  `Flow.__init__`, `CompiledFlow.from_parts`, and `CompiledFlow.add_subgraph`.
- Change `FlowOutput` to `FlowOutput(name, binding)`; its `__post_init__`
  normalizes the second positional value through `as_binding`. Keep a temporary
  read-only `.from_` property only until Step 4D.
- Make `_resolve_source`, binding/ref walkers, validators, edge inference,
  manifest serialization, MAP/WAIT reserved hooks, and growth validation call
  `binding_source`.
- Make `_rens_internal` and every growth builder return a new `Binding` whose
  rewritten `.source` retains the original `.origin`.
- Manifest only `Binding.source` and `FlowOutput.binding.source`; list `origin`
  explicitly as noncanonical rather than falling through `_canonical_value`.

### Step 2D: Replace string/assert-text triples with `Assertion`

**Files**:

- Modify `src/agent_composer/nodes/base.py`.
- Modify `src/agent_composer/compose/validate/asserts.py`.
- Modify `src/agent_composer/compose/build/boundary.py`.
- Modify `src/agent_composer/compile/expand.py`.
- Modify `src/agent_composer/compile/manifest.py`.
- Modify `tests/engine/test_node_asserts.py`,
  `tests/engine/test_clone_child.py`,
  `tests/engine/test_spawner_post_assert.py`, and
  `tests/engine/test_flow_identity.py`.

#### Test first

Add tests that:

- duplicate assertions remain distinct ordered `Assertion` objects;
- `Node.run` evaluates `.expression` but reports `.display`;
- flow-assert lowering and call-post-assert lowering rewrite only `.expression`
  and preserve `.display` plus `.origin`;
- cloning/deepcopy retains origins;
- assertion origins do not affect identity, but expression/display changes do.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_node_asserts.py tests/engine/test_clone_child.py \
  tests/engine/test_spawner_post_assert.py tests/engine/test_flow_identity.py -q
```

#### Implement and commit green

Make `Node.pre_asserts`/`post_asserts` lists of `Assertion`. Replace
`LoweredAsserts` string/text maps with ordered assertion lists. Rewrite
`_lower_spawner_post_asserts` to use `dataclasses.replace(assertion,
expression=...)`. Keep only the old positional range lists as the temporary
runtime-location bridge; delete `assert_texts` now. Manifest assertions as
`{"expression": ..., "display": ...}` and omit origins.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_semantic_ir.py tests/engine/test_node_asserts.py \
  tests/engine/test_clone_child.py tests/engine/test_spawner_post_assert.py \
  tests/engine/test_flow_identity.py tests/engine/test_expand.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer tests/engine
git commit -m "refactor(engine): make bindings and assertions semantic carriers"
```

### Step 2F: Replace declarations and bare types with `Parameter`/`ValueContract`

**Files**:

- Modify `src/agent_composer/nodes/base.py`.
- Modify `src/agent_composer/nodes/binding.py`.
- Modify `src/agent_composer/nodes/start/node.py`.
- Modify `src/agent_composer/nodes/end/node.py`.
- Modify `src/agent_composer/typesys/seeding.py`.
- Modify `src/agent_composer/compose/helpers/types.py`.
- Modify `src/agent_composer/compose/nodes/base.py` and all node builders that
  assign `output_type`.
- Modify `src/agent_composer/compose/nodes/call.py` and
  `src/agent_composer/compose/nodes/case.py`.
- Modify `src/agent_composer/compose/nodes/resolver.py`.
- Modify `src/agent_composer/compose/build/pipeline.py` and
  `src/agent_composer/compose/build/inline_inputs.py`.
- Modify `src/agent_composer/compose/validate/{asserts,loop_types,references}.py`.
- Modify `src/agent_composer/compose/__init__.py` and every child-required/default
  check found by the deletion census.
- Modify agent mode files that read the declared output.
- Modify `src/agent_composer/runtime/state.py`.
- Modify `src/agent_composer/compile/manifest.py`.
- Extend `tests/engine/test_semantic_contracts.py` and relevant builder/identity
  tests.

#### Test first, in five micro-cycles

1. `Node.params`, `StartNode.params`, child signatures, and
   `ComposedFlow.input` contain the same canonical `Parameter` values; no
   `InputDecl` or `ParamDecl` survives a compose or compile boundary.
2. flow-input YAML lowers directly to `Parameter`; cover omitted input,
   explicitly supplied `null`, nullable default `null`, and invalid
   non-nullable default. `has_default` distinguishes explicit/defaulted `None`
   from omission.
3. `Node.output_contract` replaces stored `output_type`; setting the temporary
   `.output_type` property changes the contract type without discarding origins.
4. bind/start/output validation uses `contract.type`; `coerce_strings` is true
   only at a flow-input boundary. An untyped parameter accepts any stored
   Python default; a typed default is validated without coercing or replacing
   that stored value. Nested error paths and nested origin selection are
   deliberately deferred to #134 Step 5A.
5. manifests include parameter name/contract/required/has-default/default/
   coercion semantics and omit every
   contract origin.

Use:

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_semantic_contracts.py tests/engine/test_binding.py \
  tests/engine/test_start_node.py tests/engine/test_build_nodes.py \
  tests/engine/test_flow_identity.py tests/engine/test_run_state_outcomes.py -q
```

#### Implement and commit green

Delete `nodes.binding.ParamDecl` and `compose.helpers.types.InputDecl`. Make
`Node.params`, `StartNode.params`, `ComposedFlow.input`, and
`ChildSignature.inputs` use `Parameter` directly; `read_flow_inputs` lowers YAML
straight to that type. Derive string coercion from `Parameter.contract.type`
only when `coerce_strings` is true. Validate authored defaults while composing,
before a runnable flow is returned: validate typed values but retain the exact
original Python default; skip type validation for an untyped parameter. Convert
all `decl.default is None`/`is not None` omission checks to `has_default`.

Store only `ValueContract`; the temporary `.output_type` property delegates to
it and never caches a second type. Change `NodeDescriptor.output_type` to
`output_contract`. Change `ChildSignature.output` to a `ValueContract | None`.
Ensure MAP wraps only the child contract's semantic type and uses the MAP output
site for the new outer contract origin. Update the manifest census explicitly.
Update all tests, fakes, docs, and imports in this commit to construct
`Parameter`; do not leave deprecated exports.

Before running green, require this deletion census to produce no production or
test matches (comments/docstrings included, except this plan):

```bash
rg -n '\b(InputDecl|ParamDecl|type_str)\b' src tests
```

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_semantic_contracts.py tests/engine/test_binding.py \
  tests/engine/test_flow_inputs.py tests/engine/test_start_node.py \
  tests/engine/test_build_nodes.py tests/engine/test_flow_identity.py \
  tests/engine/test_run_state_outcomes.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer tests/engine
git commit -m "refactor(engine): carry parameters and value contracts through compiled IR"
```

### Step 2G: Stamp complete local origins from cursors

**Files**:

- Create `src/agent_composer/compose/source/catalog.py`.
- Modify `src/agent_composer/compose/source/__init__.py`.
- Modify `src/agent_composer/compose/nodes/base.py` and every descriptor with
  positional sites (`case`, assertions, inline bodies).
- Modify `src/agent_composer/compose/helpers/types.py`.
- Modify `src/agent_composer/compose/build/{pipeline,boundary,inline_inputs}.py`.
- Modify per-kind compose builders for CALL/MAP/LOOP, CASE, WAIT, CODE, and leaf
  inputs/outputs.
- Modify `src/agent_composer/compose/loader.py`.
- Create `tests/compose/test_source_catalog.py`.
- Create `tests/engine/test_origin_compilation.py`.

#### Test first, in seven micro-cycles

1. A node header produces `Node.origin`; each input produces
   `Binding.origin`; each assertion produces its own origin, including duplicate
   text.
2. Flow input/output declarations populate only each top-level
   `ContractOrigins.default`; recursive paths are deferred to #134 Step 5A.
   Flow outputs own binding origins.
3. START/END synthesized wiring/assertions/contracts reuse the authored boundary
   operation origins rather than generated IDs.
4. CASE arms retain ordered `when_origin` and `then_origin`; `else` has its own
   target origin.
5. MAP `over`, WAIT `until`, inline-child input/output, compact, and single-node
   def overlays retain their authored occurrence origins.
6. `SourceCatalogBuilder.register(semantic_path, cursor)` rejects duplicate
   origin keys, `freeze()` is immutable, and the resulting `SourceCatalog`
   resolves each registered frame. Duplicate mapping pairs, aliases,
   adversarial `~`/`/`, null/bool/int/string mapping keys, numeric strings, and
   integer indices cannot collide. In particular, CASE's bare YAML-1.1 `on:`
   key (`True`) is distinct from integer key `1`.
7. renamed/projected virtual overlays register their backing authored cursor;
   a wholly source-free virtual cursor returns `None` and creates no entry.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/compose/test_source_catalog.py \
  tests/engine/test_origin_compilation.py tests/engine/test_compact.py \
  tests/engine/test_defs.py tests/engine/test_inline_composite_bodies.py \
  tests/engine/test_case.py tests/engine/test_reserved_wiring_keys.py -q
```

#### Implement

Create `SourceSite` and a private mutable `SourceCatalogBuilder` in
`compose/source/catalog.py`. This builder is the **only** cursor-to-origin
construction boundary:

```python
def register(
    self,
    semantic_path: tuple[str | int, ...],
    cursor: SourceCursor,
) -> OriginRef | None: ...

def freeze(self) -> SourceCatalog: ...
```

`register` returns `None` for a source-free cursor. Otherwise it stores one
`SourceSite(document, occurrence_path, label)` and returns an `OriginRef` whose
key is encoded from both the semantic operation path and exact source
occurrence. Use this private, fully typed encoding (each component uses
JSON-Pointer `~0`/`~1` escaping before `/` joining):

```text
encode(("semantic", *semantic_path, "occurrence", *occurrence_tokens))
string x -> s:<escaped-x>
integer n -> i:<base-10-n>
null mapping key -> n:null
boolean mapping key -> b:0 or b:1
_MapStep(key, pair_ordinal) -> ("map", key, pair_ordinal)
_SeqStep(index)             -> ("seq", index)
```

Thus a mapping pair contributes the three typed segments
`s:map/<typed-key>/i:<pair_ordinal>` and a sequence item contributes
`s:seq/i:<index>`. Encode occurrence mapping keys by exact type in this order:
`None -> n:null`, `bool -> b:0|b:1`, `int -> i:<decimal>`, and escaped
`str -> s:<text>`; checking `bool` before `int` is mandatory. Other key types may
remain navigable in a cursor, but the builder raises the current located
`LoadError` if compose tries to register one. Pair ordinal is mandatory even for
the effective last duplicate. Alias occurrences retain different occurrence
keys while their `SourceSite`s deliberately resolve to the same anchor definition
range.

`freeze()` copies into an immutable mapping and rejects every duplicate key;
`SourceCatalog.resolve()` derives a `DiagnosticFrame` from the retained
document/path without reparsing. A projected/renamed overlay passes its retained
backing cursor to `register`, so lexical semantic identity changes while its
physical site remains authored; a `SourceCursor.virtual(...)` with no backing
document remains source-free.

Descriptors may retain compose-only `SourceCursor` occurrence references while
parsing, but lowering immediately puts provenance on the semantic operation:
`Node.origin`, `Binding`, `Parameter`, `Assertion`, `ValueContract`, and owned
CASE entries. Do not add `field_origins`, positional origin lists, or another
parallel range/origin map. No cursor or catalog is retained by `CompiledFlow`.

All parsing/building code calls `SourceCursor.child`, `children`,
`mapping_occurrences`, and projection methods directly. Do not recreate
`NodeReader`, `compose/helpers/reader.py`, a `site()` convenience on the cursor,
or an `id(yaml_node)` index. Thread the one local `SourceCatalogBuilder` through
the compose build context; freeze it only after local lowering is complete. A
source-free cursor produces `None` origins and no local catalog entries.

### Step 2H: Add generic origin traversal, rebasing, and catalog completeness

**Files**:

- Create `src/agent_composer/compile/origins.py`.
- Modify `src/agent_composer/compile/__init__.py`.
- Modify `src/agent_composer/nodes/base.py`.
- Modify `src/agent_composer/nodes/case/node.py`.
- Modify `src/agent_composer/nodes/code/node.py` (hook only; line mapping is Step
  5C).
- Modify CALL/MAP/LOOP nodes to expose their child flow to generic traversal.
- Modify `src/agent_composer/compose/source/catalog.py`.
- Modify `ComposedFlow` in `compose/build/pipeline.py`.
- Modify `_CallableScope.resolve` and file-cache handling in `compose/loader.py`.
- Create `tests/engine/test_origin_rebase.py`.

#### Test first, in seven micro-cycles

1. `iter_flow_origins(flow)` reaches wiring, outputs, node/common contracts,
   assertions, CASE hooks, CODE hooks, and recursively baked children without a
   `NodeKind` match.
2. `rewrite_flow_origins(flow, mapper)` returns a deep copy, rewrites contexts and
   every reachable carrier, and leaves the canonical cached child unchanged.
3. Two aliases of one cached external file rebase under distinct
   `root/uses/<alias>` namespaces without rereading or reparsing the file.
4. A diamond import keeps both lexical namespaces and may share the immutable
   underlying `SourceDocument`.
5. custom child resolvers accept either a completely source-free flow or one
   whose `OriginResolver` resolves every reachable origin; dangling or
   multiply-defined reachable origins fail as `LoadError`.
6. two CALL/MAP/LOOP nodes using the same alias and shared compiled child are
   visited/rebased once and retain one resolver namespace, while their execution
   IDs remain distinct.
7. a second rewrite of one rebased copy changes neither the canonical cached
   child nor a previously produced rebased copy; a deliberate object cycle
   terminates through the recursion guard.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_origin_rebase.py tests/engine/test_uses.py \
  tests/engine/test_uses_files.py tests/engine/test_flow_identity.py -q
```

#### Implement

Give `Node` these source-neutral hooks:

```python
def iter_local_origins(self) -> Iterator[OriginRef]: ...
def rewrite_local_origins(self, mapper: Callable[[OriginRef], OriginRef]) -> None: ...
def origin_children(self) -> tuple[Any, ...]: return ()
```

The base handles origin/context, params/contracts/assertions/output contract;
CASE and CODE override only their owned carriers; CALL/MAP/LOOP expose `child`.
`compile/origins.py` walks `Flow.wiring`, outputs, nodes, and children generically.
Both `iter_flow_origins` and `rewrite_flow_origins` own identity-based visited
sets for flows, nodes, and child containers, plus an active-recursion guard.
Mark an object before descending, rewrite each object at most once, and never
key this logic by node ID or equality. This is required because `deepcopy`
preserves shared child references. The module must not import compose or
dispatch on `NodeKind`.

Make `SourceCatalog` implement the leaf `OriginResolver` protocol. In
`compose/source/catalog.py`, define the rebased/composite resolver as one
explicit immutable table:

```python
@dataclass(frozen=True)
class _ResolutionTarget:
    delegate: OriginResolver
    old_origin: OriginRef

@dataclass(frozen=True)
class ResolverTable(OriginResolver):
    _entries: Mapping[OriginRef, _ResolutionTarget]

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[tuple[OriginRef, _ResolutionTarget]],
    ) -> "ResolverTable": ...
```

`from_entries` consumes the pair iterable, detects repeated new keys before a
dict can collapse them, copies, and freezes the result. It rejects every
duplicate new key even when both targets compare equal.
`resolve(new)` performs exactly one table lookup, then delegates with the stored
old ref. A child old key is always scoped as one opaque atomic string segment;
never parse third-party keys. Local entries target the frozen `SourceCatalog`;
rebased child entries target that child's original resolver.

`ComposedFlow` stores `origin_resolver: OriginResolver | None`.
`ComposedFlow.rebased(prefix)` copies and rewrites the compiled IR and wraps its
resolver with the exact new→old map; it never assumes the resolver is a
`SourceCatalog`. `_LoadCtx` keeps canonical file-local flows. The build/scope
context must retain the full resolved child `ComposedFlow` and its rebased
resolver contribution whenever CALL/MAP/LOOP bakes `child.compiled`; it may not
discard the resolver after extracting the graph. Alias caching contributes a
shared child resolver table once, so two nodes using the same alias reuse it;
distinct table fragments defining the same new key are rejected.

`_CallableScope.resolve` rebases only the lexical `uses:` alias result before its
alias cache stores it. Finalize every returned compose flow by collecting all
reachable origins (with the identity-safe traversal), requiring table membership
for each one, successfully resolving each frame, and rejecting any extra
resolver/table on a source-free flow. Table uniqueness proves multiplicity;
protocol calls alone do not. Cached children, resolvers, prior rebased copies,
and shared compiled children are never mutated.

### Step 2I: Prefix nested execution context during graph construction

**Files**:

- Modify `src/agent_composer/compile/expand.py`.
- Modify `src/agent_composer/nodes/loop/node.py`.
- Extend `tests/engine/test_clone_child.py`,
  `tests/engine/test_map_subgraph_builder.py`, and
  `tests/engine/test_loop_subgraph_builder.py`.

#### Test first

Pin:

- CALL child context = parent context + parent origin + child's existing
  context;
- every MAP element shares authored origins but retains distinct execution IDs;
- nested LOOP→CALL and CALL→LOOP preserve outer-to-inner order;
- `LoopNode.respawn()` copies its context and origin but does not append itself;
- synthetic MAP boundaries use the spawner operation origin, not a generated ID.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_clone_child.py tests/engine/test_map_subgraph_builder.py \
  tests/engine/test_loop_subgraph_builder.py tests/engine/test_flow_identity.py -q
```

#### Implement and commit green

In `clone_child`, after deepcopy and ID renaming, prefix each clone exactly once:

```python
clone.origin_context = (
    spawner.origin_context
    + ((spawner.origin,) if spawner.origin is not None else ())
    + clone.origin_context
)
```

Pass the spawner object/context into CALL/MAP/LOOP builders rather than infer it
from IDs. Copy origin fields in LOOP respawn. Explicitly omit origins from all
identity data.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/compose/test_source_catalog.py \
  tests/engine/test_origin_compilation.py tests/engine/test_origin_rebase.py \
  tests/engine/test_clone_child.py tests/engine/test_map_subgraph_builder.py \
  tests/engine/test_loop_subgraph_builder.py tests/engine/test_flow_identity.py \
  tests/engine/test_kind_census.py tests/engine/test_layer_direction.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer tests/engine tests/compose
git commit -m "feat(compose): compile complete opaque source origins"
```

Push `dev/engine/origin-semantic-ir`, open the #131 PR, and merge it before any
step below. Start Groups E–F only from freshly pulled merged `main` on
`dev/engine/unified-failure-diagnostics`.

---

## Coordinated #132/#133 foundation — semantic failure and diagnostic values

This group intentionally precedes the runtime failure cutover. It lets the later
coordinated commit switch runtime and CLI without needing a temporary flattened
failure API. It exists only on the unmerged coordinated branch.

### Step 3F: Add `Failure` and `Diagnostic` to the leaf module

**Files**:

- Modify `src/agent_composer/diagnostics.py`.
- Modify `tests/engine/test_diagnostics.py`.

#### Test first

Add tests that:

- `Failure` accepts exactly the nine closed design codes, exact `str | None`
  fields, and an immutable outer-to-inner `tuple[OriginRef, ...]` trace;
- `Diagnostic` is frozen, contains ordered `DiagnosticFrame`s and notes, and
  carries the semantic failure fields without an `OriginRef`;
- neither value imports YAML, compose, nodes, compile, or runtime;
- neither value is exported from package root before Step 4D.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_diagnostics.py -q
```

#### Implement

Add `FailureCode` as a `Literal`, one `FAILURE_CODES` frozen set, `Failure`, and
`Diagnostic` exactly as designed. Validate the closed code and field/trace types
in `Failure.__post_init__`; keep formatting and source resolution out of the
leaf. Do not yet change events, runtime, `RunResult`, CLI, or package-root
exports.

### Step 4A: Make `LoadError` accept exactly one `Diagnostic`

**Files**:

- Modify `src/agent_composer/compose/helpers/errors.py`.
- Modify `src/agent_composer/compose/source/catalog.py` only for a pure
  cursor-to-frame helper used by immediate load diagnostics.
- Modify `tests/engine/test_cli_load_errors.py` and
  `tests/cli/test_render_diagnostic.py`.

#### Test first

Test that:

- `LoadError(Diagnostic(...)).diagnostic` is the exact value and `str(error)` is
  only the diagnostic message;
- old `line=`, `lines=`, `notes=`, `range=` and string-only construction raise
  `TypeError`;
- an unlocated factory creates `Diagnostic(code="compose.load", ...)`;
- `SourceCatalog.resolve(origin)` creates one resolved frame and satisfies
  `OriginResolver`;
- prepending an external/def/import frame is pure and preserves frame order.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_cli_load_errors.py tests/cli/test_render_diagnostic.py -q
```

#### Implement

Keep one small compose-local helper:

```python
def load_diagnostic(message: str, *, cursor: SourceCursor | None = None,
                    code: str = "compose.load",
                    notes: tuple[str, ...] = ()) -> Diagnostic: ...
```

`LoadError.__init__` accepts only `Diagnostic`. Do not duplicate frame/text/range
fields on the exception. `load_diagnostic` may create a frame from the cursor's
document/path immediately, but it does not register an origin and does not add a
`SourceCursor.site()` API.

### Step 4B: Convert cursor, parse, and descriptor errors

**Files**:

- Modify `compose/source/document.py`, `compose/source/cursor.py`,
  `compose/parse/{file,nodes}.py`, `compose/helpers/{types,uses}.py`, and all
  `compose/nodes/*.py` error construction.
- Update focused parser/node/error tests.

#### Test first, in four micro-cycles

1. malformed YAML uses PyYAML's problem mark to make one frame;
2. Pydantic `loc` walks the `SourceCursor` to the precise top-level site;
3. missing/wrong descriptor fields use the current/field cursor;
4. type/default/CODE construction errors retain the precise cursor frame.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_parser.py tests/engine/test_node_parse.py \
  tests/engine/test_read_type.py tests/engine/test_errors.py \
  tests/engine/test_code_inline.py -q
```

#### Implement

Replace every old `LoadError(message, range=...)` in these files with
`LoadError(load_diagnostic(message, cursor=cursor))`. Do not turn a range back
into a cursor. Thread `SourceCursor` directly through helper signatures that
currently carry only `SourceRange`; do not recreate `NodeReader` or
`compose/helpers/reader.py`.

### Step 4C: Convert whole-graph, nested, and resolver errors

**Files**:

- Modify `compose/desugar/*.py`, `compose/edges/*.py`,
  `compose/validate/*.py`, `compose/build/*.py`, and `compose/loader.py`.
- Modify `src/agent_composer/cli/run.py` load rendering only.
- Update `tests/engine/test_errors.py`, uses/defs/inline tests, and
  `tests/engine/test_cli_load_errors.py`.

#### Test first, in five micro-cycles

1. bad refs, wiring parity, types, handles, and cycles resolve precise ordered
   frames rather than line/range lists;
2. duplicate/cycle notes live only in `Diagnostic.notes`;
3. nested defs prepend the def declaration frame once;
4. external import/cycle/version errors prepend lexical import frames in order;
5. `_render_load_error(error)` consumes only `error.diagnostic` and never reparses
   source text.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_errors.py tests/engine/test_defs.py \
  tests/engine/test_uses.py tests/engine/test_uses_files.py \
  tests/engine/test_inline_composite_bodies.py \
  tests/engine/test_cli_load_errors.py tests/cli/test_render_diagnostic.py -q
```

#### Implement and commit green

Convert all remaining compose `LoadError` construction. A caught nested
`LoadError` is re-raised with a new `LoadError(prepend_frame(...))`; never copy
message/line/range fields. Make `render_diagnostic(diag)` iterate its own frames,
using each frame's text/label/range. Keep the old runtime rendering functions
until the coordinated cutover.

```bash
rg -n 'LoadError\([^D]|line=|lines=|range=|\.line\b|\.lines\b|\.range\b' \
  src/agent_composer/compose src/agent_composer/cli/run.py
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_errors.py tests/engine/test_cli_load_errors.py \
  tests/engine/test_defs.py tests/engine/test_uses_files.py \
  tests/cli/test_render_diagnostic.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer/diagnostics.py src/agent_composer/compose \
  src/agent_composer/cli/run.py \
  tests/engine tests/cli
git commit -m "refactor(compose): establish canonical failure diagnostics"
```

The `rg` output above is an inspection list, not a zero-match assertion. Allowed
matches are `SourceRange`/`DiagnosticFrame`/`SourceCursor` internals and unrelated
third-party exception attributes; no legacy `LoadError` constructor or
`LoadError.line|lines|range` consumer may remain. The full suite must pass before
this Group E commit.

---

## Coordinated #132/#133 cutover — schema 2, reducer, result, and runtime

Steps 3A–3E and 4D form one coordinated commit. Run the named focused test after
each micro-step, but do not commit an intermediate public shape.

### Step 3A: Classify evaluator stages into `EvaluationFailure(Failure)`

**Files**:

- Modify `src/agent_composer/nodes/binding.py`.
- Modify `src/agent_composer/nodes/base.py` and `nodes/start/node.py`.
- Modify `src/agent_composer/runtime/eval_node.py`.
- Modify `tests/engine/test_eval_node.py` and
  `tests/engine/test_eval_node_raw.py`.

#### Test first

Exercise the real boundary for:

- missing/invalid expression → `binding.evaluate`;
- bound value contract mismatch → `contract.input`;
- false/evaluation-error assertion → `assertion.failed`;
- node body raise → `node.execute`;
- unsupported result or non-spawner Grow → `contract.node_result`.

Each test asserts `EvaluationFailure.failure`, `node_id`, `cause_type`, traceback,
and `origin_context + operation_origin` order. Add a source-free Python-built node
case whose trace is empty.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_eval_node.py tests/engine/test_eval_node_raw.py -q
```

#### Implement

Split `_evaluate_node_stream` into narrow try/except stages. Let binding errors
carry parameter and operation origin; let input-contract errors carry the
contract's top-level/default origin until #134 adds structured path selection;
let `NodeAssertFailed` carry its `Assertion`; and call
`node.exception_origin(exc)` for body exceptions. Add one helper that constructs
`Failure` and omits `None` origins. `_HostStop` and stream callback errors still
escape unchanged.

### Step 3B: Separate expected node-data rejection from host faults

**Files**:

- Modify `src/agent_composer/compile/model.py` and
  `src/agent_composer/compile/expand.py`.
- Modify `src/agent_composer/runtime/engine.py` and
  `src/agent_composer/runtime/state.py`.
- Modify `tests/engine/test_runtime_recorded_dispatch.py` and growth tests.

#### Test first

Exercise:

- wrong committed type → `contract.output`, using the output contract's
  top-level/default origin (#134 adds nested path precision);
- invalid route → `contract.route`, using CASE target origin when known;
- a closed compile seed-shape rejection → `contract.growth`;
- each runtime-known node-produced fragment, budget, reference-depth, topology,
  and prune rejection → `contract.growth`;
- arbitrary injected builder `RuntimeError` escapes by identity;
- reducer/topology/digest/append errors escape and produce no failure record.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_runtime_recorded_dispatch.py \
  tests/engine/test_run_state_growth.py tests/engine/test_grow_outcome.py -q
```

#### Implement

Define `GrowthRejected` as the shared source-neutral expected-node-data contract
in `compile/model.py`. Split ownership exactly at the existing layer boundary:

- `compile.expand.build_grow()` translates only its closed malformed/unsupported
  **seed-shape** rejections into `GrowthRejected`;
- `runtime/state.py` raises/translates `GrowthRejected` only for its existing
  known node-produced fragment, node-budget, reference-depth, fragment-topology,
  and prune validations;
- `runtime/engine.py::_build_growth` catches only `GrowthRejected` when turning
  expected node data into `contract.growth`.

Delete `_GrowBuilderError` and every broad wrapper. An arbitrary exception from
the injected growth builder, reducer, log, topology code, or invariant check
escapes by object identity. `_prepare_data_failure(code, operation_origin, exc)`
accepts a complete semantic code and never inspects exception text to classify
it.

### Step 3C: Replace the failure record payload and bump schema to 2

**Files**:

- Modify `src/agent_composer/runtime/log.py`.
- Modify `tests/engine/test_run_records.py` and
  `tests/engine/test_run_log.py`.

#### Test first

Add schema-2 round trips for every failure field and multi-frame origin trace.
Assert canonical JSON contains only `{"failure": ...}` for a
`node_failed` payload; rejects unknown codes, malformed origins, duplicate fields,
schema 1, and a node-failure whose `Failure.node_id` is null. Assert memory/JSONL
retain equal `Failure` values.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_run_records.py tests/engine/test_run_log.py -q
```

#### Implement

Set `SCHEMA_VERSION = 2`. Change `NodeFailedRecord` to one `failure: Failure`.
Add strict `_OriginPayload` and `_FailurePayload`; delete `_SourceSpanPayload` and
`_SourceRangePayload`. Encode/decode the trace as an ordered list and return a
tuple. Do not read schema 1.

### Step 3D: Carry the same Failure through reducer, observers, and `RunResult`

**Files**:

- Modify `src/agent_composer/events.py`.
- Modify `src/agent_composer/runtime/state.py`.
- Modify `src/agent_composer/run.py`.
- Modify state/run/recovery/golden tests.

#### Test first, in four micro-cycles

1. `prepare_node_failed(state, failure)` validates identity and stores one
   canonical `NodeFailedRecord.failure`.
2. reducer state, `NodeFailed.failure`, and `RunFailed.failure` carry the same
   immutable object after append; append failure changes neither state nor
   observers.
3. terminal-unreachable derives `graph.terminal_unreachable` from the first
   required END `Binding.origin`, appends no failure record, and emits no
   `NodeFailed`.
4. fresh and terminal replay produce equal `RunResult.failure`; terminal replay
   emits no invocation-local events.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_run_state_outcomes.py \
  tests/engine/test_runtime_recorded_dispatch.py \
  tests/engine/test_run.py tests/engine/test_run_recovery_prefixes.py \
  tests/engine/test_golden_run_results.py -q
```

#### Implement

- `NodeFailed` and `RunFailed` contain only `failure` (plus `node_id` only if it
  is retained as a convenience equal to `failure.node_id`; prefer no duplicate).
- `_FailureDelta` contains one `Failure`; remove all source-copy helpers.
- `_TerminalDelta` contains a derived `Failure | None` instead of error strings.
- `RunResult` fields are exactly `input`, `status`, `run_id`, `log`, `output`,
  `failure`, `events`. Remove flattened fields, not aliases.
- Update progress reporting to print `event.failure.message`.

### Step 3E: Add the compose diagnostic resolver

**Files**:

- Modify `ComposedFlow` in `src/agent_composer/compose/build/pipeline.py`.
- Modify `src/agent_composer/compose/source/catalog.py`.
- Create `tests/engine/test_failure_diagnosis.py`.

#### Test first

Test that `ComposedFlow.diagnose(failure)`:

- resolves trace entries in order into `DiagnosticFrame`s and marks the final
  frame primary by order, not by a separate field;
- copies code/message/node/cause/traceback exactly;
- returns an unlocated diagnostic when `origin_resolver` is `None` or a
  third-party resolver cannot resolve the complete trace;
- catches a resolver exception without changing/masking the original failure;
- needs no compiled-graph walk and performs no YAML parse.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_failure_diagnosis.py -q
```

#### Implement

`ComposedFlow` stores `origin_resolver: OriginResolver | None`; it never exposes
or type-assumes a YAML `SourceCatalog`. Delete any diagnostic dependency on
`compiled`. `diagnose` calls only the protocol's `resolve` method, in trace
order. If any origin is unresolved or resolution raises, return `Diagnostic`
with the original semantic fields, no frames, and one non-behavioral note that
source presentation was unavailable.

### Step 4D: Cut the CLI to the resolved diagnostic and delete legacy carriers

**Files**:

- Modify `src/agent_composer/cli/run.py`.
- Modify `src/agent_composer/cli/chat/command.py` and
  `src/agent_composer/cli/chat/tools.py`.
- Modify every repository consumer returned by the scoped `RunResult`/failure
  census, including all tests that currently read `.error`, `.error_type`,
  `.traceback`, or `.locator` from the replaced result/event/record classes.
- Modify package charters/exports that still mention legacy carriers.
- Modify `src/agent_composer/__init__.py`, `docs/api.md`, `docs/engine.md`, and
  `README.md` for the final public `.failure`/diagnosis contract.
- Create `tests/engine/test_failure_api_census.py`.
- Delete legacy source-location code from events/nodes/compose/loader.
- Replace `tests/cli/test_call_traceback_range.py` with origin-frame assertions.
- Modify `tests/cli/test_inline_composite_traceback.py`,
  `tests/cli/test_render_diagnostic.py`, `tests/engine/test_source_span.py`,
  `tests/engine/test_node_range.py`, and `tests/engine/test_run_locator.py`.

#### Test first

Pin:

- runtime CLI calls `loaded.diagnose(result.failure)` exactly once;
- the same renderer handles load and run diagnostics;
- nested frames render outermost first and the final operation last;
- `--engine-trace` uses `Diagnostic.traceback` only;
- chat command/tool failures read `result.failure.message` and never
  `RunResult.error`;
- no source/resolver failure changes the run exit status or message.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/cli/test_render_diagnostic.py tests/cli/test_call_traceback_range.py \
  tests/cli/test_inline_composite_traceback.py tests/engine/test_cli_load_errors.py \
  tests/engine/test_run_locator.py -q
```

#### Implement and delete

Make `_render_run_error(result, loaded, engine_trace=False)` require
`result.failure` and render `loaded.diagnose(...)`. Make `_render_load_error`
render `error.diagnostic`. Delete `_field_range`, `_top_ranges`, `_node_range`,
`_resolve_range`, `_strip_clone_suffix`, `_last_segment`, `_walk_call_frames`,
`_render_frame_stack`, `_render_notes`, singular/plural renderer aliases, and
top-frame reparsing.

At this atomic cutover, and not before it, export `Failure`, `Diagnostic`, and
`OriginRef` from `agent_composer.__init__`; keep `SourceRange`,
`DiagnosticFrame`, and `OriginResolver` available only from
`agent_composer.diagnostics`. Remove every replaced root export in the same
commit.

Before editing consumers, capture the exact migration set and update every
class-specific match in source **and tests**:

```bash
rg -n '\b(result|res|fresh|resumed|run_result)\.(error|error_type|traceback|locator)\b|\b(NodeFailed|RunFailed|NodeFailedRecord)\([^\n]*(error|locator)' \
  src tests
```

Do not mechanically replace unrelated agent/model/terminal-controller error
events. Add `tests/engine/test_failure_api_census.py`: inspect dataclass fields
for `RunResult`, `NodeFailedRecord`, `NodeFailed`, and `RunFailed`; parse source
AST imports/names for the explicit legacy classes; and assert `FlowOutput` has no
`from_` field/property while `Edge.from_` remains valid.

Delete:

- `SourceSpan` and the temporary `events.SourceRange` re-export;
- old `events.Diagnostic`;
- `FieldRanges`;
- Node/descriptor `range`, `field_ranges`, wiring labels, and positional range
  lists;
- `SourceFrame`, `ComposedFlow.source`, CALL/MAP/LOOP `child_source`;
- `_frame_ranges`, `top_frame_ranges`, `_section_lines`, boundary range helpers;
- temporary `.from_`, `.output_type`, and declaration `.type` properties after
  `rg` confirms no production reader remains.

Run the explicit-symbol deletion ratchet:

```bash
rg -n '\b(SourceSpan|SourceFrame|FieldRanges|assert_texts|pre_assert_ranges|post_assert_ranges|child_source|InputDecl|ParamDecl)\b' \
  src/agent_composer tests/engine tests/cli
rg -n '\b(result|res|fresh|resumed|run_result)\.(error|error_type|traceback|locator)\b' \
  src/agent_composer tests
```

The first command may match only intentional negative assertions in
`test_failure_api_census.py`; the second must have no matches. Do not grep
generic `.error`, because unrelated error-bearing event contracts are valid.
`FlowOutput.from_`, `Node.output_type`, and the exact replaced dataclass fields
are checked through import/AST assertions, so `Edge.from_` is never caught by a
broad textual deletion rule.

#### Commit the coordinated cutover green

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_eval_node.py tests/engine/test_eval_node_raw.py \
  tests/engine/test_runtime_recorded_dispatch.py \
  tests/engine/test_run_records.py tests/engine/test_run_log.py \
  tests/engine/test_run_state_outcomes.py tests/engine/test_run.py \
  tests/engine/test_run_recovery_prefixes.py \
  tests/engine/test_failure_diagnosis.py \
  tests/engine/test_failure_api_census.py tests/cli -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer tests docs/api.md docs/engine.md README.md
git commit -m "feat(engine): cut over to canonical durable failures"
```

Push the coordinated branch and open one PR containing both `Fixes #132` and
`Fixes #133`; merge it before #134 starts. The PR must list both full-suite runs
(Group E and Group F) and state that Group E never merged independently.

---

## Slice 5 — precision completion

### Step 5A: Resolve nested value-contract paths to exact YAML sites

**Files**:

- Modify `src/agent_composer/typesys/values.py`.
- Modify `src/agent_composer/compose/helpers/types.py`.
- Modify `src/agent_composer/compose/nodes/base.py`,
  `src/agent_composer/compose/build/boundary.py`, and
  `src/agent_composer/compose/build/inline_inputs.py`.
- Modify `src/agent_composer/nodes/binding.py`,
  `src/agent_composer/nodes/start/node.py`, and
  `src/agent_composer/runtime/state.py`.
- Modify `tests/engine/test_types.py` and `tests/engine/test_segments.py`.
- Create `tests/engine/test_contract_origin_precision.py`.

#### Test first

First pin `TypeCheckError.path` for top-level `()`, nested record fields, the
first deterministic missing/unknown field, list indices, records in lists,
nullable ancestors, and variant mismatch. Then cover input and output failures
for those paths and source-free Python graphs. Assert recursive
`read_value_contract` entries plus exact/longest-ancestor/default
`ContractOrigins.at` selection and the final trace origin.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_contract_origin_precision.py tests/engine/test_types.py \
  tests/engine/test_segments.py tests/engine/test_run.py -q
```

#### Implement

Give `TypeCheckError(message, *, path=())` an immutable tuple path. Thread a
private path through `_build_for_type` and `_build_scalar_or_list`, prefixing
field names and indices while unwinding and preserving existing messages.

Add `read_value_contract(cursor, registry, catalog_builder)`; keep
`read_type(value, registry)` as the source-free semantic helper. While
recursively reading the type, register each nested field/list-element cursor
under its `ValuePath`, sort entries canonically, and use the top-level declaration
origin as `ContractOrigins.default`. Binding, START, and committed-output checks
select `contract.origins.at(error.path)` and never parse the message. This step
is the sole authority for structured paths, recursive contract-site compilation,
and nested selection; #131 supplied only the carrier and top-level default.

### Step 5B: Complete CASE, boundary, MAP, WAIT, assertion, and child precision

**Files**:

- Modify `src/agent_composer/nodes/case/node.py` and compose CASE lowering.
- Modify MAP/WAIT reserved binding errors.
- Modify boundary/inline child lowering and origin context if tests expose gaps.
- Create `tests/engine/test_failure_origin_matrix.py`.

#### Test first

Use a table that runs and diagnoses:

- root binding and duplicate assertion;
- input declaration and flow output/boundary failure;
- searched CASE `when`, on-form `when`, `then`, and `else`;
- MAP `over`, per-item input, and WAIT `until`;
- def, external use, inline CALL, MAP element, LOOP body, and nested
  CALL→MAP→LOOP failures.

Assert semantic code, execution `node_id`, exact outer-to-inner origin keys, and
resolved frame labels/ranges. Preserve MAP iteration execution IDs while origins
remain equal.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_failure_origin_matrix.py tests/engine/test_case.py \
  tests/engine/test_reserved_wiring_keys.py -q
```

#### Implement

Make each CASE arm carry `when_origin` and `then_origin`; include the default arm
as an ordinary ordered entry with no condition. A CASE condition error carries
its arm origin, and invalid route data selects the target origin. Make reserved
binding hooks use the `Binding` object so errors retain `over`/`until` origins.
Fix only gaps exposed by the matrix; do not add WAIT reference behavior from #100.

### Step 5C: Add YAML-blind executable-line origins to CODE

**Files**:

- Modify `src/agent_composer/compose/source/cursor.py`.
- Modify `src/agent_composer/nodes/code/node.py`.
- Modify `src/agent_composer/compose/nodes/code.py`.
- Modify `src/agent_composer/compile/manifest.py`.
- Extend `tests/compose/test_source_cursor.py`.
- Extend `tests/engine/test_code_inline.py` and create
  `tests/engine/test_code_origin.py`.

#### Test first, in four micro-cycles

1. literal `|`, `|-`, and `|+` bodies map each executable line to its physical
   content origin despite inferred/explicit indentation and blank/trailing lines;
2. `SourceCursor.scalar_style` and `scalar_content_line_ranges()` handle literal
   chomping/indent indicators without exposing `_Occurrence.yaml_node`; folded
   `>`, quoted, plain, projected, and source-free virtual cursors exercise the
   documented whole-scalar/no-source fallbacks;
3. `SyntaxError.lineno` and a nested helper/runtime exception select the deepest
   traceback frame belonging to that node's compiled code-object tree;
4. two CODE nodes sharing a filename cannot steal each other's location; line
   maps/code-object caches do not affect fingerprints.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/compose/test_source_cursor.py \
  tests/engine/test_code_inline.py tests/engine/test_code_origin.py \
  tests/engine/test_flow_identity.py -q
```

#### Implement

Add these compose/source-owned APIs without exposing the retained YAML node:

```python
@property
def scalar_style(self) -> str | None: ...

def scalar_content_line_ranges(self) -> tuple[SourceRange, ...] | None: ...
```

For literal `|` styles, return one physical content range per constructed source
line, excluding the block header and honoring inferred/explicit indentation,
blank lines, and `|-`/`|`/`|+` chomping. Folded `>`, quoted, and plain scalars
return `None`, directing callers to the whole `cursor.range`; a projected cursor
delegates to its backing cursor; a source-free virtual cursor returns
`scalar_style=None`, ranges `None`, and has no origin.

Compose constructs an immutable executable-line→`OriginRef` tuple using only
those public cursor methods and passes it to `CodeNode`; fallback styles map all
executable lines to the CODE-field origin, and wrapper lines use that same
origin. `CodeNode` never receives a cursor/YAML node. CODE stores a frozen
compiled code-object tree and overrides
`exception_origin(exc)` by scanning traceback frames deepest-first and matching
code-object identity. It knows no YAML, `SourceRange`, line label, or catalog.
Syntax construction errors expose only executable line/origin back to the compose
descriptor, which constructs the load `Diagnostic`.

### Step 5D: Prove formatting-only replay and presentation failure isolation

**Files**:

- Extend `tests/engine/test_run_recovery_matrix.py`.
- Extend `tests/engine/test_failure_diagnosis.py` and
  `tests/engine/test_origin_rebase.py`.

#### Test first

Run a failure to JSONL, insert comments/blank lines and reindent the YAML without
changing semantic occurrence paths, load the edited flow, and resume the terminal
prefix. Assert:

- fingerprint and `Failure` remain equal;
- origin keys are unchanged;
- diagnosis uses the edited document's current ranges;
- replay emits no events and makes no write;
- a missing/broken resolver still returns failed status and the original
  failure.

Also cover duplicate keys, merge precedence, recursive aliases, two aliases of
one file, and a diamond import with the slice-1 parse counter still at one parse
per physical file.

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_run_recovery_matrix.py \
  tests/engine/test_failure_diagnosis.py tests/engine/test_origin_rebase.py \
  tests/compose/test_source_document.py -q
```

#### Commit precision completion green

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_contract_origin_precision.py \
  tests/engine/test_failure_origin_matrix.py tests/engine/test_code_origin.py \
  tests/engine/test_run_recovery_matrix.py \
  tests/engine/test_failure_diagnosis.py tests/engine/test_origin_rebase.py \
  tests/compose/test_source_cursor.py tests/compose/test_source_document.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add src/agent_composer tests/engine tests/compose
git commit -m "fix(compose): complete precise nested failure origins"
```

Push `dev/compose/failure-origin-precision`, open `Fixes #134`, and merge it
before starting #135 from fresh `main`.

---

## Slice 6 — maintained OOD and end-to-end run-flow maps

### Step 6A: Pin Mermaid rendering and validation

**Files**:

- Modify `mkdocs.yml`.
- Create `package.json` and generated `package-lock.json`.
- Create `scripts/check_mermaid.py`.
- Modify `.github/workflows/publish.yml`.
- Create `tests/docs/test_architecture_docs.py`.

#### Test first

The Python test asserts:

- every `mermaid` fence is closed and non-empty;
- `docs/architecture.md` has `flowchart`, `classDiagram`, and
  `sequenceDiagram` blocks;
- `mkdocs.yml` declares the SuperFences custom formatter and exact browser URL
  `https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js`;
- `package.json` pins both `mermaid` exactly to `11.12.1` and
  `@mermaid-js/mermaid-cli` exactly to `11.12.0` (no `^` or `~`).

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/docs/test_architecture_docs.py -q
```

#### Implement

Configure `pymdownx.superfences` with this exact custom fence formatter:

```yaml
- pymdownx.superfences:
    custom_fences:
      - name: mermaid
        class: mermaid
        format: !!python/name:pymdownx.superfences.fence_code_format
```

Add the exact jsDelivr URL above under `extra_javascript`. Make `package.json`
private with exact dev dependencies `mermaid: "11.12.1"` and
`@mermaid-js/mermaid-cli: "11.12.0"`, then commit the generated lockfile.
`scripts/check_mermaid.py` extracts
each fence to a temporary `.mmd`, invokes
`node_modules/.bin/mmdc`, reports `path:block-index`, and leaves no generated
assets in the tree.

Add a `docs` job to `publish.yml`: checkout, Python 3.11, Node 20, install
`.[docs]`, `npm ci`, run the Mermaid checker, then `mkdocs build --strict`.

### Step 6B: Draw the package ownership and contract-spine class diagrams

**Files**: modify `docs/architecture.md`; extend
`tests/docs/test_architecture_docs.py`.

#### Test first

Require the class diagram and responsibility table to name:

`SourceDocument`, `SourceCursor`, `SourceSite`, `SourceCatalog`,
`OriginResolver`, `Binding`, `Parameter`, `Assertion`, `ValueContract`,
`CompiledFlow`, `EvaluationFailure`, `Failure`,
`NodeFailedRecord`, `RunState`, `NodeFailed`, `RunFailed`, `RunResult`,
`DiagnosticFrame`, and `Diagnostic`.

Require the package flowchart to show compose as the only YAML owner and runtime
as kind-blind, with no runtime→compose or runtime→`compile.expand` edge.

#### Implement

Replace the large ASCII package map with one curated Mermaid flowchart and one
class diagram. Follow them with a responsibility table containing exactly one
owner and one-sentence meaning for each architectural class. Do not emit an
automatic 150+-class graph; private/per-kind helpers remain documented in their
package charters and focused pages.

### Step 6C: Draw success, recorded failure, derived failure, replay, and rendering

**Files**: modify `docs/architecture.md`; extend the docs test.

#### Test first

Require sequence participants for YAML/compose, run boundary, dispatcher,
RunLog, reducer, `ComposedFlow.origin_resolver`, and CLI. Require labeled
branches for:

- success (`Output` → succeeded record → `RunResult.output`);
- recorded failure (`EvaluationFailure` → `NodeFailedRecord` →
  `RunResult.failure`);
- derived terminal-unreachable (causal success/route record, no failure record);
- replay (same reducer, no historical observers);
- diagnosis/rendering after execution.

#### Implement

Add one sequence diagram with `alt` branches rather than separate drifting
diagrams. Keep source resolution after `RunResult`; neither runtime nor the log
may point to `SourceDocument` in the diagram.

### Step 6D: Synchronize docs, charters, public API, and project skills

**Files**:

- Modify `docs/{architecture,compose,engine,api,reference}.md` as applicable.
- Modify `src/agent_composer/README.md`.
- Modify package charters in `src/agent_composer/{__init__,nodes,compile,compose,runtime}/__init__.py`
  and create `compose/source/__init__.py` charter if slice 1 did not.
- Modify `.claude/skills/engine/{SKILL.md,reference.md,templates/node_kind/WIRING.md}`.
- Modify `.claude/skills/structure/{SKILL.md,reference.md}`.
- Modify `README.md` only for changed public `RunResult.failure`/diagnosis API.

#### Test first

Extend `tests/docs/test_architecture_docs.py` to reject the legacy terms as
current contracts (`SourceFrame`, `SourceSpan`, flattened `RunResult.error`,
schema 1) and to require the final ownership chain. Add/update public import
tests for `Failure`, `Diagnostic`, and `OriginRef`.

#### Implement and commit green

Document:

- one parse/document and compose-only YAML ownership;
- semantic IR carrier responsibilities;
- exact nine failure codes and host-fault boundary;
- schema 2 and formatting-independent replay;
- `ComposedFlow.diagnose(result.failure)`;
- how a future Python front end supplies no origins or an opaque complete
  `OriginResolver`;
- the node-kind template's requirement to expose kind-specific origins through
  node-owned hooks, never a runtime kind branch.

The composing-agents skill need not change because YAML syntax does not change;
mention this in the PR rather than editing it.

```bash
npm ci
conda run -n agentcomposer python scripts/check_mermaid.py docs
conda run -n agentcomposer python -m mkdocs build --strict
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/docs/test_architecture_docs.py tests/engine/test_layer_direction.py \
  tests/engine/test_kind_census.py -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
git add mkdocs.yml package.json package-lock.json scripts/check_mermaid.py \
  .github/workflows/publish.yml docs src/agent_composer/README.md \
  src/agent_composer/*/__init__.py .claude/skills README.md tests/docs
git commit -m "docs: add maintained architecture and run-flow maps"
```

---

## Final adversarial review and end-to-end verification

### Step 7A: Run structural deletion and ownership audits

```bash
rg -n 'import yaml|from yaml' src/agent_composer \
  -g '*.py'
rg -n '\b(SourceSpan|SourceFrame|FieldRanges|child_source|assert_texts|pre_assert_ranges|post_assert_ranges|InputDecl|ParamDecl)\b' \
  src/agent_composer tests
rg -n '\b(result|res|fresh|resumed|run_result)\.(error|error_type|traceback|locator)\b' \
  src/agent_composer tests
conda run -n agentcomposer env PYTHONPATH=src python -m pytest \
  tests/engine/test_failure_api_census.py \
  tests/engine/test_layer_direction.py tests/engine/test_kind_census.py -q
```

Expected: the PyYAML audit names only
`compose/source/document.py` and `compose/source/literals.py`; the explicit
legacy-symbol audit may name only intentional negative string literals in the
census test; the class-specific result-field audit has no matches; the import/
dataclass/AST census passes; and the kind census is zero. Generic `.error` is
not audited because unrelated terminal-controller and agent events legitimately
own error fields.

### Step 7B: Run full tests and docs from a clean status

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest tests/engine -q
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q
npm ci
conda run -n agentcomposer python scripts/check_mermaid.py docs
conda run -n agentcomposer python -m mkdocs build --strict
git status --short
```

Record exact versions and results in the issue/PR. Do not claim completion from
focused tests alone.

### Step 7C: Adversarial code review, PR, and merge

Review specifically:

- a custom resolver with dangling origins;
- aliases, duplicate keys, recursive anchors, diamond imports;
- MAP/LOOP execution-ID vs authored-origin separation;
- all nine failure code boundaries;
- arbitrary builder/reducer/log/observer errors staying host exceptions;
- append failure before apply/observer;
- terminal-unreachable replay with no failure record;
- source-free Python graph run/failure/replay;
- formatting-only terminal replay;
- CODE nested code objects and same-filename isolation;
- renderer/resolver failure not masking semantic failure;
- fingerprints/digests ignoring all provenance;
- diagrams matching final class names and ownership.

Fix every confirmed finding with a focused regression, rerun the complete pytest
suite, and commit only when both are green. Then
open the #135 PR with `Fixes #135`, link the posted plan comment, list both exact
Mermaid pins and every check, obtain review, merge, switch to `main`, pull
`--ff-only`, and rerun the docs checker plus narrow smoke tests on merged `main`.
Work is complete only after that audit.
