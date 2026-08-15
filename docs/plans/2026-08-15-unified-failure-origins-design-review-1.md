VERDICT: NEEDS_REVISION

## Summary Assessment

The selected direction is sound: opaque flow-local `OriginRef`s, a compose-owned
source catalog, one immutable `Failure`, and resolution only at the presentation
boundary are simpler than the current symbolic-locator-plus-range system. The
existing `SpanLoader` also demonstrates that safe construction and mark collection
can happen in one PyYAML loader pass.

The design is not implementation-ready, however. It does not define how independently
compiled/cached child catalogs become one collision-free parent catalog, and it does
not define the provenance-bearing semantic carrier types needed to replace the current
parallel maps. It also assumes every failed run passes through `NodeFailedRecord`, which
is false for the existing terminal-unreachable path. Those gaps affect identity,
durability, replay, aliases, and nested growth rather than just implementation detail.

The revision should keep the four-authority model, but make ownership and transformations
concrete enough that each origin has exactly one carrier and every failure path has one
durable derivation.

## Critical Issues (must fix)

### 1. Flow-local origin keys have no defined catalog composition or rebasing rule

Each child is compiled before it is embedded. The current file resolver caches and returns
the same `ComposedFlow` by resolved path (`compose/loader.py:329-347`), while CALL/MAP/LOOP
bake `child.compiled` directly. If every independently compiled child uses keys such as
`root/nodes/check/input/x`, two different external files collide in the parent catalog.
The same cached child can also be reached through two aliases or a diamond import, so simply
mutating/rebasing the cached child at the first callsite makes the second embedding wrong.

The design's `uses:reviewer/...` example suggests rebasing, but no operation, owner, or cache
semantics are specified. The revision must define:

- a deterministic namespace for root, each `defs:` scope, inline lexical children, and
  recursively imported files;
- whether origins are finalized during child compilation or rebased on embedding;
- how rebasing copies a cached child without reparsing its physical `SourceDocument`;
- how one physical file imported through multiple aliases and diamond paths behaves;
- how a custom `child_resolver` contributes a catalog; and
- a compose-finalization invariant that every non-`None` origin reachable from the compiled
  graph resolves exactly once in the returned catalog.

The rule must remain stable across formatting-only edits and cannot fall back to absolute
paths, content digests, runtime IDs, or generated private IDs, per the stated decisions.
Without this rule nested external failures can resolve to the wrong document even though all
individual keys and ranges are valid.

### 2. “Semantic operations carry origins” needs actual canonical IR carrier types

The current IR cannot hold the proposed provenance without new shapes:

- `Flow.wiring` is `dict[node_id][param] -> raw literal/string`
  (`compile/model.py:76-94`), and `bind_params` reads those raw values directly
  (`nodes/binding.py:125-168`).
- assertions are strings plus `assert_texts` and positional range lists
  (`nodes/base.py:281-337`);
- CASE arms are plain `Case(handle, when)` values; and
- output validation reads a bare `Node.output_type`.

Adding parallel `binding_origins`, `assert_origins`, or `output_origin_paths` maps would
recreate the overlap the design is intended to remove. The revision should name the
canonical source-neutral operation values and their owning layer, for example a binding
value containing `(source, origin)`, an assertion value containing
`(expression, display_text, origin)`, and a CASE arm containing its own origins. It must
also define the output-contract shape at nested-field granularity.

The transformation rules must cover all non-obvious sites, not only ordinary authored
nodes: the top-level START wiring override, child START wiring replacement in
`clone_child` (`compile/expand.py:237-244`), MAP per-item binding, MAP/WAIT reserved wiring,
flow-boundary assertions, synthesized inline calls, cloned child END contracts, and LOOP
`respawn` (which currently manually copies only selected fields at
`nodes/loop/node.py:129-147`). Manifest/fingerprint code must canonicalize semantic values
while excluding provenance.

Nested output precision additionally requires a source-neutral structured error path.
The current type checker recurses into record fields and list elements without prefixing or
retaining a path (`typesys/values.py:467-489`), so the claimed `output.score` path does not
exist today. Define the path contract and how it selects the one operation origin. A
contract-local origin tree/index is acceptable if it is the sole provenance attached to
that contract; a second YAML/range map is not.

### 3. Not every terminal failure has a `NodeFailedRecord`

The proposed chain

`EvaluationFailure -> NodeFailedRecord(Failure) -> state/events/result`

is not exhaustive. A valid route or success record can leave END unreachable. The reducer
then derives `_TerminalDelta(succeeded=False)` (`runtime/state.py:2740-2788`) and materializes
`RunFailed(error_type="TerminalSkipped")` (`runtime/state.py:3242-3257`). There is deliberately
no extra failure record; the causal durable record remains `NodeRoutedRecord` or
`NodeSucceededRecord`.

The revision must choose and document one authority-preserving model:

- derive a canonical `Failure(code="terminal.unreachable", node_id=None, ...)`
  deterministically from the causal outcome record during both live reduction and replay,
  and narrow the claim that every `Failure` is stored verbatim in `NodeFailedRecord`; or
- introduce an explicit durable terminal-failure transition and redesign preparation so it
  still obeys one append-before-apply authority.

Do not append an after-the-fact second record after the causal outcome has already made state
terminal. Whichever model is selected must make live and replay results equal and explain why
node-less `RunFailed` has no `NodeFailed` observer.

### 4. The post-cutover load-error contract is missing

The goal promises one clean load-time/runtime diagnostic framework and says the parallel
`LoadError.line`, `.lines`, and `.range` channels will be removed, but the design never states
what `load_flow` raises afterward. “Malformed YAML creates a diagnostic directly” is not an
exception/API contract, and retaining the present `LoadError` shape would create a fifth
location authority.

Define one final shape. A simple option is a compose-owned `LoadError` whose only structured
payload is one already-resolved `Diagnostic` (with `str(error)` delegating to its message).
Ordinary cursor errors and malformed YAML can both use it; nested external-load errors can add
ordered frames without exposing origins outside compose. Pydantic `ValidationError.errors()`
locations must be translated before the current catch-and-stringify boundary. The design must
also state how notes and multi-site errors such as cycles are represented without resurrecting
`line`/`lines` aliases.

### 5. A shared immutable `SourceDocument` conflicts with current mutating parse passes

One parse is feasible, but the current “fresh tree per parse” assumption permits mutation.
`parse/nodes.py:142-162` renames key scalars and constructed mappings in place, and CASE parsing
replaces `reader.value` to undo YAML 1.1 `on:` coercion (`compose/nodes/case.py:131-138`). Reusing
one document makes those mutations observable by defs, aliases, catalogs, and later passes.

The revision must declare `SourceDocument` and its node tree read-only after construction and
make normalization/desugaring use occurrence-local cursor overlays. It must define structural
path semantics for:

- duplicate mapping keys (the current reader deliberately chooses the effective last pair at
  `compose/helpers/reader.py:98-119`);
- merge-key precedence;
- aliases that reuse the same PyYAML node identity but occur at multiple semantic paths; and
- normalized names such as surface `input` versus internal `inputs`.

The range index must therefore be occurrence-path based, not merely `id(yaml_node)` based,
even though multiple occurrence keys may intentionally resolve to the same definition-site
range. This is required to make the selected alias definition-site policy and unique origin
keys coexist.

### 6. Failure-code and host-boundary semantics are still deferred

`Failure.code` is described as stable and machine-facing, so its initial vocabulary and
classification are part of the public/wire design, not a planning detail. The representative
list omits existing cases including input-contract failure, invalid route data, and terminal
unreachability. It also does not distinguish an assertion that evaluates false from an
expression-evaluation defect if that distinction is intended.

The current dispatcher turns every injected growth-builder exception into a flow failure
(`runtime/engine.py:344-351`) while reducer/state invariant errors propagate. The revision must
state which builder, route, output, serialization, and malformed-program errors are author flow
failures versus host/compiler bugs. Otherwise `contract.growth` can accidentally durable-log a
programming defect that should abort outside the flow model.

Inline CODE precision also needs a concrete source-neutral protocol. A `ScalarNode` provides a
whole-scalar mark, not a general executable-line mapping for literal, folded, indented, and
chomped YAML scalar styles. Either restrict inline CODE to a documented scalar style or specify
and test the physical-to-constructed line algorithm. Define how CODE selects an `OriginRef`
from an exception while retaining the original `cause_type` and traceback, without learning
YAML concepts.

### 7. The requested ownership and run-flow diagrams need a reproducible tool contract

The design has no documentation/tooling slice for the newly required package/class ownership
and YAML-to-engine success/failure diagrams. Mermaid is the best fit for this repository only
with constraints: GitHub renders it natively, and MkDocs Material supports flow, class, and
sequence diagrams without committing generated images. It should be selected explicitly over
PlantUML/D2 because those add a Java/binary or generated-asset pipeline for little benefit here.

Do not assume Mermaid is currently configured. `mkdocs.yml:59-70` enables plain SuperFences but
has no `mermaid` custom fence. Material 9.7.6's client integration fetches a floating
`https://unpkg.com/mermaid@11/...` when no local runtime exists, `pyproject.toml:51` leaves the
Material version open-ended, and CI does not build docs (`.github/workflows/publish.yml:26-37`).
That combination is neither offline nor reproducible and will not catch invalid diagram syntax.

Add a delivery slice that:

- configures the Mermaid custom fence and an exact pinned/vendored Mermaid 11 runtime;
- validates diagrams and runs `mkdocs build --strict` in CI;
- limits syntax to the subset shared by the pinned renderer and GitHub; and
- adds at least (1) a package/class ownership diagram showing the diagnostics leaf,
  compose source model, semantic IR carriers, runtime records/events/results, and CLI resolver,
  and (2) a run-flow diagram with YAML load, normal success, recorded node failure,
  terminal-unreachable failure, replay, and diagnostic resolution branches.

## Suggestions (nice to have)

- Put `OriginRef`, `Failure`, `SourceRange`, `DiagnosticFrame`, and `Diagnostic` in one
  dependency-free diagnostics contract module/package below `events`; then let `events` embed
  `Failure`. Update package charters and extend the layer-direction ratchet to forbid that leaf
  importing engine/front-end packages.
- Make `SourceCatalog` and inline-CODE line tables immutable/shareable under `deepcopy`; MAP can
  clone many nodes and should not duplicate source text or large line maps per element.
- Validate `OriginRef.key` as an exact non-empty string, define segment escaping, and test
  adversarial node/def/alias names so two structural paths cannot stringify to the same key.
- Exclude all origins and catalogs from `flow_manifest`, but add tests proving that semantic
  binding/type/assert content still affects the fingerprint while comments and coordinates do
  not.
- Even with no legacy logs, increment or deliberately redefine the run-record schema version
  rather than silently assigning two incompatible payloads the same version.
- Treat a missing catalog entry as graceful at presentation time, but make it a compose build
  invariant for compose-produced origins. Graceful fallback should protect users from stale or
  third-party resolvers, not hide a compiler bug in newly loaded YAML.
- Renderers should catch resolution/presentation errors around `diagnose`, retain the original
  `Failure`, and optionally attach the resolver error as a note under engine-trace mode.

## Verified Claims

- The repeated-parse claim is accurate for a normal top-level file: `parse_file` calls both
  `yaml.safe_load` and `yaml.compose` (`compose/parse/file.py:219-251`), then `_load_flow` invokes
  `_nodes_reader`, `_section_lines`, `_boundary_field_ranges`, and `_boundary_assert_ranges`
  (`compose/loader.py:404-471`). Each multi-node def adds another `_def_nodes_reader` span parse.
- `SpanLoader` already proves safe construction and span collection can share one loader/tree
  (`compose/helpers/spanloader.py:34-104`), including current definition-site behavior for
  merged/anchored nodes.
- Keeping `OriginRef`/`Failure` in a new dependency-free leaf preserves the actual import
  ratchets: nodes can import the leaf without importing compile/runtime, runtime can import it
  without `compile.expand`/compose, and compose remains the only YAML owner.
- The runtime failure funnel is kind-blind today: `eval_node` catches exceptions generically
  (`runtime/eval_node.py:153-181`), and the dispatcher branches on `Output | Route | Grow |
  EvaluationFailure`, not `NodeKind` (`runtime/engine.py:257-342`). The proposed carrier does not
  require a kind branch.
- Append-before-apply can be retained: failure preparation already produces a canonical record
  before `commit_prepared`; changing the record payload to one immutable `Failure` does not
  require moving the append boundary.
- Programmatically constructed nodes/flows can remain source-free if every provenance field is
  optional. Nothing about `OriginRef` requires YAML; the coupling risk comes from where the
  semantic carrier classes are placed, not from the opaque key itself.
- The current external resolver's physical-file cache supports the parse-once goal, but its
  shared `ComposedFlow` values are precisely why catalog rebasing/composition must be explicit.
- Existing aliases intentionally report the anchor definition range because aliases share YAML
  node identity (`compose/helpers/spanloader.py:49-54`); the selected policy is implementable if
  occurrence identity and displayed range are kept distinct.
