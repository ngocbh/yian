# Unified Failures and Source Origins

**Date:** 2026-08-15

**Issue:** [#110](https://github.com/ngocbh/agent-composer/issues/110)

**Related issues:** [#81](https://github.com/ngocbh/agent-composer/issues/81),
[#86](https://github.com/ngocbh/agent-composer/issues/86),
[#91](https://github.com/ngocbh/agent-composer/issues/91),
[#100](https://github.com/ngocbh/agent-composer/issues/100), and
[#103](https://github.com/ngocbh/agent-composer/issues/103)

**Status:** Validated design; implementation planning remains

## Goal

Give load-time and runtime failures one clean, structured contract while keeping
all YAML parsing and source lookup inside `compose/`. A physical YAML file is read
once and parsed once. The engine carries only optional opaque provenance, so a
future Python-authored graph can use the same compiler and runtime without knowing
about YAML or providing source metadata.

The resulting design has four authorities:

1. `compose.SourceDocument` is the single parsed representation of one YAML file.
2. `OriginRef` is the engine-visible, source-neutral identity of an authored site.
3. `Failure` is the one durable semantic description of an uncaught flow failure.
4. `Diagnostic` is a resolved, presentation-only view produced by an authoring
   front end.

No layer carries two competing descriptions of the same location or failure.

## Non-goals

- Retry, fallback, default, catch, or other author-visible recovery strategies.
  Those need their own durable transition semantics and should build on `Failure`.
- Changing the node result algebra (`Output | Route | Pause | Grow`). An uncaught
  exception remains an evaluator failure, not an ordinary node result.
- Giving the runtime any YAML parser, YAML path vocabulary, or source renderer.
- Preserving old JSONL failure payloads. There are no existing run logs that need
  migration, so the wire format will make a clean cutover.
- Designing the future Python graph-authoring API. This work only ensures that
  Python-authored `CompiledFlow`s can omit source metadata and still run correctly.

## Current Problems

A plain valid flow currently undergoes six full YAML document parses:

1. `yaml.safe_load` in `parse_file`;
2. `yaml.compose` for top-level lines;
3. `SpanLoader` for the top-level node reader;
4. `SpanLoader` for section lines;
5. `SpanLoader` for boundary input ranges;
6. `SpanLoader` for boundary assertion ranges.

Each eager multi-node `defs:` body adds another full parse. A file with two defs is
therefore parsed eight times even though the CLI reads it from disk once.

Failure data is similarly duplicated. The evaluator, failure record, reducer delta,
node event, run event, and public result repeatedly copy some combination of
`node_id`, message, exception type, symbolic locator, concrete range, and traceback.
The CLI then scans invocation-local events and resolves source metadata yet again.

That duplication already causes observable drift:

- nested child failures can carry the correct concrete input range while the CLI
  chooses a node header or `code:` range instead (#103);
- output failures use symbolic `output` while descriptor maps use `outputs`, and a
  CLI-only alias repairs the mismatch;
- load errors carry `line`, `lines`, and `range` as parallel authorities (#91);
- replayed terminal results lose information because `RunResult` omits the failing
  node and exception type, then the CLI tries to recover them from events.

## Layer Boundary

The dependency direction is:

```text
source-neutral diagnostics contracts
          │
          ├── nodes / compile / runtime
          │
          └── compose (YAML front end)
                    │
                    └── package run boundary / CLI
```

The shared contracts import neither YAML nor `compose`. A small leaf module owns
`OriginRef`, `Failure`, and the source-neutral resolved diagnostic types. `events`
embeds `Failure`; it does not define another failure shape.

`compose/` alone owns:

- `SpanLoader` and all PyYAML interaction;
- `SourceDocument`, source cursors, structural YAML paths, and source catalogs;
- generation and resolution of origin keys;
- load-time source diagnostics;
- YAML labels, text, and source frames.

`nodes`, `compile`, and `runtime` may carry an optional `OriginRef`, but never
inspect it. They never import `yaml` or `compose`.

## Source-neutral Contracts

### OriginRef

```python
@dataclass(frozen=True)
class OriginRef:
    key: str
```

The key is stable and unique within one compiled flow identity. It is opaque to the
engine and serialized unchanged in durable records. Only the producing front end
assigns meaning to it.

Compose may internally generate keys resembling:

```text
root/nodes/score/input/x
root/defs/enrich/nodes/fetch/code
uses:reviewer/nodes/check/asserts/0
```

That spelling is not an engine API. It demonstrates the identity rule: comments,
whitespace, indentation, and absolute line numbers do not participate. Runtime node
IDs and clone suffixes also do not participate.

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

`code` is the stable machine-facing category; `message` is user-facing. `node_id`
is execution identity and deliberately remains separate from source identity.
`trace` is outermost-to-innermost, with the final element naming the precise failing
operation. `cause_type` and `traceback` retain Python debugging information without
serializing exception objects.

Representative code namespaces include `binding.evaluate`, `assertion.failed`,
`node.execute`, `contract.output`, and `contract.growth`. The implementation plan
will census current failure funnels before finalizing the closed initial vocabulary.

The exact same immutable value is embedded in:

```text
evaluator failure outcome
    -> NodeFailedRecord
    -> reducer state
    -> NodeFailed / RunFailed
    -> RunResult.failure
```

These containers retain their distinct lifecycle meanings but do not flatten or
copy the failure fields. `RunResult` no longer requires the CLI to scan events for
the authoritative failure.

### Diagnostic

A `Diagnostic` is a resolved presentation value, not a durable engine fact. It
contains the failure/load-error code and message plus zero or more ordered source
frames, notes, and an optional traceback. A frame contains a display label, source
text or a source-text handle, and a generic range.

`ComposedFlow.diagnose(failure)` resolves origin references through its compose-owned
catalog. A future Python front end may provide a Python-specific resolver. With no
resolver, the failure still displays its code, node ID, message, cause type, and
traceback.

## One-parse Compose Source Model

Each physical YAML file is parsed once:

```python
SourceDocument.parse(text, label)
    -> constructed value
    -> YAML node tree
    -> structural-path range index
```

The one `SpanLoader` pass performs both safe construction and mark collection,
including anchors and merge keys. `SourceDocument` retains the text, constructed
value, YAML root, and range index. A lightweight cursor pairs a structural path with
the corresponding value/node and can descend without reparsing.

The root loader, node parser, every `defs:` body, boundary inputs, assertions,
outputs, and source diagnostics all receive cursors into the same document. Each
uncached external YAML file creates its own document once and then enters the
existing load-context cache.

This deletes the need for independent `_nodes_reader`, `_def_nodes_reader`,
`_section_lines`, `_boundary_field_ranges`, `_boundary_assert_ranges`, and
`top_frame_ranges(text)` parses. It also removes the node-identity lifetime hazard
created by repeatedly constructing unrelated YAML trees.

A compose-owned `SourceCatalog` maps each `OriginRef` to a source site containing
the document identity, structural path, and current range. The split between source
document and site remains private to compose; the engine sees the single key only.

Malformed YAML is the sole direct-coordinate exception. When parsing fails before
a document or stable site exists, compose creates a diagnostic directly from
PyYAML's problem mark.

## Origin-carrying Compilation

Source location belongs to the semantic operation that can fail, not to a global
string-keyed range map. The compiled contracts therefore gain optional provenance
at the relevant granularity:

- every authored or synthesized node has a default origin;
- input bindings carry their value origin;
- assertions carry their indexed item origin;
- output contracts carry their declaration/binding origin;
- CASE arms and other ordered entries carry indexed origins;
- boundary input/output/assert operations carry real author origins.

Compose supplies these values while lowering YAML. A programmatic/Python builder
supplies `None` unless it has its own provenance system.

Desugaring must preserve an existing origin or assign one explicit authored origin
to each generated semantic operation. A synthesized node never inherits another
node's entire source map, fabricates a coordinate, or depends on a generated private
identifier for source lookup.

Structured validator paths must survive to this boundary. Pydantic locations such
as `llm_config.provider` and type-checker paths such as `output.score` resolve through
the same source cursor instead of being collapsed to a section line.

The cutover removes `SourceSpan`, `FieldRanges`, positional assertion/CASE range
side channels, legacy node/section line maps, and singular/plural renderer aliases.

## Runtime Failure Flow

The evaluator catches binding or node exceptions and constructs one `Failure`.
Binding evaluation uses the binding's origin; a raw node exception falls back to the
node origin. A node-owned exception may select a more precise optional origin, but
it cannot name a YAML field or range.

Dispatcher-owned contract failures use the origin of the operation being committed,
such as the output contract. The runtime continues to match only on its closed
worker-result and record variants; it never dispatches on `NodeKind`.

Append-before-apply remains unchanged:

```text
evaluate
  -> construct Failure
  -> prepare NodeFailedRecord(failure)
  -> append record
  -> apply the reducer
  -> publish events
```

The JSONL codec serializes the new `Failure` payload directly. Because there are no
existing logs to preserve, old failure payload readers and compatibility aliases are
removed rather than carried indefinitely. Recovery folds the new record through the
same reducer and yields an equal `RunResult.failure`.

Host/infrastructure errors remain outside `Failure`: invalid log access, fingerprint
mismatch, corrupt records, reducer invariant violations, and observer/renderer bugs
are ordinary boundary exceptions.

## Nested Provenance

Each compiled node has an optional default origin and an ordered origin context.
When a CALL, MAP, or LOOP grows a child graph, generic graph construction prefixes
the parent context plus callsite origin to every child clone. This is ordinary
provenance propagation; runtime does not inspect the keys or branch on node kind.

When a child operation fails:

```python
failure.trace = node.context + (operation_origin_or_node_origin,)
```

MAP iterations may share source origins while retaining distinct execution IDs such
as `review#3/check`. Source identity describes authored program structure; execution
identity describes one invocation.

The resolved trace directly produces outer-to-inner diagnostic frames. The CLI no
longer walks runtime IDs, strips `#`/`~` clone suffixes, follows baked child graphs,
or chooses a fallback range ahead of a precise leaf range. This resolves #103 by
removing the competing lookup path rather than patching its precedence.

Special cases use the same rule:

- duplicate assertions and CASE items use their sequence indices;
- MAP/WAIT reserved bindings use the binding origin (#86);
- output/type failures use the output-contract origin;
- inline CODE receives a compose-generated mapping from executable Python lines to
  opaque origins, letting the CODE node select an origin without understanding YAML;
- flow boundary nodes receive the real input/output/assertion origins;
- aliases retain the current definition-site behavior unless occurrence-site
  tracking is designed separately.

If a reference cannot resolve during presentation, rendering falls back to the
unlocated failure. Diagnostic resolution must never replace or hide the failure.

## Formatting-neutral Recovery

Flow identity deliberately excludes comments and source coordinates. Persisting only
concrete ranges would therefore make a resumed error point into stale text after a
formatting-only edit.

Durable failures instead store stable `OriginRef`s. Loading the semantically
equivalent reformatted flow rebuilds the same origin keys and maps them to the new
ranges. The failure record remains authoritative while presentation follows the
current source layout.

An origin key only needs stability within the compiled flow identity. It is not a
global source URI, filesystem path, content digest, or public YAML-path syntax.

## Python-authored Graphs

The executable contracts treat all provenance as optional. A Python-authored graph
can construct the same compiled nodes, bindings, assertions, and output contracts
with no origins and no source catalog. Runtime behavior, durability, and failure
codes remain identical; only source frames are absent.

A future Python front end may assign its own `OriginRef`s and resolve them to Python
files or notebook cells. That capability requires no runtime change because origin
keys are already source-neutral and opaque.

No YAML document, cursor, structural path, catalog, or range map belongs on
`CompiledFlow` merely to serve the YAML front end. The compose-level wrapper retains
the catalog beside the compiled program and resolves failures after execution.

## Alternatives Considered

### Keep the current symbolic locator plus concrete range

This is the smallest patch: fix #103 precedence, add missing public fields, and
centralize a few helpers. It retains two authorities, renderer aliases, stale ranges
after formatting, and repeated failure-field copying. Rejected.

### Put resolved YAML locations directly in engine failures

This simplifies immediate rendering but makes durable coordinates stale under
formatting-neutral replay and couples engine contracts to one authoring front end.
It also makes Python-authored graphs inherit irrelevant YAML concepts. Rejected.

### Resolve all locations by walking the graph after failure

This keeps runtime data small but recreates the current CLI problem: execution IDs,
growth clones, source structure, and exact failing operations must be reverse
engineered after the fact. Nested failures already demonstrate that this drifts.
Rejected.

### Opaque origin plus compose-owned catalog

This design keeps one stable durable identity, one source parser, and one resolver,
while keeping execution independent of authoring format. Selected.

## Delivery Slices

The architecture is unified, but implementation should land as small dependent
issues/PRs with green tests at every boundary:

1. **Single-parse SourceDocument foundation.** Introduce the immutable document,
   cursors, and path index; prove one parse per physical file without changing
   diagnostics.
2. **Origin-carrying compiled contracts.** Add `OriginRef` and attach optional origins
   to semantic operations; replace shape-specific source maps.
3. **Canonical Failure carrier.** Replace flattened failure fields and the JSONL
   payload across evaluator, records, reducer, events, and public results.
4. **Nested diagnostic cutover.** Propagate traces, add compose resolution, switch the
   CLI to one renderer, and delete all legacy resolution paths. This closes #91 and
   #103 rather than patching them independently.
5. **Inline CODE mapping and remaining precision cases.** Remove source padding and
   prove embedded-code line mapping, duplicate items, boundary sites, and synthetic
   origins.

#100 is a live reference-validation hole, not merely a location bug. Restore its
WAIT validation on `main` in a focused prerequisite/follow-up rather than hiding the
behavior change inside source plumbing. #86 is an acceptance case for origin-carrying
bindings and may close through the cutover.

Retry/fallback/catch work (#63) depends on the canonical failure carrier but remains
a separate semantic design.

## Acceptance Tests

### Parse and ownership

- A plain file, a file with multiple `defs:`, and each uncached external file each
  instantiate exactly one full-document YAML loader.
- `runtime` and `compile` import neither `yaml` nor `compose`.
- A Python-built graph compiles, runs, fails, and replays without source metadata.

### Location precision

- Root, `defs:`, external `uses:`, CALL, MAP, and LOOP failures resolve the correct
  outer-to-inner frames.
- Input bindings, duplicate assertions, CASE entries, boundary operations,
  MAP/WAIT reserved bindings, output contracts, and inline CODE tracebacks point to
  their precise authored sites.
- Malformed YAML reports PyYAML's problem mark without requiring a catalog.
- Nested failures never substitute a node header or `code:` field for a precise
  innermost binding origin.

### Durability and public parity

- Memory and JSONL logs round-trip an equal `Failure`.
- Live execution and terminal replay return equal `RunResult.failure` values even
  when replay emits no invocation-local events.
- A formatting-only YAML change preserves the flow fingerprint and origin keys while
  resolving to the new line/column ranges.
- Library and CLI callers see the same failure code, node ID, message, trace, cause,
  and optional traceback.

### Engine invariants

- Append-before-apply and recovery-prefix tests remain green.
- The runtime kind-census remains zero.
- Growth provenance works recursively without `NodeKind` dispatch.
- A missing/broken diagnostic resolver degrades to an unlocated failure and never
  changes run status or masks the original failure.
