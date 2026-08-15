# Inline Composite Bodies — Final Revised Design

**Date:** 2026-08-14

**Issue:** [#6](https://github.com/ngocbh/agent-composer/issues/6)

**Status:** Implementation-ready

## Changes from v2

V3 preserves the accepted surface and resolver/build design, with six corrections
from the second adversarial review and read-only spike:

1. Parent masking is occurrence-local. It shallow-copies every occurrence's
   constructed mapping, `MappingNode` pair list, and key scalar nodes. Original
   anchor/merge nodes and value nodes remain read-only.
2. Mappings and lists are literal bindings. Their values are inferred recursively,
   but every nested string is `str`, even when it contains `${...}`. Only a
   top-level string is a binding template.
3. Dotted walks propagate nullability from the root and every intermediate record
   to the inferred result, including MAP item paths.
4. Canonical record keys and variant tags use canonical JSON quoting; structural
   record fields are sorted in both generated labels and generated `Type` values.
5. A list stored as `LIST_ANY` is proven when it has a fully proven `element`.
   Only a list with no element type is unproven.
6. Lazy-build stack frames retain the requesting binding's label and range, making
   located self/mutual dependency paths concrete rather than aspirational.

## Goal and Authoring Surface

Allow a one-off CALL, MAP, or LOOP body to be written directly on the composite:

```yaml
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      text: ${item.text}

    nodes:
      classify:
        kind: agent
        input:
          text: ${input.text}
        prompt: Classify ${text}
        output: str

    output: ${classify.output}
```

Every CALL, MAP, and LOOP chooses exactly one child form:

- named: `call: <defs-or-uses-name>`;
- inline: a non-empty `nodes:` mapping plus required `output:`.

`call:` and `nodes:` are mutually exclusive. `output:` is legal on the composite
only with inline `nodes:`. It accepts the same single binding or named binding map
as a normal flow `output:`; the synthesized child must still satisfy the existing
CALL, MAP, or LOOP codomain contract. Inline bodies may contain one or many nodes
and may recursively contain inline composites.

The enclosing `input:` must be a mapping with string keys. Its values are caller
bindings evaluated in the parent; its keys become the private child's parameters.
The child reads them through `${input.<key>}`. It is a closed anonymous function,
not a closure: direct `${item}` and parent-node references inside the body fail the
ordinary child reference checks. For MAP, `${item}` is legal only in the outer MAP
`input:` values.

This is Compose-only sugar. Runtime kinds, growth, scheduling, run records,
checkpoints, replay, and recovery do not change.

## Configuration Semantics

Only `nodes:` and `output:` belong to the synthesized child. All outer fields keep
their current named-form behavior exactly:

- CALL keeps its outer `input:` and current `asserts:` behavior.
- LOOP keeps its carried `input:`, predicate/limit, and assertion behavior.
- MAP keeps `over:`, `input:`, and `parallel:` on the driver. Although its
  descriptor inherits the parsed `asserts:` key, current validation rejects all MAP
  assertions; the inline form remains rejected too.
- Common `env:` is baked on the outer CALL/MAP/LOOP node, not copied to the child.
  Inner nodes may author their own `env:` normally.
- CALL/MAP/LOOP have no callsite `llm_config:` today, and #6 does not add one. The
  existing flow-level LLM cascade descends into an inline baked child exactly as it
  descends into an equivalent named child without child-local configuration.

Per-callsite LLM configuration and MAP-result assertions require separate language
contracts and are out of scope.

## One Per-file Callable Scope

The loader creates one callable scope after the file registry/external resolver but
before eager defs:

```python
class CallableScope:
    def resolve(self, flow_id: str, version: str | None = None) -> ComposedFlow: ...
    def register_private(
        self,
        flow_id: str,
        owner: tuple[str, ...],
        label: str,
        build: Callable[[], ComposedFlow],
    ) -> None: ...
    def eager_load_defs(self) -> None: ...
```

It owns defs/uses, the external resolver, private declarations, successful child
caches, and in-progress guards. `_assemble`, `_load_def`, and recursive inline
assembly share it; `BuildContext.resolver` receives `scope.resolve`.
`register_private` installs a lazy builder before its enclosing descriptor can
resolve it. `resolve` builds once and caches. Registration is idempotent only for the
same ID and complete lexical owner; any other collision is loud.

This lets eager defs contain inline bodies because the scope exists before
`eager_load_defs()`. Each external YAML file owns its own scope. Inline bodies, MAP
elements, and LOOP iterations do not create resolvers.

## Private Identity

Reserve the exact callable prefix `__inline_flow_` at authored callable-name sites:
`defs:` names, `uses:` aliases, named `call:` values, and whole-value `call(...)`
callees. Validate those names before lowering. Do not reserve all `__` names or this
prefix for graph node IDs.

Lexical owners are unambiguous tuples:

```text
("root", "node", <id>, "inline")
("def", <def-name>, "node", <id>, "inline")
(<parent-owner...>, "node", <nested-id>, "inline")
```

The private ID is `__inline_flow_` plus the full SHA-256 digest of the tuple's
canonical JSON array encoding. Root/def identity, def name, and all enclosing inline
nodes participate; line numbers, comments, traversal counters, and object identity do
not. Formatting is identity-neutral, while moving/renaming the owner changes program
identity. The scope still collision-checks every registration.

`__start__`, `__end__`, and `__call_N` remain graph node IDs and never enter the
callable scope.

## Occurrence-local Located Overlays

The file is parsed into one read-only located root. The original constructed values,
YAML node tree, and span table are never changed. This is required because aliases
reuse constructed objects and YAML node identities, while merge expansion can reuse
key/value nodes from an anchor.

Before parsing one `nodes:` occurrence, create a shallow located overlay:

1. Copy that occurrence's constructed nodes mapping.
2. Shallow-copy its `MappingNode`, replace its `.value` with a new pair list, and
   shallow-copy **every** key scalar node in that list. Reuse value nodes read-only.
3. Give the wrapper mapping the original mapping range and every copied key the
   original key range through a small overlay span map that delegates all other
   lookups to the read-only base spans.
4. For every child occurrence, shallow-copy its constructed body mapping and its
   `MappingNode`/pair list/key nodes by the same rule. Install that distinct body
   overlay in both the parent overlay's constructed mapping and the corresponding
   parent YAML pair. Therefore `a: *body` and `b: *body` receive separate occurrence
   wrappers even though their read-only source value node is shared.

This operation is shallow by design. Original value nodes, including an inline
`nodes:` subtree and `output:` scalar/mapping, retain their identity and original
ranges. Recursive inline assembly creates a fresh occurrence overlay when it later
parses that retained subtree. No copied wrapper or key is installed back into the
anchor, alias target, merge source, or global root.

After overlaying, extraction handles each composite occurrence:

1. Validate exactly one of named `call:` or inline `nodes:`. Reject missing,
   empty, or non-mapping `nodes:`, missing `output:`, named `call:` plus `output:`,
   and a non-mapping/non-string-keyed outer `input:` at their own ranges.
2. Retain readers for the occurrence's inline `nodes:` and `output:` plus each outer
   `input:` value range.
3. Mask only the occurrence overlay: in its constructed body replace `nodes:` with
   generated `call: <private-id>` and remove `output:`; in its copied YAML pair list
   rename the copied `nodes` key to `call` and remove the `output` pair. The original
   `nodes` value node remains the synthetic call field's range anchor.
4. Parse the masked parent, run existing descriptor desugars, associate retained
   bodies with final owner descriptors, and register their private builders.
5. Derive `SourceFrame` maps and `n_lines` from those parsed masked descriptors,
   never by passing unlowered inline syntax to ordinary `parse_nodes`.

The same overlay helper serves top-level nodes, `defs/<name>/nodes`, recursively
inline nodes, and `top_frame_ranges(text)`. A nested child frame uses label
`inline body of node '<id>'`, the declaring file's text, and absolute original
ranges. Generated START input-declaration ranges point to individual outer `input:`
values. The child output range is the retained inline `output:` range and is threaded
to reference validation, so a bad nested output points to the authored value.

## Lazy Cached Parent Construction

Parent assembly builds descriptors through one cache. A request frame retains both
the node being requested and the binding that requested it:

```python
@dataclass(frozen=True)
class BuildRequest:
    node_id: str
    via_label: str | None       # e.g. "a.input.x"
    via_range: SourceRange | None

built: dict[str, BuildResult] = {}
building: list[BuildRequest] = []

def build_one(node_id: str, via_label=None, via_range=None) -> BuildResult:
    if node_id in built:
        return built[node_id]
    if any(frame.node_id == node_id for frame in building):
        raise located_type_cycle(building, node_id, via_label, via_range)
    building.append(BuildRequest(node_id, via_label, via_range))
    try:
        result = descriptors[node_id].build(ctx)
        built[node_id] = result
        return result
    finally:
        building.pop()

for node_id in descriptors:              # source order retained
    build_one(node_id)                    # inference-time results are reused
```

An inline private builder infers each parameter from the final outer descriptor.
A `${producer.output...}` request calls `build_one(producer, binding_label,
binding_range)`, so forward references work. Named and inline composite producers
resolve/bake through the shared callable scope. No descriptor is built twice.

On a cycle, find the earlier stack occurrence and format the intervening request
labels plus the closing request, for example `a.input.x -> b.input.y -> a`. Anchor
the `LoadError` at the closing requesting binding's range (falling back to the first
available request range). Keeping every frame's label/range also permits future
multi-range rendering without reconstructing provenance. Ordinary DAG cycle
validation remains authoritative for cycles unrelated to signature inference.

## Strict Input-type Inference

Inline inference is a strict helper, separate from the lenient
`_output_value_type`/post-build call check. It produces a proven structural `Type` or
a located message directing the author to an explicitly typed named `defs:` body.

There are two string modes:

- `infer_binding(value)` applies template semantics only when the **top-level**
  binding value is a string.
- `infer_literal(value)` is used recursively inside a mapping/list. Every string in
  this mode is literal `str`, including `${producer.output}`, `${item.x}`, and `$$`.
  This matches runtime `_resolve_source`, which returns every non-string top-level
  container unchanged; nested strings are neither evaluated nor reference-walked.

The exact table is:

| Source | Result |
| --- | --- |
| YAML `null`, bool, int, float | `None`, `bool`, `int`, `float`; test bool before int. |
| Top-level string with no span | `str`. |
| Top-level string with literal text plus spans, or multiple spans | `str`, matching template concatenation. Its refs remain subject to ordinary validation. |
| Top-level string containing exactly one direct-reference span | Resolve by the direct-reference rules below. |
| Top-level single span containing a scalar literal | The literal's scalar type. |
| Top-level single span containing list expression, arithmetic, comparison, boolean/unary operation, builtin, coalesce, default, or required | Reject; there is no static expression signature algebra. |
| Mapping in either mode | Closed record inferred recursively with `infer_literal`; keys must be strings. Empty mapping is an exact empty record. Sort fields by key in the resulting `Type`. |
| Non-empty list in either mode | Infer elements with `infer_literal`. Structurally equal elements produce `list[T]`; int/float mixtures widen to `list[float]`. |
| Empty list | Reject because no element type is provable. |
| Other heterogeneous list, including null mixed with another type or different record shapes | Reject; never invent `Any`/`object` as a fallback. |
| Nested string at any container depth | Always literal `str`; do not scan `${...}` or `$$`. |
| Unsupported Python/YAML object | Reject. |

Direct references resolve as follows:

- `${input.name}` begins with the parent's declared `InputDecl.type`.
- `${node.output}` calls `build_one(node, binding_label, binding_range)` and begins
  with the producer's declared output type.
- `${item}` is permitted only while inferring an outer MAP `input:`. Infer the MAP
  `over:` through this same strict top-level table, require a list with a proven
  element, and begin with that element type.
- Any further path component requires a proven closed-record `fields` map and an
  existing field. `${system...}`, invalid/retired heads, `${item...}` outside a MAP
  input, opaque producers, and unknown fields fail strict inference.

Nullability is accumulated during a dotted walk. Initialize
`path_nullable = root.nullable`. Before and after each field step, OR the current and
selected field's nullability into it. Return a canonicalized copy of the final field
with `nullable = final.nullable or path_nullable`. Thus an optional root record, any
optional intermediate record, an optional field, or a nullable MAP element makes the
final inferred value nullable, matching runtime dotted misses. The same algorithm is
used for input, producer, and item paths.

A type is fully proven when all of its structural refinements are proven:

- records have a real `fields` map; each field is required or nullable, and every
  field type is recursively proven;
- every list storage kind, **including `LIST_ANY`**, is proven when `element` is not
  `None` and that element is recursively proven;
- a list with `element is None` is unproven;
- scalars, explicit unrefined `object`, variants, date/datetime/file, and required
  `None` are valid declared types.

The `LIST_ANY` rule preserves exact structural types whose storage tag has no more
specific carrier: `list[None]`, nested lists, and lists of date/datetime/file can all
carry a proven `element`. It does not permit empty/heterogeneous literal fallback.

The existing multi-output fallback uses a non-null `None` field omitted from the
record's `required` set as an opaque sentinel. Reject that invariant violation; a
genuine required `None` field is present in `required`, while a genuine optional
field has `nullable=True`.

## Generated Input Declarations and Canonical Labels

`_assemble` accepts exactly one input source:

```python
inputs_section: dict | None          # authored declarations
input_decls: list[InputDecl] | None  # already-resolved declarations
```

Top-level flows and named defs still call `read_flow_inputs`. Inline children pass:

```python
InputDecl(
    name=key,
    type_str=canonical_type_label(canonical_type),
    default=None,
    required=not canonical_type.nullable,
    type=canonical_type,
)
```

Canonicalization recursively copies the inferred `Type`, sorting record fields by
Unicode code-point key order and recomputing `required` against those fields. It also
sorts variant tags. It does not mutate the producer's type.

`canonical_type_label` is deterministic and unambiguous:

- scalar labels use their author vocabulary (`str`, `int`, `float`, `bool`, `date`,
  `datetime`, `object`, `file`, `None`);
- list labels are `list[<element-label>]`, based on the structural element even when
  storage is `LIST_ANY`;
- variant labels are `Literal[<tag>,...]`, with tags sorted and each tag encoded by
  `json.dumps(tag, ensure_ascii=False, separators=(",", ":"))`;
- record labels are `{<key>:<type>,...}`, with sorted fields and every key encoded by
  the same canonical JSON-string function;
- nullable types wrap the non-null label as `Optional[<label>]`.

JSON quoting/escaping prevents a key such as `"a: int, b"` or a tag containing a
quote/comma from colliding with another structure. The label is display/manifest
identity data and is never reparsed; the structural `Type` remains authoritative.
Generated declarations have no defaults, and `required` follows root nullability.

## Author-facing Diagnostics

Each private ID has label `inline body of node '<id>'`. An internal, non-authorable
`child_label` on the lowered descriptor is used by `_resolve_child`, child-binding
checks, MAP codomain errors, and LOOP field/type-contract errors instead of the
private ID. Runtime nodes retain the private `flow_id` for stable manifest identity,
while ordinary runtime failures render the retained nested source frame.

Complete negative diagnostics—syntax, inference, cycles, MAP/LOOP contracts, nested
references, and runtime inner-node failures—must not contain `__inline_flow_`.

## Verification

Focused tests cover:

- inline CALL/MAP/LOOP; one-node, multi-node, and recursively nested bodies;
- equivalence to named defs and unchanged defs/uses resolution;
- forward producer refs, single-build caching, and located self/mutual type cycles
  whose messages contain the retained binding path;
- all strict-inference rows, nullable root/intermediate record walks, nullable MAP
  record elements, and proven `LIST_ANY` element cases;
- a nested `${producer.output}` string in a literal mapping/list remaining literal
  `str`, creating no producer edge, and arriving unchanged in the child;
- two different node occurrences using the same YAML `*alias`: each receives its own
  lexical private ID/child and masking one never changes the other or anchor;
- a `<<:`-merged inline body: expanded shared keys remain intact and the occurrence
  compiles with precise original ranges;
- generated declaration fields, JSON escaping for adversarial keys/tags, sorted
  structural fields, and stable manifests;
- exclusive-shape/reserved-prefix errors and absence of private IDs in diagnostics;
- outer-frame preservation, individual binding/output ranges, and nested runtime
  traceback frames;
- stable fingerprint under comments/line movement and mismatch after semantic edits;
- current CALL/LOOP assertions, MAP assertion rejection, driver `env:`, and equal
  flow-level LLM cascade for named/inline unconfigured children;
- MAP/LOOP expansion and recovery through unchanged runtime behavior; and
- kind-census and import-layer ratchets.

Documentation changes accompany code in `docs/syntax.md`, the engine README, and the
engine/composing-agent skills. A load-tested authoring template demonstrates concise
MAP and LOOP bodies.

## Out of Scope

- implicit output/terminal selection;
- runtime-derived or `Any`/`object` fallback inference;
- recursive template evaluation inside literal containers;
- lexical capture or nested defs/uses/typedefs namespaces;
- callsite `llm_config:` or new MAP assertion semantics;
- runtime, scheduler, run-log, checkpoint, or recovery changes.
