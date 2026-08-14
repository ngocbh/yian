# Inline Composite Bodies

**Date:** 2026-08-14

**Issue:** [#6](https://github.com/ngocbh/agent-composer/issues/6)

**Status:** Validated design

## Goal

Make one-off CALL, MAP, and LOOP bodies concise. Today an author must lift even a
single-node body into top-level `defs:` and then name it with `call:`. The new form
lets the composite node define that child directly with `nodes:` and `output:`.

This is Compose-only sugar. It compiles to the same closed child-flow contract as
an authored `defs:` entry, and it does not change runtime scheduling, growth,
durability, or replay.

## Authoring Surface

Every CALL, MAP, and LOOP chooses exactly one child form:

- **Named child:** `call: <defs-or-uses-name>`.
- **Inline child:** a non-empty `nodes:` mapping plus a required `output:`.

`call:` and `nodes:` are mutually exclusive. The inline body may contain one or
multiple nodes; one-off single-node bodies are the primary ergonomic use case, but
an artificial one-node restriction would add no useful invariant.

```yaml
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      text: ${item.text}

    nodes:
      normalize:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: app.steps:normalize

      classify:
        kind: agent
        input:
          text: ${normalize.output}
        prompt: Classify ${text}
        output: str

    output: ${classify.output}
```

The corresponding named form remains available for reusable or explicitly typed
children:

```yaml
defs:
  classify_one:
    input:
      text: str
    nodes:
      classify:
        kind: agent
        input:
          text: ${input.text}
        prompt: Classify ${text}
        output: str
    output: ${classify.output}

nodes:
  each:
    kind: map
    over: ${input.items}
    call: classify_one
    input:
      text: ${item.text}
```

The inline child adds no configuration wrapper. Only `nodes:` and `output:` belong
to the synthesized body. `input:`, `asserts:`, `llm_config:`, `env:`, ordering,
MAP options, and LOOP drivers remain properties of the enclosing composite and
retain the same meaning for named and inline children.

## Function Boundary

An inline body is an anonymous function, not a lexical closure. The enclosing
`input:` mapping has two compile-time roles:

1. Its values are the caller bindings evaluated in the parent scope.
2. Its keys are the synthesized child's parameter names.

The child reads those values through `${input.<name>}`. It cannot directly read
the parent's `${item}` or another parent node's output. For MAP, `${item}` is valid
only in the enclosing MAP bindings:

```yaml
each:
  kind: map
  over: ${input.items}
  input:
    id: ${item.id}
  nodes:
    fetch:
      kind: agent
      input:
        id: ${input.id}
      prompt: Fetch details for ${id}
      output: str
  output: ${fetch.output}
```

This explicit boundary preserves flow-as-a-function, structural determinism,
cross-flow validation, and durable replay. Parent data needed by the child must be
named in the enclosing `input:` map.

## Input-Type Inference

A normal `defs:` entry declares parameter types. The inline form instead derives
each synthesized parameter type from its enclosing binding. The inference pass
accepts only types that the compiler can prove statically, including:

- declared flow inputs;
- declared producer outputs and their record fields;
- MAP `${item}` and its fields when `over:` has a known list element type; and
- literals with an unambiguous natural type.

The pass must not silently widen an unknown type to `object`. If any parameter type
cannot be established, loading fails at that exact binding with an actionable
message directing the author to use a named, explicitly typed `defs:` entry.

The inferred signature is fixed at load time and participates in the compiled
child and flow fingerprint. Runtime values do not determine or alter it.

## Compose Lowering

Lowering stays in `compose/` and occurs after located node parsing and static type
resolution but before the enclosing composite descriptor is built.

For each inline body, Compose:

1. Retains a located reader for the authored `nodes:` and `output:` subtree.
2. Infers a typed child input declaration from the enclosing `input:` bindings.
3. Recursively assembles the body into an ordinary `ComposedFlow`, using the
   declaring file's typedef registry and named callable resolver.
4. Registers that child under a deterministic private callable ID.
5. Replaces the inline fields on the enclosing descriptor with the generated
   `call:` target.
6. Runs the existing CALL/MAP/LOOP build, signature, type, and graph passes.

The runtime therefore receives the same baked child graph it receives for a named
def. CALL still returns the child value, MAP still wraps the child codomain as a
list and expands once per element, and LOOP still enforces and threads `'a -> 'a`.

Lowering is recursive. An inline body may itself contain an inline CALL, MAP, or
LOOP, and its assembly applies the same pass.

## Resolver Ownership and Private Identity

The feature does not create a resolver per inline body. Each loaded YAML file keeps
its existing composite callable resolver and gains one captured table:

```python
private_children: dict[str, ComposedFlow]
```

Top-level assembly, authored defs, and recursively inline bodies in that file share
the resolver and table. Resolution checks a generated private ID in this table and
otherwise follows the existing authored `defs:` then `uses:` path. External YAML
files retain their own per-file resolver, as they do today. MAP elements and LOOP
iterations never create resolvers; they reuse the child already baked at load.

Generated callable IDs use the specific reserved prefix
`__inline_flow_`, followed by a digest of the lexical callsite path. The identity
must not depend on object identity, traversal counters, or source line numbers.
Comments and line movement therefore leave it stable, while renaming or moving the
owning node changes the program identity as expected.

Only this exact prefix is reserved in authored `call:`, `defs:`, and `uses:` names;
the language does not reserve all double-underscore names. Existing `__start__`,
`__end__`, and `__call_N` values are graph node IDs inside a `CompiledFlow`, not
callable IDs, and never enter the private callable table.

The private identifier must not appear in author-facing diagnostics. Frames use a
label such as `inline body of node 'each'` and retain the original file text and
absolute source ranges.

## Validation and Diagnostics

Compose rejects these shapes before ordinary node construction:

- both `call:` and `nodes:`;
- neither `call:` nor `nodes:`;
- an empty or non-mapping inline `nodes:`;
- inline `nodes:` without `output:`; and
- `output:` on a named composite without inline `nodes:`.

Generated parameter names exactly match the enclosing `input:` keys. Unknown
parameter types point to the individual binding. Invalid references inside the
body follow ordinary closed-flow validation: direct `${item}` capture or a parent
node reference fails at the authored inner field.

Load failures name the owning composite and the inline body without exposing its
private ID. Runtime failures retain both the outer callsite and a child source
frame whose node and field ranges point into the original nested YAML.

## Verification

Tests will cover:

- CALL, MAP, and LOOP with one-node inline bodies;
- multi-node bodies and recursively nested inline composites;
- behavioral equivalence with manually written named defs;
- unchanged named `call:` resolution through `defs:` and `uses:`;
- type inference from flow inputs, producer fields, MAP items, and literals;
- actionable failure for a binding whose type is not statically known;
- mutually exclusive forms, missing output, empty nodes, and reserved-prefix use;
- rejection of `${item}` and parent-node capture inside the child;
- precise load-error and nested runtime-failure source locations;
- stable fingerprints under comments and line movement;
- fingerprint mismatch after an inline semantic change;
- MAP/LOOP expansion, durable replay, and recovery;
- identical composite assertions and LLM configuration behavior for named and
  inline forms; and
- the kind-census and layer-direction ratchets.

Documentation changes accompany implementation in `docs/syntax.md`, the package
README, and the engine and composing-agent skills. A load-tested authoring template
will demonstrate the concise MAP/LOOP form.

## Alternatives Rejected

**Mapping-valued `call:`.** Allowing `call:` to hold either a string or a complete
flow mapping represents named function versus lambda directly, but keeps a wrapper
and duplicates the child input signature. The selected `nodes:` form better targets
the one-off-body ergonomics.

**A `body:` or `child_config:` wrapper.** This disambiguates scopes but adds the
indentation and ceremony the feature is meant to remove. The inline grammar needs
only `nodes:` and `output:`.

**Lexical capture.** Letting inner nodes read `${item}` or parent outputs directly
would require closure capture, hidden dependencies, new type rules, and new replay
identity. Explicit enclosing inputs preserve the existing function boundary.

**Restrict inline bodies to exactly one node.** Single-node bodies motivate the
feature, but the same lowering naturally handles a small multi-node body. Enforcing
one node would be arbitrary and would not simplify the compiler meaningfully.

**Physically insert generated entries into authored `defs:`.** The semantics are
equivalent, but mutating the public dictionary complicates source paths and exposes
collision behavior. A resolver-private table preserves the normal child-flow path
without creating an author-visible name.

## Out of Scope

- Implicitly choosing a terminal node when `output:` is absent.
- Falling back to an untyped or `object` child parameter.
- A new lexical namespace for nested `defs:`, `uses:`, or `typedefs:`.
- Runtime node kinds, scheduler branches, records, or recovery behavior.
- Automatic capture of parent values not named in the enclosing `input:`.
