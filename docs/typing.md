# Types — the typed value system

> The `typesys` package (`values.py` + `types.py` + `pool.py`) — the **leaf** of the
> layer ladder, below `nodes`. Everything above depends on it; it depends on nothing
> internal. For the authoring surface (`type:` / `typedefs:`), see
> [Flow syntax → Types](syntax.md#types).

## Why a flow is typed

A flow is a function: typed values in, a typed value out, and every node produces a typed
value the next one reads. Two forces make the typing non-negotiable:

- **Durability.** A run can pause and resume in another process, so the engine serializes
  the whole pool to JSON and back. JSON is lossy — `1` and `1.0` come back the same. So
  every value carries its own type tag and decodes back to *exactly* the type it had.
- **Loud failure at the boundary.** A node returns a plain value; the engine checks it
  against the node's declared output type *at the moment it is written* to the pool — so a
  wrong shape fails there, not three nodes later.

```
   types.py  → authoring surface: "list[Signal]" / typedefs: → a runtime Type
   values.py → the leaf: ValueKind, the TypedValue containers, the write-boundary check
   pool.py   → VariablePool: the run's typed memory + the ${...} resolver
```

## `ValueKind` — the closed vocabulary

Every value is one of a fixed set of kinds. The enum's `.value` is the *one* spelling —
author token, wire tag, and display name at once (it is `str`, never `string`):

```
scalars   None  str  int  float  bool  date  datetime  object  file
lists     list[Any]  list[str]  list[int]  list[float]  list[bool]  list[object]
```

`date`/`datetime` are ISO-8601 strings, never auto-inferred. `file` is a reserved
placeholder (not yet authorable).

## `TypedValue` — self-describing, losslessly serializable

Each value is a frozen pydantic model pinning its `ValueKind` as a `Literal`
discriminator (`IntegerValue`, `ListStringValue`, …). Their union is *discriminated*:

```python
AnyValue = Annotated[Union[NoneValue, StringValue, IntegerValue, ...],
                     Field(discriminator="kind")]
```

On a round-trip pydantic reads the `kind` tag and decodes to the right subclass — so
`{"kind": "int", "value": 1}` becomes `IntegerValue(1)`, not a float. **That losslessness
is the primitive checkpoint/resume is built on.** Node code never touches a `TypedValue`:
`.to_object()` gives the plain Python value; wrapping happens only at the pool's write
boundary.

## Two constructors: infer vs. check

```
build_value(v)               → wrap v in its NATURAL type (infer)
build_value_as(declared, v)  → wrap v as `declared`, RAISE on mismatch (check)
```

`build_value` infers the kind (`bool` checked before `int`; empty list → `list[Any]`).
`build_value_as` is the **write-boundary check** `VariablePool.set` calls — it validates
against the declared type, raising `TypeCheckError`; the only coercion is widening (`int`
fills a `float` slot).

## `Type` — structure beyond the bare tag

A `ValueKind` says how a value *stores*; a `Type` says what it must *be* — a frozen
dataclass whose `kind` is the storage tag, with optional refining fields:

| Form | `Type` shape |
|------|--------------|
| scalar / flat list | `kind` only |
| **record** | `kind=OBJECT`, `fields={name: Type}`, `required={names}` (closed) |
| **variant** (enum) | `kind=STRING`, `tags={labels}` |
| **typed list** | `kind=LIST_*`, `element=<Type>` |
| **Optional[X]** | X's `Type` with `nullable=True` |

Authors write types as Python-typing strings; `parse_type` parses them via Python's own
`ast` (so the vocabulary *is* typing: `list[X]`, `Optional[X]`, `Literal[...]`, a bare
PascalCase registry name). `Union[...]` is rejected — model a tagged union as a
discriminated record. `read_typedefs` builds the per-flow `TypeRegistry`, `resolve_type`
resolves a parsed expr against it into a leaf `Type`.

## `VariablePool` — the run's typed memory

The runtime state primitive the engine owns. Narrow by design:

- **A node produces exactly ONE value.** `store[node_id]` is a single `TypedValue`;
  "multiple outputs" are fields of one `object`. Writes are **write-once** and typed.
- **Three read namespaces**, all via `${...}`:

  | Reference | Resolves to |
  |-----------|-------------|
  | `${input.X}` | field `X` of the flow's input |
  | `${node.output[.path]}` | node `node`'s value, dotting into an object |
  | `${system.X}` | a host-injected ambient (run id / clock); reserved, run-global |

  An unrecognized head resolves to `None` (the missing-ref-is-falsy contract — see
  [Expressions](expressions.md)). Because values are typed *objects*,
  `${x.output.ratio}` traverses **into** an object.
- **Lossless serialization.** `dumps()`/`loads()` round-trip the whole pool through the
  `AnyValue` union — the state half of a durable checkpoint.

## Why this shape

- **Durable by construction** — the discriminated union makes checkpoints lossless.
- **One vocabulary** — `ValueKind.value` is author token, wire tag, and display name at once.
- **Fail at the boundary** — nodes stay pure; `build_value_as` catches a wrong type early.
