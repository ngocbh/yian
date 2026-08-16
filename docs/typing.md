# Types — the typed value system

> The `typesys` package (`values.py` + `types.py` + `pool.py`) — the **leaf** of the
> layer ladder, below `nodes`. Everything above depends on it; it depends on nothing
> internal. For the authoring surface (`type:` / `typedefs:`), see
> [Flow syntax → Types](syntax.md#types).

## Why a flow is typed

A flow is a function: typed values in, a typed value out, and every node produces a typed
value the next one reads. Two forces make the typing non-negotiable:

- **Durability.** A run can be recovered in another process by replaying its
  authoritative record prefix. The initial record carries the whole pool and later
  success records carry typed outputs. JSON alone is lossy — `1` and `1.0` can collapse
  to the same representation — so every value carries its own type tag and decodes back
  to *exactly* the type it had.
- **Loud failure at the boundary.** A node returns a plain value; the engine checks it
  against the node's declared output type before appending its success record and reducing
  the write into the pool — so a wrong shape fails there, not three nodes later.

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
`{"kind": "int", "value": 1}` becomes `IntegerValue(1)`, not a float. **That
losslessness is what lets the run-log reducer reconstruct the same pool from canonical
records.** Node code never touches a `TypedValue`: `.to_object()` gives the plain Python
value. Initial inputs are wrapped at the pool boundary; a node output is wrapped and
validated while its canonical success record is prepared, before append and reducer apply.

## Two constructors: infer vs. check

```
build_value(v)               → wrap v in its NATURAL type (infer)
build_value_as(declared, v)  → wrap v as `declared`, RAISE on mismatch (check)
```

`build_value` infers the kind (`bool` checked before `int`; empty list → `list[Any]`).
`build_value_as` is the declared-type check used by both `VariablePool.set` and the
run log's `canonical_typed` success-record boundary. It raises `TypeCheckError` on a
mismatch; the only coercion is widening (`int` fills a `float` slot).
`TypeCheckError.path` is an immutable tuple relative to the declared type: string
segments name record fields and integer segments are actual list indices. A
mismatch at the checked root uses `()`; a nested container mismatch keeps its
current field/index path. The message and ordinary `ValueError.args` stay unchanged,
while input/output boundaries use the path to select the exact or nearest authored
contract origin before recording a failure.

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

## Authored contract origins

Compose compiles each authored root- or `defs:`-level flow input into a `Parameter`
whose `origin` identifies the selecting key. Its `ValueContract` records the type value
at the contract root and at each recursively authored nested record field. A physical
node `output:` declaration populates the same root/nested `ContractOrigins` on the
node's canonical `ValueContract`. Compose reads that retained cursor at the descriptor's
existing per-kind build position, so source compilation does not move ahead of
kind-owned validation.
For a packed declaration such as `count: "int = 3"`, `default_origin` is a distinct
semantic key that resolves to the same scalar range as the contract root.

When a boundary check reports a `TypeCheckError.path`, origin selection tries that exact
contract path, then its nearest authored ancestor and the contract root. An input failure
then falls back to the parameter key; an output failure uses the effective commit target's
contract and finally the executing node. A scalar typedef reference or scalar `list[...]`
spelling has no separately authored nested sites, so nested failures fall back to its root.
Full and compact root flows, both `defs:` forms, and inline children retain physical node
output sites; external-flow views rebase those origins through the same resolver mechanism
as other origins. Generated and Python-built contracts, including inferred inline-flow
signatures, remain source-free.

Node-output contract provenance is distinct from flow-output binding provenance. The former
describes the type declaration on `Node.output_contract`; the latter describes the value
expression on `FlowOutput.binding`. A compact root or flat single-node definition therefore
has authored contract origins even though its generated flow-output reexport remains
origin-free.

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
  `AnyValue` union. The run-record codec uses the same typed representation for the
  complete initial pool and for each committed node output.

## Why this shape

- **Durable by construction** — the discriminated union makes canonical record replay
  lossless.
- **One vocabulary** — `ValueKind.value` is author token, wire tag, and display name at once.
- **Fail at the boundary** — nodes stay pure; `build_value_as` catches a wrong type early.
