# Types — the typed value system

> The `typesys` package (`values.py` + `types.py` + `pool.py`) — the **leaf** of the
> layer ladder, below `nodes`. Everything above depends on it; it depends on nothing
> internal. For the authoring surface (`type:` / `typedefs:`), see
> [Flow syntax → Types](syntax.md#types).

## Why a flow is typed

A flow is a function: typed values in and a typed value out. Value-producing
nodes pass typed values to consumers; routing nodes govern control without
binding a value. Two forces make the typing non-negotiable:

- **Durability.** A run can be recovered in another process by replaying its
  authoritative record prefix. The initial record carries the whole pool and later
  success records carry typed outputs. JSON alone is lossy — `1` and `1.0` can collapse
  to the same representation — so every value carries its own type tag and decodes back
  to *exactly* the type it had.
- **Loud failure at the boundary.** A value-producing node returns a plain value;
  the engine checks it against the declared output type before appending its success
  record and reducing the write into the pool — so a wrong shape fails there, not
  three nodes later.

```
   types.py  → authoring surface: "list[Signal]" / typedefs: → a runtime Type
   values.py → the leaf: ValueKind, the TypedValue containers, the write-boundary check
   pool.py   → VariablePool: the run's typed value store + the ${...} resolver
```

## `ValueKind` — the closed vocabulary

Every value is one of a fixed set of kinds. The enum's `.value` is the *one* spelling —
author token, wire tag, and display name at once (it is `str`, never `string`):

```
scalars   None  str  int  float  bool  date  datetime  object  file
lists     list[Any]  list[str]  list[int]  list[float]  list[bool]  list[object]
```

`date`/`datetime` are strings in one fixed form, never auto-inferred:

| Kind | Form | Example |
|---|---|---|
| `date` | `YYYY-MM-DD` | `2026-05-21` |
| `datetime` | `YYYY-MM-DDTHH:MM[:SS[.fff]][±HH:MM]` | `2026-05-21T14:30:00+07:00` |

A space may replace `T`, `.fff` is 1–6 fraction digits, and the offset may be `Z`,
which means `+00:00`. The same strings are accepted on every supported Python version.
Other ISO 8601 forms, such as `20260521` or `2026-W21-4`, are rejected with an error
that shows the format. The value keeps the text as written.

In a flow file, yian's loader reads an unquoted timestamp as a date or datetime first
(see [YAML values](syntax.md#yaml-values)). A WAIT `until:` or a `case`
value gets its ISO text, so YAML forms such as `2026-05-21 14:30:00 -5` work there; a
typed default must be quoted (see [syntax](syntax.md#types)), and so must a timestamp
anywhere else.

`file` is a reserved placeholder (not yet authorable).

A `float` is finite. NaN, infinity, and `complex` numbers are not yian values, at any
depth. The first boundary that sees one rejects it and names where it sits:

- a run input: `run_flow` raises `ValueError` before the log exists;
- a binding's value (expression result, literal, or default): the node
  fails with `binding.evaluate`;
- a node output: the node fails with `contract.output`, naming the field path;
- an `llm_config` value: a load error, or `ValueError` before the run for a host config;
- the run-log codecs (`canonical_typed`, `canonical_seed`, `encode_pool`): `TypeCheckError`
  with a path. Nothing is recorded as `null` in its place;
- a model reply: in an AGENT's structured reply it is a schema deviation, retried
  under `retries:`; in a tool call's arguments the AGENT fails with `node.execute`
  before any call of that reply runs (`tool call 'lookup' arguments at x: non-finite
  number nan is not a yian value`). With session memory a reply holding one anywhere,
  provider metadata included, can't be stored: that model call is recorded `failed`
  and the reply is not kept, so a text or tool AGENT fails with `node.execute` and a
  structured AGENT retries, spending `retries:` even on a metadata NaN the model
  can't fix.

A temporary inside an expression may be infinite or complex; only a binding's final
value is checked.

An `int` has no size limit. The run log and the run's outputs record it exactly,
however many digits it has, and `resume_flow` reads it back unchanged. Rendering one
as text (embedded in a string or in prompt text) still follows Python's digit limit;
see
[Expressions → Arithmetic follows Python](expressions.md#arithmetic-follows-python).

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
mismatch; the only coercion is widening (`int` fills a `float` slot at any depth:
a record field, a list item, or a union member's field).
An `int` too large for a float fails at its path (`int is too large for float`).
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
| **union** | `kind=OBJECT`, `variants=(record, ...)`: two or more non-nullable records |
| **Optional[X]** | X's `Type` with `nullable=True` |

For a refined specialized list, `Type.element.nullable` permits `None` alongside
the element's ordinary values; a bare or non-nullable specialized list still
rejects it. The existing `LIST_ANY` and explicit `None`-element semantics remain
unchanged. A nullable specialized list retains its `LIST_*` discriminator and
carrier, so `AnyValue` and `VariablePool` round-trips preserve null items without
lowering it to `LIST_ANY`. Cross-flow compatibility applies the same structural
check recursively: a strict element source may fill a nullable element sink, but
not the reverse.

Authors write types as Python-typing strings; `parse_type` parses them via Python's own
`ast` (so the vocabulary *is* typing: `list[X]`, `Optional[X]`, `Literal[...]`, a bare
PascalCase registry name, `Union[A, B, ...]`). `read_typedefs` builds the per-flow
`TypeRegistry`, `resolve_type` resolves a parsed expr against it into a leaf `Type`.
`read_typedefs` resolves every typedef as it builds the registry, even one nothing uses,
so an unknown constructor or name, bad bounds, a recursive record, or an alias cycle fails
there. The error names the typedef and field (`typedef 'Signal' field 'tags': unknown type
constructor 'lst' in 'lst[str]'`), and its `path` is `(typedef, field)` so the loader can
point at that field.

`Union[...]` members are records. `resolve_type` flattens nested unions and aliases, drops
repeated members, and orders the members canonically (by tag, else by
`canonical_type_label`), so authored order is not identity. A union with one distinct
member is that member. The members must be told apart by the data alone:

- **Tagged.** A tag field is one every member requires as a non-nullable single-value
  `Literal`, with a different value in each (`union_tag_fields` derives it; nothing is
  stored). `build_value_as` picks the member by the tag (the first tag field by name), then
  runs that member's record check. A missing tag fails at `(tag,)` with
  `missing tag field 'action' (expected one of ['buy', 'sell'])`; an unknown one with
  `'hold' is not a tag of the union (expected one of ['buy', 'sell'])`.
- **Untagged.** The members' shapes can't overlap (`types_overlap`, conservative), so a
  value satisfies at most one, and that member's check runs. No match fails at `()` with
  `value matches no member of <label>: <member>: <reason>; ...`.

Otherwise resolution fails: `Union[Buy, Sell]: members 'Buy' and 'Sell' can match the same
value; add a tag field every member declares as a single-value Literal, ...`. Like every
typedef, a union typedef resolves in `read_typedefs`, so a bad union fails even when
nothing uses it. Compatibility is member by member: a source union
fits a sink union when each source member fits some sink member, a record fits a union
when it fits some member, a union fits `object`, and a union never fits a record (load
can't tell which member arrives). `union_field` describes a field read on a union (every
member has it, some do, or none do); `restrict_union` keeps some members. The manifest
emits a union as `"union": [...]` in canonical order.

A [typed binding](syntax.md#typed-bindings-name-type-value) compares its declared
type with the type inferred for its value through `compile.validation.type_relation`:

- **fits** — every inferred value is a declared value; no check is needed beyond the
  usual runtime one.
- **narrower** — some inferred values may miss the declared type (Optional into
  non-Optional, `str` into a `Literal`, an unbounded number into a bounded one, an
  `object` value into a record, a union into one of its members); the binding loads
  and the binder checks it at runtime.
- **disjoint** — no non-null inferred value fits; loading fails. Records count as
  closed, so an extra required field makes them disjoint, and sharing only `None`
  is not overlap.

Unions compare member-wise: each inferred member takes its best relation against the
declared members (a non-union side counts as one member), and the results combine to
fits when all fit, disjoint when all are disjoint, and narrower otherwise. Narrowing
views apply first, so a `case` branch's value has only the branch's members.

This relation agrees with `types_compatible`, the check for wiring, on what fits, with
two differences. It treats records as closed, so a record fits only when its fields are
a subset of the declared fields at every level. It lets `date` and `datetime` fit
`str`, since they are stored as strings. Where wiring rejects a source that is only
narrower, such as a tag-less `str` into a `Literal`, a typed binding accepts it and
checks the value at runtime.

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

## `VariablePool` — the run's typed value store

The runtime state primitive the engine owns. Narrow by design:

- **Each node id has at most one value slot.** A value-producing node commits one
  `TypedValue`; CASE routes without binding a value. "Multiple outputs" are fields
  of one `object`. The engine binds each logical target once; stored values are typed.
- **Three read namespaces**, all via `${...}`:

  | Reference | Resolves to |
  |-----------|-------------|
  | `${input.X}` | field `X` of the flow's input |
  | `${node.output[.path]}` | node `node`'s value, dotting into an object |
  | `${system.X}` | a host-injected ambient: `today` (`date`), `now` (`datetime`), or `run_id` (`str`); reserved, run-global. These are scalars, so `${system.today.year}` is a load error |

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
- **Fail at the boundary** — nodes cannot mutate graph state; `build_value_as`
  catches a wrong type early.
