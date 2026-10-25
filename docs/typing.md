# Types — the typed value system

> The `typesys` package (`values.py` + `types.py` + `pool.py`) — the **leaf** of the
> layer ladder, below `nodes`. Everything above depends on it; it depends on nothing
> internal. For the authoring surface (`typedefs:`, `output:` types, and tagged slots),
> see [Flow syntax → Types](syntax.md#types).

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
   types.py  → authoring surface: "list(Signal)" / typedefs: → a runtime Type
   values.py → the leaf: ValueKind, the TypedValue containers, the write-boundary check
   pool.py   → VariablePool: the run's typed value store + the ${...} resolver
```

## `ValueKind` — the closed vocabulary

Every value is one of a fixed set of kinds. The enum's `.value` is the kind's storage
tag, the name the run log and the wire use (it is `str`, never `string`). It is not
authored syntax: authors write types with parentheses, such as `list(str)`.

```
scalars   None  str  int  float  bool  date  datetime  object  file  inbox
lists     list[Any]  list[str]  list[int]  list[float]  list[bool]  list[object]
type-only Any
```

`ANY` is type-only: no value is stored as `Any`. An `Any` value is any JSON value except
null, and it keeps its natural kind (`str`, `int`, `object`, `list[Any]`, ...), so the run
log records `"hello"` as a `str`. Type labels spell the list kinds with parentheses:
`list[str]` is the type `list(str)`, and `list[Any]` holds a `list(Any)` or a
`list(Optional(Any))`.

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
(see [YAML values](syntax.md#yaml-values)). A WAIT `until:` (TS yian: no WAIT) or a `case`
value gets its ISO text, so YAML forms such as `2026-05-21 14:30:00 -5` work there; a
typed default must be quoted (see [syntax](syntax.md#types)), and so must a timestamp
anywhere else.

Changed (TS yian): an unquoted date or timestamp is text everywhere (see the note in
[YAML values](syntax.md#yaml-values)), so this paragraph doesn't apply. A default fills a
`str` slot, and a `date` or `datetime` slot checks its text: `!date 2026-01-01` loads, and
`!date 2026-10-14T10:00:00Z` fails with `date must look like 2026-05-21`. A YAML timestamp
form such as `2026-05-21 14:30:00 -5` is a load error.

`file` is a file reference, held as a `FileRef` with `uri` (text), `mime`, and `name`
(text or null). Where the loader knows a value is a `file`, a field read on it is typed
at load: `${f.uri}` is `str`, `${f.mime}` and `${f.name}` are `Optional(str)`, and any
other field is a load error. A file from an untyped output isn't checked (see
[References](syntax.md#references-naming-a-value)). Storage and transfer are not built
yet. Its JSON form is the mapping
`{"uri": ..., "mime": ..., "name": ...}`. A `file` slot reads that mapping, with exactly
those keys, as a `FileRef`. An `object` slot reads a nested `FileRef` as its mapping, but
a `FileRef` at the top of an `object` slot fails, and `Any` and `list[Any]` reject files.
A typedef that names `file` (`FileT: file`) passes `isinstance` for the JSON form too.

Changed (TS yian): `inbox` is a handle on a top-level inbox, the value of an `Inbox(T)`
input, held as an `InboxRef` with one field, `inbox` (the inbox's name, non-empty text). Its
JSON form is the mapping `{"inbox": ...}`, and its typed wire is
`{"kind": "inbox", "value": {"inbox": ...}}`. An `Inbox(T)` slot reads that mapping, with
exactly that key, as an `InboxRef`; text, a mapping with other keys, `null` and a list fail
with `TypeCheckError`. Every other type refuses a handle, except that an `object` slot reads a
handle nested in its value as the `{"inbox": ...}` mapping, as it reads a nested `FileRef`; a
handle at its top fails. Load treats `Inbox(T)` as disjoint from every other type: `Inbox(T)`
relates only to another `Inbox(U)`, as `T` to `U`. `inbox` is never inferred from a record;
only an `InboxRef` infers it.

A `float` is finite. NaN, infinity, and `complex` numbers are not yian values, at any
depth. The first boundary that sees one rejects it and names where it sits:

- a run input: `run_flow` raises `ValueError` before the log exists;
- a binding's value (expression result, literal, or default): the node
  fails with `binding.evaluate`;
- a node output: the node fails with `contract.output`, naming the field path;
- an `llm_config` value: a load error for a literal; `ValueError` before the run for a
  host config or a `${...}` value resolved at run start; `binding.evaluate` for a
  `${...}` value resolved when its agent runs;
- the run-log codecs (`canonical_typed`, `canonical_seed`, `encode_pool`): `TypeCheckError`
  with a path. Nothing is recorded as `null` in its place;
- a model reply: in an AGENT's structured reply it is a schema deviation, retried
  under `retries:`; in a tool call's arguments the AGENT fails with `node.execute`
  before any call of that reply runs (`tool call 'lookup' arguments at x: non-finite
  number nan is not a yian value`). With session memory a reply holding one anywhere,
  provider metadata included, can't be stored: that model call is recorded `failed`
  and the reply is not kept, so a text or tool AGENT fails with `node.execute` and a
  structured AGENT retries, spending `retries:` even on a metadata NaN the model
  can't fix. (Python yian; TS yian has no session memory.)

Changed (TS yian): infinity and minus infinity are float values; an open range end admits its
infinity (`float(0.0..)` takes infinity), and a range closed at that end refuses it. NaN is
refused by the build boundary and host intake with `NaN is not a yian value`. Text or a record
key holding a lone surrogate is not a yian value (`text holding a lone surrogate is not a yian
value`); a value written in a flow file fails the load (see [Nesting limit](#nesting-limit)).
`-0` becomes `0` when a float becomes a value.

A temporary inside an expression may be infinite or complex; only a binding's final
value is checked.

## Integer size limit

An `int` has at most 4300 decimal digits, Python's default int/str limit; a host can
lower the limit with `YIAN_MAX_INT_DIGITS` (see [Defaults](defaults.md)). The sign and
leading zeros don't count, a hex or octal literal counts the digits of its decimal
value, and exactly 4300 digits is allowed. A longer int is not a yian value, like NaN. The message gives the digit count, never the digits:

```
input 'n': integer with 5000 digits is larger than yian allows (4300 digits)
```

The first boundary that sees one rejects it and names where it sits:

- an int literal in the flow file (a value, a default, a bare `call(...)` argument, a
  number in an expression, or an `int(lo..hi)` range end): a load error at the literal;
- a run input: `run_flow` raises `ValueError` before the log exists, and `yian run`
  reports it for `-i`, `--inputs`, and the interactive prompt (which asks again);
- a binding's value or default, a value entry, a MAP `over` list, a LOOP seed, or a
  flow output: the node fails with `binding.evaluate`;
- a node output: the node fails with `contract.output` at the field path;
- a model reply: a structured reply is retried under `retries:`, and tool-call
  arguments fail the AGENT with `node.execute` before any call of that reply runs;
- a host tool's result, which the model reads as text: the AGENT fails naming the tool
  (`tool 'lookup' result: ...`);
- `RunResult.backward(feedback)`: such feedback raises `ValueError` before any
  backward attempt starts. An AGENT's backward response holding one is rejected at its
  path (`backward response at self_feedback[1]: ...`).

A temporary inside an expression may be larger: `${10 ** n > 0}` is a bool. Writing
one as text fails like any other expression error, with the same message: embedded
text (`"v=${10 ** n}"`), a prompt, `%`, `join`, or `render_as_json`; see
[Expressions → Arithmetic follows Python](expressions.md#arithmetic-follows-python).

yian never changes Python's own int/str digit limit. The rule holds with that limit at
its default (4300) or off; a host that lowers it below `YIAN_MAX_INT_DIGITS` makes
valid ints fail to convert.

Changed: ints of any size used to be values. An int over 4300 digits now fails at
the first boundary. A run log written then that holds one fails to resume or replay
with `run-record JSON: integer with N digits is larger than yian allows (4300
digits); a log written while yian accepted such ints can't be read`, and the log is
left unchanged. A memory store written then that holds one may fail to open.

Changed (TS yian): an `int` has no digit limit; an int of any size is a value at every
boundary above.

## Nesting limit

A yian value nests at most 100 levels by default, counting itself;
`YIAN_MAX_VALUE_NESTING` can lower it (see [Defaults and limits](defaults.md)). Each
dict, list, tuple, pydantic model (such as a `FileRef`), or dataclass instance is one
level; a `TypedValue` wrapper and a scalar add none. A record or list
built around a value is a new value one level deeper, so wrapping a 100-level value
fails. `reject_deep_nesting(value, where)` checks without recursion and raises
`TypeCheckError` whose `path` locates the first container past the limit:

```
<where> at <path>: value nests deeper than 100 levels, the limit for a yian value (each record and list is a level)
```

Changed (TS yian): the levels are arrays, records and `FileRef`s; a `TypedValue` wrapper and a
scalar add none.

An expression has the same bound, checked when it is parsed; see
[Expressions → Expression nesting limit](expressions.md#expression-nesting-limit).
A flow file's YAML has its own, larger bound of 200 levels (`YIAN_MAX_YAML_NESTING`),
and expands to at most 100,000 nodes, each alias counting as a copy of what it names
(`YIAN_MAX_YAML_NODES`); see [Syntax → YAML values](syntax.md#yaml-values).

A value written out in a flow file (a node input, MAP `over:`, WAIT `until:`, a value
entry, an input or AGENT variable default, a flow `output:`, an `env:` value) is
checked at load with the counting the run-time check would apply to it, so one nested
deeper is a located load error, even when YAML aliases build its depth while the text
stays shallow. A record flow output's fields count the record's level. `${...}` text
counts as a scalar there, so the boundaries below still check computed values and
values in a flow built in Python.

Changed (TS yian): that load check also refuses text or a record key holding a lone surrogate
(only a double-quoted `\uD800` escape can write one): `node 'use' input 'topic': text holding a
lone surrogate is not a yian value`, at the value. It refuses an AGENT's `prompt:` and
`system_prompt:` text holding one too, at the field (`node 'a' field 'prompt': ...`), and a
HUMAN_INPUT's `prompt:`, `adaptive_questions.prompt:` and literal `questions:` text, the last at
the text (`node 'ask' field 'questions' at [0].question: ...`), a CASE's `on:` and `when:`
text, at the field (`node 'g' field 'cases[1].when': ...`), a LOOP's `while:` and `until:`
text, at the field (`node 'lp' field 'while': ...`), a node's `asserts:` text, at the assert,
among the node's other assert errors (`node 'x' field 'asserts[0]': ...`), and a flow's
`asserts:` text, at the assert (`flow field 'asserts[0]': ...`). Python yian loads such a value.
With HUMAN_INPUT removed, an `ask`'s `prompt:` and `questions:` text are node inputs, checked as
one.

A type nests at most 100 levels by default (`YIAN_MAX_TYPE_NESTING`). In type text
every constructor is a level, `Optional(` and `Union(` included, so `list(Optional(int))`
nests 2. A record written as a mapping, or a list of single-key maps, in a node's
`output:` counts differently: each record is a level, and a field's type adds the
records and lists it resolves to, so a typedef name counts as deep as its definition
(an `Optional(...)` or a union adds only its deepest member's). A deeper record is a
load error at the record or field that crosses the limit, aliases included:

```
type at r.k.k nests deeper than 100 levels, the limit for a type (each record and list is a level); its type 'list(list(int))' adds 2
```

The `; its type ...` part appears when a field's type pushes the record over.
An AGENT's `output:` nests at most 16 levels, counted the same way
(`YIAN_MAX_AGENT_OUTPUT_NESTING` can only lower it), since a model's reply schema is built
from it; see [Syntax → AGENT](syntax.md#agent-the-llm-backed-agent-node).

A typedef counts the same way, and a chain of typedefs, each naming the next, holds at
most 100 of them. A typedef past either bound fails the load at the typedef, and at the
field that crosses the limit for a record:

```
bad typedefs: typedef 'T100' field 'a': type nests deeper than 100 levels, the limit for a type (each record and list is a level); its type 'T99' adds 100
bad typedefs: typedef 'A100': it ends a chain of 101 typedefs, each naming the next, more than 100, the limit for a type; write its type with fewer typedefs in between
```

A type anywhere else (an `output:` type, a tag, a flow `input:`) also stays within 100
levels once its typedef names are expanded, so `list(T)` over a 100-level `T` fails at
load: `type nests deeper than 100 levels once its typedef names are expanded, the limit
for a type (each record and list is a level)`.

The first boundary that sees a deeper value rejects it:

- a run input may nest 100 levels; a deeper one makes `run_flow` raise `ValueError`
  (a `TypeCheckError`) naming `input 'x'` and the path, before the log exists;
- a binding's value (expression result, literal, or default) or a MAP `over` list:
  the node fails with `binding.evaluate` at the binding or default. `${[x]}` around a
  100-level `x` fails, and MAP items may nest 99 levels. An AGENT variable default
  fails `node.execute`. Only a binding's final value is checked, not temporaries
  inside an expression. A literal or default written out in the flow file never gets
  here, since the load check rejects it first;
- a node output, checked whole when the node succeeds: the node fails with
  `contract.output` at the field path. A flow output is its END's output, so
  `output: ${n.output}` carries a 100-level value, but a record flow output
  `{t: ${n.output}}` adds a level and fails at `__end__`. A MAP's result list adds a
  level too, so a MAP whose children return 100-level values fails at its fan-in END
  (`<map>/__end__`); a LOOP body's record output likewise counts its level;
- `RunResult.backward(feedback)`: deeper feedback raises `ValueError` before any
  backward attempt starts;
- a model reply, checked before the non-finite check: an AGENT's deeper structured
  value self-corrects under `retries:` like a NaN (`structured output at t.a...`), and
  deeper tool-call arguments fail the AGENT with `node.execute` before any call of that
  reply runs (`tool call 'lookup' arguments at x.a...`). With session memory that
  model call is recorded `failed` and the reply is not kept (a JSON-text structured
  reply is kept, since it is text). (Python yian; TS yian has no session memory.)

START's input record holds the input values, so it alone may sit one level over the
limit. Replaying or resuming a recorded log does not re-check the limit, so logs
recorded before it still replay. The run-log codecs (`canonical_typed`,
`canonical_seed`, `encode_pool`) do not apply it; they see only values these
boundaries bounded.

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

`build_value` infers the kind (`bool` checked before `int`; an empty list is the untyped
list, labelled `list(Optional(Any))`).
`build_value_as` is the declared-type check used by both `VariablePool.set` and the
run log's `canonical_typed` success-record boundary. It raises `TypeCheckError` on a
mismatch. Three conversions apply at any depth (a record field, a list item, or a union
member's field): an `int` filling a `float` slot widens; a file's JSON form filling a
`file` slot becomes a `FileRef`; and a `FileRef` nested in an `object` value or a
`list(object)` item becomes its mapping, copying only the containers on its path. An
`Any` value is stored as given: an int stays an int and `True` stays a bool.

Changed (TS yian): an `int` filling a `float` slot stays an int at any depth. A `float`
slot given an int stores kind `int`, a `list(float)` keeps its int items, and a record's
`float` field keeps the int; slots only check. Likewise `buildValue` infers a list of ints
and floats as `list(float)` with its ints kept (`[1, 2.5]`, where Python stores `[1.0, 2.5]`).

Changed (TS yian): `buildValue` walks its whole value as a JSON value (files kept) and refuses
a part that isn't one; Python's `build_value` doesn't walk, and its `canonical_typed` runs
`reject_non_json` first.

`normalize_files(typ, value)` applies only the file conversions, with no check and no
widening. Typed bindings and `T(...)` constructors deliver their value through it, and
the runtime uses it to restore files in values it decodes from the run log.

`object`, `Any`, `list(object)` items, and `list(Optional(Any))` items must be JSON
values: `None`, a `bool`, `int`, `float`, or `str`, a `dict` with only `str` keys and
JSON values, or a list or tuple of JSON values. The check walks the value without recursion and fails at
the bad part's path: a non-text key with `object keys must be text, got 1` at the
mapping, anything else (a date object, a set, a pydantic model, a dataclass, a
`FileRef` in an `Any` value) with `<value> is not a JSON value`, and a container inside
itself with `value contains itself`. An `object` value or `list(object)` item reads a
nested `FileRef` as its mapping before the walk. Depth and non-finite numbers are left
to the boundary checks above, so a recorded value replays. A value or key shown in a
type-check message is cut to `YIAN_MAX_SHOWN_VALUE_CHARS` (200) characters: a longer
rendering keeps its first 199 and ends with `…` (see [Defaults and limits](defaults.md)).
A node output with no declared type (and each untyped field of a partial record output)
goes through the same walk at `canonical_typed`, with `FileRef` leaves allowed at any
depth; its message names the path (`output at d.x[1]: datetime.date(2026, 1, 1) is not a
JSON value`). An already-built `TypedValue` (a replayed record) is not walked. Nothing
types those files, so a reader sees each one as its mapping.
An `int` too large for a float fails at its path (`int is too large for float`).
Changed (TS yian): no int converts to a float, so this error doesn't exist.
`TypeCheckError.path` is an immutable tuple relative to the declared type: string
segments name record fields and integer segments are actual list indices. A
mismatch at the checked root uses `()`; a nested container mismatch keeps its
current field/index path. The message and ordinary `ValueError.args` stay unchanged,
while input/output boundaries use the path to select the exact or nearest authored
contract origin before recording a failure. The path stays exact however long its keys
are; only its rendering in a message is cut: each key to `YIAN_MAX_SHOWN_VALUE_CHARS`,
and a rendered path longer than four times that keeps its start and end around `…`.
Before any of these checks, a node output is read once, through Python iteration, into
plain containers, and every output check and the stored value use that copy: a `dict`,
`list`, or `tuple` subclass counts as what its `items()` or iteration yields (a list
whose iteration yields nothing is stored as `[]`).

Changed (TS yian): JSON values are `null`, a boolean, an int (`bigint`), a float (`number`),
text, and arrays and records of those, so the dict and tuple wording above reads as records and
arrays; anything else (a `Date`, a `Map` or `Set`, a class instance) is `<value> is not a JSON
value` (`<Date object> is not a JSON value`). A value shown in a type-check message is in the text form (`"bad" does not match
declared type int`, where Python writes `'bad'`). A name quoted in the prose keeps Python's
repr (`missing tag field 'action'`), but a list of names and a quoted key in a value path are in
the text form too: `record missing required fields: ["confidence","value"]`, `unknown inputs
"a", "b"` and `output at a["my key"]`, where Python writes `['confidence', 'value']`,
`'a', 'b'` and `a['my key']`.

Changed (TS yian): host intake replaces the plain-container read. A host value (such as a tool's
result or a CODE body's return value) enters through `fromHost(value, where)`, which returns a
fresh yian value or throws `TypeCheckError` with `path` and the message
`<where>[ at <path>]: <reason>`:

| Host value | yian value |
|---|---|
| `null`, a `boolean`, a well-formed `string` | itself |
| `bigint` | int |
| `number`, an integer within ±(2^53−1) | int (`-0` is `0`) |
| any other `number` (`±Infinity`, `1e20`, `2.5`, `9007199254740992`) | float |
| `NaN` | error: `NaN is not a yian value` |
| a string or record key holding a lone surrogate | error: `text holding a lone surrogate is not a yian value` |
| a `FileRef` its constructor built | itself |
| plain array (a hole is `undefined`), plain record (prototype `Object.prototype` or `null`) | a fresh array or record, its items taken in the same way |
| `undefined`, at any depth | error: `undefined is not a yian value; use null` |
| a container inside itself | error: `value contains itself` |
| an accessor property (getter or setter) on a record or array index | error: `accessor property is not a yian value` |
| a `Proxy` | error: `Proxy is not a yian value` |
| anything else | error: `<Name> is not a yian value` (`Date is not a yian value`) |

Like `JSON.stringify`, intake skips symbol keys, non-enumerable keys and extra array
properties. A `TypedValue` handed in is refused. Intake reads only own data properties, so no
host code runs while it reads a value. `fromHost` refuses a top-level `undefined` too: the tool
caller reads it as `null` first, and a CODE body returning it fails. Depth is checked
separately, by `rejectDeepNesting`.

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
| **Optional(X)** | X's `Type` with `nullable=True` |
| **Any** | `kind=ANY` only (type-only; the value keeps its natural kind) |
| **Inbox(T)** (TS yian) | `kind=INBOX`, `element=<T>`; never nullable |

For a refined specialized list, `Type.element.nullable` permits `None` alongside the
element's ordinary values; a bare or non-nullable specialized list still rejects it. An
authored `list(Any)` is `LIST_ANY` with an `Any` element, and `list(Optional(Any))` is
`LIST_ANY` with an `Optional(Any)` element. A `LIST_ANY` with no element is a list whose
items weren't proven: a tool's untyped array or tuple, or a mixed or empty literal. It
has the same values and label, `list(Optional(Any))`, and compatibility reads its items
as `Optional(Any)`, but MAP items over it are checked at runtime and an inline body's
parameter still needs a proven element. A nullable specialized list retains its `LIST_*`
discriminator and carrier, so `AnyValue` and `VariablePool` round-trips preserve null
items without lowering it to `LIST_ANY`. Cross-flow compatibility applies the same
structural check recursively: a strict element source may fill a nullable element sink,
but not the reverse.

Authors write types in yian's own grammar: Python typing's vocabulary, spelled with
parentheses (`list(X)`, `Optional(X)`, `Literal(...)`, `dict(K, V)`, a bare PascalCase
registry name, `Union(A, B, ...)`) and inclusive numeric ranges (`int(1..12)`,
`float(0.0..1.0)`, `int(0..)`, `int(..100)`; either end may be open). `parse_type` is a
hand-written tokenizer and recursive-descent parser that reads `0..1` as a range, not as
`0.` and `.1`. It builds the same `TypeExpr` AST as before, so resolution is unchanged,
and `format_type_expr` prints the parenthesized spelling. The old bracket spelling
(`Optional[str]`, `int[1, 12]`) is a load error that shows the new spelling; a private
legacy parser reads it only to print that message. Wire values keep their own names: a
`ValueKind` such as `list[str]` is a storage tag, not authored syntax.

A type definition (a typedef, or a node's `output:`) is written as a type. A slot's type
is a YAML tag on its value, read by `compose/helpers/tags.py` with the same parser (see
[Typed slots](syntax.md#typed-slots-yaml-tags)). A tag can't hold `,`, `[`, `]`, `{` or
`}` and ends at the first space, so a multi-argument type is named in `typedefs:` and
tagged by its name. `read_typedefs` builds the per-flow
`TypeRegistry`, `resolve_type` resolves a parsed expr against it into a leaf `Type`.
`read_typedefs` resolves every typedef as it builds the registry, even one nothing uses,
so an unknown constructor or name, bad bounds, a recursive record, or an alias cycle fails
there. The error names the typedef and field (`typedef 'Signal' field 'tags': unknown type
constructor 'lst' in 'lst(str)'`), and its `path` is `(typedef, field)` so the loader can
point at that field.

`Union(...)` members are records. `resolve_type` flattens nested unions and aliases, drops
repeated members, and orders the members canonically (by tag, else by the canonical
label in the older bracket spelling, `Optional[int]` where `canonical_type_label` prints
`Optional(int)`), so authored order is not identity. A union with one distinct
member is that member. The members must be told apart by the data alone:

- **Tagged.** A tag field is one every member requires as a non-nullable single-value
  `Literal`, with a different value in each (`union_tag_fields` derives it; nothing is
  stored). `build_value_as` picks the member by the tag (the first tag field by name), then
  runs that member's record check. A missing tag fails at `(tag,)` with
  `missing tag field 'action' (expected one of ['buy', 'sell'])`; an unknown one with
  `'hold' is not a tag of the union (expected one of ['buy', 'sell'])`. Changed (TS yian):
  `missing tag field 'action' (expected one of ["buy","sell"])` and `"hold" is not a tag of
  the union (expected one of ["buy","sell"])`.
- **Untagged.** The members' shapes can't overlap (`types_overlap`, conservative), so a
  value satisfies at most one, and that member's check runs. No match fails at `()` with
  `value matches no member of <label>: <member>: <reason>; ...`.

Otherwise resolution fails: `Union(Buy, Sell): members 'Buy' and 'Sell' can match the same
value; add a tag field every member declares as a single-value Literal, ...`. Like every
typedef, a union typedef resolves in `read_typedefs`, so a bad union fails even when
nothing uses it. Compatibility is member by member: a source union
fits a sink union when each source member fits some sink member, a record fits a union
when it fits some member, a union fits `object`, and a union never fits a record (load
can't tell which member arrives). `union_field` describes a field read on a union (every
member has it, some do, or none do); `restrict_union` keeps some members. The manifest
emits a union as `"union": [...]` in canonical order.

A [tagged slot](syntax.md#typed-slots-yaml-tags) compares its tag's type with the type
inferred for its value through `compile.validation.type_relation`. A tag checks and
narrows. It never fills in a field, and it converts only how files are held
(`normalize_files`): a `file` position reads a file's JSON form as a `FileRef`, and an
`object` position reads a nested `FileRef` as its mapping.

- **fits** — every inferred value is a declared value; no check is needed beyond the
  usual runtime one.
- **narrower** — some inferred values may miss the declared type (Optional into
  non-Optional, `str` into a `Literal`, an unbounded or wider-ranged number into a
  bounded one, an `object` value into a record, a union into one of its members, or an
  `Any` value into another type); the binding loads and the binder checks it at
  runtime with `build_value_as`, failing with `binding.type`.
- **disjoint** — no non-null inferred value fits; loading fails. Records count as
  closed, so an extra required field makes them disjoint, and sharing only `None`
  is not overlap. A type whose every non-null value holds a `file` is disjoint from
  `Any`, and so is `None`. Load treats `file` as disjoint from `object`, `Any`, and a
  record shaped like a file's JSON form, so a typed `file` slot still needs a
  `file`-typed source. The JSON form is accepted at runtime where load can't tell: an
  untyped producer, a seed, or a value decoded from the run log.

Unions compare member-wise: each inferred member takes its best relation against the
declared members (a non-union side counts as one member), and the results combine to
fits when all fit, disjoint when all are disjoint, and narrower otherwise. Narrowing
views apply first, so a `case` branch's value has only the branch's members.

This relation agrees with `types_compatible`, the check for wiring, on what fits, with
two differences. It treats records as closed, so a record fits only when its fields are
a subset of the declared fields at every level. It lets `date` and `datetime` fit
`str`, since they are stored as strings. Where wiring rejects a source that is only
narrower, such as a plain `str` into a `Literal` or an `Any` value into an `int`, a
tagged slot accepts it and checks the value at runtime. Downstream readers see the tag's
type.

In wiring, every non-null type without a `file` in it fits `Any`, and a nullable one
fits `Optional(Any)`. An `Any` source fits only `Any` and `Optional(Any)`, since load
can't tell which JSON value it holds; the CALL and LOOP errors say to tag the slot or
narrow the value with `isinstance` in a `case`. `types_overlap` treats `Any` as
overlapping every type except `None`. Since a `file` slot reads a file's JSON form,
`file` overlaps plain `object`, `Any`, and every record that admits the mapping
`{uri, mime, name}`, so an untagged union whose members overlap that way is a load error
that asks for a tag field.

A value with `${...}` nested in a record or list is inferred structurally, field by
field, so each field keeps its type. A record that omits a required single-value
`Literal` field is not filled in; the check fails and names the field.

## The built-in Message type

TS yian. `Message` is a built-in record type that every flow can name wherever a type goes: a
tag (`!list(Message)`), a flow, CODE or END `output:`, and a typedef field, alias or union member.
An AGENT list prompt splices it with all its parts (see
[A list prompt](syntax.md#a-list-prompt)). Its fields:

| Field | Type | Meaning |
|---|---|---|
| `role` | `Literal(user, assistant, tool)` | Who sent it. There is no `system` role: the system prompt is the agent's own. |
| `parts` | `list(<part>)` | The content, in order. |
| `author` | `Optional(str)` | Who wrote it, when that isn't the flow's user or the model; non-empty when set. Model routes send only a user message's author, in an `<authored_message>` wrapper. |
| `protocol` | `Optional(str)` | Opaque replay data a model reply carries: the protocol that produced it. Keep it; don't write it. |
| `provider_payload` | `Optional(object)` | Opaque provider data at the message level. Keep it; don't write it. |

Each part is a record told apart by its `type`, and each may carry an opaque
`provider_payload: Optional(object)`. Reasoning and payload parts keep their replay record
(protocol, provider, model) inside it.

| `type` | Fields |
|---|---|
| `text` | `text: str` |
| `reasoning` | `text: str` |
| `refusal` | `text: str` |
| `tool_call` | `id: str`, `name: str`, `arguments: Optional(object)`, `raw_arguments: Optional(str)` |
| `tool_result` | `call_id: str`, `name: str`, `content: str`, `is_error: Optional(bool)` |
| `media` | `mime: str`, `file: file`, `name: Optional(str)` |

- **Media** is a `file` in the run's `media` folder. Inline or URL media has no `Message` form,
  and no model route sends media yet, so a spliced media part fails the node.
- **The name** is reserved: a typedef named `Message` is a load error. The part names aren't
  reserved. There is no `Message(...)` constructor and no `isinstance(x, Message)` yet. An AGENT
  can't output a `Message`, since a model can't write a `file`.
- **Storage:** a `Message` is stored as `object`, and a list of them as `list[object]`. Its files
  are restored by the commit target's contract (see
  [`VariablePool`](#variablepool-the-runs-typed-value-store)).
- **Labels:** the canonical label of a `Message` type is the expanded record.
- **Nesting:** a `Message` adds three levels (the message, its parts list and a part) above a tool
  call's `arguments`, and counts three levels toward a typedef that names it.
- **An AGENT's second value:** `${x.new_messages}` is typed `list(Message)`, so a typed binding,
  a field read or a flow output reading it is checked against that type (see
  [An AGENT's new messages](syntax.md#an-agents-new-messages)).

## Authored contract origins

Compose compiles each authored root- or `defs:`-level flow input into a `Parameter`
whose `origin` identifies the selecting key. Its `ValueContract` records the type value
at the contract root and at each recursively authored nested record field. A physical
node `output:` declaration populates the same root/nested `ContractOrigins` on the
node's canonical `ValueContract`. Compose reads that retained cursor at the descriptor's
existing per-kind build position, so source compilation does not move ahead of
kind-owned validation.
For a tagged declaration with a default, such as `count: !int 3`, `default_origin` is a
distinct semantic key that resolves to the same scalar range as the contract root.

When a boundary check reports a `TypeCheckError.path`, origin selection tries that exact
contract path, then its nearest authored ancestor and the contract root. An input failure
then falls back to the parameter key; an output failure uses the effective commit target's
contract and finally the executing node. A scalar typedef reference or scalar `list(...)`
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

  Changed (TS yian): `now` is the run's clock in Python's `isoformat()` spelling with
  `+00:00` and `today` its date, from one reading of the clock, so they always agree; the
  real clock has millisecond resolution, so a fraction ends in `000`.

  An unrecognized head resolves to `None` (the missing-ref-is-falsy contract — see
  [Expressions](expressions.md)). Because values are typed *objects*,
  `${x.output.ratio}` traverses **into** an object.
- **Lossless serialization.** `dumps()`/`loads()` round-trip the whole pool through the
  `AnyValue` union. The run-record codec uses the same typed representation for the
  complete initial pool and for each committed node output.

  Changed (TS yian): `store` and `system` are `Map`s; `toWire()`/`fromWire()` give and
  strictly read the wire object `{store, system, start_id}` (each value as `{kind, value}`),
  and `dumps()`/`loads()` write and read it with yian's JSON codec.
- **Files keep their type in typed containers.** The run log stores a file nested in a
  container as its JSON form. The runtime reads each committed value back through its
  commit target's contract, so every position the contract types as `file` holds a
  `FileRef`: a `list(file)` item, a record's `file` field, a union member's field, or a
  LOOP-carried `file`. The run result reads through the flow's inferred output type the
  same way. Live runs, replay and resume, and learnable trace reads agree; the log bytes
  don't change. A `${x.f}` dotted step reads a dict key or a file's `uri`, `mime`, or
  `name`. An untyped output or seed keeps its nested files as mappings. A MAP gets a
  list type only when its body's output has one exact type: a typed whole value, or a
  record whose every field is typed. A MAP whose body returns a record with an untyped
  field has no inferable list type, so readers of `${m.output}` in the parent, and a run
  result read from it, see its nested files as mappings too.

## Why this shape

- **Durable by construction** — the discriminated union makes canonical record replay
  lossless.
- **One storage vocabulary** — `ValueKind.value` names a kind in the run log and on the
  wire alike.
- **Fail at the boundary** — nodes cannot mutate graph state; `build_value_as`
  catches a wrong type early.
