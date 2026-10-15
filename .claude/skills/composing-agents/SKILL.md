---
name: composing-agents
description: Use when authoring a Yian flow — writing or editing a flow YAML, building an agent workflow, wiring nodes (agent / code / tool / case / human_input / wait / call / map), or answering "how do I write a flow that …". Trigger on "create an agent", "write a flow", "compose a workflow", "add a node to my flow", "how do I branch / loop / call another flow", or any `yian run` flow-authoring request.
---

# Composing an agent flow

In Yian, an **agent** is a composable typed transformation represented
by a node, a flow, or a child flow. A **flow is a function**: a typed `input:`, a
graph of `nodes:`, and a typed `output:`. The concrete **AGENT node**
(`kind: agent`) is the LLM-backed leaf. `${...}` references infer data edges;
`depends_on:` and `runs_after:` add ordering-only edges. The human owns the
structure; the LLM only fills the AGENT node leaves.

Conceptual vocabulary: [`docs/agent.md`](../../../docs/agent.md). Full authoring
reference: [`docs/syntax.md`](../../../docs/syntax.md). Working flows:
[`examples/`](../../../examples). This skill is the authoring workflow + the
parts that bite.

**Companions:** [`reference.md`](reference.md) — operators, the three expression
contexts, type forms, recipes, and gotchas. [`templates/`](templates/) — minimal
loadable starter flows for each shape (single AGENT node, AGENT node → CODE node pipeline,
branch+join, tool use, human gate, named and inline `call`/`map`/`loop`
composition); copy one and edit.

## The workflow — author a flow in order

1. **State the function.** Write the one-line job, then its **signature**: what
   typed values come in (`input:`) and what comes out (`output:`). If you can't
   name the input and output types, the design isn't done.
2. **Name reusable types** in `typedefs:` (enums, records, aliases).
3. **Decompose into leaf nodes.** Each node is one step: an LLM call (`agent`),
   deterministic code (`code`), a registered tool (`tool`), a branch (`case`), a
   blocking human gate (`human_input`), a blocking UTC deadline (`wait`), a child-flow call
   (`call`/`map`), or an iterated body (`loop`). A reusable child lives in
   `defs:`/`uses:`; a one-off child may put `nodes:` + `output:` directly on
   CALL/MAP/LOOP. Prefer many small typed nodes over one mega-prompt.
4. **Wire by reference, not by edges.** A node consumes another by putting
   `${producer.output}` in its `input:` block. That reference *is* the edge.
5. **Bind the flow `output:`** to the terminal node(s).
6. **Add `asserts:`** for invariants that must hold (boundary or post checks).
7. **Validate, then run** (see "Validate & run").

## The flow skeleton

```yaml
id: my-flow                  # stable identifier (no spaces)
name: my_flow                # display name
description: One line.       # optional

typedefs:                    # optional — named/composed types
  Verdict: Literal(go, no_go, wait)

input:                       # the parameters — each typed with a YAML tag
  question: !str             # required
  audience: !Optional(str)   # nullable; omitted -> null
  lookback: !int 30          # default-fill when omitted

nodes:                       # the body — a map keyed by node id (no per-node id:, no wrappers)
  step_a:
    kind: agent
    input:
      question: ${input.question}
    output: str
    prompt: "Answer: ${question}"

output: ${step_a.output}     # one value, OR a record {name: ${x}, ...}; may be typed: !str ${step_a.output}

asserts:                     # optional — boolean invariants
  - step_a.output != ""
```

There is **no `edges:` block**, no `__start__`/`__end__`, no per-node `id:`, and
no body wrappers — a node body is flat. Flows, nodes, and `defs:` entries all write
`input:`/`output:`; the old plural `inputs:`/`outputs:` is a load error.

**The `output:` shape sets the return.** `output: ${x}` returns the value of `x` as
is. `output: {name: ${x}, ...}` always returns a record, even with one field
(`output: {order: ${x}}` returns `{order: x}`, read by a caller as
`${child.output.order}`) or none (`output: {}` returns `{}`). No `output:` (or
`output: null`) returns `None`. Either form may declare types with tags:
`output: !Size ${x}`, or a field `size: !Size ${x}` under a block `output:`.
`output: !T {...}` is one typed value, not a record of fields.
Every key in a flow must be text: quote a `true`/`1`/`null`/date key (`'true': ${x}`).
Text fields (`code:`, `tool_id:`, `call:`, `on:`, `kind:`, `node_name:`, asserts, `uses:` refs)
must be text: quote a number, boolean, date, or a value starting with `[` or `{`
(`on:` takes a reference, `kind:` a kind name).
A scalar literal (`output: 5`, a field `x: 5`) is returned as written.

Node IDs are the keys under `nodes:`. A node id must be an identifier: ASCII
letters, digits, and `_`, not starting with a digit (`fetch_data`, not `fetch-data`
or `2x`), so that `${fetch_data.output}` can reference it. The expression keywords
`and`, `or`, `not`, `in`, `true`, `false`, `null`, and `none` can't be node ids
either. In compact mode the flow `id:`, and for a single-node `defs:` entry the def
name, also names the node and must be an identifier; a full-form flow id or a
multi-node def name may keep hyphens.

A `nodes:` entry is a node (a mapping with `kind:`) or a **value entry**, a tagged value
such as `total: !int ${a.output + b.output}` (see "Values in `nodes:`" under Typed
slots). An untagged entry without `kind:` is a load error.

A mapping can't repeat a key (two nodes with one id, a field written twice): it's
a load error at the second key. Keys a `<<:` merge brings in may still be overridden.

Unquoted values follow YAML 1.2's core rules plus unquoted dates and `<<:` merges:
only `true`/`false` are booleans, so `yes`/`no`/`on`/`off` are text (CASE's `on:`
key too); `017` is 17 (octal is `0o17`); `1e3` is a float; `1:30` and `1_000` are
text; `2026-10-14` is a date. An unquoted date-shaped value that is not a real
date (`2026-13-45`, `2026-01-01T10:00:00+05:99`) is a load error at the value;
quote it to keep it as text. A bare `call(...)` argument reads the same way, as
one scalar (`x=None` is text; write `x=null`). An unquoted date in an untyped
output (a flow or body `output:` without a tag) is a load error; quote it. So is a
number there that is not finite (`.nan`, `.inf`, `1e400`). A `yes`
in a tagged binding or default, a call input, a value entry, `steerable:`, or
`inherit:` that needs a `bool` fails with "`yes` is text; write `true`". Under a `bool` `on:`, write
`when: true`: a `when: yes` there is a load error with the same fix.

Never write a secret in a flow file: every written value is part of the flow's
fingerprint and the file is recorded in full. Pass credentials as flow inputs or from
the host (`llm_config.api_key` only from the host). No name is special: a literal
under `password` loads like any value. See `docs/syntax.md`, "Secrets stay out of
flow files".

### Compact mode — when the flow is a single node

For the "one flow, one node" case, drop the `nodes:` map and the redundant
`output: ${node.output}` line — put the node `kind:` and its fields at the top
level. The flow `id:` names the node; the flow `input:` is the node signature
(auto-wired by name); the flow `output:` is the node's output type, re-exported.
A full flow declares its output type with a tag instead
(`output: !str ${greet.output}`).

```yaml
id: hello
name: hello
input:
  name: !str                 # auto-wired into the node as ${input.name}
output: str                  # the node's output TYPE (no tag) — also the flow output
kind: agent
prompt: "Write a warm one-sentence greeting to ${name}."
```

Allowed only for the **leaf kinds** (`agent`, `code`, `tool`, `human_input`) —
`case`/`call`/`map`/`loop` need the full `nodes:` form. See
[`templates/compact.yaml`](templates/compact.yaml).

## References — how you wire values

| Write | Means |
|-------|-------|
| `${input.X}` | field `X` of the flow input |
| `${node.output}` | node `node`'s whole (typed) value |
| `${node.output.field}` | dot into an object value |
| `${name}` (bare) | a local input inside `prompt:` (AGENT or `human_input`), a `human_input` question's text, an AGENT `system_prompt:`, or `case` `when:`; AGENT prompts also see their own `variables:` |

A whole-string `${ref}` resolves to the **typed value**; embedded in surrounding
text it is **stringified**.

Prompt templates are strict: a missing reference or any span whose final value is
null raises, including an embedded span. `:-`, `|`, and `:?` still fire on a missing
or null operand (`${note :- "none"}`); only the fallback must resolve. Binding
templates stay non-strict—a whole null remains typed null and an embedded null
contributes empty text.

When the loader knows a value's type, reading a field that type can't have is a load
error: a field on a scalar (`str`, `int`, `Literal(...)`, ...) or a list, or a field a
record doesn't declare. This covers `${input.X...}`, `${node.output...}`, and a MAP's
`${item...}` in bindings, `case` conditions, and flow outputs, and the carried fields a
LOOP `while:`/`until:` predicate reads. Reading a declared field
through an `Optional(...)` record still gives `null` when the whole value is null, and
values whose type isn't known at load (`object`, `Any`, an untyped output, or anything
below them) aren't checked.

On a `Union(...)`, a field every member has reads anywhere (one type if the members
agree, the merged `Literal` if every member's field is a string `Literal`, else
untyped). A field only some members have
is a load error outside a narrowing `case` branch (see `case` below): `reference
${d.output.qty} reads field 'qty', which only union members with action buy, sell have;
read it inside a case on ${d.output.action}`. Flow outputs are never narrowed. A LOOP
`while:`/`until:` predicate narrows with a top-level `and` part before the read, as a
searched `case` `when:` does: `until: isinstance(r, Done) and r.answer == "ok"` (a check
under `or`/`not` doesn't narrow); unnarrowed, the error says to check the member first
with `isinstance(r, <Type>) and ...`.

**Operators inside `${...}`** — one expression grammar, everywhere:

| Form | Meaning |
|------|---------|
| `${a + b}`, `${a * b + 1}` | arithmetic — `+ - * / % **`, unary minus |
| `${a == b}`, `${x in [1, 2]}` | comparisons / membership / `and`/`or`/`not` |
| `${[a, b]}`, `${upper(x)}` | list literal / pure builtin call (null or wrong-typed args fail: `${upper(x :- "")}`) |
| `${isinstance(x, Buy)}` | type test; `T` is a `typedefs:` name or `str`/`int`/`float`/`bool`/`date`/`datetime`/`object` (not `Any`) |
| `${Buy(action="buy", qty=n)}`, `${Buy(x)}` | constructor: build a typedef record from exactly these fields, or check and narrow `x` (like `!Buy ${x}`) |
| `${X:-default}` | value, else `default` if absent |
| `${X:?"msg"}` | required — fail with the literal `msg` if absent (quote the message) |
| `${a \| b \| c}` | first present among peers (e.g. the branches of a routing-only `case`) |
| `$$` | a literal `$` (universal escape — in a prompt `$$` renders a single `$`) |

Arithmetic is Python's: `+` concatenates text and lists, `*` repeats them, `%` formats text
(`${"%.2f" % p}`), a bool is 0/1 (but a CASE `on:` never equates a bool with a number). No
cost limits; an operator Python can't complete (overflow), or writing an int longer than 4300
digits as text (a template, a prompt, `%`, `join`, `render_as_json`), fails like an
expression error.

`and`, `or`, `not`, and `in` are reserved operators, so none can be the root
name of a reference. A dotted field may use those names (`${record.output.in}`).

Inside an expression, name values directly and group with parentheses:
`${a :- (b :- "lit")}` or `${a | b | "lit"}`. A `${...}` inside an expression
(`${a :- ${b}}`, `when: "${a.output} > 5"`) is a load error that shows the rewrite.
The `:-`/`:?` RHS is an expression, so a string default must be **quoted**
(`${x:-"today"}`; bare `today` reads a variable). A child-flow call is the
whole-value `call(...)` directive (see below), never a `${flow(args)}` span.
Arithmetic on a value that is null when a `case` skips its node is a load error;
give it a default: `${(o.output.qty :- 0) + 1}`. This holds in a searched `when:` too:
`(o.output.qty :- 0) + 1 > 1`, not `((o.output.qty + 1) :- 0) > 1`.

For child-flow output contracts, a direct ref preserves its exact or partial contract;
selecting a known partial-record field yields an exact type. A comparison/membership or
`not`/`and`/`or` result is proven `bool`. This permits a
LOOP body field such as `exited: ${ask.output == "/exit"}` without a CODE fold.
Other computed result types remain unknown unless an explicit typed node supplies them.
In LOOP records, an unknown field does not disable enforcement of proven siblings,
including when `while:` stops before the first iteration or a run is replayed.

## Types

Scalars: `str`, `int`, `float`, `bool`, `date`, `datetime`, `object`, `Any`, `None`.
Forms use parentheses: `list(X)`, `Optional(X)`, `Literal(a, b)` (an enum of tags),
named **records**, `Union(A, B, ...)` of records, and `dict(K, V)`. Brackets
(`Optional[str]`, `int[1, 12]`) are a load error that shows the new spelling.
`date` is `YYYY-MM-DD`; `datetime` is `YYYY-MM-DDTHH:MM[:SS[.fff]][Z|±HH:MM]` (a
space may replace `T`; `.fff` is 1–6 digits); other forms are rejected.

`Any` is any JSON value except null (object, list, text, number, bool); `Optional(Any)`
adds null, and `list(Optional(Any))` is a list that may hold null (`list(Any)` can't).
Write `Any`, never `any` (load error). An `Any` value fills only an `Any` input: tag the
slot (`n: !int ${x.output}`, checked at runtime) or narrow it with `isinstance` in a
`case`. `object` is a JSON object: text keys and JSON values only, so a dict with a
non-text key, a date object, or a set fails the node. A node output with no `output:`
follows the same rule but may also hold files; a date, set, or non-text key in it fails
`contract.output` with its path (`output at d: ...`). A `file` is a `FileRef` (`uri`,
`mime`, `name`; read them as `${f.uri}`). It keeps its type inside typed containers: a
`list(file)` item, a record `file` field, or a LOOP-carried `file` reaches CODE as a
`FileRef` (`x["f"].uri`, not `x["f"]["uri"]`). A `file` slot also reads the mapping
`{uri, mime, name}`, and an `object` slot reads a nested file as its mapping; untyped
outputs keep nested files as mappings. A MAP's list is typed only when its body's output
has one exact type (a typed whole value, or a record with every field typed); over a
body returning a record with an untyped field, `${m.output}` readers in the parent and
the run result see nested files as mappings. A list the loader can't type (a
tool's untyped array, a mixed or empty literal) is `list(Optional(Any))`; its items are
checked at runtime when passed on.

Use `int(1..12)` or `float(0.0..1.0)` for inclusive numeric ranges. Either end may
be open: `int(0..)` is 0 or more, `int(..100)` is at most 100; `int(..)` is an error
(write `int`). The lower must not exceed the upper (`0..1` reads as a range). Integer
ends are exact, up to 4300 digits; float ends must be finite, and integer ends in a
float domain must convert without rounding. Booleans are invalid. Ranges compose
with lists, records, aliases, and nullable types. Defaults and supplied values are
checked; cross-flow sources must fit the destination domain, with an open end
counting as unbounded (an `int(0..)` source into `int(0..100)` needs a tag).

A `float` is finite: a binding whose value is NaN, infinity, or complex (say an
overflowing `${x * 10}`, or `${(0 - 8) ** 0.5}`) fails its node with
`binding.evaluate`, so keep the value in range (a `case` guard or a `code` clamp)
or return null explicitly (a `case` routed to a value entry
`no_value: !Optional(float) null`, or a `code` node returning `None`). An `int` has at
most 4300 digits: a longer literal is a load error, and a longer value fails where it
crosses a boundary (input, binding, output, model reply, feedback) with
`integer with 5000 digits is larger than yian allows (4300 digits)`. A temporary
inside an expression may be longer (`${10 ** n > 0}`); writing one as text fails.

A value nests at most 100 levels (each record and list is one): a binding, MAP
`over` list, or node output nested deeper fails its node, and wrapping a 100-level
value (`${[x]}`, a record flow output, a MAP's result list) adds the level that fails.
An AGENT's tool call whose arguments nest deeper fails the node before any call runs.
A value written out in the flow file (a node input, MAP `over:`, WAIT `until:`, a value
entry, a default, a CASE value branch, a flow `output:`, an `env:` value) nested deeper
is a located load error instead, YAML aliases included; a record output's fields may
nest 99.
A type nests at most 100 levels: in type text each constructor (`Optional(` and `Union(`
included) is one; in a record written as a mapping in `output:`, each record is one and
a field's type adds its records and lists (a typedef counts as deep as its definition).
Deeper is a load error: `type at r.k nests deeper than 100 levels, the limit for a type
(each record and list is a level)`.
An expression nests at most 100 levels too (each call, list, unary operator, comparison,
`|`/`:-`/`:?`, and operator chain is one, so `a + b + ...` is one level at any length);
a deeper one is a load error, so split it into smaller bindings.
A flow file's YAML nests at most 200 levels (each mapping and list is one); deeper is a
located load error.
Each YAML document (a flow file, each `uses:` file) expands to at most 100,000 nodes
(`YIAN_MAX_YAML_NODES`): each mapping, list, and scalar counts once per place it
appears, so an alias counts every node it names again and a `<<:` merge source counts
in full; more is a load error at the first mapping or list, in post-order, past the
limit, so don't build big values by stacking aliases.
CALL, MAP, and LOOP bodies nest at most 50 levels too (each body is one, whether a
`defs:` entry, an inline `nodes:` body, a `call(...)` directive, or a `uses:` flow); a
deeper chain is a load error at the `call:` that crosses the limit, so move short
bodies into their callers or split the work into separate steps.
A chain of `uses:` files, each importing the next, is held to 50 files the same way,
counted apart from call depth; a longer one fails at the alias that crosses it.
These limits are the defaults of `YIAN_MAX_VALUE_NESTING` (which also sets expression
nesting), `YIAN_MAX_YAML_NESTING`, and `YIAN_MAX_CALL_NESTING`; each can only be
lowered. `YIAN_MAX_TYPE_NESTING` sets the type limit (at most 125). `docs/defaults.md`
lists every default and limit and its variable.

Only values supplied to the top-level flow (CLI, JSON inputs, `run_flow`) may undergo
string coercion; a `call`, `map`, or `loop` edge is not a flow boundary.

```yaml
typedefs:
  Basket: list(str)                    # alias (aliases compose)
  Decision: Literal(go, no_go, wait)   # enum — one of these tags
  Signal:                              # record — fields recurse
    score: float
    note: Optional(str)
  Buy:                                 # union members: records with a tag field
    action: Literal(buy)
    qty: int
  Sell:
    action: Literal(sell)
    price: float
  Order: Union(Buy, Sell)              # one of the members
```

A type definition is written as a plain type, with no tag: typedefs and the
`output:` declaration of a node (AGENT, CODE, TOOL, HUMAN_INPUT) or of a compact flow.

`Literal` tags and record field names are non-empty text. `Literal("", a)`, an empty
field name (`"": int`), or a lone surrogate such as `"\uD800"` fails the load, the same
way wherever the type appears: in `typedefs:`, an input tag, or `output:`.

Every typedef is checked at load, even one nothing uses; a bad field type, an unknown
name, or a record that contains itself fails at that typedef field
(`typedef 'Signal' field 'tags': unknown type constructor 'lst' ...`).

A union's members must be told apart by the data: a **tag field** every member declares
as a single-value `Literal` with distinct values (any name), or shapes that can't
overlap. Otherwise the typedef fails with `members 'Buy' and 'Sell' can match the same
value; add a tag field ...`. A `file` field overlaps an `object` or `Any` field or a
record with `uri`, `mime`, `name`, so such an untagged union needs a tag field too.
Write `Union(...)`, never `A | B` (load error with a hint);
a nullable union is `Optional(Order)`, not `Union(A, None)`.
A union fills a union or `object` input member by member, never a record input: narrow
it in a `case` first, or with a tag (`buy: !Buy ${d.output}`), checked at runtime.

## Typed slots — YAML tags

One rule: a slot's type is a YAML tag on its value (`name: !T value`); a type
definition is written as a type. `name: !T` with no value is an empty tagged scalar,
not null. Full rules: [`docs/syntax.md`](../../../docs/syntax.md#typed-slots-yaml-tags).

**Where.** Flow inputs (root, `defs:`, single-node defs, compact flows); node `input:`
bindings, including LOOP seeds, HUMAN_INPUT context, and CALL/MAP/inline-body
arguments; flow `output:` fields and a single `output: !T ${...}`; agent
`variables:`; value entries in `nodes:`; and records or lists nested in any of these.

**Flow inputs.** Every entry is a tag with an optional default:

```yaml
input:
  qty: !int                          # required
  history: !list(Message) []         # default []
  model: !Optional(str)              # may be omitted; gives null
  note: !Optional(str) null          # default null
```

A default is literal YAML checked against the type at load: `${...}` inside it is
not evaluated, a null default needs an `Optional` type, and nothing is coerced
(write `!int 30`, not `!int "30"`; quote a date string, `!date "2026-01-01"`). A
compatible default keeps its authored value, so `!float 30` stays the integer `30`.
Only values supplied at a flow boundary may be string-coerced. An inline record type
(`cfg: {a: int}`) is a load error: name the record in `typedefs:` and tag the input
(`cfg: !Cfg`). `!object` is an unstructured mapping (any JSON object); `!Any` takes
any JSON value but null. An `--input` value for an `Any` input stays text; use
`--inputs` for a structure. Agent variables always need a
value (`num: !int(1..12) 4`).

**Bindings and outputs.**

```yaml
quote:
  kind: code
  input:
    size: !Size ${size.output}       # str producer into a Literal: checked at runtime
    discount: !float 0               # YAML literal: checked at load
    order: !Buy                      # a nested record, checked as a Buy
      action: buy
      qty: ${size_qty.output}
  output: float
  code: ...
output:
  size: !Size ${size.output}
  price: !float ${quote.output}
```

**Writing the tag.**
- Inside a tag there are no commas. Name a multi-argument type (`Union`, a
  multi-value `Literal`, `dict(K, V)`) in `typedefs:` and tag with its name
  (`Decision: Union(Buy, Sell)`, then `!Decision`). One-argument forms and ranges are
  fine inline: `!Optional(str)`, `!list(Buy)`, `!int(1..12)`.
- A tag ends at the first space, so `!Union(A, B)` is a load error that says to name
  it in `typedefs:`. `[ ] { } ,` in a tag are load errors (strict YAML 1.2 forbids
  them, and flows stay valid for YAML 1.2 tools such as VS Code's).
- Inside a `{...}` flow mapping, a comma splits an unquoted multi-argument type, so
  quote it or use block style. After a tag inside `{...}`, put a space before
  `,`/`}`/`]`, or use block style.
- Tags anywhere else are load errors: on typedefs, node `output:` declarations, other
  fields, keys, and `<<:` merge sources. Tags under `x-*` keys are ignored. YAML's own
  `!!` tags keep their YAML meaning.

**Tags check.** A tag never fills in anything and converts only how files are
held (a file's mapping at a `file` slot becomes a file; a nested file at an `object`
slot becomes its mapping). `n: !float ${i}` still delivers an `int`.
- **Load:** the declared type is compared with the inferred one. Disjoint is a load
  error (a `str` declared `int`; records are closed, so an extra required field
  is disjoint). Narrower (non-Optional over Optional, a `Literal` or range subset, a
  record over `object`, an `Any` value into another type (not `None` or `file`), a
  union member over the union) loads and is checked at
  runtime, as is a value with no inferred type (arithmetic, builtin calls, untyped
  producers).
- **Values:** YAML 1.2 core rules plus unquoted dates apply and quoted values stay
  text: `!int "5"` is an error, `!str 30` and `!str 2026-01-01` fail with a hint to
  quote (`!str "30"`), and `!bool yes` fails: `yes` is text, write `true`.
- **Records:** a record missing a required single-value tag field (`action:
  Literal(buy)`) fails, and the error names the field. Write the field.
- **Runtime:** the value is validated before the consuming node runs, with no
  conversion (an int bound as `float` stays an int; `"30"` is not an int). A miss
  fails the consumer (END for a flow output) with code `binding.type`, located at
  the binding. A committed output widens an int in any typed `float` slot (the
  whole value, a list item, a record field); an untyped record field stays as given.
- **Downstream sees the declared type** — this is how an opaque value becomes
  typed. CALL/MAP/LOOP parents see a child's declared output type; a CALL or LOOP
  checks a tagged argument or seed against the child's or body's parameter; an inline
  body's parameter takes the declared type (so arithmetic or `[]` needs no `defs:`
  body); a LOOP field the body omits keeps the seed's declared type, and a typed seed
  checks the starting value.
- **Optional:** an untyped nullable source into a strict child or LOOP body
  parameter is still a load error (also a child output that is `Optional` because a
  `case` may skip its node); `x: !int ${maybe.output}` accepts it with a runtime check.
- **Replay:** declared types are in the flow fingerprint, so a run can't resume
  after a type changes; recorded flow outputs are re-validated on resume.

**Nested values.** `${...}` inside mapping and list values of a slot is evaluated,
and each field keeps its type. `$$` is a literal `$` at any depth. A nested
`call(...)` is a load error; hoist it to its own node or binding. Each nested
`${...}` acts like its own input for skips: a node whose nested whole-value read's
producer was skipped is skipped; a flow output gives null at that nested position. A
literal list with `${` under MAP `over:` or WAIT `until:` is a load error; write the
list as an expression `${[a, b]}`.

**Values in `nodes:`.** A tagged `nodes:` entry is a value, read like a typed binding;
other nodes read it as `${<id>.output}`:

```yaml
nodes:
  count: !int 3                              # scalars are tagged too
  greeting: !str Hello, ${input.name}
  total: !int ${a.output + b.output}
  exit_result: !object {exited: true, reply: ""}
  decision: !Buy                             # a block record, checked as a Buy
    action: buy                              # the tag never fills this in
    conviction: ${score.output}
```

A tagged entry is always a value, even with a `kind:` key inside. An untagged mapping
without `kind:`, scalar, or list is a load error that says to add `kind:` or a tag. A
span-free literal is checked at load; nested `${...}` and tags work, `$$` is a `$`, and
a top-level unquoted `call(f, ...)` calls the flow (a quoted `!str "call(f)"` is text).
A written-out `!object` mapping keeps its record, so reading a field it doesn't have is
a load error; `!object ${x.output}` and a `!Any {...}` entry stay unchecked. The entry
lowers to a CODE node with its id (in traces, failures, and the run log like any node)
and runs like one: on its branch when a `case` routes to it, else when its inputs are
ready. A runtime type miss fails it with `binding.type`. An entry that reads itself is
a cycle.

**Constructors.** Inside `${...}` there are no YAML tags, so a typedef name builds or
checks a typed value: `${Buy(action="buy", qty=size.output)}` builds a record from
exactly the given fields and checks it; `${Buy(x)}` checks and narrows `x`, exactly
like `!Buy ${x}`. Only typedef names are constructors (`${int(x)}` is an unknown
function: yian never converts). They work where `isinstance` does, not in prompts.
`isinstance(x, Buy)` stays the boolean check. See
[`docs/expressions.md`](../../../docs/expressions.md#constructors-t).

**Old forms are load errors that show the new spelling:** `qty: int = ${x}` (says
`` `qty: !int ${x}` ``), `name: T = default` inputs, agent `T = default` variables,
untagged input entries (`qty: int` -> `qty: !int`), inline record input types, and
bracket types anywhere (`Optional[str]` -> `Optional(str)`, `int[1, 12]` ->
`int(1..12)`). A value whose left side isn't a type (`note: a = b`) is plain text.

## Node kinds (the closed set — you compose these, you don't define new ones)

### `agent` — the LLM-backed AGENT node
```yaml
classify:
  kind: agent                  # no tools: one model call; tools:/skills: run the tool loop
  input:
    text: ${input.text}        # bindings — these infer the data edges
  output: Literal(positive, neutral, negative)
  prompt: |-
    Classify the sentiment of: ${text}
    Answer with exactly one of: positive, neutral, negative.
```
**An AGENT node's `output:` may be any shape.** Only a bare, untagged `str` keeps the
node a text producer. A `Literal(...)` enum, record, union, non-string scalar, or list
switches it to **structured generation** — the engine derives a schema from
`output:` and the model emits a conforming value (native structured output, or a
JSON fallback for providers that lack it), validated at the write boundary and
retried up to `retries:` times (default 2, `YIAN_AGENT_RETRIES`; a non-negative integer)
on deviation, a NaN or infinity in the reply, a value nested deeper than 100 levels, or a
reply that skips the structured-output tool call included. A root `Literal(...)` returns
its bare tag to the flow. A union returns the matching member's record; each member's
schema pins its tag and rejects unknown fields. An `object` or `Any` output is an open
mapping or any JSON value; where structured output is strict (Azure, the OpenAI Responses
API, Anthropic with `thinking` on) yian asks for that whole schema through a non-strict or
unforced tool instead, and still validates and retries the reply. A range such as
`int(0..10)` goes to Azure in the field's description rather than as `minimum`/`maximum`,
and is still checked on the reply. Use a downstream `code`
node only for deterministic post-processing.
`prompt:` must be text: quote a number, boolean, date, or a value starting with `[` or `{`;
otherwise it is a load error at the value.
An AGENT with `tools:` or `skills:` runs the tool loop; without them it makes one model
call (text, or structured for a typed `output:`). `mode:` was removed; writing it is a
load error.
To give an AGENT node ordinary tools: `tools: [tool_id, ...]` (always a list, even for one).
A `tools:` name may also be a host-filled tool group (see "Tool groups" below).
To set who the model is: `system_prompt: "You are ${name}, ${title}."`. It
**replaces** the framework's default system instruction (omit it to keep the default)
and renders like `prompt:` — bare local inputs and the node's own `variables:`, with
located load errors for unknown names. Granted skills, then recalled memory, append
after it. Every model request of the node (one model call, tool rounds, structured generation)
uses it; it is part of the flow fingerprint, and with `learnable=True` the rendered text
is recorded for backward. Compact flows accept it at the top level; there is no
flow-level default.
To give it host-supplied Claude/Codex skills: `skills: [skill-name, ...]` or
`skills: all`. The system message lists each
granted skill's name and description; the model calls the built-in
`load_skill(name, path=None)` to read `SKILL.md` or a bundled file. `load_skill` is
reserved: never list it in `tools:`. When a flow grants skills, `yian run` supplies
`~/.claude/skills`, `~/.agents/skills`, and the project's `.claude/skills`/`.agents/skills`
(personal wins). A library host
passes `skills=` (`{}` for none); without it, or with a listed skill missing, the run
fails before it starts.
To let a watching human steer or interrupt the agent, add `steerable: true` (requires
`str` output or no `output:`; anything else is a load error). Without tools, an interrupt
cuts the streaming text reply.
The host passes `run_flow(..., steering=inbox)`. Queued steers join as user messages at
the top of the next iteration; an interrupt ends the loop at the next safe point and the
node **succeeds** with its text so far, so downstream nodes and loops continue. Leave it
off for checks and graders that must not be talked out of their job.
A node may pin its model via `llm_config:`; otherwise the process defaults captured
once when the run initializes apply. `inherit: false` skips authored flow/parent/CLI
layers but still receives those provider/model defaults.
Node, `defs:`, and flow `llm_config:` are all checked at load against the same
fields; `inherit:` is node-only. A non-finite number (`openai: {temperature: .nan}`)
is a load error.
The top level holds only `provider`, `model`, and `inherit:`. Request settings go in a
block named after the provider (`openai`, `azure`, `xai`, `deepseek`, `qwen`, `qwen-cn`,
`glm`, `glm-cn`, `minimax`, `minimax-cn`, `openrouter`, `vllm`, `anthropic`, `google`,
`ollama`), written exactly as that provider's API request body and sent as written:
`openai` is the Responses API, `anthropic` the Messages API, `google` Gemini
`generateContent` (camelCase, `thinkingLevel: HIGH`), `ollama` `/api/chat` (`think`,
`options`, `keep_alive`, ...), the others their vendor's Chat Completions. A node uses
the block of its resolved provider; the others are inert but still checked.

```yaml
llm_config:
  provider: anthropic
  model: claude-opus-4-8
  anthropic: {max_tokens: 16000, thinking: {type: adaptive}, output_config: {effort: high}}
  openai: {reasoning: {effort: high}}      # used only if a layer picks provider: openai
```

Fields yian builds (messages, system prompt, tools, tool choice, structured output,
streaming, conversation state, `n`/`candidateCount`) are a load error in a block. A
limit of one model is the provider's to enforce: `temperature` on newer Claude models
fails with the provider's error, so prefer `max_tokens`, `output_config.effort`, and
`thinking` there. A ChatGPT-subscription `openai` agent is re-checked at run start
without `temperature`/`max_output_tokens`. Retired keys fail with their block form:
`anthropic_effort` → `anthropic: {output_config: {effort: …}}`,
`openai_reasoning_effort` → `openai: {reasoning: {effort: …}}`,
`google_thinking_level` → `google: {generationConfig: {thinkingConfig: {thinkingLevel: …}}}`,
`temperature` → `<provider>: {temperature: …}` (google `generationConfig.temperature`,
ollama `options.temperature`).
OpenAI subscription and API-key connections both use `provider: openai`;
`yian auth openai` selects the method and saves the model.
Never put `api_key` in a flow's `llm_config:`, nor `base_url` except beside
`provider: azure` (there it is the resource endpoint, required, and `model` is
the deployment name); loading fails. A
Python host passes them in `run_flow(..., llm_config={"provider": ...,
"api_key": key, "base_url": url})`, and the key again to
`resume_flow`/`result.backward` as `api_key=key`. They reach every AGENT of that
provider, `inherit: false` ones included.
No provider or model is built in. Process defaults are `YIAN_DEFAULT_PROVIDER`/
`YIAN_DEFAULT_MODEL`, then the login saved by `yian auth <provider>` (one per
provider, each with its model); a default model applies only to its provider. An
agent that no layer gives a provider and model fails the run before any node runs:
`Agent 'write' has no provider or model; set llm_config.provider/model, pass
--provider/--model, or run 'yian auth <provider>'`. Grant ordinary Yian
tools explicitly; retired `provider: codex`, `sandbox`, `cwd`, and `builtin_tools`
settings are rejected.
Tune a node's non-model behavior with a static `env:` block (flow-level default +
per-node override, node wins): e.g. `env: {max_tool_iterations: 300}` bounds an
AGENT's tool-calling loop (default `-1` = no cap, `YIAN_AGENT_MAX_TOOL_ITERATIONS`).
`env:` values are literals, checked at load against the 100-level value nesting limit,
and must not contain themselves.
`retries` bounds typed-output correction, not tool, transport, authentication, or stop
failures.

### Agent-owned variables — `variables:`

Use tagged defaults for local prompt settings, alongside the AGENT node's inputs.
A multi-value `Literal` is named in `typedefs:` (`Tone: Literal(concise, detailed,
technical)`) and tagged by name:

```yaml
writer:
  kind: agent
  variables:
    num_sentences: !int(1..12) 4
    tone: !Tone concise
  input:
    topic: ${input.topic}
  prompt: Write ${num_sentences} ${tone} sentences about ${topic}.
  output: str
```

Every variable needs a tag and an explicit, valid default, including nullable
variables (`note: !Optional(str) null`). Names cannot overlap inputs. Defaults are
literal values; `${...}` inside a default string stays text. A default nested deeper
than 100 levels is a load error; a NaN or infinity default fails the node when it
runs. Each invocation receives a fresh copy and repeated prompt references read the
same local value.
Ownership follows the authored AGENT path plus name across MAP items and LOOP
iterations. These declarations create no data edges. Model settings still belong in
`llm_config:` even when a
variable has the same name, such as `temperature`.

### Agent memory — `long_term_memory:`

`long_term_memory:` names one or more memories on a full flow (`nodes:` form, root
file or `defs:` entry) or on an AGENT node. Each memory has a name and a type. YAML
holds public behavior config only, never storage paths, credentials, or Python
imports. Three forms:

- `long_term_memory: episodic` — one memory named after its type;
- `long_term_memory: {notes: episodic, chat: session}` — one memory per `name: type`;
- `long_term_memory: {notes: {type: episodic, config: {max_episodes: 50}}}` — long
  form; only `type` and `config` are allowed.

`null` is the same as leaving the key out. `{}` and the retired
`{name: X, config: ...}` form are errors; `name`, `type`, and `config` cannot be
memory names. Built-in types and their config keys: `episodic` (`max_episodes`,
`recall_limit`), `reflective` (`max_items`, `recall_limit`), `session`
(`history_budget_chars`, `recent_turns`), `profile` (`max_chars`,
`input_budget_chars`). Values are integers of at least 1 (`recent_turns` at least
0); episodic and reflective limits may also be `null`. A bad key or value fails at
load. Any other type is custom and needs a Python host. `storage`,
`store`, `path`, and `api_key` are rejected in config.

A memory's key is its name prefixed by the `uses:` aliases it was imported
through: `notes` in the root file, `lib.notes` in a file imported as `lib`. The same
key declared twice in one file is one memory, so type and config must agree; two
libraries that both declare `notes` stay independent. Keys that differ only by case
are an error. An AGENT recalls each enclosing flow's memories, outermost first,
through CALL, MAP, and LOOP, then its own; a repeated key appears once. A child's
declaration never binds to a parent's memory. Other node kinds and anonymous inline
CALL/MAP/LOOP bodies cannot carry `long_term_memory:`; use a named `defs:` flow.
Each declaration's form and config are checked for every def and import; the rules
that compare declarations (one type and config per key, session placement, case
collisions, alias names) apply only to declarations reachable from the root.

A `session` memory is declared on exactly one AGENT, never on a flow, and an AGENT
declares at most one. It supplies a structured transcript, while other memories
supply notes; preserve transcript roles and tool results, and do not flatten
history into a note. Every call site of that AGENT shares one conversation:
repeated calls, LOOP iterations, and MAP items queue in admission order, not MAP
input order, so use non-session memory for independent items. History updates
automatically during forward, without learning. Explicit API backward
self-feedback can become committed session notes for later calls, without another
model call; recall is pinned when the run opens. `history_budget_chars` (default
32_000) and `recent_turns` (default 4) control API summaries of older context; raw
history and recent whole turns stay intact, even above the soft budget.

`yian run` builds every built-in memory in RAM, or on disk with
`--memory-root DIR` (key `lib.notes` lives at `DIR/<flow id>/lib/notes.sqlite`) and
repeatable `--memory-storage KEY=PATH`. It cannot run a flow that declares a custom
type, and it never runs backward learning. `yian chat` and its `run_flow` tool
supply no long-term memory: declared memories stay off (session conversations are
not kept). Supply the same storage on unfinished
`--resume`; terminal inspection opens none. A storage file has one owner until its
backend closes. From Python, pass
`memory=build_memories(flow, root=..., storage=..., supplied=..., factories=...)` to
`run_flow`; `memory=None` leaves declarations dormant.
Independently, `learnable=True` retains
execution for a later explicit `result.backward(feedback)` call; forward returns
before memory learning or learned-mutation commits. `result.backward()` with no
external feedback still gives every memory key one learn call, while unreached
AGENT backward model calls are skipped. The separate outcome never changes forward
values or updates local variables. Terminal `resume_flow` inspects without learning;
it does not retry a backward attempt. See the [Python API](../../../docs/api.md#explicit-backward-feedback)
for feedback shapes and repeated backward attempts.

### `code` — deterministic Python
`code:` is either a **`module:function`** reference (import + call) or **inline source** —
a bare body that reads the `inputs` dict and `return`s a value (the engine wraps it as
`def main(inputs)` and runs it in-process; same one-dict convention, so a body promotes to
a reference by copy-paste). See [`templates/inline-code.yaml`](templates/inline-code.yaml).
The fingerprint holds a reference's `module:function` text, not the function's source:
keep a referenced function unchanged until a resumable run finishes.
```yaml
verdict:
  kind: code
  input:
    s: ${score.output.signal}
  output: Signal               # one typed value; an AGENT node supports this too
  code: pkg.mod:fn             # a reference — OR an inline `code: |` body that returns
```

### `tool` — a registered tool (no LLM)
```yaml
typedefs:
  Fact: {text: str, source: str}
nodes:
  news:
    kind: tool
    tool_id: get_facts         # a registered or per-run tool id
    input:                     # the tool's own argument names; a ${...} ref or a literal
      symbol: ${input.topic}
      limit: 10
    output: list(Fact)         # optional; every returned value is checked against it
```
Register the tool in Python so the id resolves:
```python
from typing_extensions import TypedDict   # typing.TypedDict fails in pydantic on 3.10/3.11
from yian.tools import register_tool

class Fact(TypedDict):
    text: str
    source: str

@register_tool("get_facts")
def get_facts(symbol: str, limit: int = 10) -> list[Fact]:
    "Fetch recent facts for a symbol."   # docstring = the description the model sees
    ...
```
Or give it to one run only: `run_flow(flow, inputs, tools={"get_facts": get_facts})`.
That mapping is the run's whole tool set: registered tools are hidden. To keep
them, pass `tools={**registered_tools(), "get_facts": get_facts}`.

**Typed TOOL nodes.** Every returned value must fit `output:`; model expected outcomes
as union variants (`Union(Filled, Interrupted)`) and route them with `case`. At run
start, unknown or missing inputs and input or output types that can never fit raise
`ValueError`, with no coercion (`qty: "10"` into `int` fails). See `docs/syntax.md`.

**Tool groups.** When the tools an AGENT may use change between runs (for example,
tools a permission system grants), grant a group name and let the host fill it:
`tools: [toolbox, workspace, ask_user]` in YAML, and
`run_flow(..., tools={"toolbox": [t1, t2], "workspace": [], "ask_user": ask})` in Python.
A list value is a group; each item is a `BaseTool` (named by `.name`) or a def-defined
function (named by `__name__`). The model sees and calls the group's tools by their
own names. Groups may share the same tool object; the model sees it once. The run
fails before it starts if a granted group is missing, two different tools share a
name (or a group tool's name is a top-level key), or a group has a tool named
`ask_user` or `load_skill`; the handler stays the top-level `ask_user` entry. `yian run`
fills no groups, so such flows need a Python host. Pass groups again to `resume_flow`;
restarted nodes use the contents passed there.
Note `tool` uses `tool_id:` + `input:`, with an optional `output:`. ML inference
goes through a host tool (`kind: tool`); `kind: model` was removed.

### `case` — branching (no `input:`)
It takes one branch; the other targets skip, and so does every node that plainly
reads (or `depends_on`) a skipped one. A searched `case` that reads a node only under
a default that keeps a binding running (`(a.output :- 0)`) runs when that node is
skipped, like the binding would; `(a.output :- b.output)` skips it only when both are. Read the taken branch's result
with `${route.output}` (or `${route.output.field}`). A routing-only `case` (never read
via `${route.output}`) may take none when no `when:` matches and there is no `else:`;
over a `Literal` node output or flow input it still needs a `when:` for every tag.
```yaml
route:
  kind: case
  on: ${synth.output}          # simple form: match a Literal value
  cases:
    - when: go
      then: go_brief
    - when: no_go
      then: no_go_brief
  else: more_info_brief        # runs when no `when:` matches
```
Searched form (first true `when:` wins, no `on:`): `when: "score.output >= 0.5"`.
Each outermost `:-`/`:?`/`|` reads like its binding: `(score.output :- 0) >= 0.5` routes
when `score` is skipped, `(a.output :- b.output) >= 1` is skipped only when both are.
`:-` binds loosest, so parenthesize it: `x.output and (a.output :- false)`.
`cases:` is a list of `{when:, then:}` mappings; any other shape is a load error.
Routing on a `Literal` node output or flow input (a LOOP body's carried field, a child
flow's input) is **exhaustiveness-checked** — omitting a tag with no `else:` is a load
error.

**Matching.** Each `when:` is a typed value of the `on:` type (text, a number, or a bool;
an unquoted date is its ISO text; quote a datetime). Ints and floats compare numerically,
a bool never equals a number, and quoted values are text. Where the `on:` type is known, a
`when:` it can't hold is a load error even with an `else:` (``case node 'route': `when: 1`
is an int but `on: ${input.n}` is `str`; quote it: `when: "1"` ``). An untyped `on:` also
compares exactly: `when: 1` doesn't match the text `"1"`. Two `when:`s with the same
value (`when: 1` and `when: 1.0`; not `1`, `true` and `"1"`) are a load error at the
later one.

**Branches.** A `case` only routes. A `then:`/`else:` is a sibling node (an untagged
unquoted word) or a child call (an unquoted `call(...)` as the whole value). Any node
name works as a target, including `default` and a value entry. For a fixed or computed
result, route to a value entry:
```yaml
exited: !Turn                  # a value entry: the exit record itself
  transcript: ${input.transcript}
  exited: true
route:
  kind: case
  on: ${ask.output}
  cases:
    - when: /exit
      then: exited
  else: call(respond, transcript=${input.transcript}, message=${ask.output})
```
A value in a branch (a tagged value such as `!str reply`, quoted text, a number, a bool,
`null` or an empty `else:`, a date, a mapping, a list, unquoted text with `${`) is a load
error that shows the rewrite (``case node 'route' then: a branch names a node or calls a
flow; put the value in its own entry in `nodes:` and name it:
`route_then_0: !str approve` and `then: route_then_0` ``). An unquoted word that names
no node suggests the nearest node and a value entry (``Did you mean `reply`? For the
text, add a value entry and name it: `replyy_text: !str replyy` ``). A `case` has no
`output:` (a load error: tag the result where you read it). `${route.output}` reads
whichever branch ran; type it with a tag where you read it
(`output: !Turn ${route.output}`). A CASE read through
`${case.output}` (or routed to by one) needs `else:` or an `on:` Literal with every tag
listed, or a non-nullable `bool` with `when: true` and `when: false`. `${case.output}`
works as a whole binding or inside text, not in a coalesce, `:-` default, expression,
`when:`/`on:`, or assert.

**Skipped branches in a flow output.** An output field that reads only nodes a `case`
may skip (`${order.output}`, `${a.output | b.output}`, `${a.output :- b.output}`)
returns `null` when they were skipped, typed `Optional(T)`; the loader reads it as if
`:- null` were written. This holds in `defs:` children and inline CALL/LOOP/MAP bodies.
A join over an exhaustive `case` (`else:`, every tag of a non-nullable `Literal`, or a
non-nullable `bool` with `when: true` and `when: false`) is never null. A tagged
`x: !str ${order.output}` loads and fails with `binding.type` when skipped. Templates
and comparisons are unchanged (embedded null renders as empty text). Arithmetic on such a
value is a load error asking for `:-`.

**Narrowing a union.** `on: ${d.output.action}` over a union's tag field narrows
`d.output` in each branch: a `when: buy` target (and nodes chained to it by plain refs or
`depends_on`) can read `Buy`-only fields; `else:` sees the unlisted members. A join
`${b.output | s.output}` of `buy` and `sell` branches sees both. `on:` must be one direct
ref to the tag (not a coalesce); a `when:` that isn't a tag is a load error (``case node
'route': `when: buyy` is not a tag of `on: ${d.output.action}` (`Literal(buy, sell)`);
use one of: buy, sell``). A searched `case`
narrows with a top-level `and` part `isinstance(<ref>, T)`:
`when: isinstance(d.output, Buy) and d.output.qty > 5`. Its `else:` doesn't narrow.
A searched `case` whose `when:` reads a node only under such a default doesn't pass
that node's narrowing to its branches. Inline
child bodies are typed before narrowing; read member fields in a branch node.

### `human_input` — a guaranteed human gate
**Always** calls the run's `ask_user` handler and blocks its worker until the
host returns a typed answer (validated against `output:`). It stays inside the
live node call and returns the answer as an ordinary node output.
```yaml
approve:
  kind: human_input
  input:
    plan: ${draft.output}             # context the prompt may reference (bare ${plan})
  prompt: |-
    Here is the plan:

    ${plan}

    Approve as-is, or revise? (approve / revise)
  output: Approval                    # a typed answer, e.g. a Literal enum
```
Instead of a typed `prompt:`+`output:`, a gate can present **questions** (1–4,
`YIAN_MAX_QUESTIONS`; AskUserQuestion-shaped) — the answer is a record keyed by each
`header`; `output:` then defaults to `object`. `prompt:` is optional; `questions:`
and `adaptive_questions:` are mutually exclusive. The gate itself never chooses or
calls an LLM: `adaptive_questions:` is LOAD-TIME sugar that lowers to a
synthetic composer AGENT node + gate; the composer carries the gate's `case` routes,
`depends_on:`, and `runs_after:`, and waits for the gate's other inputs, so the composer makes
no LLM call when the gate can't run. The gate only calls the host's installed interaction tool.
```yaml
ask:                                  # static questions list
  kind: human_input
  input:
    proj: ${input.proj}               # ${proj} renders into question text
  questions:
    - question: "Which framework for ${proj}?"
      header: Framework               # answer key -> {Framework: <label>}
      options: [ {label: React, description: a lib}, {label: Vue, description: a fw} ]
    - { question: "Any notes?", header: Notes }   # no options -> free-text
ask2:                                 # LLM-composed questions (desugared at load)
  kind: human_input
  input:
    ctx: ${research.output}
  adaptive_questions: { prompt: "Design 1-3 questions for: ${ctx}", retries: 3 }
```
(On an adaptive gate, `__questions` is reserved for the generated composer wire;
do not declare it in `input:`.)
Questions may also come from an upstream node: `questions: ${qs}`.
The prompt, the adaptive brief, and a literal list's question text (`question`, option
`label`/`description`) read the gate's `input:` names bare and are checked at load; `header`
is not a template. A bad brief is reported as `node '<gate>' adaptive_questions.prompt`,
and a bad `llm_config` as `node '<gate>' adaptive_questions.llm_config`. A cycle through
the gate and a model-config error at run start also name the gate, not `<gate>__compose`.
`adaptive_questions:` takes only `prompt`, `llm_config`, and `retries`; any other key is a
load error.
`questions:` is a list or exactly one `${name}` naming a declared input;
anything else (a mapping, a number, text around the ref, `${a.b}`, a call) is a
load error. Put free text in `prompt:`.
The prompt, the adaptive brief, and each question's `question`, `header`, `label`, and
`description` must be text: quote a number, boolean, date, or a value starting with `[` or `{`.

### `wait` — block until an absolute UTC deadline
```yaml
settle:
  kind: wait
  until: ${input.as_of}
  # order a downstream node after it with depends_on: [settle]
```
`until:` resolves to a `date` (`2026-05-21`) or `datetime`
(`2026-05-21T14:30[:00][Z|+07:00]`). Date and naive datetime values
are UTC; aware values are normalized to UTC. A past/current deadline returns
immediately. Otherwise WAIT holds one worker until the remaining wall-clock
duration elapses, then returns `None` as an ordinary node output.
An `until:` that can never be a date or datetime fails at load: a literal that is not a
YAML date/datetime or date text, or a reference typed `int`, `bool`, `list(...)`, etc.
(a `str` reference loads and is parsed at run).

### `call` / `map` — composition over child flows
A child is itself a flow. `call` runs it once; `map` runs it per list element
(`${item}` is the current element). Name a reusable child with `call:`:
```yaml
uses:
  research-one: 03-research-one   # bind an external sibling flow to a local alias

nodes:
  each:
    kind: map
    over: ${input.topics}         # a list(T)
    call: research-one            # node value is list(U)
    input: { topic: "${item}" }   # per-element call args (sink-bind the child's params)
```
`call:` resolves **defs-first, else a `uses:` alias** (an external flow on the
search path, by filename; `alias@v1` adds a version guard). A CALL's record fields
are available as `${one.output.field}`; a MAP returns a list at `${each.output}`,
so project element fields in a downstream CODE/MAP step. `parallel:` is accepted
but currently inert; run-level `num_workers` controls MAP concurrency.
`over:` must be a list: a non-list literal (quoted `call(...)` text included) or a
reference whose type is not a list or `Any` (nullability aside) fails at load.

Each argument must fit the child's input type at load: a reference by its type
(`${item...}` through the `over:` element type, or each element of a literal `over:`
by value), a YAML literal by its value (`"2026-01-01"` fits `date`; `9` for
`int(1..3)` does not), `${x :- d}` by `x` without its null plus the fallback `d`, and
`${a | b}` operand by operand. A `call`/`map` edge is not a flow boundary, so nothing
is coerced across it: `"3"` does not fit `int` (write `3`), an unquoted `2026-01-01`
(a YAML date) fits neither `date` nor `str` (quote it), and text from an untyped
source into an `int` input fails at the child's start (`contract.input`). A `str`
reference does not fit an `int` or `date` input: type the source (`list(date)`), or
tag the binding (`t: !int ${input.n}`) for a value narrower than the input.
A LOOP seed is checked the same way against its body's input.

For a one-off child, replace `call:` with a non-empty `nodes:` body and add the
required `output:` key:

```yaml
each:
  kind: map
  over: ${input.topics}
  input:
    topic: ${item}                # generates child parameter `topic`
  nodes:
    normalize:
      kind: code
      input:
        topic: ${input.topic}
      output: str
      code: |
        return inputs["topic"].strip()
  output: ${normalize.output}
```

Use exactly one of `call:` or `nodes:`. The `output:` key is required; CALL may
set it to null and return `None`, and `output: {...}` returns a record even with one
field. MAP requires a non-empty `output:`; LOOP requires at least one carried field.
The inline child cannot capture the parent pool: pass parent values and MAP's item
through the outer `input:`, then read generated `${input.X}` parameters and
child-node outputs inside. Run-global `${system.today}`,
`${system.now}`, and `${system.run_id}` remain available. Types are inferred
strictly from YAML null/bool/int/float/string values, structural containers, or
direct typed refs; refs preserve structure/nullability, and only a lone
whole-value `${item...}` lazily requires a proven `over:` element type. An
optional element admits actual null list items; a specialized list keeps its
storage kind and preserves `None` through lossless round-trips. Cross-flow
bindings accept strict-to-nullable elements but reject nullable-to-strict. If
inference cannot prove a type, declare it with a tag
(`n: !int ${input.a + input.b}`) or write a named `defs:` child with an explicit
signature. `${...}` nested in mapping and list arguments is evaluated, and each
field keeps its type. Other
fields keep their outer-node meaning (LOOP drivers, `env:`, and CALL/LOOP
asserts; MAP asserts remain invalid); `parallel:` remains accepted but inert.
Inline child flow configuration is not a separate surface, and `llm_config:`
remains invalid on composite nodes. The enclosing flow's LLM cascade still
reaches body AGENT nodes. See
[`templates/inline-composites.yaml`](templates/inline-composites.yaml).

### `loop` — iterate a body until a predicate goes false (the engine's `while`)
Threads ONE carried record from each turn into the next: the body reads the carried
record and returns the fields that change (`'a -> 'a`), and the loop re-runs while
`while:` holds.
```yaml
turn:
  kind: loop
  call: chat_turn              # the body (a def or a `uses:` file)
  input:                       # the SEED carried record — defines 'a
    messages: []
    exited: false
  while: not exited            # pre-check over the carried record
```
A LOOP's `input:` is a mapping of carried fields; `input: ${seed.output}` is a load
error, so list each field (`n: ${seed.output.n}`).
A tagged seed (`n: !int 0`) checks the starting value; a field the body omits
keeps its declared type. Every seed must fit the body's input like a CALL argument
(`n: "1"` into `int` fails; write `1`). Tag an `Optional` reference that is always set
(`n: !int ${input.m}`); a seed that may be null, such as a cursor a
`while: cursor != null` guards, needs an `Optional` body input, narrowed inside the
body with a tag where a strict value is needed.
A named body's returned field must fit the body's own input, since the next iteration
binds it there: returning `Optional(int)` into `n: !int` is a load error that asks you to
make the input `Optional(int)` or tag the seed (`n: !int ${input.m}`) to leave the
null check to runtime; an `Any` field needs `isinstance` narrowing in a `case` or a typed
producer, and any other mismatch (`float` into `int`) needs the body's input and output
types to agree.
`loop` accepts the same inline `nodes:` + `output:` form instead of `call:`; its
generated parameters are the carried `input:` fields, so the same `'a -> 'a`
contract still applies. Those parameters are typed from the seeds, and each returned
field must fit its seed's type, so type a `${null}` seed the body fills:
`seen: !Optional(list(str)) ${null}`. A tagged `note: !str "start"` seed accepts an
`Optional(str)` field and rejects a null at the next iteration (`contract.input`); an
untyped `"start"` seed asks for `note: !Optional(str) "start"`.
The body's `output:` returns a **subset** of the carried record (at least one field);
each omitted field keeps its previous value, with nothing to declare. A body field
that is not carried is a load error. The body reads only a **subset** of the carried
names. Names are checked at build; each returned field must keep its carried type,
compared at load for proven body/seed pairs and always enforced at runtime and replay.
Returning null sets a nullable field to null, and so does a field that reads a node a
`case` skipped (it does not keep the previous value). A literal `${null}` is typed
`None`, so it does not fit an `Optional(str)` seed; return a nullable-typed producer
instead. A body may return one field: `output: {done: ...}`. An empty `output: {}` returns no
field and is rejected. A `while:`/`until:` that reads only fields the body never
returns cannot change. Without `max:` it is a load error; with `max:` it fails if it
reaches the cap. The node's value and the LOOP node's own `asserts:` see the merged
carried record; a body def's own `asserts:` see the body's own output. A body that
blocks (a `human_input` leaf) makes the loop a live chat REPL: the same run threads
each answered turn. Pick **exactly one**
driver: `while:` (pre-check, 0+ runs), `until:` (post-check / do-while, 1+ runs,
stop when the predicate becomes true), or `times: N` (fixed count). `max:` is
optional for `while:`/`until:` but **redundant and rejected** with `times:`. A
`while:`/`until:` is an expression string; `while: true` or `until: 1` is a load error.

#### Conversational REPL — the `loop` shape used by `yian chat`
A chat is a `loop` whose body blocks each turn at `ask` (`human_input`), then a `case`
picks the turn's record (the carried `{transcript: str, exited: bool}` `Turn`). Exact
`/exit` and `/quit` route to `exited: !Turn {...}`, a value entry that keeps the
transcript and sets `exited: true`; `else:` calls a `respond` def whose `reply` AGENT
node and transcript fold return the same record. The body's output is
`!Turn ${route.output}`: the tag types whichever branch ran as the carried record.
Typed exit is flow-owned — a host must not intercept it — while terminal EOF may be
normalized to `/exit`. See [`templates/chat.yaml`](templates/chat.yaml).

## Run-ordering without data (`depends_on` / `runs_after`)
Both gate a node on another **settling** even when no value flows. `depends_on:
[x]` also **co-skips** the dependent if `x` skipped; `runs_after: [x]` orders only
(the dependent still runs). Use these for `wait`, side-effecting tools, etc.
Each is a list of node ids: a bare `depends_on: x` is a load error.
Naming a `case` itself in either is not a branch: the dependent runs whichever branch
the `case` takes, and is skipped only when the `case` is.

## The three expression contexts (one grammar)
One `${...}` grammar; the context decides what happens to the result:
- **Bindings** (`input:`/`output:` values): **evaluated** to a typed value — refs,
  literals, arithmetic (`+ - * / % **`), list literals, comparisons, `:-`/`:?`,
  `|`, and pure builtins. A child-flow call is the whole-value `call(...)`
  directive, not a `${...}` span. A tag (`name: !T value`) declares the value's
  type (see "Typed slots"); `${T(...)}` builds or checks a typedef value inside
  an expression. A builtin called with arguments it doesn't take is a load error.
- **`when:` / `asserts:`**: **tested** as a boolean — `== != < <= > >=`,
  `in`/`not in`, `and`/`or`/`not`, parens, arithmetic operands. Bare, or one
  whole `${...}`, are equivalent; braces on a ref inside (`${a} > 5`) are a load
  error, as is calling a function other than a builtin, `isinstance`, or a `typedefs:`
  constructor `T(...)`, or a builtin with arguments it doesn't take (even where
  `and`/`or` would skip it). A flow-level
  `asserts:` reads `input.X` (boundary, before any node) or `node.output` (post,
  after the terminal). An exact unqualified flow-input name is shorthand (`age`
  means `input.age`) and wins over a same-named node; keep fields explicit as
  `input.profile.age`.
  `${system.X}` is **not** allowed there (it is ambient; a `system` ref is a
  LoadError — move the check into a body node).
- **Prompts**: free text with embedded `${...}` spans (each stringified). A span may
  call only a builtin; a builtin called with arguments it doesn't take is a load error.

`isinstance(x, T)` and constructors `T(...)` work in bindings, `when:`, node/flow
asserts, LOOP `while:`/`until:`, and flow outputs. They are load errors in prompts and
`questions:`, WAIT `until:`, MAP `over:`, and a `call`/`loop` node's own asserts.

> Bindings wire, conditions test, nodes compute.

## Inline `call(...)` — a flow call as a binding's whole value
`call(f, arg=${ref}, k=lit)` written as a binding's **whole value** (or a `case`
`then:`/`else:` target) desugars at load into an anonymous `call` node; the host
becomes `${<synth>.output}`. Every authored node `input:` mapping participates,
including HUMAN_INPUT context and LOOP carried-record seeds. Keyword args only, each at
most once; a bare arg is one plain scalar under the YAML value rules above, and a quoted
arg is text; nesting is allowed (`call(a, x=call(b, y=1))`). The synth node runs only when
the node it is written in (its host) can: it carries the host's `case` routes,
`depends_on:`, and `runs_after:` (a nested branch call's inner calls carry the branch), and
waits for the host's other inputs, skipping whenever a skipped input skips the host (a
`:-`-defaulted input only makes it wait). Calls in the same node don't wait on each other
unless one reads the other, and a nested call also carries its outer call's other
arguments (in `call(f, x=call(g), y=${a.output})`, `g` waits for `a` and is skipped with
it). A call in
the flow `output:`/`asserts:` has no host. An **unquoted** value whose trimmed text starts
with `call(` must be **exactly one** well-formed directive, else a load
error names the problem (unbalanced parens, trailing text after `)`, flow id not a bare name
/ missing comma, positional or repeated keyword, a keyword value that reads as two keywords
like `a=1 b=2`, a bare value YAML can't build like `d=2026-13-45`). A **quoted** value (`"call(f) tail"`, or a quoted arg `x="call(g)"`) is
text — the escape for literal `call(` text; a quoted call holding `${...}` is a load error.
A `call(` mid-value or inside `${...}` is literal text. A flow `asserts:` item follows the
rule quoted or not (an assert is an expression, never text). The old
`${flow(args)}` form (a flow call *inside* braces) is a **load error**: hoist an
embedded flow call to a named `call` node, and split a coalesce-of-calls
(`${a() | b()}`) into per-node outputs. Pure builtins (`${upper(x)}`) stay legal
inside `${}`.

Do not confuse this with an inline child body: `call(f, ...)` invokes an already
named child flow and synthesizes a CALL node; `kind: call` + `nodes:` defines a
one-off child's body on that node.

## Prompt names are local
Inside a `prompt:` or an AGENT `system_prompt:` use names from the node's own `input:`
block, written bare (`${name}`). An AGENT prompt also sees its own `variables:`. Pool references (`${input.x}`,
`${other.output}`) go in the `input:` block — bind there, then use the local name
in the prompt. The same rule covers a `human_input` `prompt:` and its question text
(`question`, option `label` and `description`); the loader checks all of them, so an
unknown name, a pool reference, an unknown function, a builtin called with arguments it
doesn't take, or a span that doesn't parse is a load error at that text.

## Effects from inside an AGENT node — `ask_user`
An AGENT node granted `tools: [ask_user]` can synchronously ask the human
**only if the model decides it needs to**. The tool blocks inside the same live
model/tool loop and its returned answer becomes the ordinary tool result. The model
may pass 2-4 short `options`, shown as choices; the human can still type an answer. For a
*guaranteed* gate use `human_input`; for "ask only if needed" use `ask_user`.
`ask_user` on an AGENT node returns its answer to the same live tool loop; HUMAN_INPUT and
WAIT return ordinary node outputs from their live calls. The host passes the handler
per run: `run_flow(..., tools={"ask_user": handler})`. `yian run` and `yian chat` do
this for you. Without a handler, a flow whose AGENT grants `ask_user` fails before the run
starts, and a `human_input` node fails when it asks. There is no authored
`entry:`/`controls:` field on an AGENT node or event-mode WAIT.

## Validate & run
- **Quick validate** (loads + compiles, no model): `yian run <flow>.yaml` will load
  first; or from Python `load_flow(text, search_paths=[flow_dir])`.
- **Run:** `yian run <flow>.yaml [--inputs file.json] [--input k=v]...`. The
  `--inputs` file must be UTF-8 text holding one JSON object; anything else fails
  with exit code `2`. JSON inputs load first; repeatable `--input` flags override individual keys but remain a
  string/scalar surface. With an interactive stdin (TTY), every declaration still
  missing is prompted. Keep four outcomes distinct: a value, `Set null`,
  omit/use-default, and cancellation. Nullable boolean and `Literal(...)` selects
  include `Set null`. Null-capable open scalar/list/object inputs first show
  omit/default when allowed, `Set null`, and `Enter value`; Enter opens the existing
  scalar or typed-JSON editor. Optional blank there returns to the palette rather
  than omitting, while a direct non-null-capable optional/defaulted editor still
  blanks to omission. The null-only `None` type has `Set null` and no Enter action.
  Treat text `null` for a string input or a `Literal` tag `null` as an ordinary
  string. Keep root JSON `null` invalid in the structured editor—choose `Set null`
  instead—while allowing nested null where the nested contract accepts it.
  Cancellation cancels the run.
  Without a TTY, missing optional/defaulted declarations stay absent. Missing
  required declarations fail with exit code `2` before provider or run-log setup;
  explicit null and native structures require `--inputs`, while ordinary scalars
  may use the string-valued `--input` or `--inputs`.
  `human_input` and `ask_user` are separate live-run interactions, unchanged by
  the initial-input omission policy. The host broker reads their answers from its
  input stream while the run remains live. WAIT does not prompt and also remains
  in the live run.
- A flow with only `code`/`tool` nodes runs with **no provider**; any `agent` node
  needs a provider/model configured (see `docs/installation.md`).

## Authoring checklist
- [ ] Every `input:` entry is tagged (`name: !T` or `name: !T default`); every type used is a scalar/form or a `typedefs:` entry, spelled with parentheses.
- [ ] Every data dependency uses `${...}` in `input:`; ordering-only dependencies use `depends_on:` or `runs_after:`. `input:` is always a mapping of names to bindings, never a whole `${node.output}`.
- [ ] Every `agent` `output:` is typed — bare `str` for free text, or a
  `Literal(...)`/other supported shape for structured generation.
- [ ] Every `prompt:` and `system_prompt:` references local inputs or, for an AGENT, its own declared variables.
- [ ] Every variable has a tagged default (`n: !int 4`) and a name distinct from local inputs.
- [ ] Every `long_term_memory:` declaration is on a full flow or `agent`, uses one of the three forms with only allowlisted built-in config keys (or a custom type a Python host builds), and puts each `session` on exactly one AGENT.
- [ ] Each `case` whose result is used has an `else:`, lists every tag of a non-nullable `Literal` `on:`, or is a non-nullable `bool` `on:` with `when: true` and `when: false`; its result is read with `${case.output}` and tagged where read; every branch names a node (a value entry for a fixed result) or calls a flow.
- [ ] `tool` ids are registered or passed per run with `tools=` (AGENT `tools:` may name a host-filled group); a `code` `module:function` is importable (or an inline `code: |` body `return`s a value); `call`/`uses:` targets resolve on the search path.
- [ ] Every CALL/MAP/LOOP has exactly one child form: named `call:`, or non-empty inline `nodes:` plus an `output:` key; inline parent/item dependencies cross through outer `input:`.
- [ ] A child-flow call is written `call(f, ...)` as a whole value, never `${flow(args)}`; string `:-` defaults and `:?` messages are quoted.
- [ ] Every typed slot uses a tag (`name: !T value`), never `name: T = value`; multi-argument types in a tag are named in `typedefs:`.
- [ ] No secret is written in the file: credentials come from flow inputs or the host.
- [ ] It loads cleanly (`yian run` / `load_flow`) before you wire a model.
