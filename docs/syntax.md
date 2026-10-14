# Flow syntax

A flow is a **function**: a typed `input:`, a graph of `nodes:`, and a typed
`output:`. `${...}` references infer data edges; `depends_on:` and `runs_after:`
(each a list of node ids, e.g. `depends_on: [fetch]`) add ordering-only edges. The human owns the structure; the LLM only fills the
AGENT node leaves. See [Agent](agent.md) for the conceptual model and terminology;
this page is the YAML reference.

## The flow shape

A flow file is Docker-Compose-shaped: metadata scalars at the top, then the
interface and body as top-level sections.

```yaml
id: momentum               # stable identifier
name: momentum             # display name
description: ...           # optional, one line
typedefs: { ... }          # optional — named/composed types
input:   { ... }           # parameters — typed
nodes:   { ... }           # body — a map keyed by node id
output:  ${x} | { ... }    # return — one value, or a record; bindings, optionally typed
asserts: [ ... ]           # optional — boolean invariants
```

There is **no `edges:` block**, no `__start__`/`__end__`, no per-node `id:`, and
no body wrappers — a node body is flat.

Flows, nodes, and `defs:` entries all spell their signature `input:` and `output:`.
The old plural `inputs:`/`outputs:` keys are a load error at any level.

The shape of `output:` sets what the flow returns:

- `output: ${x}` (one binding string) returns the value of `x` as is.
- `output: {name: ${x}, ...}` returns a record keyed by name. This holds for any
  number of fields: `output: {order: ${x}}` returns `{order: x}`, and `output: {}`
  returns `{}`. A caller reads a field by name, as in `${child.output.order}`.
- No `output:`, or `output: null`, returns `None`.

A record field name is a key, and every key must be text (see
[YAML values](#yaml-values)), so quote a `true`, `null`, or `1` field name
(`'true': ${x}`). A scalar YAML literal in place of a binding (`output: 5`, or a
field `x: 5`) is returned as written.

Changed: a non-string field name used to name its output by its Python text (`True`,
`None`, `1`), and `1:` beside `'1':` dropped one output silently.

Either form may declare its type with a [typed binding](#typed-bindings-name-type-value):
`output: Size = ${x}`, or a field such as `size: Size = ${x}` under `output:`.

Changed: a one-field record used to return its bare value, and `output: {}` used to
return `None`. For the old result, write `output: ${x}`, or drop `output:` to return
`None`. Flows whose `output:` is a record of zero or one field, and flows that call,
map, or loop over one, got a new fingerprint. Runs recorded under the old rule fail the fingerprint
check and can't be resumed or replayed.

Changed: a flow output that reads a node a `case` skipped used to fail the run with
`graph.terminal_unreachable`; it now returns `null` (see
[Skipped branches and flow outputs](#skipped-branches-and-flow-outputs)). Flows with such
an output got a new fingerprint, as did flows that call, map, or loop over one or over a
child whose output is written `${x :- null}`. Runs recorded before can't be resumed or
replayed. An explicit `${x :- null}` is now typed `Optional[...]`, which adds two load
errors: `on: ${x.output :- null}` over a `Literal` with a tag missing and no `else:` is
non-exhaustive, and a named-def LOOP seed written `${x :- null}` into a non-`Optional`
body field is rejected.

Node IDs are the keys under `nodes:`. A node id must be an identifier: ASCII
letters, digits, and `_`, not starting with a digit (`fetch_data`, not `fetch-data`
or `2x`), so that `${fetch_data.output}` can reference it. The expression keywords
`and`, `or`, `not`, `in`, `true`, `false`, `null`, and `none` can't be node ids
either. In compact mode the flow `id:`, and for a single-node `defs:` entry the def
name, also names the node and must be an identifier; a full-form flow id or a
multi-node def name may keep hyphens.

Changed: a node id that isn't an identifier, or is an expression keyword, used to load,
and failed only when a `${...}` referenced it, with an error that didn't name the node
(for `fetch-data`, split references `${fetch}` and `${data.output}`). It is now a load
error at the node that suggests a rename. To migrate, rename the node and every
`${...}`, `then:`, `else:`, `depends_on:`, and `runs_after:` that names it; the renamed
flow gets a new fingerprint, so its earlier runs can't be resumed or replayed. A compact
flow or single-node def whose id isn't an identifier never loaded; the error now names
the id. Other flows load as before, with the same fingerprint.

A mapping can't repeat a key: two nodes with one id, a field written twice, or a
section written twice is a load error at the second key, such as `duplicate key 'a'
(first at line 4)`. Keys a `<<:` merge brings in may still be overridden.

Changed: a repeated key used to load silently, and the last one won. Remove or rename
the earlier copy; the flow then runs as before.

### YAML values

yian reads an unquoted value with YAML 1.2's core rules, the rules editors such as
VS Code's YAML extension use, and keeps two older extensions: an unquoted date or
timestamp is a date, and a `<<:` merge key merges a mapping. Quoted and block (`|`,
`>`) values are always text.

| Unquoted | Reads as |
|---|---|
| `true`, `false` (also `True`, `TRUE`, ...) | boolean |
| `yes`, `no`, `on`, `off`, `y`, `n` (any case) | text |
| `null`, `~`, or nothing | null |
| `17`, `-3`, `017` | integer (`017` is 17) |
| `0o17`, `0x1F` | integer (octal 15, hex 31) |
| `1.5`, `.5`, `1e3` | float (`1e3` is 1000.0) |
| `1:30`, `1_000`, `0b101` | text |
| `2026-10-14`, `2026-10-14T10:00:00Z` | date, datetime |

The same rules apply to keys, to input and output defaults, to the literal after
` = ` in a typed binding, and to a bare `call(...)` argument. A `yes`/`no`/`on`/`off`
where a `bool` is expected fails with the fix, such as "`yes` is text; write `true`":
in a typed binding or default, a call input, a `case` value, `steerable:`, or
`llm_config` `inherit:`, including inside a list or record.

Every mapping key in a flow file must be text after YAML reads it: node ids, input
and output names, typedef names and fields, keys in a written-out record value, and
keys in `x-*` blocks alike, as in JSON. An unquoted `true`, `1`, `1.5`, `null`, or date
key is a load error at that key:

```
the key `True` was read by YAML as a boolean, but names in a flow must be text. Quote it ("True") or choose another name.
```

After quoting, the usual name rules apply: `"True"` is a valid node id, while `"1"`
still fails as a node id because it isn't an identifier. A `yes`, `no`, `on`, or `off`
key is already text (CASE's `on:` is the text `on`).

Changed: a non-text key used to fail differently depending on where it was (a
site-specific error in some places, a raw `TypeError` or a silently renamed name in
others); every such key now fails at the key with the error above. Keys that used to
load, such as `1:` in an `x-*` block or in a written-out record value, now fail too;
quote them.

Changed: yian used to read values with YAML 1.1's rules: an unquoted `yes`, `no`,
`on`, or `off` was a boolean, `1:30` was the number 90, `1_000` was 1000, and `017`
was octal 15, while `1e3` was text. Write `true`/`false` for a boolean, `90` or
`1000` for those numbers, `0o17` for octal, and quote a value you mean as text. A
`when: yes` on a `Literal[yes, no]` `on:` now matches the text `yes`; under a `bool`
`on:` it is a load error, so write `when: true`.
A CASE key spelled `On:` or `ON:` used to mean `on:`; write `on:`.

Changed: a bare `call(...)` argument used to read with the expression grammar:
`1e3`, `0x1F`, `0o17`, `+12`, `.5`, `1.`, `.inf`, `~`, an empty value, and dates
were text, while `None`/`none` and mixed-case `tRUE`/`nULL` were null or booleans.
They now read as the table says, as in a named `input:`, which also changes such a
flow's fingerprint. A bare date passed to any call input (`date` included) and a
bare number such as `1e3` passed to a `str` input now fail at load. Write `null`
for null and quote a value you mean as text (`x="1e3"`). Under a secret name,
`password=None` now fails; write `null`. A bare int too long for Python to convert
used to load and now fails at load, as in a named `input:`.

### Compact mode — when the flow *is* one node

The common case is "one flow, one node." Writing a full `nodes:` map plus a
redundant `output: ${greet.output}` wiring step for it is noise, so a flow whose
body is a single node can be written **inline**: drop the `nodes:` map and put the
node's `kind:` and its fields at the top level.

```yaml
# hello.yaml — the compact form
id: hello                  # names BOTH the flow and its single node
name: hello
input:
  name: str                # the node signature — auto-wired by name
output: str                # the node's output TYPE — re-exported as the flow output
kind: agent
prompt: |-
  Write a short, warm one-sentence greeting addressed to ${name}.
```

This desugars to the canonical one-node flow below before compile — same IR, same
behavior:

```yaml
id: hello
name: hello
input:
  name: str
nodes:
  hello:                   # keyed by the flow id
    kind: agent
    input:
      name: ${input.name}  # each flow input auto-wired by name
    output: str
    prompt: |-
      Write a short, warm one-sentence greeting addressed to ${name}.
output: ${hello.output}    # the single node's output, re-exported
```

Rules:

- The flow `input:` is the node's signature — each parameter is auto-wired into the
  node by name (`name` → `${input.name}`), so you refer to it bare in the prompt.
- The flow `output:` is the node's output **type**; the flow returns that node's
  output (no explicit `output: ${...}` line). A full flow declares its output type
  with a [typed binding](#typed-bindings-name-type-value) instead:
  `output: str = ${greet.output}`.
- Any other field (`prompt:`, `system_prompt:`, `tools:`, `skills:`, `llm_config:`,
  node-local `asserts:`, …) is the node body.
- Allowed only for the **value-producing leaf kinds** — `agent`, `code`, `tool`,
  `human_input`. `case`/`call`/`map` reference other nodes a one-node flow has none
  of, so they need the full `nodes:` form.

## References — naming a value

Everywhere you wire data you use a `${...}` reference:

| Write | Means |
|-------|-------|
| `${input.X}` | field `X` of the flow's input |
| `${node.output}` | node `node`'s whole value |
| `${node.output.field}` | dot into an object value |
| `${name}` / bare `name` | a local input in a node's `prompt:` (AGENT or `human_input`), a `human_input` question's text, an AGENT's `system_prompt:`, or a `case` `when:`; AGENT prompts also see their own declared `variables:` |

A whole-string `${ref}` resolves to the **typed value**; embedded in surrounding
text it is stringified.

Prompt templates are strict: a missing reference or any span whose final value is
null raises, even when the span is embedded in text. `:-`, `|`, and `:?` still see a
missing or null operand, so `${note :- "none"}` renders `none` for a null `note`; only
the fallback (the last `|` operand) must resolve. Binding templates retain their
non-strict behavior—a whole null remains a typed null, while an embedded null
contributes empty text. A flow output that reads a node a `case` skipped is null too
(see [Skipped branches and flow outputs](#skipped-branches-and-flow-outputs)).

When the loader knows a value's type, reading a field that type can't have is a load
error: a field on a scalar (`str`, `int`, `Literal[...]`, ...) or a list, or a field a
record doesn't declare. This covers `${input.X...}`, `${node.output...}`, and a MAP's
`${item...}` in bindings, `case` conditions, and flow outputs, and the carried fields a
LOOP `while:`/`until:` predicate reads. Reading a declared field
through an `Optional[...]` record still gives `null` when the whole value is null, and
values whose type isn't known at load (`object`, an untyped output, or anything below
them) aren't checked.

On a [union](#unions-of-records-union), a field every member has can be read anywhere. Its
type is the members' type when they agree, the merged `Literal` when every member's field
is a string `Literal` (a tag field, for one), and untyped otherwise; it is nullable when
any member's field is `Optional` or not required. A field only some members have is a load
error outside a `case` branch that narrows the value (see [`case`](#case-branching)):

```
reference ${d.output.qty} reads field 'qty', which only union members with action buy,
sell have; read it inside a case on ${d.output.action}
```

An untagged union says `which only 1 of 2 union members have; check the member with
isinstance(d.output, <Type>) in a case first`. A field no member has is an unknown field.
The rule also covers a node's `${output...}` asserts, flow asserts, MAP `${item...}`,
and flow outputs, which are never narrowed. To return a member field, read it in a
branch and return the `case` result with `${case.output}`; to use one per item, pass
`${item}` to the child and narrow there.

A LOOP `while:`/`until:` predicate can't sit in a `case` branch, so it narrows with a
top-level `and` part instead: an `isinstance(r, T)` part lets the parts after it read `T`'s
fields, as in a searched `case` `when:`. A check under `or` or `not`, or after the read,
doesn't narrow. Unnarrowed, the error points at that form:

```yaml
until: isinstance(r, Done) and r.answer == "ok"
```

```
reference ${r.answer} reads field 'answer', which only 1 of 2 union members have; check
the member first with `isinstance(r, <Type>) and ...`
```

!!! important "Prompt names are local"
    Inside a `prompt:` or an AGENT `system_prompt:` use names from the node's own
    `input:` block, written bare, like `${name}`. An AGENT prompt also sees its own
    `variables:`. Pool references
    (`${input.x}`, `${other.output}`) belong in the `input:` block, not the
    prompt. Bind it there, then refer to the local name in the prompt.
    The same rule covers a `human_input` `prompt:` and its question text
    (`question`, and each option's `label` and `description`). The loader checks
    all of them, so an unknown name, a pool reference, an unknown function, or a
    span that doesn't parse is a load error at that text.

Changed: a non-text `prompt:` used to crash the loader with a raw `TypeError` (AGENT, an
adaptive brief), load silently (AGENT `0`, `false`, `[]`, `{}`; `human_input` lists and
mappings), or load and fail when the gate ran (other `human_input` values). A non-text
question field was a raw validation dump at `questions:`. Now each is a load error at
the value; quote it.

### Operators inside `${...}`

A `${...}` is **one expression**. The same grammar works everywhere `${...}`
appears — bindings, conditions, prompts. Inside it you may write:

| Form | Meaning |
|------|---------|
| `${a + b}`, `${a * b + 1}` | arithmetic — `+ - * / % **`, unary minus |
| `${a == b}`, `${x in [1, 2]}` | comparisons — `== != < <= > >=`, `in` / `not in` |
| `${a and b}`, `${not a}` | boolean — `and` / `or` / `not` |
| `${[a, b, c]}` | a list literal (elements are themselves expressions) |
| `${upper(name)}`, `${join(items, ", ")}` | a pure builtin call (with dotted access on the result: `${fn(x).field}`); a null or wrong-typed argument fails, so default it: `${upper(name :- "")}` |
| `${X:-default}` | value, else `default` if absent |
| `${X:?"msg"}` | required — fail with the literal `msg` if absent (quote the message) |
| `${a \| b \| c}` | first present among peers (n-ary coalesce — for branch joins) |
| `$$` | a literal `$` (the scanner's universal escape — outside a span; in a prompt `$$` renders a single `$`) |

The operator words `and`, `or`, `not`, and `in` are reserved and cannot be the
root name of a reference. They may still name a dotted field, for example
`${record.output.in}`.

The `:-` default and `:?` message RHS are themselves expressions, so a **string**
default must be quoted: `${input.as_of:-"today"}`, not `:-today` (bare `today`
would read a *variable* named `today`).

`${...}` marks an expression inside a YAML string, so it never appears inside an
expression. Inside one, name values directly and group with parentheses: a default
chain is `${a :- (b :- "lit")}` or `${a | b | "lit"}`, and a default that reads
another value is `${input.as_of :- system.today}`. A `${...}` written inside an
expression is a load error that names the location and shows the rewrite:

```
`${...}` can't appear inside an expression; write `${input.as_of :- system.today}`
```

Changed: a `${...}` inside an expression used to load in some places. `${a :- ${b}}`
read `b` as a reference. A condition such as `when: "${a.output} > 5"` worked, but
`${a :- ${b :- "lit"}}` failed with a misleading error about an unknown input. All
of these are now load errors that show the rewrite (`a.output > 5`). A condition is
written bare, or as one whole `${...}` (see
[Expression contexts](#expression-contexts)).

- A searched `case` condition `"${a.output:-0} >= 1"` gave its gate an optional input,
  so the `case` still ran when `a` was skipped. The rewrite `(a.output :- 0) >= 1`
  does too (see the [searched form](#searched-form)). The default still covers a null
  value or an absent flow input.
- Each `on:` match now lowers to a condition on a bare `__on` (`__on == "<value>"`)
  instead of `${__on}`. So every flow with an `on:` `case` gets a new fingerprint,
  and resuming an old run log of one fails with a fingerprint error.
- A case value read in a bare condition or assert (`gate.output == "x"`) used to
  load, and could route wrong. It is now the same load error as the braced
  spelling.

Arithmetic on a value that is null when a `case` skips its node is a load error that
asks for a `:-` default, such as `${(order.output.qty :- 0) + 1}` (see
[Skipped branches and flow outputs](#skipped-branches-and-flow-outputs)).

Arithmetic is Python's. `+` concatenates text and lists (`${xs + [item]}`), `*` repeats
them (`${"ab" * 3}`), and `%` with text on the left formats (`${"%.2f" % price}`). A bool
counts as 0/1 (`${true + 1}` is `2`), but a CASE's `on:` never equates a bool with a
number. There are no cost limits. Float overflow in `+ - * /` gives `inf`; an operator
Python can't complete (`**` overflow, an int too large to mix with a float as in
`${10 ** 400 * 1.0}`, a repetition count too large, a missing `%(key)s`) fails like any
other expression error. See
[Expressions → Arithmetic follows Python](expressions.md#arithmetic-follows-python).

A whole-string `${expr}` resolves to the **typed value** (a number stays a
number, a list a list); embedded in surrounding text it is stringified. A typed value
may hold an int of any size, and the run records every digit. Rendering it as text
is limited: embedded in text or in a prompt, a value holding an int too long to
render (past 4300 digits by default) fails like any other expression error: a binding
fails with `binding.evaluate`, and a prompt fails its node. An int literal written in
the YAML has the same limit; a longer one is a load error.

Changed: an operator Python can't complete, or a value too long to render as text, in a
binding or an `asserts:` used to raise out of `run_flow`; now the run fails like any
other expression error. In a prompt or a CASE `when:` the node already failed; the
message now names the operator or value.

NaN, infinity, and complex numbers are not yian values. A binding whose value is
NaN, infinity, or complex at any depth, whether an expression result, a literal, or a
default, fails its node with `binding.evaluate`. The message names the input and the
path, like `input 'y' at a[1]: non-finite number inf is not a yian value`; a default
reads `default for 'x': ...`. A negative base to a fractional power is complex:
`v: ${(0 - 8) ** 0.5}` fails with `input 'v': complex number (…) is not a yian value`.
A temporary inside an expression is fine: the embedded template `"v=${x * 10}"` is
the text `v=inf`. Only a binding's final value is checked. See [Expressions → Non-finite results](expressions.md#non-finite-results).

A value nests at most 100 levels; each record and list is a level, and a record or
list built around a value is one level deeper. A binding (expression result, literal,
or default) or a MAP `over` list nested deeper fails its node with
`binding.evaluate`, naming the path (`input 'y' at [0].a...: value nests deeper than
100 levels, ...`): `${[input.x]}` around a 100-level `x` fails, and MAP items may
nest 99 levels. A run input may nest 100 levels; `run_flow` rejects a deeper one
before the run starts. A record flow output such as `{t: ${n.output}}` adds a level,
so wrapping a 100-level output fails `contract.output` at `__end__`. See
[Types → Nesting limit](typing.md#nesting-limit).

!!! note "Flow calls are not expressions"
    A `${...}` computes over values — it does **not** invoke a flow. To call a
    child flow inline, use the `call(...)` directive (see [`call` / `map`](#call-map-composition-over-child-flows)),
    not `${flow(args)}` (which is a load error).


## Types

The type vocabulary is Python typing. Scalars: `str`, `int`, `float`, `bool`,
`date`, `datetime`, `object`, `None`. Containers/forms: `list[X]`, `Optional[X]`,
`Literal[...]` (an enum of tags), named **records** (dataclass-style), and
`Union[A, B, ...]` of records.

Numeric domains may add inclusive bounds: `int[1, 12]` or `float[0.0, 1.0]`.
Both endpoints are required and the lower endpoint must not exceed the upper.
Integer bounds are exact, up to Python's digit limit (4300 digits by default).
Float bounds must be finite; integer endpoints used in a float domain must convert without rounding. Booleans and computed
endpoint expressions are invalid. Bounds work inside aliases, records, lists,
and `Optional[...]`.
Defaults and supplied values must satisfy them. Cross-flow wiring requires the
source domain to fit inside the destination domain; an unbounded source cannot
establish that guarantee for a bounded destination. Enums follow the same rule:
a `Literal[...]` destination accepts a source whose tags are a subset of its own,
not a plain `str`. To pass a wider source anyway, declare the destination type
with a [typed binding](#typed-bindings-name-type-value), which checks the value
at run time (`mode: Literal[a, b] = ${input.s}`), or route the call under a
`case` on `isinstance(input.s, Mode)`, where `Mode` names the `Literal[...]` in
`typedefs:`.

Name reusable types in `typedefs:`:

```yaml
typedefs:
  Ticker: str                          # alias
  Basket: list[Ticker]                 # aliases compose
  Decision: Literal[go, no_go, wait]   # enum — one of these tags
  Signal:                              # record — fields recurse
    score: float
    note: Optional[str]                # nullable field
```

`Literal` tags and record field names are non-empty text. `Literal["", a]`, an empty
field name (`"": int`), or a lone surrogate such as `"\uD800"` fails the load, the same
way wherever the type appears: in `typedefs:`, `input:`, or `output:`.

Every typedef is checked when the flow loads, even one nothing uses. A record can't
contain itself, directly or through other typedefs. A mistake fails at the typedef
field that holds it:

```
bad typedefs: typedef 'Signal' field 'tags': unknown type constructor 'lst' in 'lst[str]'
```

`Optional[X]` (nullable) and a `= default` (omission-fill) are orthogonal. A
required input has neither; `lookback: int = 30` defaults when omitted; and
`Optional[date]` with no default may be omitted, resolving to null. An explicit
`Optional[int] = null` is a present default, distinct from having no `=`. A null
default for a non-nullable type is invalid, as is an empty `=`.

Typed storage for `list[Optional[X]]` accepts actual null items because its
refined element type is nullable. When `X` has a specialized list storage kind,
those items remain `None` without lowering that kind to `list[Any]`. Cross-flow
bindings compare element contracts too: `list[X]` may fill
`list[Optional[X]]`, but the nullable source cannot fill the strict sink.

The default RHS is a YAML literal validated against the declared type while the
flow loads. Stored defaults are never string-coerced: write `int = 30`, not
`int = "30"`; a date string must likewise be quoted, such as
`date = "2026-01-01"`. Compatible payloads retain their authored representation,
so `float = 30` remains the integer `30`. String coercion applies only to values
supplied at a flow boundary: the inputs of the top-level flow, given on the command
line, in a JSON inputs file, or to `run_flow`. A `call`, `map`, or `loop` edge into a
child flow is not a flow boundary, so nothing is coerced across it (see
[`call` / `map`](#call-map-composition-over-child-flows)).

A supplied string coerces to `int` or `float` when it parses as one. For `bool`,
`1`, `true`, `yes`, `y`, and `on` become true and `0`, `false`, `no`, `n`, and
`off` become false, ignoring case and surrounding whitespace. Any other string
is left as text, so the declared type rejects it: `--input approved=ture` fails
with `input 'approved': 'ture' does not match declared type bool`.

### Unions of records — `Union[...]`

`Union[A, B, ...]` is a value that is one of several records:

```yaml
typedefs:
  Buy:
    action: Literal[buy]
    qty: int
  Sell:
    action: Literal[sell]
    qty: int
    price: float
  Hold:
    action: Literal[hold]
    note: str
  Decision: Union[Buy, Sell, Hold]
```

Every member is a record. Nested unions and aliases flatten, repeated members drop, and
member order doesn't matter. The loader must be able to tell the members apart from the
data alone, in one of two ways:

- **A tag field.** Every member declares the same field as a single-value `Literal`, with
  a different value in each. Any field name works (`action`, `kind`, `type`). A value is
  checked against the member its tag names; the tag must be present.
- **Shapes that can't overlap.** For example `{text: str}` and `{num: int}`. A value is
  checked against the one member it satisfies.

Otherwise the union fails to load, at its typedef even when nothing uses it:

```
Union[Buy, Sell]: members 'Buy' and 'Sell' can match the same value; add a tag field
every member declares as a single-value Literal, with a different value in each member,
e.g. kind: Literal[buy]
```

Other load errors:

| Write | Error |
|-------|-------|
| `Buy \| Sell` | `use Union[Buy, Sell] (\| is not a type operator)` |
| `Union[Buy, None]` | `None is not a union member ...; use Optional[Union[...]]` |
| `Union[Buy, Optional[Sell]]` | `member 'Optional[Sell]' is nullable; use Optional[Union[...]] instead` |
| `Union[Buy, str]` | `member 'str' is not a record` |

For a nullable union write `Optional[Decision]` or `Optional[Union[Buy, Sell]]`.

At run time a value with no tag fails with `missing tag field 'action' (expected one of
['buy', 'hold', 'sell'])`, and an unknown tag with `'wait' is not a tag of the union
(expected one of [...])`. An untagged union that matches no member fails with
`value matches no member of ...`, listing each member's reason. Nothing is coerced.

Wiring checks members: a union fills a union input when each of its members fits some
member of the input's union, and a record fills a union input when it fits a member. A
union fills an `object` input, but never a record input: narrow it in a `case` branch
first (see [`case`](#case-branching)), or declare the member with a
[typed binding](#typed-bindings-name-type-value) (`buy: Buy = ${decision.output}`),
which loads and is checked at runtime. Reading fields of a union is covered under
[References](#references-naming-a-value).

## Typed bindings — `name: Type = value`

Any binding may declare the type of its value. The form matches a flow-input
declaration: the type, ` = `, then the value.

```yaml
id: price
name: price
typedefs:
  Size: Literal[small, medium, large]
input:
  text: str
nodes:
  size:
    kind: code
    input:
      text: ${input.text}
    output: str                          # any str, as far as the loader knows
    code: |
      return "large" if len(inputs["text"]) > 20 else "small"
  quote:
    kind: code
    input:
      size: Size = ${size.output}        # narrower than str: checked at runtime
      discount: float = 0                # a YAML literal: checked at load
    output: float
    code: |
      base = {"small": 1.0, "medium": 2.0, "large": 3.0}[inputs["size"]]
      return base - inputs["discount"]
output:
  size: Size = ${size.output}            # a typed flow output field
  price: float = ${quote.output}
```

A single output value takes the same form: `output: Size = ${size.output}`.

### Where

- A node's `input:` bindings, including a LOOP's starting state and HUMAN_INPUT
  context bindings.
- A flow's `output:` fields, and a flow's single `output:` value. This gives a
  full flow a declared output type.
- Root flows, `defs:`, and inline CALL/MAP/LOOP bodies all accept them.

`call(...)` keyword arguments do not. A compact flow keeps `output: Type`, the
node's type.

### Quote text that contains ` = `

The YAML scalar style decides whether a value is typed:

- An unquoted value containing ` = ` is a typed binding, unless it starts with
  `call(`: then it is a
  [`call(...)` directive](#inline-call-a-flow-call-as-a-bindings-whole-value), so
  `call(f, a = 1)` needs no quotes. The text before the first ` = ` must be a type:
  a built-in, a typedef, or a composite such as `Optional[int]` or `list[Signal]`.
  Otherwise loading fails and suggests quoting:

  ```text
  `a` in `qty: a = b` is not a type. If you meant the text "a = b", quote it: `qty: "a = b"`
  ```

  A close miss adds a hint: `qty: imt = 3` ends with ``Did you mean `int`?``.
- A quoted (`'...'`, `"..."`) or block (`|`, `>`) value is never typed. Its `${...}`
  spans still evaluate. A quoted whole value starting with `call(` is text, not a
  call.
- So quote any value whose text contains ` = `, including an expression such as
  `'${input.y:-"a = b"}'`.
- `int=3` (no spaces) is plain text, unlike a flow-input declaration.
- An unquoted value ending in ` =` with a type on its left is an error: the value
  is empty.
- Use block-style mappings. A plain scalar inside `{...}` can't contain `${`, so
  `input: {qty: int = ${x}}` does not parse.

```yaml
input:
  note: "a = b"                     # the text: a = b
  fallback: '${input.y:-"a = b"}'   # an expression whose text contains " = "
  qty: int = ${size.output.qty}     # typed
```

### The value

The value after ` = ` is read in this order:

- A value starting with a quote, `[`, or `{` is read as YAML. A quoted string
  that contains `${` or starts with `call(` is then a template or call, without
  its quotes: `v: str = "${input.a} = 2"` binds `1 = 2`. (Here the quotes belong to
  the value after ` = `; a quoted *whole* binding value is text.) A list or record is a
  literal, and the strings inside it stay text, so `v: list[str] = ["${x}"]`
  binds the one-item list `["${x}"]`.
- Otherwise, a value containing a `${...}` template or expression, or starting
  with `call(`, stays as written.
- Anything else is a YAML literal, like an input default: `rounds: int = 0` binds
  the int `0`, `items: list[str] = []` an empty list, and `x: Optional[int] = null`
  a null.

Literals are checked against the type at load, read with the flow's
[YAML value rules](#yaml-values). So `str = 30` and `date = 2026-01-01` fail with a
hint to quote the value (`str = "30"`), and `bool = yes` fails with
"`yes` is text; write `true`".

The whole binding is one plain YAML scalar, so it can't contain `: ` or ` #`
anywhere, even inside quotes after ` = ` (`msg: str = "Status: ${x}"` does not
parse; drop the type and quote the whole value instead). A record literal needs
keys YAML reads as keys without `: `, so write `v: Rec = {"a":1}`, not `{a: 1}`.

### Load check

The loader compares the declared type with the type it infers for the value:

- **Disjoint** types fail at the binding, such as a `str` value declared `int`.
  Records are closed, so a record with a required field the declared record lacks
  is disjoint.
- A **narrower** declared type loads and is checked at runtime: non-Optional over
  Optional, a `Literal` or bounded subset, a record over `object`, or a union member
  over the union (explicit narrowing, e.g. `buy: Buy = ${decision.output}`).
  A flow output over a node a `case` may skip is `Optional`, so
  `order: str = ${order.output}` loads and fails with `binding.type` when `order`
  is skipped.
- Unions compare member by member: a member declared as its union fits, and a
  record that matches no member is disjoint. Inside a narrowing `case` branch the
  value has the branch's members, so `buy: Buy = ${decision.output}` in the `sell`
  branch fails at load.
- A MAP's lone `${item}` or `${item.field}` is inferred from the element type of
  `over:`, so `t: str = ${item}` over a `list[int]` fails at load.
- A value whose type can't be inferred (arithmetic, a builtin call, an untyped
  producer) is checked at runtime only.

### Runtime check

Before the consuming node runs, the value is validated against the declared type.
Nothing is converted: an int bound as `float` stays an int, and the string `"30"`
is not an int. A failure fails the consuming node (END for a flow output) with the
failure code `binding.type`, and the diagnostic points at the binding.

A flow's committed output is stored like any typed node output: an int in a
`float` slot is widened to a float at any depth, whether the slot is the whole
value, a `list[float]` item, or a typed record field. An untyped field of a record
output keeps its value as returned.

### Downstream

Consumers see the declared type, not the inferred one. This is how an opaque
value becomes typed.

- A CALL, MAP, or LOOP parent sees a child's declared output type.
- A CALL checks a typed argument's declared type against the child's parameter.
- An inline body's parameter takes the declared type, so a value the loader can't
  type (arithmetic, an empty list) works without a named `defs:` body.
- A typed LOOP seed checks the starting value. A field the body omits keeps the
  seed's declared type; a field the body returns follows the body's output type.
- An untyped nullable source into a strict child parameter is still a load error,
  including a child output that is `Optional` because a `case` may skip its node.
  A typed binding (`x: int = ${maybe.output}`) accepts the narrowing and checks it
  at runtime.

Declared types are part of the flow fingerprint, so a run can't resume after a
type changes. Recorded flow outputs are re-validated on resume.

## Per-node configuration — `env:`

Any node may carry an `env:` block — a mapping of **static** config keys the node's
own implementation reads. It is the generic knob for "tune this node's behavior"
that isn't data flow (`input:`) or model selection (`llm_config:`). Keys are plain
strings; each key's meaning and type is the consuming node's concern (the engine
never interprets `env`; compose stores the merged mapping on the node for its own
implementation to read).

A flow can set a default `env:` for **every** node under it; each node can override
individual keys. On build the two merge per key, **node wins**:

```yaml
env:                            # flow layer — default for every node
  max_tool_iterations: 150
nodes:
  researcher:
    kind: agent
    prompt: Investigate ${topic}.
    tools: [search]
    env:
      max_tool_iterations: 300  # this node overrides the flow default
```

The merge is **static, at build time** (`{**flow_env, **node_env}`) — there is no
runtime cascade and no per-field resolution across parent flows. It is also
**flow-local**: a called child flow does *not* inherit its parent's flow `env:` —
it has its own. A multi-node `defs:` entry sets its own `env:` at the same level as
its `nodes:`. Values must be literals (no `${...}` refs). A key with a secret name
can't hold a literal value (see
[Secrets stay out of flow files](#secrets-stay-out-of-flow-files)).

Keys in use today:

| Key | Node kind | Meaning |
|-----|-----------|---------|
| `max_tool_iterations` | `agent` | Model turns the tool-calling loop runs before it gives up with an `AgentLoopError`. If the reply on the last allowed turn asks for tools, the node fails without running them, since no later model call could read their results. Default `-1` (**no cap** — run until a final answer). Set a positive int to bound it; any other value (`0`, other negatives, a non-int) fails at load, on the flow `env:` or the AGENT node's `env:`. |

## Secrets stay out of flow files

Every value written in a flow file is part of the flow's fingerprint (see
[Durable recovery](api.md#durable-recovery-with-resume_flow)), so a credential
doesn't belong there. Declare it as a flow input and pass the value when the flow
runs, or have the host supply it.

**Secret names.** `password`, `passwd`, `secret`, `client_secret`, `private_key`,
`access_token`, `refresh_token`, `api_key`, and any name ending in `_api_key`,
`_password`, or `_secret`. Matching ignores case, camelCase, `-` versus `_`, and
leading or trailing `_`, so `clientSecret`, `Client-Secret`, `DB_PASSWORD`, and
`OPENAI_API_KEY` match. `.` and spaces are not separators. Broader names such as
`key`, `user`, `auth`, `token`, `sig`, `signature`, `page_token`, `secret_key`, and
`apikey` are not on the list.

**The rule.** Under a secret name the value must be a `${...}` reference or `null`.
Anything else fails at load: text (even `''`), a number, a boolean, a list, a
mapping, a `$${...}` escape (it is literal text), and malformed template text. Text
around a reference is fine (`'Bearer ${input.token}'`). In a binding position, a
whole-value [`call(...)` directive](#inline-call-a-flow-call-as-a-bindings-whole-value)
counts as a binding, and its own keyword arguments follow the same rule.

The rule applies to:

- node `input:` and other binding fields, such as MAP `over:`, including the value
  of a typed binding (`password: str = ${input.password}`);
- flow `output:`;
- `case` `then:` and `else:` values (a value branch has no name, so only secret keys
  nested in it are checked);
- `call(...)` keyword arguments;
- declaration defaults (flow `input:`, `defs:` inputs) and AGENT `variables:`;
- `env:` on a node, a flow, or a `defs:` entry.

A secret key nested in a literal mapping or list under any other name is checked
too, and the error gives its path, such as `conn.items[0].password`. In `env:`,
defaults, and nested literal text, `${...}` is not evaluated: it stays text. It
holds no secret, so it passes, but it doesn't bring a value in either; a value
comes in through a node `input:` binding.

The error names the slot and the fix, and never repeats the value. This flow fails
at load:

```yaml
nodes:
  login:
    kind: code
    input:
      user: admin
      password: hunter2
    output: str
    code: |
      return inputs["user"]
```

```
node 'login': input 'password' is a secret name, so its value can't be written in the flow file; pass it as an input (${input.password}) or from the host
```

Declare the input and pass the value when the flow runs:

```yaml
input:
  password: str
nodes:
  login:
    kind: code
    input:
      user: admin
      password: ${input.password}
    output: str
    code: |
      return inputs["user"]
```

A default under a secret name fails with `input 'password': a secret name can't have
a default value in the flow file; drop the default and pass the value when the flow
runs`. The error for an `env:` entry points the value to a node's `input:`. Model keys
have their own rule: a flow file can't set `llm_config.api_key` at all (see
[`llm_config`](#llm_config-the-model-selection-cascade)). Memory config keeps its own,
broader key rule: it rejects credential keys at any depth, whatever the value (see
[Agent memory](#agent-memory-long_term_memory)).

Changed: a literal value under a secret name used to load, and it was left out of the
fingerprint. It now fails at load with the error above. Move the value to a flow input
or have the host supply it. The fingerprint also used to leave out or mask values it
guessed were credentials; it now covers every value written in the flow. Flows the
old guess touched get a new fingerprint, so their earlier runs can't be resumed or
replayed: a flow with any input, wiring key, or `env:` key named like a credential
(`token`, `signature`, `auth`, `secret`, and similar), even with no literal value, or
such a key nested in a literal mapping; any `${...}` under such a name; and any literal
or mapping key holding URL userinfo (`https://user:pass@host`), credential-like query
values (`key=`, `user=`, `sig=`, `*_token=`), or `host=… user=…`, `;`-separated, or
JDBC connection text. Other flows keep their fingerprints.

## Agent memory — `long_term_memory:`

`long_term_memory:` names one or more memories on a full flow (the ordinary
`nodes:` form, root file or `defs:` entry) or on an AGENT node. Each memory has
a **name** and a **type**. YAML holds public behavior only; the host builds the
backend and decides where it is stored.

```yaml
id: support
name: support
long_term_memory:
  notes: episodic
  lessons:
    type: reflective
    config: {max_items: 100, recall_limit: 12}
input: {question: str}
nodes:
  answer:
    kind: agent
    long_term_memory: {chat: session}
    input: {question: "${input.question}"}
    output: str
    prompt: Answer ${question}.
output: ${answer.output}
```

Three forms are accepted:

| Form | Meaning |
|---|---|
| `long_term_memory: episodic` | Sugar for `{episodic: episodic}`: one memory named after its type. |
| `long_term_memory: {notes: episodic, chat: session}` | One memory per entry, `name: type`. |
| `long_term_memory: {notes: {type: episodic, config: {...}}}` | Long form with literal `config`. Only `type` and `config` are allowed. |

`long_term_memory: null` is the same as leaving the key out. `{}` is an error.
The retired `{name: X, config: ...}` form is also an error; `name`, `type`,
and `config` are reserved and cannot be memory names. Names and type names match
`[A-Za-z_][A-Za-z0-9_-]*`.

Built-in types, the config keys they accept, and the values allowed. Each value
is an integer; a wrong key or value is a load error.

| Type | Config keys | Values |
|---|---|---|
| `episodic` | `max_episodes`, `recall_limit` | at least 1, or `null` for the stored or default limit |
| `reflective` | `max_items`, `recall_limit` | at least 1, or `null` for the stored or default limit |
| `session` | `history_budget_chars`, `recent_turns` | `history_budget_chars` at least 1, `recent_turns` at least 0 |
| `profile` | `max_chars`, `input_budget_chars` | at least 1 |

Any other type name is a custom type that the Python host builds (see
[building memories](api.md#building-memories-with-build_memories)). In any
type, `storage`, `store`, and `path` are rejected as config keys, and
`api_key` is rejected anywhere: storage locations and credentials belong to
the host.

**Memory keys.** A memory is identified by its key, the name prefixed by the
`uses:` aliases it was imported through: `notes` in the root file, `lib.notes`
in a file imported as `lib`, `b.d.notes` through two imports. Declaring the
same key in two places of one file (for example on the flow and on an AGENT)
is the same memory. The type and config must then agree. Two libraries that
both declare `notes` stay independent (`liba.notes`, `libb.notes`), and so does
one file imported under two aliases. Keys that differ only by case are an
error, as is an alias that does not match the name pattern when it scopes a
memory.

**Stacking.** An AGENT recalls, in order: each enclosing flow's memories,
outermost first, through CALL, MAP, and LOOP; then its own. A repeated key
appears once, at its outermost position. An AGENT inside an imported child
recalls the importing flow's memories by position, but a child's declaration
never binds to a parent's memory. `long_term_memory:` is invalid on other node
kinds and on anonymous inline CALL/MAP/LOOP bodies; use a named `defs:` flow
when the child needs flow-level memory. The key itself is rejected there, even
as `long_term_memory: null`; `null` means absent only where the key is allowed.

**Session.** A `session` memory must be declared on an AGENT, never on a flow.
Each session key has exactly one declaring AGENT, and an AGENT declares at most
one session. Every call site of that AGENT shares one conversation; repeated
calls, LOOP iterations, and MAP items take turns in admission order. History
updates automatically during forward, without enabling learning.

Each declaration's form and config are checked for every def and import. The
rules that compare declarations (one type and config per key, session
placement, case collisions, alias names) apply only to declarations reachable
from the root.

**Supplying memory.** `yian run` builds every built-in memory in RAM, or
on disk with `--memory-root DIR` / `--memory-storage KEY=PATH` (see
[CLI memory](cli.md#memory-and-storage)). It does not build custom types; a
flow that declares one runs from Python. From Python, pass
`memory=build_memories(flow, ...)` to `run_flow`; `memory=None` leaves
declarations dormant. `learnable=True` independently retains accepted execution
for an explicit `result.backward(feedback)`. Every memory key then gets one
learn call. The CLI does not run backward. See
[the Python API](api.md#explicit-backward-feedback).

## Node kinds

Every node has a `kind:`. The set is closed — you compose flows from these, you
do not define new kinds.

### `agent` — the LLM-backed AGENT node

```yaml
classify:
  kind: agent
  input:
    text: ${input.text}        # bindings — these infer the data edges
  output: Literal[positive, neutral, negative]
  prompt: |-
    Classify the sentiment of: ${text}
    Answer with exactly one of: positive, neutral, negative.
```

`prompt:` must be text. A value YAML reads as a number, boolean, date, list, or
mapping is a load error at that value; quote it:

```
node 'classify' (kind=agent): `prompt:` must be text, but YAML read it as an integer; put it in quotes
```

A `null` `prompt:` still counts as an absent `prompt:`. This used to load or crash differently; see
the note under [References](#references-naming-a-value).

An AGENT with `tools:` or `skills:` runs the tool loop; without them it makes one
model call (text, or structured for a typed `output:`).

`mode:` was removed. A flow that still sets it fails to load with a message that
says to delete the line; with `mode: plain` and `tools:`, those tools now run.

An AGENT node's `output:` may be any declared shape a model can generate, which
excludes `file` and `None` anywhere in the type (a flow that declares one fails at
load; produce such a value in a CODE node). Only a bare, untagged `str`
keeps the node a **text producer**. A `Literal[...]` enum, record, union, non-string
scalar, or list switches the node to **structured generation**: the engine derives a
schema from the declared `output:` and asks the model to emit a conforming value
(via the provider's native structured output, or a JSON prompt-injection fallback
for providers that lack it). A root `Literal[...]` is constrained internally as a
single `value` field and returned to the flow as the bare tag. `Optional[X]` allows
null in the schema wherever it appears: at the root, as a list element, or as a
record field. A root `Optional[str]` stays a text producer, so it never returns
null. The result is
validated while its canonical success record is prepared, like every other node
output.

A [union](#unions-of-records-union) `output:` asks for one of the members: each member's
schema forbids unknown fields and pins its tag with a `const`. The node returns the
matching member's record. A bad reply's retry error points at a field of the member the
tag selects, at the tag field when the tag is missing or unknown, or at the union itself
when an untagged union matches no member.

```yaml
extract:
  kind: agent
  input:
    text: ${input.text}
  output:                       # a record shape -> structured generation
    name: str
    score: int
  prompt: |-
    Extract the person's name and a 0-10 score from: ${text}
```

If the model deviates from the schema, the engine feeds the error back and retries
up to `retries:` times (default 2). A NaN or infinity in the reply (`{"score": NaN}`),
or a value nested deeper than 100 levels, is a deviation too. `retries:` must be a
non-negative integer (`0` means no retry); anything else fails at load:

```yaml
extract:
  kind: agent
  retries: 3                    # extra self-correction attempts (default 2)
  output: {name: str, score: int}
  prompt: ...
```

A node can pin its own provider/model; otherwise the process defaults snapshotted
when the run initializes apply (see [Installation](installation.md)).

#### `llm_config` — the model-selection cascade

Model selection cascades **per field, most-specific wins**. Each field (provider,
model, a field of a [provider block](#provider-blocks-request-settings), …) is
resolved independently: an AGENT node fills only the fields it leaves unset from the
layer outside it. Precedence, most specific first:

1. the AGENT node's own `llm_config:`
2. the enclosing flow's `llm_config:`, then each parent flow outward
3. the CLI `--provider` / `--model` flags (`yian run … --provider anthropic`)
4. process provider/model defaults, read exactly once when the run initializes

Process defaults use environment overrides, then saved authentication choices,
then the provider's built-in fallback. OpenAI is the default provider. An unset
model uses the selected provider's default; it never inherits another provider's
saved or environment model. An explicit model in an authored layer still follows
the per-field precedence above.

A flow can set defaults for every AGENT node under it with a top-level `llm_config:`:

```yaml
llm_config:                     # flow layer — every AGENT node inherits these
  provider: anthropic
  anthropic:
    max_tokens: 8000
nodes:
  drafter:
    kind: agent
    prompt: Draft a summary.
    llm_config:
      model: claude-opus-4-8    # fills a field the flow leaves unset
      anthropic:
        output_config: {effort: high}   # joins the flow's max_tokens
```

Every `llm_config:` in a flow file (node, `defs:` entry, or top level) is checked
when the flow loads: an unknown key such as `modle:`, a bad value such as
`openai: {temperature: hot}`, a non-finite number such as
`openai: {temperature: .nan}`, or an unknown provider fails at that `llm_config:`.
`inherit:` is valid only in a node's `llm_config:`. A host
`run_flow(..., llm_config=...)` layer with an unknown key fails before any node runs.

A flow file cannot set `api_key` at any level, nor `base_url` except beside
`provider: azure`; loading such a flow fails. A library host passes the provider
key, and optionally the endpoint it goes to, in
`run_flow(..., llm_config={..., "api_key": key, "base_url": url})` instead (see
[the Python API](api.md)). The key and its `base_url` go to every AGENT of that
provider, including one with `inherit: false`.

For `provider: azure`, `model` is the deployment name and `base_url` the resource
endpoint, such as `https://<resource>.openai.azure.com/`. Both are ordinary
`llm_config:` fields there, so a flow can set them at any level; the key stays in
`AZURE_OPENAI_API_KEY`. An Azure agent whose resolved config has no `base_url`
fails before the run's first node.

The CLI flags are the *outermost authored* layer — they fill gaps, they do **not**
override an AGENT node or flow that set the field. The run resolves the complete
static tree of AGENT nodes up front, so later environment changes cannot alter a model selection.
To ignore the authored outer layers, set `inherit: false` in a node's
`llm_config:`. The node then uses its own dict plus the snapshotted process
provider/model defaults; it ignores enclosing flow, parent-flow, and CLI layers:

```yaml
  grader:
    kind: agent
    prompt: Grade it.
    llm_config:
      provider: openai
      model: gpt-5.5
      inherit: false            # own config + process defaults; no flow/CLI layers
```

`inherit:` must be `true` or `false`; a quoted `"false"` or `null` fails at load.

#### Provider blocks — request settings

The top level of `llm_config:` selects the model: `provider`, `model`, `inherit:`
(node only), and the host-only `api_key` and `base_url` (a flow sets `base_url` only
beside `provider: azure`). Everything else about the request goes in a block named
after a provider: `openai`, `azure`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`,
`glm-cn`, `minimax`, `minimax-cn`, `openrouter`, `vllm`, `anthropic`, `google`, or
`ollama`. A node uses the block of its resolved provider.
The other blocks are inert, but they are still checked at load, so one flow can carry
settings for several providers and each node takes the block of the provider it
resolves to:

```yaml
llm_config:
  provider: openai
  model: gpt-5.5
  openai:
    reasoning: {effort: high}
    max_output_tokens: 4096
  anthropic:                    # used only when a layer selects provider: anthropic
    max_tokens: 16000
    thinking: {type: adaptive}
    output_config: {effort: high}
```

A block mirrors the request body of the API yian calls for that provider, with the
same names, nesting, and values, and yian sends it as written. There is no
translation layer: copy a field from the provider's API reference and it means what
the reference says. The load check follows that reference too. An unknown key, a wrong
type, a value outside the documented enum or bounds, or a non-finite number fails at
that `llm_config:`, naming the block, the key path, and what is allowed. The error
never shows the value.

| Provider | API yian calls | Reference |
| --- | --- | --- |
| `openai` | OpenAI Responses (`POST /v1/responses`) | [platform.openai.com](https://platform.openai.com/docs/api-reference/responses/create) |
| `azure` | Azure OpenAI Chat Completions | [learn.microsoft.com](https://learn.microsoft.com/azure/ai-foundry/openai/latest) |
| `xai` | xAI Chat Completions | [docs.x.ai](https://docs.x.ai/developers/rest-api-reference/inference/chat-completions) |
| `deepseek` | DeepSeek Chat Completion | [api-docs.deepseek.com](https://api-docs.deepseek.com/api/create-chat-completion) |
| `qwen`, `qwen-cn` | Alibaba Cloud Model Studio OpenAI-compatible Chat Completions | [alibabacloud.com](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions) |
| `glm`, `glm-cn` | Z.AI Chat Completion | [docs.z.ai](https://docs.z.ai/api-reference/llm/chat-completion) |
| `minimax`, `minimax-cn` | MiniMax OpenAI-compatible Chat Completions | [platform.minimax.io](https://platform.minimax.io/docs/api-reference/text-openai-api) |
| `openrouter` | OpenRouter Chat Completions | [openrouter.ai](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) |
| `vllm` | a vLLM server's Chat Completions | [docs.vllm.ai](https://docs.vllm.ai/en/latest/serving/online_serving/) |
| `anthropic` | Anthropic Messages (`POST /v1/messages`) | [platform.claude.com](https://platform.claude.com/docs/en/api/messages) |
| `google` | Gemini `generateContent` | [ai.google.dev](https://ai.google.dev/api/generate-content) |
| `ollama` | Ollama `POST /api/chat` | [github.com/ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion) |

A few providers need a note:

- `openai` on a ChatGPT subscription (`yian auth openai`, no host `api_key`) reaches
  the subscription endpoint, which accepts the Responses body without `temperature`
  and `max_output_tokens`. The block is checked against that narrower schema when the
  run starts, and a block that sets either fails before the first node, naming the
  agent.
- `google` uses the REST names and enum spellings: camelCase keys and upper-case
  levels, such as `google: {generationConfig: {thinkingConfig: {thinkingLevel: HIGH}}}`.
- `anthropic` with `thinking` on (any `type` other than `disabled`) gets structured
  output through the JSON-schema output format (`output_config.format`), because the
  Messages API refuses thinking beside a forced tool call.
- `ollama` takes what the Ollama client sends: `think` (`true`, `false`, or a level),
  `options` (`temperature`, `num_ctx`, …), `keep_alive`, `logprobs`, and
  `top_logprobs`. It has no default `think`, so the model's own default applies.

An omitted field is not sent, so the provider's default applies. The one exception is
Anthropic `max_tokens`, which the Messages API requires; the client library fills it in
when the block leaves it unset.

Some request fields are built by yian, so a block cannot set them: the model, the
messages (`messages`, `input`, `contents`), the system prompt (`system`,
`instructions`, `systemInstruction`), tools and tool choice (`tools`, `tool_choice`,
`toolConfig`, `parallel_tool_calls`), the output format and structured output
(`response_format`, `text.format`, `output_config.format`, `responseMimeType`,
`responseSchema`, Ollama `format`), streaming (`stream`, `stream_options`),
conversation state (`previous_response_id`, `conversation`, `store` on Responses),
and the number of replies (`n`, `candidateCount`). Setting one is a load error that
says the field is built by yian and why.

The block check covers what the API reference documents for every model. A limit of
one model is left to the provider: sent as written means the provider's or the client
library's error replaces a silent drop. For example, `anthropic: {temperature: 0.5}`
passes the load check, but newer Claude models refuse `temperature`, so the node fails
with the provider's error.

Blocks follow the cascade one level deep: a node's block and the flow's block of the
same name merge per block field, the node's fields winning. A value inside a field is
replaced whole, so a node's `thinking: {type: disabled}` replaces the flow's whole
`thinking` mapping. `null` means unset at every level.

The provider-specific keys yian used to translate are gone. Each one is a load error
that shows its block form:

| Removed key | Write instead |
| --- | --- |
| `anthropic_effort: high` | `anthropic: {output_config: {effort: high}}` |
| `openai_reasoning_effort: high` | `openai: {reasoning: {effort: high}}` |
| `google_thinking_level: high` | `google: {generationConfig: {thinkingConfig: {thinkingLevel: HIGH}}}` |
| `temperature: 0.2` | `<provider>: {temperature: 0.2}`; `google: {generationConfig: {temperature: 0.2}}`; `ollama: {options: {temperature: 0.2}}` |

#### OpenAI authentication

Run `yian auth openai` to select ChatGPT subscription sign-in or an OpenAI API key,
then choose the default model. Both use `provider: openai`, one model call or
the tool loop, explicit Yian tool grants, typed output, SessionMemory, and
backward learning.

The retired `provider: codex`, `sandbox`, `cwd`, and `builtin_tools` settings
must be removed when migrating. Native thread references cannot be resumed as
message histories; use fresh SessionMemory. Yian tools execute on the host.
See [authentication setup](installation.md#openai-authentication).

#### Agent-owned variables

Declare local prompt settings in `variables:`. Every entry needs a type and an
explicit default:

```yaml
id: summarize
name: summarize
input:
  topic: str
nodes:
  writer:
    kind: agent
    variables:
      num_sentences: int[1, 12] = 4
      temperature: float[0.0, 1.0] = 0.4
      tone: Literal[concise, detailed, technical] = concise
      instruction: str = "Summarize clearly"
    input:
      topic: ${input.topic}
    prompt: ${instruction}. Write ${num_sentences} ${tone} sentences about ${topic}.
    output: str
output: ${writer.output}
```

Variable names must be unique and cannot overlap the node's input names. Defaults
are validated while loading, including bounds and enum choices. Nullable variables
still require an explicit default, such as `Optional[str] = null`; a quoted empty
string is a valid `str` default. Default strings are literal text, so `${...}`
inside a default is not evaluated as a binding. A NaN or infinity default, or one
nested deeper than 100 levels, fails the node when it runs.

Each invocation starts with a fresh copy of these defaults. Repeated `${name}`
references read the same local value. A variable belongs to its authored AGENT
path and name: MAP items and LOOP iterations of that node share its declaration,
while distinct authored callsites have distinct owners. Variables introduce no
data dependencies or flow outputs. A variable named `temperature` is prompt data;
configure model temperature in the provider block of `llm_config:`.

#### System prompt — `system_prompt:`

Every AGENT model request starts with a system message. By default it is a fixed
framework instruction. `system_prompt:` replaces it with your own:

```yaml
id: chat
name: chat
input:
  name: str
  title: str
  message: str
nodes:
  reply:
    kind: agent
    input:
      name: ${input.name}
      title: ${input.title}
      message: ${input.message}
    system_prompt: "You are ${name}, ${title}. Speak as ${name} in the first person."
    prompt: ${message}
    output: str
output: ${reply.output}
```

- It **replaces** the framework instruction; it does not add to it. Omitting it
  keeps the framework instruction exactly as before.
- It is a template with the same rules as `prompt:`: it reads the node's own
  `input:` names and `variables:`, written bare. An unknown name or a pool
  reference such as `${input.name}` is a load error at the `system_prompt:` line.
  A null value at run time fails the node before any model call.
- The value must be a non-empty string.
- Granted [`skills:`](#agent-skills-skills) are listed after it, then recalled
  [`long_term_memory:`](#agent-memory-long_term_memory) sections. Memory recall
  receives the rendered system prompt without the skill listing.
- Every model request of the node uses it: one model call, each tool-loop
  round, and structured generation.
- It is part of the flow's identity. Changing it changes the flow fingerprint, so a
  run log cannot be resumed with a different system prompt. With `learnable=True`,
  the rendered text is recorded with the node's evidence, so backward learning
  sees what the model saw.
- A compact single-node flow accepts `system_prompt:` at the top level.
- There is no flow-level default; set it on each AGENT node that needs one.

#### Agent skills — `skills:`

A skill is a folder with a `SKILL.md` file, in the format Claude Code and Codex
share: YAML frontmatter with `name` and `description`, then Markdown instructions.
Other files in the folder (`references/`, `scripts/`, ...) come with the skill.
Grant skills to an AGENT by name:

```yaml
reviewer:
  kind: agent
  skills: [code-review, release-notes]
  input:
    diff: ${input.diff}
  prompt: Review this change. ${diff}
  output: str
```

The model sees only each granted skill's name and description, in its system
message. It reads a skill when it needs it by calling the built-in tool
`load_skill(name)`, which returns the `SKILL.md` instructions, or
`load_skill(name, path)`, which returns a file inside the skill folder, such as
`references/checklist.md`. Paths outside the folder and hidden files (such as
`.env`) are rejected, and files must be UTF-8 text of at most 256 KiB.

- `skills: [name, ...]` grants exactly those skills. Names use lowercase letters,
  digits, and single hyphens, up to 64 characters.
- `skills: all` grants every skill the host supplies for the run. Use it in a
  shared flow when each run should see a different set.
- Omitting `skills:` grants none; the system message is unchanged.

YAML only names skills. The host decides which skills exist for a run: `yian run`
reads your Claude Code and Codex skill folders (see [CLI](cli.md#skills)), and
Python hosts pass `run_flow(..., skills=load_skills([...]))` (see
[API](api.md#skills)). A flow that grants skills needs a host `skills=` mapping, and
each listed name must be in it; otherwise the run fails before it starts. An AGENT
with `skills:` runs the tool loop. `load_skill` is a reserved name: never list it in
`tools:`.

#### Steering a running agent — `steerable:`

`steerable: true` lets the host steer or interrupt a running AGENT, like
pressing Esc or typing a follow-up in a coding-agent chat:

```yaml
worker:
  kind: agent
  steerable: true               # default false
  tools: [search]
  prompt: Research the question and answer it.
  output: str                   # required: `str` or no `output:`
```

- **Steer.** At the top of each loop iteration the node reads the messages the
  host queued for it and sends each one to the model as a user message with the
  next request. Steers never land between the results of one reply's tool calls.
  A steer that arrives during the final model call is not read; it stays with the
  host for the next input.
- **Interrupt.** The node checks for an interrupt at the top of each iteration,
  as each streamed chunk after the first arrives, and before each tool call. It
  then stops and **succeeds** with the text so far, so the flow or loop
  continues. A reply that already arrived is kept whole, including a streamed
  reply whose last chunk had arrived. Otherwise a streamed reply is cut at the
  last chunk shown. Tool calls it did not run get an error result ("Not run:
  interrupted by the user."). Without streaming, an interrupt during a model
  call takes effect when that call returns. An interrupt for a node that has
  already finished is ignored.
- Session memory records steers with the model request that carried them, and
  closes an interrupted turn as `interrupted`.
- An AGENT without `tools:` or `skills:` makes one model call. Steers queued
  before it go with that request, and an interrupt cuts its streaming text reply
  at the last chunk shown.

`steerable: true` is a load error on an AGENT with a non-`str` output, because a
structured value cannot be partial. A model question from
`ask_user` is still answered through `ask_user`, not by a steer. Python hosts pass
the inbox as `run_flow(..., steering=inbox)` (see [API](api.md#steering)); a
node without `steerable: true` never reads it.

### `code` / `tool` — the other leaves

These are the non-LLM computational leaves — a Python callable or a registered
tool. Both consume `input:` bindings and may declare a typed `output:`. A TOOL
node's `input:` names are the tool's own arguments. Its return value is bound as
the node's `output` for downstream references; see
[Typed TOOL nodes](#typed-tool-nodes) for how both sides are checked.

Running an ML model is a host capability: the host serves it behind a tool, and
the flow calls it with `kind: tool`. `kind: model` was removed; loading it is an
error that says so.

A TOOL node's `tool_id:` and an AGENT node's `tools:` grant name either a tool
registered with `register_tool` or one the host passes to that run with
`run_flow(..., tools={...})`. When a host passes `tools=`, the run sees only
those tools. The flow text is the same either way. `tools:` is always a list of
names, even for one tool (`tools: [search]`); `tools: search` fails at load.

An AGENT node's `tools:` entry may also name a **tool group** that the host fills
for each run. The YAML looks the same; only the host knows which names are groups:

```yaml
nodes:
  worker:
    kind: agent
    prompt: ${input.task}
    tools: [toolbox, workspace, company, ask_user]
```

The host passes each group as a list of tools:
`run_flow(..., tools={"toolbox": [t1, t2], "workspace": [...], "company": [], "ask_user": ask})`.
`worker` sees every tool in the three groups plus `ask_user`, and the model calls
them by their own names. The graph stays fixed; only the groups' contents change
between runs, and the YAML does not show them. The run fails before it starts when
an AGENT grants a name the host did not supply (a missing group or tool), when two
different tools share a name, or when a group contains a tool named `ask_user` or
`load_skill`. Groups may share the same tool; the model sees it once.
`ask_user` stays a separate top-level entry. `yian run` fills no groups, so a
flow that grants one needs a Python host. See [Tool groups](api.md#tool-groups).

#### Typed TOOL nodes

A TOOL node may declare `output:` of any type, including a union of records.
Every value the tool returns is checked against it before downstream nodes see
it. A mismatch fails the node with `contract.output`, located at `output:`.
Without `output:`, any JSON-like value passes.

Expected outcomes are variants of the output type; route them with `case` on the
tag:

```yaml
typedefs:
  Filled: {type: 'Literal[filled]', qty: int}
  Interrupted: {type: 'Literal[interrupted]', by: str, partial: 'Optional[int]'}
  OrderResult: Union[Filled, Interrupted]
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: OrderResult
  route:
    kind: case
    on: ${place.output.type}
    cases:
      - when: filled
        then: on_filled
      - when: interrupted
        then: on_interrupted
  on_filled:
    kind: code
    input:
      qty: ${place.output.qty}
    code: 'return f"filled {inputs[''qty'']}"'
    output: str
  on_interrupted:
    kind: code
    input:
      by: ${place.output.by}
    code: 'return "stopped by " + inputs["by"]'
    output: str
output: ${on_filled.output | on_interrupted.output}
```

Interrupts belong to the host: its tool returns the `Interrupted`
variant, and yian validates it like any other value. A tag the union lacks fails
at the gate (`'rejected' is not a tag of the union`). An exception from the tool
still fails the node.

When a run starts, before any record is written, every static TOOL node is
checked against the run's tool. That includes nodes that may never run: an
unreached `case` branch and the child flows of `call`, `map`, and `loop` nodes
(an uncalled `defs:` entry is not checked). The run raises
`ValueError` for:

- an input the tool doesn't take:
  `TOOL node 'place' input 'qtty': tool 'place_order' takes no such input; it takes ['qty', 'sym']`;
- required inputs the node doesn't pass, all named in one error:
  `TOOL node 'place' is missing inputs ['qty', 'sym'] that tool 'place_order' requires`;
- an input whose type can never fit, by the same disjoint rule as a
  [typed binding](#load-check):
  `TOOL node 'place' input 'qty': tool 'place_order' takes int, but the binding is "10" (str)`;
- an `output:` the tool's return annotation can never satisfy:
  `TOOL node 'place' output: declared int, but tool 'place_order' returns str`.

A literal is compared by its exact value, a typed binding (`qty: int = ${x}`) by
its declared type, and a reference by its inferred type. An unproven binding is
skipped. A narrower binding, such as `Optional[int]` into `int` or `str` into a
`Literal`, starts; the tool's own argument validation checks it on each call.
Nothing is coerced: the string `"10"` or `true` into an `int` input, or a number into
a `date` or `datetime` input, fails here, though the tool would have converted it. An unknown `tool_id:` still fails only when its node
runs. How yian reads a host tool's types is in
[Typed tools](api.md#typed-tools).

Changed: a coercible value such as `qty: "10"` into an `int` tool input used to be
converted by the tool on each call; it now fails at run start. Input names and
missing inputs used to fail only when the node ran; they are now checked at run
start, even for nodes that may never run.

A `code` node's `code:` field is one of two shapes:

**Reference** — a `module:function` token: import and call it in-process.

```yaml
verdict:
  kind: code
  input:
    s: ${score.output.signal}
  output: str
  code: pkg.mod:fn             # module:function
```

Only the `module:function` text is part of the flow fingerprint, not the function's
source. Keep the function unchanged while a run may be resumed; an edited function
passes the resume check and runs in the restarted nodes. See
[Durable recovery](api.md#durable-recovery-with-resume_flow).

**Inline** — real source shipped with the flow. Write a **bare body** (no `def`) that
reads the node's inputs via the `inputs` dict and **`return`s** a value; the engine
wraps it as `def main(inputs):` and runs it **in-process**:

```yaml
verdict:
  kind: code
  input:
    rating: ${score.output.rating}
    label:  ${score.output.label}
  output: str
  code: |
    lean = "positive" if inputs["rating"] >= 0 else "negative"
    return f"{inputs['label']}: {lean}"
```

Inline notes:

- **Same one-dict convention as reference mode** — inline reads `inputs["x"]` exactly as
  a `module:function` callable receives `inputs`, so a body promotes to a reference by
  copy-paste. A body **must `return`** a value (a body with no `return` is rejected at
  load).
- **In-process, no isolation — only run flows you trust.** The trust model is
  *author == operator*: inline `code:` executes arbitrary Python **with your privileges**,
  the same capability the reference form already has. There is **no sandbox**, so loading a
  flow you didn't author — one written by someone else, or generated by an agent — runs
  *their* code as *you* (remote code execution). Keep flow YAML operator-authored and
  version-controlled. A runaway body is not killed; interrupt the run to stop it. A killable,
  sandboxed runtime (and an opt-in gate for untrusted sources) is planned.
- **Output must be JSON-representable** (`str` / `int` / `float` / `bool` / `list` /
  `object` / typedefs); a non-serializable return — even nested — fails at the node.
- **Imports** resolve against the process `sys.path` (cwd + `PYTHONPATH`), same as the
  reference form. A dotted token with no colon (`pkg.mod.helper`) is treated as a likely
  typo'd reference and rejected at load with a "did you mean `pkg.mod:helper`?" hint.
- **`file:` / folder / repo sources are planned**, not yet available.

### `case` — branching

A `case` node routes — it has no `input:`. It takes one branch; the other targets
are skipped, and so is every node that plainly reads a skipped node or names it in
`depends_on`. A node the loader lifted out of a skipped node (an inline `call(...)` in
its `input:`, an adaptive composer) is skipped too, whether that node is a target on an
untaken branch or is skipped through its inputs or `depends_on`. A node that names the
`case` itself in `depends_on:` or `runs_after:` is not on a branch: it runs whichever
branch the `case` takes, and is skipped only when the `case` is. A searched `case`
that reads a node only under a default that keeps a binding running, such as
`(a.output :- 0)`, is not skipped with it (see the [searched form](#searched-form)). Read its result with
`${case.output}`. A routing-only `case` may take
none when no `when:` matches and there is no `else:` (see
[Exhaustiveness](#exhaustiveness)); a flow output that reads a skipped node returns
null (see [Skipped branches and flow outputs](#skipped-branches-and-flow-outputs)).

Changed: reading `${case.output}` of a `case` that isn't
[exhaustive](#exhaustiveness) used to load, and returned null when no branch ran; it is
now a load error. Add an `else:`, or list every tag of a non-nullable `Literal` `on:`.
An unquoted `then:` or `else:` word that names no node used to fail with a `KeyError`;
it is now a load error that suggests the nearest node, or quoting. An empty `else:` used
to mean no `else:`; it is now a load error unless the `case` declares an `Optional`
`output:` (then it is the value `null`). Node ids may no
longer start with `__value_` or `__case_`. Flows without value branches or a typed
`case` keep their fingerprints.

Changed: a node with `depends_on:` naming a `case` used to be skipped whenever the `case`
took a `when:` branch rather than `else:`, so a flow output reading it returned null, and
a branch target that also named its own `case` in `depends_on:` never ran. Both now run on
every branch the `case` takes.

Changed: a `then:` naming a node `default` used to share the `else:` route, so both
branches ran (and with no `else:`, `default` ran when nothing matched); it is now an
ordinary node. The `else:` route's handle is `#else` (was `default`) in
`NodeRouted.handle`, run logs, memory episodes, and the CLI's `✓ gate → #else` line.
Flows and child flows with a `case` `else:` get new fingerprints and expansion digests, so
resuming an old run log of one, or of a `case` that matched nothing, fails with a
fingerprint, digest, or replay error. Stored memory episodes keep `default`; they are not
replayed.

Changed: a `when:` in the `on:` form used to be compared as text, so `when: 1` and
`when: true` never matched an int or bool `on:`; it is now a typed literal of the `on:`
type. `when: 1` on a `str` `on:` used to match `"1"`; it is now a load error that says to
quote it. Where the `on:` type isn't known at load (an untyped output, `object`, an
expression whose type isn't proven, e.g. `${input.n + 1}`), the runtime comparison is
exact too: `when: 1` no longer matches the text `"1"`, so type the producer or quote the
value. A `when:` that can't be a value of the `on:` type (a typo'd tag, a wrong scalar
type) is a load error even with an `else:`. An unquoted datetime `when:` is a load error;
quote it. Flows with a number or bool `when:` get new fingerprints.

Changed: a routing-only `case` over a `Literal` flow input (also a LOOP body's carried
field or a child flow's input) that omits a tag with no `else:` used to load and run no
branch for that tag; it is now a load error, as for a `Literal` node output. Two `when:`s
with the same value (`when: 1` twice, `when: 1` and `when: 1.0`) used to load with the
later one dead; that is now a load error.

Changed: a searched `when:` that read a node under `:-`, `:?` or `|`
(`(a.output :- 0) >= 1`) used to skip the `case` with that node; now the `case` runs
when a binding of that part would (see the [searched form](#searched-form)). A call on
such a value, as in `(upper(a.output) :- "x") == "X"`, then fails the run when `a` is
skipped, as it does in a binding; give the argument the default
(`upper(a.output :- "")`). A `:-` or `|` over two node reads used to skip it
when either node was skipped; now only when both are. When such a `case` runs with the
node skipped, its branches don't inherit that node's routing: a union narrowed on the
way to that node isn't narrowed in them, a branch's arithmetic on a value that may be a
skipped null can be a [load error](#skipped-branches-and-flow-outputs), and a flow
output over the branches may be null. Arithmetic in such a `when:` on a value that may
be a skipped null is now a load error. A `case` whose searched `when:` reads a node
under `:-`, `:?` or `|` gets a new fingerprint; others keep theirs.

Simple form — match a value:

```yaml
route:
  kind: case
  on: ${synth.output}          # a Literal value
  cases:
    - when: go
      then: go_brief
    - when: no_go
      then: no_go_brief
  else: more_info_brief        # runs when no `when:` matches
```

Matching:

- Each `when:` is a value of the `on:` type: text, a number, or a bool. An unquoted date
  is its ISO text; quote a datetime.
- Ints and floats compare numerically (`when: 2` matches `2.0`), a bool never equals a
  number, and quoted values are text (`when: "1"` matches only the text `1`).
- A `when: yes` (or `no`, `on`, `off`, `y`, `n`) on a `bool` `on:` is text, so it is a
  load error that says to write `true` or `false`.
- Two `when:`s with the same value are a load error at the later one (`when: 1` and
  `when: 1.0` are the same value; `when: 1`, `when: true` and `when: "1"` are not).
- Where the `on:` type is known (node outputs, flow inputs, a LOOP body's carried fields,
  a child flow's inputs), a `when:` it can't hold is a load error, even with an `else:`:

```text
case node 'route': `when: 1` is an int but `on: ${input.n}` is `str`; quote it: `when: "1"`
case node 'route': `when: nogo` is not a tag of `on: ${synth.output}` (`Literal["go","no_go"]`); use one of: go, no_go
case node 'route': `when: 1.0` duplicates `when: 1` (first at line 7), so it never matches; remove one
```

Searched form — first true `when:` wins (a boolean expression, no `on:`):
{: #searched-form }

```yaml
gate:
  kind: case
  cases:
    - when: "score.output.signal >= 0.5"
      then: bullish
  else: cautious
```

`cases:` is a list of `{when:, then:}` mappings. Any other shape (a `cases:`
mapping without `- `, a bare word, a list item that is not a mapping) is a load
error at that field.

A `when:` reads a node the way a binding does. A plain read
(`score.output.signal >= 0.5`) skips the `case` with `score`. Each outermost `:-`, `:?`
or `|` decides as a binding of just that part, `${...}`, would:

- `when: "(score.output :- 0) >= 0.5"` (or `score.output | 0`) still routes when
  `score` is skipped; the default applies.
- `when: '(score.output :? "need a score") >= 0.5'` routes too; testing that case fails
  the run with `need a score`.
- `(a.output :- b.output) >= 1` is skipped only when both `a` and `b` are;
  `(a.output :- input.seed) >= 1` is skipped with `a`, as `${a.output :- input.seed}`
  would be.
- `a.output > 5 or (a.output :- 0) < 1` is skipped with `a`, by its plain read.

Cases are still tested in order, and `and` / `or` short-circuit, so a later case's `:?`
isn't evaluated once an earlier case matches. `:-` binds loosest:
`x.output and a.output :- false` defaults the whole condition, so write
`x.output and (a.output :- false)`.

Read the taken branch's output with `${gate.output}`:

```yaml
output: ${gate.output}         # bullish's output or cautious's output
```

Routing on a `Literal` node output or flow input (a LOOP body's carried field, a child
flow's input) is **exhaustiveness-checked**: omitting a tag with no `else:` is a
load error. A routing-only `case` over any other value may match nothing; then no
branch runs. A `case` that [produces a value](#exhaustiveness) must always match.

#### Branches

A `then:` or `else:` is one of:

- a sibling node: an unquoted word (`then: reply`);
- a child flow call: an unquoted `call(...)` that is the whole value
  (`else: call(respond, message=${ask.output})`, see
  [Inline `call(...)`](#inline-call-a-flow-call-as-a-bindings-whole-value)). A nested
  call in its arguments (`then: call(outer, x=call(inner))`) runs only on that branch too;
- a value: anything else (`else: "approve"`).

An unquoted word always names a node. A word that names no node is a load error
that suggests the nearest node and quoting:

```text
case node 'route' then: `replyy` names no node. Did you mean `reply`? If you meant the text "replyy", quote it: `then: "replyy"`
```

A `yes`, `no`, `on`, or `off` is a word too, so `else: yes` names a node. When the
case's `output:` takes a boolean, the error adds ``or write `true` for the boolean``.

A quoted branch (`"..."`, `'...'`, or a block `|`/`>`) is always a value:

- `else: "approve"` is the text `approve`.
- `then: "reply"` is the text `reply`, even when a node is named `reply`.
- `then: "call(f)"` is the text `call(f)`. A quoted call that holds `${...}` is a
  load error; drop the quotes to call the flow.
- `then: "${x}"` evaluates its span like any quoted binding. A lone span keeps
  its type.

A value can be text, a number, a bool, `null`, a list, or a record (a block
mapping), nested to any depth:

- `${...}` spans may appear anywhere inside a value. They are evaluated when the
  branch runs.
- Words nested in a list or record are text: `- x` is the text `x`.
- `$$` is a literal `$`: `"costs $$5"` is `costs $5`.
- A YAML date (`2026-01-02`) becomes its ISO text.
- A `call(...)` nested in a value, an unquoted `Type = value` branch (the
  `case`'s `output:` gives the type), and a non-finite number (`.inf`, `.nan`)
  are load errors.

A whole-value span that reads a skipped node skips the value branch too, like any
node input, and the `case` result with it. When a span reads a node that may be
skipped, give it a `:-` default. A value that reads its own `case` is a load
error. So is a value that plainly reads another branch of the same `case` (outside
a coalesce or a `:-` default), since that branch never runs when this one does.

```yaml
decide:
  kind: case
  on: ${review.output}         # Literal[ok, bad, unsure]
  output: str
  cases:
    - when: ok
      then: "approve"          # quoted: the text approve
    - when: bad
      then: write_rejection    # unquoted: the node write_rejection
  else: "ask ${input.owner}"   # text with a span
```

#### `output:` — the type of the result

A `case` with a value branch declares `output:`. Every branch must fit it by the
[typed-binding rule](#load-check): a branch whose type is disjoint from `output:`
is a load error at the branch, and a narrower one is checked at runtime (failure
code `binding.type`). Readers of `${case.output}` see the declared type. `null`
is a valid branch value only when `output:` is `Optional[...]`.

A typed `case` checks the taken branch's value on every run, even when nothing
reads it. The failure is located at the `case`'s `output:`.

This turn of a chat loop returns one `Turn` record from either branch. Exact
`/exit` text is a record value; anything else calls the `respond` flow, which
returns a `Turn`:

```yaml
route:
  kind: case
  on: ${ask.output}
  output: Turn                 # Turn: {transcript: str, exited: bool}
  cases:
    - when: /exit
      then:                    # a value branch: the record itself
        transcript: ${input.transcript}
        exited: true
  else: call(respond, transcript=${input.transcript}, message=${ask.output})
```

#### Reading the result

`${case.output}` reads the taken branch's value, and `${case.output.field}` reads
one field of it. Use it as a whole binding or inside text
(`note: "got ${route.output}"`). It is a load error in a coalesce, in a `:-`
default, inside an expression (`${route.output == "yes"}`), in a `when:` or `on:`, and in
asserts.

#### Exhaustiveness

A `case` produces a value when it declares `output:`, has a value branch, or is
read through `${case.output}`. A `case` that such a `case` routes to produces one
too. It must always match: it needs an `else:`, or an `on:` whose type is a
non-nullable `Literal` with every tag listed in a `when:`, or a non-nullable `bool` with
`when: true` and `when: false`. A flow input counts
as an `on:` type, and inside a narrowing branch only the tags that reach the
`case` count. Only a tagged union's member narrows: a branch on a plain `Literal`
doesn't, so a nested `case` on that same `Literal` still lists every tag or has an
`else:`. A value-producing `case` that may match nothing fails to load:

```text
case node 'gate' produces a value, so it must be exhaustive: add an `else:` (on: ${input.x} is str, not a Literal or bool)
```

The reason in parentheses is one of:

- `on: ... is str, not a Literal or bool`: the `on:` type has no tags and isn't `bool`;
- `a searched case needs an else:`;
- `${input.choice} may be null`: the `on:` type is `Optional[...]`;
- `enum tag(s) ['b'] not covered`: a tag has no `when:`;
- `bool value(s) [false] not covered`: a `bool` `on:` lacks `when: false`.

A routing-only `case` keeps "no match, nothing runs", and routing on a `Literal`
node output or flow input is still checked for missing tags.

#### Synthetic nodes

Each value branch becomes a CODE node `__value_N`, the branch target. A typed
`case` gets a CODE result node `__case_<id>` that checks the taken branch against
`output:`; `${case.output}` reads it. Both are recorded like any node and appear
in traces and failures (a bad branch value fails at `__case_<id>`). Node ids may
not start with the reserved prefixes `__value_` and `__case_`.

#### Skipped branches and flow outputs

A flow output field that reads only nodes a `case` may skip returns `null` when they
were skipped. This covers a reference (`${order.output}`, `${order.output.qty}`), a
coalesce (`${a.output | b.output}`), and a default chain (`${a.output :- b.output}`),
in a root flow, a `defs:` child, and an inline CALL/LOOP/MAP body. The loader reads it
as if `:- null` were written: `${order.output}` becomes `${order.output:-null}`.

```yaml
id: orders
name: orders
input:
  approve: bool
nodes:
  gate:
    kind: case
    cases:
      - when: ${input.approve}
        then: order
    else: rejected
  order:
    kind: code
    output:
      qty: int
    code: |
      return {"qty": 2}
  rejected:
    kind: code
    output: str
    code: |
      return "no"
output:
  order: ${order.output}       # {order: null} when approve is false
```

- **May be skipped:** a `case` target, or a node that plainly reads one (or names it
  in `depends_on`), under some combination of `case` choices. A node that names the
  `case` itself in `depends_on` runs on every branch the `case` takes. A searched
  `case` whose `when:`s read a node only under a default that keeps a binding
  running (`(a.output :- 0)`) runs when that node is skipped. A `case` that
  runs takes one branch, or none unless it is exhaustive: it has `else:`, or its `on:`
  is a non-nullable `Literal` node output or flow input with every tag listed, or a
  non-nullable `bool` with `when: true` and `when: false` (a null `on:` value matches
  nothing).
  So a join over an exhaustive `case`'s branches (`${bullish.output | cautious.output}`,
  or `${gate.output}`) is never null. Conditions of different `case`s are
  treated as independent.
- **Type:** the field is `Optional[T]` when its reference has type `T`, or when every
  reference of its coalesce has type `T`, and untyped otherwise. An explicit `${x :- null}` is `Optional[T]`
  too. A typed `qty: int = ${order.output.qty}` loads and fails with `binding.type`
  when `order` is skipped; `qty: Optional[int] = ...` returns null. A parent that
  passes such a child output untyped to a non-`Optional` input fails to load; narrow
  it with a typed binding (`qty: int = ${c1.output.qty}`) or give it a default.
- **Templates and comparisons** are unchanged:
  `'${order.output.qty} and ${rejected.output}'` renders ` and no`, and
  `${order.output.qty == 2}` is `false`.
- **Arithmetic** (`+ - * / % **`, unary minus) on such a value is a load error at the
  binding: in a flow output, and in a node binding (including a `case`'s `on:`) or a
  searched `case`'s `when:` whose node still runs when the value's node is skipped:

  ```
  output 'q': ${order.output.qty + 1} does arithmetic on order.output.qty, which is null
  when node 'order' is skipped; give it a default with :-, e.g. (order.output.qty :- <default>)
  ```

  A node that also plainly reads `${order.output}` is skipped with it, so its
  arithmetic loads. Arithmetic in flow-level `asserts:` is not checked. The default
  side of `:-` is checked as if it always runs: `${a.output :- (b.output + 1)}` is an
  error even when `b` runs whenever `a` is skipped, since `a` might be null while it runs.
  In a `when:`, `((a.output.n + 1) :- 0) > 1` is an error and
  `(a.output.n :- 0) + 1 > 1` loads.
  A null guard is reported too: `${a.output != null and a.output.n + 1 > 1}` is an
  error though `and` stops at the guard; write `${(a.output.n :- 0) + 1 > 1}` instead.
- In a LOOP body the null sets the carried field (see [`loop`](#loop-re-run-a-body-under-a-predicate-or-a-fixed-count));
  a MAP returns null for each element whose branch was skipped.

#### Narrowing a union in a branch

A `case` branch can read fields only some [union](#unions-of-records-union) members have.
An `on:` over the union's tag field narrows the value in each branch:

```yaml
route:
  kind: case
  on: ${decide.output.action}  # decide.output is a Decision
  cases:
    - when: buy
      then: place_buy          # sees decide.output as Buy
    - when: sell
      then: place_sell         # sees decide.output as Sell
  else: log_hold               # sees the unlisted members: Hold
place_buy:
  kind: code
  input:
    qty: ${decide.output.qty}  # a Buy field; a load error outside the branch
  code: ...
```

- Two `when:`s with one `then:` give that target the union of both members. `else:`
  gives the members no `when:` lists, keeping an `Optional[...]` union's null. Once
  every tag is listed, `else:` sees only null for an `Optional[...]` union and leaves a
  non-optional one unnarrowed.
- `on:` must be one direct reference to a tag field: `${node.output[.field...].tag}` or
  `${input.x[.field...].tag}`. A coalesce or embedded text doesn't narrow.
- A `when:` that isn't a tag is a load error: ``case node 'route': `when: buyy` is not a
  tag of `on: ${decide.output.action}` (`Literal["buy","hold","sell"]`); use one of: buy,
  hold, sell``. Without `else:` every tag needs a
  `when:` (`enum tag(s) ['hold'] not covered`).
- Narrowing follows the branch target's chain: a node that reads it through a plain
  reference, or names it in `depends_on`, sees the same members. A join
  `${b.output | s.output}` of a `buy` and a `sell` branch sees `Buy` or `Sell`.
  Narrowing doesn't pass through an optional read (`${b.output:-0}`) or `runs_after`,
  nor through a searched `case` whose `when:` reads that node only under a default
  that keeps a binding running (`(a.output :- 0)`): its branches see the value
  unnarrowed.
  A node that several `case`s route is narrowed only when every one of them narrows the
  same value, and then sees the members any of them allows.
- A nested `on:` over a field of the narrowed member narrows again, and its exhaustiveness
  counts only that member's tags.
- A value branch sees the narrowed type like a routed node: under `when: buy` its
  spans may read `${decide.output.qty}`.
- A `call` target can pass the narrowed value to a child whose input is the member
  (`order: Buy`). An inline child body's inputs are typed before narrowing, so a member
  field bound there fails with `cannot infer type for inline binding`. An inline
  `call(...)` lifted out of a routed node (a nested call in a target's arguments, or a
  call in a branch node's `input:`) carries that node's route and waits for its inputs,
  so it is narrowed like it.

A searched `case` narrows with [`isinstance`](expressions.md#isinstance-the-type-test): each
top-level `and` part `isinstance(<ref>, T)` of a `when:` narrows that reference to `T`
in that branch and its chain. `T` may be a member, a union of members, or any other type;
a `T` the reference's type already fits, such as `object`, doesn't narrow. First true `when:` still wins, and checks may overlap. The searched form's `else:`
narrows nothing. A later `and` part of the same `when:` may already read `T`'s fields:

```yaml
gate:
  kind: case
  cases:
    - when: isinstance(decide.output, Buy) and decide.output.qty > 5
      then: big_buy            # sees decide.output as Buy
    - when: isinstance(decide.output, Sell)
      then: place_sell
  else: log_other
```

### `human_input` — a deterministic human gate

A `human_input` node **always** invokes the run's `ask_user` handler at a
fixed point and blocks its worker until the human answers. The answer is
validated against the declared `output:` before the flow continues. This is one
live node call; the invocation stays active until the answer arrives.

```yaml
approve:
  kind: human_input
  input:
    plan: ${draft.output}      # context the prompt may reference (bare ${plan})
  prompt: |-
    Here is the plan:

    ${plan}

    Approve as-is, or revise? (approve / revise)
  output: Approval             # a typed answer, e.g. a Literal enum
```

The same invocation stays live until the host returns an answer. Instead of (or
alongside) a `prompt:`, a gate may carry **questions** —
AskUserQuestion-shaped multiple-choice/free-text prompts. Each question is
`{question, header, options:[{label, description}], multi_select}`; `options`
(omit ⇒ free-text) and `multi_select` (default `false`) are optional. 1–4
questions, headers unique. The host always offers a free-text **"Other"** escape.
The gate's answer is a **record keyed by header** — `{header: label}`, or
`{header: [labels]}` for a `multi_select` question — so `output:` defaults to
`object`. `prompt:` is optional once a node has questions; `questions:` and
`adaptive_questions:` are **mutually exclusive**.

**Static questions** — a literal list (with `${...}` templating from `input:`):

```yaml
ask:
  kind: human_input
  input:
    proj: ${input.proj}
  questions:
    - question: "Which framework for ${proj}?"   # ${proj} renders from input
      header: Framework                          # answer key -> {Framework: <label>}
      options:
        - { label: React, description: A component library. }
        - { label: Vue, description: A progressive framework. }
      multi_select: false        # optional, default false
    - question: "Any notes for the build?"       # no options -> free-text
      header: Notes
  # output omitted -> defaults to object: {Framework: ..., Notes: ...}
```

`question` and each option's `label` and `description` are templates like `prompt:`:
they read the gate's `input:` names, bare. `header` is not a template. A pool
reference is a load error at that string, with a hint to bind the value in `input:`:
`node 'ask' questions[0].options[1].label: ${input.proj} is not a declared input`.

`prompt:`, `adaptive_questions.prompt:`, and each question's `question`, `header`,
`label`, and `description` must be text. A value YAML reads as non-text is a load error
at that value, as for an AGENT `prompt:`; quote it. A blank question field says
``node 'ask' (kind=human_input): `questions[0].header:` must be text, but it is empty;
write the text``.

Changed: most `human_input` prompt or question text with an unknown name, a pool
reference, an unknown function, or a span that doesn't parse used to load and fail the
gate when it ran. A `:-` default on a name the gate doesn't declare, or on a pool
reference (`prompt: "Hi ${input.user :- 'there'}"`), used to render the default every
time, silently. Both are now load errors. To migrate, bind the value and its default in
`input:` and read the local name:

```yaml
greet:
  kind: human_input
  input:
    user: ${input.user :- "there"}
  prompt: "Hi ${user}"
```

**Adaptive questions** — an LLM composes the questions from context. The block
**desugars at load** into a synthetic composer AGENT node (`<node>__compose`, output
`list[Question]`) wired into the gate; the runtime gate never calls an LLM. The
composer runs only when the gate can: it carries the gate's `case` routes,
`depends_on:`, and `runs_after:`, and waits for the gate's other inputs, so no LLM call
happens on a branch the gate isn't on or when a skipped input skips the gate. So a node
that reads `${<node>__compose.output}` can't also be read by the gate or be in its
`depends_on:` or `runs_after:`: that is a cycle.
The input name `__questions` is reserved for that generated wire and cannot be
authored on an adaptive gate.
The brief reads the gate's `input:` names, bare, and is checked at load like an AGENT
prompt. An error names the gate and points at the brief:
`node 'ask' adaptive_questions.prompt: ${nope} is not a declared input or variable`.

```yaml
ask:
  kind: human_input
  input:
    ctx: ${research.output}
  adaptive_questions:
    prompt: "Design 1-3 questions with options for: ${ctx}"  # required (LLM brief)
    llm_config: { model: ... }   # optional — the composer's provider/model
    retries: 3                   # optional — self-correction attempts
```

`adaptive_questions.mode:` was removed; writing it is a load error.

Changed: a bad brief used to be reported on `node 'ask__compose' prompt` and point at
the gate's `prompt:` line, or at the node when the gate had no `prompt:`.

**Upstream questions** — read the list from an author-written upstream node:

```yaml
ask:
  kind: human_input
  input:
    qs: ${composer.output}            # composer.output is a list[Question]
  questions: ${qs}
```

`questions:` is either a list or exactly one `${name}` naming a declared input;
anything else (a mapping, a number, text around the ref, `${a.b}`, a call) is a
load error. Put free text in `prompt:`.

### `wait` — block until an absolute UTC deadline

```yaml
settle:
  kind: wait
  until: ${input.as_of}
  # downstream nodes order after it with depends_on: [settle]
```

`until:` must resolve to a `date` (`2026-05-21`) or `datetime`
(`2026-05-21T14:30[:00][Z|+07:00]`); see
[the accepted forms](typing.md#valuekind-the-closed-vocabulary). A date means
midnight UTC; a naive datetime is UTC; an offset-aware datetime is normalized to
UTC. A deadline at or before the current instant returns immediately. Otherwise
WAIT occupies one worker until the remaining wall-clock duration elapses, then
returns `None`; it does not split the run into another execution segment.
Engine/host shutdown wakes the worker cooperatively.

### `call` / `map` — composition over child flows

`call` runs a child flow once; `map` runs it over a list (`${item}` is the
current element). A reusable child is named by `call:`: resolution is
`defs:` first, then a declared `uses:` alias.

```yaml
each:
  kind: map
  over: ${input.tickers}       # a list[T]
  call: child_flow             # node value is list[U]
  input:
    ticker: ${item}
```

The `input:` map contains the child's arguments. A `call` re-exports the child's
output (`None` when the child has no `output:`; `{}` for `output: {}`); a `map`
requires a non-empty child `output:` and returns `list[U]` in `over:` order. If the
child's `output:` is a record, even of one field, the MAP returns a list of records.
`parallel:` is accepted but currently inert; MAP concurrency is controlled by
the run's `num_workers`, while collection order remains the `over:` order.

Each argument must fit the child's input type, checked at load. Nothing is coerced,
because a `call`/`map` edge is not a flow boundary. An untyped
[`loop`](#loop-re-run-a-body-under-a-predicate-or-a-fixed-count) seed that reads no
reference is checked the same way against a named body's input. Each argument is
checked as follows:

- A reference is compared by its type. A MAP's `${item}` or `${item.field}` is typed
  from the element type of `over:`; over a literal `over:` list, each element is
  checked by value. A `str` source does not fit an `int` or `date` input, and an
  `Optional` source does not fit a strict one.
- A YAML literal is validated by value: `"3"` does not fit an `int` input (write `3`),
  `5` fits a `float` input, `9` does not fit `int[1, 3]`, and `5` does not fit `str`.
  `null` fits only an `Optional` input. An unquoted `2026-01-01` (a named input or
  a bare `call(...)` argument) is a date to YAML, not text, so it fits neither a
  `date` nor a `str` input, alone or inside a list or record: quote it
  (`"2026-01-01"`), as in a typed binding or a default.
- `${x :- fallback}` checks `x` without its null, then the fallback: a constant by
  value, a reference (`input.y` or `${input.y}`) by its type. `:- null` fits only an
  `Optional` input. `${x :? message}` checks `x` without its null, and `${a | b}`
  checks every operand, all but the last without its null.
- A source whose type the loader can't prove is checked, again with no coercion, when
  the child starts: the text `"3"` from an untyped `code` output into an `int` input
  fails with `contract.input`. For a value narrower than the input, such as an
  `Optional` source you know is set, use a
  [typed binding](#typed-bindings-name-type-value): `t: int = ${input.n}`.

```text
call node 'each': binding 'day' — child expects 'date', source is 'str'
call node 'c': binding 'n' — child expects 'int', but the value "3" does not fit; text is not converted between flows, so write 3 without quotes
call node 'c': binding 'day' — child expects 'date', but the value 2026-01-01 does not fit; quote it to keep it as text: "2026-01-01"
```

Changed: a `${item}`, literal, `:-`, or `|` argument used to be checked only when the
child started, after the work upstream had run, and the child's start converted text
for an `int`, `float`, or `bool` input. These used to run and now fail at load:

- a literal, `:-` fallback, or `|` operand written as text for a number or bool, such
  as `n: "3"`, `${input.n :- "3"}`, or `${input.n | "3"}` into an `int` input (write
  `3`), and a `${item}` over a literal `over: ["1", "2"]` into an `int` input;
- an unquoted date or datetime literal, alone or inside a list or record, or as an
  element of a literal `over:` list, into any input, `str` included (quote it);
- a `:-` or `|` whose reference doesn't fit the input, which used to run whenever the
  reference was null, such as `${input.m :- 3}` or `${input.m | 3}` into an `int` input
  with `m: Optional[str]` (fix the reference's type);
- a `${item}` over a `list[str]` into a `date` input (type the list `list[date]`), and a
  `${item}` over a `list[Optional[str]]`, or `${x :- null}`, into a strict input (use a
  typed binding).

Text from a source the loader can't type, such as an untyped `code` output, is no
longer converted when the child starts either: it fails with `contract.input`. The same
holds for a `loop` seed that reads a reference or is typed: it is not checked against
the body's input at load (a returned field is checked against the body's output), so a
mismatch fails with `contract.input` when the body starts. Type the source, or convert
the value in the producing node.

Changed: a literal `loop` seed that doesn't fit a named body's input, such as `n: "1"`
into an `int` input, used to fail when the body started, or not at all when the loop
never ran. It now fails at load. So does a `${...}` seed that reads no reference, such
as `${null}`, for a field the body doesn't return.

#### Inline child bodies — skip a one-off `defs:` entry

When a CALL, MAP, or LOOP body is used only once, put the child graph directly
on the composite node. This is load-time sugar for a private child flow; the
output uses the same compiled-child execution path as a named `defs:` body, while
keeping occurrence-local identity and diagnostics.

```yaml
nodes:
  clean_one:
    kind: call
    input:                         # arguments become the child parameters
      text: ${input.text}
    nodes:                         # the private child body
      clean:
        kind: code
        input:
          text: ${input.text}      # reads the generated child parameter
        output: str
        code: |
          return inputs["text"].strip()
    output: ${clean.output}        # child return binding

  clean_each:
    kind: map
    over: ${input.texts}
    input:
      text: ${item}                # current item becomes a child parameter
    nodes:
      clean:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: |
          return inputs["text"].strip()
    output: ${clean.output}
```

Inline-body rules:

- Write **exactly one** of `call:` or `nodes:`. Inline `nodes:` must be a
  non-empty mapping and requires `output:`; `output:` is not valid beside a
  named `call:`. The key itself is required: CALL may use `output: null` and
  return `None`, while MAP requires a non-empty output and LOOP requires at least
  one carried field. As in any flow, `output: {...}` returns a record, even with
  one field.
- The outer composite `input:` is still the argument map. Its keys become the
  anonymous child's typed input parameters, and the body reads them as
  `${input.<key>}`.
- The child is closed to its parent's pool: it cannot capture parent inputs,
  parent nodes, or MAP's `${item}`. Pass those values through the outer `input:`
  map, then read the generated inputs and child nodes inside. The run-global
  `${system.today}`, `${system.now}`, and `${system.run_id}` values remain
  available as in every flow.
- Parameter types are inferred strictly. YAML null/bool/int/float/string values,
  closed literal records, non-empty homogeneous literal lists, and direct typed
  references (`${input.x}`, `${producer.output.field}`, or MAP `${item.field}`)
  retain their structural type. A nullable record path produces an optional
  parameter. If a type cannot be proved, loading fails; declare it with a
  [typed binding](#typed-bindings-name-type-value) (`n: int = ${input.a + input.b}`),
  whose declared type becomes the parameter type, or use a named `defs:` child with
  an explicit `input:` signature.
- MAP proves the `over:` element type only when an untyped outer argument is a
  lone whole-value `${item...}` reference. An opaque `over:` therefore remains
  legal when the inline body does not need the item, or when the item binding is
  typed (`t: str = ${item}`). When item typing is requested, `over:` must have an
  exact proven list element type. That proof retains
  element nullability, so an optional element admits actual null items and
  preserves them as `None` in typed storage.
- Strings nested inside literal lists or mappings stay literal. For example,
  `payload: {text: "${input.text}"}` passes the characters `${input.text}`; nested
  containers are not scanned for references.
- Existing composite configuration keeps its existing owner and meaning:
  LOOP drivers still configure the outer composite, `env:` stays on that node
  (inner body nodes may set their own `env:` normally), CALL/LOOP node asserts
  keep their normal semantics, and MAP node-local asserts remain invalid.
  `parallel:` remains an accepted outer MAP field but is currently inert.
  `llm_config:` remains invalid on a composite node; the enclosing flow's LLM
  cascade still descends into body AGENT nodes exactly as it does for an unconfigured
  named child. There is no separate inline-child
  `llm_config:`/`env:`/`asserts:` surface; configure individual body AGENT nodes or use
  `defs:` when the child needs its own supported flow-level configuration or
  assertions.

Inline bodies may contain further CALL/MAP/LOOP nodes, including further inline
bodies. Each occurrence remains a separate lexical child.

#### Inline `call(...)` — a flow call as a binding's whole value

Declaring a whole `call` node for a one-off child call is verbose. The `call(...)`
**directive** is sugar: written as a binding's **whole value**, it desugars at
load into an anonymous `call` node, and the host binding is rewritten to reference
that node's output. This applies to every authored node `input:` mapping, including
HUMAN_INPUT context bindings and LOOP carried-record seeds.

```yaml
# on a node `brief` (routed by a case, with `runs_after: [fetch]`) —
news: call(enrich, topic=${input.topic})   # inline directive (sugar)

# desugars to a sibling that carries `brief`'s route and ordering:
__call_0:
  kind: call
  call: enrich
  input:
    topic: ${input.topic}
  runs_after: [fetch]   # copied from `brief`, as are its `depends_on:` and case routes
news: ${__call_0.output}
```

A route can't be written on a node, so the example doesn't show it: a `case` branch
that routes to `brief` routes `__call_0` too. Nor does it show the wait on `brief`'s other
inputs: if `brief` also reads `notes: ${draft.output}`, `__call_0` waits for `draft` and is
skipped when `draft` is.

Rules:

- **Whole-value only.** An unquoted value (a binding value or a `case`
  `then:`/`else:` route target) whose trimmed text starts with `call(` must be
  exactly one well-formed directive. Anything else is a load error that names the
  problem: unbalanced parentheses, text after the closing `)` (`call(f) tail`), a
  first argument that is not a bare flow id (`call(f x=1)`), a positional argument,
  a keyword given twice, a keyword without one name before its `=`, or a keyword
  value that reads as two keywords (a missing comma, `call(f, a=1 b=2)`; `==`,
  `>=`, and a glued `a=b` stay part of the value).
- **Quote it for text.** A quoted value (`'...'`, `"..."`, or a block `|`/`>`) is
  text, never a call: `note: "call(f) tail"` is the text `call(f) tail`. A quoted
  `call(...)` that holds `${...}` is a load error; drop the quotes to call the flow.
  A `call(` in the *middle* of a value, or inside a `${...}` span, is also text.
- **Keyword args only** — `call(f, x=1, note=${a.output})`. Each arg value is a
  `${...}` expression or a literal. A bare literal is one plain YAML scalar read
  with the [YAML values](#yaml-values) rules (`window=30` binds the int `30`, `1e3`
  the float 1000.0, `~` null, `None` the text `None`); brackets, braces, anchors,
  tags, and `<<` are not parsed, so `x=[1]` is the text `[1]`. A quoted arg
  (`x="call(g)"`, `x="1e3"`) is text. A bare value YAML can't build, such as
  `d=2026-13-45`, is a load error that says to quote it. Each keyword appears at
  most once.
- **Nesting is allowed** — `call(outer, x=call(inner, y=1))`. An unquoted arg that
  starts with `call(` follows the same whole-value rule.
- **Runs only when its host can.** The call runs only on the `case` branches that reach
  the node it is written in (its host), is skipped when a host `depends_on:` node is
  skipped, and waits for the host's `runs_after:` nodes. It also waits for the host's
  other inputs and is skipped whenever a skipped input skips the host; an input with a
  default (`${x.output :- "none"}`) doesn't skip the host, so the call only waits for
  `x`. Calls written in the same node don't wait on each other unless one reads the
  other, and a nested call also carries its outer call's other arguments: in
  `call(f, x=call(g), y=${a.output})`, `g` waits for `a` and is skipped when `a` is. A
  call in the flow `output:` or `asserts:` has no host. A node that reads the call's output by
  id (`${__call_0.output}`) is skipped with it, and if the host also reads that node or
  names it in `depends_on:` or `runs_after:`, the flow has a cycle (a load error).
- **In a flow `asserts:` item** a value starting with `call(` follows the same rule,
  quoted or not: an assert is an expression, so quoting it does not make it text.
- **A flow call inside `${...}` is retired.** The old `${flow(args)}` form (a flow
  call *inside* braces) is a load error that points at the `call(...)` directive.
  An **embedded** flow call (`pe=${relevance(t=x)}` — a call inside surrounding
  text or a `${}` span) must be **hoisted** to a named `call` node and referenced
  by output. A **coalesce of calls** (`${a(x=1) | b(y=2)}`) has no whole-value form
  and is a load error — hoist each call to its own node, then coalesce the outputs.
  (Pure builtins like `${upper(x)}` stay legal inside `${}` — only *flow* callees
  moved out.)

This directive and an inline child body solve different verbosity problems.
`call(named, x=...)` is a whole-value binding that invokes an already named
`defs:`/`uses:` child flow and synthesizes the surrounding CALL node. A
`kind: call` node with `nodes:` defines that one-off child body in place. It
is invalid to combine `call:` and `nodes:` on the same node.

### `loop` — re-run a body under a predicate or a fixed count

`loop` runs a child flow (the **body**) over and over, threading one **carried
record** from each iteration into the next. It is the engine's `while`/`do-while`/
`for`: the body reads the carried record and returns the fields that change for the
*next* one (`'a -> 'a`), and the loop re-runs it under one of three drivers — a
pre-check predicate (`while:`), a post-check predicate (`until:`), or a fixed count
(`times:`).

```yaml
turn:
  kind: loop
  call: chat_turn              # the body flow (a def or a file)
  input:                       # the SEED carried record
    messages: []
    exited: false
  while: not exited            # pre-check predicate, over the carried record
```

A one-off LOOP body can be inline too:

```yaml
repeat:
  kind: loop
  times: 3
  input:
    n: 0
  nodes:
    bump:
      kind: code
      input:
        n: ${input.n}
      output:
        n: int
      code: |
        return {"n": inputs["n"] + 1}
  output: ${bump.output}
```

Here the generated child signature is `n: int`, and its record output `{n: int}`
returns the whole carried record.

- **`input:`** is the seed carried record — its field names and types define `'a`.
  A typed seed (`n: int = 0`) checks the starting value. A field the body omits
  keeps the seed's declared type; a returned field follows the body's output type.
  An untyped seed that reads no reference is checked against a named body's input
  like a [`call` argument](#call-map-composition-over-child-flows), and nothing is
  coerced: `n: "1"` into an `int` input is a load error (write `1`), and `null` or
  `${null}` fits only an `Optional` input. This holds even when a `while:` loop would
  run zero times.
- **`call:` / `nodes:`** names a reusable body or defines a one-off body inline.
  The body's `output:` returns a **subset** of the carried record: at least one
  carried field, and any number of them. Each field the body omits keeps its value
  from the previous iteration; there is nothing to declare. A body output field
  that is not a carried field is a load error (`body output field(s) [...] are not
  carried record fields [...]`). The body still reads only its carried record: the
  fields it reads (`${input.X}`) must be a subset of the carried names. This
  contract is checked twice: field **names** at build, and each statically proven
  **returned** field against a statically proven seed source at load. A returned
  field must keep its carried type. An omitted field takes its type from its seed
  binding: a typed seed gives its declared type, and an untyped literal seed stays
  untyped (it is still checked by value against a named body's input). Every proven
  returned field is enforced at runtime and replay even when its seed was opaque at
  load. Each output field's type
  is inferred from its binding shape (see [expressions](expressions.md)): a lone
  direct ref preserves an exact or partial contract, a known partial-record field is
  exact, a proven comparison or boolean expression is `bool`, and a
  **concatenating template** (literal text around spans) is `str`. Therefore a body may
  write `exited: ${ask.output == "/exit"}` directly, and a carried transcript may grow
  through a plain template binding, without a CODE fold node. If a computed field's
  type is not proven, its declared name is still checked but its type stays opaque;
  opaque never means a concrete `None` value and does not erase type checks for proven
  sibling fields. LOOP enforces the closed field set and every proven sibling on normal
  iterations, zero-iteration `while:` commits, and durable replay. A literal `${null}`
  seed is compatible with a nullable body field, including inside a typed list, when
  the body is a named `defs:` body or the seed is typed. An inline body's parameters
  are typed from the seeds, and each returned field must fit its parameter at load,
  since the next iteration binds it there. An untyped `${null}` seed gives the type
  `None`, so with an inline body, type the seed: `seen: Optional[list[str]] = ${null}`.
  Over a typed seed (`note: str = "start"`), a returned field typed `Optional[str]`
  loads, and the next iteration's input check rejects a null (`contract.input`). Over
  an untyped seed (`note: "start"`) the same field is a load error that asks you to
  declare `note: Optional[str] = "start"`.
- **Omitted is not null.** An omitted field keeps its value. A returned null sets the
  field to null, and so does a field that reads a node a `case` skipped (see
  [Skipped branches and flow outputs](#skipped-branches-and-flow-outputs)). When the
  seed is typed, the returned field's type must accept it: a
  producer typed `Optional[str]` fits an `Optional[str]` seed. A literal `${null}` is
  typed `None`, and an `Optional[str]` seed does not fit `None`, so that load fails.
- **One field is enough.** `output: {done: ...}` returns the record
  `{done: ...}`. An empty `output: {}`, or a reference to an empty record, returns
  no carried field and is rejected.
- **Merged views.** The loop's value (`${loop.output.x}` downstream) and the LOOP
  node's own `asserts:` see the merged carried record. A body def's own flow-level
  `asserts:` see the body's own output (and its inputs through `${input.X}`).
- **Exactly one** of `while:` / `until:` / `times:` per loop node selects the
  driver. Zero or more than one is a load-time error (`exactly one of
  while:/until:/times: is required`).

A chat body carries `person` and `talking_to` unchanged every turn and returns only
what changes:

```yaml
defs:
  turn:
    input:
      person: str               # reads only what it needs
    nodes:
      ask:
        kind: human_input
        prompt: "You:"
        output: str
      reply:
        kind: agent
        input:
          message: ${ask.output}
          person: ${input.person}
        output: str
        prompt: "You are ${person}. ${message}"
    output:                     # person and talking_to carry over unchanged
      exited: ${ask.output == "/exit"}
      reply: ${reply.output}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      person: ${input.person}
      talking_to: ${input.talking_to}
      exited: false
      reply: ""
    while: not exited
```

**`while:` — pre-check (0+ runs).** A predicate evaluated on the carried record
*before* each iteration; 0 iterations run if the seed already fails it. It is a
record-scoped boolean over bare refs — every ref must name a **carried record
field**, and a field read below it (`state.count`) must be one the carried field's
type has (a typo'd name or field is rejected at load, not silently read as falsy; a
field only some union members have needs an `isinstance` part before it, see
[References](#references-naming-a-value)) — written in the canonical bare form:
`while: not exited`. `max:` is optional. The predicate is an expression string: a
YAML `true`, `false`, number, or list is a load error, and so is an empty string.

**`until:` — post-check / do-while (1+ runs).** Same record-scoped predicate
syntax as `while:` (bare refs, no `${}`), but checked *after*
each iteration: the body always runs at least once, and the loop **continues
while the predicate is FALSE and stops the moment it becomes TRUE**. `max:` is
optional.

```yaml
retry:
  kind: loop
  call: attempt
  input:
    ok: false
  until: ok                    # post-check; runs once, then stops when ok is true
  max: 5                       # optional runaway guard
```

**`times: N` — fixed count.** The body runs exactly `N` times, with no predicate.
`N` must be a plain integer `>= 1`. **`max:` is redundant here and REJECTED** — the
count already bounds the loop, so supplying both is a load-time error (`max: is
redundant with times:`).

```yaml
poll:
  kind: loop
  call: step
  input:
    n: 0
  times: 3                     # exactly 3 runs; do NOT also give max:
```

- **`max:`** (for `while:`/`until:`) is an optional runaway guard. Without it (or
  with `max: null`) the loop has no iteration limit: it runs until its predicate ends
  it (or the run is interrupted). When given, it is a plain integer `>= 1`, and a
  loop that would run more than `max` iterations fails loudly (`LoopMaxExceeded`).
- A `while:`/`until:` predicate that reads only fields the body never returns cannot
  change. Without `max:`, such a loop could never end, so it is a load error
  (`` this loop can never end: `until: done` reads `done`, which the body never
  updates; return `done` from the body, or add `max:` ``). With `max:`, it loads, and
  fails with `LoopMaxExceeded` if it reaches the cap.

The node's value is the final merged carried record (committed under the loop
node's id once the loop stops).

A body may itself block (e.g. a `human_input` leaf): the worker stays inside that
live iteration until the host answers, then the same run advances into the next
iteration. This is the shape a chat REPL takes.

Each continuing iteration **prunes** the previous iteration's body nodes from the
live graph, along with the nodes grown under them by a nested CALL, MAP, or LOOP. So
the live graph stays about one iteration, whatever the body. The run log still keeps
every iteration, and so does the execution record a `learnable=True` run keeps for
backward. Resume and backward read those, not the live graph. The engine's node budget
(10,000 live nodes, "expansion exceeded node budget") is checked before pruning. So it
limits how big one iteration can get (the finished iteration plus the next body), not
how many iterations run.

The next iteration starts as soon as the body `output:` is ready, as a CALL's caller
continues as soon as its child's output is. A body node the output doesn't read (a
side step such as logging, notifying, or saving) may still be running then, so it can
overlap the next iteration, whatever `num_workers` is. It runs to completion, and the
run waits for it before it ends; a failure there fails the run. Until it finishes, it
stays live with the nodes it reads and the nodes grown under it, and the next iteration
that continues after it has finished prunes them.

## Effects from inside an AGENT node — the `ask_user` tool

An AGENT node can be granted the ordinary `ask_user` tool. Unlike
`human_input` (which always asks), `ask_user` is *model-chosen*: the node calls it
**only if** it decides it needs a fact it cannot supply. The tool blocks inside
the same live model/tool loop, then its returned answer is fed back as the tool result.

```yaml
assistant:
  kind: agent
  tools: [ask_user]            # explicitly grant the capability; tools run the loop
  input:
    request: ${input.request}
  output: str
  prompt: |-
    The user asks: ${request}
    If a detail essential to a good answer is missing, call ask_user ONCE to get it.
    Otherwise answer directly.
```

The model may pass up to four short `options` with its question. `yian run`
and `yian chat` show them as a choice list with an "Other" answer; the human
can always type their own answer. Prompt for options when the choices are clear,
for example "call ask_user with the two approaches as options".

For a *guaranteed* gate use `human_input`; for "ask only if needed" use
`ask_user`.

All three authorable blocking surfaces are live calls: `ask_user` returns the
human's answer to the same model/tool loop, HUMAN_INPUT returns it as the node's
output, and WAIT returns `None` at its deadline.
There is no authored `entry:`/`controls:` surface on an AGENT node and no event-mode WAIT.

## Expression contexts

One `${...}` grammar serves every context — the difference is what each context
*does* with the result, not what it can parse:

- **Bindings** (`input:` / `output:` values): the expression is **evaluated** to a
  typed value. Refs, literals, arithmetic (`+ - * / % **`), list literals,
  comparisons, `:-`/`:?`, `\|`, and pure builtins all work. To invoke a child flow
  as a binding's whole value, use the `call(...)` directive — not a `${...}` span.
  A binding may declare its value's type as `name: Type = value` (see
  [Typed bindings](#typed-bindings-name-type-value)); quote a value whose text
  contains ` = `.
- **`when:` / `asserts:`**: the expression is **tested** as a boolean. Same
  operators. Write conditions in the canonical **bare** form (`a.output > 5`) — no
  `${}`. Braces around the whole expression (`${a.output > 5}`) load and evaluate
  identically (an expression field is already in expression mode, so the wrapper
  is a redundant no-op), but bare is the form used throughout the docs and seeds.
  Braces on a ref inside the expression (`${a.output} > 5`) are a `${...}` inside
  an expression: a load error that shows the rewrite.
- **Prompts**: free text with embedded `${...}` spans (each stringified into the
  text); spans may use the full grammar, but reference only the node's own declared
  inputs. `isinstance` is the exception: compute it in an input binding (see
  [Expressions → `isinstance`](expressions.md#isinstance-the-type-test)).

> Bindings wire, conditions test, nodes compute.


## Asserts

`asserts:` are boolean invariants; a false (or raising) one fails the run loudly.
A top-level `asserts:` runs over `input.X` (a boundary check, before any node)
or `node.output` (a post check, after the terminal). An unqualified single name
matching a declared flow input is shorthand (`age >= 18` means `input.age >= 18`)
and wins over a same-named node; node results still require `node.output`. The
shorthand does not extend through fields: write `input.profile.age`, not
`profile.age`. `${system.X}` is **not**
allowed in a flow-level assert — it is ambient (there is no producer node to draw
the check's dependency edge from), so it raises a `LoadError` at load time. If you
need a system-dependent check, put it in a body node's `asserts:` or a `case`.

A flow-level post assert reads its producer over an **optional** edge (the assert
must not deadlock the run when its producer sits behind an untaken `case` branch).
So a producer that is **skipped** resolves to `null` for the assert: a check like
`maybe.output == null` **silently passes** when `maybe` is skipped, rather than
failing. Assert on a value you know is produced, or gate the check behind the same
branch as its producer. Arithmetic on such a producer in a flow-level assert is not
checked at load.

```yaml
asserts:
  - synth.output in ["go", "no_go", "needs_more_info"]
```

A node may also carry its own `asserts:`. A node-local assert is **PRE** if it
reads only the node's inputs (checked before the node runs) and **POST** if it
reads `output` (checked once the node's value is committed). Both fail the run
loudly, exactly like a flow-level assert.

```yaml
nodes:
  classify:
    kind: agent
    prompt: ...
    output: str
    asserts:
      - output in ["go", "no_go"]   # POST — reads the node's own output
```

This holds for a `call` node too: its POST `asserts:` fire when the call's value
is committed, and may read `${output}` **and** the call's declared inputs
(`${name}`), matching leaf-node semantics. (`map` nodes still reject node-local
`asserts:` at load time — assert a `map`'s result with a flow-level or downstream
check instead.) A `call` or `loop` node's own asserts can't use `isinstance`; assert
inside the child flow.

## Next

- [Examples](examples.md) — these constructs in working flows.
- [Python API](api.md) — run and resume flows from code.
