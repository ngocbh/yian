---
title: "Types"
description: "Scalars, records, Optional, unions, Literal, files and narrowing."
---

Every input, output and value in a flow can have a type. yian checks types when the flow loads,
so most wiring mistakes fail before anything runs, and checks the rest when values arrive. This
page explains the types you can write and where they go.

## Where types go

A type goes in one of two places:

| Place | How you write it | Example |
|---|---|---|
| A type definition: a `typedefs:` entry, or the `output:` of an `agent`, `code` or `tool` node | As plain text | `output: Optional(str)` |
| A slot, a place that holds a value: a flow input, a node input, a value entry, a flow output | As a YAML tag before the value | `qty: !int ${size.output}` |

A tag on a type definition is a load error, and so is a type without a tag on a flow input. See
[YAML tags](#yaml-tags) for how to write one.

## Scalars

| Type | Values |
|---|---|
| `str` | Text |
| `int` | A whole number of any size; [code](/code-and-tools/) gets one past 2^53 as a `bigint` |
| `float` | A number; an int fits too |
| `bool` | `true` or `false` |
| `date` | Text in the form `2026-10-14` |
| `datetime` | Text in the form `2026-10-14T10:30:00Z`; seconds, a fraction and the offset are optional, and a space may replace the `T` |
| `object` | Any JSON object: a mapping with text keys |
| `Any` | Any JSON value except null |
| `None` | Only null |
| `file` | A file reference (see [Files](#files)) |

`int` and `float` may carry an inclusive range. Either end may be left open:

| Type | Values |
|---|---|
| `int(1..12)` | 1 to 12 |
| `int(0..)` | 0 or more |
| `int(..100)` | 100 or less |
| `float(0.0..1.0)` | 0.0 to 1.0 |

Write `Any`, not `any`; `none`, `List` and `Dict` also work, for `None`, `list` and `dict`. An
unknown name is a load error that suggests a close match.

## Containers

| Type | Values |
|---|---|
| `list(X)` | A list of `X` |
| `Optional(X)` | An `X`, or null |
| `Literal(a, b, c)` | One of these texts, written as names or quoted strings, not numbers |
| `Union(A, B)` | A value of one of these records (see [Unions](#unions)) |
| `dict(K, V)` | The same as `object`; `K` and `V` must name known types |
| `Inbox(X)` | A handle on an inbox of `X` events (see [Inboxes and waiting](/inboxes/)) |
| `FileMemory` | A memory, written only as a flow input's tag (see [Memory](/memory/)) |
| `VectorDBMemory(T)` | A memory of items whose fields are the record `T`, or text only when written `VectorDBMemory`; a flow input's tag only (see [Memory](/memory/#vectordbmemory)) |
| `Message` | A chat message, built in (see [Agents](/agents/)) |

Types use parentheses, not brackets: `Optional[str]` is a load error.

`list(X)` holds no null unless `X` allows it: write `list(Optional(int))` for a list that may
hold null. A `list(int)` fits where a `list(Optional(int))` is expected, not the other way round.

## Records and typedefs

Name types under `typedefs:`. A typedef is a mapping of field names to types (a record), or any
other type (an alias):

```yaml
id: signals
name: signals
typedefs:
  Ticker: str                          # an alias
  Basket: list(Ticker)                 # aliases compose
  Decision: Literal(go, no_go, wait)   # one of these texts
  Signal:                              # a record
    score: float
    note: Optional(str)
input:
  basket: !Basket [AAPL, MSFT]
  signal: !Signal {score: 0.8}
nodes:
  count: !int ${len(input.basket)}
output: ${input.signal.score}
```

- Records are closed: a value with a field the record doesn't declare fails the check. An
  agent's record reply is the exception: it drops such a field (see
  [Structured output](/agents/#structured-output)).
- Every field is required, except an `Optional(...)` field, which may be left out.
- A tag never fills a field in. A `Buy` with `action: Literal(buy)` must still write
  `action: buy`.
- A record written inside `{...}` splits at each comma, so quote a type with a comma in it:
  `Msg: {role: "Literal(user, assistant)", text: str}`, or write the record in block style.
- A record can't contain itself, directly or through other typedefs.
- A typedef can't take a built-in name, such as `str`, `Optional`, `Message` or the memory
  types `FileMemory` and `VectorDBMemory`, and can't name a memory type.
- Every typedef is checked when the flow loads, even one nothing uses.

`Literal` values are text. One `Literal` fits where another is expected when its values are a
subset.

## Unions

`Union(A, B, ...)` is a value that is one of several records. yian must be able to tell the
members apart from the data alone, in one of two ways:

- **A tag field.** Every member has the same field as a one-value `Literal`, with a different
  value in each, as `action` below.
- **Shapes that can't overlap**, such as `{text: str}` and `{num: int}`.

```yaml
id: union-demo
name: union demo
typedefs:
  Buy:
    action: Literal(buy)
    qty: int
  Sell:
    action: Literal(sell)
    price: float
  Order: Union(Buy, Sell)
input:
  order: !Order
nodes:
  action: !str ${input.order.action}
output: ${action.output}
```

A field every member has, such as `action`, can be read anywhere. A field only some members
have, such as `qty`, needs [narrowing](#narrowing) first.

| Write | Load error |
|---|---|
| `Buy \| Sell` | `use Union(Buy, Sell) (\| is not a type operator)` |
| `Union(Buy, None)` | `None is not a union member ...; use Optional(Union(...))` |
| `Union(Buy, str)` | `member 'str' is not a record` |
| Two members that can match one value | `members 'A' and 'B' can match the same value; add a tag field ...` |

For a union that may be null, write `Optional(Order)`. When a run checks a value, a missing tag
field fails with `missing tag field 'action' (expected one of ["buy","sell"])`.

## Files

A `file` is a reference to a file, with three fields:

| Field | Type |
|---|---|
| `uri` | `str` |
| `mime` | `Optional(str)` |
| `name` | `Optional(str)` |

Read them as `${f.uri}`, `${f.mime}` and `${f.name}`; any other field is a load error. Where a
`file` is expected, a JSON mapping with exactly the keys `uri`, `mime` and `name` is read as a
file, so you can pass one in an `--inputs` file. Write null for an unknown `mime` or `name`; a
mapping that leaves a key out isn't a file.

## YAML tags

A slot's type is a YAML tag: `!`, then the type, then the value.

```yaml
id: tags-demo
name: tags demo
typedefs:
  Size: Literal(small, large)
input:
  text: !str                         # a required input: a tag and no value
  rounds: !int(1..5) 3               # a default
nodes:
  size: !Size small                  # a literal: checked at load
  count: !int ${input.rounds}        # an expression
  order: !object                     # a record, with tags inside
    qty: !int ${input.rounds}
    note: !str ""
output: !Size ${size.output}
```

- A tag ends at the first space and can't hold `,`, `[`, `]`, `{` or `}`. Name a type with
  several arguments in `typedefs:` and tag with its name: `Order: Union(Buy, Sell)`, then
  `!Order`. One-argument types work inline: `!Optional(list(str))`, `!Literal(draft)`.
- Inside `{...}` or `[...]`, a `,` ends the tag: `{qty: !int 5, note: !str ""}` works.
- A tag with no value is allowed only on a flow input. A binding needs a value:
  `w: !int ${...}`.
- `output: !T {...}` is one value of type `T`. Without the tag, `output: {...}` is a record of
  flow outputs.
- `!include` is the one tag that isn't a type: it reads a text value from a file. Where a tag
  goes, `!include(T) PATH` stands for `!T` with the file's value: for `!include(str)` its text
  as written, for any other `T` the YAML value it holds (see
  [Prompts in files](/agents/#prompts-in-files)).

### What a tag checks

A tag on a literal, or the value an `!include(T)` reads, is checked when the flow loads, using
YAML's rules for unquoted values:

| Unquoted value | Reads as |
|---|---|
| `true`, `false` | A boolean |
| `yes`, `no`, `on`, `off` | Text |
| `null`, `~`, nothing | null |
| `17`, `017`, `0x1F`, `0o17` | An int (17, 17, 31, 15) |
| `1.5`, `.5`, `1e3`, `.inf` | A float |
| `1_000`, `1:30` | Text |
| `2026-10-14` | Text, checked as a date when the slot is a `date` |
| `2026-10-14 10:30:00` | Text, checked as a datetime when the slot is a `datetime` |

An unquoted date or time must be a real one. Other forms YAML reads as a time, such as
`2026-1-5 9:00:00`, fail with a message to quote them.

So `!str 30` fails with ``quote it to keep it as text: `v: !str "30"` ``, `!bool yes` fails
with `` `yes` is text; write `true` ``, and `!int "5"` fails because quoted text stays text.

A tag on a `${...}` compares the type yian infers for the value with the tag:

| The value's type | Result |
|---|---|
| Fits the tag | Loads |
| Can't fit, such as `str` for `!int` | Load error: `declared int, but the value is str` |
| Wider, such as `str` for `!Size`, or `Optional(int)` for `!int` | Loads; checked when the value arrives |
| Unknown, such as arithmetic or an untyped result | Loads; checked when the value arrives |

A failed check at run time fails the reading node with `binding.type`. A check never converts
a value: an int in a `float` slot stays an int.

A check that fails inside a list or record says where, here in the second leg of a flow output
tagged `!list(Leg)`: `flow output at [1].mode: "boat" is not one of Literal(car, train)`. Flow
inputs, node inputs and node outputs name the place the same way.

## Wiring between flows

When a value goes into a child flow's input (through `call`, `map` or `loop`), its type must fit
the input's type:

| Source | Target | Loads? |
|---|---|---|
| `list(int)` | `list(Optional(int))` | Yes |
| `Buy` record | `object` | Yes |
| `Literal(small)` | `Literal(small, large)` | Yes |
| `int(1..10)` | `int(0..100)` | Yes |
| `str` | `Literal(small, large)` | No |
| `Optional(str)` | `str` | No |
| `int(0..)` | `int(0..100)` | No |
| `object` or `Any` | `int` or a record | No |
| `Order` (a union) | `Buy` (a member) | No |

To pass a wider value anyway, tag the binding, which checks it when it runs (`size: !Size
${input.s}`), or narrow it first. A tagged loop seed relaxes only the `Optional` row: under
`acc: !int 0`, the body may return `Optional(int)`, and a returned `null` fails the next run of the
body. Any other row stays a load error for a loop seed, tagged or not (see
[Repeat with `loop`](/control-flow/#repeat-with-loop)).

## Narrowing

Narrowing proves that a value is a narrower type, so the reads after it can use that type.
Three checks narrow a value:

| Check | Proves |
|---|---|
| `isinstance(x, Buy)` | `x` is a `Buy` |
| `x.action == "buy"` | `x` is the member whose tag is `buy` |
| `x != null` | `x` isn't null |

`not`, `and` and `or` carry these through: in `x != null and x.qty > 0`, `x.qty` is read only
after the check. A check written after a read doesn't narrow it. In a condition, comparing a tag
field with a value no member has, such as `x.action == "bye"`, is a load error that suggests a
close match.

Where narrowing applies:

- **A `case` branch.** Nodes on a branch see what the branch proves. This is how you read a
  field only some union members have.
- **Conditions:** `asserts:`, `case` conditions, and a `loop`'s `while:` and `until:` narrow
  by every check before each read.
- **Null checks** narrow in any expression.
- **A tag** narrows by checking: `buy: !Buy ${input.order}` loads, and fails at run time when
  the order is a `Sell`.

```yaml title="order.yaml" check=runnable
id: order
name: order
typedefs:
  Buy:
    action: Literal(buy)
    qty: int
  Sell:
    action: Literal(sell)
    price: float
  Order: Union(Buy, Sell)
input:
  action: !str
  amount: !int
nodes:
  order:
    kind: code
    input:
      action: ${input.action}
      amount: ${input.amount}
    output: Order
    code: |
      if (inputs.action === "buy") return { action: "buy", qty: inputs.amount };
      return { action: "sell", price: Number(inputs.amount) / 100 };
  route:
    kind: case
    on: ${order.output.action}
    cases:
      - when: buy
        then: bought
      - when: sell
        then: sold
  bought: !str "buy ${order.output.qty}"
  sold: !str "sell at ${order.output.price}"
output: ${bought.output | sold.output}
```

```console
$ yian run order.yaml --input action=buy --input amount=3
buy 3
```

`bought` reads `qty`, which only `Buy` has. It loads because the `buy` branch routes it. The
same read outside a branch fails:

```text
reference ${order.output.qty} reads field 'qty', which only union members with action buy
have; read it inside a case on ${order.output.action}
```

## Inputs from the command line

The top-level flow's inputs may arrive as text, from `--input k=v`. yian converts text to the
input's type before the run starts:

| Input type | Text it converts |
|---|---|
| `int` | A whole number, such as `42` |
| `float` | A number, such as `2.5`, `1e3` or `-inf` (`inf` and `infinity` in any case) |
| `bool` | `true`, `yes`, `y`, `on`, `1` and `false`, `no`, `n`, `off`, `0`, in any case |
| `Any` | None: `--input x=3` gives the text `"3"` |

Text that doesn't convert fails the run: `--input approved=ture` gives
`input 'approved': "ture" does not match declared type bool`. Pass lists, records, and numbers
for an `Any` input in a JSON file with `--inputs`. Values passed to a child flow are never
converted.

## Limits

A value, a type and a YAML file each nest a limited number of levels. Environment variables
can change each limit within its bounds. See [Reference](/reference/).
