---
title: "Expressions"
description: "The ${...} expression language: operators, comprehensions, null rules and builtins."
---

Expressions compute small values inside a flow: they read inputs and results, do arithmetic,
compare, and format text. This page explains the `${...}` language. One grammar serves every
place you write an expression.

## An example

```yaml title="digest.yaml" check=runnable
id: digest
name: digest
input:
  names: !list(str) [ada, grace, alan]
  limit: !Optional(int)
nodes:
  shown: !list(str) "${[upper(n) for n in input.names[:(input.limit :- 2)]]}"
  digest: !str "${join(shown.output, ', ')} and ${len(input.names) - len(shown.output)} more"
output: ${digest.output}
```

```console
$ yian run digest.yaml --input limit=1
ADA and 2 more
```

`shown` takes the first `limit` names (2 when `limit` is omitted) and upper-cases each one.
`digest` joins them and counts the rest.

## Where expressions go

| Place | How you write it | What yian does with it |
|---|---|---|
| Values: a node's `input:`, a value entry, a flow's `output:` | `${...}`, alone or inside text | Computes a typed value |
| Conditions: `asserts:`, `case` conditions, a `loop`'s `while:` or `until:` | A bare expression, such as `qty > 0` | Tests it as true or false |
| Prompts: an agent's `prompt:` and `system_prompt:` | `${...}` inside text | Writes each value into the text |
| Model settings: values in `llm_config:` | `${...}` | Computes the setting (see [Models and providers](/models/)) |

A condition may also be written as one whole `${...}`: `when: ${qty > 0}` means the same as
`when: qty > 0`.

### Values and text

A value that is exactly one `${...}` gives a typed value: `${input.qty}` is an int, and
`${[1, 2]}` is a list. A `${...}` inside other text is written into the text:

| The value | Written as |
|---|---|
| Text | As is |
| null | Nothing (in a prompt, an error; see [Null rules](#null-rules)) |
| A number, boolean, list or record | Compact JSON, such as `[1,2]` or `{"a":1}` |

Write `$$` for a literal `$`: `"costs $$5"` is `costs $5`. Text outside `${...}` is never parsed,
so `(a|b)` in a prompt stays as written.

You can't put a `${...}` inside an expression. Name values directly and group with parentheses:
write `${input.a :- input.b}`, not `${input.a :- ${input.b}}`. The load error shows the rewrite.

### Prompts

A prompt reads the node's own names, bare: the names in its `input:` block (and, for an agent,
its `variables:`). Bind a value in `input:`, then use it in the prompt:

<!-- yian-test {"replies": ["Bonjour, Ada !"]} -->
```yaml title="greet.yaml" check=runnable
id: greet
name: greet
input:
  person: !str
  language: !Optional(str)
nodes:
  greet:
    kind: agent
    input:
      name: ${input.person}
      language: ${input.language}
    output: str
    prompt: Greet ${name} in ${language :- "English"}.
output: ${greet.output}
```

```console
$ yian run greet.yaml --input person=Ada --input language=French
Bonjour, Ada !
```

- `${input.person}` in a prompt is a load error; bind it in `input:` first.
- A prompt span whose value is null fails the node. Give it a default, as with `language` above.
- A prompt may call `join`, `upper`, `lower` and `render_as_json`. Compute anything else, such as
  `len`, `isinstance` or a constructor, in an input.

## Literals

| Literal | Examples | Notes |
|---|---|---|
| Integer | `0`, `42`, `-3` | Any size |
| Float | `2.5`, `0.1` | Digits with a `.`; no exponent |
| Infinity | `inf`, `-inf` | A float |
| Text | `"hello"`, `'hello'` | No escapes; to include `"`, use `'...'` |
| Boolean | `true`, `false` | |
| Null | `null`, `none` | |
| List | `[1, "a", input.x]` | Items are expressions |

There is no record literal. Write a record as a YAML mapping (`!object {a: 1}`) or with a
[constructor](#types-in-expressions).

## References

| Write | Reads |
|---|---|
| `input.qty` | The flow input `qty` |
| `total.output` | The result of node `total` |
| `total.output.amount` | A field of that result |
| `f.uri`, `f.mime`, `f.name` | A field of a [file](/types/#files) |
| `system.today`, `system.now` | Today's date and the current time, in UTC, read once per run |
| `system.run_id` | The run's id |
| `item` | The current item, inside a `map` (see [Control flow](/control-flow/)) |
| `reply.new_messages` | The messages an agent added (see [Agents](/agents/)) |
| `qty` | A node's own input, in its prompt and asserts |

A dot reads a field. When the loader knows a value's type, reading a field the type doesn't have
is a load error. `and`, `or`, `not` and `in` can't start a reference, but they can name a field:
`${record.output.in}`.

## Operators

From lowest to highest precedence:

| Operators | Meaning |
|---|---|
| `a \| b \| c` | The first value that isn't null |
| `a :- b`, `a :? "msg"` | A default; a required value (see [Null rules](#null-rules)) |
| `or` | Either is true |
| `and` | Both are true |
| `not` | Negation |
| `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in` | Comparisons |
| `+`, `-` | Add, subtract; join two texts or two lists |
| `*`, `/`, `%` | Multiply, divide, remainder |
| `-x` | Negative |
| `**` | Power, grouped from the right |
| `xs[a:b]`, `f(x)`, `x.field` | Slice, call, field |

Use parentheses to group. Comparisons don't chain: write `0 < x and x < 10`, not `0 < x < 10`.

### Arithmetic

- Two ints give an exact int, however large: `${2 ** 100}` is
  `1267650600228229401496703205376`.
- `/` always gives a float: `${7 / 2}` is `3.5`, and `${4 / 2}` is a float too.
- `%` takes the sign of the left side: `${-7 % 3}` is `-1`.
- A float on either side gives a float: `${1 + 1.5}` is `2.5`.
- `+` joins two texts (`${"a" + "b"}`) or two lists (`${xs + [item]}`).
- A boolean counts as 0 or 1: `${true + 1}` is `2`.
- Dividing by zero fails, and so does arithmetic on null or on kinds that don't mix, such as
  text times a number. `%` doesn't format text; use a `${...}` inside text instead.

### Comparisons

- `==` compares numbers by value (`1 == 1.0` is true), and texts, lists and records by content.
  Values of different kinds are unequal: `"1" == 1` is false.
- `<`, `<=`, `>` and `>=` compare two numbers or two texts. A null on either side gives false.
  A date is text such as `2026-10-14`, so two dates compare in calendar order.
- `in` tests for a substring (`"ell" in "hello"`), a list item (`3 in [1, 2, 3]`), or a record
  key (`"a" in settings.output`).

### Conditions and truth

`and`, `or` and `not` test whether each value is true, and give `true` or `false`. These
values count as false: `false`, `null`, `0`, `0.0`, `""`, `[]` and a record with no fields.
Everything else counts as true. `and` and `or` stop as soon as the answer is known, so in
`x != null and x.qty > 0` the field is read only when `x` isn't null.

## Slices and comprehensions

`xs[a:b]` takes the items of a list from index `a` up to, not including, `b`:

| Write | Gives, for `[1, 2, 3, 4, 5]` |
|---|---|
| `xs[1:3]` | `[2, 3]` |
| `xs[2:]` | `[3, 4, 5]` |
| `xs[:-2]` | `[1, 2, 3]` |
| `xs[-2:]` | `[4, 5]` |

A negative index counts from the end, and an index past either end stops there. There is no
single index (`xs[0]`) and no step. Inside the brackets a `:` ends a bound, so a default needs
parentheses: `xs[(n :- 0):]`.

A list comprehension builds a list from another:

```yaml check=fragment
names: ${[m.text for m in messages if m.role == "user"]}
flat: ${[x for row in rows for x in row]}
```

Add more `for` and `if` clauses after the first; the leftmost `for` is the outer loop. A variable
is local to the comprehension. It can't be named `input`, `output`, `system` or `item`.
Parenthesize a list or condition that uses `:-`, `:?` or `|`: `${[m for m in (ms :- [])]}`.

Quote a YAML value that holds a slice or a comprehension, as in the first example, since YAML
reads `[` and `:` itself.

## Null rules

A value may be null: an optional input that was omitted, a node a `case` skipped, or a field
that holds null. These forms handle it:

| Write | Gives |
|---|---|
| `x?.name` | `x.name`, or null when `x` is null |
| `x :- "none"` | `x`, or `"none"` when `x` is null |
| `a \| b \| "none"` | The first of `a`, `b` that isn't null, else `"none"` |
| `x :? "x is required"` | `x`, or the run fails with the message `x is required` |
| `x != null and x.name == "Ada"` | `true` when `x` isn't null and its name is Ada; `x.name` is read only after the check |

A `.` read on a value whose type allows null is a load error: write `x?.name`, or check
`x != null` first. A `?.` makes the rest of the chain null too: `${x?.a.b}` is null when `x` is.

The default after `:-` is an expression, so quote text: `${input.when :- "today"}`. A bare
`today` would read a value named `today`. Defaults also cover a skipped node: in
`${(order.output.qty :- 0) + 1}`, a skipped `order` gives `1`.

## Builtins

| Builtin | Gives |
|---|---|
| `len(x)` | The number of items in a list, or of characters in a text |
| `upper(s)`, `lower(s)` | The text in upper or lower case |
| `join(list, sep="\n")` | The items as text, joined by `sep` |
| `render_as_json(value, indent=2)` | The value as indented JSON text |

Arguments may be given by name: `${join(items, sep=", ")}`. A null argument fails the call, so
give a default: `${upper(name :- "")}`. Calling a builtin with arguments it doesn't take, or a
name that isn't a builtin, is a load error. A flow isn't a function: to run a child flow from a
value, use the `call(...)` form (see [Control flow](/control-flow/)).

`len` gives an int and the others give text, so `${upper(name).x}` is a load error.

## Types in expressions

YAML tags can't go inside `${...}`, so two forms work with [types](/types/) there:

| Write | Gives |
|---|---|
| `isinstance(x, Buy)` | Whether `x` is a valid `Buy` |
| `Buy(x)` | `x`, checked as a `Buy`, like `!Buy ${x}` |
| `Buy(action="buy", qty=2)` | A `Buy` record with exactly those fields, checked |

`Buy` is a name from `typedefs:`. A constructor that can never succeed, such as a text where
the type wants an int, is a load error; others are checked when they run.

`isinstance` and field checks also narrow a value for the reads after them; see
[Narrowing](/types/#narrowing).

## Limits

An expression nests at most 100 levels: each call, list, comparison, default and chain of
operators is one level. A deeper one is a load error; split it across values. See
[Reference](/reference/) for every limit.
