---
title: "Expressions"
description: "The ${...} expression language: operators, conditionals, indexes, comprehensions, null rules and builtins."
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
| Model settings: values in `llm_config:` | `${...}`, with no `isinstance` or constructor | Computes the setting (see [Models and providers](/models/)) |

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
- A prompt may call the builtins that give text: `join`, `upper`, `lower`, `render_as_json`,
  `str`, `strip`, `lstrip`, `rstrip` and `replace`. Compute anything else, such as `len`, `split`,
  `isinstance` or a constructor, in an input.

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
| `draft.new_messages` | The messages agent `draft` added (see [Agents](/agents/)) |
| `qty` | A node's own input, in its prompt and asserts |
| `output`, `output.amount` | A node's own result, in its asserts |

A dot reads a field. When the loader knows a value's type, reading a field the type doesn't have
is a load error. `and`, `or`, `not` and `in` can't start a reference, but they can name a field:
`${record.output.in}`.

## Operators

From lowest to highest precedence:

| Operators | Meaning |
|---|---|
| `a if c else b` | `a` when `c` is true, else `b` (see [Conditionals](#conditionals)) |
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
| `xs[i]`, `xs[a:b]`, `f(x)`, `x.field` | Index, slice, call, field |

Use parentheses to group. Comparisons don't chain: write `0 < x and x < 10`, not `0 < x < 10`.

### Arithmetic

- Two ints give an exact int, however large: `${2 ** 100}` is
  `1267650600228229401496703205376`.
- `/` always gives a float: `${7 / 2}` is `3.5`, and `${4 / 2}` is a float too.
- Put spaces around `/` between names: `n/2` reads as one name, a load error that shows `n / 2`.
- `%` takes the sign of the left side: `${-7 % 3}` is `-1`.
- A float on either side gives a float: `${1 + 1.5}` is `2.5`.
- `+` joins two texts (`${"a" + "b"}`) or two lists (`${xs + [item]}`).
- A boolean counts as 0 or 1: `${true + 1}` is `2`.
- Dividing by zero fails, and so does arithmetic on null or on kinds that don't mix, such as
  text times a number. `%` doesn't format text; use a `${...}` inside text instead.

### Comparisons

- `==` compares numbers by value (`1 == 1.0` is true), and texts, lists and records by content.
  A boolean counts as 0 or 1 (`true == 1` is true); other values of different kinds are
  unequal: `"1" == 1` is false.
- `<`, `<=`, `>` and `>=` compare two numbers or two texts. A null on either side gives false.
  A date is text such as `2026-10-14`, so two dates compare in calendar order.
- `in` tests for a substring (`"ell" in "hello"`), a list item (`3 in [1, 2, 3]`), or a record
  key (`"a" in settings.output`).

### Conditions and truth

`and`, `or` and `not` test whether each value is true, and give `true` or `false`. These
values count as false: `false`, `null`, `0`, `0.0`, `""`, `[]` and a record with no fields.
Everything else counts as true. `and` and `or` stop as soon as the answer is known, so in
`x != null and x.qty > 0` the field is read only when `x` isn't null.

### Conditionals

`a if c else b` gives `a` when `c` is true and `b` otherwise, testing `c` as `and` and `or` test
their values. Only the branch it picks runs, so in `x.qty if x != null else 0` the field is read
only when `x` isn't null. It has the lowest precedence: `x :- y if c else z` means
`(x :- y) if c else z`, and `a if c else b if d else e` means `a if c else (b if d else e)`.
A test that is itself a conditional needs parentheses: `a if (b if c else d) else e`.

In a comprehension, `[a if c else b for x in xs]` computes each item, and an `if` after a `for`
stays a filter.

## Indexes, slices and comprehensions

`xs[i]` reads one item of a list, and `xs[a:b]` takes the items from index `a` up to, not
including, `b`:

| Write | Gives, for `[1, 2, 3, 4, 5]` |
|---|---|
| `xs[0]` | `1` |
| `xs[-1]` | `5` |
| `xs[1:3]` | `[2, 3]` |
| `xs[2:]` | `[3, 4, 5]` |
| `xs[:-2]` | `[1, 2, 3]` |

A negative index counts from the end. An index out of range fails the run with an error naming
the index and the length, unless `:-`, `|` or `:?` handles it: `${xs[0] :- "none"}`. A slice bound
past either end stops there. There is no step. Inside the brackets a `:` ends a bound, so a
default there needs parentheses: `xs[(n :- 0):]`, `xs[(i :- 0)]`.

Text works the same way, by character: for `s` = `"héllo"`, `s[1]` is `"é"` and `s[-3:]` is
`"llo"`. A record has no index; read its fields with a dot. A dot may follow a subscript or
parentheses: `xs[0].name`, `xs[0]?.name`, `(a if c else b).name`.

A list comprehension builds a list from another:

```yaml check=fragment
names: ${[m.text for m in messages if m.role == "user"]}
flat: ${[x for row in rows for x in row]}
```

Add more `for` and `if` clauses after the first; the leftmost `for` is the outer loop. A variable
is local to the comprehension. It can't be a keyword, `input`, `output`, `system` or `item`,
start with `__`, or reuse the name of an enclosing comprehension's variable.
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

A `.` read is a load error on a value whose type allows null, and right after a node that may
be skipped, whatever its type: write `x?.name`, or check `x != null` first. A `?.` makes the rest of the chain null too: `${x?.a.b}` is null when `x` is.

The default after `:-` is an expression, so quote text: `${input.when :- "today"}`. A bare
`today` would read a value named `today`. Defaults also cover a skipped node: in
`${(order.output?.qty :- 0) + 1}`, a skipped `order` gives `1`.

## Builtins

| Builtin | Gives |
|---|---|
| `len(obj)` | The number of items in a list, or of characters in a text |
| `upper(s)`, `lower(s)` | The text in upper or lower case |
| `join(value, sep="\n")` | The items of a list as text, joined by `sep` |
| `render_as_json(value, indent=2)` | The value as indented JSON text; an infinite float fails, as JSON has no infinity |
| `startswith(s, prefix)`, `endswith(s, suffix)` | Whether the text starts or ends with the other |
| `split(s, sep=null)` | A list of the parts of `s` between each `sep`; with no `sep`, the words between runs of whitespace |
| `strip(s, chars=null)`, `lstrip(...)`, `rstrip(...)` | The text without whitespace, or the characters in `chars`, at both ends, the start or the end |
| `replace(s, old, new)` | The text with every `old` replaced by `new` |
| `str(x)` | The text `${x}` writes into text, such as `[1,2]` for a list, `1` for `1.0` and `true` for `true`; not Python's `str` |
| `int(x)` | An int from text such as `" -42 "` or `"1_000"`, a float cut toward zero, or a boolean as 0 or 1 |
| `float(x)` | A float from text such as `"2.5"`, `"1e3"` or `"inf"`, or from an int or a boolean |
| `round(x, n=null)` | `x` rounded to an int, or to `n` decimal places, ties to even |
| `abs(x)` | The number without its sign |
| `min(xs)`, `max(xs)` | The least or greatest of a list of numbers, or of texts |
| `sum(xs)` | The sum of a list of numbers; `0` for an empty list |
| `any(xs)`, `all(xs)` | Whether any, or every, item of a list is true |

Arguments may be given by the names in the table: `${join(items, sep=", ")}`. A null argument
fails the call, so give a default: `${upper(name :- "")}`. A `null` for an argument whose default
is `null` is the same as leaving it out. Calling a builtin with arguments it doesn't take, a name
that isn't a builtin, or an argument whose type the builtin never takes (`upper` of an int) is a
load error. A flow isn't a function: to run a child flow from a value, use the `call(...)` form
(see [Control flow](/control-flow/)).

Where Python has the same function, the builtin gives Python's result:

- Text is counted and split by character (code point), as `len` counts it.
- Whitespace is Python's set: U+0009 to U+000D, U+001C to U+001F, U+0020, U+0085, U+00A0,
  U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F and U+3000. `int` and `float` skip it
  around a number, all but U+001C to U+001F, as Python does.
- `int` and `float` read Python's syntax with ASCII digits only. `float("nan")` fails, since a
  value is never NaN, and so does `float` of an int too large for a float.
- `round` works on the float's exact value: `round(2.5)` is `2`, `round(2.675, 2)` is `2.67`.
  `round(x)` gives an int; `round(x, n)` a float for a float `x` and an int for an int.
- `sum` keeps ints exact and adds floats as Python 3.12 does: `sum([0.1, 0.2, 0.3])` is `0.6`.
- `min` and `max` compare texts by code point, and give the first of equal items.

The result types follow the table: `len` and `int` give an int, `float` a float, `round`, `abs`
and `sum` an int or a float as above, `startswith`, `endswith`, `any` and `all` a boolean,
`split` a `list(str)`, `min` and `max` an item of their list, and the others text, so
`${upper(name).x}` is a load error.

## Types in expressions

YAML tags can't go inside `${...}`, so two forms work with [types](/types/) there:

| Write | Gives |
|---|---|
| `isinstance(x, Buy)` | Whether `x` is a valid `Buy` |
| `Buy(x)` | `x`, checked as a `Buy`, like `!Buy ${x}` |
| `Buy(action="buy", qty=2)` | A `Buy` record with exactly those fields, checked |

`Buy` is a name from `typedefs:`; `isinstance` also takes a built-in type, as in
`isinstance(v, int)`. A constructor that can never succeed, such as a text where
the type wants an int, is a load error; others are checked when they run.

`isinstance` and field checks also narrow a value for the reads after them; see
[Narrowing](/types/#narrowing).

## Limits

An expression nests at most 100 levels: each call, list, comprehension, index, slice,
conditional, comparison, unary operator, `:-`, `:?` or `|`, and chain of operators is one level.
A deeper one is a load error; split it across values. See [Reference](/reference/) for every
limit.
