# Expressions — the one `${...}` grammar

> The `expr` package (`grammar.py` + `expressions.py` + `template.py` + `builtins.py`).
> It sits above [`typesys`](typing.md) and below `nodes`, and is **pool-agnostic**: every
> layer takes a `resolve` callable, never the pool itself. For what you may *write* inside
> `${...}`, see [Flow syntax → References](syntax.md#references-naming-a-value).

## One grammar, three readers

Everywhere a flow wires data—a node input, a `when:` condition, a flow `output:`,
or an AGENT node's task `prompt:`—the author writes `${...}`. There is **one** grammar, and the same parse
tree is read three ways:

- **evaluate** it against a `resolve` callable → a value (at run time), and
- **infer** a result type when the AST proves one → a `Type` (at load time), and
- **ref-walk** it → the set of references it reads (at load time, to infer edges).

That single source of truth is why the graph the compiler infers matches exactly what the
runtime evaluates. Three stacked layers plus a builtin registry:

```
   grammar.py     PARSE ONLY — one Lark grammar → a parse tree          (parse_expr)
   expressions.py evaluate / infer a tree via injected resolvers        (eval_expr,
                  infer_expr_type) + ref-walk/rewrite                    (expr_refs)
   template.py    the ${...} scanner: split text into literals + spans  (eval_binding,
                  drive the engine per span; strict prompt rendering     render_template_record)
   builtins.py    TEMPLATE_FNS — the pure builtins any ${...} expression may call
```

## Layer 1 — grammar: parse only

`parse_expr(text)` builds a parse tree and does nothing else. One grammar parses every
construct:

| Form | Example |
|------|---------|
| reference | `a`, `a.b.c`, `node#0.output` (graph-expansion segments) |
| arithmetic / comparison | `a + 1`, `x < 5`, `x in [1, 2]` |
| boolean | `a and b`, `not a` |
| list literal | `[1, "x", a]` |
| builtin call | `upper(name)`, `join(items, ", ")` |
| coalesce / default / required | `a \| b \| c`, `a :- d`, `a :? "msg"` |
| type test | `isinstance(a, Buy)` |

Two load-bearing points: one shared `NAME` terminal with an optional call-suffix (a naive
separate-ref-plus-call grammar won't parse under LALR); and `#`/`/` are in `NAME` (runtime
ids like `node#0`) while `-` is not (so `a - b` is subtraction).

Changed (TS yian): `ts/src/expr/grammar.ts` is a hand-written parser, not Lark, and builds
the same tree (`Tree` with `data` and `children`, `Token` with `type` and `value`). A
number's digits are ASCII `0`–`9`, as a name's letters already were; Python's also took
other scripts' digits. Whitespace inside `not in` follows JavaScript's `\s`, which differs
from Python's only on rare control and format characters. A parse error still starts
`could not parse expression '<text>': `, but the rest is TS's own text naming what it found,
at a 1-based line and a UTF-16 column: `unexpected ')' at line 1, column 2`.

Changed (TS yian): **list slices.** A postfix subscript `x[lower:upper]` follows any operand: a
reference, a call result, parentheses, a list literal or another subscript (`xs[1:][:2]`).
Either bound may be omitted (`xs[1:]`, `xs[:-1]`, `xs[:]`). Postfix binds tightest, as Python's
primary does: `-xs[1:]` parses as `-(xs[1:])` and `xs[1:] ** 2` as `(xs[1:]) ** 2`. The tree is
`subscript[base, slice[lower, upper]]`, with an omitted bound as `omitted[]`.

The colon rule: a `:` directly inside a subscript's brackets ends a bound, and the lexer tries it
before `:-` and `:?`. Pending operators don't hide the brackets, so `xs[:-1]`, `xs[a:-b]`,
`xs[a :- b]` and `xs[n - 1:-1]` are all Python slices. Inside parentheses, a call or a list
within a bound, the escapes work as before, so a default in a bound needs parentheses:
`xs[(n :- 0):]`. These are parse errors, each naming the fix:

| Text | Message |
|---|---|
| `xs[a:?b]` | ``inside a slice, `:` ends a bound; write `(a :? b)` for a required value`` |
| `xs[::2]`, `xs[1:2:1]`, `xs[n :- 0:]` | ``slices take no step; for a default in a bound, use parentheses: `xs[(n :- 0):]` `` |
| `xs[0]` | ``indexing `xs[i]` is not supported; `xs[i:i + 1]` gives a one-item list`` |
| `xs[1:].a` | `unexpected '.'`: a list has no fields |

Changed (TS yian): **list comprehensions.** `[elem for x in xs]`, then any mix of `for y in ys`
and `if cond` clauses, as in Python: `[m.text for t in history for m in t if m.role == "user"]`.
The leftmost `for` is the outermost loop. A target is one bare name (yian has no tuples). The
elem is a full expression, escapes included. Each iterable and each condition is Python's
`or_test`, the `or` level and above, so an escape there needs parentheses:
`[m for m in (input.history :- [])]`. The tree is `list_comp[elem, clause, ...]` with clauses in
source order, each `comp_for[NAME, iterable, SOURCE]` (`SOURCE` keeps the iterable's text for
messages) or `comp_if[cond]`.

`for` and `if` are keywords only where an operator is expected, right after a complete operand,
at any bracket depth: in a list's brackets `for` after the first element starts a comprehension,
and `if` after an iterable or a condition adds a filter. Where an operand is expected they stay
names, so `[for, if]` is still a two-item list.

Scoping is lexical, as in Python. A variable is in scope in the elem and in every clause after
its own `for`, nested comprehensions included; the first iterable runs in the enclosing scope, so
`[x for x in x.items]` iterates the outer `x`. Inside the comprehension the variable wins over
any outer name: it may share a name with a node, an input, a typedef, a LOOP field or a
builtin, and a call such as `len(...)` or `Msg(...)` is never a variable read. Every parse ends
with a pass that turns each variable read into a `comp_var` node, so reference walks (edges,
name checks, guards, CASE renames) skip variables and see only the outer reads. These are parse
errors, each naming the fix:

| Text | Message |
|---|---|
| `[a, b for x in xs]` | ``a comprehension has one element before `for`; write `[[a, b] for x in xs]` `` |
| `(x for x in xs)`, `f(x for x in xs)` | `` `for` makes a list comprehension only: write `[x for x in xs]` `` |
| `[x if c else y for x in xs]`, `(a if c else b)` | ``conditional expressions (`a if c else b`) aren't supported yet`` |
| `[x for 1 in xs]`, `[x for in in xs]`, `[x for a#0 in xs]` | ``a comprehension's `for` takes a name, then `in` `` |
| `[x for x in a :- []]`, `[x for x in xs if a \| b]` | ``a comprehension's iterable and `if` stop before `:-`, `:?` and `\|`; write `(a :- [])` `` |
| `[x for item in xs]`, `[x for __x in xs]` | `'item' can't be a comprehension variable: input, system, item, output and names starting with __ are reserved` |
| `[x for x in a for x in b]`, `[[x for x in a] for x in b]` | `comprehension variable 'x' is already bound here; rename one` |

The reserved names are the keywords, `input`, `system`, `item`, `output` and names starting with
`__` (CASE's generated locals use that prefix). `[x for x in [x for x in a]]` is allowed: the
inner comprehension sits in the first iterable, which runs in the enclosing scope.

`and`, `or`, `not`, and `in` are reserved operators and cannot be the exact root
name of a reference. They remain legal as dotted field names, such as
`record.output.in`.

Changed (TS yian): `inf` is a reserved constant for infinity, like `true` and `null`, not
a reference; `-inf` is unary minus on it. After a dot it stays a field name (`x.inf`). It
is an expression keyword, so a node can't be named `inf` (see
[Flow syntax](syntax.md#the-flow-shape)).

A chain of same-precedence operators is one flat node, like an `and` or `or` chain:
`a + b - c` is one `sum`, `a * b / c % d` one `product`, and `a ** b ** c` one `power`.
The evaluator applies a `sum` or `product` left to right and a `power` right to left,
so the results are Python's and a chain of any length needs no recursion per operator.

### Expression nesting limit

An expression nests at most 100 levels, the same bound as a
[value](typing.md#nesting-limit), so `YIAN_MAX_VALUE_NESTING` sets both (see
[Defaults and limits](defaults.md)). Each call, list literal, unary operator
(`-`, `not`), comparison, escape (`|`, `:-`, `:?`), `and`/`or` chain, and operator chain is
one level (in TS yian, so is a subscript, like a call, and a comprehension, like a list; its
clauses and variable reads add none); a plain reference adds none, and
parentheses add none of their own. Mixing
precedence nests: `a * b + c` is a `product` inside a `sum`, two levels, and
`((a + b) + c) + d` is three chains, three levels. `-1` is one level. `parse_expr` checks
the depth first, without recursion, so no later walk meets a tree deep enough to
overflow it, and a deeper expression is a load error at its field:

```
expression nests deeper than 100 levels, the limit for a yian expression (each call, list, unary operator, comparison, and chain of operators is a level); split it into smaller bindings or steps
```

## Layer 2 — the one evaluator

`eval_expr(tree, resolve, item=None, mode=...)` is the single evaluator for every
construct. It is pool-agnostic — `resolve` turns a reference path into a value, so the same
evaluator serves the pool, a prompt's local record, and the map-body `item` scope.

**Three locked semantics** (`case` routing to `else:` depends on them):

1. a **missing reference is `None`** (non-strict modes),
2. **arithmetic over that `None` raises** — a missing operand never silently becomes `0`,
3. an **ordered comparison with `None` is `False`** — so an absent value routes to `else:`.

Changed (TS yian): comparisons. `==` and `!=` compare deeply across kinds: numbers by exact
value (`${2 ** 60 == 2.0 ** 60}` is `true`; a bool counts as 0 or 1), text by code units, lists
item by item, records by keys and values, files by `uri`, `mime` and `name`; any other pair is
unequal. Ordering takes only two numbers (by exact value, a bool as 0 or 1) or two texts (by
UTF-16 code units); any other pair, lists included (`${[1] < [2]}`), is an error naming the
operator and the operand kinds, and a null operand still gives `false`. `in` finds text in
text, an equal item in a list, or a text key in a record; any other right side, or a non-text
left side with text on the right, is an error.

When the `None` would come from a node a `case` may skip, Compose rejects the arithmetic at
load and asks for a `:-` default (see `syntax.md`, "Skipped branches and flow outputs").
In TS yian this looks through a slice: `${maybe.output[1:] + xs}` is refused like
`${maybe.output + xs}`.

Changed (TS yian): **slice semantics.** `xs[a:b]` on a list of length n follows Python's
`slice.indices` with step 1: an omitted `a` is 0 and an omitted `b` is n, a negative bound adds
n, both bounds clamp to `[0, n]`, and `a >= b` gives `[]`. Bounds are ints of any size, and a
bool counts as 0 or 1, as in arithmetic.

| Case | Binding / condition | Strict (prompt) |
|---|---|---|
| the base is `null` or misses | a miss, as `.field` on null is, so `${xs[1:] :- []}` falls back; the bounds aren't read | raises `unresolved reference ${xs}` |
| the base is text, a record or a number | raises `slicing takes a list, got str` | same |
| a bound is `null` or misses | raises `slice bound is null; omit it to mean the end` | same |
| a bound is a float or text | raises `slice bounds take int, got float` | same |

Changed (TS yian): **comprehension semantics.** The first iterable is evaluated in the
enclosing scope; each later clause runs once per binding of the clauses before it and sees
those variables. An `if` keeps a binding when its condition is truthy, and after the last clause
the elem is appended, so the result is a new list in iteration order. Variables are local:
`[[y for y in xs], y]` reads the outer `y` (a miss if there is none). An item is returned as
is, as in Python, so `[x for x in [null, 1]]` gives `[null, 1]`; `.field` on a null item misses,
as on any null. The clauses run in a loop, not by recursion, so any number of them evaluates.

| Case | Binding / condition | Strict (prompt) |
|---|---|---|
| the elem misses | a `null` item, as `[a, 1]` gives `[null, 1]` | raises `unresolved reference ${m.text} in expression` |
| a condition misses | `null`, falsy: the item is dropped | raises |
| an iterable is `null` or misses | raises ``comprehension `for m in input.history` iterates over null; give a default, e.g. `(input.history :- [])` `` | same |
| an iterable is text, a record or a number | raises ``comprehension `for c in name` takes a list, got str`` | same |

A comprehension never misses, so `${[...] :- []}` never falls back. A flow output that
iterates a node a `case` may skip, `${[x for x in maybe.output]}`, loads, and fails at run time
with the null-iterable error when `maybe` is skipped; write `(maybe.output :- [])`.

Quote a YAML value that holds a slice: in a plain scalar `xs[a: b]` (a space after the colon)
breaks the scalar, and in an unquoted flow mapping the `[` and `]` break it. Write
`q: "${xs[a: b]}"`.

In bindings, CASE conditions, LOOP predicates, flow outputs, prompts, and asserts the
loader rejects field reads a known type can't have, including a field only some members of
a union have outside a narrowing `case` branch (or, in a condition, without a check before
it that narrows it; see
[Flow syntax → Narrowing inside a condition](syntax.md#narrowing-inside-a-condition)), so a
`None` there is a null, skipped, or untyped value, not a misspelled field.

**`ResolveMode`** — how the caller wants a miss treated:

| Mode | A missing reference… | Used by |
|------|----------------------|---------|
| `BINDING_NONE` | becomes `None` (coalesce/default may fire) | binding values |
| `CONDITION_FALSY` | `None`, falsy through comparisons | `when:` / `asserts:` |
| `STRICT_RAISE` | raises, naming the path — no silent blank | prompt rendering |

In every mode, `|`, `:-`, and `:?` see a miss on a non-final operand as `None`, so
`${note :- "none"}` works in a prompt; the fallback (the last `|` operand) stays strict.
Strictness also applies to the span's final value: a prompt span that evaluates to
null raises before whole-value return or embedded concatenation. Bindings remain
non-strict, so an embedded null contributes an empty string.

**Two safety rules:** dotted access reads only a dict key or a file's `uri`, `mime`, or
`name`, never `getattr` (so `${x.__class__}` is just a missing key, on a file too);
builtin calls dispatch through `TEMPLATE_FNS` only (no arbitrary callable is reachable).
So `${doc.output.uri}` reads a file's `uri` in bindings, templates and prompts, `case`
`on:` and `when:`, and asserts. When the loader knows the value is a `file`, it checks
the read: `uri` is `str`, `mime` and `name` are `Optional(str)`, and any other name is a
load error (see [Flow syntax → References](syntax.md#references-naming-a-value)).

### Arithmetic follows Python

Operators run on the resolved *values* with Python semantics:

| Expression | Result | Rule |
|------------|--------|------|
| `${7 + 2}`, `${"a" + "b"}`, `${xs + [item]}` | `9`, `"ab"`, `xs` with `item` appended | `+` adds numbers and concatenates text and lists |
| `${"ab" * 3}`, `${[0] * 3}` | `"ababab"`, `[0, 0, 0]` | `*` repeats text and lists by an int |
| `${"%.2f" % price}`, `${"%(k)s" % record}` | `"3.14"`, the `k` field as text | `%` with text on the left is printf-style formatting |
| `${7 % 3}` | `1` | `%` with a number on the left is the remainder |
| `${true + 1}`, `${true > 0}`, `${true == 1}` | `2`, `true`, `true` | a bool counts as 0/1 in arithmetic, ordering and equality |

Equality in a CASE's `on:` form is stricter: a bool never equals a number there.

Changed (TS yian): the operators follow this table instead:

| Operands | Result |
|---|---|
| int `+`, `-`, `*`, `%` int; int `**` int ≥ 0 | an exact int of any size |
| `/` | always a float, the one nearest the exact quotient: `${10 ** 400 / 10 ** 399}` is `10.0` |
| `%` | the remainder takes the left operand's sign: `${-7 % 2}` is `-1` |
| text `%` anything | an error: ``text formatting with `%` was removed; use a template`` |
| text or list `*` int | an error: repetition is gone |
| an int with a float, int `**` negative int | the int becomes the nearest float, then float arithmetic |
| `/` or `%` by zero, `0 ** negative` | an error: `division by zero: 1 / 0` |

`**` with a finite non-zero base and an integral exponent `n` (an int, or a float with no
fraction) with `|n| ≤ 1100` is correctly rounded when its result is a float: `${10 ** -4}` is
`0.0001` and `${1.1 ** 10}` is `2.5937424601000023`. Any other float `**` is JavaScript's
`**`, whose last bit may differ between Bun and Node. Operand kinds an operator doesn't take
fail naming the operator and the kinds, with the operands in the text form:
``type error evaluating "ab" * 2: unsupported operand kinds for `*`: `str` and `int` ``.

There are no cost limits. A huge `**` or repetition takes whatever time and memory
Python needs.

Changed (TS yian): repetition is gone; a huge `**` or `*` takes whatever time and memory the
runtime needs, up to `integer result is too large for this runtime`.

Float `+`, `-`, `*` and `/` that overflow give `inf`, which the
[non-finite rule](#non-finite-results) catches at the binding. A result Python refuses to
compute fails like any other expression error:

- `**` overflow: `${2.0 ** 5000}`;
- an int too large to mix with a float: `${10 ** 400 * 1.0}`;
- a repetition count too large: `${"a" * 10 ** 20}`;
- a missing `%(key)s`: `${"%(k)s" % record}` when `record` has no `k`;
- a value holding an int longer than 4300 digits written as text: embedded in text
  (`"v=${10 ** 5000}"`), anywhere in a prompt, formatted with `%` (`${"%d" % big}`,
  any conversion), or passed to `join` or `render_as_json`. The message gives the
  digit count: `integer with 5001 digits is larger than yian allows (4300 digits)`.

An int longer than 4300 digits is not a value, so a whole-string `${10 ** 5000}`
fails its binding (see [Types → Integer size limit](typing.md#integer-size-limit)).
A temporary inside an expression may be that long: `${10 ** 5000 > 0}` is `true`. An
int literal longer than 4300 digits is a load error.

Changed (TS yian): none of the failures in this list exist. `**` overflow and an int past the
float range in float arithmetic give `inf` (see [Non-finite results](#non-finite-results));
repetition and `%` formatting are gone; and there is no digit limit, so an int of any size is a
value and becomes text in full, and int literals have no length limit. An int result too large
for the runtime's `bigint` fails instead: `integer result is too large for this runtime`
(`${10 ** 10 ** 10}`). A float literal too large for a float is a load error: `float literal
with 400 digits before the decimal point is out of range; write inf for infinity`.

The failure is `binding.evaluate` in a binding, `node.execute` at the CASE node in a
searched `when:`, `assertion.failed` in a node or flow `asserts:`, and `node.execute` at the
AGENT or HUMAN_INPUT node whose prompt can't render. (TS yian has no HUMAN_INPUT; an `ask`'s
`prompt:` is an input binding.)

### Non-finite results

NaN, infinity, and complex numbers are not yian values. Float `+`, `-`, `*` and `/`
that overflow give `inf` (`${input.x * 10}` with `x` = `1e308`; `**` and int-to-float
overflow fail instead, see [Arithmetic follows Python](#arithmetic-follows-python)), and
`**` with a negative base and a fractional exponent gives a complex number
(`${(0 - 8) ** 0.5}`). A binding whose final value holds NaN, infinity, or a complex
number at any depth fails its node with `binding.evaluate`, naming the input and path
(`input 'y' at a[1]: non-finite number inf is not a yian value`,
`input 'v': complex number (…) is not a yian value`). A temporary inside an expression
is fine: only the binding's final value is checked, so the embedded template
`"v=${x * 10}"` is the text `v=inf`. A non-finite literal written in an output without
a type (`g: .nan`) fails earlier, at load (see
[Flow syntax → YAML values](syntax.md#yaml-values)). Flows that used to pass an infinite value along
now fail. To fix one, keep the value in range before it reaches the binding (guard it
with a `case`, such as checking the base is not negative before `**`, or compute it in
a `code` node), or return null explicitly: a `case` that routes to a value entry
`no_value: !Optional(float) null`, or a `code` node that returns `None`.

Changed (TS yian): infinity is a float value, not an error. A float result past the float
range, `**` and an int converted to a float included, is `inf` (`${2.0 ** 5000}`). `**`
follows IEEE `pow`: `${1 ** inf}` is `1.0`. An operation with no numeric result fails, naming
the operator and the operands: `- on Infinity and Infinity has no numeric result` (also
`${inf * 0}` and `${(0 - 8) ** 0.5}`), so NaN never exists, and there are no complex numbers.

A list literal adds a level of nesting (`${[x]}` is one level deeper than `x`), and
a binding whose final value nests deeper than 100 levels fails its node
with `binding.evaluate` (see [Types → Nesting limit](typing.md#nesting-limit)).

**Compile-time companions** (never evaluate): `infer_expr_type(tree, ref_type_fn)` returns
a type only for statically fixed results: scalar literals, exact references,
comparisons/membership, `not`/`and`/`or`, and `isinstance`. Other expressions remain
unknown rather than borrowing an operand's type. `expr_refs(tree)` collects the reference
paths an expression reads — this is how the compiler infers data edges.
`rewrite_expr_refs` rewrites only the reference *leaves* by source position — how refs are
re-namespaced when a child flow is inlined.

**The condition surface** — `when:` / `until:` / `while:` / `asserts:` admit two
spellings that parse and evaluate identically: bare `a > 5` and whole-span `${a > 5}`.
Both route through this one path, so they can never drift. A `${...}` inside an
expression (`${a} > 5`, `${x :- ${y}}`) is rejected by `parse_expr` with an error that
shows the rewrite (`a > 5`, `x :- y`); `scan_template` reports a template span's whole
fixed span (`${x :- y}`).

## Layer 3 — the `${...}` scanner

A binding or prompt is text with `${...}` spans embedded in it. `scan_template(text)`
splits it into literal runs and spans, parsing **only** the span interiors — so operator
characters in free text stay literal (a prompt line `stance (positive|negative)` is not an
expression). `$$` escapes a literal `$`. Evaluation then follows one rule:

- a value that is **exactly one** `${...}` span resolves to the **typed** value (a float
  stays a float, an object a dict);
- a span **embedded** in surrounding text is **stringified** into it;
- text with no span is a plain literal.

Changed (TS yian): an embedded value is written in the text form: text as itself, `true`/`false`,
a number in JavaScript spelling (`1.0` is `1`, infinity is `Infinity`), and a list, record or file
as compact JSON (`[1,"a"]`, `{"k":"v"}`); record keys that look like array indexes (`"0"`, `"17"`)
come first, here and in `render_as_json`.

The same rule applies to every string nested in a record or list value of a slot: the
binder evaluates each nested string as its own template, so `{n: ${a.output}}` gives a
record whose `n` is the typed value. `$$` is the escape at any depth, and a nested
`call(...)` is a load error (see
[Flow syntax → Nested values](syntax.md#nested-values)).

Compile-time output typing mirrors this: a lone direct-reference span preserves the
producer's exact or partial contract (including provenance), and a known field selected
from a partial record is an exact typed passthrough. A statically proven computed span gets its result type
(`==`, membership, and `not`/`and`/`or` produce `bool`); and a concatenating template
(text around spans, or 2+ spans) is always `str`. Unsupported computed expressions remain
untyped, never the type of one reference leaf. Compose retains each declared output's
optional field-local contract, deriving names from that one representation. An opaque
field is therefore not misrepresented as actual `None`, MAP can tell an opaque result
from a missing declaration, and LOOP enforces every proven sibling at load when its seed
source is also statically proven, then unconditionally at commit and replay even when
another field remains opaque. A proven `${null}` source fits a nullable
sink, including recursively inside lists, matching runtime coercion. This lets a loop body
compute a carried boolean directly, for example
`exited: ${ask.output == "/exit"}`.

A nested record or list value is typed structurally: each field takes the type of its
own value. A [tag](syntax.md#typed-slots-yaml-tags) on a slot (`name: !T value`) replaces
the inferred type with its declared one. The loader rejects it only when the two are
disjoint; the binder checks the value at runtime. A tag never fills in a value, and it
converts only how files are held: in a `file` slot a file's JSON form becomes a file, and
in an `object` slot a nested file becomes its mapping. Numbers are delivered as given
(`!float ${i}` delivers an int).

A **prompt** is the strict context: it reads local inputs as bare `${name}`;
AGENT prompts also read their own declared variables. It renders in
`STRICT_RAISE` mode and may call a formatting builtin (`render_as_json`, `join`, `upper`,
`lower`). In TS yian, `len` isn't one: compute a length in an input binding. A host that
registers its own function named `len` replaces the shipped one, and a prompt may call it.

## builtins — pure expression functions

A template span may be a plain reference or a call to a pure builtin. Prompts use
local values; bindings and conditions use their existing resolver. The same
closed registry serves these contexts. Builtins are pure and read-only (formatting; TS yian's
`len` measures) and create no graph node.

| Builtin | Does |
|---------|------|
| `render_as_json(value, indent=2)` | pretty-print a value as JSON |
| `join(value, sep="\n")` | join a list's elements, each stringified |
| `upper(s)` / `lower(s)` | case-fold text |
| `len(obj)` | TS yian: the length of text in code points or of a list in items, an `int`; bindings and conditions only |

Changed (TS yian): `join` writes each element in the text form, and `sep` must be text.
`render_as_json` uses yian's own writer with Python's layout and floats in the text form (`1.0`
renders as `1`, infinity as bare `Infinity`, so the text is not strict JSON), and takes only an
int indent of 0 or more. `upper`/`lower` use JavaScript's non-locale case mapping, which differs
from Python's only where the engines' Unicode versions differ.

Changed (TS yian): `len` measures rather than formats, and gives an `int`: `len("é😀")` is 2, as
Python's `len` counts code points. Anything but text or a list fails:
`len() takes str or list, got record`. A prompt can't call it.

Typedef names are PascalCase and every shipped builtin is lowercase, so the loader refuses a
typedef named like one (`typedef name 'len' must be PascalCase (an uppercase identifier)`). A
host-registered function isn't checked; a builtin wins over a typedef of the same name in a call.

A builtin never turns null into text. A null argument, a null `join` element, a
non-text `upper`/`lower` argument, or a non-list `join` argument fails the
evaluation, like arithmetic over null. Give a default when the value may be
missing: `${upper(input.name :- "")}`. A null nested inside a `render_as_json`
value still renders as `null`.

`render_as_json` of text renders that text as a JSON string, so nesting
re-encodes: `render_as_json(render_as_json(x))` is the JSON string of
`render_as_json(x)`'s text, the same as Python's `json.dumps(json.dumps(x))`.
A file, at any depth, renders as its mapping `{"uri": ..., "mime": ..., "name": ...}`.
Each level escapes the quotes and backslashes of the one inside it, so the text
roughly doubles per level. Render a value once.

New builtins register with `@register_template_fn()`, before loading a flow that
calls one.

Changed (TS yian): a host registers a builtin with `registerTemplateFn(name, {params, call})`,
where `params` lists `{name, default}` in order and a param with a `default` is optional. A
call's arguments are checked against the declared `params`, as Python checks a function's
signature. A builtin gets yian values, ints as `bigint`, and its result is read as a host value
(see [Types → host intake](typing.md#two-constructors-infer-vs-check)), so an integral `number`
within ±(2^53−1) comes back as an int. A builtin must be pure and deterministic: replay and
learning bind a node again from the run log instead of reading recorded inputs, so yian may call
it again with the same arguments and expects the same result.

Every call is checked before it runs, wherever it is written (bindings, flow outputs,
conditions, prompts): an unknown function, or arguments the builtin doesn't take, is a
load error. Set in Python, a condition with either fault, or a binding, flow output, or
prompt that calls a builtin with arguments it doesn't take, makes `run_flow` raise
`ExpressionError` before the run. A `typedefs:` constructor `T(...)` is not an unknown
function; set in Python, its name must be in the node's type table.

Each shipped builtin returns text, except `len`, which returns `int`, so a field read on its
result (`${upper(x).f}`) is a load error wherever it is written: `reference ${upper(...).f}
reads field 'f' on type str, which has no fields`, or `reference ${len(...).f} reads field 'f'
on type int, which has no fields`. A host-registered builtin's result, even one registered
under a shipped name, isn't checked. A value bound from a builtin call (`v: ${upper(x)}`)
is not typed at load, so a later `${v.f}` is not checked.

## `isinstance` — the type test

`isinstance(x, T)` is `true` when the value of `x` satisfies the type named `T`:

```yaml
input:
  is_buy: ${isinstance(decide.output, Buy)}
```

It is a special form, not a `TEMPLATE_FNS` builtin. `T` is a bare type name, never a
reference: a `typedefs:` name, or one of `str`, `int`, `float`, `bool`, `date`,
`datetime`, `object`. `list(...)` or `Optional(...)` would read as a call, so give it an
alias in `typedefs:` first. Each flow resolves the names against its own `typedefs:`
(a `uses:` child against its own, and a parent can't name a child's type). An unknown name
is a located load error: `unknown type 'list'; isinstance takes a typedefs: name or one of
str, int, float, bool, date, datetime, object`. `Any` is not a target, since it would
only test for null: write `x != null`. The call must be exactly
`isinstance(<expr>, <TypeName>)`: two positional arguments, and no field read on the
result.

The check is the pool's write-boundary check (`build_value_as`), so a value passes exactly
when it could be written to a slot of type `T` (an `int` passes `float`). With a typedef
that names `file` (`FileT: file`), a file's JSON form `{uri, mime, name}` passes
`isinstance(x, FileT)`, since a `file` slot reads it. A missing `x` is
checked as `None` in the non-strict modes, so it fails a record and passes an
`Optional(...)` alias. Load-time inference types the result `bool`, and the ref-walk and
rewrites skip `T`.

Compose gives each node a type table (`Node.types`) of the names its expressions use, and
evaluates with it. `isinstance` works in:

- bindings: node inputs, MAP per-item inputs, `call(...)` arguments, LOOP seeds;
- `case` `when:` conditions, where a searched `case` also narrows its branch (see
  [Flow syntax → `case`](syntax.md#narrowing-a-union-in-a-branch));
- node asserts and flow asserts;
- LOOP `while:` / `until:`;
- flow outputs.

In every condition (`when:`, asserts, `while:` / `until:`), a check narrows the reads after
it through `and`, `or` and `not`: an `isinstance`, a tag comparison, or a null check
`x != null`, which narrows `Optional(T)` to `T` (see
[Flow syntax → Narrowing inside a condition](syntax.md#narrowing-inside-a-condition)). A
`while:` null check also lets an `Optional` seed fill a named body's strict input (see
[Flow syntax → `loop`](syntax.md#a-while-null-check-guards-the-body)).

It is a load error in:

- prompts and HUMAN_INPUT question text: `isinstance is not a prompt function; compute it
  in an input binding`;
- WAIT `until:` and MAP `over:`: `isinstance is not supported in a wait node's until:`
  (TS yian has no WAIT or HUMAN_INPUT; an `ask`'s and a `sleep`'s inputs are bindings, as
  above);
- a `call` or `loop` node's own asserts, which run on the child flow's END:
  `isinstance in a call node's asserts is not supported; assert inside the child flow`.

An inline child body can't take a whole-value `${isinstance(...)}` argument (`cannot infer
type for inline binding`); declare the input in a `defs:` flow instead.

## Constructors — `T(...)`

There are no YAML tags inside `${...}`, so a typedef name works as a constructor there.
It builds or checks a typed value:

```yaml
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
nodes:
  order:
    kind: tool
    tool_id: place_order
    input:
      decision: ${Buy(action="buy", conviction=score.output)}   # build
      checked: ${Buy(pick.output)}                               # check and narrow
```

- **Keyword form.** `Buy(action="buy", conviction=score.output)` builds a record from
  exactly the given fields and checks it against `Buy`. Nothing is filled in: leaving out
  `action` fails the check, and the error names the field.
- **Positional form.** `Buy(x)` checks `x` against `Buy` and narrows it, exactly like
  `!Buy ${x}` on a slot. A value that is not a valid `Buy` fails the evaluation.
- Either form delivers the value as given, except how files are held: a file's JSON form
  at a `file` field becomes a file, and a file nested at an `object` field becomes its
  mapping.
- Only `typedefs:` names are constructors. There are no conversion functions, so
  `${int(x)}` is an unknown function.
- `isinstance(x, Buy)` stays the boolean test: it returns `false` where `Buy(x)` fails.

Compose resolves the name at load into the node's type table (`Node.types`), as for
`isinstance`, and load-time inference types the result as the typedef. A constructor is
allowed where `isinstance` is, and is a load error where `isinstance` is, including in
prompts.

## Retained expression routes

With `learnable=True`, binding evaluation records the route in the same walk that
produces the delivered value. References retain the actual accepted producer;
record/list construction, selection, coalescing, concatenation, and repetition
preserve their structural routes. Equal values never establish source identity.
An explicit backward call uses those routes without evaluating the expression or
calling a registered function again. Opaque operations stop mechanical transfer
and retain local context; literal/default inputs have no invented upstream source.

In TS yian, a list slice keeps a `list` route of its base's route projected to each kept
index, so feedback on a slice item reaches that item of the base. `len`, like every builtin
call, is an `opaque` route. A list comprehension keeps a `list` route of its elems' routes, and
while it runs a variable carries its item's route (the iterable's route projected to the item),
so feedback on `[m.text for t in history for m in t]`'s item reaches `history[i][j].text`. Its
`if` conditions aren't part of the route, as a CASE condition isn't.

Changed (TS yian): **where to compute.** A non-derived node (AGENT, TOOL, CODE) records its
binding's routes in learnable runs: a computed list as a `list` route with one child per item,
a computed scalar (`m.role + ": " + m.text`, `len(...)`) as a route that records its value.
Derived nodes (START, END, CALL, LOOP, MAP) build their routes in memory and commit pointers,
prompts are rebuilt rather than recorded, and a CASE `when:` is evaluated untraced. So computed
values belong in prompts, `when:` or derived bindings, and an AGENT binding should be a plain
reference: flatten a history in a CALL's binding and measure it in a `when:`. Computed bases
cost more: a slice or comprehension over a computed list (`a.output + b.output`) in a learnable
non-derived binding records the base's route once per item, since only a `source` route stays
flat when projected. There, iterate a plain reference, or move the computation into a derived
node.

Changed (TS yian): where `len()` and list slicing differ from Python. Yian refuses slicing text,
`len` of a record, a `null` slice bound, a step and an index, where Python gives a value; each
message names the fix. A slice of `null` or a miss is a miss (`:-` catches it), where Python
raises. `len(obj=x)` works, since a builtin has no positional-only parameters. A prompt can't
call `len` (`unknown prompt function 'len'; prompts call only formatting builtins ...`).

Changed (TS yian): where list comprehensions differ from Python. Yian refuses iterating text
or a record, generator expressions, tuple targets, conditional expressions (`a if c else b`),
the reserved variable names, rebinding a variable within a comprehension, a target holding `#`
or `/` (graph-expansion separators), and an escape in an iterable or `if` without parentheses;
each message names the fix. Yian gives a value where Python raises: an elem miss is a `null`
item and a condition miss is falsy outside strict mode, and a name read in a clause before its
own `for` reads the outer name (`[y for x in xs if y for y in ys]`), where Python raises
`UnboundLocalError`; in a flow the name checks usually make that an unknown-name load error.
Reads through a variable aren't type-checked at load yet: `m.txt` on a `list(Msg)` loads and
misses at run time.

Feedback is ordinary text, a partial record, an exact-length positional list, or
`None`. Whole-value text stays unchanged; interpretation context accompanies it
separately. Validate every nested length before treating an all-`None` selection
as empty. See [the Python API](api.md#explicit-backward-feedback) for examples and
failed-forward rules.

## Why this shape

- **One grammar, one evaluator, one ref-walk** — a binding, a condition, and a prompt can
  never parse the same text differently.
- **Pool-agnostic** — a `resolve` callable serves the pool, a local record, and the `item`
  scope; `expr` never imports `nodes`/`compile`/`runtime`.
- **Parse/eval separation** — the same tree feeds evaluation *and* the compile-time walks,
  so the inferred graph matches what runs.
- **Safe by construction** — dict-key and file-field access and a closed builtin registry
  keep an expression from reaching arbitrary Python.
