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

`and`, `or`, `not`, and `in` are reserved operators and cannot be the exact root
name of a reference. They remain legal as dotted field names, such as
`record.output.in`.

## Layer 2 — the one evaluator

`eval_expr(tree, resolve, item=None, mode=...)` is the single evaluator for every
construct. It is pool-agnostic — `resolve` turns a reference path into a value, so the same
evaluator serves the pool, a prompt's local record, and the map-body `item` scope.

**Three locked semantics** (`case` routing to `else:` depends on them):

1. a **missing reference is `None`** (non-strict modes),
2. **arithmetic over that `None` raises** — a missing operand never silently becomes `0`,
3. an **ordered comparison with `None` is `False`** — so an absent value routes to `else:`.

When the `None` would come from a node a `case` may skip, Compose rejects the arithmetic at
load and asks for a `:-` default (see `syntax.md`, "Skipped branches and flow outputs").

In bindings, CASE conditions, LOOP predicates, and flow outputs the loader rejects field
reads a known type can't have, including a field only some members of a union have outside a
narrowing `case` branch or an `isinstance` conjunct before it (see `syntax.md`), so a `None`
there is a null, skipped, or untyped value, not a misspelled field. Some assert reads check
only the name, so a `None` in an assert can still be a dotted miss.

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

**Two safety rules:** dotted access is dict-key lookup only, never `getattr` (so
`${x.__class__}` is just a missing key); builtin calls dispatch through `TEMPLATE_FNS` only
(no arbitrary callable is reachable).

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

There are no cost limits. A huge `**` or repetition takes whatever time and memory
Python needs.

Float `+`, `-`, `*` and `/` that overflow give `inf`, which the
[non-finite rule](#non-finite-results) catches at the binding. A result Python refuses to
compute fails like any other expression error:

- `**` overflow: `${2.0 ** 5000}`;
- an int too large to mix with a float: `${10 ** 400 * 1.0}`;
- a repetition count too large: `${"a" * 10 ** 20}`;
- a missing `%(key)s`: `${"%(k)s" % record}` when `record` has no `k`;
- a value holding an int too long to render as text (past Python's limit, 4300 digits
  by default), embedded in text or anywhere in a prompt: `"v=${10 ** 5000}"`.

An int of any size is a value: a whole-string `${10 ** 5000}` binds the int, and the
run log, outputs, and `resume_flow` keep every digit. Rendering it as text fails. An
int literal written in the YAML may have at most Python's limit (4300 digits by
default); a longer one is a load error.
`render_as_json` writes all the digits, so `${render_as_json(big)}` is text you can
embed or put in a prompt.

The failure is `binding.evaluate` in a binding, `node.execute` at the CASE node in a
searched `when:`, `assertion.failed` in a node or flow `asserts:`, and `node.execute` at the
AGENT or HUMAN_INPUT node whose prompt can't render.

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
`"v=${x * 10}"` is the text `v=inf`. Flows that used to pass an infinite value along
now fail. To fix one, keep the value in range before it reaches the binding (guard it
with a `case`, such as checking the base is not negative before `**`, or compute it in
a `code` node), or return null explicitly: a `case` value branch `then: null` (with
`output: Optional[...]`), or a `code` node that returns `None`.

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

A [typed binding](syntax.md#typed-bindings-name-type-value) (`name: Type = value`)
replaces the inferred type with its declared one. The loader rejects it only when
the two are disjoint; the binder checks the value at runtime.

A **prompt** is the strict context: it reads local inputs as bare `${name}`;
AGENT prompts also read their own declared variables. It renders in
`STRICT_RAISE` mode and may call a builtin.

## builtins — pure expression functions

A template span may be a plain reference or a call to a pure builtin. Prompts use
local values; bindings and conditions use their existing resolver. The same
closed registry serves these contexts. Builtins are read-only formatting and
create no graph node.

| Builtin | Does |
|---------|------|
| `render_as_json(value, indent=2)` | pretty-print a value as JSON |
| `join(value, sep="\n")` | join a list's elements, each stringified |
| `upper(s)` / `lower(s)` | case-fold text |

A builtin never turns null into text. A null argument, a null `join` element, a
non-text `upper`/`lower` argument, or a non-list `join` argument fails the
evaluation, like arithmetic over null. Give a default when the value may be
missing: `${upper(input.name :- "")}`. A null nested inside a `render_as_json`
value still renders as `null`.

New builtins register with `@register_template_fn()`.

In a condition the loader checks every call: an unknown function, or arguments the
builtin doesn't take, is a load error. Register a builtin with
`@register_template_fn()` before loading a flow that calls it.

## `isinstance` — the type test

`isinstance(x, T)` is `true` when the value of `x` satisfies the type named `T`:

```yaml
input:
  is_buy: ${isinstance(decide.output, Buy)}
```

It is a special form, not a `TEMPLATE_FNS` builtin. `T` is a bare type name, never a
reference: a `typedefs:` name, or one of `str`, `int`, `float`, `bool`, `date`,
`datetime`, `object`. The grammar has no `[...]` in an expression, so give `list[...]` or
`Optional[...]` an alias first. Each flow resolves the names against its own `typedefs:`
(a `uses:` child against its own, and a parent can't name a child's type). An unknown name
is a located load error: `unknown type 'list'; isinstance takes a typedefs: name or one of
str, int, float, bool, date, datetime, object`. The call must be exactly
`isinstance(<expr>, <TypeName>)`: two positional arguments, and no field read on the
result.

The check is the pool's write-boundary check (`build_value_as`), so a value passes exactly
when it could be written to a slot of type `T` (an `int` passes `float`). A missing `x` is
checked as `None` in the non-strict modes, so it fails a record and passes an
`Optional[...]` alias. Load-time inference types the result `bool`, and the ref-walk and
rewrites skip `T`.

Compose gives each node a type table (`Node.types`) of the names its expressions use, and
evaluates with it. `isinstance` works in:

- bindings: node inputs, MAP per-item inputs, `call(...)` arguments, LOOP seeds;
- `case` `when:` conditions, where a searched `case` also narrows (see
  [Flow syntax → `case`](syntax.md#narrowing-a-union-in-a-branch));
- node asserts and flow asserts;
- LOOP `while:` / `until:`, where a top-level `and` part also narrows (see
  [Flow syntax → References](syntax.md#references-naming-a-value));
- flow outputs.

It is a load error in:

- prompts and HUMAN_INPUT question text: `isinstance is not a prompt function; compute it
  in an input binding`;
- WAIT `until:` and MAP `over:`: `isinstance is not supported in a wait node's until:`;
- a `call` or `loop` node's own asserts, which run on the child flow's END:
  `isinstance in a call node's asserts is not supported; assert inside the child flow`.

An inline child body can't take a whole-value `${isinstance(...)}` argument (`cannot infer
type for inline binding`); declare the input in a `defs:` flow instead.

## Retained expression routes

With `learnable=True`, binding evaluation records the route in the same walk that
produces the delivered value. References retain the actual accepted producer;
record/list construction, selection, coalescing, concatenation, and repetition
preserve their structural routes. Equal values never establish source identity.
An explicit backward call uses those routes without evaluating the expression or
calling a registered function again. Opaque operations stop mechanical transfer
and retain local context; literal/default inputs have no invented upstream source.

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
- **Safe by construction** — dict-only access and a closed builtin registry keep an
  expression from reaching arbitrary Python.
