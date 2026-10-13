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
   builtins.py    TEMPLATE_FNS — the pure formatters a prompt may call
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

**Three locked semantics** (`case default` routing depends on them):

1. a **missing reference is `None`** (non-strict modes),
2. **arithmetic over that `None` raises** — a missing operand never silently becomes `0`,
3. an **ordered comparison with `None` is `False`** — so an absent value routes to `default`.

In bindings, CASE conditions, and flow outputs the loader rejects field reads a known
type can't have, including a field only some members of a union have outside a
narrowing `case` branch (see `syntax.md`), so a `None` there is a null, skipped, or untyped
value, not a misspelled field. LOOP predicates and some assert reads check only the
name, so a `None` in a condition can still be a dotted miss.

**`ResolveMode`** — how the caller wants a miss treated:

| Mode | A missing reference… | Used by |
|------|----------------------|---------|
| `BINDING_NONE` | becomes `None` (coalesce/default may fire) | binding values |
| `CONDITION_FALSY` | `None`, falsy through comparisons | `when:` / `asserts:` |
| `STRICT_RAISE` | raises — no silent blank | prompt rendering |

Strictness also applies to the span's final value: a prompt span that evaluates to
null raises before whole-value return or embedded concatenation. Bindings remain
non-strict, so an embedded null contributes an empty string.

**Two safety rules:** dotted access is dict-key lookup only, never `getattr` (so
`${x.__class__}` is just a missing key); builtin calls dispatch through `TEMPLATE_FNS` only
(no arbitrary callable is reachable). Arithmetic runs on the resolved *values* with Python
semantics — `str + str` concatenates, `list + list` extends (`${xs + [item]}`).

**Compile-time companions** (never evaluate): `infer_expr_type(tree, ref_type_fn)` returns
a type only for statically fixed results: scalar literals, exact references,
comparisons/membership, `not`/`and`/`or`, and `isinstance`. Other expressions remain
unknown rather than borrowing an operand's type. `expr_refs(tree)` collects the reference
paths an expression reads — this is how the compiler infers data edges.
`rewrite_expr_refs` rewrites only the reference *leaves* by source position — how refs are
re-namespaced when a child flow is inlined.

**The condition surface** — `when:` / `until:` / `while:` / `asserts:` admit three
spellings that parse and evaluate identically: bare `a > 5`, mixed `${a} > 5`, whole-span
`${a > 5}`. All route through this one path, so they can never drift.

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
| `join(value, sep="\n")` | join an iterable's elements |
| `upper(s)` / `lower(s)` | case-fold |

New builtins register with `@register_template_fn()`.

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
- LOOP `while:` / `until:`;
- flow outputs.

It is a load error in:

- prompts and HUMAN_INPUT `questions:`: `isinstance is not a prompt function; compute it
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
