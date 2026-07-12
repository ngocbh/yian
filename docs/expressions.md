# Expressions — the one `${...}` grammar

> The `expr` package (`grammar.py` + `expressions.py` + `template.py` + `builtins.py`).
> It sits above [`typesys`](typing.md) and below `nodes`, and is **pool-agnostic**: every
> layer takes a `resolve` callable, never the pool itself. For what you may *write* inside
> `${...}`, see [Flow syntax → References](syntax.md#references-naming-a-value).

## One grammar, two readers

Everywhere a flow wires data — a node input, a `when:` condition, a flow `output:`, an
agent `prompt:` — the author writes `${...}`. There is **one** grammar, and the same parse
tree is read two ways:

- **evaluate** it against a `resolve` callable → a value (at run time), and
- **ref-walk** it → the set of references it reads (at load time, to infer edges).

That single source of truth is why the graph the compiler infers matches exactly what the
runtime evaluates. Three stacked layers plus a builtin registry:

```
   grammar.py     PARSE ONLY — one Lark grammar → a parse tree          (parse_expr)
   expressions.py evaluate a tree via a `resolve` callable              (eval_expr)
                  + the compile-time ref-walk (expr_refs) + ref-rewriter
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

Two load-bearing points: one shared `NAME` terminal with an optional call-suffix (a naive
separate-ref-plus-call grammar won't parse under LALR); and `#`/`/` are in `NAME` (runtime
ids like `node#0`) while `-` is not (so `a - b` is subtraction).

## Layer 2 — the one evaluator

`eval_expr(tree, resolve, item=None, mode=...)` is the single evaluator for every
construct. It is pool-agnostic — `resolve` turns a reference path into a value, so the same
evaluator serves the pool, a prompt's local record, and the map-body `item` scope.

**Three locked semantics** (`case default` routing depends on them):

1. a **missing reference is `None`** (non-strict modes),
2. **arithmetic over that `None` raises** — a missing operand never silently becomes `0`,
3. an **ordered comparison with `None` is `False`** — so an absent value routes to `default`.

**`ResolveMode`** — how the caller wants a miss treated:

| Mode | A missing reference… | Used by |
|------|----------------------|---------|
| `BINDING_NONE` | becomes `None` (coalesce/default may fire) | binding values |
| `CONDITION_FALSY` | `None`, falsy through comparisons | `when:` / `asserts:` |
| `STRICT_RAISE` | raises — no silent blank | prompt rendering |

**Two safety rules:** dotted access is dict-key lookup only, never `getattr` (so
`${x.__class__}` is just a missing key); builtin calls dispatch through `TEMPLATE_FNS` only
(no arbitrary callable is reachable). Arithmetic runs on the resolved *values* with Python
semantics — `str + str` concatenates, `list + list` extends (`${xs + [item]}`).

**Compile-time companions** (never evaluate): `expr_refs(tree)` collects the reference
paths an expression reads — this is how the compiler infers data edges. `rewrite_expr_refs`
rewrites only the reference *leaves* by source position — how refs are re-namespaced when a
child flow is inlined.

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

Compile-time output typing mirrors this: a lone single-ref span is a typed passthrough; a
concatenating template (text around spans, or 2+ spans) is always `str`. That is what lets
a loop body carry a `str` field with a plain `${...}` template — no fold node needed.

A **prompt** is the strict context: it reads only the node's own declared inputs (bare
`${name}`), renders in `STRICT_RAISE` mode, and may call a builtin.

## builtins — the prompt formatters

A prompt span may be a plain reference or a call to a pure builtin over the node's
already-bound inputs. A deliberate, bounded bend of "all computation is a node": builtins
are read-only formatting, mint no graph node, and are unavailable in bindings/`when:`.

| Builtin | Does |
|---------|------|
| `render_as_json(value, indent=2)` | pretty-print a value as JSON |
| `join(value, sep="\n")` | join an iterable's elements |
| `upper(s)` / `lower(s)` | case-fold |

New builtins register with `@register_template_fn()`.

## Why this shape

- **One grammar, one evaluator, one ref-walk** — a binding, a condition, and a prompt can
  never parse the same text differently.
- **Pool-agnostic** — a `resolve` callable serves the pool, a local record, and the `item`
  scope; `expr` never imports `nodes`/`compile`/`runtime`.
- **Parse/eval separation** — the same tree feeds evaluation *and* the compile-time walks,
  so the inferred graph matches what runs.
- **Safe by construction** — dict-only access and a closed builtin registry keep an
  expression from reaching arbitrary Python.
