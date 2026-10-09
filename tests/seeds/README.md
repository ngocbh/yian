# Seed gallery — Yian flow syntax

These are **example flows in the `Yian` contract**: a Compose-inspired,
edge-free, single-value I/O model. They load and run via the Compose loader
(`yian.load_flow` / `run_flow`); the CODE nodes call `tests.seeds.fns`.

> Negative fixtures for failure modes that lack a canonical check live under
> `_future/errors/`. Supported syntax, including blocking HUMAN_INPUT and WAIT,
> stays in the main gallery.

The goal: every behavior and every syntax convention shows up in at least one seed, so the
gallery doubles as the spec.

---

## The shape of a flow

A flow is a function `'a -> 'b`. A file is **Docker-Compose-shaped**: metadata scalars, then
the flow's interface and body as **top-level sections** — no `edges:`, no `__start__`/`__end__`
nodes, no per-node `id:`, no body wrappers.

```yaml
id: …            # metadata
name: …
description: …
input:  { … }   # the flow's parameters   — source types (+ default/optional)
nodes:   { … }   # the body — a MAP keyed by node id
output: { … }   # the flow's return       — sink bindings
asserts: [ … ]   # optional — boolean checks over ${input.X}/${node.output}
x-…:             # optional — extension keys, ignored by the engine
```

- **`input:`** is the parameter signature (source side — types). **`output:`** is the return
  (sink side — bindings). Because `output:` can bind from several nodes, multi-output flows are
  natural (the flow returns one object); there is no single-terminal rule.
- Flow-input refs are `${input.X}`. The return is never referenced — nothing reads the "output
  node" (there isn't one).

## One value in, one value out

Every node (and the flow) produces **exactly one value** — a scalar, an **object**, or a
**list**. "Several outputs" = fields of one object. There are no per-key node outputs.

## No `edges:` — the graph is inferred

| Edge kind | Inferred from |
|---|---|
| **data** | a `${X.output}` ref inside a node's `input:` (also `over:` / `on:` / `when:`) |
| **control** | a `case` node's `then:` / `else:` targets |
| **ordering** | an explicit `depends_on: [id, …]` (run-after with no data binding — side effects) |

A cycle is a compile error.

## Node anatomy — flat bodies, keyed by id

`nodes:` is a **map keyed by the node id** (the key *is* the id — no `id:` field). The body is
**flat**: `kind:` + the kind's own fields, no `agent:`/`code:`/… wrapper.

```yaml
nodes:
  score:                       # ← the node id (map key)
    kind: agent
    node_name: Relevance score  # optional — a human display label
    depends_on: [warmup]       # optional — explicit ordering edge(s)
    input:                    # sink bindings (consumed)
      topic: ${input.topic}
    output: float             # source type (produced) — Python typing
    prompt: "…"                # ← kind field, flat on the node
```

Per kind, the flat fields are: **agent** → `prompt` (+ `tools`/`mode`/`llm_config`); **code**
→ `code: module:function`; **call** → `call:` + `input:` (a single application); **map** →
`call:` + `over:` + `input:` (+ `parallel:`) — `List.map` over the `over:` list, `${item}` per
element; **case** → `cases:` + `else:` (+ `on:`); **model** → `model_id:` (+
`weights_uri:`/`runtime:`); **tool** → `tool_id:` + `input:`.

A `call` also has an **inline form** — `call(f, arg=${ref}, k=lit)` written as the whole value of any binding
(a node input, a TOOL `arg:`, a `map`'s `over:`, a flow output). It is sugar: it desugars
**at load** into an anonymous `call` node and the host binding becomes `${<that node>.output}`.
Keyword args only; each arg value is a full binding (a `${…}` ref/coalesce, else a YAML-scalar
literal); a nested `call(g, …)` arg desugars inner-first. It cannot capture `${item}` (the synth
node has no map-element scope — use a named `map` node with `over:`). Seed 21.

## Declarations are native YAML — source side = types, sink side = bindings

The YAML *value* of a declared field is read as-is, recursively:

| Where | Side | The value is a … |
|---|---|---|
| `node.output:` | source (produced) | **type** (Python annotation) |
| top-level `input:` | source (parameters) | **type** (`Optional[X]` = nullable; `TYPE = default`) |
| `node.input:` | sink (consumed) | **binding** |
| top-level `output:` | sink (returned) | **binding** |
| `call:` `input:` | sink (call args) | **binding** |

**Types** (source side) — annotations are **Python `typing`**. A string leaf is a scalar/generic; a
map is a record (a dataclass — fields recurse):
```yaml
output: str                              # scalar: str / int / float / bool / date (Any = opaque)
output: list[str]                        # generic: list[X], dict[K, V], Optional[X]
output: { decision: str, why: str }      # record (inline — OK: no [ ] or ${…} inside)
output:                                  # record (block style — identical)
  decision: str
  why: str
output:                                  # nested record
  rating: float
  meta: { source: str, asof: date }
```
(Anonymous inline records **are field-checked** at compile time, like named records — a dotted read
of a field the record doesn't declare is a compile error. Only
`Any` stays opaque. Naming a type in `typedefs:` adds reuse + cross-flow structural equivalence, not
"more checking".

**Bindings** (sink side) — a reference, a literal, an operator form, or a coalesce (see below).

## User-defined types (`typedefs:`)

A top-level `typedefs:` section lets a flow **name and compose** types, **Python-typing style** — a
record is a *dataclass*, annotations use `typing`:

```yaml
typedefs:
  Topic: str                          # STRING = alias (any type expression)
  Bundle: list[Topic]                 #   aliases compose -> list[str]
  Amount: float
  Category: Literal[pro, con, mixed]  # enum
  Choice:                             # tagged (payload) union: a sequence of bare-tag | {tag: payload}
    - defer
    - approve: { count: int }
    - reject: RejectInfo
  RejectInfo:                         # MAP = record (a dataclass — field: annotation)
    count: int
    limit: Amount
  Rating:
    category: Category
    score: float
    note: Optional[str]               # nullable field
  Plan:
    rating: Rating
    history: list[Rating]             # list over a user type
```

Use a type name anywhere a type is expected (`input:`, a node `output:`); dotted access into a
named record is **type-checked at compile time** (`${analyze.output.rating.category}`). Rules:

- **Scalars** are Python: `str`/`int`/`float`/`bool`, plus `date`; opaque = `Any`. **Generics**:
  `list[X]`, `Optional[X]`, `dict[K, V]`.
- **Enums** are `Literal[a, b, c]` (tag-only) — routed by `case … on`. Model tagged choices as a
  discriminated record with a `Literal` field, then route on that field. `Union[...]`, payload-union
  sequences, and the retired `kind: match` form are not supported.
- **Names**: PascalCase; must not shadow a scalar (`str`/`int`/…/`date`) **or** a typing constructor
  (`Any`/`Optional`/`Literal`/`Union`/`List`/`Dict`).
- **Equivalence** across call/map boundaries is **structural** and recursive through records and
  lists; recursive or alias-cycle definitions are rejected.
- *Status:* the Compose loader resolves scalars, `list[X]`, `Optional[X]`, records, `Literal` enums,
  and aliases. `dict[K, V]` currently resolves to a lenient object placeholder; structural key/value
  checking remains unsupported.

## References & interpolation

| Form | Means |
|---|---|
| `${input.X}` | a field of the flow's input |
| `${<node>.output}` | node `<node>`'s whole value |
| `${<node>.output.field[.sub]}` | dot into an object value |
| `${<case>.output}` | a `case` node's **taken-branch value** (desugars to a coalesce over its branch targets) |
| `${item}` | inside a `map` node body only (`kind: map` with `over:`) — the current element |
| `${system.X}` | host-ambient (run id / clock / tenant); reserved |
| `${name}` / bare `name` | that node's own declared input `name` — as `${name}` inside an AGENT prompt, or bare `name` in a `case` `when:` |

**Operator forms** inside `${…}` — one expression grammar (arithmetic + our coalesce):

| Form | Means |
|---|---|
| `${a + b}`, `${a * b + 1}` | arithmetic — `+ - * / % **`, unary minus |
| `${a == b}`, `${x in [1, 2]}` | comparisons / membership / `and`/`or`/`not` |
| `${[a, b]}`, `${upper(x)}` | list literal / pure builtin call |
| `${X:-default}` | value, else `default` if X is null/absent |
| `${X:?"err"}` | value, else **fail** with the literal message `err` (required; quote it) |
| `${a \| b \| c}` | first-present among **peers** — n-ary coalesce (branch-joins) |
| `$$` | a literal `$` (universal escape — in a prompt `$$` renders a single `$`) |

Nesting a ref is allowed: `${a:-${b:-"lit"}}`. The `:-`/`:?` RHS is an
expression, so a **string** default must be quoted (`${x:-"today"}`; bare
`today` reads a variable). Rule of thumb: `:-` for *value-or-default*, `|` for
*whichever-ran*.

**Whole-string vs embedded:** a value that is *exactly* `${ref}` resolves to the **typed** value
(could be a list/object); a `${ref}` embedded in surrounding text is **stringified**. So
`briefs: ${map.output}` is a real list; `"see ${map.output}"` is its string form.

## Bindings must be block form (not inline)

Bindings are **unquoted, one per line**. They must **not** sit in an inline flow-mapping
(`input: { topic: ${…} }`) — there the `}` in `${…}` (or a `[` from `list[…]`/`Literal[…]`/
`Union[…]`/`dict[…]`) would break the map. Always use block form for `input:`/`output:`/`call`
maps, and for any record/payload with a generic-typed field. (Quoted values — an AGENT
`prompt:`, a `case` `when:` — *may* stay inline; their quotes protect the `}`.)

## Three expression contexts

One `${…}` grammar; the context decides what happens to the result:

| Context | What it does |
|---|---|
| **Bindings** (`input:`/`output:` values) | **evaluated** to a typed value — refs / literals / arithmetic / lists / `:-` `:?` / `\|` / pure builtins. (A child-flow call is the whole-value `call(…)` directive, not a `${…}` span.) |
| **`when:` / `asserts:`** | **tested** as a boolean — `== != < <= > >= in not in`, `and`/`or`/`not`, parens, arithmetic operands. Canonical **bare** form (no `${}`); `${…}`-wrapped spellings load and evaluate identically. |
| **Prompts** | free text with embedded `${…}` spans (each stringified). |

"Bindings wire, conditions test, nodes compute."

## Branching — the `case` node (SQL CASE)

A `case` node **routes only** (no `input:`). Two forms, like SQL:

```yaml
gate:                            # searched form — each when: is a boolean
  kind: case
  cases:
    - when: "score.output >= 0.5"
      then: positive
  else: cautious                 # unconditional fallback

route:                           # simple form — match a value with on:
  kind: case
  on: ${classify.output}
  cases:
    - when: pro                 # here when: is a VALUE to match (not a boolean)
      then: pro_note
    - when: con
      then: con_note
  else: mixed_note
```

Exactly one branch runs; the others are skip-flooded and never write a value, so a join uses a
coalesce to pick whichever ran: `${pro_note.output | con_note.output | mixed_note.output}`.
The **value-case shorthand `${<case>.output}`** is sugar for exactly this coalesce over the
case's branch targets (= the taken branch's value; seed 22).

## Defaults / optionality

`Optional[X]` (nullable — *type-level*) and `= default` (omission-fill —
*parameter-level*) are **orthogonal**, not a choice. A top-level param is:

- **required** ⇔ its type is **not** `Optional` **and** it has **no** `= default` (`topic: str`).
  Omitting it is a hard missing-input error.
- **`Optional[date]`** (no default) → omitted yields **null**.
- **`window: int = 30`** → omitted yields **30** (the RHS is always a YAML literal, never a type ref).
- **`Optional[int] = null`** has an explicit null default, distinct from no `=`.
- **`Optional[date] = "2026-01-01"`** → omitted yields that ISO date string; an
  explicitly supplied null shadows the default and is allowed by the nullable type.

The default RHS is validated against its declared type while the flow loads. It
cannot be empty and is never string-coerced: `int = "30"` is invalid, while
`int = 30` is valid. Compatible values retain their authored payload, so
`float = 30` keeps the integer `30`. Only supplied boundary strings may be
coerced.

A flow input is a string annotation (`TYPE`, `TYPE = literal`, or
`Optional[TYPE]`) or a mapping that directly declares a record type. A mapping is
not a `{type:, default:}` wrapper. Quote the whole annotation when a structured
default contains YAML mapping syntax:

```yaml
input:
  budget: Amount = 1000.0
  bundle: Bundle = ["ACME"]
  prior: 'Rating = {category: pro, score: 0.0, rationale: neutral}'
```

Applied at every flow boundary (top-level run plus a call/map child seeding its own parameters). Defaults live
on flow `input:` only — internal nodes never declare defaults; a sink that may be absent uses
`${X:-default}` or a coalesce `${a | b}` (e.g. to pick whichever `case` branch ran). *(Distinct layers:
`= default` sets a **parameter's** default in the signature; `${X:-default}` is a
**use-site** fallback for a possibly-null reference.)*

## Constraints — the `asserts:` section

No `min`/`max`/`options`/`label` on fields. Value constraints (range, enum membership,
required, cross-field invariants) are a top-level **`asserts:`** list of boolean checks over
`${input.X}` / `${node.output}`; any false fails the run.
```yaml
asserts:
  - input.topics != []                 # non-empty
  - input.window * 2 <= 365           # arithmetic is allowed here
```

## Extensibility — anchors + `x-`

The schema is strict (`extra='forbid'`) **except** keys matching `x-*`, which it ignores. Use
them for custom metadata and to park reusable YAML anchor blocks:
```yaml
x-agent-defaults: &agent_defaults    # parked under x- so the strict schema ignores it
  output: str
nodes:
  pro:
    kind: agent
    <<: *agent_defaults              # merged in by YAML before the engine sees it
    input:
      topic: ${input.topic}
    prompt: "…"
```

---

## The gallery

| File | Shows |
|---|---|
| `00-hello-agent.yaml` | minimal: top-level `input:`/`output:`, one AGENT, scalar value, bare return, inferred graph |
| `01-structured-agent.yaml` | AGENT **object** output + `${node.output.field}` dotted access + a CODE consumer |
| `02-case.yaml` | `case` (searched form: `when`/`then`/`else`) + join via a **coalesce** binding |
| `03-research-one.yaml` | a child flow (CODE→AGENT, object output) — the `call` target below; a param `default` |
| `04-call.yaml` | `call` (typed function application) re-exporting a callee's value |
| `05-call-map.yaml` | a `map` node (`kind: map` + `over:` + `parallel:`) over a list + `${item}` + `:-` default + `asserts:` + `node_name` + multi-output |
| `06-case-on.yaml` | `case … on` (simple form: value match) + a 3-way n-ary `\|` coalesce join |
| `07-model-rating.yaml` | `MODEL` kind — `model_id`/`weights_uri`/`runtime`, `int` type, object output |
| `08-tool-news.yaml` | `TOOL` kind — `tool_id` + untyped `input:`; whole-string (typed) vs embedded (stringified) `${…}` |
| `09-interpolation-ops.yaml` | the operator forms — `${X:?}` / nested `${X:-…}` / `$$` escape / `${system.X}` |
| `10-asserts-arithmetic.yaml` | arithmetic + `and`/`or`/`not`/`in` in `when:`/`asserts:` |
| `11-reuse-anchors.yaml` | `x-*` extension keys + YAML anchor (`&`/`*`/`<<:`) reuse + `llm_config` |
| `12-depends-on.yaml` | run-ordering edges — `depends_on` (co-skip) / `runs_after` (pure order); a side-effect node, no data binding, + `bool` output |
| `13-types-objects.yaml` | the type grammar — Python scalars, `list[X]`, inline/block/nested objects (records), deep dotted access |
| `14-agent-tools.yaml` | AGENT knobs — `mode` / ordinary `tools` including `ask_user` / `llm_config` + structured output |
| `17-effects-human-wait.yaml` | effects — blocking `HUMAN_INPUT` (typed answer) + absolute UTC `WAIT` (`until:`) + `case` on the answer + a `depends_on` side-effect |
| `18-research-pipeline.yaml` | realistic end-to-end — fan-out reviewers → typed-`View` synth → `case … on` stance → multi-output + `asserts:` |
| `19-binding-stances.yaml` | every input-binding stance — required (plain, co-skip) / optional (`:-`) / branch-join (`\|`) / fail-loud (`:?`); pins **per-input readiness** |
| `20-call-defs.yaml` | a `call:` resolving to an in-file `defs:` callable (a multi-node sub-flow inline) — `defs:` section + `call` defs-first; loads resolver-free |
| `21-inline-call.yaml` | an INLINE `call(enrich, topic=${input.topic})` call directive — desugars to a synth `call` node on the in-file def; loads resolver-free |
| `22-case-value.yaml` | the **value-case**: `output: ${gate.output}` = the taken branch's value (desugars to the branch coalesce — seed 02's hand-written join) |
| `23-asserts-scopes.yaml` | **asserts at every scope**: FLOW (boundary/post + a named `call` node's `.output` in a flow assert), DEF-child (a `defs:` callable's `asserts:`, enforced at the call seam), and NODE (per-node contract — `${name}` PRE, `${output}` POST) |
| `27-expr-ops.yaml` | the **unified `${...}` grammar** — arithmetic (`+ - * / % **`, unary minus, parens), string `+`, list literals + `list + list`, comparison / `and`/`or`/`not` / `in`/`not in`, `\|` / quoted `:-` / quoted `:?`, the `upper`/`lower`/`join`/`render_as_json` builtins (in bindings, a prompt, and `asserts:`), `$$` escape |
| `28-refine-loop.yaml` | the **`loop`** node — `until:` post-check predicate + `max:` guard over an in-file `defs:` body, the `'a -> 'a` carried-record contract; reuses the new expression forms (arithmetic + a builtin) in the body; loads resolver-free |
| `29-inline-code.yaml` | inline **`code:`** source — a bare body reading `inputs`, wrapped as `def main(inputs)` and run **in-process** (the alternative to the `module:function` reference form); same one-dict convention as reference mode |
| `30-reference-code.yaml` | reference **`code:`** source — a `module:function` token imported and called **in-process**; the reference twin of seed 29 (same body as its inline block), showing one computation both ways |

Every node kind (AGENT/CODE/MODEL/TOOL/call/map/case/loop/HUMAN_INPUT/WAIT) and every supported
convention appears in at least one seed. Seed 17 pins blocking HUMAN_INPUT and WAIT's
absolute UTC deadline surface. Tagged data is modelled as a discriminant record (a `Literal`
field) routed by `case … on <field>` — the `kind: match` + payload-union design was dropped.

## Negative gallery — `errors/`

`errors/` holds flows that are **supposed to fail**, at compile **or** at runtime — they pin
loud, source-located failures and are the loader's negative test fixtures
(`tests/engine/test_errors.py`).

**The rule:** every behavior here gets accompanying error fixtures in `errors/` covering its
expected **compile-time** (`load_flow` → `LoadError`, located at the `.yaml` line) **and
runtime** (`run_flow` → `status="failed"`) failure modes that the engine can actually capture;
each isolates one failure, with a matching test. Failure modes whose check isn't built yet live
in `_future/errors/` until that check exists. See `errors/README.md` for the full table and rule.

HUMAN_INPUT and WAIT are blocking nodes. Reusable CODE helpers for the seeds, including
`one_line_summary`, `fetch_facts`, `prime_cache`, `build_outline`, and `confirm_action`, live in
`tests.seeds.fns`. The `errors/` fixtures fail before or without running a CODE callable.

## Supported conventions

The supported contract uses Compose-style top-level
`input:`/`nodes:`/`output:`/`asserts:`; no `edges:`/`__start__`/`__end__`; keyed `nodes:` map
with flat bodies; `case`/`when`/`then`/`else`/`on`; references `${…}` with `:-`/`:?`/`|`
and `$$` escape; unquoted block-form bindings; strict schema + `x-*` + YAML anchors; and a
**Python-typing** type system (`str`/`int`/`float`/`bool`, `list[X]`/`Optional[X]`, dataclass
records, `Literal` enums, discriminated records, inline `TYPE = default`) — `typedefs:` registry.
