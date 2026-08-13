# Reference — engine internals

Lookup companion to [`SKILL.md`](SKILL.md). The skill is the *workflow* (design
from the functional model → fit invariants → plan → implement); this is the
quick-reference and the [`templates/`](templates/) index. The canonical design
text is [`src/agent_composer/README.md`](../../../src/agent_composer/README.md);
the authoring surface is [`docs/syntax.md`](../../../docs/syntax.md).

## Templates

| Template | Use |
|----------|-----|
| [`templates/node_kind/node.py.template`](templates/node_kind/node.py.template) | the `Node` subclass skeleton |
| [`templates/node_kind/__init__.py.template`](templates/node_kind/__init__.py.template) | the package charter / re-export |
| [`templates/node_kind/test_node.py.template`](templates/node_kind/test_node.py.template) | unit + load/run test skeleton |
| [`templates/node_kind/WIRING.md`](templates/node_kind/WIRING.md) | the cross-file edits to make `kind: xxx` authorable |

## The node contract (the most portable idea)

A node is a **pure function of its bound input record**. A kind implements
`_run_impl(inputs, **caps) -> NodeResult`; the base `Node.run` is a concrete
**template method** the engine drives — it fires the node's asserts around the body:
`apply_defaults(record) → pre_asserts(record) → _run_impl(record, **caps) →
post_asserts({**record, "output": value})`. Assert firing is **record-only** (no pool),
so it preserves node purity, and it happens **exactly once** here in the node (not in
`eval_node`). `_run_impl` returns ONE of the closed sum:

| Return | Meaning |
|--------|---------|
| `Output(value)` | the one produced value; the engine writes it under the node id. Optional `Output(value, commit_as=<id>)` redirects the write to another id (and fires *that* id's out-edges) — the node-chosen commit target; `None` (the default) commits under the node's own id. |
| `Route(handle)` | a routing-only outcome (a CASE picks an out-edge `handle`); carries no value, writes no pool entry. The engine emits `NodeRouted` and takes the chosen edge, skip-flooding the siblings. |
| `Pause(reason)` | dormant custom/artificial compatibility. The engine emits `PauseRequested`, parks the leaf, and later delivers a command value as its `Output`; no shipped authorable node returns it. |
| `Grow(seed, prune)` | describe one deterministic graph expansion without constructing it. `seed` is the kind-shaped but runtime-opaque builder input (CALL record, MAP records, or LOOP carried record/index); `prune` names ids to retire in the same step. Compile-owned `build_grow(spawner, seed)` constructs the self-describing `Flow`, and the engine splices it generically via `_apply_grow`. The canonical seed is persisted for durable replay. Only spawner kinds (`is_spawner`) may return it. |

A streaming kind is a generator that yields `StreamChunk` and *returns* a
`NodeResult`. **Failure is not a variant** — a node `raise`s and the engine
boundary turns it into `NodeFailed`. `Node.on_failure` is a dormant future seam,
not currently driven by `eval_node`.

**Invariants:** a node never receives the pool (the `eval_node` seam binds its
inputs); a node never writes the pool (it *describes* `Output(value)`, the engine
performs the write). Keeps nodes pure and the state immutable (`let`-bindings).

Growth uses the same inversion boundary. Node modules never import `compile`: a
spawner returns a canonical seed only. `compile.expand.build_grow` is the sole
kind-aware constructor; package-root `run.py` injects it into `FlowEngine` for live
growth and checkpoint replay. Runtime may consume `compile.model` topology but never
imports `compile.expand`, so the splice remains kind-blind.

**Capabilities (`caps`)** are narrow, engine-owned effects passed as keyword args to
`run`; each is gated by a node trait. A mapped call gets `caps['bind_item']`
(per-element bind, gated on `binds_per_item`). AGENT and WAIT get the exact run-owned
`caps['stop_event']` (gated on `needs_stop_event`): AGENT checks it between external
model/tool effects, while WAIT blocks on it so host shutdown or sibling failure wakes
the deadline wait. AGENT is the deliberate model-access exception: its mode imports
`llm_clients` and calls `model_from_config` directly, while its granted tools use the
ordinary tool registry.

## Node-owned traits/hooks (how the core stays kind-blind)

The engine core reads these node members instead of branching on `node.kind`. Defaults sit on the
base `Node`; a kind overrides only what it needs.

| Member | Where the core reads it | Default → overrides |
|--------|-------------------------|---------------------|
| `is_spawner: bool` | result boundary: permit `Grow` only from a graph-expanding node | `False` → `True` on call/map/loop |
| `binds_per_item: bool` | read boundary (`eval_node`): bind PER ELEMENT via a `bind_item` cap vs bind `params` once | `False` → `True` on `map` |
| `bind_reserved(wiring, pool) -> dict` | read boundary: reserved keys pre-resolved into the record before `run` | `{}` → `wait` `{"until": ts}`, `map` `{"over": [...]}` |
| `reserved_wiring_keys() -> set` | load-time (compile passes: `check_wiring_parity`, ref-scan): the NAMES of those reserved keys so validation gates on traits, not `node.kind` | `∅` → `wait` `{"until"}`, `map` `{"over"}` |
| `grow_depth_delta: int\|None` | growth core: REF-depth increment stamped on the spliced spawners + terminal (positive bounded by `MAX_REF_DEPTH`) | `None` (loop/non-REF, no depth work) → `1` call/map |
| `is_loop: bool` | growth core (`_apply_grow`): gate the origin-keyed single-live-record ledger invariant (the driver self-respawns, so each iteration's grow is attributed to `origin_id` and supersedes the prior record) | `False` → `True` on `loop` |
| `needs_stop_event: bool` | read boundary (`eval_node`): pass the exact cooperative run-stop event | `False` → `True` on agent/wait |

The rest of the splice (add subgraph, enforce `MAX_TOTAL_NODES`, mint one uniform `GrowRecord`,
finish/mark the spawner, apply the origin `commit_as` to the derived terminal, schedule the
subgraph's `start_id`) is
uniform. The old per-kind `_grow_*_residual` switch is gone. Success-path routing (CASE branch/skip,
subflow commit-under-spawner) rides `Outcome` data (`Route.handle`, `Output.commit_as`). A
spawner's own `post_asserts` (a `call` node's `asserts:`) are **lowered onto the cloned child
`END`** at grow time (`expand._lower_spawner_post_asserts`): `${output}` stays; a declared input
`${X}` becomes a synth END param wired off the cloned child START's coerced output, so the assert
fires through the child END's ordinary `run` wrapper — not re-checked at any commit site. Live
growth and durable replay both pass the canonical seed persisted in the growth ledger through
the same injected `compile.expand.build_grow`, so lowering is identical on resume. The moved
assert's failure names the **child END id**, not the spawner.

AGENT is now an ordinary non-spawner leaf. Its model/tool loop, including an
explicitly granted `ask_user`, stays inside one live `_run_impl`; post-asserts fire
on the final `Output` through the base `Node.run` template. HUMAN_INPUT and WAIT
are likewise live leaves returning `Output`.

## `NodeKind` (closed vocabulary)

`NodeKind` is a closed enum (no registry/metaclass), but the engine **core does not dispatch on
it** — `runtime/engine.py` + `runtime/eval_node.py` branch only on the closed `Outcome` sum and on
node-owned traits/hooks (below). Ordinary behavior lives in each node's own `_run_impl`; growth
construction's sanctioned closed kind match lives in `compile.expand.build_grow`, which
interprets node-produced seeds outside the runtime core. A ratchet test
(`tests/engine/test_kind_census.py`) holds the core's `NodeKind`/`*Expansion` dispatch count at 0.
Kind-specific diagnostics stay with their kind. CODE owns
`nodes/code/watchdog.py`, which wraps only its reference/inline in-process Python
body and logs one soft-budget warning without killing it. AGENT, HUMAN_INPUT, and
WAIT are never timed by that watchdog.

| Authorable kinds | Internal-only (loader-synthesized) |
|------------------|------------------------------------|
| `AGENT`, `CODE`, `MODEL`*, `TOOL`, `CASE`, `HUMAN_INPUT`, `WAIT`, `CALL`, `MAP`, `LOOP` | `START`, `END` |

\* `MODEL` parses but `run` raises — the ML-serving seam was removed as dead
plumbing; re-add when real serving lands.

## OCaml analogue map (design from this)

| Our construct | OCaml concept | Why it holds |
|---------------|---------------|--------------|
| a flow | a function `'a -> 'b` | typed in, typed out; composes |
| an agent | a flow whose leaf is an LLM loop | no special contract |
| `call:` / `uses:` | function application / a module ref | nests to any depth |
| the variable pool bind | `let (node_id, key) = ...` | immutable, no mutation |
| `NodeKind` + `match` | a variant + exhaustive `match` | closed set, no registry; the runtime core matches on `Outcome`/traits, while compile's `build_grow` interprets spawner seeds by kind |
| `Output \| Route \| Pause \| Grow` | a sum type (the result) | failure is a `raise`, not a case |
| dormant artificial pause / resume | algebraic effect + handler | retained compatibility only; authorable blockers return `Output` live |
| a package charter (`__init__`) | a module signature (`.mli`) | a narrow declared interface |

When borrowing runtime mechanics from a prior worker engine: **borrow** the
correctness-critical parts (single-writer dispatcher, 3-state edge join,
outputs-before-successors, recursive skip-flood, layered checkpoint, discriminated
pause reasons); **drop** scale/framework baggage (external DBs, dynamic worker
scaling, plugin registries, heavy layering, multi-tenancy).

## Non-negotiable invariants (the keep/simplify/drop bar)

A change is a design smell if it breaks any of these — most are the functional
model made enforceable.

- **Deterministic structure** — the author fixes the call graph; the LLM fills
  leaf boxes. A flow never rewrites itself. No agentic routing.
- **A flow is a function** — explicit typed input/output signature; an agent gets
  no special contract.
- **Composable / recursive** — a node can *be* a flow (`call:`/`uses:`), nestable
  to any depth. Never assume a node is a leaf; preserve recursion through
  compile + run; the dormant checkpoint replay must also preserve existing growth.
- **Privileges no output type or domain** — prefer the most general primitive that
  composes over a use-case-specific feature.
- **Node never writes the pool** (purity); **typed, losslessly-serializable state**
  (`TypedValue` / `VariablePool` — the basis for checkpoints + `${...}` refs).
- **Blocking authorable effects are live** — AGENT `ask_user`, HUMAN_INPUT, and
  WAIT return `Output` from the same invocation and never create pause topology.
  The generic `Pause`/`RunCheckpoint` API remains only for custom/artificial
  compatibility until issue #117. While it remains, its checkpoint records
  `num_workers`; `FlowEngine.restore(..., num_workers=N)` and `resume_flow`
  preserve or explicitly override that drive mode.
- **Dependency-light core** — no DB / heavy frameworks; external capabilities enter
  through injected seams (plain callables). *Exception:* the AGENT node imports
  langchain + `llm_clients` and builds its model via `model_from_config`.
- **`llm_config` cascade resolved once at run start** — `resolve_llm_cascade`
  (`compile/`) walks the static call tree top-down, per-field fill-the-gap
  (most-specific wins), deep-copying each CALL/MAP/LOOP child for per-callsite isolation,
  and bakes the effective dict onto every `AgentNode`. It also returns a complete map
  keyed by stable slash-separated static paths. The CLI config is the outermost
  authored layer; the run boundary reads process provider/model defaults exactly once
  at initialization and passes them as the explicit final fallback. `inherit: false`
  skips authored flow/parent/CLI layers but still receives those process defaults.
  `apply_effective_llm_map` replays a recorded complete map without reading ambient
  settings and deep-copies every callsite again. Until the run-log cutover, a durable
  checkpoint resume resolves before `FlowEngine.restore`, whose replay re-clones
  children from the static graph.
- **Stable program and expansion identity** — compile-owned `flow_manifest`
  canonicalizes execution semantics, `flow_fingerprint` hashes it, and
  `expansion_digest` covers an expanded fragment's entry/end ids and topology.
  Render-only ranges, comments, and object identity do not affect these values.
- **Closed `NodeKind`; the core is kind-blind** — `NodeKind` stays a closed enum, but the engine
  core dispatches on `Outcome` + node-owned traits/hooks, never on `node.kind` (ratchet: census 0).
  **single-writer** (workers are pure
  executors, the dispatcher is the only mutator); a pooled drive signals cooperative stop
  before fully joining workers after an abnormal dispatch exit, but not after merely yielding
  a terminal pause; **single-process CLI target**.
- **AGENT structured output** — a non-text `output:` Type switches an agent from text
  producer to structured generation: `type_to_schema` (`nodes/agent/structured.py`)
  derives a pydantic schema, the mode generates a conforming value (native
  `with_structured_output`, or a JSON prompt-injection fallback gated by
  `supports_native_structured(provider, model)`), with `retries:`-capped self-correction.
  Three-part contract: **generate-tries** (the schema asks), **boundary-enforces**
  (`pool.set(..., declared=output_type)` validates on both the primary path and
  any `commit_as`-redirected write), **retry-catches** (a deviation is fed back and
  re-asked). A bare `str`/`Literal[...]` keeps the text path.

## Layer ladder (where code goes)

```
events  <-  typesys  <-  nodes  <-  compile  <-  compose
                        ^   ^
            expr  ──────┘   └──────  llm_clients

compile.model <── runtime ──> suspension
compose + compile.expand + runtime <── package-root run.py
```

Arrows never reverse: a package imports only lower-level or peer packages. See the
`structure` skill. An upward import means the code is in the wrong package (extract
the shared contract to `common.py` / a leaf, or invert via a seam). Dynamic growth is
the concrete example: nodes return seeds, compile owns `build_grow`, `run.py` injects
it, and runtime never imports `compile.expand`. Spawner seed schemas live in their
owning node packages. The scheduling-state vocabulary sits in the internal
`_state.py` leaf shared by compile and suspension, so those consumers do not import
each other.

## The `expr` layer (one grammar, everywhere `${...}` appears)

Bindings, conditions, and prompts share ONE expression grammar — the three
divergent dialects the engine grew (binding coalesce/default, `when:`
boolean/arithmetic, prompt builtin-call) are unified. Three modules, one pipeline
(parse → AST → evaluate / ref-walk):

- **`grammar.py` — parse only.** `parse_expr` compiles a span into a Lark tree
  (arithmetic `+ - * / % **`, comparisons, `and`/`or`/`not`, `in`/`not in`, list
  literals, pure builtin calls with dotted access, `:-` default / `:?` required /
  `|` coalesce). No evaluation here. The evaluator walks this **restricted AST** —
  there is no Python `eval`; a builtin call resolves only against the
  `TEMPLATE_FNS` whitelist, so a span can compute but cannot reach arbitrary code.
- **`expressions.py` — evaluate + ref-walk.** `eval_expr` walks the tree over a
  resolver; `expr_refs` collects the reference paths a span reads (the compile-time
  companion, for edge inference + scope checks); `evaluate_when` is the `when:`
  boolean. `RequiredError` (a `${x:?"msg"}` over an unbound ref) lives here, its
  natural home now that the evaluator raises it.
- **`template.py` — the string surface.** `scan_template` splits literal text from
  `${...}` spans; `eval_binding` renders a binding value; `render_template_record`
  renders a strict prompt; `$$` is the scanner's universal escape (a prompt `$$`
  renders a single `$`).

**Three resolve modes** decide what a MISSING reference does, one per context:
`BINDING_NONE` (a binding → `None`, so `|`/`:-` can coalesce), `CONDITION_FALSY`
(a `when:` → falsy), `STRICT_RAISE` (a prompt → raise; no silent blank).

**`call(...)` desugar — a flow call is NOT an expression.** A child-flow call is a
compile-time directive recognized at TWO sites, by ONE whole-value rule (only the
entire trimmed value, never embedded in a span): a binding's whole value, and a
`case` `then:`/`else:` target. At load it desugars into an anonymous `call` node
(`__call_N`) and the host is rewritten to `${<synth>.output}`. A flow call inside
a span (`${flow(args)}`) is a hard `LoadError` — hoist an embedded call to a named
node; a coalesce-of-calls (`${a() | b()}`) is likewise rejected. Pure builtins
(`${upper(x)}`) stay legal inside `${}`.

**Raw-string public API** (what callers outside `expr` use, re-exported from the
`expr` package `__init__`): `eval_binding` (render a binding), `expr_refs_of` (its
reference paths), `evaluate_when` (a condition), plus `render_template_record` /
`prompt_refs` for prompts. Two families back the reference walkers. Extraction:
`expr_refs_of` reads only `${...}` spans (bindings, templates, prompts), while
`condition_refs` parses a WHOLE expression (a `when:`/`on:`/loop predicate/assert
written bare, mixed, or whole-span) so the bare spelling's refs are seen — using
the template walker on a bare condition finds nothing. Rewriting (spelling-
preserving, for the `case` desugar): `rewrite_expr_refs` / `rewrite_template_refs`
(span walkers) and `rewrite_condition_refs` (whole-expression). Callers pass
source strings + a resolver; they never touch the grammar or AST.

## Located errors (precise source lines)

A runtime failure points at the **exact YAML line** it originates from, not just the
node header. The mechanism is a structured locator produced at the failure site and
resolved to a line at the CLI boundary — never a text heuristic.

- **`SourceSpan(node, kind, key)`** (in `events`, the leaf) — `kind ∈ {input, assert,
  input_decl, field}`; `key` is the input name / assert expr / input-decl name / field
  name; `node` is the node id, or `None` for a flow-level location.
- **Carriers:** `NodeFailed.locator` (node-level) and `RunFailed.locator` /
  `RunResult.locator` (flow-level). All default `None`.
- **Producers:** `BindingError` stamps a node-less `input` span (`bind_params` knows the
  param name, not the node); the node's `run` wrapper raises `NodeAssertFailed` carrying an
  `assert` span when a pre/post assert fails (incl. the flow's boundary asserts lowered onto the
  START node's `post_asserts` and its post asserts lowered onto the END node), and `eval_node`'s
  funnel fills any missing node id via `replace(loc, node=node.id)`;
  `StartNode._run_impl` stamps an `input_decl` span on the e08 `TypeCheckError` (the flow's boundary
  `asserts:` are lowered onto START's `post_asserts` and fire through the same `run` wrapper
  — `run.py` stamps none itself);
  the engine's typed write boundary stamps a `field` span (`key="output"`) on the
  `NodeExecutionError` it raises when a node's value fails its declared `output:` Type.
- **Concrete range (resolved at the failure site):** ALONGSIDE the symbolic `locator`, a concrete
  **`SourceRange`** (start/end line+col, 1-based; in `events`) is attached. For an `input` /
  `input_decl` / `field` locator the funnel keys the node's recorded `field_ranges` on the
  locator's **`(kind, key)`** tuple — kind-blind (on the locator, never `node.kind`), and
  `(kind, key)`-tagged so an input never collides with a like-named field. `field_ranges` is a
  **`FieldRanges`** (`nodes/base.py`), a dict subclass whose `get` **raises `TypeError` on a
  non-tuple key**: a bare-string lookup can never match, and every caller falls back
  (`or node.range`), so without the guard a wrong key silently yields a caret on the wrong line
  — which is exactly how two dead lookups survived review and a green suite. Any new code path
  that builds the map must keep the type (the parse stamp, `_finish`'s copy onto the runtime
  node, the START boundary map, the inline-call desugar all do). An **assert** is located
  BY POSITION: the `run` wrapper resolves the failing assert's own `pre_assert_ranges[i]` /
  `post_assert_ranges[i]` (each aligned 1:1 with `pre_asserts` / `post_asserts`) and raises
  `NodeAssertFailed(range=…)` — so two identical assert texts get distinct carets (asserts are NOT
  in `field_ranges`). It rides `NodeFailed.range` and its flow-level twin `RunResult.range` /
  `RunFailed.range` (all default `None`; the flow twin is the boundary node's resolved range —
  `StartNode.field_ranges[("input_decl", name)]` for an e08 input-decl,
  `StartNode.pre_assert_ranges` / `EndNode.post_assert_ranges` for a boundary / post assert).
- **Resolution (CLI boundary):** `cli/run.py` prefers the pre-resolved `nf.range` →
  `result.range`; only if both are absent does `_resolve_range` map the span itself, via
  `_top_ranges` (the flow's `SourceFrame` maps, else `loader.top_frame_ranges(text)`): a
  `field` locator → that field's range, else `_node_range` (node-kind best field, e.g. a code
  node's `code:`, → node HEADER range). The winning `SourceRange` is boxed via a
  **`Diagnostic`** (severity + message + primary range + notes) through `render_diagnostic`.
- **Cross-flow call traceback** — when a failure is inside a *called child*, its node id is
  runtime-NAMESPACED (`gate/approve`, `outer/via/boom`, `gate#0/inner` for a map element).
  A `call`/`map` node bakes its child flow's render-only **`SourceFrame`** (`compose/loader`:
  label + source text + node→`SourceRange` / field→`SourceRange` maps, built by `_frame_ranges`
  off the parsed descriptors) onto `child_source` at load — for a `defs:` child the text is the
  PARENT file (inner nodes at their absolute parent lines, via `_def_nodes_reader`); for an
  external `uses:` child it is that file's own text, label = its filename. The frame is frozen
  and `__deepcopy__`-returns-self, so `clone_child`'s per-callsite deep-copy shares the one
  instance and it is exempt from the node-purity scan (it carries raw `${...}` YAML, but is
  metadata, not a wiring source). At error time `cli/run.py:_walk_call_frames` splits the
  namespaced id and walks segment-by-segment through the baked IR (`node.child`/
  `node.child_source`), collecting one frame per level; `_render_frame_stack` boxes them
  stacked, most-recent-call-last (Python-traceback style), falling back to the single-frame
  box when fewer than two frames resolve. Each call-chain frame anchors at the node HEADER
  line; the failing leaf at its precise field / kind-fallback line. A frame title names WHERE
  it lives: a top/external file frame is the filename alone, a `defs:` frame is
  filename-qualified (`<file> defs:<name>`) since its nodes physically live in that file.

## Design-note template (step 3 of the workflow)

Before coding any engine change, write this short note and confirm any non-obvious
choice (CLAUDE.md "ask when uncertain"):

```
Construct:    <name> — what it consumes -> what it produces (its type signature)
OCaml analogue: <the concept you're borrowing> — how it maps to nodes/pool/runtime
Keep / drop:  <what you borrow from prior engines; what scale baggage you drop>
Lands in:     <layer/package on the ladder>  (new package? write its charter)
Seam:         <injected callable, if it touches an external dependency> | none
Invariants:   <which of the non-negotiables it touches; how it stays within them>
Tests:        <the tests/engine cases that will prove it>
```
