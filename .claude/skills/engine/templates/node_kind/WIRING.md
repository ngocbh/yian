# Wiring a new authorable leaf node kind

The Node subclass (from `node.py.template`) is only half the job — a kind also
needs a `compose/nodes/<kind>.py` module (its descriptor, which carries `parse`,
source-origin attachment, and `build`) plus one entry in the `_DESCRIPTORS` tuple
in `compose/nodes/__init__.py`,
so a flow author can write `kind: xxx` in YAML. This is the file-by-file checklist
for an **authorable leaf** kind (one a flow names directly, like `code`/`tool`).

> Internal-only kinds (`START`/`END`) are *synthesized* by the loader, not authored —
> they have no `compose/nodes/<kind>.py` module and expand at load/runtime instead. This
> checklist is for a leaf you want authors to write.
>
> A normal authorable leaf returns `Output` or `Route` and skips the conditional
> spawner step below. If the new kind can return `Grow`, it is a spawner as well
> as authorable and must follow §2a; that requirement does not apply to ordinary
> leaves.

Use `tool` (`compose/nodes/tool.py`) as the simplest real leaf reference. Use
`code` only when the new kind also needs semantic source sites beyond the common
node header. Build bottom-up by dependency, **one test green before the next
step** (CLAUDE.md).

## 1. `nodes/base.py` — add the closed-vocabulary tag

Add a member to the `NodeKind` enum — the single source of the tag. It is a closed
vocabulary (no registry/metaclass); the engine core does NOT dispatch on it, so a new
leaf needs no engine change — its behavior is entirely in its own `_run_impl`. A
spawner additionally owns its growth traits/hooks (`is_spawner`,
`grow_depth_delta`, `is_loop`) and the compile interpretation described in §2a.
The other node-owned hooks split into execution-semantic capabilities
(`binds_per_item`, `bind_reserved`, `reserved_wiring_keys`, `needs_stop_event`,
`skipped_output_target`) and the identity-neutral diagnostic projections
`exception_origin` and `failure_trace`. They remain available to either shape.
The base `Node.run` is a concrete template method that fires the node's asserts
around `_run_impl`; do not override it.
`skipped_output_target()` is a pure, deterministic topology hook: ordinary nodes
inherit baked `commit_as` or their own id. A result-carried `Output.commit_as`
redirects success only. If absence must cross that same boundary, override the
hook using only execution state represented in the flow manifest, as LOOP does
with `origin_id`; return an existing node id. The redirect must have exactly one
unfinished physical owner and an inert target—an expanded spawner or an unknown
placeholder with no incoming work, never ready, in flight, or already committed, and with
every outgoing edge still unknown. Do not model an OR join or multiple writers
with this hook.
Programmatic assertions are ordered `Assertion(expression, display, origin)`
carriers, never raw strings or a side text map.
Every non-`None` programmatic node signature is an exact `list` of exact
`Parameter` carriers; never introduce subclasses, alternate containers, a
parallel declaration type, or a source on the parameter itself.
Every enforced programmatic node output is an exact `ValueContract` stored in
`Node.output_contract`; `None` means unenforced. Node-side code reads the
contract's `.type` when it needs the semantic type and retains the carrier when it
clones or rewrites the node. Python callers pass `output_contract=`; do not add
another output-Type field or compatibility property.
`Node.origin` is optional opaque diagnostic provenance and is unrelated to LOOP's
`origin_id`. A programmatic/generated node defaults to `None`; preserve an existing
origin when cloning, and never interpret it as execution state. Nodes expose only
their local origin hooks; they never import `compile.rebase_flow_origins` or
`diagnostics.ResolverTable`, because whole-graph rebasing belongs to `compile`.
If a node adds kind-owned origin slots, validate them at construction and extend
`iter_local_origins()` / `rewrite_local_origins()` explicitly in deterministic
order. Do not discover provenance by reflection or include it in execution
semantics or the manifest.
`Node.origin_context` is separate generated outer-to-inner spawner ancestry. It
starts empty on direct/static nodes and stays outside local origin traversal,
rewriting, manifests, fingerprints, and expansion digests. Expected failure
producers call `node.failure_trace(operation_origin)`, which appends that operation
origin or the node origin to the context, omitting only an absent local suffix and
preserving order/duplicates; never reconstruct this rule in runtime by kind.

```python
class NodeKind(str, Enum):
    ...
    XXX = "xxx"
```

## 2. `nodes/xxx/` — the Node subclass + charter

- `nodes/xxx/node.py` — copy `node.py.template`, rename `XxxNode`, set
  `kind = NodeKind.XXX`, implement `_run_impl` (the base `Node.run` wrapper fires asserts
  around it — do not override `run`).
- `nodes/xxx/__init__.py` — copy `__init__.py.template` (re-export only; the
  package charter per the `structure` skill).

### 2a. Spawner only — seed-first growth construction

Skip this section for a simple leaf. A kind whose `_run_impl` may return `Grow`
must preserve the layer boundary:

- Return only `Grow(seed=<canonical value>, prune=<ids>)`. The node owns and
  documents the seed schema but never constructs a `Flow` and never imports
  `compile`.
- Set `is_spawner = True` and only the node-owned growth traits the generic splice
  needs (`grow_depth_delta` or `is_loop`).
- When a builder bakes `commit_as` onto a generated terminal, its inherited
  `skipped_output_target()` must send absence through the same inert spawner
  anchor. Test both terminal success and a skipped terminal settling the
  spawner's successors.
- Add the kind-aware interpretation to the closed match in
  `compile/expand.py::build_grow(spawner, canonical_seed)`. It validates the seed
  and returns one deterministic, self-contained `Flow` fragment.
- Propagate diagnostic ancestry during that compile-owned construction. Prefix
  every cloned child and generated boundary with `spawner.failure_trace()`; if a
  clone already has generated ancestry, append it after that prefix without
  deduplication. A self-respawning driver copies only `spawner.origin_context`,
  so its local authored origin is appended once by `failure_trace()`. Do not add
  a runtime kind branch or derive ancestry from runtime node ids.
- Add focused tests proving the node returns only the canonical seed and
  `build_grow` reconstructs the expected entry/end ids, topology, and wiring.
- A returned `Flow` owns canonical `Binding` values. Its constructor may accept
  raw builder sources and wraps them once; an existing carrier keeps its origin.
  Never assign a raw value into `fragment.wiring` after construction. When a
  builder rewrites a source, return a new carrier with the original origin.

Do not add a runtime kind branch or a `runtime -> compile.expand` import. The
package-root run boundary injects `build_grow` into the kind-blind runtime for live
growth and replay.

## 3. `compose/nodes/xxx.py` — descriptor (subclass of `NodeDescriptor`)

This one module owns the kind's front-end handle: a frozen `XxxDescriptor` dataclass
that **subclasses `NodeDescriptor`**. The base holds the common contract — the shared
fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + `inputs`/`outputs` + the
`origin`/`range`/`field_ranges` source metadata), the key-legality check, the template
`parse(node_id, cursor)` classmethod, the immutable `with_source_origins(...)`
attachment hook, and the sink derivations (`output_contract` / `sink_params` /
`sink_wiring` / `_finish`). A leaf kind diverges only where it must. Copy
`tool.py`; consult `code.py` only for extra source-origin attachment.

- Declare the kind's ClassVars at the top of the dataclass body:
  - `kind = "xxx"` — the surface string (drives the `KINDS`/`BINDING_FIELDS` keys).
  - `allowed = NodeDescriptor.allow("field_a", "field_b", ...)` — the commons plus this
    kind's own keys (`allow` unions onto the base allowed set).
  - `required = NodeDescriptor.require("field_a", ...)` — the mandatory flat keys.
  - `binding_fields = ("inputs", ...)` — the descriptor fields carrying `${...}` binding
    strings (walked kind-blind by `compose/nodes/bindings.py`). Empty tuple if none.
  - (optional) `keyword_remap = {...}` — author-key → field-name aliases, if any.
- Declare only the **kind-specific** dataclass fields (`inputs`/`outputs` come from the
  base). Give each a type; use `field(default_factory=...)` for mutable defaults.
- **`parse` is inherited** — the base template checks key legality, reads the common
  fields, and records each field's `SourceRange`. Override the `_normalize_cursor` /
  `_validate_fields` hooks only if the kind needs pre-normalization or extra checks
  (see `case.py` / `human_input.py` for examples). Most leaves need neither.
- **`with_source_origins` is inherited as a no-op** — override
  `with_source_origins(cursor, *, site_path, catalog_builder)` only when the kind
  owns semantic source sites beyond the common node header. Return an immutable
  replacement descriptor carrying only opaque `OriginRef` values. Register sites
  while the located cursor exists; never retain the cursor, coordinates, or a
  parallel source map. CODE is the reference: one whole-field fallback plus an
  ordered tuple for exact literal lines.
- The assembly conductor stamps `NodeDescriptor.origin` from the authored node-header
  occurrence after parsing. After desugaring/building, it also stamps source-backed wiring
  values directly on their canonical `Binding`: ordinary `input:` entries plus node-owned
  reserved fields (currently MAP `over:` and WAIT `until:`). Only an ordinary input's key
  stamps a matching `Parameter`; reserved wiring has no formal parameter. A kind must not
  mint these common origins or keep a parallel source map; kind-owned sites use only the
  descriptor hook above.
- **Override `build`** — an instance method `build(self, ctx: BuildContext) -> BuildResult`
  that constructs `XxxNode(...)`, sets the output contract / params / wiring, and returns
  `self._finish(node, wiring)` (which stamps `.origin` plus the legacy ranges and wraps a
  `BuildResult`).
  The inherited `self.output_contract(ctx.registry)` returns a `ValueContract | None`;
  assign that carrier directly to `node.output_contract`. Retain an existing carrier when
  re-exporting a child output; construct a new carrier only when the kind changes the semantic
  type (for example, MAP deriving `list[U]`). The authored `outputs:` field does not change.
  The one input is the immutable `BuildContext`: `ctx.registry` (the `TypeRegistry`),
  `ctx.resolver` (the child-flow resolver — only call/map/loop read it), plus
  `ctx.source_file` and optional `ctx.resolve_origin` (only CODE reads these, as an opaque
  Python traceback label and a pre-finalization Compose presentation seam). Other leaf
  kinds read only `ctx.registry`. Do not add positional line hints or parse source below
  Compose; CODE maps wrapped lines through its source-neutral origin carriers.

`BuildResult.wiring` is still pre-IR Compose data and may remain raw. Final assembly
canonicalizes source-backed ordinary input and reserved wiring entries with their value
origins; the shared `CompiledFlow.from_parts` boundary normalizes any remaining raw wiring.
Do not add a second normalization map to a descriptor.

## 4. `compose/nodes/__init__.py` — add to the `_DESCRIPTORS` tuple

Import the module and add its descriptor class to the `_DESCRIPTORS` tuple. `KINDS`
(kind-string → descriptor class) and `BINDING_FIELDS` are derived from that tuple, so
one entry registers both:

```python
from agent_composer.compose.nodes import xxx
...
_DESCRIPTORS = (
    ...,
    xxx.XxxDescriptor,
)
```

The shared dispatch (`parse/`, `build/pipeline.py::_assemble`'s build loop,
`bindings.py`) looks the class up in `KINDS` and calls `cls.parse(...)`,
`desc.with_source_origins(...)`, and `desc.build(...)` kind-blind — no isinstance
branch to add.

## 5. `compose/__init__.py` — re-export the descriptor

If anything outside `compose/` references `XxxDescriptor`, add it to the imports +
`__all__`, mirroring `CodeDescriptor` (re-exported from `compose.nodes.code`).

## 6. `tests/engine/test_xxx.py` — prove it

Copy `test_node.py.template`. At minimum: a direct `_run_impl()` unit test and one
`load_flow` + `run_flow` end-to-end. A kind with extra source-origin carriers also
tests source-free defaults, deterministic traversal/rewrite, external rebasing,
and manifest/fingerprint neutrality. For a spawner, also test its canonical seed
and the matching `build_grow` arm (§2a). Run:

```bash
PYTHONPATH=src pytest tests/engine/test_xxx.py -q
```

## 7. GitHub issue + docs

Close the tracking GitHub issue when the work lands, referencing the commit/PR
(CLAUDE.md "Zeroth rule"). If the kind is part of the authoring surface, add it to
`docs/syntax.md` and the `composing-agents` skill.

---

### Quick map

| Touchpoint | File | Anchor |
|------------|------|--------|
| Enum tag | `nodes/base.py` | `class NodeKind` |
| Node impl | `nodes/xxx/node.py` + `__init__.py` | new package |
| Growth builder *(spawner only)* | `compile/expand.py` | `build_grow` closed match |
| Descriptor (`NodeDescriptor` subclass) | `compose/nodes/xxx.py` | copy `tool.py`; CODE for extra source sites |
| Register | `compose/nodes/__init__.py` | `_DESCRIPTORS` tuple |
| Re-export | `compose/__init__.py` | imports + `__all__` |
| Test | `tests/engine/test_xxx.py` | new file |
| Issue/docs | GitHub issue, `docs/syntax.md` | — |
