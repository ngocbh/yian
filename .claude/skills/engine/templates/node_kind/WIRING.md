# Wiring a new authorable leaf node kind

The Node subclass (from `node.py.template`) is only half the job — a kind also
needs a `compose/nodes/<kind>.py` module (its descriptor, which carries `parse` +
`build`) plus one entry in the `_DESCRIPTORS` tuple in `compose/nodes/__init__.py`,
so a flow author can write `kind: xxx` in YAML. This is the file-by-file checklist
for an **authorable leaf** kind (one a flow names directly, like `code`/`tool`).

> Internal-only kinds (`START`/`END`) are *synthesized* by the loader, not authored —
> they have no `compose/nodes/<kind>.py` module and expand at load/runtime instead. This
> checklist is for a leaf you want authors to write.
>
> A simple leaf returns `Output`, `Route`, or `Pause` and skips the conditional
> spawner step below. If the new kind can return `Grow`, it is a spawner as well as
> authorable and must follow §2a; that requirement does not apply to ordinary leaves.

Use `code` (`compose/nodes/code.py`, the simplest real leaf) as the reference
implementation throughout. Build bottom-up by dependency, **one test green before the
next step** (CLAUDE.md).

## 1. `nodes/base.py` — add the closed-vocabulary tag

Add a member to the `NodeKind` enum — the single source of the tag. It is a closed
vocabulary (no registry/metaclass); the engine core does NOT dispatch on it, so a new
leaf needs no engine change — its behavior is entirely in its own `_run_impl`. A
spawner additionally owns its growth traits/hooks (`grow_depth_delta`,
`grow_restamps_self`, `is_loop`) and the compile interpretation described in §2a.
The other node-owned capability traits (`binds_per_item`, `bind_reserved`,
`needs_llm`) remain available to either shape. The base `Node.run` is a concrete
template method that fires the node's asserts around `_run_impl`; do not override it.

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
  needs (`grow_depth_delta`, `grow_restamps_self`, or `is_loop`).
- Add the kind-aware interpretation to the closed match in
  `compile/expand.py::build_grow(spawner, canonical_seed)`. It validates the seed
  and returns one deterministic, self-contained `Flow` fragment.
- Add focused tests proving the node returns only the canonical seed and
  `build_grow` reconstructs the expected entry/end ids, topology, and wiring.

Do not add a runtime kind branch or a `runtime -> compile.expand` import. The
package-root run boundary injects `build_grow` into the kind-blind runtime for live
growth and replay.

## 3. `compose/nodes/xxx.py` — descriptor (subclass of `NodeDescriptor`)

This one module owns the kind's front-end handle: a frozen `XxxDescriptor` dataclass
that **subclasses `NodeDescriptor`**. The base holds the common contract — the shared
fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + `inputs`/`outputs` + the
`range`/`field_ranges` parse metadata), the key-legality check, the template
`parse(node_id, reader)` classmethod, and the sink derivations (`output_type` /
`sink_params` / `sink_wiring` / `_finish`). A leaf kind diverges only where it must.
Copy `code.py`.

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
  fields, and records each field's `SourceRange`. Override the `_normalize_reader` /
  `_validate_fields` hooks only if the kind needs pre-normalization or extra checks
  (see `case.py` / `human_input.py` for examples). Most leaves need neither.
- **Override `build`** — an instance method `build(self, ctx: BuildContext) -> BuildResult`
  that constructs `XxxNode(...)`, sets `output_type` / `params` / wiring via the inherited
  `self.output_type(ctx.registry)` / `self.sink_params()` / `self.sink_wiring()`, and returns
  `self._finish(node, wiring)` (which stamps `.range`/`.field_ranges` and wraps a `BuildResult`).
  The one input is the immutable `BuildContext`: `ctx.registry` (the `TypeRegistry`),
  `ctx.resolver` (the child-flow resolver — only call/map/loop read it), and the inline-CODE
  traceback anchors `ctx.source_file`/`ctx.code_line`/`ctx.code_field_line` (only the CODE build
  reads them). A leaf kind reads only `ctx.registry`.

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
`bindings.py`) looks the class up in `KINDS` and calls `cls.parse(...)` / `desc.build(...)`
kind-blind — no isinstance branch to add.

## 5. `compose/__init__.py` — re-export the descriptor

If anything outside `compose/` references `XxxDescriptor`, add it to the imports +
`__all__`, mirroring `CodeDescriptor` (re-exported from `compose.nodes.code`).

## 6. `tests/engine/test_xxx.py` — prove it

Copy `test_node.py.template`. At minimum: a direct `_run_impl()` unit test and one
`load_flow` + `run_flow` end-to-end. For a spawner, also test its canonical seed
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
| Descriptor (subclass `NodeDescriptor`) | `compose/nodes/xxx.py` | copy `code.py` |
| Register | `compose/nodes/__init__.py` | `_DESCRIPTORS` tuple |
| Re-export | `compose/__init__.py` | imports + `__all__` |
| Test | `tests/engine/test_xxx.py` | new file |
| Issue/docs | GitHub issue, `docs/syntax.md` | — |
