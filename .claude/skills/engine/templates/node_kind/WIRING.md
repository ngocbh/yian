# Wiring a new authorable leaf node kind

The Node subclass (from `node.py.template`) is only half the job — a kind also
needs a `compose/kinds/<kind>.py` module (its descriptor + `parse` + `build`, all
registered into the shared `KINDS` table) so a flow author can write `kind: xxx` in
YAML. This is the file-by-file checklist for an **authorable leaf** kind (one a flow
names directly, like `code`/`tool`).

> Internal-only kinds (`START`/`END`) are *synthesized* by the loader, not authored —
> they have no `kinds/<kind>.py` module and expand at load/runtime instead. This
> checklist is for a leaf you want authors to write.

Use `code` (`compose/kinds/code.py`, the simplest real leaf) as the reference
implementation throughout. Build bottom-up by dependency, **one test green before the
next step** (CLAUDE.md).

## 1. `nodes/base.py` — add the closed-vocabulary tag

Add a member to the `NodeKind` enum — the single source of the tag. It is a closed
vocabulary (no registry/metaclass); the engine core does NOT dispatch on it, so a new
leaf needs no engine change — its behavior is entirely in its own `run` (and, for a
spawner, the node-owned growth traits/hooks: `binds_per_item`, `bind_reserved`,
`iter_boundary_records`, `grow_depth_delta`, `grow_restamps_self`, `is_loop`, `needs_llm`).

```python
class NodeKind(str, Enum):
    ...
    XXX = "xxx"
```

## 2. `nodes/xxx/` — the Node subclass + charter

- `nodes/xxx/node.py` — copy `node.py.template`, rename `XxxNode`, set
  `kind = NodeKind.XXX`, implement `run`.
- `nodes/xxx/__init__.py` — copy `__init__.py.template` (re-export only; the
  package charter per the `structure` skill).

## 3. `compose/kinds/xxx.py` — descriptor + parse + build + register

This one module owns everything the front end needs for the kind. It carries (a) the
frozen descriptor, (b) `parse(node_id, reader)` — read the located node body, recording
each field's `SourceRange`, (c) `build(desc, *, registry, resolver, ...)` — construct the
runtime node + wiring, stamping `.range`/`.field_ranges`, and (d) `register()` — wire the
`KindSpec` into `KINDS` and the binding fields into `BINDING_FIELDS`. Copy `code.py`.

- Define an `XxxDescriptor` dataclass. Keep the common tail (`node_name` / `depends_on` /
  `runs_after` / `env`) and the located-parse metadata (`range` / `field_ranges`, both
  `compare=False`); add your kind's own fields. Declare `XXX_REQUIRED` / `XXX_ALLOWED` /
  `XXX_BINDING_FIELDS` frozensets at module top.
- Write `parse(node_id, reader)` — field-legality check against `XXX_ALLOWED | _COMMON_FIELDS
  | {"kind"}`, then record each present field's `reader.range_of(key)` into `field_ranges`.
- Write `build(desc, *, registry, resolver=None, ...)` — construct `XxxNode(...)`, set
  `output_type` / `params` / wiring via the shared `build.py` helpers, stamp `node.range` /
  `node.field_ranges`. Leaf kinds ignore `resolver`.
- Add `register()` and call it at import (bottom of the module):

```python
def register() -> None:
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS
    KINDS["xxx"] = KindSpec(descriptor_cls=XxxDescriptor, parse=parse, build=build)
    BINDING_FIELDS["xxx"] = XXX_BINDING_FIELDS

register()
```

## 4. `compose/kinds/__init__.py` — import for the register side effect

Add a bottom-of-file import so the module's `register()` runs at package import
(mirroring the existing `from agent_composer.compose.kinds import code as _code`):

```python
from agent_composer.compose.kinds import xxx as _xxx  # noqa: E402,F401
```

The shared dispatch (`stages/parse.py` `_parse_node`, `build.build_leaf_node`,
`bindings.py`) consults `KINDS` / `BINDING_FIELDS` kind-blind — no isinstance branch to add.

## 5. `compose/__init__.py` — re-export the descriptor

If anything outside `compose/` references `XxxDescriptor`, add it to the imports +
`__all__`, mirroring `CodeDescriptor` (re-exported from `compose.kinds.code`).

## 6. `tests/engine/test_xxx.py` — prove it

Copy `test_node.py.template`. At minimum: a direct `run()` unit test and one
`load_flow` + `run_flow` end-to-end. Run:

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
| Descriptor + parse + build + register | `compose/kinds/xxx.py` | copy `code.py` |
| Register import | `compose/kinds/__init__.py` | bottom-of-file imports |
| Re-export | `compose/__init__.py` | imports + `__all__` |
| Test | `tests/engine/test_xxx.py` | new file |
| Issue/docs | GitHub issue, `docs/syntax.md` | — |
