---
name: structure
description: Use when adding a file, creating a folder/package, or deciding where a function or class belongs — any "where does this code go?" question. Enforces the framework — each folder is a self-contained package with a charter in its __init__.py, imports flow one way (lower-level or peer only), and no circular imports. Trigger on "add a … module", "where should I put …", "create a package for …", "this file is getting big", or any new directory.
---

# How to structure the code

A reader (human or model) should be able to open any folder, read its
`__init__.py`, and know **what lives here, what it may touch, and what it must
never touch** — without reading the implementation. Structure is a contract, not
a side effect. Follow this every time you add a file or a folder.

For a copy-paste charter template, a real example, and a "where does it go?"
table, see [`reference.md`](reference.md).

## The one rule that generates the others

**Each folder is a self-contained package. Imports flow one direction: a package
may import only *lower-level* packages or its *peers* — never a higher-level
package, never in a cycle.**

Everything below is a consequence of making that rule legible and enforceable.

## 1. Every package declares a charter in `__init__.py`

The moment you create a folder, write its charter as the `__init__.py` module
docstring — *before* the first real module. A package with no charter is a bug.
The charter has four parts, in this order:

```python
"""<one line: what this package IS>.

<2–4 sentences: its role — the single responsibility it owns and why it exists
as its own package rather than living in a neighbor.>

Knows about:   <the lower-level/peer packages it may import, by name.>
Never imports: <the higher-level packages that import IT — name them so the
               direction is impossible to get wrong. Also list forbidden heavy
               deps, e.g. "no DB, no server, no web framework".>
"""
```

This is not boilerplate. "Knows / Never imports" is the dependency direction
written down where it gets read. When you later wonder "can `typesys` import
`runtime`?", the answer is already in `typesys/__init__.py`. See
[`src/agent_composer/__init__.py`](../../../src/agent_composer/__init__.py) for the
worked top-level charter with the full layer diagram.

After the docstring, `__init__.py` does **only** re-exports: expose the package's
public names and list them in `__all__`. No domain logic or classes are defined
inline. A composition-root facade may use a small standard-library-only
`__getattr__` export table when eager imports would load every higher layer while
someone imports a leaf submodule; cache each resolved value and preserve the
same public API. The `__init__` is the package's public face; everything else is
private until re-exported.

## 2. Order packages into layers; let imports flow leaf-to-root

Before adding a package, place it on the dependency ladder. The engine reads
(leaf → root):

```
diagnostics  <-  events
      ^
      └────────  nodes  <-  compile  <-  compose
                   ^   ^
      typesys + expr    └──────  llm_clients

_state + compile runtime contracts <── runtime
compose + compile.expand + runtime <── package-root run.py
```

`diagnostics.py` is the lowest source-neutral value leaf. It owns concrete
ranges, opaque origins, and the authoring-format-neutral resolution protocol;
`events` imports it. The package facade resolves its declared public exports
lazily so importing this leaf does not bootstrap compose or runtime.

The package-root run boundary is the composition root: it may import compose,
compile's concrete builders, and runtime. Lower layers never import it. **Arrows
never reverse.** In particular, nodes return canonical growth seeds and never
import `compile`; runtime may consume compile's topology, identity, and validation
contracts but never imports `compile.expand`. `run.py` injects
`compile.expand.build_grow` into runtime instead. Runtime owns run-record/log
codecs, append-before-apply reducer state, and dispatch; `run.py` owns fresh-log
creation, locked prefix recovery, and
log closure. Spawner seed schemas are owned by their node packages; the internal
`_state.py` leaf carries the scheduling-state vocabulary shared by compile and
runtime.
`tests/engine/test_layer_direction.py` ratchets these boundaries.
When you add a package, name what it sits *above* and what it sits *below*, and
record that in its charter's "Knows / Never imports".

The Compose front end has its own ownership ladder:

```
compose/source  <-  helpers  <-  nodes  <-  parse  <-  desugar  <-  edges/validate  <-  build  <-  loader
```

`compose/source` exclusively owns PyYAML and the private constructed YAML graph.
It constructs each physical document once as a `SourceDocument` and exposes
stable `SourceCursor` navigation plus immutable `SourceCatalog` resolution of
opaque origins; no other compose package may import PyYAML, inspect private graph
slots, or reconstruct source by dumping and reparsing it.
Source-aware parsing and inline extraction consume documents/cursors through that
public interface; descriptors and nodes carry derived ranges afterward.
Text-accepting parse conveniences delegate construction to `SourceDocument` before
doing semantic work; they do not own a second parser.

One package-private catalog builder is shared across a physical document's root,
definitions, and inline bodies. Assembly registers authored node headers and ordinary
node-input key/value occurrences directly on the canonical `Parameter`/`Binding`
carriers, then stores a frozen snapshot on `ComposedFlow._local_catalog`. That field
stays private and document-local: it is not a complete resolver for embedded external
flows until their origins are lexically rebased.

`compose/loader` owns the state whose lifetime is a load graph or one authored
file. Its resolved-path context guards cycles and caches only successfully
compiled external flows, so aliases and diamonds reuse one physical source load
and compiled flow.
Its file-local callable scope owns authored `defs:`/`uses:`, compiler-private
callables, and their shared success cache. The loader resolves defs and uses eagerly;
only compiler-private builders stay lazy. Recursive local/private construction is
guarded separately from cross-file recursion. A failed callable attempt rolls back
new private builders, nested cache entries, and catalog sites together. These are
distinct caches and guards with distinct keys and lifetimes. Neither `compile` nor
`runtime` knows about YAML, `SourceDocument`, `SourceCursor`, or loader-private callable scope;
they receive ordinary compiled contracts.

Compose-only anonymous children are a front-end ownership example. Located,
read-only `SourceCursor` projection belongs in `compose/desugar`; strict generated
signature proof and child assembly belong in `compose/build`; the file-lifetime
private callable table/cache belongs in `compose/loader`, beside the authored
`defs:`/`uses:` scope it extends. `build` depends only on a small registrar/
resolver protocol, never imports the loader back, and no private-name mechanism
is interpreted by `compile` or `runtime` (the generated string remains ordinary
`flow_id` data).

If you find yourself needing an upward import, you've put the code in the wrong
package — see §5.

## 3. Where a new function or class goes

Ask, in order:

1. **Does an existing package already own this responsibility?** If a package's
   charter covers it, add it there. Don't create a new package for a function
   that belongs in one that exists.
2. **Is it the public behavior of a kind/entity?** Put it in that entity's own
   module (e.g. a node kind in `nodes/<kind>/`), and re-export from the package
   `__init__` if callers outside need it. Kind-specific execution support belongs
   there too: CODE's soft-budget watchdog is `nodes/code/watchdog.py`, not a
   runtime-wide timer around every node.
3. **Is it shared *contract* across the entities of a package** — a dataclass, a
   base type, a registry, a `register_*` decorator, an enum the submodules share?
   Put it in `common.py`. Concrete submodules import from `common`; `common`
   imports from no sibling. This is the standard cure for circular imports.
4. **Is it a pure stateless helper** (no shared state, just a function over its
   args)? Put it in `utils.py`.
5. **None of the above / it's a new responsibility?** It's a new package — go to §4.

`common.py` vs `utils.py` is a real distinction, not a synonym: `common` = shared
**types/contract**; `utils` = shared **helper functions**. A package may have
both (e.g. `nodes/agent/modes/common.py` is the mode contract; a `modes/utils.py`
holds pure helpers).

## 4. When to create a new package (folder)

Create a folder when a responsibility is **distinct, has more than one file's
worth of code, or needs its own charter to state what it must not touch**. A
folder for a single 10-line function is overhead; a folder for "everything about
node X" or "the typed value system" earns its keep.

Checklist when you create one:
- [ ] Write the `__init__.py` charter (§1) first.
- [ ] Place it on the layer ladder (§2); fill in "Knows / Never imports".
- [ ] Confirm no existing peer already owns this (§3.1).
- [ ] `__init__` re-exports only; one entity/responsibility per submodule.

## 5. Never create a circular import

A cycle means two packages are really one responsibility split wrong, or a
contract is living in the wrong place. Fixes, in preference order:

1. **Extract the shared contract** to a lower/leaf module (`common.py`, or a new
   leaf package both can depend on). Both formerly-cyclic packages now import
   *down* into it.
2. **Invert via a seam** — if a lower-level package needs behavior from a
   higher-level one, it should take it as an injected callable/parameter, not
   import it. The high level wires the low level; the low level stays a leaf.
   Seed-first growth is the repository example: runtime takes a `grow_builder`
   callable, and package-root `run.py` supplies compile-owned `build_grow`.
3. **Merge** the two packages if they genuinely can't be separated — better one
   honest package than two that import each other.

Never paper over a cycle with a function-local (deferred) import. That hides a
structural error; fix the structure.

## 6. Self-explaining over commented

The structure should answer "what is this and what may it touch" before anyone
reads a line of logic — via package charters, module docstrings, the layer map,
and `common`/`utils` naming. Reserve inline comments for *why* something
non-obvious is done (per CLAUDE.md "Comments explain why"). If you need a
paragraph of comments to explain *what* a module is, the charter or the file
split is doing too little.

## Quick checklist (run before finishing any structural change)

- [ ] New folder → charter in `__init__.py` (what / role / Knows / Never imports).
- [ ] `__init__.py` re-exports only; `__all__` set.
- [ ] Imports point down or sideways only — no upward, no cycle.
- [ ] Nodes do not import compile; runtime does not import `compile.expand`.
- [ ] Only `compose/source` imports PyYAML or touches the private source graph.
- [ ] Source-aware Compose entry points use `SourceDocument`/`SourceCursor`; no phase reparses source.
- [ ] `compile` and `runtime` remain YAML- and compose-source-blind.
- [ ] Shared types → `common.py`; pure helpers → `utils.py`; entity behavior → its own module.
- [ ] Code added to an existing package's responsibility, not a redundant new one.
