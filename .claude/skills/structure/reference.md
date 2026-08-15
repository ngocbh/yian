# Reference — package charter & layout

## Charter template (copy into a new `__init__.py`)

```python
"""<one line: what this package IS>.

<2–4 sentences: the single responsibility it owns, and why it's its own package.>

Knows about:   <lower-level/peer packages it may import, by name>
Never imports: <higher-level packages that import IT; plus forbidden deps>
"""

# re-exports only — no domain logic, no inline classes
from <pkg>.<submodule> import <PublicName>

__all__ = ["<PublicName>"]
```

## Real example (from this repo)

`src/agent_composer/typesys/__init__.py` — the typed value system + variable pool,
an independent engine leaf (it imports nothing else in the engine; `nodes`,
`compile`, `compose`, and `runtime` import *it*, never the reverse):

```python
"""Typed runtime state: the typed value system + the variable pool."""

from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import TypedValue, build_value
# … (the full module re-exports the typed value + type surface)

__all__ = ["TypedValue", "VariablePool", "build_value"]
```

The worked top-level charter — with the full leaf-to-root layer diagram and the
"nothing here imports a DB or a server" boundary — is
`src/agent_composer/__init__.py`. Read it to see the dependency direction written
down for the whole engine. It is also the deliberate facade exception: a
standard-library-only lazy export table prevents importing the lower
`diagnostics` submodule from bootstrapping compile, compose, or runtime while
preserving the declared package-root API.

## Where does it go? (one-glance table)

| The thing you're adding                          | Goes in                       |
|--------------------------------------------------|-------------------------------|
| Public behavior of one entity/kind               | that entity's own module      |
| Shared dataclass / base type / enum with no stronger domain name | `common.py` |
| Cohesive public contract vocabulary | a named module such as `contracts.py` |
| Pure stateless helper function                   | `utils.py`                    |
| A whole new distinct responsibility (>1 file)    | new package (write charter)   |
| Something an existing package already owns        | that package (don't make a new one) |

Kind-specific support follows the entity row: for example, CODE's in-process
soft-budget watchdog lives in `nodes/code/watchdog.py`. Putting it around every
evaluation in `runtime/` would both misstate ownership and broaden behavior.

## Import-direction smell test

- Importing *upward* (leaf reaches into a root package) → wrong package, or needs a seam.
- Two packages import each other → extract shared contract to `common.py`/a leaf, or merge.
- `from x import y` written *inside a function* to dodge an error → hidden cycle; fix the structure.

## Dependency-inversion example: graph growth

Spawner nodes own the schema of a canonical `Grow(seed, prune)` but do not know how
to compile it. `compile.expand.build_grow` is the only kind-aware constructor, while
runtime knows only the injected `(spawner, seed) -> Flow` callable. Package-root
`run.py` wires them together. This keeps both forbidden edges absent:

- no module under `nodes/` imports `agent_composer.compile`;
- no module under `runtime/` imports `agent_composer.compile.expand` (imports of
  compile's execution contracts remain allowed).

`tests/engine/test_layer_direction.py` parses imports with `ast` and enforces these
rules, including equivalent relative-import spellings. Spawner seed schemas live
with their owning nodes; the scheduling-state vocabulary lives in the internal
`_state.py` leaf because both compile and runtime consume it. Runtime owns the
run-record/log codecs, append-before-apply reducer, and dispatcher, while the
package-root `run.py` owns log lifecycle and recovery. Placing the shared state
vocabulary in a neutral leaf keeps both consumers from owning the other's contract.

## Compose front-end ownership: source once, project thereafter

The Compose front end is an adapter above the representation-neutral compiler:

```
compose/source  <-  helpers  <-  nodes  <-  parse  <-  desugar  <-  edges/validate  <-  build  <-  loader
```

| Responsibility | Owner | Must not leak to |
|---|---|---|
| PyYAML construction and the private YAML node/occurrence graph | `compose/source` | every other compose phase, `compile`, `runtime` |
| Stable location-aware navigation and immutable projections | public `SourceDocument` / `SourceCursor` contracts from `compose/source` | private document slots or PyYAML nodes |
| Semantic descriptors and flow sections | `compose/parse` consuming a document/cursor | another YAML construction pass |
| Source-preserving inline extraction | `compose/desugar` projecting cursors | PyYAML and loader state |
| Compiled-flow assembly | `compose/build` consuming descriptors and retained cursor sources | PyYAML and concrete loader implementation |
| Resolved-file reuse, cycles, and one file's callable namespace | `compose/loader` | `compile` and `runtime` |

Only modules under `compose/source` may import PyYAML or traverse the retained
parser graph. A text-accepting convenience in `compose/parse` must delegate the
single construction to `SourceDocument`. Source-aware parsing and inline extraction
navigate its cursors; descriptors and nodes carry derived ranges afterward. No phase
copies, renders, or reparses YAML to reconstruct source.

The loader owns two intentionally separate caches:

- `_LoadCtx.flows` is keyed by resolved absolute path, shared across one external
  load graph, and populated only after a child has compiled successfully. It
  makes aliases and diamond imports reuse the same physical source load.
- `_CallableScope._built` is keyed by file-local callable name and memoizes every
  successful compiler-private child, authored `defs:` flow, and resolved `uses:`
  alias. Defs load eagerly before the parent and uses resolve eagerly during
  assembly; only compiler-private builders stay lazy. `_in_progress` rejects
  recursive private/def construction, while `_LoadCtx.loading` guards recursive
  external-file resolution.

`compile` owns representation-neutral flow contracts, and `runtime` executes
compiled flows. Neither layer imports Compose source contracts, knows that YAML
was the authoring format, or interprets loader-private names.

## Front-end ownership example: inline composite children

Inline CALL/MAP/LOOP bodies do not earn a runtime package or a new compile
abstraction. Their responsibilities follow the existing compose phases:

| Responsibility | Owner | Must not know |
|---|---|---|
| Located, occurrence-local `SourceCursor` projection and stable lexical private identity | `compose/desugar/inline_bodies.py` | PyYAML/private source graph, later `build`, concrete loader scope, runtime |
| Strict generated input-type proof | `compose/build/inline_inputs.py` | loader state, runtime |
| Association, lazy parent-descriptor dependencies, child assembly | `compose/build/pipeline.py` | concrete loader implementation |
| File-private callable table, one-build cache, recursion guard | `compose/loader.py` | runtime |

`build.pipeline` accepts a narrow registrar/resolver protocol; the higher loader
supplies its concrete file scope. That seam prevents the tempting `build ->
loader` back-edge. Once lowered, the child is an ordinary compiled CALL/MAP/LOOP
child, so neither `compile` nor `runtime` needs to know the private prefix or
authored inline shape.
