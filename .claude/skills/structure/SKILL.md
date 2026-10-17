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
[`src/yian/__init__.py`](../../../src/yian/__init__.py) for the
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
defaults                                    (stdlib only; below every package)
diagnostics  <-  events
      ^
      └────────  nodes  <-  compile  <-  compose
                   ^   ^
      typesys + expr    └──────  llm_clients
                                 llm          (beside llm_clients; llm_clients will build on it)

compile runtime contracts <── runtime
compose + compile.expand + runtime <── package-root run.py
compile memory table + memory backends <── package-root memory_host.py <── cli
```

`diagnostics.py` is the lowest source-neutral value leaf. It owns concrete
ranges, opaque origins, the authoring-format-neutral resolution protocol, and
immutable exact routing across resolver namespaces; `events` imports it. The
package facade resolves its declared public exports lazily so importing this leaf
does not bootstrap compose or runtime.

`defaults.py` is a stdlib-only leaf below every package: it holds every predefined
default and limit, each overridable by a `YIAN_<NAME>` environment variable read
once per process. Code reads a value through `defaults.current()` instead of a
module constant; `test_defaults_leaf_imports_only_stdlib` ratchets the leaf.

The package-root run boundary is the composition root: it may import compose,
compile's concrete builders, and runtime. Lower layers never import it. **Arrows
never reverse.** In particular, nodes return canonical growth seeds and never
import `compile`; runtime may consume compile's topology, identity, and validation
contracts but never imports `compile.expand`. `run.py` injects
`compile.expand.build_grow` into runtime instead. Runtime owns run-record/log
codecs, scheduler-state vocabulary, append-before-apply reducer state, and dispatch; `run.py` owns fresh-log
creation, locked prefix recovery, delayed backward handles, and log closure.
A retained handle holds private program/data and memory references, not open
sessions or a lock; its guarded call reopens authority and fresh recorded-snapshot
resources. Spawner seed schemas are owned by their node packages; runtime's
`common.py` owns the scheduling-state vocabulary used by its reducer.
`tests/engine/test_layer_direction.py` ratchets these boundaries.
When you add a package, name what it sits *above* and what it sits *below*, and
record that in its charter's "Knows / Never imports".

Feedback contracts follow the same ladder. `expr/lineage.py` owns ordinary
feedback validation, immutable source/composition routes, and route transfer;
it imports only the standard library and `typesys.values` (the int digit limit).
`nodes/binding.py` owns transient binding capture and never imports `nodes/base.py`
back. `nodes/base.py` owns
`BackwardContext`, `BackwardResult`, and node-facing validation, while
`nodes/contracts.py` owns source-neutral execution scopes, positions, input
connections, and child boundaries within its existing restricted import set.
`compile/execution_layout.py` derives and
atomically applies authored node/flow paths and boundaries without reading memory
declarations or creating another child-cloning mechanism. `runtime/contracts.py` owns
forward captures and backward outcome, failure, and attribution carriers shared by records and
consumers; it never imports runtime log, state, trace, driver, or the package-root
run boundary. `runtime/trace.py` derives accepted execution and producer indexes
from records, preparing sparse deltas for the state reducer and exposing paired
source reads to the evaluator. It also owns accepted terminal identity and
reconstructs detached backward contexts from recorded invocations. It may consume
runtime contracts and log codecs, but never imports the backward driver or
`compile.expand`. `runtime.learning` sits above contracts/log/trace and below
state/drivers. It owns pure root/transfer ordering, full attribution validation,
attempt projections, and owner-context construction. Reducer replay uses those
rules without callbacks or context materialization. `runtime/backward.py` consumes
history in one reverse accepted-outcome walk and owns callback invocation plus
the explicit attempt driver. `runtime.memory_runtime` opens injected sessions,
stages memory proposals, and delivers one detached batch per memory key. Neither reconstructs
graph topology or interprets node kinds. `FlowEngine` schedules forward only;
both drivers use the same state prepare/append/reduce authority. No legacy
auto-learning driver or per-key attempt state remains.
`compile.expand` constructs growth recipes alongside topology; `runtime.state`
validates them through the shared growth boundary before `runtime.trace` joins
them to accepted binding routes. Runtime never dispatches on spawner kind.

Memory conversation ownership follows the same lower-layer boundary.
`memory/contracts.py` owns immutable conversation carriers, `MemoryContent`,
`MemoryContextRequest`, `MemoryContext`, and `TranscriptSource`/`ConversationTurn`.
Memory recall returns text or a rich content contribution; session transcripts and
notes share this interface. Only the admitted context holds a live turn. `memory/conversation.py` owns the pure fold,
strict codecs, and digests. `memory/session.py` owns the concrete
`SessionMemory` backend: views, FIFO leases, atomic history/note publication, and
derived summary state. `memory/summary.py` owns pure whole-turn selection and
summary-cache prefix proofs. The backend depends on those lower modules and
injected stop/projection/summary callables, never on nodes, runtime, compile,
compose, or provider SDKs. The `memory/storage/` package sits below the
memory backends: `sqlite.py` owns single-host SQLite lifetime locking and atomic
finite-JSON replacement for session, episodic, and reflection, and `markdown.py`
owns the versioned `MarkdownStore` document interface and `FileMarkdownStore`. Backends validate semantic candidates and save before
publishing RAM. Storage never imports concrete backends or serializes their live
handles, queues, or callbacks. Optional host-only paths do not reach runtime,
public snapshots, or YAML; the contracts-only facade does not export the adapter.
There is no runtime conversation projection or recovery channel, and the run log
holds no conversation records. Keep mutable backend coordination and stored
conversation state out of the scheduler. Runtime admits a recalled transcript
source only from a key the recorded table marks `conversation_eligible` and
binds authored scope and accepted invocation identity; `eval_node` owns the lazy
turn lifetime through execution/cleanup, never provider message encoding.
`nodes/agent/memory.py` owns stateless API message encoding and effect
publication through the turn. It may import memory contracts and provider message
types, but owns no history or queue. It injects safe projection and the current
unbound model into optional memory-owned context preparation; summary callbacks
run outside the backend lock under its existing lease.
`nodes/agent/live.py` owns transient AGENT observation: streaming a chat reply,
bracketing model/tool calls, and projecting provider usage. It imports `events`
and publishes only through evaluator-supplied sinks; it never records, retains,
or prices anything.
`nodes/agent/llm_values.py` holds the helpers for `llm_config` layers that hold
values (`${...}`): value leaves and references, which lower values still
contribute, the reserved node/flow binding, and its resolution against a pool.
Package-root `memory_host.py` builds the memories a flow declares
(`build_memories`) and owns their lifetime in a `MemoryRegistry`. It reads the
compiled memory table and imports only the selected built-in backend modules.
`cli/memory.py` parses `--memory-root` and `--memory-storage KEY=PATH` and calls
it; `cli/run.py` passes the registry to the run and closes it only after engine
work drains. Storage flags remain host-only; runtime never imports either module
or constructs memory. `cli/mcp_config.py` reads `~/.yian/mcp.json` and each
`--mcp-config` file (`mcpServers` JSON) into unopened `McpServer`s; `cli/run.py`
passes the servers the flow grants to the run unopened, as tool groups, and the run
opens them after its own checks and closes them when it ends. The CLI opens servers
itself only to probe for an unresolved tool name.

The `memory` facade exports contracts only: concrete session, episodic,
reflection, and profile backends must be imported from their own modules, not eagerly from
`memory/__init__.py`. Keep mutable queue/lease state private to the backend;
public snapshots contain conversation values, not coordination objects. The
memory import ratchet in `tests/engine/test_layer_direction.py` enforces this
direction, including direct-facade and relative-import spellings.

`skills/` is a leaf beside `memory/`: the frozen `Skill` value, the `load_skill`
tool schema and listing, and confined bundled-file reads. It imports only
langchain-core and the standard library, never nodes, compile, compose, runtime,
`run.py`, or CLI (`test_skills_leaf_imports_no_engine_or_front_end_layer`).
Reading skill folders parses YAML frontmatter, so it lives in `compose/skills.py`
and uses `compose.source.literals.parse_yaml_literal`.

`tools/` is a leaf beside `skills/`: `registry.py` owns `register_tool`, the
global registry, and the per-run `ToolSet`; `types.py` reads a tool's declared
argument and return types as yian `Type`s for the run-start TOOL check.
`context.py` holds `ToolContext`; `docstrings.py` parses tool docstrings (stdlib
only); `schema.py` builds a tool's argument model from a function signature;
`function_tools.py` holds `FunctionTool` and `function_tool`; `errors.py` holds
`ReportedToolError`, whose message an AGENT's model reads as the call's result,
`PermanentToolError` and `TransientToolError`, which steer a TOOL node's retries, and
`ToolCheckError`, raised when a check yian runs on a call fails (stdlib only).
`mcp.py` holds `mcp_server`, the `McpServer` it returns (the MCP SDK client on its own
thread and loop, exposing the listed tools as `FunctionTool`s), and the `McpError`
classes; it imports the MCP SDK and httpx2 only when a server opens.
`tools/` imports langchain-core, pydantic, `typesys`, the skills leaf, and
`yian.defaults`, never nodes, compile, compose, runtime, `run.py`, or CLI. The run
boundary compares the types; `tools/` only reads them.

`llm/` is a leaf beside `llm_clients/`: yian's provider-neutral message, stream
and error vocabulary, an httpx2 transport that refuses redirects, SSE and NDJSON
framing, auth objects, routes, the executor, and test helpers. It imports only the
standard library, httpx2, and `yian.defaults`, never another yian package, a
provider SDK, or langchain (`test_llm_leaf_imports_only_declared_leaves`).
`llm_clients` will build on it; it never imports `llm_clients`.

`llm_clients/auth/` is a private authentication leaf below model construction and
host settings. It owns immutable credential/selection carriers, provider login
and refresh protocols, and private atomic storage. It imports no model SDK,
CLI, node, memory, compiler, or runtime module. Callers inject presentation and
HTTP effects; only the CLI prompts, and only an explicit save activates staged
credentials and the provider's model; it becomes the default provider only when
none is saved. Refresh preserves the selected
connection identity. Keep tokens out of model configuration and history.
`llm_clients/connection.py` combines saved selections with environment defaults
and supplies pinned request-time credential callbacks. The run host owns the
snapshot; compiler resolution stays pure and receives provider-scoped defaults.
`llm_clients/layers.py` checks one `llm_config` layer (`validate_layer`), resolves an
agent's merged layers with the host defaults and saved models (`resolve_agent_config`),
and checks a complete or recorded config (`require_concrete_selection`); the
compile-time cascade calls it per agent.
`llm_clients/connection_settings.py` checks the connection settings that travel
with a key: OpenAI organization and project ids and extra request headers (names,
values, refused names, caps). It imports only pydantic, the standard library, and
`yian.defaults`, never `config.py` or `auth/`, so the `LLMConfig` carrier, the auth
store, `yian auth`, and resume all share it.
`llm_clients/model_list.py` lists a host's models with only the key and connection
settings passed in, over the `auth/http.py` transport. It imports no model SDK, so the
OpenAI-compatible default endpoints live in `factory.py` beside the routes, where
both it and the OpenAI client read them.
`llm_clients/provider_blocks.py` holds the `llm_config` provider-block schemas: one
pydantic class per provider API (`PROVIDER_BLOCKS`) with its reserved paths,
`block_schema` (the class for the API a call actually reaches, such as the ChatGPT
subscription endpoint), and `merge_request_body` (how a client sends a block as
written). It imports only pydantic and the standard library; `config.py` builds
`LLMConfig`'s block fields from it and the client wrappers send the blocks.

Compiled-origin traversal follows the same direction. Semantic carriers and
nodes own their local origin slots and static-child hooks; `diagnostics` owns the
exact resolver table; and `compile/origins.py` owns flow wiring, outputs,
identity guards, copying, recursion, and graph-wide rebasing into that table.
Nodes never import the walker, and the walker never inspects arbitrary payloads
or assigns meaning to opaque source keys.

The Compose front end has its own ownership ladder:

```
compose/source  <-  helpers  <-  nodes  <-  parse  <-  desugar  <-  edges/validate  <-  build  <-  loader
```

`compose/source` exclusively owns PyYAML and the private constructed YAML graph.
It constructs each physical document once as a `SourceDocument` and exposes
stable `SourceCursor` navigation plus immutable `SourceCatalog` resolution of
opaque origins. Cursor-owned `ScalarSource` metadata exposes whole-scalar fallback
and exact literal constructed-line ranges; catalog sites retain semantic line indices,
never coordinates. Its catalog module owns both typed document-local key encoding
and the package-private lexical `uses:` key scope; scoped child keys stay opaque,
and exact membership does not replay source. No other compose package may import
PyYAML, inspect private graph slots, or reconstruct source by dumping and
reparsing it.
Source-aware parsing and inline extraction consume documents/cursors through that
public interface. Descriptors retain derived ranges inside Compose, and
`NodeDescriptor._finish()` copies node/field locations to the Compose-owned
`BuildResult` for validation. Semantic nodes receive only source-neutral origins;
no YAML range crosses assembly.
Text-accepting parse conveniences delegate construction to `SourceDocument` before
doing semantic work; they do not own a second parser.

One package-private catalog builder is shared across a physical document's root,
definitions, and inline bodies. `compose/helpers/types.py` owns the one source-aware
recursive type compiler. For retained root/definition flow inputs, declaration keys
stamp `Parameter.origin`, type values and authored nested record fields
populate `ContractOrigins`, and the default written after the tag stamps
`default_origin`; START reuses those carriers. Scalar alias/list type expressions fall
back to the contract root. Python/source-free declarations and inferred inline
parameters remain source-free.
`compose/source` records each local YAML tag and constructs the node as if untagged;
`compose/helpers/tags.py` is the one reader every slot uses (`read_tag_type`,
`read_tagged_value`, `suggest_tag`, and the ledger that makes a tag outside a slot a load
error). Nested tags and nested `${...}` become a `ValueTemplate`, the carrier
`nodes/contracts.py` owns with its shared string walkers (`iter_source_strings`,
`map_source_strings`, `binding_input_groups`); `nodes/binding.py` evaluates it.
`typesys/types.py` owns the parenthesized type grammar.
`compose/helpers/output_contracts.py` separately owns source-neutral binding-template
result contracts: exact-or-partial direct-reference projection/provenance, proven
computed types, and exact-or-partial closed-record assembly. Node resolution, inline
input inference, and validation import this lower helper; it never owns YAML navigation
or graph-stage policy. `ChildSignature` and the producer map preserve the source-neutral
`OutputContract` sum owned by `nodes/contracts.py`; no temporary Compose metadata becomes
a second runtime authority.
`compose/helpers/isinstance_types.py` collects the type names `isinstance` calls use and
resolves them against the owning flow's `typedefs:`; `compose/build` stores the result on
`Node.types` after the START/END boundary is synthesized, so the evaluator never sees a
registry. `isinstance_types.py` also decides which union members an `isinstance` check
narrows to. `compose/helpers/condition_guards.py` (syntax only) says what a condition
proves at each read and check through `and`/`or`/`not`, and
`compose/helpers/narrowing_facts.py` turns that into narrowing facts, narrowed read types,
and impossible-check errors; CASE, LOOP, and node and flow asserts share both, which is why
they sit in `helpers` rather than `validate`. `compose/validate/narrowing.py` computes union
narrowing from built CASE descriptors and edges and hands each node a view of the producer
and flow-input contracts; the reference, call-type, and exhaustiveness validators read that
view, while the field-read rule itself stays in `nodes/contracts.py`. Narrowing is load-time only and adds
nothing to runtime.
`compose/validate/skips.py` models which nodes of one flow level CASE routing may skip,
mirroring the runtime join rule over compiled edges (`case_exhaustive.exhaustive_cases`
says which CASEs always take a branch). `compose/build` uses it to give a flow output
whose producers may all be skipped the explicit null default and to reject arithmetic
over a maybe-skipped value; runtime is unchanged.
`compose/validate/llm_config_sources.py` checks the sources of `llm_config` values where
they are written (AGENT, composer and flow layers): each needs a proven type or a tag, and
only a type disjoint from its field (`provider`, `model`, a simple provider-block field,
a whole record or block) is a load error.
For authored node outputs, `compose/build` injects a Compose-owned optional reader through
`BuildContext`; the base descriptor invokes it only when an output-authoring leaf reaches
its established contract step. This preserves earlier kind-owned validation order while
letting the reader select the normalized physical `output:` cursor. Its default origin owns
the declaration root, nested record fields own exact paths, and scalar alias/list interiors
fall back to the nearest authored ancestor or root. Compact roots and flat definitions use
source-aligned projections that retain top-level `output:` for their synthesized node.
Direct/virtual builds and generated contracts without an authorable type occurrence remain
source-free. External aliases use the existing lexical graph rebasing for every retained
carrier. Assembly separately registers
authored node headers, ordinary node-input keys and bound values, source-backed reserved
MAP/WAIT wiring values, and source-backed flow-output values directly on the canonical
carriers. A kind-blind descriptor hook registers
additional kind-owned sites while the cursor is available; CODE uses it for the whole
`code:` scalar and exact literal constructed lines, then carries only opaque origins below
Compose. After those registrations, CODE may resolve one load-time syntax site through the
builder's package-private read-only lookup. That lookup replays the retained document without
copying, freezing, or reparsing and never installs a resolver. Assembly also registers each authored
flow/node assertion by source-order index; the retained item cursor supplies its
`Assertion.origin`, so equal text at different positions stays distinct. Validation may use
the cursor's concrete range inside Compose, but no aligned range enters the semantic node.
Full roots and multi-node definitions retain flow assertion cursors and reuse
the lowered carriers on START/END. Compact roots and flat single-node definitions project
their assertions onto the synthesized node instead. Flow-input declaration keys and ordinary
node-input keys belong to `Parameter`; ordinary node-input, reserved, and flow-output bound
values belong to `Binding`.
Flow outputs
are stamped after rewrites and reused by END; output mapping keys stay origin-free.
Generated compact-root and single-node-definition reexports have no authored binding
occurrence and stay origin-free. This flow-output `Binding` rule is separate from the
physical `output:` contract provenance retained by the compact/flat node projection.
Without a retained occurrence, Compose mints no origin and
preserves any explicit programmatic origin. Private assembly leaves the resolver unset while
its shared catalog grows. The loader validates and lexically rebases external children per alias, performs the
only successful-load catalog freeze at the physical-file boundary, and installs the exact
`ComposedFlow.origin_resolver`; source-free graphs use `None`.
`ComposedFlow` retains no parallel source text, labels, or range maps: runtime
presentation reaches authored documents only through that resolver.

`compose/loader` owns the state whose lifetime is a load graph or one authored
file. Its resolved-path context guards cycles and caches only successfully
compiled external flows with complete resolvers, so aliases and diamonds reuse one
canonical physical source load and compiled flow.
Its file-local callable scope owns authored `defs:`/`uses:`, compiler-private
callables, and their shared success cache. Each external alias is a rebased graph copy;
repeated calls through that alias reuse the copy without mutating the path cache. The
loader resolves uses, then defs, eagerly; only compiler-private builders stay lazy.
Recursive local/private construction is
guarded separately from cross-file recursion. A failed callable attempt rolls back
new private builders, nested cache entries, and catalog sites together. These are
distinct caches and guards with distinct keys and lifetimes. Neither `compile` nor
`runtime` knows about YAML, `SourceDocument`, `SourceCursor`, or loader-private
callable scope; they receive ordinary compiled contracts.

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
   there too, not in a runtime-wide hook around every node.
3. **Is it shared *contract* across the entities of a package** — a dataclass, a
   base type, a registry, a `register_*` decorator, an enum the submodules share?
   Put it in `common.py`. Concrete submodules import from `common`; `common`
   imports from no sibling. This is the standard cure for circular imports.
4. **Is it a pure stateless helper** (no shared state, just a function over its
   args)? Put it in `utils.py`.
5. **None of the above / it's a new responsibility?** It's a new package — go to §4.

A whole-graph operation over the compiled IR belongs with `compile`, not in one
node kind or in runtime. Origin traversal is the concrete example: nodes expose
their own fields and children, while the compile walker joins those views without
reflection or `NodeKind` dispatch.

`common.py` vs `utils.py` is a real distinction, not a synonym: `common` = shared
**types/contract**; `utils` = shared **helper functions**. A package may have
both. The same split can use domain names: `nodes/agent/context.py` holds the
invocation contract and `nodes/agent/text.py` the shared helpers.

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
   leaf package both can depend on), then have both packages import *down* into it.
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
