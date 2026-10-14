# yian — the engine

The implementation of **Yian**: a from-scratch flow engine. A flow is
authored as **Docker-Compose-shaped YAML**, read by the `compose/` loader into a
compiled IR, and executed by the engine's own scheduler — there is no compilation
down to a third-party graph runtime.

The public terminology and node/flow/child-flow distinction are defined in
[`docs/agent.md`](../../docs/agent.md).

The engine is, at heart, a **small functional programming language**—one whose
leaf computations may be LLM-backed AGENT nodes. If you know OCaml, you already
know the shape of it:

- A **flow is a function**: typed inputs in and one typed output value out. An
  **agent** is the broader conceptual unit—a composable transformation that may
  be represented by one node, a flow, or a child flow. The concrete **AGENT
  node** (`kind: agent`) is the LLM-backed leaf.
- Flows **compose**. CALL, MAP, and LOOP apply child flows, so a large
  transformation is built recursively from small ones—function composition, not
  a monolith.
- Computation is **pure at the boundary**: a node returns an outcome and the
  engine binds it; a node never receives or mutates the variable pool. Each kind
  implements `forward`; the base `Node.run` template applies defaults and fires
  record-only asserts around that body. A flow's own asserts are lowered onto its
  synthesized START/END boundary nodes, so top-level and nested-flow asserts use
  the same ordinary node path. `Node.backward` accepts recorded context and an
  ordered feedback tuple, returning input, self, and variable channels. Its base
  implementation returns empty channels; structural nodes own their routing rules.
  AGENT backward reads one live field, the unrecorded host `api_key` in its node's
  `llm_config`, because the record never holds the key.
- The **program structure is fixed by the author**, as a function's call graph is
  fixed by its source. The LLM fills leaf boxes; it does not rewrite the graph at
  runtime. The authoritative record prefix makes engine state recoverable; it
  does not make arbitrary external effects exactly once.

## Layout

```
diagnostics.py semantic failures, resolved presentation, origins/resolver routing [leaf]
json_codec.py JSON text for values holding ints of any size; stdlib only     [leaf]
typesys/      typed value system (values) + variable pool                 [leaf]
events.py     transient run/node observer vocabulary; imports diagnostics
expr/         one ${...} grammar: parse -> AST -> evaluate / infer type / ref-walk
memory/       memory contracts, pure conversation fold/codecs, optionally durable backends
skills/       host-supplied SKILL.md values, confined file reads, load_skill [leaf]
nodes/        pure Node contract + per-kind implementations; spawners return seeds
              (start, end, agent, code, tool, case, call, map, loop,
               human_input, wait)
compile/      compiled IR + validation; identity-safe origin traversal/rebasing;
              growth builder; stable identity; static-path LLM config maps
compose/      Compose-YAML front end: text -> CompiledFlow; source/ owns YAML
              construction/navigation and opaque-origin resolution; later phases
              lower file-private children; skills.py reads skill folders
              (source/helpers/nodes/parse/desugar/edges/validate/build/loader)
runtime/      run records/logs, scheduler-state vocabulary, reducer-owned state,
              forward dispatcher/worker pool,
              explicit backward driver and pure learning/context projections
run.py        public run/recovery boundary and retained RunResult.backward handle
memory_host.py build_memories/MemoryRegistry: host-side construction of declared memories
llm_clients/  provider clients, authentication, LLMConfig, model_from_config, model listing
tools/        tool registry + per-run ToolSet (run tools and the ask_user handler)
cli/          `yian` commands + main-thread terminal broker
_settings.py  environment-based default provider/model
```

The main import direction is acyclic: `diagnostics` is the lowest
source-neutral leaf; `events` imports it, and nodes may combine it with the
independent `typesys` and `expr` leaves before `compile <- compose`. Runtime
consumes the event leaf plus compile's topology, identity, and validation
contracts, but never imports `compile.expand` or `compose`. Its `common.py`
module owns the scheduler-state vocabulary used by the reducer.
Compose imports `SourceRange` directly from `diagnostics`; `events` does not
re-export that front-end presentation coordinate.
`json_codec` imports only the standard library and sits below every package; any
layer that reads or writes value JSON may import it.
The package facade lazy-loads its declared root exports, so importing a leaf
submodule does not bootstrap those higher layers.
Package-root `run.py` sits above `compose`, `compile.expand`, and `runtime`: it
consumes a `ComposedFlow`, injects the kind-aware growth builder, and owns run-log
creation, recovery, and delayed backward resource ownership. None of those
packages imports it back. `memory_host.py` sits beside `run.py`: it reads
`compile`'s memory table and constructs `memory/` backends, which `memory/`
itself may not do.

`diagnostics.Failure` is the canonical source-neutral semantic failure value. It
is a frozen product of `code`, `message`, optional execution `node_id`, an
outermost-to-innermost `tuple[OriginRef, ...]` trace, optional Python
`cause_type`, and optional debugging `traceback`. The exception class is retained
only as diagnostic data; consumers must branch on the closed semantic code,
never on `cause_type` or traceback text. `Failure` is imported from the leaf
module inside lower layers and is lazy-exported from the package root for public
result consumers.

`diagnostics.Diagnostic` is the frozen, resolved presentation product:
`code`, `message`, ordered `DiagnosticFrame` values, free-standing `notes`, and
the optional `node_id`, `cause_type`, and `traceback` copied from a failure. It
has no execution meaning and owns no source-resolution behavior. For a
Compose-authored graph, `ComposedFlow.diagnose(failure)` is the sole front-end
seam that turns a `Failure` into this value. It reads only the failure and the
flow's `OriginResolver`, resolves the complete trace in its existing
outermost-to-innermost order, and preserves duplicate origins. Resolution is
atomic: an empty trace yields an ordinary unlocated diagnostic, while any
missing, incomplete, malformed, or raising resolver for a non-empty trace yields
no frames and exactly the generic note `Source presentation was unavailable.`;
semantic and debug fields remain unchanged.

Compose load failures use the same presentation product directly. A
`LoadError` contains exactly one `Diagnostic`; a Compose-owned factory resolves
retained document/cursor ranges into source-owning frames before the exception
crosses the front-end boundary. Definition and import wrappers create a new
diagnostic with outer context prepended. They never mutate an inner error or ask
the CLI to combine coordinates with separately supplied source text.

Every node builds failures through one source-neutral rule:
`Node.failure_trace(operation_origin=None)` returns its generated, outermost-to-
innermost `origin_context` followed by the supplied operation origin, or by the
node's own origin when no operation origin is supplied. Only an absent local
suffix is omitted; order and duplicates are preserved. Direct and static nodes
start with an empty context. The context is diagnostic-only generated ancestry,
not authored provenance: origin traversal and rewriting ignore it, and manifests,
fingerprints, and expansion digests exclude it.

Compile-owned CALL/MAP/LOOP expansion prefixes cloned children with the
spawner's complete `failure_trace()`. MAP's generated boundaries receive the
same context. A LOOP respawn instead copies only the driver's outer
`origin_context`, so the loop's own origin is appended exactly once when the
next driver builds a failure. Evaluator, dispatcher, and terminal-unreachable
failure producers remain kind-blind and call the same node-owned rule. Runtime
results expose only the canonical `Failure`; runtime and load failures both reach the same
`render_diagnostic()` through `ComposedFlow.diagnose()`. Missing or broken source
resolution therefore stays unlocated instead of falling through to runtime-id or
graph inference.

The closed codes are `binding.evaluate`, `binding.type`, `contract.input`,
`assertion.failed`, `node.execute`, `contract.node_result`, `contract.output`,
`contract.route`, `contract.growth`, and `graph.terminal_unreachable`. They cover
expected flow-owned rejection. `EvaluationFailure` owns exactly one `Failure`. The
evaluator assigns its first six codes by
operation stage: expression/default/required binding, typed-binding checks,
supplied-input contracts, base-wrapper assertions, node execution, and result
validation. Each producer
passes its best operation origin, if any, through `Node.failure_trace()`; the
node origin is the local fallback after any generated ancestry. Host control,
stream callbacks, unexpected binder faults, and malformed compiled/setup state
escape this expected-failure sum.

The dispatcher assigns the next three codes structurally from the closed result
sum: rejected `Output`, `Route`, and `Grow` data become `contract.output`,
`contract.route`, and `contract.growth`. Output rejection uses the effective
target contract origin selected by the rejected value's structural
`TypeCheckError.path` (exact site, nearest ancestor, then contract default), then
the executing node origin. Non-type canonicalization failures use the contract
default. Route and growth use the executing node origin. The trace is empty only
when generated ancestry is empty and neither the operation nor node fallback
supplies a local origin. Unsupported growth kinds and arbitrary builder, compiler,
spawner/fragment-copy, digest, topology-materialization, reducer, append, or log
faults escape rather than becoming flow failures.

At the failure boundary introduced in schema 7, every diagnostic trace item
becomes an exact base `OriginRef(key)`: front-end-specific subclasses are valid in memory, but class
identity is intentionally absent from the single-key wire. The reducer carries
that canonical immutable `Failure` through `NodeFailed`, terminal `RunFailed`,
and `RunResult.failure` without flattened aliases; live carriers share the same
object, while recovery reconstructs an equal canonical value. A route that first
skip-floods END latches
`graph.terminal_unreachable` with the causal router and its `failure_trace()`. That
`NodeRoutedRecord` persists the derived failure; replay structurally revalidates
the route and restores the recorded value, including its complete trace, even if
the supplied graph now carries rebased origins. No additional terminal record
is added, and an explicit node failure still supersedes the latch. A flow from
`load_flow` never reaches this latch through its outputs: the loader gives an output
whose producers a CASE may skip the explicit null default (`${x.output:-null}`), so its
END edge is optional. The runtime path remains for hand-built graphs.

### OpenAI model transport

`llm_clients/openai_subscription.py` constructs a direct ChatGPT subscription
model through the normal LangChain chat interface, with injected access
credentials and optional caller-owned HTTP clients. It performs no login,
credential persistence, tool dispatch, or conversation storage. The OpenAI provider
selects this transport for saved subscription authentication;
API-key authentication uses the same Yian model/tool/memory pipeline.

`llm_clients/auth/` owns staged login, private atomic credential storage, token
refresh, and authenticated model discovery. `yian auth openai` activates credentials
and the provider/model default together. `llm_clients/connection.py` snapshots
nonsecret method, credential identity, endpoint, and store location into the
effective model map. Request-time callbacks resolve credentials and refresh
tokens for that identity. Reconnection or a changed environment key fails a
pinned request rather than changing its account or billing method.

The fresh-run host reads environment and saved defaults once. Model defaults
are provider-scoped, with environment overrides above saved choices; explicit
authored fields retain their existing per-field precedence. Recovery and
backward calls retain the recorded connection. OpenAI records predating this
metadata retain their environment API-key semantics, never today's subscription
selection. Connection metadata is internal and is not an authored YAML field.

`LLMConfig.api_key` is a `SecretStr` excluded from every dump, and `LLMConfig`
validation errors hide input values. Clients use it before the provider's key
environment variable. `run_flow` pops the run layer's `api_key` before the cascade
and `ConnectionSnapshot.bind` marks the effective config of every static agent of
the run layer's provider with the nonsecret `_host_key: true` (OpenAI agents also
record a `host` connection with no credential identity). `bind_api_key` then puts
the key only on the run-local `llm_config` of marked agents, so `RunStarted`,
events, results, and manifests never hold it. `resume_flow(api_key=...)` and
`RunResult.backward(api_key=...)` bind a re-supplied key the same way on the
replayed program; AGENT backward takes it from its live node config, the one live
field it reads. `split_model_config` fails a marked config that has no key, so
agents marked for the key never fall back to the environment. A key that marks no
agent is dropped from the result and ignored by replay. Providers can echo a
rejected key, so `AgentNode` and `ProfileMemory.learn` scrub the key and its
masked fragments from their errors (`llm_clients.redact`) before the kind-blind
runtime records the failure. Authored layers cannot set `api_key` or `_host_key`,
because they reach the recorded effective map; the loader rejects a flow- or
def-level `api_key`. The loader checks flow- and def-level `llm_config:` against
`LLMConfig` as it does a node's, and the cascade rejects unknown keys in the host
layer, so a typo never reaches the recorded map.

Request settings live in provider blocks: `LLMConfig` has one optional block per
provider, named after it (`openai`, `qwen-cn`, `anthropic`, ...), beside `provider`,
`model`, `api_key`, and `base_url`. `llm_clients/provider_blocks.py` holds one class
per API (`PROVIDER_BLOCKS`), each mirroring the request body of the API yian calls
for that provider, with the same names, nesting, and allowed values, plus a table
of reserved paths that yian builds itself (messages, system prompt, tools,
structured output, streaming, conversation state, reply count). A block is checked
in three places. The loader validates every authored layer against `LLMConfig`, so
every block, used or not, is checked at load. The cascade (`_validate_layer`)
rejects unknown and retired keys in the host layer, and `merge_llm_config` merges a
block per block field, one level deep, replacing nested values whole. At run start
`check_model_config` re-checks each agent's resolved block against the schema of
the API the call actually reaches (`block_schema`): an `openai` agent on a ChatGPT
subscription connection without a host key is checked against
`OpenAISubscriptionBlock`, which has no `temperature` or `max_output_tokens`. The
clients send the block as written (`merge_request_body`), with no translation and
no defaults except Anthropic `max_tokens`, so a limit specific to one model surfaces
as the provider's or client library's error. Retired keys (`temperature`,
`anthropic_effort`, `openai_reasoning_effort`, `google_thinking_level`) fail with
their block form (`retired_key_error`); recorded runs may still carry them as null.

`LLMConfig.base_url` is where a config's key is sent, so a host key is never sent
to a `base_url` from an authored layer. This covers authored layers only: when the
host passes a key without `base_url`, the provider's default endpoint or its
environment variables (`OPENAI_BASE_URL`, `VLLM_BASE_URL`, `ANTHROPIC_BASE_URL`,
and so on) pick the endpoint, as before, and some clients
read them on every call, including on resume. A host that wants a fixed endpoint
passes `base_url` with the key. `LLMConfig` accepts it only beside `api_key` and
`provider`, requires `https` (`http` only for local hosts) with no credentials,
query, or fragment, checks the exact string it stores (`validate_base_url` rejects
whitespace, control and non-ASCII characters, bad ports, and over-long URLs rather
than normalising them), never echoes it in errors, and rejects it for `ollama`.
The `anthropic` and `google` clients, whose key headers httpx re-sends on a
cross-origin redirect, do not follow redirects from a `base_url`. An unset
`base_url` is dropped from dumps, so existing records and snapshots are unchanged.
`run_flow` pops it with the key (`pop_host_credentials`) and
`ConnectionSnapshot.bind(host_base_url=...)` writes it into the recorded effective
config of each marked agent; an OpenAI agent's `host` connection takes it as its
endpoint, and `split_model_config` fails a `base_url` that does not match a host
connection's endpoint. The loader, `AgentNode`, and the cascade reject an authored
`base_url` (`host_only_field_error`), and the cascade fails an effective config
whose `base_url` lacks the `_host_key` marker, so a Python-built or mutated flow
cannot redirect the key. It also re-checks a recorded `base_url` with
`validate_base_url`, so an edited log fails when resume starts. Resume and
backward reuse the recorded `base_url` with the re-supplied key; `ProfileMemory`
takes one beside its own key; `ReflectiveMemory` rejects it, except an Azure
endpoint.

`provider: azure` is the one exception. Its `base_url` is the resource endpoint and
its `model` the deployment name, and the client reads neither from the environment
(`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT_NAME` are ignored). Any layer,
including a flow file, may set `base_url` beside `provider: azure` without a key
(`host_only_field_error` allows it, and `pop_host_credentials` leaves it in a keyless
host layer), because the environment key it goes to is already readable by flow code.
The cascade bakes it into the recorded effective map, so resume reuses it, and
`check_model_config` fails an Azure config without one before the first node runs.
A host Azure `api_key` still needs its own `base_url` (`pop_host_credentials`), which
`bind` records over any authored one, so a host key never goes to an authored
endpoint. `ProfileMemory` and `ReflectiveMemory` take a keyless Azure `base_url` in
their own `llm_config`.

The transport requires a completed Responses event before returning a model
reply. Early EOF and incomplete replies raise rather than exposing tool calls
for execution. Cooperative cancellation closes the stream; network operations
have an explicit timeout. Access-source refresh/locking remains the caller's
responsibility. The OpenAI dependency is pinned to the version tested against
these request and streaming hooks.

Both the direct transport and the existing OpenAI API-key client preserve
Responses content blocks, including opaque continuation data. Agent output and
evaluation evidence extract visible text separately; SessionMemory retains the
full provider message through its existing codec. The API history projection
preserves Responses function-call blocks only when they match the ordered,
canonical tool calls exactly. Ambiguous calls remain inert context. Projection
version 2 invalidates summaries made with the older block projection.
Summary extraction accepts validated reasoning blocks alongside visible text,
and still rejects tool calls, unsupported content, and empty summaries.

## Semantic IR carriers

`nodes/contracts.py` is the source-neutral vocabulary that joins opaque
provenance from `diagnostics` with semantic `typesys.Type` values. Each class has
one responsibility:

| Carrier | Meaning |
| --- | --- |
| `Binding` | one flow-owned input or output expression/literal, with an optional declared `ValueContract` |
| `Parameter` | one formal input, its optional contract, and default/coercion policy |
| `Assertion` | one executable expression plus stable author-facing display text |
| `ValueContract` | one recursive semantic `Type` plus nested origin selection |
| `PartialRecordContract` | one closed record shape with optional field-local value contracts |
| `ContractOrigins` | immutable root/nested origin lookup for a value contract |

For a refined specialized list, `Type.element.nullable` permits `None` alongside
the element's ordinary values; a bare or non-nullable specialized list still
rejects it. Existing `LIST_ANY` and explicit `None`-element semantics remain
unchanged. The nullable specialized list retains its `LIST_*` discriminator and
carrier through canonical value, pool, and run-record round-trips. Cross-flow
compatibility recursively applies the same nullable-source-to-strict-sink rule
to list elements. The singleton `None` source is compatible with every nullable
sink, matching runtime coercion, but remains incompatible with a strict non-`None`
sink.

All origins are optional, so Python-authored values default to source-free and
may supply an opaque origin explicitly. The carriers are frozen only at their
fields: arbitrary binding/default payloads are retained by identity.
`iter_origins()` visits only declared provenance slots, while `map_origins()`
transforms those slots without inspecting or copying semantic payloads.
`Flow` construction is the single binding boundary:
every direct `flow.wiring[node_id][parameter]` value and every
`FlowOutput.binding` is canonical `Binding`, while raw Python-authored values are
accepted and wrapped origin-free. Expansion rewrites only `Binding.source` and
retains `Binding.origin` and `Binding.contract`; pre-IR Compose dictionaries and
runtime-only overrides remain raw. `Binding.contract` is set only for a typed binding
(`name: Type = value`); its contract origins are traversed and rebased with the
binding's own origin. Declared binding types are part of the manifest: wiring types
in a separate `binding_types` map and output types under a `type` key, so untyped
flows keep their fingerprints. `FlowOutput.from_` is a temporary read-only source view during the
remaining migration.

`Node.origin` is the same source-neutral provenance shape for an authored node
context. Compose stamps the authored node-header occurrence for root, `defs:`,
and inline-body nodes; direct Python construction defaults to origin-free, and
loader/desugar-synthesized nodes remain so. Clones preserve an existing origin.
`Node.origin_context` separately carries generated outer-to-inner spawner
ancestry for nested failure traces; it is empty on direct/static nodes and is
neither enumerated nor rewritten as local authored provenance. Both fields are
diagnostic-only and excluded from manifests, fingerprints, and expansion
digests. A semantic `Node` carries no YAML `range`, `field_ranges`, or aligned
assertion-range lists. Compose retains concrete locations on cursors,
descriptors, and `BuildResult` only until its validators finish, then discards
them before constructing `CompiledFlow`.

CODE also owns two source-neutral diagnostic slots for its authored executable
field. `code_origin` is the whole-`code:` fallback, while
`code_line_origins` is an immutable tuple aligned with the constructed lines of
an exact literal block. Quoted and folded scalars carry only the whole fallback;
source-free Python construction defaults to `None` and `()`. Compose transfers
these origins from `CodeDescriptor` to `CodeNode`, and clones preserve them.
Inline CODE compiles the normal unpadded `def main(inputs):` wrapper once and
retains its recursive module/function/lambda/comprehension code-object tree.
`CodeNode.exception_origin()` walks an exception traceback outer-to-inner and
selects the deepest frame whose code object is owned by identity (`is`), never by
filename, equality, or hash membership. Wrapper line `n` selects authored slot
`n - 2`; an unavailable slot, wrapper-only frame, or post-body serialization
failure falls back to `code_origin`, then `Node.origin`. Reference mode uses the
whole `code_origin` as its invocation site, then the node fallback. Compose applies
the same line selector to load-time
`SyntaxError.lineno` and reports `SyntaxError.msg`.

CASE carries one optional origin on each ordered `Case`, identifying that exact
`cases[N].when` occurrence even when predicate text is duplicated. Compose
registers those indexed sites before lowering and preserves the origin on the
rewritten predicate; searched CASE also assigns each generated input `Binding`
the origin of the first predicate that introduced its read, while `on:`
owns the generated `__on` binding origin. Searched lowering allocates one `__rN`
per distinct read: a reference `${path}`, a node-output reference inside a
`:-`/`:?`/`|` as `${path :- null}` (the same value on an optional edge), and, per
co-skipping escape, an unread join input `${p1.output | p2.output}` over its
producers (a required group). The binding co-skip rule then gives each edge, so the
gate skips as a binding of each escape would; the `when:` itself stays lazy.
`CaseNode.exception_origin()` locates its own active `forward` frame by
code-object and node identity, then selects the active `Case` by identity. Failures outside predicate evaluation fall back
to `Node.origin`. These origins participate in ordinary traversal and rebasing,
but remain diagnostic-only and manifest-neutral.

The AGENT node adds no provenance carrier for structured output: it reuses the
existing
`ValueContract.origins` on `Node.output_contract`. After structured-generation
retries are exhausted, `AgentNode.exception_origin()` accepts a Pydantic error
or its directly wrapped native-parser cause only when tracebacks prove both the
exact native/fallback generator and the exact generated schema validation. It
does not traverse deeper cause chains. It normalizes the first error location by
leaving record fields unchanged and removing only the generated scalar `value`
or list `items` wrapper, validates that path against the expected `Type`, and
selects `ContractOrigins.at()` (exact, nearest ancestor, then contract root).
Unrelated tool/provider validation, non-validation errors, and source-free
contracts keep the node fallback. Retry behavior and the original exception are
unchanged.

`Node.pre_asserts` and `Node.post_asserts` are ordered `Assertion` lists. Parsed
YAML remains raw text until the inline-call pass crosses each flow assertion into
one carrier; node-local validation creates one fresh carrier per authored
occurrence. Compose indexes those occurrences by source order and derives the
carrier's opaque `OriginRef` from the retained item cursor, so equal assertion
text at two positions remains two distinct sites. Flow carrierization preserves
an explicit programmatic origin
instead of replacing it. Inline-call desugaring and flow/CALL lowering replace only
`.expression`; child cloning and LOOP respawn retain every carrier field. Thus
`.display` and `.origin` survive every rewrite. Synthesized START/END nodes reuse
the lowered flow carriers.
`Node.run` evaluates the expression, reports the display text, and diagnoses a
failure through that exact `Assertion.origin`. Manifests include expression,
display, order, and duplicate count while excluding origin.
An expression that doesn't parse, or calls an unknown function or a builtin with
arguments it doesn't take, is malformed setup, not a failed assertion:
Compose rejects it in YAML with a located `LoadError`; for a carrier set in Python,
`run_flow` and `resume_flow` (of an unfinished run) raise `ExpressionError` (a
`ValueError`) before writing any record, and `Node.run` raises it from setup
validation before the body runs. One that parses but raises while evaluating still
does not hold. A CASE `when:` and a LOOP `while:`/`until:` set in Python follow the
same rule; the error names the case handle or the predicate keyword.

`Node.params` has one compiled shape too: an ordered list of exact `Parameter`
carriers (or `None` for a directly constructed no-input node). The flow still owns each
parameter's `Binding`; the node owns the formal name, optional `ValueContract`,
required/default presence, and string-coercion policy. Binding validates stored
defaults without replacing their authored payload and never converts a string:
`run_flow` applies the coercion policy (`coerce_inputs`) to the top-level flow's
supplied inputs before START binds them, so a value crossing a CALL/MAP/LOOP edge
into a child START is validated unconverted. A bound null shadows a default but still passes
through contract validation, so it is accepted only when the parameter is untyped
or its contract accepts null (a nullable or `None`-kind type). A present typed
`Binding` is checked after its source resolves and before the parameter contract:
`build_value_as` validates the value against `Binding.contract` and the built value
is discarded, so the payload reaches the node unconverted. A miss is `binding.type`.

NaN, infinity, and complex numbers are not yian values;
`typesys.values.reject_nonfinite(value, where)` finds one at any depth and raises
`TypeCheckError` with its path (`... complex number <repr> is not a yian value` for a
complex). The first boundary that sees one rejects it. The binder checks each resolved value and filled default, and fails `binding.evaluate` at the binding or default origin
(`input 'y' at a[1]: non-finite number inf is not a yian value`); temporaries inside
an expression are not checked. MAP checks its `over` list the same way, and an AGENT
variable default fails `node.execute` at the default. `run_flow` raises `ValueError` for
a run input before the log exists. `LLMConfig` rejects a non-finite value, so authored
config is a load error and host config fails before the run starts. Every run-log codec
(`canonical_typed`, `canonical_seed`, `encode_pool`) rejects one with a path; there is
no opt-in. So a node output fails `contract.output` at the field path (`output at
stats.a: ...`), selecting the declared field's origin, and a growth seed fails as a
growth rejection (`contract.growth`). Nothing is recorded as null in its place. A
top-level complex returned by a CODE node fails `contract.output` like NaN. An int
too large for a float in a `float` slot fails the type check at its path.

Ints of any size are values (an int literal in the YAML keeps Python's digit limit,
a located load error past it). Run-log JSON goes through `yian.json_codec`, which
writes and reads ints of any size without changing Python's process-wide
`sys.set_int_max_str_digits`; yian never calls it. Value bytes decode through
`runtime.log.decode_typed_json`: pydantic's strict JSON path first, and the codec for
a document with an int past the digit limit. Converting a long int costs time
quadratic in its digits (about 0.2 s each way at 100,000 digits). Type-check messages
show such an int as `<int too large to display>`. Rendering one as text (template
interpolation, prompt text) still fails like any other expression error, as does an
operator Python can't complete (`binding.evaluate` in a binding, `node.execute` for a
prompt).

An AGENT checks what the model sends back. `run_agent` passes `invoke_model` a
`check_reply` that rejects a tool call whose arguments hold a NaN or infinity, before any call of
that reply runs, so the AGENT fails `node.execute` (`tool call 'lookup' arguments at
x: ...`). Structured generation marks every `float` slot `allow_inf_nan=False` (the
JSON schema is unchanged), so `{"score": NaN}` is a `finite_number` schema error that
self-corrects under `retries:` and, once spent, selects the declared field's origin;
a freeform slot is checked after unwrapping (`structured output at k: ...`; on the
native path with a transcript the raw check names it first, `model reply at
tool_calls[0].args.value.k: ...`). With a
transcript, `invoke_model` also rejects a raw reply holding one anywhere, provider
metadata included, since only finite JSON can be stored: the model effect is observed
`failed` (or `interrupted` when a stop is pending) and the reply is not stored. On the
native path the carrier's own schema error is raised instead when it has one, so the
retry and origin match a run without memory. Without a transcript, provider metadata
is not checked.

A yian value nests at most `typesys.values.MAX_VALUE_NESTING` (100) levels, counting
itself; each dict, list, tuple, pydantic model (such as `FileRef`), or dataclass
instance is a level, and a `TypedValue` wrapper adds none. `reject_deep_nesting(value,
where, *, enclosing_levels=0)` walks with an explicit stack and raises `TypeCheckError`
with the path (`<where> at <path>: value nests deeper than 100 levels, the limit for a
yian value (each record and list is a level)`). The same boundaries apply it, depth
before the non-finite walk: the binder fails a resolved value, filled default, or MAP
`over` list with `binding.evaluate` at its origin, so `${[x]}` around a 100-level `x`
fails and MAP items may nest 99; an AGENT variable default fails `node.execute`;
`run_flow` raises `ValueError` for a run input before the log exists;
`RunResult.backward` raises `ValueError` for deeper feedback before an attempt starts;
and an AGENT checks the model's reply as for NaN: `run_agent`'s `check_reply` fails
deeper tool-call arguments with `node.execute` before any call runs, and structured
generation checks the unwrapped value (`structured output at ...`), which
self-corrects under `retries:`. On the native path with a transcript the structured
check also runs in `check_reply`, on the carrier's parsed value, so a deeper value is
observed `failed` and never stored (its raw reply could exceed memory's 128-level
bound, and the raw non-finite walk recurses); a carrier with a parsing error whose raw
reply is deeper raises that parsing error there instead, as for NaN. Live success
admission (`prepare_node_succeeded`) checks a node output whole and fails
`contract.output` at the field path with the declared field's origin. A record built
around a value is one level deeper, so a record flow output wrapping a 100-level value
fails at `__end__`, and a MAP whose children return 100-level values fails at its
fan-in END (`<map>/__end__`). A node with the `commits_input_record` trait (START)
passes `enclosing_levels=-1`, since its record wraps input values already checked.
Replay validation does not re-apply the limit, so logs recorded before it still
replay, and the run-log codecs never apply it; they see only values these entry
checks bounded.

Numeric `Type` values may retain an inclusive `bounds` pair for INTEGER or
NUMBER. Shape validation rejects malformed, reversed, or incompatible
refinements; integer endpoints remain arbitrary precision, and NUMBER endpoints
require finite, lossless float representation. Value checks preserve the failing
structural path. Cross-flow compatibility requires interval containment, including
integer-to-number widening. Inline type copying, nested generated schemas, and
canonical labels preserve bounds. Manifests include bounds only when present,
preserving the identity of unbounded declarations. Enum (`tags`) compatibility
follows the same rule: the source tags must be a subset of the sink's, and an
untagged source cannot fill an enum sink.

A union is `Type(kind=OBJECT, variants=(member, ...))`: two or more non-nullable record
members in canonical order (by tag, else by canonical label), deduplicated by equality
because `Type` holds dicts and is unhashable. `Optional[Union[...]]` sets `nullable` on
the union, never on a member. Tag fields are derived (`union_tag_fields`), not stored. A
tagged union's value check picks the member by the first tag field; an untagged one, whose
members `types_overlap` proved disjoint at resolution, checks the one member the value
satisfies. Compatibility, labels, inline type copying, structured-output schemas, and
manifests (`"union": [...]`, only when present) all walk the arm. `union_field` and
`restrict_union` are the read and narrowing helpers.

`Node.types` is the node's `isinstance` type table, `dict[str, Type]`, empty by default.
Compose fills it after the START/END boundary is synthesized, from the names the node's
wiring sources, asserts, CASE `when:`s, and LOOP predicate use, resolved against the
owning flow's `typedefs:`. Every evaluation entry point takes `types=None`; `eval_node`
(binding), node asserts, CASE routing, and the LOOP predicate pass `self.types`, and
`LoopNode.respawn` copies it. Manifests emit `types` only when non-empty, so nodes
without `isinstance` keep their fingerprints.

`AgentNode.variables` is an ordered list of the same `Parameter` carriers, each
with an exact `ValueContract` and an explicit typed default. Variable names are
local and cannot overlap inputs. Compose compiles their declaration-key, type,
and default origins through its shared reader; AGENT's origin hooks retain them
through graph rebasing. Variable declarations affect authored identity, while
their diagnostic origins do not. Empty variable inventories leave existing
fingerprints unchanged.

AGENT resolves a detached copy of its defaults once per invocation and renders
one prompt from inputs plus those local values. Default strings remain literal
data. Rendering requires no execution scope and does not change model config or
graph bindings. `Node.declared_variables()` exposes an empty tuple by default;
AGENT returns its canonical declarations for common feedback target validation.
The stable owner is `(execution_scope.static_path, variable.name)`, so repeated
MAP/LOOP occurrences retain one authored owner and distinct callsites retain
distinct owners. This declaration inventory does not itself activate backward.

`AgentNode.system_prompt` is an optional template. When set, AGENT renders it
from the same inputs-plus-variables record as the prompt, and the result replaces
`DEFAULT_SYSTEM`; when `None`, `run_agent` receives `DEFAULT_SYSTEM` unchanged. The
skill listing and then memory sections append after this effective system prompt,
and every model request (one model call, tool loop, structured generation) sends it. It is recorded
as `EvaluationEvidence.base_system_prompt`, so backward and recovery read what the
model saw. The manifest includes the template only when set, so flows without one
keep their fingerprints.

`TypeCheckError.path` contains only contract-relative record names and actual list
indices. A supplied-value rejection resolves that path through `ContractOrigins`
(exact, nearest ancestor, then default) before falling back to the parameter
origin. An invalid authored default remains `binding.evaluate` and uses its
default origin rather than the value-contract site. A typed binding whose value
misses its declared type fails `binding.type`, located at the declared type's
origin for the failing path, then the binding's origin, then the parameter's.
Parameter/default/contract origins do not affect manifest identity. Compose
`input:` lowers directly into this same exact carrier shape; `ComposedFlow.input`
and START reuse the same carriers without a parallel front-end declaration type.
`compose/helpers/types.py` is the one source-aware recursive type compiler. For
each authored root or `defs:` flow input, its key owns `Parameter.origin`, its type
value owns the `ContractOrigins` root and authored nested record fields,
and a packed `TYPE = default` owns `default_origin`. Scalar alias and list
type expressions have no interior YAML occurrence, so nested failures fall back to
the contract root. Python/source-free declarations and inferred inline-child
parameters keep empty origins; external aliases use ordinary lexical graph rebasing.

That declaration ownership is distinct from ordinary node-input wiring. Final
assembly puts the selecting-key origin on the sink `Parameter` and the bound-value
origin on its flow-owned `Binding`. Source-backed reserved wiring carries only the
value origin on its canonical `Binding`: MAP `over:` and WAIT `until:` have no
corresponding `Parameter`. A desugar may change `Binding.source` without moving that
value origin. When no source occurrence is retained, Compose mints no origin and
preserves any explicit programmatic origin; source-backed projections inherit their
backing site.
Source-backed bare and
mapped flow-output values on root flows, multi-node definitions, and inline children stamp
the final `FlowOutput.binding` after desugaring rewrites; the synthesized END boundary
reuses that exact binding. Output mapping keys carry no origin. Compact-root and
single-node-definition reexports are generated without an authored binding occurrence and
remain origin-free. That binding provenance is separate from node-output contract
provenance: a physical `output:` type declaration owns the root and authored nested sites
on `Node.output_contract`, including when compact/flat lowering generates the reexport.

`Node.output_contract` is the canonical compiled output declaration: an exact
`ValueContract`, a `PartialRecordContract`, or `None` when the output is unenforced.
The partial arm is a closed record: every field name is enforced, proven fields retain
their recursive contracts, and opaque fields retain presence without pretending their
type is `None`. Runtime success and replay validate the effective commit target through
this same sum. Exact contracts also drive node-owned structured generation. For
an AGENT node,
only an untagged `str` opts out to free text; a tagged string (`Literal[...]`) uses the
generated scalar `value` wrapper and is unwrapped back to its bare tag. A live type
rejection resolves its contract-relative structural path through the matching exact or
field-local origins. The path is consumed before `Failure` construction and is not part
of the run-record wire. Manifests include the semantic exact/partial contract but omit
origins, so changing or rebasing provenance does not change flow or expansion identity.
Compose and runtime growth reference validation share the source-neutral closed-field walk
in `nodes/contracts.py`, so a partial record cannot expose a provably absent field. That
walk, `output_contract_field_error`, also rejects any field on a scalar or list type; an
opaque `object` (no declared fields) and `file` stay unchecked. On a union it reads a
field every member has with the shared type (the merged `Literal` when every member's
field is a string `Literal`, untyped when the types otherwise differ) and rejects a field
only some members have, naming the tag to `case` on or suggesting `isinstance`. Runtime
growth passes `partial_union_fields=True`: grown fragments come from flows the loader
already checked under narrowing, and that narrowing is not recorded, so member-only reads
are allowed there.

Compose's narrowing pass (`compose/validate/narrowing.py`, `compute_narrowing`) gives each
node a view of the producer and flow-input contracts with narrowed union bases rebuilt. A
fact for a base path (`<node>.output[.path]` or `input.<name>[.path]`) is `Members(S)` (a
subset of the base's own union members, with nullability) or `Exact(T)`. An `on:` CASE
over a union tag field gives each `when:` handle its tagged members and `else:` the
unlisted ones. A searched CASE's top-level `isinstance(<ref>, T)` conjunct gives its
handle a fact; its `else:` carries none. A later conjunct of the same `and` chain is
checked against the narrowed base (`Narrowing.for_param`). Facts flow in Kahn's order
over data, CASE, and `depends_on` edges: control edges join by union, each required data
group joins its producers by union, and all contributions meet by intersection. Optional
groups and `runs_after` contribute nothing; nodes on a cycle get no facts and are left for
`reject_cycles`. The call-type and CASE-exhaustiveness checks read
`Narrowing.for_node(id)`, and the reference and typed-input checks read `for_param`, which
falls back to it; flow outputs are checked unnarrowed. A CASE that produces a value (it
declares `output:`, has a value branch, or is read as `${case.output}`) must be
exhaustive: an `else:`, every tag of a non-nullable enum `on:`, or `true` and `false` for
a non-nullable `bool` `on:`, with a flow-input `on:` resolved for it too; `when:` values
compare as typed values, not text. An `on:` CASE's `when:` values are typed literals, and
`compose/validate/case_exhaustive.py` (`check_case_when_values`) runs one membership check
right after narrowing: where the `on:` type is known, a `when:` it can't hold (a non-tag,
a wrong scalar type) is a located load error, even next to an `else:`. Before that, it
rejects a `when:` equal to an earlier one (typed equality: `1` and `1.0` are equal; `1`,
`true` and `"1"` are not) at the later one, for any `on:`. A routing-only CASE over a
`Literal` node output or flow input (a LOOP body's carried field, a child flow's input)
needs a `when:` for every tag unless it has an `else:`. Compose lifts each
CASE value branch to a synthetic `__value_N` CODE node and gives a typed CASE one
`__case_<id>` CODE result node that checks the taken branch; this is load-time desugar,
and the runtime is unchanged. A node a desugar pass lifts out of an authored node (an
inline `call(...)` in its binding, an inner call of a nested branch call, an adaptive
composer) records that node as its `host`, and `compose/edges/hosts.py` copies the host's
CASE route edges and `depends_on`/`runs_after` ordering edges onto it, so it runs only on
the host's branches and in the host's order. It also gets data gates: data edges from the
producers of the host's other inputs, grouped like the host's input (group
`<node>.<group>`), so the runtime join skips it when those inputs skip the host and
otherwise only makes it wait. Nodes lifted out of the same host don't gate each other, but
a nested call is gated on its outer call's other arguments. Inline child bodies and LOOP
seeds are typed while nodes build, before the pass, so they see unnarrowed types. Runtime
routing is unchanged.

Physical Compose declarations derive exact `ValueContract`s. Output binding derivation,
builders, child signatures, producer maps, and validators retain the complete
`OutputContract` sum; algorithms project an exact type only where one is required. The
shared `compose/helpers/output_contracts.py` preserves exact or partial whole-value
references, projects independently proven fields, assigns source-free contracts to
results proven by `expr.infer_expr_type`, and keeps unsupported computations opaque.
Record assembly produces an exact record when every field is exact, otherwise a
closed `PartialRecordContract` with the proven siblings. `ChildSignature.output` follows
the child's END: a record (`output: {...}`, even with one field or none) gets the record
contract keyed by every name, a value (`output: ${x}`) re-exports that binding's contract,
and CALL preserves it. END's `output_contract` comes from declared output types: a
value END takes its one output's `Binding.contract`, and a record END takes an exact
record when every output is typed, else a partial record with the untyped fields
opaque; with no typed output it stays `None`. END commits and replays through it, so
parents see a child's declared output type and recorded outputs are re-checked on
resume. A LOOP's contract is the merged carried
record: each field the body returns comes from the body's contract, and each omitted
field comes from its seed binding (a literal seed stays opaque). MAP derives `list[U]`
only from an exact child contract because the output-contract sum has no partial-list
arm. Strict inline input inference remains separate: it accepts literals, direct
exact references, and proven fields projected from partial records, while rejecting
whole or opaque partial reads. LOOP uses the same canonical carrier for proven seed
checks at load and for every zero- or later-iteration commit, replay, origin lookup,
and expansion identity; no transient field-evidence authority exists.
There is no parallel output-Type API: Python-authored nodes pass `output_contract=` explicitly.
For a physical node `output:` declaration, Compose injects a source-aware reader into the
descriptor's existing per-kind build position. The shared recursive compiler returns the
canonical contract with its declaration root and each authored nested record field; this
preserves kind-owned validation order without a parallel source map or second type walk.
Full and compact roots, multi-node and flat definitions, and inline children retain or
project the needed cursor. External aliases use ordinary lexical graph rebasing. Scalar
aliases and scalar `list[...]` expressions have no interior declaration, so a nested
failure falls back to the contract root. Generated declarations and Python-authored nodes
remain source-free.

### Compiled origin traversal and rebasing

Nodes own their provenance layout. The base `Node` enumerates and rewrites its
node origin, output contract, parameters, and assertions. CODE extends those
hooks with its whole-source fallback and ordered exact-line origins; CASE
extends them with its ordered predicate origins;
CALL/MAP/LOOP also own their cached `child_inputs` origins and expose the baked
child through `origin_children()`. These hooks remain source-neutral and never
import `compile`.

`compile.origins` joins those hooks with flow-owned wiring and declared outputs.
`iter_flow_origins()` walks wiring, outputs, nodes, and static children in stable
order, guarding flow, node, and frozen-carrier identity so shared or cyclic
graphs terminate. `rewrite_flow_origins()` rewrites an isolated working copy by
identity, then copies that finalized graph for return so payload dict/set hashes
are rebuilt after frozen carrier values change. Aliases, payload back-references,
and deliberate cycles survive while the authored graph remains unchanged. Binding sources,
parameter defaults, node environments, and render metadata are opaque even when
they contain an `OriginRef`. Origins remain excluded from manifests,
fingerprints, and expansion digests.

`diagnostics.ResolverTable` composes opaque namespaces without owning any source
format. Each public origin key routes to exactly one delegate resolver and one origin
in that delegate's namespace. Duplicate public keys are rejected; unknown
origins return `None`; delegate exceptions propagate; and malformed delegate
results are rejected. The table never parses keys or introduces resolver precedence.

`rebase_flow_origins(flow, mapper, resolver=...)` pairs a pure graph rewrite with
those exact routes. It preserves first structural-visit order while deduplicating
reachable old origins by their opaque key, calls `mapper` once per distinct old key,
and rejects collisions between distinct mapped keys before rewriting through
the precomputed map. The result is an independent graph of the same concrete type
plus a table containing only its reachable routes. An originful graph requires a
resolver; a source-free graph requires `None` and returns an empty table. The
operation defines no lexical key policy and knows nothing about Compose or YAML.

## One-pass source ownership

`compose/source/` is the only production Compose package that imports PyYAML.
One loader class (`compose/source/scalars.py`) reads plain scalars with YAML 1.2
core-schema rules plus unquoted timestamps and `<<` merge keys, for documents,
for isolated literals (`parse_yaml_literal`), and for one bare plain scalar
(`read_plain_scalar`, which reads an inline `call(...)` argument), without
touching PyYAML's global resolver or constructor tables. A plain scalar its tag
can't build, such as `2026-13-45`, is a located YAML error that says to quote
it. Root text and each uncached external file are constructed once as
`SourceDocument`s, whose public cursor interface is immutable and defensive.
They retain the original text and label while keeping PyYAML's
composed/constructed graph private. YAML aliases and merge mappings are
interpreted without mutating that graph. Persistent pair/arc ropes keep merge
topology compact, and completed mapping summaries bound repeated merge-DAG
traversal. Public value snapshots use a stack-safe iterative copier that
preserves shared and recursive aliases. Within one successful file-import graph,
the loader's resolved-path cache also ensures that each external physical file
is read, parsed, and compiled once; aliases and diamond imports reuse that
canonical `ComposedFlow` as the source for their lexically rebased views.

Source-aware parsing and inline extraction navigate stable `SourceCursor` values.
A physical cursor combines a definition range with an occurrence path (and a
mapping-key range when applicable); a virtual cursor is a defensive, source-free
value snapshot; a projected cursor replaces or renames values while explicitly
retaining the physical sources of selected children. While the cursor is still
available, a descriptor may attach opaque origins for kind-owned semantic sites.
Cursors and descriptors retain concrete ranges through parsing and desugaring;
`NodeDescriptor._finish()` copies the descriptor's header and field locations
for the built node to Compose-only `BuildResult.node_range` / `field_ranges`, and
its `field_labels` (author-facing names a desugar sets for a synth node's fields) to
`BuildResult.field_labels`. Post-build validators
consume that metadata, and assembly drops it before `CompiledFlow`; the semantic
node receives only the opaque origins. No phase reparses text or touches parser
objects.
For an unchanged source-backed string, `SourceCursor.scalar_source` returns immutable
`ScalarSource` metadata: the whole scalar range and exact zero-based constructed-line
ranges when the scalar is a literal `|` block. The document derives those ranges from
the real block token during its existing parse. Other scalar styles expose the whole
range but no exact line tuple; virtual and value-changing cursors expose no scalar
provenance. `SourceCursor.scalar_style` likewise reports an unchanged scalar's
authored YAML style (`"plain"`, `"single"`, `"double"`, `"literal"`, or `"folded"`),
also through projections, and `None` otherwise.

`SourceCatalog` is Compose's immutable document-local `OriginResolver`. Its private
builder maps typed semantic paths to opaque origins while retaining the physical
document occurrence and a key/value anchor or semantic constructed-line index.
Resolution replays that occurrence and reselects the current range; line coordinates
are never stored in the site identity. Unavailable exact line registration leaves the
site free for a whole-scalar fallback. Resolution neither reparses YAML nor creates a
parallel range map. Virtual or unanchored cursors produce no origin. One builder is shared by
the root flow, its `defs:`, and its inline bodies. Private assembly leaves the public
resolver unset; the loader freezes the builder once when the physical-file root is
complete.
After the common node-header origin is stamped, assembly invokes
`NodeDescriptor.with_source_origins(...)` on every normalized descriptor. The
default is an immutable no-op. `CodeDescriptor` uses the hook to register the
whole `code:` scalar and, when `ScalarSource` exposes an exact literal tuple,
each constructed line under the node's lexical site path. It stores only the
returned `OriginRef` values, never a cursor or coordinate map. The same path runs
for roots, definitions, and inline children; virtual source produces no origins.
`CaseDescriptor` uses the same hook for each indexed `cases[N].when` and optional
`on:` value, then transfers those origins to the ordered `Case` values and
lowering-generated `Binding` values.
`compose/helpers/types.py` owns the sole source-aware recursive type compiler.
Final assembly uses it for retained root/definition flow inputs: declaration keys
stamp `Parameter.origin`, type values and authored nested record fields
populate `ContractOrigins`, and packed defaults stamp `default_origin`. START reuses
those carriers. Scalar alias/list type expressions fall back to the contract root;
Python/source-free declarations and inferred inline parameters remain source-free.
At each output-authoring leaf's established build position, an injected Compose reader
uses the retained normalized node cursor to compile a physical `output:` declaration into
the same canonical `ValueContract` shape. Compact roots and flat definitions project their
top-level declaration onto the node; multi-node definitions and inline children retain
their own node cursors. External child views rebase the resulting root/nested origins.
Virtual/generated declarations and direct Python descriptor builds remain source-free.
Final assembly separately registers authored node headers, ordinary node-input keys
and bound values, source-backed reserved MAP/WAIT wiring values, CODE source fields,
CASE predicate/`on:` sites, assertions, and source-backed flow-output values directly
on their canonical semantic carriers; it does not build a parallel location map. An
ordinary node input's key
belongs to its sink `Parameter`, while its value belongs to `Binding`. An assertion's semantic
path includes its authored list index, and that item's cursor supplies
`Assertion.origin`. Runtime assertion failure diagnosis uses that carrier origin;
no positional assertion range crosses into the semantic node. Full root flows and
multi-node definitions retain a flow assertion cursor; their lowered carriers
are reused by START/END. Compact roots and flat single-node definitions instead
project their `asserts:` into the synthesized node, so those sites are
node-local. Flow outputs are stamped after rewrites on `FlowOutput.binding`, and
END reuses that binding; mapping keys are not origin sites. For sites without
retained occurrences, Compose mints no origin and preserves any explicit
programmatic origin. Generated compact-root and single-node-definition reexports
are such source-free binding sites even though their physical type declaration originates
the node's output contract. Runtime output validation selects the effective target
contract's exact/nearest/default origin from `TypeCheckError.path`, then falls back to the
executing node.
At each `uses:` alias, the loader validates the canonical child's complete resolver,
copies and rebases its compiled origin graph under a collision-free lexical key, and
caches that alias view. The physical resolved-path cache remains canonical and is never
mutated. Finalization filters the document catalog and alias tables to the exact reachable
origin set and installs one immutable `ResolverTable` as
`ComposedFlow.origin_resolver`; a source-free graph carries `None`. Repeated calls through
one alias reuse its view, while distinct aliases and diamond branches remain disjoint.

`ComposedFlow` stores no source-text or range snapshot. Its optional
`origin_resolver` is the sole presentation authority for runtime failures, while
load diagnostics are resolved before they leave Compose.

## Inline composite lowering

An authored CALL, MAP, or LOOP may contain a non-empty `nodes:` plus a required
`output:` instead of naming a child with `call:`. This is entirely a `compose/`
front-end feature. `desugar.inline_bodies` retains the located child body and
masks that occurrence as an ordinary private `call:` before the normal
parse/build/validate pipeline runs. YAML anchors, aliases, and merged mappings
are not mutated: extraction works on occurrence-local cursor projections while
keeping the original child cursors and source ranges read-only.

Each source file owns one child-flow scope. Alongside its authored `defs:` and
`uses:` namespace, that scope has a resolver-private table for retained inline
children. A child's stable identity is `__inline_flow_` plus a digest of its
complete lexical owner path (root or def, enclosing node, and nested inline
occurrences), not a process counter or object identity. Authored defs, aliases,
and call targets cannot use that reserved prefix, and author-facing diagnostics
show the lexical child label rather than the opaque digest.

Private child builders are registered during assembly and invoked lazily. The
file scope builds each child flow at most once on successful resolution, shares
the same cache across private children/defs/uses, and guards private/def
construction in progress. The default file resolver's load context guards
cross-file recursion; an injected resolver is responsible for any equivalent
protection.
Generated child parameters come from the enclosing composite's `input:`
bindings. Their types are proved structurally; parent producer descriptors are
themselves built on demand through one success cache, and a MAP's `over:` element
is proved and cached only if a lone whole-value `${item...}` binding requests it.
Nested literal container strings are deliberately not scanned as references.

The result is an ordinary compiled child baked onto the existing CallNode,
MapNode, or LoopNode. No private resolver state reaches execution, no runtime
kind or graph-rewrite rule is added, and live execution/recovery continue through
the same kind-blind `Grow`/record/reducer path as named children.

## Node contract

A node is pure at the graph-state boundary: it receives a bound input record and
cannot mutate engine-owned state. It returns exactly one member of the closed
node-outcome sum:

| Node outcome | Meaning |
|--------|---------|
| `Output(value, commit_as=None)` | one value for the engine to type-check and bind under the resolved success target |
| `Route(handle)` | one named outgoing branch (every edge with that handle: the target and the nodes lifted out of it), with redirect-aware sibling skip-flooding; the handle must match an outgoing edge, except the no-match route `ELSE_HANDLE` (`"#else"`), which may have none |
| `Grow(seed, prune)` | one canonical deterministic expansion description |

Success and absence resolve their logical output boundaries through distinct
carriers. Success resolves an explicit `Output.commit_as`, then the baked
`Node.commit_as`, then the physical node id. A skip cannot observe result-carried
`Output.commit_as`, so the reducer instead asks the pure, deterministic
`Node.skipped_output_target()` hook, which may consult only manifest-backed node
state. The base hook returns baked `commit_as` or the node id; LOOP returns its
manifest-backed `origin_id` to mirror its dynamic success redirect. A redirected
target has exactly one unfinished physical owner and must be an inert logical
anchor: an expanded spawner or an unknown placeholder with no incoming work, never a
ready, in-flight, or already-committed node; all of its outgoing edges must still
be unknown. The reducer marks the physical node skipped, deduplicates logical
targets, and floods the target's outgoing edges.

Failure is a `raise`, not another result variant. `eval_node` structurally turns
expected binding, assertion, body, and result rejection into
`EvaluationFailure(Failure)`. Result preparation constructs the same canonical
value for expected output, route, and growth rejection. Schema 9 carries that
value unchanged through the durable record, reducer, observer events, and public
result; the durable outcome may also retain optional source-neutral evaluation
evidence. Host control, stream callbacks, unexpected binder faults, malformed
setup, and arbitrary host/compiler faults escape. A streaming kind may yield
transient `StreamChunk` values and then return one of the same three results.
A node declaring `needs_live_events` instead receives in-body `emit_stream` and
`emit_event` capabilities, each only when the host supplied the matching
callback. They publish transient chunks and in-loop observers without changing
the node result; a sink exception escapes as the host's own exception.

`eval_node(..., capture_bindings=True)` can also return a detached
`BindingCapture` independently of model/tool evidence. It retains the actual
input record after coercion, validation, and wrapper defaults, before assertions
or body mutation. An incomplete binding has `bound_inputs=None`; partial routes
remain available for inspection. The node-owned `bind_reserved(..., bind=None)`
hook can use the injected `bind(name, source, normalize)` callable to retain a
reserved input's route and normalized value. Materialized hook results without
that call have literal routes. MAP passes `bind_item(element, index)` its actual
enumeration index and projects each item's routes from the captured `over`
collection, including concatenated or repeated collections.

Expression capture uses the existing evaluator walk. A paired resolver returns
the value, presence, and optional source together; neither capture nor feedback
transfer invokes the source expression again. `expr.lineage.ValueRoute` retains
selected arms, projections, collection construction, concatenation, repetition,
and the evaluated contributors to scalar operations and interpolation. Ordinary
registered function calls are opaque: feedback stops there with local context.
`BindingCapture.evaluated_keys` separately records which inputs evaluated a
template expression, since even an evaluated expression can return a literal
route. Literal and escaped strings do not set that marker.

Feedback values are ordinary strings, dictionaries, exact-length positional
lists, or `None`. A string addresses a whole value and stays unchanged during
transfer; structural context accompanies it separately. Dictionaries select
known fields, and a positional list must match its recorded list's full length
at every nested level. `None` and empty selections carry no effective feedback;
an empty string is still a supplied message. `BackwardResult` keeps independent
messages in ordered tuples across its input, self, and variable channels.
Input messages may be structured; self and variable feedback contain text
messages only.
`validate_backward_result` checks the complete result against the detached
`BackwardContext`, including actual input shapes and declared target names,
before any input message is routed. Forward execution does not invoke backward.
Routing receives the accepted canonical input value for shape validation when a
supported forward object had an unavailable temporary expression snapshot. It
preserves the recorded route and stopping boundary. An opaque projection with
unavailable parent shape retains its selected feedback and path without inventing
parent fields or list lengths; literals still stop without opaque-stop evidence.
Structural backward uses the recorded invocation layout: START maps named inputs,
END follows its layout (a value maps its one input to path `[]`, a record maps each
field to `[name]`, and a MAP fan-in list maps each slot to its index), and a stopping
LOOP routes its carried record. The manifest encodes a value END like a record END
with the same names, so value, absent and multi-field outputs keep their fingerprints;
a zero- or one-field record adds `keyed: true`. `backward_input_names()` supplies each
node's eligible input inventory: END uses its stored output names or list slots, and
CASE excludes its generated control parameters. Assertion-only parameters do not
become feedback targets. Behavioral nodes inherit the empty stopping
implementation until they define their own transformation.

`compile.execution_layout` derives execution locations independently of memory
declarations. Each `ExecutionScope` separates a root-relative authored node path,
its containing flow path, a flow boundary role, and instantiated child positions.
The root flow path is empty, so a direct graph needs no flow ID or artificial
boundary nodes solely for layout. A node's `MemoryScope.static_path` is the same
root-relative locator as its execution `static_path`.

Resolve and apply execution layout after the existing run-configuration pass
makes private child copies at each callsite. Resolution is pure. Application
validates the full layout, current graph, exact target objects, aliases, and all
existing scopes before stamping any node; reapplication is idempotent. It rejects
shared or replaced targets instead of overwriting scopes at conflicting paths.
The derived `Node.execution_scope` field has an explicit manifest exclusion and
does not change authored flow fingerprints.

LOOP templates start with an explicit driver position at iteration zero. Child
instantiation adds CALL, MAP-item, or LOOP-body positions to the enclosing path;
a nested LOOP keeps its own initial driver position. Each continuing driver
replaces its previous driver position. Authored node and flow paths stay stable,
and memory occurrence paths and memory-key stacks remain separate.

Growth fragments carry source-neutral input-connection recipes and actual child
START/END boundary pairs. CALL and LOOP recipes select a named materialized input
route; MAP recipes select an explicit item index and then a name. Synthetic MAP
fan-out and collector nodes inherit the spawner's execution scope when present,
with `boundary=None`. Expansion validation and its digest include these derived
connections and execution topology; authored fingerprints exclude them.

The runtime core is **kind-blind**. It dispatches on this result sum and consults
node-owned execution-semantic traits/hooks (`is_spawner`, `commits_input_record` (START's
input record may go one level over the value nesting limit), `binds_per_item`, `item_source_key`,
`bind_reserved`, `validate_execution_setup`, `grow_depth_delta`, `is_loop`,
`needs_stop_event`, `needs_tools`, `needs_memory_context`, `needs_skills`,
`needs_evaluation_trace`, `needs_live_events`, `needs_steering`, `memory_scope`, and
`skipped_output_target`) plus the diagnostic-only projections
`exception_origin` and `failure_trace`; it never branches behavior on `NodeKind`.
A kind label may appear in diagnostic text. The diagnostic projections do not
affect program identity. The closed kind-aware
growth match lives in `compile.expand.build_grow`, outside runtime.
`compile.model.GrowthRejected` crosses that boundary only when a legal
CALL/MAP/LOOP kind supplies a malformed seed. Unsupported kinds and builder or
compiler helper faults remain raw; runtime marks only explicit seed, fragment,
budget, and prune-policy rejection as expected growth data. The kind census
test keeps runtime's `NodeKind`/expansion dispatch count at zero.

## One append-only authority

Every drive uses one `RunLog` and matching reducer-owned `RunState` over the
run's accepted record authority.
`MemoryRunLog` is the process-local implementation; `JsonlRunLog` is the durable
implementation, with a nonblocking exclusive advisory lock, canonical JSONL, and
an fsync per append. Both accept the same closed record vocabulary:

```
RunStartedRecord | RunResumedRecord | NodeStartedRecord |
NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord |
MemoryCommitRequestedRecord |
MemoryCommitSucceededRecord | MemoryCommitFailedRecord |
BackwardStartedRecord | BackwardReadyRecord | BackwardFinishedRecord
```

Records use schema 9, and readers accept only schema 9. A schema-7 or schema-8
log is rejected with `run log schema N predates schema 9 and is not supported`;
there is no migration. Preparation and replay retain each envelope's canonical
bytes, including trace comparison, prepared-record tamper checks, and append
verification.

The recorded `learnable` flag controls forward retention:
`learnable=True` requires every accepted node outcome to carry a `ForwardCapture`;
`learnable=False` omits it. The capture contains canonical bound inputs or an
incomplete-binding marker, input routes, indexed item captures, execution scope,
detached declared input/variable inventories, and optional node-owned data. The
pure `capture_backward_context` hook supplies static node layout at the existing
input capture boundary: after defaults and before assertions for the base run
template, or before invoking a custom run wrapper. Its empty default is
independent of the common declaration inventories. Expected early binding
failures retain the known inventories with no usable input record. Capture
faults remain host errors.
The transient `BackwardCapture` carries names, scope, and node data through
`EvaluatedOutcome`; only the accepted `ForwardCapture` is durable. A reserved
`input_output_paths` map links eligible names to output field/index paths, keeping
structural routing in node code and its context separate from feedback values.

An opt-in invocation capture capability also accepts one completed dynamic
snapshot. AGENT publishes from its actual variable-resolution/render operation,
retaining `prompt_template`, `variables`, `variable_domains`, `rendered_prompt`,
and `effective_llm_config`. Locals established before a later failure remain
available; unresolved variables or an unrendered prompt are explicitly `None`,
distinct from an empty record/string. The evaluator immediately detaches the
snapshot, rejects repeated or post-close writes, and keeps capture faults as host
faults. Supported values use canonical seed conversion, which rejects NaN,
infinity, and complex numbers like every codec; static capture validation is unchanged. Model configuration comes
only from the accepted RunStarted map entry selected by the authored execution
path, never a dynamic ID or mutable ambient/node config. No default or prompt is
evaluated a second time to capture it.

Model/tool evaluation evidence is separate and may
have no memory scope; recalled sections still require
a matching scope and follow its memory-key stack order. `NodeFailedRecord` owns one canonical
nested `Failure`. Terminal
success remains derived from the END value. The first
route whose skip closure reaches END persists its causal `Failure` on that
`NodeRoutedRecord`; replay revalidates the route before restoring it, and no
additional terminal record is written.

Runtime handles every transition in one order:

```
prepare and validate -> append -> apply reducer -> publish transient observers
```

Preparation cannot mutate live state. If append fails, its state transition is
not applied and its outcome observer is not published. `FlowEngine` is the sole
forward ready-frontier consumer; the explicit backward driver later uses the
same prepare/append/reduce authority. Pooled workers only evaluate durably
started nodes and return raw results to the dispatcher. Stream chunks remain
callback-only and never enter the authoritative log.

`runtime.trace.RetainedTrace` derives execution history from those accepted
records. An occurrence is identified by the run ID and its accepted
`NodeStartedRecord` sequence; outcome sequence separately defines accepted order.
Sequence zero denotes the root input source. The current writer index maps each
logical output target to its actual producing execution, including redirected
outputs. A binding lookup reads the value, presence, and producer from one
immutable entry at the existing read point. A pooled consumer may therefore start
before the producer it eventually reads has started.

Trace changes follow the same prepare/append/apply order as scheduler state.
Rejected proposals retain their accepted canonical failure; discarded worker
results do not become occurrences. Replay rebuilds the indexes through the same
reducer and retains reconstructed node backward callables separately from the
serialized records. Pruning removes live names while preserving accepted history
and its callable references. It does not rerun expressions or model calls to
recover provenance.

The trace also retains the actual root output writer or causal failure source,
independently of the last completed node and live graph pruning. Its
`backward_context` method reconstructs detached invocation context without
calling node hooks. `runtime.backward.compute_backward` walks accepted outcomes
in reverse order, gathering every incoming message before invoking an occurrence
once. It validates the entire returned result before transferring any input
feedback. Every accepted occurrence remains in its result inventory, including
unreached nodes with empty feedback. Root and child input projections use the
accepted START result, preserving feedback for literal and defaulted inputs.

AGENT backward validates retained node data and uses the exact recorded model
configuration for at most one tool-free call. It supplies authored/rendered
prompt, resolved locals/domains, delivered inputs, output/failure, ordinary
feedback and downstream context. The private `_evaluation_evidence` context key
comes only from accepted evidence and cannot be supplied by node capture. Base
system prompt, ordered recalled sections, and visible interactions remain
separate recorded components; tool calls/results are inert JSON in the prompt.
There is no tool grant/loop, recall, formatter replay, or config fallback.
Unreached AGENTs do no backward model work. Validated variable feedback is advice,
not an optimizer or a variable/model update.

`BackwardAttribution` references the sending occurrence, returned input message,
route contribution, and incoming parent positions. `AttributedBackwardResult`
keeps those references alongside the received messages and validated result.
Materialized downstream context and ordering keys remain transient. Captured
structural paths establish message ancestry only when the complete returned input
channel matches their prescribed transfer; transformed custom feedback carries
its own message identity.

Accepted growth installs candidate child-input connections through the same trace
delta. A connection targets a child START input, or a fragment node's input that has
a default and no wiring, such as the next LOOP driver's field that the body omits.
A child's accepted capture remains unchanged; its effective input route
joins those candidates only for fields already captured. A spawner selects its
next-hop seed routes from that effective input route, so a LOOP field the body omits
keeps its lineage across iterations. A field in
`evaluated_keys` keeps its actual second binding route, even when the result is
literal or null. Failed binding remains incomplete. Consuming or pruning a child
removes its pending connections, while accepted boundary associations and input
routes remain in history. Replay derives the same connections after validating
the rebuilt expansion and digest; it never infers sources from matching values.

The package-root `run_flow` boundary commits the complete initialization record
before lending its state/log pair to `FlowEngine`. A direct runtime caller has the
same obligation: apply `RunStartedRecord` to a matching `RunState` and `RunLog`
before driving the engine. The engine borrows this authority and never closes it;
the boundary or direct caller owns its lifetime.

## Host-supplied skills

`skills/` is a leaf. It exports the frozen `Skill` value and `SkillError`;
`skills/catalog.py` holds confined bundled-file reads, and `skills/tool.py` holds
the `load_skill` tool schema and the system-message listing. It imports only
langchain-core and the standard library. `compose/skills.py` reads skill folders
from disk, because only `compose/source/` may parse YAML and `parse_yaml_literal`
is its safe entry point.

Before it opens the log, for fresh runs and for resume, the run boundary
(`run.py`) walks static nodes and their `origin_children()` for the `needs_skills`
trait. A non-empty grant needs a `skills=` mapping, and every listed name must be
in it. `FlowEngine(skills=...)` treats the mapping as opaque. The evaluator passes
it only to nodes whose `needs_skills` trait is `True` (`AgentNode`), so runtime
stays kind-blind.

The AGENT renders the sorted listing into `base_system_prompt`, after the effective
system prompt. The memory context opens with the effective system prompt without
the listing, so skill grants do not change recall queries. The AGENT
serves `load_skill` calls through the same `invoke_tool_effect` path as registered
tools and records each result as evidence. A non-empty grant is AGENT execution
identity (`sorted(names)` or `"all"`); an empty grant is omitted so older
fingerprints hold.

## Host steering inbox

`nodes/base.py` defines the runtime-checkable `Steering` protocol:
`drain(node_id) -> list[str]` and `interrupted(node_id) -> bool`. `run_flow` and
`resume_flow` accept `steering=`, check it with `require_steering` before opening
the log, and pass it through `FlowEngine(steering=...)` to `eval_node`. The
evaluator supplies `caps['steering']` only to nodes whose `needs_steering` trait
is `True`. `AgentNode` reports its authored `steerable:` flag, so runtime stays
kind-blind. The capability wrapper turns inbox exceptions and wrong return types
into `_CapabilityFault`, so they escape as host errors, not node failures.

`run_agent` in `nodes/agent/loop.py` reads the inbox. `steerable:` is rejected on
non-`str` output, both at YAML load and in `AgentNode.validate_execution_setup`, so
a steerable AGENT always takes the text path. An AGENT without `tools:` or `skills:`
runs that path once: one model call, where steers queued before it ride along and an
interrupt cuts the streaming reply. At the top of each iteration `run_agent` checks
`interrupted`, then drains steers into `messages` and `new_messages` as
`HumanMessage`s, so `invoke_model` records them with the next model request. It checks `interrupted`
again before each tool call, and `stream_reply` checks it as each chunk after the
first arrives, before showing it. A stream that runs out is never cut, so a
single-chunk reply, or an interrupt that lands on the last chunk, returns the
complete reply and the tool loop skips its calls (unless a call's arguments hold
a NaN or infinity, which fails the node first). A mid-stream interrupt sends
the `final=True` chunk and raises `ReplyInterrupted` (a leaf in
`nodes/agent/interrupts.py`, shared by `live.py` and `memory.py`) with the visible
text so far and `usage_of` the chunks received; `observe_model_call` reports that
usage on `ModelCallFinished` with no `error`, and `invoke_model` observes the model
effect as `interrupted` with the partial reply (it holds only text, so it is always
finite JSON). `record_skipped_tool` records each
call not run as an `interrupted` tool effect paired with an error `ToolMessage`,
and `observe_skipped_tool_call` emits `ToolCallStarted` then
`ToolCallFinished(error="interrupted")` for it. `run_agent` then calls the turn's
optional `interrupt()` (the session `_Turn` closes as `interrupted`) and returns
the latest reply's visible text as a normal `Output`.

Closing the stream generator must release the provider connection.
`NormalizedChatAnthropic._stream` closes the SDK's HTTP response itself, because
langchain_anthropic iterates the SDK stream without closing it.

No run-log record kind is added, and `NodeSucceeded` carries no interrupted
flag. A finished node replays its recorded output without calling the inbox. A
node retried after a crash or failure sees earlier steers only through
session-memory history.

## Host-injected agent memory

Authored `long_term_memory` declarations name memories and their types (see
[flow syntax](../../docs/syntax.md#agent-memory-long_term_memory)); they never name
Python classes, storage locations, credentials, or tenant identity. Compose parses
each entry into `MemoryDecl(name, type, config, site, origin)`. Its
`DeclSite(scope, owner, node)` holds the import scope (the `uses:` aliases from the
root file, outermost first), the lexical flow owner, and the AGENT node id, or
`None` for a flow declaration. `_rebase_external_child` calls
`compile.memory_layout.scope_memory_declarations` to prepend the import alias, so
the memory key `".".join((*scope, name))`, such as `notes` or `lib.notes`, is fixed
before any per-call-site copy. One key is one memory: the two call sites of one
def share it, and one file imported under two aliases has two.

`compile.memory_layout.resolve_memory_layout` is the single check authority. It is
pure, reads the authored graph, and runs at load (errors become a located
`LoadError`), in `build_memories`, and in `run_flow` setup when memory is enabled.
It checks type and config agreement per key, session placement, scope-segment
names, and casefold-unique keys, then returns a key-sorted memory table and one
target per placement. A placement's stack lists the enclosing flows' keys,
outermost first through CALL, MAP, and LOOP, then the AGENT's own keys; a repeated
key keeps its first position. Dynamic MAP items and LOOP iterations contribute
occurrences under their template's stack rather than adding keys. Runtime remains
kind-blind: it consumes only each executing node's compiled `MemoryScope` and an
injected recall callable.

`run_flow(..., memory=registry)` never constructs memory. Package-root
`memory_host.build_memories(flow, root=..., storage=..., supplied=..., factories=...)`
returns a `MemoryRegistry` keyed by memory key. A `storage` value must be a `str`
or `os.PathLike[str]` local file path; anything else is a `MemorySetupError` naming
the key. `run_flow` requires every declared
key, rejects one object under two keys, and checks a registry's types against the
table; `resume_flow` and delayed backward repeat that type check. It opens one run session per key and stores the memory table, every
placement's `memory_stacks`, and the sessions' exact public snapshots in
`RunStartedRecord`. `memory=None` leaves authored declarations dormant. For each
opted-in AGENT node evaluation, recall returns one source-neutral section per key in
stack order. `run_agent` renders those sections as canonical JSON system
material labelled by memory key. The
encoding prevents memory text from breaking or spoofing the envelope, but the
text remains model-visible system material and is not isolated from prompt
injection. Accepted invocation evidence is recorded beside the ordinary node
outcome, including visible model retries within that invocation. Reducer-rejected
node proposals are represented by their accepted canonical failure. Discarded
late siblings and transient stream chunks do not become learning evidence.

The `yian run` host passes `--memory-root DIR` and repeated
`--memory-storage KEY=PATH` to `build_memories`; without them, built-in memories
are process-local. A flow that declares a custom type fails setup in the CLI,
`--resume` included, and runs through the Python API with `supplied=` or
`factories=`. Backends open storage only on the first `open_run` or `reset`, so
terminal inspection and dormant logs acquire no memory resources. The CLI closes
the registry after engine work drains. Runtime does not construct memories, and
CLI activation does not enable backward learning or change Python's `memory=None`
behavior.

`learnable=True` enables retention only, with or without enabled memory.
Forward returns before any backward hook, learner, or commit. An explicit
`RunResult.backward(feedback=None)` starts the separate attempt driver after
terminal success or failure. It leaves forward values, failure, and events
unchanged and exposes a separate `BackwardOutcome`; `RunResult.learning` is the
learning projection of that observed outcome. With no effective feedback, the
node callback fold does no work, but every memory key still learns.

`runtime.learning` owns pure root seeding/projection, ordered message transfer,
complete attributed-result validation, and owner-context construction. The
callback driver imports those helpers; state/replay never imports the driver or
invokes callbacks. `RuntimeBackwardState` is the sole attempt authority and holds
the recorded memory-key order. Forward execution keeps no per-memory episode
inventory: `project_owner_contexts` is the only episode projection.

`project_owner_contexts` consumes the recorded memory table and stacks and every
accepted occurrence, including CODE, MAP, growth, causal route failures, and early
binding failures. A key with no occurrence gets a context too.
`MemoryEpisode.execution_seq` is the physical accepted start sequence, not
completion order. Each retained exact `MemoryScope` must equal its recorded stack;
the stacks select receiving keys and memory episode occurrences.
Feedback `owner_static_path` and `occurrence_path` retain the original
root-relative `ExecutionScope` coordinates, explicitly absent if unavailable.
Each ordinary feedback entry preserves its channel, target name, original value,
message ordinal, and contribution context. Self/variable messages remain
separate ordered entries with empty contribution context. Received/generated
input and opaque-stop entries reconstruct context through the existing route
interpreter; literal/default feedback remains local without a fabricated source.
Defined JSON/materialization failure becomes a memory-attributed context failure
before Ready; invalid record authority remains an error. Memory JSON allows 128
levels, and run material at the value limit nests about 108, so it always fits. A
context that still cannot be built (for example from a log recorded before the
limit) raises `OwnerContextError` carrying the owner occurrence's `BackwardContext`;
each occurrence's feedback is validated on its own so the failure is attributed to
it. The driver records a `memory.invalid_context` `BackwardFailure` naming that
occurrence (`scope`, `run_id`, `node_id`, `node_started_seq`) with the message
`memory learn context could not be constructed for node '<id>': <cause>`; replay
requires the named occurrence to be accepted and to receive that memory. Only a
failure in root fallback feedback carries no node. `MemoryRuntime.propose_learning`
rebuilds each `LearnContext` instead of deep-copying it, which re-validates and
detaches the feedback without recursing through a value at the limit.

The explicit driver accepts `BackwardStartedRecord`, completes the single reverse
fold and every `MemoryProposal`, then accepts one `BackwardReadyRecord`.
No per-node progress or per-memory learning records exist. A known computation
failure finishes without partial root feedback; a known learner/context/proposal
failure retains the completed root result and full attribution, discards temporary
proposals, and commits nothing. Unexpected errors and host/cancellation controls
escape without a fabricated Finished.

After Ready, each memory key receives exactly one batch in recorded key order,
including empty batches. Its positive `attempt_id` and deterministic
`memory_commit_id` are validated on `MemoryCommitRequestedRecord` before the
backend call. Backend mutations are detached from accepted authority. A success
or known failure is recorded afterward; a known failure stops later keys.
`BackwardFinishedRecord` settles the attempt. Commit dispositions are
`unrequested`, `unknown`, `succeeded`, and `failed`; a returned success without an
accepted observation remains unknown. Known learner/commit failure makes learning
failed regardless of backend retry hints. All successful confirmations without
Finished still project an interrupted backward outcome and incomplete learning.
An accepted failed confirmation instead projects failed learning.

Only an unfinished attempt blocks a new Started, and the driver closes it first
with an interrupted Finished that projects its prefix. Every explicit call is a
new attempt that recomputes everything from original snapshots with fresh
sessions; Ready proposals are never resumed. Every requesting attempt uses
`memory_commit_id(..., attempt_id)` (`ac-memory-v3`) over the run ID, attempt,
memory key, and mutations, so identical batches in different attempts never share
an ID. `LearnContext` carries no attempt
identity; backends tell attempts apart only by commit ID. No rollback, rebase,
or automatic recovery of backward work occurs.

Each memory key learns once per attempt, in sorted key order. Its batch holds
every accepted occurrence whose recorded stack contains the key, of any kind.
Unattributed root feedback reaches only keys owned by exactly
`DeclSite((), ("root",), None)`, so an imported file's flow memory never receives
it. Node feedback propagation remains a runtime operation; a memory callback does
not mutate the authored graph. Concrete implementations live behind the
source-neutral `Memory`/`MemorySession` protocols, and one backend instance holds
one memory's state. The provided `ReflectiveMemory` is a backend whose model call
and query rewrite are backend-owned. The provided
`yian.memory.episodic.EpisodicMemory` is a model-free
backend. Its learner deterministically proposes one canonical append per owned
episode; commit applies a complete batch atomically and idempotently, then bounds
the memory by its `max_episodes` setting. Contextual recall ranks lexical overlap
and then recency up to `recall_limit`; context-free recall returns the full
bounded history chronologically. Sessions read immutable snapshots, including
after newer commits, while stale sessions cannot learn or commit. Public JSON
snapshots support the existing recovery boundary or restoration into a pristine
instance. Both default to process-local state and support host-only `storage=path`.

The provided `yian.memory.profile.ProfileMemory` keeps one markdown
document in a host `MarkdownStore`; `build_memories` supplies a
`FileMarkdownStore` for a path and an `InMemoryMarkdownStore` otherwise. Its session
snapshot records only the pinned `profile_version` and `text_sha256`, so resume
and backward reopen that exact version. Every stack that holds the memory recalls
the full document. Learning makes one
tool-free model call over the feedback, the current document, and budgeted
episodes with their visible interactions, and returns the whole rewritten
document. Commit is a compare-and-set store write that never calls a model. Only
`profile.version_mismatch` is retryable, because only it guarantees nothing was
written. An attempt killed during commit stays `unknown`; the store's write
provenance holds its `commit_id`, so hosts check there before retrying.
`FileMarkdownStore` rejects the same non-file paths as SQLite storage
(`storage.path`) and, like it, resolves a symlinked path to its target.
`ProfileMemory.close()` does no I/O; afterwards the backend and its sessions
raise `profile.closed`; a session closed on its own raises
`profile.session_closed`.

### Optional backend durability

`memory/storage/sqlite.py` holds `SQLiteStorage`, a lower-level SQLite value adapter shared by the
episodic, reflective, and session backends. One instance owns a local file for
its lifetime; a second owner fails immediately. Backends validate complete semantic candidates,
commit detached finite JSON in a transaction, then publish RAM. Private inventory
includes commit tombstones and adoption flags; SessionMemory also preserves its
derived summary cache. Public snapshots retain their existing formats and exact
opening pins. No callback, queue, handle, or provider client is serialized.

Host constructor values and YAML `config` stay outside private storage. An
unset episodic or reflective limit adopts the saved value, or else the built-in
default; an explicit value that differs from the saved one raises
`episodic.config_conflict` or `reflective.config_conflict`. Recorded run model
settings remain pinned. Current session history settings apply on reopen,
and a larger recent-turn count must recover raw context hidden by older caches.
Paths stay outside YAML and runtime records. `storage=None` performs no storage I/O.
A configured path opens on the first `open_run` or `reset`, under the backend's
lock, and PID ownership applies from construction.

Backend `close()` permits idle run handles and invalidates them, but fails
immediately while operations or queued turns are active. Closing before the first
open does no I/O; a later `open_run` or `reset` raises the backend's `*.closed`
error. In-flight model and
summary callbacks remain protected even after their run handle closes. Cleanup
must drain ownership without replacing a primary exception; inherited handles
check PID before any lock, including context finalizers. Failed I/O poisons the
handle until close/reopen. Reset preserves commit deduplication and never deletes
provider resources. This does not supply exactly-once external execution or
shared-cluster storage.

### Session conversation foundation

`yian.memory.session.SessionMemory` supplies conversation continuity
for AGENT execution through `run_flow`. It
owns message history updates directly. The CLI builds declared built-in
memories through `build_memories`.
`SessionMemory` is a concrete backend;
`MemorySession` remains the run-handle protocol. `MemorySession.recall()` returns
text or `MemoryContent(text, transcript)`. SessionMemory contributes both pinned
learned notes and a `TranscriptSource`; other memory types can supply plain text or
the same rich carrier. This is one context interface with distinct content forms:
notes are instructions/facts, while the transcript retains structured messages.
`MemoryContextRequest` combines recall inputs, accepted invocation identity, and
message compatibility. The admitted `MemoryContext` carries ordered sections and
an optional `ConversationTurn`; live handles never enter snapshots or run logs.
The backend has no provider session references or start/resume callbacks. Existing
message-history digests retain reserved null fields in their wire format; nonempty
legacy native references require fresh memory and cannot be adopted.

The ownership split is a pure state fold plus an effectful interpreter:
`memory/contracts.py` defines immutable carriers, `memory/conversation.py` owns
`advance_conversation`, strict JSON codecs, and digests, and `memory/session.py`
owns live views and FIFO leases. The fold and backend
import only the standard library and lower memory modules, never nodes, runtime,
compile, compose, or providers. The facade exports contracts only; concrete
backends are imported explicitly from their own modules.

`MemoryRuntime.open_context` recalls every key in the node's stack once, keeps
nonempty text sections in stack order, and admits a transcript only from a key
whose recorded `MemorySpec.conversation_eligible` is true (exactly one AGENT
owner). A transcript from any other key, or a second transcript in one context,
raises `MemoryRecallError`. Runtime does not inspect backend classes or detect
transcript methods on sessions. Text-only memories contribute through the same entry point. The engine
captures accepted run/start identity before observers or worker submission;
`eval_node` binds that identity and authored scope into a lazy, single-use
`open_memory_context` callback. The callback takes current instructions, rendered
prompt, and compatibility, never owner IDs.
`AgentRunContext.memory_context` replaces separate recall and conversation fields.
The evaluator keeps the lease through the node body, generator drain,
post-assertions, and result-shape validation. Completion means wrapped execution
finished, not that the dispatcher subsequently accepted typed output. Cleanup
preserves primary errors; no live lease travels in worker results. Nodes without
the trait and recalls without a transcript open no conversation turn.

`nodes/agent/memory.py` is a stateless provider adapter, not a second store.
It encodes detached finite-JSON LangChain messages and actual effect observations
through the admitted turn. Provider plus message format define compatibility;
compatible model changes retain history. Current instructions, recall, and tool
grants apply on every call. Historical system instructions do not become current
instructions. Restoring history never dispatches tools. Context-only projections
make missing tool results unknown and schema pseudo-calls inert; raw history is
unchanged. Replies, including native structured raw replies, are saved before
cancellation or parsing. Memory faults bypass structured/provider retry loops.

`SessionMemory(history_budget_chars=32_000, recent_turns=4)` accepts a positive
history budget and nonnegative prior-turn count as keyword settings, from host code
or the declaration's `config`. A session memory is declared on exactly one AGENT,
and each instance holds one conversation with one FIFO queue. MAP items, LOOP
iterations, the call sites of one def, and concurrent runs on the same instance
take turns, and no fork mode exists. Run setup rejects a session key whose
placements resolve different conversation compatibility. FIFO is admission
order, not a MAP input-index guarantee. A turn reads its live
baseline only after acquiring the lease. The run handle's detached opening
snapshot stays pinned as live turns advance history. Public
version-3 snapshots contain namespace, generation, one conversation view or null,
and learned notes, not transient queues, leases, summary cache, or commit bookkeeping.
Older snapshot versions fail with `session.invalid_snapshot`. Storage defaults to
process-local. Reopening an older
snapshot against a used backend reads latest compatible live history without
rewinding it; a pristine backend can restore only the history in the snapshot.

The recalled `TranscriptSource.open_turn(request, check_stopped=check)` acquires
the conversation lease.
Its `request` and `observe` methods validate and publish directly and atomically
in the backend. Provider/tool calls and stop callbacks run outside the condition
lock. There is no checkpoint sink, prepared-apply callback, or reconciliation
channel. Cooperative stop checks can cancel queued work; actual observations are
retained before honoring a stop during an effect. A model or tool call that
raises while a stop is pending is observed `interrupted` and the run stops
instead of failing (a host callback fault still escapes). If the stop is
honored only after the failure is recorded, the effect stays `failed` and the
run still stops. Conversation progress is independent of `learnable`.

API context preparation uses an optional concrete-turn `prepare_history`
capability without changing mandatory protocols. A second optional capability,
`interrupt()`, asks the turn to close as `interrupted` when its body returns
normally; a steerable AGENT calls it after a host interrupt, and a turn without
it closes as usual. `memory/summary.py` owns pure
admission-led selection and cache-prefix proofs; SessionMemory owns cache
publication. Project and summarize callbacks run outside the Condition under the
existing FIFO lease. They receive detached values, never runtime state.
The stateless API adapter projects selected invocations safely, excludes old
system/developer instructions, and uses the current unbound model for a summary
before tool binding or structured-output generation. A callback model/response
failure remains an ordinary node failure; codec/backend failures remain capability
faults. Do not turn a summary failure into an answer or structured retry.

The soft character budget applies to historical projection and summaries, not
fresh instructions/input, tool schemas, or learned notes. Preserve the latest
four prior whole admissions plus the active invocation, including failed or empty
interrupted turns. Keep all raw history. A valid oversized summary is retained
without truncation or automatic retry; cached prefixes extend only when needed.
Cache identity proves the exact covered raw prefix and projection version, not
the current live-view revision. Old snapshots never rewind this derived state.
The final structured pass uses prepared prior context plus a fresh safe projection
of the active invocation; it must not re-expand old history or retain summaries
and synthetic repairs as new raw messages. Reset clears derived summaries.

Explicit API backward learning reuses the existing self-feedback as learned
notes through session `learn`, `commit`, and contextual `recall`.
There is no extra model call. Feedback's accepted execution sequence joins to
the episode's root-relative authored leaf path; the memory learns notes only from
invocations it admitted. Commit validates the whole deterministic
mutation batch before atomic publication and rejects conflicting commit IDs.
Notes are separate from raw conversation values and never rewrite history.
Recall returns every note of the instance from the run-opening pinned snapshot,
not the latest admission-time history. Context-free recall supplies empty text
alongside the transcript source. New
commits affect later runs. Reset clears history and notes while retaining commit
deduplication to prevent replay from resurrecting cleared notes.

Failed/interrupted turns keep completed observations and leave unobserved requests
pending, not falsely successful. A new invocation can continue that history;
successful closure checks only its own effects. Restoring context never itself
executes stored tools, but ordinary node retry may resend requests and repeat
actions. The backend owns no provider or tool client. A changed exact compatibility
descriptor requires reset; provider-specific compatibility rules are not part of
this backend.

Closing a run handle cancels its queued tickets and prevents further requests,
but an active lease stays held until its owner exits. This prevents overlap while
an external call is still finishing and lets the owner retain its actual result.
Released turns cannot append late observations. Cancellation stays effective even
if caller code catches the original stop exception.

Backend-wide `reset()` fails immediately and unchanged with `session.busy` while
any turn is active or queued. When idle, it clears every local conversation and
advances the generation, invalidating old handles.
It does not delete external provider history or files. The
[Python API guide](../../docs/api.md#sessionmemory-foundation) shows host injection
and the lower-level snapshot/reset interface.

## Durable recovery

`run_flow(..., log=path)` exclusively creates a new JSONL authority. Its first
record captures the run ID, authored-flow fingerprint, complete initial typed
pool, START wiring, worker count, fully resolved LLM configuration, the required
`learnable` retention flag, and optional memory table, stacks, and snapshots. Expansion
records retain canonical growth seeds plus topology digests, so replay rebuilds
the same nested CALL/MAP/LOOP graph without re-running the spawner.

`resume_flow(flow, log=path)` locks an existing log, durably removes bytes after
the final complete newline, validates its canonical contiguous prefix, and checks
the supplied flow fingerprint. It folds historical records silently through the
same reducer and injected growth builder used live. An eligible nonterminal primary
prefix gets exactly one `RunResumedRecord`, which puts durably started nodes with no
durable outcome back ahead of the ready frontier before dispatch continues. A
terminal primary returns its observed backward prefix without appending records
or invoking graph nodes, backward callbacks, learners, or commits. Recovery uses the
recorded input, run ID, and complete LLM map rather than current ambient defaults.

An ordinary node retry receives a new start sequence.
A retry may resend model requests or repeat tool actions; there is no exactly-once
guarantee or exact mid-request continuation. Runtime does not reconstruct a
conversation or fabricate completion for unknown effects. SessionMemory owns the
latest saved history, and the run log is not a replacement memory store.

When an unfinished recorded run enabled memory, recovery requires an
implementation for every recorded memory key and verifies that opening each session reproduces the exact
recorded snapshot. Terminal inspection opens no memory; supplied mapping references
remain available to the returned handle for later explicit attempts.

A delayed handle owns private program/data snapshots and memory references, not
open forward resources. The run boundary anchors durable paths at acquisition,
reopens and replays authoritative records per call, and restores fresh sessions
from recorded public snapshots. Process-local reopening validates and detaches
every schema/run/sequence carrier without reopening the original closed log.
A nonblocking per-handle guard covers replay, work, prefix refresh, and cleanup;
durable log locks exclude competing handles. Invalid submissions preserve that
result's last observed objects. Append/cleanup errors preserve the primary fault
and accepted prefix; a handle that cannot refresh its authority becomes unusable
instead of reusing stale proposals or requests.

This provides deterministic engine-state recovery, not exactly-once external
effects. An eligible interrupted node with a durable start and no outcome is
retried. Callers using the optional SessionMemory turn capability consume the
backend's latest saved history, but requests and external effects may repeat.

## Blocking effects and structured drain

Fresh CLI and Python runs default to eight workers. An explicit `num_workers=0`
selects serial evaluation; positive values bound concurrent node execution.
Resume without a worker override preserves the latest recorded count.
Each pooled node inherits a fresh copy of the host's context variables, preserving
host cancellation for nested flows while isolating node-local context changes.

Authorable blocking effects stay inside their node calls. An AGENT node may invoke the
explicitly granted `ask_user` tool, HUMAN_INPUT always invokes the same run-owned
handler, and WAIT blocks on its absolute UTC deadline. The model sees only
`ask_user`'s `question` and optional `options`; the tool cleans `options` leniently
and never fails a call over them. `questions`, `answer_schema`, `title`, and
`node_id` are injected, so only HUMAN_INPUT's trusted call can set them. The AGENT node checks the
run-owned stop signal between model/tool effects; WAIT blocks on that signal. Each
returns `Output` from the same invocation.

Tools reach nodes the same kind-blind way. `run_flow` and `resume_flow` build one
immutable `ToolSet` from `tools=` before any log work. `FlowEngine` passes it to
`eval_node`, which adds `caps["tools"]` only for nodes whose `needs_tools` trait
is set (TOOL, HUMAN_INPUT, and AGENT). `tools=None` reads the global registry. A
mapping is the run's whole tool set. Hosts that want both pass
`{**registered_tools(), ...}`; that copy is taken at call time, so tools must be
registered before the run starts. `ask_user` is always per run and never falls
back to the registry, so concurrent runs in one process keep separate handlers.
Pooled nodes of one run can call the same handler at once, so it must be
thread-safe. Tools are host callables: the log does not record them, and a host
passes them again on resume.

A `tools=` value that is a list or tuple is a named tool group. The `ToolSet`
flattens every group into its tool map, keeps each group's ordered names, and
rejects a group tool whose name is reserved (`ask_user`, `load_skill`), is a
top-level key, or names a different tool in another group, so each name calls one
tool. The same tool object may sit in several groups, since host permission groups
overlap. An AGENT's `tools:` grant may name a group; the tool loop calls
`ToolSet.expand`, which drops repeated names, before binding tools and before its
grant check, so the model and the check see each tool name once.
Before any log work, fresh runs and unfinished resumes walk static nodes and their
`origin_children()` through the kind-blind `Node.granted_tools()` hook (AGENT
returns its `tools:`) and raise `ValueError` when a grant does not resolve in the
run's `ToolSet`. The walk covers nodes that may never run, such as an unreached
branch or a LOOP body. Only the authored names enter the
fingerprint; group contents are host capabilities, so a resume may pass different
contents and restarted nodes use them.

The same walk then reads the kind-blind `Node.host_tool_call()` hook. TOOL returns a
`HostToolCall`: its `tool_id`, input names, proven input types, and declared output
type. The compose pass stamps `ToolNode.input_types` at load from the bindings' proven
types; they are derived from wiring, so they stay out of the fingerprint. For each
call whose tool resolves, `tools.types.tool_types` reads the tool's declared argument
and return types, and the boundary compares them with `type_relation`. An unknown
input, a missing required input, or a disjoint input or output type raises
`ValueError` before any log work; an unresolved `tool_id` is left to the node. The
`output:` gate is the ordinary success path: the engine validates each returned value
against the node's output contract, so a host reports an expected outcome such as an
interrupt by returning a variant of that type. An exception still fails the node.

Pooled drives own their executors through a full join: an abnormal dispatch exit
signals cooperative stop before joining, so the boundary never releases its run-log
lock while a worker can still produce an outcome. When a node outcome fails the run,
the dispatcher publishes that failure's durable observers before the join, so a host
can withdraw a sibling's pending `ask_user` prompt instead of waiting for its answer.

## Dynamic growth and stable identity

A spawner node returns `Grow(seed, prune)` without importing `compile`. The
kind-aware `compile.expand.build_grow(spawner, seed)` converts the canonical seed
into a deterministic `Flow`; package-root `run.py` injects that callable into both
live execution and replay. Seed values are resolved data: the child START receives
each one verbatim, so text containing `${...}` (a shell script, an LLM answer) is never
evaluated a second time inside the child. Runtime validates and splices the fragment
generically, remaining kind-blind. During that compile-owned construction, CALL/MAP/LOOP
prefix child clones with `spawner.failure_trace()`; MAP also stamps its generated
boundaries, while a LOOP's next driver keeps only the enclosing
`origin_context`. A LOOP merges the next carried record at respawn: the next driver
gets a parameter default for each field the body omits, from the canonical seed, and
only returned fields are wired from the body END. Expansion digests cover entry/end
IDs and topology but exclude that diagnostic context, while loop pruning retires each
finished iteration's body nodes. A pruned id is retired with the live nodes grown under
it once it and all of them are settled (taken, skipped, or expanded; none ready or
running). Until then they stay live, along with any pruned id they read, and each later
growth retires them once settled (`RunState.deferred_prune`); a self-pruning LOOP driver
is always retired. The reducer finds grown nodes from accepted growth records
(`RunState.growth_owner`). `NodeExpandedRecord.prune_ids` keeps
only the ids the node asked for; replay re-derives the rest.
A growth costs its fragment plus its prunes, not the run's graph so far. The reducer
checks the fragment against the live state through indexes: the adjacency lists and
`RunState.growth_index`, a derived reverse lookup of which live nodes name each node and
which nodes each owner grew. Commit applies the change to the live containers in place
after the record is appended.

Compile also owns stable identity for the static program. `flow_manifest`
canonicalizes execution semantics, including every authored value under any name
(nothing is redacted), `flow_fingerprint` hashes that manifest, and
assertion identity includes executable and display text in list order while
remaining origin-neutral. `expansion_digest` applies the same projection to
dynamic fragments. CODE identity includes its executable `code` text (for a
reference, the `module:function` token, not the callable's source) but excludes
`code_origin`, `code_line_origins`, and the retained compiled/owned-code caches,
while CASE identity includes predicate handle/expression semantics but excludes
each predicate origin. Thus source formatting, origin rebasing, and Python object
identity cannot change program or expansion identity. Because every authored value is
identity, compose rejects a literal value under a secret name at load:
`compose/validate/secret_literals.py` checks node bindings, case values, and flow
outputs before typed bindings are split, `compose/helpers/secret_literals.py` holds the
walker and messages that defaults, `env:`, and `call(...)` arguments share, and the one
list of secret names is the stdlib-only leaf `secret_names.py`. `resolve_llm_cascade` returns a
complete map keyed by slash-separated paths of static AGENT nodes. The run boundary reads
process provider/model defaults and nonsecret connection selections once at fresh-run initialization;
`apply_effective_llm_map` restores the
recorded map during recovery without consulting the environment.
