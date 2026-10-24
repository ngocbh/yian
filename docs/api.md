# Python API

Use the engine as a library: load a flow, run it, and optionally retain an
authoritative append-only log for recovery. An AGENT node's `ask_user` call,
HUMAN_INPUT, and WAIT stay inside live node calls; their enclosing nodes return
ordinary outputs. For the node/flow/agent distinction, start with
[Agent](agent.md).

```python
from pathlib import Path

from yian import load_flow, run_flow

flow_path = Path("hello.yaml")
loaded = load_flow(
    flow_path.read_text(),
    search_paths=[flow_path.parent],
    source_label=flow_path.name,
)
result = run_flow(
    loaded,
    {"name": "Ada"},
    llm_config={"provider": "ollama", "model": "llama3.2:3b"},
)
print(result.status, result.output)
```

## Public surface

These names are exported from the top-level `yian` package:

| Name | What it is |
|------|------------|
| `load_flow` | parse + compile + validate Compose-YAML into a `ComposedFlow` |
| `run_flow` | drive a `ComposedFlow` to a terminal, returning a `RunResult` |
| `build_memories` | build one backend per memory key a `ComposedFlow` declares, returning a `MemoryRegistry` |
| `MemoryRegistry` | read-only memory-key -> backend mapping with declared `types` and `close()`; pass it as `run_flow(..., memory=)` |
| `Failure` | immutable semantic failure (`code`, message, node, origin trace, debug cause) |
| `ComposedFlow` | a compiled, validated flow with an optional resolver and `diagnose(failure)` |
| `CompiledFlow` | the compiled flow IR (`loaded.compiled`) |
| `VariablePool` | the typed value pool |
| `evaluate_when` | evaluate a boolean `when:`/assert expression |
| `LoadError` | bad-flow exception containing one canonical `diagnostic` |
| `FlowValidationError` | raised on a failed validation check |
| `ExpressionError` | raised on a bad `${...}` expression |
| `ToolContext` | the per-call context a host tool receives when its first parameter asks for it (see [Tool context](#tool-context)) |
| `FunctionTool` | a host tool given as a name, a description, an argument JSON Schema, and an `on_invoke` call |
| `function_tool` | build a `FunctionTool` from a function, optionally overriding its name and description |
| `PermanentToolError` | an exception a tool raises when another attempt can't help; a TOOL node doesn't retry it (see [TOOL retries](#tool-retries)) |
| `TransientToolError` | an exception a tool raises to ask a TOOL node to retry, optionally after `retry_after` seconds |
| `ReportedToolError` | an exception a tool raises for a failure the calling AGENT's model reads as the call's result (see [TOOL retries](#tool-retries)) |

The MCP names (`mcp_server`, `McpServer`, and the `McpError` classes) live in
`yian.tools` (see [MCP servers](#mcp-servers)).

Recovery, the result type, and the steering protocol live in `yian.run`:

```python
from yian.run import RunResult, Steering, resume_flow
```

The lower-level `FlowEngine`, `RunState`, `RunLog`, `MemoryRunLog`, and
`JsonlRunLog` interfaces are scoped to `yian.runtime`. A direct engine
caller must first apply a `RunStarted` transition to a matching state/log pair;
the package-root functions below own that setup for normal callers.

## `load_flow`

```python
load_flow(
    text: str,
    *,
    child_resolver: ChildResolver | None = None,
    search_paths: list[str | Path] | None = None,
    source_label: str = "<flow>",
) -> ComposedFlow
```

Parses, compiles, and validates the flow `text`, including every authored input
default. Pass `search_paths` so that a flow's `call:` / `uses:` references resolve
relative to the flow file's directory — typically `search_paths=[flow_dir]`—or
inject `child_resolver` for host-provided external flow resolution. An injected
resolver takes precedence over local search paths. A flow a custom resolver returns
counts its own `uses:` chain through `ComposedFlow.import_depth`; one built without
`load_flow` counts 0. Pass the filename as
`source_label` when `text` came from a file. Raises
`LoadError` on a bad flow (compile-time errors surface here, not at run time);
its immutable `.diagnostic` owns the message, notes, and every ordered
`DiagnosticFrame`, including frames from imported files. A YAML document nested
deeper than `YIAN_MAX_YAML_NESTING` (200) levels, or one that expands to more than
`YIAN_MAX_YAML_NODES` (100,000) nodes when each alias counts as a copy of what it
names, raises a located `LoadError` (see [Syntax → YAML values](syntax.md#yaml-values)).

```python
from yian import LoadError

try:
    loaded = load_flow(text, source_label="flow.yaml")
except LoadError as error:
    print(error.diagnostic.message)
    for frame in error.diagnostic.frames:
        print(frame.label, frame.range.start_line)
```

The returned `ComposedFlow.origin_resolver` has one exact route for every
reachable source origin. It is `None` only for a source-free graph. External
files remain canonical in the path cache; each `uses:` alias embeds a rebased
copy whose routes still resolve to the original file.

Changed (TS yian): `loadFlow(text, { childResolver, searchPaths, sourceLabel, baseDir, codeRoot,
env })` returns a Promise of the `ComposedFlow`, and a `LoadError` is its rejection. A
`childResolver(path, version)` may return the flow or a Promise of it. Every `uses:` file
is read before the flow assembles. `baseDir` is the directory a CODE node's
`./file.ts:fn` reference resolves against (default: the first search path, else the
current directory); a file read through the search path resolves against its own
directory, so a `childResolver` that loads a child file passes the child's own directory
as its `baseDir`. With `codeRoot`, a CODE node's `./file.ts:fn` reference whose file's real
path is outside the real path of `codeRoot` (symlinks are followed on both sides) is a
`LoadError` at the `code:` value. This covers the files the search path reads. A flow a
`childResolver` returns was checked by its own load, so a resolver that loads a file passes
`codeRoot` on. An inline `code:` body and a module's own imports aren't confined; without
`codeRoot`, nothing is. A relative `codeRoot` resolves against the current directory, and one
that doesn't exist refuses every reference. The check runs at load: the file is imported by
path when the node runs, so a file replaced after the load isn't checked again; `resumeFlow`
with `codeRoot` checks the recorded flow's files again (see [Durable recovery with
`resume_flow`](#durable-recovery-with-resume_flow)). `env` is the environment this load reads
the `YIAN_*` values from. The default is the process environment, or the calling load's when a
`childResolver` calls `loadFlow` without `env`.
A bad `YIAN_*` value in `env` rejects with `DefaultsError`, not `LoadError`.

## `run_flow`

```python
run_flow(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 8,
    on_event: Callable[[Any], None] | None = None,
    on_stream: Callable[[StreamChunk], None] | None = None,
    llm_config: dict[str, Any] | None = None,
    memory: Mapping[str, Memory] | None = None,
    learnable: bool = False,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    context: Any = None,
) -> RunResult
```

Coerces supplied `inputs` against the flow's declared types, snapshots the
complete LLM configuration, initializes the run log, and drives the flow to a
terminal. Stored defaults are not coerced. Every `inputs` key must name a
declared input; an unknown name raises `ValueError` listing the declared inputs,
and `inputs` that is not a mapping raises `TypeError`, both before the log is
created. NaN, infinity, and complex numbers are not yian values: an `inputs` value
holding one at any depth raises `ValueError` naming where it sits
(`input 'x' at a[1]: ...`), also before the log is created. An `inputs` value may
nest 100 levels (each record and list is a level; `YIAN_MAX_VALUE_NESTING`); a deeper
one raises `ValueError` naming the input and path, before the log is created (see [Types
→ Nesting limit](typing.md#nesting-limit)). A non-finite value in an `llm_config`
provider block, such as `{"openai": {"temperature": float("nan")}}`, is rejected too: in
a flow file it is a load error, and in the host `llm_config` it raises `ValueError`
before the run starts when an AGENT uses it. An int longer than 4300 digits
(`YIAN_MAX_INT_DIGITS`) is not a yian value either: an `inputs` value holding one raises
`ValueError` before the log is created (`input 'n': integer with 5000 digits is larger
than yian allows (4300 digits)`; see [Types → Integer size
limit](typing.md#integer-size-limit)). A node assertion, CASE
`when`, or LOOP `while`/`until` set in Python whose expression doesn't parse (or calls
an unknown function or a builtin with the wrong arguments) raises `ExpressionError` (a
`ValueError`) naming the node and the field (assertion list and assertion, case handle,
or predicate keyword), before the log is created; `resume_flow` raises it for an
unfinished run before appending any record. Likewise a binding, flow output, AGENT
prompt, or `human_input` text set in Python whose `${...}` span doesn't parse, calls a
builtin with the wrong arguments, or calls any other function (in a binding, one that is
not a `typedefs:` constructor in the node's type table) raises `ExpressionError` naming
the node and field before the log is created; `resume_flow` raises it for an unfinished
run before appending any record.

- `log` — an optional JSONL path. `None` still uses the same record/reducer path
  with a process-local log. A path creates a new durable log exclusively; an
  existing file is never replaced.
- `run_id` — host-injected run identity (readable as `${system.run_id}`); minted
  when omitted and retained by every record.
- `num_workers` — defaults to `8`; `0` runs serially, and a positive value bounds
  the worker pool.
- `on_event` — receives post-durable observer events as they occur. While an
  AGENT node runs it also receives transient in-loop observers
  (`ModelCallStarted`, `ModelCallFinished` with `TokenUsage`, `ToolCallStarted`,
  `ToolCallFinished`), strictly between that node's `NodeStarted` and its outcome.
  In-loop observers are never durable, never members of `RunResult.events`, and
  never replayed on resume; filter them with
  `isinstance(event, yian.events.IN_LOOP_EVENTS)`. Observers from
  concurrently running nodes interleave (runs are pooled by default); correlate
  them by `node_id` and `exchange_id`. `usage` is the provider's own count
  (`None` when unreported) and `error` is an exception type name only, or
  `"interrupted"` for a tool call a [steering](#steering) interrupt skipped.
  Changed (TS yian): `conformance/schema/events.schema.json` describes each
  event's JSON form, with the snake_case keys hosts forward. `ReplayDegraded`
  (`node_id`, `reason`) is in-loop too: the AGENT's model left out replay state
  it couldn't send back (another provider's or account's reasoning, an unknown
  record version), while the replayed turns' text and tool calls were still
  sent. It arrives once per reason per node start. Every reasoning part an
  adapter produces is tagged with the route that wrote it; after a switch of
  protocol, provider, model (on Anthropic any Claude model may replay) or
  account, that reasoning and its signatures and ids are left out, never turned
  into text.
  Changed (TS yian): `ModelCallRetried` (`node_id`, `consumer`, `attempt`,
  `error: {kind, message}`, `retry_in`) reports a model call outside a forward
  node that failed with a retryable error and is sent again after `retry_in`
  seconds; it arrives before the wait. `consumer` is `"backward"` for AGENT
  backward (under the AGENT's `node_id`) or `"memory:<key>"` for a memory's
  learner (under `node_id` `"__memory__"`), and `error` is the provider error's
  kind and message, scrubbed of credentials. It is transient and not in
  `IN_LOOP_EVENTS`, since it comes from outside a node. A `ReplayDegraded`
  published during backward or with `node_id` `"__memory__"` comes from outside a
  forward node too. Python yian publishes no retry event.
  Changed (TS yian): a learner's `ModelCallRetried` and `ReplayDegraded`
  reach the `onEvent` given to `buildMemories`, never a run's `on_event`, since
  learners run outside any run.
- `on_stream` — receives transient `StreamChunk` values. Chunks are neither
  durable records nor members of `RunResult.events`. With `on_stream` set, AGENT
  chat replies stream visible text as `StreamChunk(node_id, "text", chunk,
  exchange_id=...)`; structured replies do not stream. A completed streamed call
  ends with one empty `final=True` chunk. A failed call ends without one and its
  `ModelCallFinished.error` is set; a host discards that exchange's partial text.
  The node value is authoritative: streamed text can differ from it in the
  whitespace between content blocks. Without `on_stream`, AGENT calls `invoke`
  exactly as before.
  Changed (TS yian): every AGENT call streams on the wire; structured replies
  still send no chunks. `ModelCallStarted`, `ModelCallFinished` and `StreamChunk`
  carry `attempt`, starting at 1. A retried call ends the failed attempt with
  `ModelCallFinished` holding its `error` and no final chunk, then starts
  `attempt + 1` in the same exchange. A structured reply that fails the output
  check finishes its call without `error`; the correction is sent as a new
  exchange.
- `llm_config` — the outer authored layer of the configuration cascade for AGENT nodes.
  The resolved map and process defaults (`YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL`,
  then the logins saved by `yian auth`) are captured at initialization. There is no
  built-in provider or model: when no layer gives an AGENT a provider and model,
  `run_flow` raises `ValueError` before the log is created, naming the agent and the
  fixes (`Agent 'write' has no provider or model; set llm_config.provider/model, pass
  --provider/--model, or run 'yian auth <provider>'`). An
  `api_key` entry, which needs `provider`, is the provider key for this run: every
  AGENT whose effective provider matches, including an `inherit: false` one, uses
  it instead of the provider's saved login or key environment variable.
  An empty key is rejected. The key never enters the run log, events, results, or
  manifests, and validation errors do not echo it. When a provider rejects the key
  and echoes it, in full or masked like `sk-abc****wxyz`, the key and those
  fragments are replaced with `[REDACTED]` before the failure is recorded. The same
  applies to whichever credential the call used instead: the provider's saved login
  or its key environment variable. The log
  records only a nonsecret marker on those agents, so pass the key again as
  `resume_flow(..., api_key=key)` or `result.backward(..., api_key=key)`; without
  it, agents marked for the key fail when they call their model instead of reading
  the environment. A key that marks no agent, and that no agent resolving its config
  when it runs could use, is ignored. The result of `run_flow`
  or `resume_flow` keeps the key in memory for its own `backward` calls. Flow files
  cannot set `api_key`. `ReflectiveMemory` rejects it, because its snapshots record
  the model configuration, so it always uses the saved login or environment
  credentials.
  `ProfileMemory` accepts one in its own `llm_config`, with `provider`, and does not
  record it.

  Besides `provider`, `model`, `api_key`, `base_url`, and the connection settings
  (`organization`, `project`, `headers`; below), the run layer takes the
  same provider blocks as a flow file (see
  [Provider blocks](syntax.md#provider-blocks-request-settings)): one block per
  provider, named after it, holding request-body fields of the API that provider's
  calls reach. They join the cascade per block field. Write a dashed block name as a
  dict key:

  ```python
  llm_config={
      "provider": "qwen-cn",
      "model": "qwen-plus",
      "qwen-cn": {"enable_thinking": True},
  }
  ```

  A retired key (`temperature`, `anthropic_effort`, `openai_reasoning_effort`,
  `google_thinking_level`) raises `ValueError` showing its block form.

  This layer is literal: `${...}` in it is text. A flow file's `llm_config` may
  compute fields from `${...}` values (see
  [Values in `llm_config`](syntax.md#values-in-llm_config)). `run_flow` coerces the
  supplied `inputs` before it resolves the cascade. An agent in the root flow whose
  values read only the run's inputs is resolved then, omitted inputs taking their
  defaults, and gets every check a literal config gets. Each input those values read
  must be given (or have a default) and match its declared type; otherwise
  `run_flow` raises `ValueError` naming the field before the log is created
  (`node 'chat' llm_config.model: input 'm' is required and was not given`). A bad
  input that no `llm_config` reads stays a START failure in the log. Every other
  agent with a value resolves its config just before its model call, from the
  defaults and saved logins captured at run start; a config refused then fails that
  node with `contract.input`. Such an agent gets the host key only when it resolves
  to the key's provider, as a static agent does.

  Each layer is checked as a `yian.llm_clients.LLMConfig`. Its blocks are fields
  named like the keys, except that the dashed ones are the attributes `qwen_cn`,
  `glm_cn`, and `minimax_cn`; build such a config from a dict, as in
  `LLMConfig.model_validate({"qwen-cn": {"enable_thinking": True}})`, since the
  Python spelling is not accepted as a key. `config.block_for("qwen-cn")` returns
  the block as a plain dict without its unset fields (`{}` when it is unset).

  A `base_url` entry replaces the provider's default endpoint for the agents that
  use the key, for example an Azure OpenAI resource's v1 surface:

  ```python
  llm_config={
      "provider": "openai",
      "model": "gpt-5.4",  # the Azure deployment name
      "base_url": "https://<resource>.openai.azure.com/openai/v1/",
      "api_key": key,
  }
  ```

  A `base_url` decides where the key is sent, so it is accepted only beside
  `api_key` (and `provider`) in this run layer. Flow files cannot set it at any
  level, and a flow built in Python fails at run start if any layer has one, so
  an edited flow cannot send the host key elsewhere (an Azure endpoint, below, is
  the exception). That guarantee covers authored
  layers only. When the host passes `api_key` without `base_url`, the provider's
  default endpoint or its environment variables (such as `OPENAI_BASE_URL`,
  `VLLM_BASE_URL`, or `ANTHROPIC_BASE_URL`) pick where the
  key goes, as before; some clients read them on every call, so a resume follows
  the environment at that time. A host that wants a fixed endpoint should pass
  `base_url` with the key. It must be a lowercase `https://`
  URL without credentials, query, or fragment; `http://` is allowed only for exactly
  `localhost`, `127.0.0.1`, and `[::1]`. It is checked and used exactly as written:
  whitespace, control characters, non-ASCII characters (write an international
  host in its `xn--` form), a port outside 1–65535, and URLs over 2048 characters
  (`YIAN_MAX_BASE_URL_CHARS`) are rejected, not normalised. The OpenAI-compatible
  providers, `anthropic`,
  and `google` take it; `ollama` rejects it. For `provider: azure` it is the
  required resource endpoint: a flow file or this layer may set it beside
  `provider: azure` without a key, and an Azure `api_key` here needs one beside it,
  so a host key never goes to an endpoint a flow names. A redirect from it does not
  carry the key on: the `anthropic` and `google` clients do not follow redirects
  from a `base_url`, the `azure` client drops its `api-key` header on a redirect to
  another origin, and the OpenAI-compatible clients drop their `Authorization`
  header on one. It is not secret: the log records it on the agents marked for the
  key, and an `openai` agent's host connection uses it as its endpoint. `resume_flow` and
  `result.backward` send the re-supplied key to that recorded URL; neither the
  provider's endpoint variable nor `OPENAI_BASE_URL` changes it. `ProfileMemory`
  takes one beside its own key; `ReflectiveMemory` rejects one except an Azure
  endpoint beside `provider: azure`. Both memories take an Azure endpoint without a
  key. Changed (TS yian): no request follows a redirect, whatever the provider; a
  3xx reply fails the call with a `redirect` provider error that names only the
  target's scheme and host. Models are built by `chatModelFromConfig` in
  `llm_clients/factory.ts`, with no provider SDK: a `google` model id may carry
  Google's `models/` prefix or name a `tunedModels/` model, and an `ollama` config
  `api_key` is accepted and never sent.

  The connection settings go wherever the key goes, so each is accepted only beside
  `api_key` and `provider` in this run layer, and flow files cannot set them at any
  level (see [Connection settings](installation.md#connection-settings) for the
  rules and limits):

  ```python
  llm_config={
      "provider": "openai",
      "model": "gpt-5.5",
      "api_key": key,
      "organization": "org-...",  # sent as OpenAI-Organization
      "project": "proj_...",      # sent as OpenAI-Project
      "headers": {"X-Gateway-Token": token},
  }
  ```

  `organization` and `project` are OpenAI ids, so only `provider: openai` takes
  them. `headers` are extra request headers, such as a gateway token, for every
  provider but `ollama`; a name yian sets itself (`Authorization`, `x-api-key`,
  `Host`, `Content-Type`, ..., and for `openai` also `OpenAI-Organization` and
  `OpenAI-Project`) is refused. The agents marked for the key send them with every
  request, and a model that sends them never follows a redirect. Like `base_url`,
  `organization` and `project` are recorded on those agents. Header values are
  secrets, handled like the key: the log records only the sorted header names (as
  `_header_names`), and a value a provider echoes in an error, however short, is
  replaced with `[REDACTED]`. A saved login brings the settings `yian auth` saved
  with it instead, and an environment key brings none; a host key never picks up a
  saved login's settings. A key that marks no agent drops its settings too.
  `ProfileMemory` takes them beside its own key; `ReflectiveMemory` rejects them.

  Pass the header values again with the key: `resume_flow(..., api_key=key,
  headers={...})` and `result.backward(..., api_key=key, headers={...})`. Their
  names must be the recorded ones, compared without case, and an error names the
  missing or unrecorded header names, never a value. The two calls fail
  differently when the headers are missing. `resume_flow(..., api_key=key)` on a
  run that recorded header names raises `ValueError` before it appends anything,
  so the log stays resumable. `result.backward(..., api_key=key)` without them
  starts its attempt, and the agents marked for the key fail their model call: the
  outcome is failed with an `AuthError` cause naming the header names. A result
  from `run_flow` or `resume_flow` keeps the headers it was given for its own
  `backward` calls, as it keeps the key.

  Code that builds a model itself passes the same headers as
  `yian.llm_clients.create_llm_client(..., connection_headers={...})`, a mapping of
  name to `str` or `SecretStr`. The OpenAI-compatible, `azure`, `anthropic`, and
  `google` clients take it and keep the values out of the model's fields (its
  `repr`, `model_dump()`, and `dumpd()`). For `openai`, `OpenAI-Organization` and
  `OpenAI-Project` there carry the organization and project ids. The names and
  values follow the `headers` rules, and an `http_client` or `http_async_client`
  passed beside them must not follow redirects. The `ollama` client and the
  ChatGPT subscription model refuse it, since they send no API key.

  Changed (TS yian): a saved subscription's account id is scrubbed from errors too,
  and `ANTHROPIC_AUTH_TOKEN` isn't read. Scrubbing rewrites an error's `message`, its
  `stack` and every `cause`, and the members of an `AggregateError`. [Installation](installation.md#ts-environment)
  lists the variables TS yian reads, its base URL variables included; an empty
  variable counts as unset.
- `memory` — one backend per declared memory key, normally the
  `MemoryRegistry` from `build_memories(flow)`. `None` leaves all declarations
  dormant. A missing key, or one object under two keys, fails setup; extra keys
  are ignored. A `MemoryRegistry` built for another flow, with a different type
  for some key, fails setup with a `MemorySetupError`. `run_flow` never
  constructs memory. A mapping opens one run-scoped session per memory key,
  enables recall and optional conversation history for AGENT nodes, and records
  exact opening snapshots for recovery. Those snapshots stay pinned;
  SessionMemory reads latest live history after turn admission.
- `learnable` — retain accepted inputs, outputs, routes, and invocation context for
  a later explicit `result.backward(feedback)` call. It defaults to `False` and
  needs no memory. Forward execution never calls backward hooks, memory
  learners, or memory commits.
- `tools` — the tools this run sees, keyed by tool id. TOOL nodes, AGENT
  `tools:` grants, and HUMAN_INPUT look ids up here. `None` uses the tools
  registered with `register_tool`. A mapping, even an empty one, is the whole
  set: registered tools are hidden from the run. To keep them, pass
  `{**registered_tools(), "my_tool": fn}`. `registered_tools()` returns a copy
  taken when you call it, so register every tool before the run starts. A
  synchronous function or method becomes a tool named by its key and described
  by its docstring (see [Tool context](#tool-context)). To give tools an app
  object such as a session, pass it as `context`, or use a closure or a bound
  method; `functools.partial` and other callable objects are rejected. A
  `FunctionTool` or LangChain `BaseTool` must already use its key as its name.
  The `ask_user` entry is the run's human handler (see
  [Blocking human interaction](#blocking-human-interaction)). Tools are host
  callables, so the run log does not record them. An AGENT that grants skills
  also gets the built-in `load_skill` tool, which is not part of this mapping.
  A list or tuple value is a tool group (see [Tool groups](#tool-groups)).
  An `McpServer` from `mcp_server` is a group of the tools that server lists. One
  that isn't open belongs to the run, which opens it before the tool checks and
  closes it when `run_flow` returns or raises; one opened with `with server:`
  belongs to the host and stays open (see [MCP servers](#mcp-servers)). Opening
  raises `McpConnectError` when a server doesn't start or doesn't list its tools,
  and `ImportError` when the MCP SDK isn't installed.
  Every static AGENT grant must name a tool or group in this set, and every
  static TOOL node's `tool_id:` must name a tool in it; otherwise `run_flow`
  raises `ValueError` before the log is created. Every static TOOL node is also
  checked against its tool's argument and return types (see
  [Typed tools](#typed-tools)).
- `skills` — the skills this run may use, keyed by name (see [Skills](#skills)).
  An AGENT's `skills:` grant selects from this mapping. A listed skill that is
  missing raises `ValueError` before the run starts.
- `context` — the host's app object, such as a database session or a broker
  client. A tool whose first parameter is annotated `ToolContext` reads it as
  `ctx.context` (see [Tool context](#tool-context)). It is never recorded and is
  not part of the flow fingerprint, so pass it again to `resume_flow`.

Changed (TS yian): `runFlow(flow, inputs, { log, runId, numWorkers, onEvent, onStream,
llmConfig, memory, learnable, tools, skills, steering, context, signal, env })` returns a Promise
of the `RunResult`; a failed flow resolves too, and the errors above are its rejections. `memory`
and `skills` are `Map`s and `tools` a plain record. `signal` is the host's stop: aborting it stops
the run, which rejects with the signal's reason, and an already-aborted signal rejects before
anything is created. `env` (default `processEnv`) is the environment the run reads connections,
credentials and its `YIAN_*` values from (see [Defaults and limits](#defaults-and-limits)). Three
options are for tests: `scheduler` (the clock the system clock is seeded from and node waits run
on), `random` (the draw in [0, 1) for backoff jitter) and `models` (the model builder AGENT nodes
use). `inputs` and `llmConfig` are taken through host intake (see [Types → host
intake](typing.md#two-constructors-infer-vs-check)): a whole `number` within ±(2^53−1) arrives as
an int (`{n: 3}` holds `3n`) and any other number as a float, so `Infinity` is accepted where a
float is; NaN, an accessor property, a Proxy inside a value and a non-yian object (a `Date`, a
`Map`) are refused with `TypeCheckError`, and an `inputs` record that is a Proxy, or an `llmConfig`
that isn't a record, with `TypeError`, before the log is created.

### Tool context

A host tool is a synchronous function or bound method. Its type hints give the
argument schema, and its docstring gives the description the model sees and the
argument descriptions. A tool that needs to know which call it is serving, or
needs the host's app object, declares a first parameter annotated
`ToolContext[T]`:

```python
from yian import ToolContext, run_flow
from yian.run import resume_flow


def place_order(ctx: ToolContext[Broker], symbol: str, qty: int) -> str:
    """Place a market order.

    Args:
        symbol: The ticker to buy.
        qty: How many shares.
    """
    return ctx.context.submit(symbol, qty, client_order_id=ctx.call_id)


tools = {"place_order": place_order}
result = run_flow(flow, {}, tools=tools, context=broker, log="run.jsonl")

# After a crash, in a later process, pass the same tools and the app object again:
recovered = resume_flow(flow, log="run.jsonl", tools=tools, context=broker)
```

Here `flow` has a TOOL node with `tool_id: place_order`, and `broker` is the
host's client, which treats a repeated `client_order_id` as the same order. If
the process dies after the broker accepts the order but before the node's result
is recorded, the resumed run re-runs the node with the same `ctx.call_id`, and
the broker returns the first order instead of placing a second one.

The context parameter is found by its type. It must be the first parameter,
positional, and annotated `ToolContext`, `ToolContext[T]`, or a subclass; its
name doesn't matter. It is hidden from the schema the model sees and from the
TOOL node's input names. `ToolContext` anywhere else, such as a later parameter,
a keyword-only one, or `Optional[ToolContext]`, raises `TypeError` when the tool
is built: in `run_flow` or `resume_flow` before the log is created or opened for
a `tools=` entry, and at decoration for `register_tool`. An `async def` tool raises
`TypeError` the same way, since yian calls tools synchronously.

| Field | Meaning |
| --- | --- |
| `call_id` | an opaque string that names this call (rules below) |
| `attempt` | the attempt number, starting at 1 (rules below) |
| `tool_name` | the id the tool is registered under |
| `run_id` | the run's id, or `""` outside a run |
| `node` | the expanded id of the calling node, such as `loop#3/place`, or `""` when the caller names none |
| `context` | the object passed as `context=`, or `None` |

The call id depends on who calls the tool:

- A TOOL node's call id is `"<run_id>/<node>"`, such as `run-7f3a/loop#3/place`.
  Each LOOP iteration and MAP item has its own expanded node id, so its own call
  id. A resumed run re-runs an interrupted TOOL node under the same call id.
- A HUMAN_INPUT node's `ask_user` call uses the same form.
- A call from an AGENT's tool loop gets an id of its own, unique per call and
  never reused for another call. It is not stable across resume: a resumed AGENT
  asks its model again and may make different calls, so reusing an id could make
  an idempotency key drop the wrong action. Call a tool whose side effect must
  happen once from a TOOL node.
- Outside a run, as with `invoke_tool` or a `ToolSet` not bound to a run, each
  call gets a fresh random id.

Treat `call_id` as an opaque string and don't parse it; the AGENT form changes
once AGENT steps are recorded. It is unique across runs only when run ids are: two
runs given the same `run_id=` give their TOOL nodes the same call ids. When a
broker limits the length or characters of its key, hash the id, for example
`hashlib.sha256(ctx.call_id.encode()).hexdigest()[:32]`.

`attempt` counts a TOOL node's attempts at the same call: 1 for the first, 2 for
the first retry, and so on (see [TOOL retries](#tool-retries)). A resumed run
that re-runs an interrupted attempt passes the next number, with the same call id.
Every other call, including one from an AGENT's tool loop, sees 1.

`context` is the same object for every tool call in the run. With the default
`num_workers=8`, nodes call tools at the same time from worker threads, so the
object must be thread-safe. The run log, events, results, and fingerprint never
hold it. Without `context=` on `resume_flow`, tools in the resumed segment see
`ctx.context is None`.

The `ask_user` handler may take `ctx` first as well:
`def handler(ctx: ToolContext[App], **request)` (see
[Blocking human interaction](#blocking-human-interaction)).

Host functions and `FunctionTool`s run directly, not through LangChain's
`invoke`, so LangChain callbacks and tracing don't fire for them. A LangChain
`BaseTool` runs as before and receives no `ToolContext`.

`ToolSet.for_run(run_id, stop_event=event)` returns a copy of a tool set bound to
one run: `run_id` becomes `ToolContext.run_id` and the base of the call ids, and
`event` is visible to each call the copy makes (`ToolSet.stop_event` returns it).
The run binds its own stop event, which is set when the run stops (Ctrl-C, a
failing sibling node, or a host stop). MCP tools watch it, so a stop cancels a
waiting call (see [MCP servers](#mcp-servers)). The copy holds exactly the event
given: `None` gives no event, even when the set was bound to one before.

**Descriptions.** The tool's description is the docstring's first text section,
everything before its first section header. Any `Word:` (or `Some words:`) line
followed by an indented block starts a section, not only `Args:`, so the text of
a `Steps:` or `Sources:` block, and everything after the first section, is not
part of the description; only `Args:`-style sections fill argument descriptions.
With no text before the first section, the description is the tool id.
Argument descriptions come from a Google `Args:` section (also `Arguments:`,
`Params:`, or `Parameters:`), a NumPy `Parameters` section, or Sphinx
`:param name:` fields; the style is detected from the docstring.
`Annotated[T, "description"]` describes an argument the docstring doesn't.
`Field(...)` defaults and constraints are kept, and a docstring or `Annotated`
description replaces a `Field` description.

**`FunctionTool`.** A tool without a Python signature, such as one generated
from an external schema, is a `FunctionTool`:

```python
from typing import Any

from yian import FunctionTool, ToolContext, run_flow


def echo(ctx: ToolContext[Any], args: dict[str, Any]) -> str:
    return args["text"]


echo_tool = FunctionTool(
    name="echo",
    description="Echo the text.",
    params_json_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    on_invoke=echo,
)
result = run_flow(flow, {}, tools={"echo": echo_tool})
```

`on_invoke` is synchronous and always receives the call's `ToolContext` and a
deep copy of the checked arguments as a plain `dict`, so changing it doesn't
change the caller's values. Before calling it, yian rejects unknown keys (unless
`additionalProperties` is `true` or a schema, which then checks the extra values),
rejects missing required keys, and checks each value against the yian type its
JSON Schema translates to (see [Typed tools](#typed-tools)); a property with no
translation is passed unchecked. An integral float such as `3.0` passes where an
integer is expected, also inside lists and records, and reaches `on_invoke` as
given. Building one raises
`TypeError` when `name` or `description` is not a string, `params_json_schema` is
not a dict, or `on_invoke` is not callable or is async, and `ValueError` when
`name` is empty or the schema's `type` is not `"object"`. An `on_invoke` that
returns an awaitable fails the call with `ToolCheckError` (a `TypeError`), which a
TOOL node doesn't retry.

`function_tool(fn, name_override=..., description_override=...)` builds a
`FunctionTool` from a function, the same way a plain function tool is built; it
also works as `@function_tool` or `@function_tool(name_override="buy")`.
`register_tool(id, description_override=...)` replaces the description of the
function or `FunctionTool` it registers. A `FunctionTool`'s name must equal its
`tools=` key or `register_tool` id.

Changed (TS yian): every tool is a `FunctionTool`, built with
`new FunctionTool({ name, description, parametersJsonSchema, execute, isConcurrencySafe })`;
a plain function is not a tool, and `registerTool(id, tool, { descriptionOverride })`
takes a `FunctionTool`. `execute(args, ctx)` gets the arguments first and the context
second, and may be sync or async. `ToolSet`'s `invoke`, `invokeTrusted` and `validate`,
and the functions `invokeTool` and `invokeToolTrusted`, return Promises.
`validate(toolId, args)` checks a call and returns a `CheckedCall`, which `invoke(call)`
runs without checking the arguments again and `isConcurrencySafe(call)` reads; `invoke(toolId, args)`
validates first. `ctx`'s fields are camelCase (`callId`, `toolName`, `runId`), `context`
is `undefined` when none is given, and `signal` is an `AbortSignal` aborted when the call
must end early (the run stops, or a sibling call fails the node); `forRun(runId, { signal })`
binds it. Tool errors, `FunctionTool` and `McpServer` are recognized by brand, not
`instanceof`, so ones built against another copy of yian (an npm install beside a
workspace one) are accepted and classified.

Arguments enter through host value intake, so a non-yian value (`undefined`, `NaN`, a
`Date`, an accessor property (getter or setter) or a `Proxy`) anywhere in them is refused,
and reach the tool as fresh host values: an int within ±(2^53−1) arrives as a `number`, a
larger one as a `bigint`, and an infinite float as a real `Infinity`, which a tool may also
return. `isConcurrencySafe(args)` is opt-in, so a tool without it is unsafe. It receives
what `execute` receives, and a call counts as safe only when it returns exactly
`true`; a throw or any other result counts as unsafe.

Changed (TS yian): an AGENT runs one reply's tool calls in groups, where Python runs them one at
a time. Consecutive safe calls run together; an unsafe call waits for them to finish, then runs
alone. `load_skill` is safe. `ask_user` isn't, and neither is an MCP tool its server's
`concurrencySafe` doesn't mark. The calls are checked (grant, then arguments) in call order, and
no call of a group starts until the group is formed. A call whose check fails fails the node
after the calls before it ran, as in Python. The model reads the results in call order, whatever
order the calls finish in, but the `ToolCallFinished` events of a group arrive in finish order.
The first call that fails aborts its siblings' `ctx.signal`, and the node fails once all of them
have settled; a `ReportedToolError` is a result, not a failure. On a stop, a call that already
finished keeps its `ToolCallFinished`, and a call not yet started never starts and gets no
events.

Changed (TS yian): in place of `function_tool`,
`tool({ name, description, parameters, output, execute, isConcurrencySafe })` builds a
`FunctionTool`. `parameters` and `output` each take a Standard Schema v1 validator with the
Standard JSON Schema interface (zod 4, ArkType; valibot through `toStandardJsonSchema`), a
validator with `parametersJsonSchema` (`outputJsonSchema`) beside it, or a raw JSON Schema.
yian imports no validator library; it reads only those interfaces.

```ts
const placeOrder = tool({
  name: "place_order",
  description: "Place a market order.",
  parameters: z.object({ symbol: z.string(), qty: z.number().int().default(1) }),
  execute: async ({ symbol, qty }, ctx) =>
    ctx.context.submit(symbol, qty, { clientOrderId: ctx.callId }),
});
```

`validate` checks the arguments against the JSON Schema first, then runs the validator once;
its output (here with `qty` filled in) is what `execute` gets on every attempt and what
`isConcurrencySafe` gets, the same object each time, so don't mutate it if you rely on
retries. The validator's issues raise `ToolCheckError`
(`tool 'place_order' arguments failed its validator: <path>: <message>; ...`).

For a tool that declares an output, `invoke` checks the result: host value intake, yian's
check against `outputJsonSchema`, which looks inside an object only when its schema sets
`additionalProperties: false`, then, when there is an output validator, the validator, which
checks the rest, and intake of its output. It returns the checked value in host form. A
failure raises `ToolCheckError` (`tool 'x' result at <path>: ...`, `tool 'x' result failed
its output validator: ...`), and so does an output validator that throws (`tool 'x' result:
the output validator threw: <message>`, the error as its `cause`) or returns no Standard
Schema result, since `execute` has already run. A tool without an output returns its raw
result. `execute`'s return type is the output validator's input form, so declare a
transforming output as a raw JSON Schema of what is recorded, or leave `output` out. An
`output` with no JSON Schema (none from the validator and no `outputJsonSchema`) makes
`tool()` throw `TypeError`; Python left such a return type unchecked.

### TOOL retries

A TOOL node with `retries: N` calls its tool again after a failed attempt, up to N
more times. The default is 0 (`YIAN_TOOL_RETRIES`), because a retry repeats any side
effect the failed attempt already had. A tool with side effects should use
`ctx.call_id` as its idempotency key: it is the same for every attempt, while
`ctx.attempt` counts them.

Every exception from the tool is retried, except:

- `PermanentToolError`, which the tool raises when another attempt can't help. A
  class that subclasses both `PermanentToolError` and `TransientToolError` counts
  as permanent;
- yian's own checks. The arguments are checked once, before the first attempt, with
  `ToolSet.validate`. The checks on the returned value (the `output:` contract, no
  NaN or infinity, the nesting limit) run after the node returns. A `FunctionTool`
  whose `on_invoke` returns an awaitable raises `ToolCheckError`. None of these is
  retried;
- a host stop, such as Ctrl-C or a sibling node's failure, which ends the node.

Changed (TS yian): the check on the returned value refuses NaN but not `Infinity`:
a host tool may return `Infinity`, which is a float value. A `tool()` validator's issues
and the checks on a declared result (see [Tool context](#tool-context)), a throw from the
output validator included, raise `ToolCheckError`, which is never retried; the result
checks run inside the call, not after the node returns. `ToolAttemptsFailed` is an
`Error` named `ToolAttemptsFailed` whose `cause` is the last attempt's error. A tool
that throws a non-`Error` value fails with an `Error` whose message is the value's text
and whose `cause` is the value; it is retried like any other failure.

A `ReportedToolError` is retried like any other failure, unless it is also a
`PermanentToolError`.

Before the retry after the k-th failure, the node waits d = min(cap, base · 2^(k−1))
seconds, drawn uniformly between d/2 and d. The base is
`YIAN_TOOL_RETRY_DELAY_SECONDS` (1) and the cap is `YIAN_TOOL_RETRY_MAX_DELAY_SECONDS`
(30), so the default waits are at most 1, 2, 4, 8, 16, 30, 30, … seconds. A tool that
raises `TransientToolError("rate limited", retry_after=5)` sets that retry's wait to
`retry_after`, up to the cap, with no jitter. `retry_after` must be a finite number
of seconds, at least 0; anything else raises `TypeError` or `ValueError` when the
error is built.

```python
from yian import PermanentToolError, ToolContext, TransientToolError


def place_order(ctx: ToolContext[Broker], symbol: str, qty: int) -> str:
    """Place a market order."""
    response = ctx.context.submit(symbol, qty, client_order_id=ctx.call_id)
    if response.status == 429:
        raise TransientToolError("rate limited", retry_after=response.retry_after)
    if response.status == 400:
        raise PermanentToolError(response.reason)
    return response.order_id
```

A waiting node keeps its worker: it occupies one pooled worker slot, or blocks a
serial run (`num_workers=0`), until the wait ends. A run stop ends the wait early.

Each retry is recorded in the run log as node steps (`NodeStepRecord`): a `failed`
step with the error's type and message and the chosen wait (`retry_in`), written
before the wait, and a `started` step written before the retry calls the tool. The
first attempt is the node's `NodeStartedRecord`, so a call that succeeds at once, or
any call of a `retries: 0` node, writes no step.

A resumed run continues the count from the log. An attempt that was interrupted (the
process crashed or the run stopped inside it) is re-run with the same `ctx.call_id`
and the next `ctx.attempt`, and uses up no retry: only recorded failures count
against `retries:`. A run that crashed or stopped during a wait waits the recorded
`retry_in` again before the next attempt.

When the call fails without a retry, the tool's exception fails the node.
After at least one retry, the node fails with `ToolAttemptsFailed` (a
`RuntimeError`, raised from the last error), and `failure.message` lists the
attempts' errors:

```text
tool 'place_order' failed after 3 attempts: attempt 1: TimeoutError: read timed out; attempt 2: interrupted; attempt 3: ConnectionError: reset by peer
```

Every attempt is listed, and an attempt with no recorded outcome shows as
`interrupted`. `failure.code` stays `node.execute`.

Changed (TS yian): an attempt with no recorded outcome shows as `unfinished` (`attempt 2: unfinished`).

An AGENT's `retries:` means something else: it bounds how often the model may
correct a structured reply that doesn't fit `output:` (see
[structured generation](syntax.md#agent-the-llm-backed-agent-node)). Tool calls
from an AGENT's tool loop are never retried, and see `ctx.attempt == 1`.

In an AGENT's tool loop, a tool that raises `ReportedToolError` doesn't fail the
node: the model reads `str(exc)` as that call's result (the exception's type name
when the text is empty) and the loop continues, so the model can fix its arguments
or try another tool. A stop pending when the tool raises still ends the node. Any
other exception from the tool fails the node. In a TOOL node a `ReportedToolError`
is an ordinary failure, as above. The message goes to the model provider and to
the session transcript, so it must not hold secrets.

```python
from pathlib import Path

from yian import ReportedToolError


def read_file(path: str) -> str:
    """Read a text file in the workspace."""
    file = Path(path)
    if not file.is_file():
        raise ReportedToolError(f"no such file: {path}")
    return file.read_text()
```

Changed (TS yian): yian classifies tool errors by their `TOOL_ERROR` brand, never by
`instanceof`, so an error thrown by a tool built against another copy of yian still
counts; `toolErrorKinds(error)` and `isPermanentToolError`, `isTransientToolError`,
`isReportedToolError` and `isToolCheckError` read it. With no multiple inheritance, a
reported error that also stops retries is `new ReportedToolError(message, { permanent:
true })`, and the wait is `new TransientToolError("rate limited", { retryAfter: 5 })`;
a bad `retryAfter` throws `TypeError` or `RangeError`. `execute` may return a Promise,
which is awaited, so no awaitable raises `ToolCheckError`.

Changed (TS yian): the model reads a tool's result in the text form, not Python's `str()`: text
as it is, `null` for `null`, and a record or list in JSON style (`{"a":1}`; see
[Expressions → the `${...}` scanner](expressions.md#layer-3-the-scanner)).
`ToolCallFinished.result` in `on_event` holds the same text. A result that isn't a yian value
(a `Date`, a class instance; see [Types → host intake](typing.md#two-constructors-infer-vs-check))
fails the node, and its `ToolCallFinished.result` is `null`. A `ReportedToolError` that a tool's
argument validator throws is read by the model like one the tool raised, as in Python, which
checks the arguments inside the call.

### Tool groups

A host can fill a named group of tools per run. The flow grants the group by
name, next to plain tool ids:

```yaml
nodes:
  worker:
    kind: agent
    prompt: ${input.task}
    tools: [toolbox, workspace, company, ask_user]
```

```python
result = run_flow(
    flow,
    {"task": "Summarize the open tickets"},
    tools={
        "toolbox": [search_tickets, read_ticket],  # functions, named by __name__
        "workspace": [read_file_tool],             # BaseTool, named by .name
        "company": [],                             # an empty group grants nothing
        "ask_user": ask,
    },
)
```

`worker` sees `search_tickets`, `read_ticket`, `read_file_tool`, and `ask_user`.
Each group item is a LangChain `BaseTool` or `FunctionTool` named by its `name`,
or a function or bound method named by its `__name__` (a lambda has no usable
name and is rejected). An `McpServer` value is a group of the server's tools (see
[MCP servers](#mcp-servers)). Groups expand before the grant check, so the model calls the tools by
their own names, never by the group name. A group is not a tool: a TOOL node's
`tool_id:` may name a tool inside a group, but not the group itself. Before the
log is created, `run_flow` raises:

- `ValueError` when an AGENT grants a group (or tool) the mapping lacks;
- `ValueError` when a TOOL node's `tool_id:` names a group
  (`TOOL node 'place' names 'toolbox': 'toolbox' is a tool group, not a tool; ...`);
- `ValueError` when two different tools share a name, in one group or across
  groups, or when a group tool's name is also a top-level key, since tools are
  called by name;
- `ValueError` when a group tool is named `ask_user` or `load_skill`. The
  `ask_user` handler stays the top-level `tools["ask_user"]` entry, and an AGENT
  that grants `ask_user` next to groups gets that handler. `load_skill` is bound
  by granting skills.

Groups may overlap: the same tool object (or the same bound method) can sit in
several groups, as when one permission set shares `web_search` with another. An
AGENT granted overlapping groups, or the same group twice, binds each tool once.

These checks cover the grants of every static AGENT node and the `tool_id:` of
every static TOOL node, including nodes that may never run: an unreached `case`
branch, the body of a MAP over an empty list, or a LOOP body. Because the YAML does
not mark which names are groups, every grant must resolve, not only group names: an
unknown plain tool id, or `ask_user` granted with no handler, fails the same way. A
TOOL node naming a tool the run lacks fails with
`TOOL node 'place' names tool 'place_ordr', which isn't in this run's tools; known tools: ['place_order']; did you mean 'place_order'?`. `run_flow` raises the `ValueError` before the log exists, and `resume_flow`
raises it for an unfinished run before appending any record; neither returns a
failed `RunResult`. (Before tool groups, an unresolvable grant failed only when
an AGENT that ran the tool loop started, as a recorded node failure.)

Groups are host capabilities, like every other entry in `tools`: the log records
neither their contents nor their tools. Only the group names in the flow are part
of its fingerprint. Each run gets its own groups, so concurrent runs never see each
other's tools, and `resume_flow` uses the groups passed to it (see
[Durable recovery](#durable-recovery-with-resume_flow)).

Changed (TS yian): each item of a group array is a `FunctionTool`, named by its `name`;
anything else throws `TypeError`. Groups follow the mapping's JS key order, so
integer-like keys come first.

### MCP servers

A Model Context Protocol (MCP) server offers tools over a standard protocol, and
can be written in any language. A host passes a server in `tools=`, and its tools
become ordinary yian tools in one group named by the server's key. The MCP SDK is
an optional extra:

```console
python -m pip install "yian[mcp]"
```

Changed (TS yian): the SDK is an optional peer dependency of the npm package, loaded only
when a server opens:

```console
npm install @modelcontextprotocol/client@2.3.1
```

Describe a server with `mcp_server` from `yian.tools`. A stdio server is a command
yian starts as a subprocess; an HTTP server is a streamable HTTP endpoint:

```python
from yian.tools import mcp_server

files = mcp_server(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    env={"LOG_LEVEL": "warn"},
    cwd="/srv/project",
)
github = mcp_server(
    url="https://mcp.example.com/github",
    headers={"Authorization": f"Bearer {token}"},
)
```

Pass exactly one of `command` and `url`. `args`, `env`, and `cwd` go with
`command`, and `headers` with `url`; `url` must be `http` or `https`. A stdio
server gets the MCP SDK's safe default environment (`PATH`, `HOME`, and a few
others) with `env` merged over it, so the host's secrets reach a server only when
passed in `env`. `timeout` and `connect_timeout` set the call and connect timeouts
(below) for this server. `label` names the server in errors and is the command's
basename, or the URL's host, by default. A bad argument raises `TypeError` or
`ValueError` at once; nothing starts until the server opens.

Changed (TS yian): `mcpServer({ command, args, env, cwd, url, headers, timeout,
connectTimeout, label, concurrencySafe })` takes camelCase options; a bad one throws
`TypeError` or `RangeError`. `concurrencySafe` (`true`, or an array of tool names) marks
the server's tools, or the named ones, safe to run beside other calls (default `false`;
see [Tool context](#tool-context)). The MCP errors carry the tool error kinds of the
table below as brands, and `isMcpError` recognizes one from any copy of yian. `url`
must start with `http://` or `https://`. A URL server's default label is the host as
WHATWG `new URL` reads it, so an IPv6 host keeps its brackets and an international
host reads as punycode.

Credentials stay with the host: the flow names the server only by its key. An
AGENT grants the server's key, which grants every tool the server lists, or single
tools by name. A TOOL node's `tool_id:` names one tool, never the server:

```yaml
nodes:
  triage:
    kind: agent
    prompt: ${input.task}
    tools: [github]
  lookup:
    kind: tool
    tool_id: search_issues
    input:
      query: ${input.query}
```

```python
result = run_flow(flow, inputs, tools={"github": github})
```

The [Tool groups](#tool-groups) rules apply. A server's tool may not share its name
with a different tool in another group (another server's, for example) or with a
top-level key, or be named `ask_user` or `load_skill`; the run raises `ValueError`
before it starts. A server inside a list group raises `TypeError`; pass it as its
own entry.

**Lifetime.** A server passed unopened belongs to the run. `run_flow` opens it
after the checks that need no tools (the other `tools=` entries, conditions,
templates, skills, and steering) and before the tool grant and type checks, and
closes it when `run_flow` returns or raises, including on Ctrl-C. When one server
fails to open, the ones already open are closed. The same server object under
several keys opens once. `resume_flow` opens a server only for an unfinished run,
after it replays the log; resuming a finished run opens nothing. Two runs can't
open one server object at once: while one run has it open, or is opening it,
another run given the same object raises `ValueError` before its log is created.

A server opened with `with server:` (or `server.open()`) belongs to the host,
which closes it. Runs use it as it is and leave it open, so several runs, even
concurrent ones, can share one process or connection:

```python
with mcp_server(url=url, headers=headers) as github:
    for task in tasks:
        run_flow(flow, {"task": task}, tools={"github": github})
```

Changed (TS yian): the host opens a server with `await server.open()`, or with
`await using github = await mcpServer({ url, headers }).open()`, which closes it when the
scope ends. `close()` returns a Promise that never rejects. `open({ signal })` takes an
`AbortSignal`; aborting it while the server starts closes the server and rejects with the
signal's reason. A stdio server runs in its own process group: `close()` ends its stdin,
waits 2 s, sends SIGTERM to the group, waits 2 s, then sends SIGKILL, so a wrapper server
(`npx`, `uvx`) leaves nothing behind.

A host-owned server yian knows is gone (its loop ended, or a call saw it closed)
raises `ValueError` at run start (`tools['github']: MCP server 'github' is closed`).
A stdio server whose process died quietly is found on the first call, which raises
`McpServerClosed`.

Changed (TS yian): the key is quoted in the text form, `tools["github"]`.

Opening starts the server, waits until it lists its tools, and keeps that list:
`server.tools` is a tuple of `FunctionTool`s in the server's order, and a later
change to the server's tool list is ignored until it is opened again. Each tool's
input schema is the server's, checked like any `FunctionTool`'s
`params_json_schema` (see [Tool context](#tool-context)), and the run-start checks
compare a TOOL node's inputs with it (see [Typed tools](#typed-tools)). An MCP tool
declares no yian return type, so only the node's `output:` checks what it returns.
`open()` raises `McpConnectError` when the server doesn't start, doesn't list its
tools within the connect timeout, lists a tool twice, lists a tool whose input
schema's `type` is not `"object"`, or lists more than 100 pages of tools. It raises
`ImportError` naming the `mcp` extra when the SDK isn't installed, and
`RuntimeError` when the server is already open.

Changed (TS yian): `server.tools` is a frozen array. A listed `outputSchema` gives no
return type, as in Python, but each call checks its result against it (see Results).
`open()` also throws `McpConnectError` when `concurrencySafe` names a tool the server
doesn't list, and a server without the tools capability opens with no tools. In place of
`ImportError` it throws an `Error` naming `@modelcontextprotocol/client`, with the import
error as its `cause`, and in place of `RuntimeError` an `Error` when the server is already
open or still opening.

`server.tools` holds plain `FunctionTool`s, so a host can wrap or rename them
before a run. Most model providers accept only tool names matching
`^[a-zA-Z0-9_-]{1,64}$`, so rename a tool whose name has a dot or a slash:

```python
import dataclasses

with github:
    renamed = [
        dataclasses.replace(tool, name=tool.name.replace(".", "_"))
        for tool in github.tools
    ]
    run_flow(flow, inputs, tools={"github": renamed})
```

A renamed tool still calls the server's tool by its original name.

**Results.** A call returns the result's structured content (`structuredContent`)
when the server sends it. A Python MCP server (`MCPServer`, formerly FastMCP) wraps
a tool's primitive return value as `{"result": ...}`, so such a tool returns a
record. Otherwise the call returns the result's content as one string: text blocks
and embedded text resources give their text; images, audio, and binary resources
give a one-line marker with the mime type and decoded size, such as
`[image image/png, 12.3 KB]`; a resource link gives `[resource <uri> <mime type>]`.
Blocks are joined by blank lines, and their bytes are dropped: media does not reach
the model yet.

Changed (TS yian): a stdio server's ints are read exactly: one within ±(2^53−1) arrives
as a `number`, and a larger one in the structured content as a `bigint`. A whole-number
float such as `1.0` arrives as the int `1`. An HTTP server's result is read by the SDK with
`JSON.parse`, so an int beyond ±(2^53−1) loses precision. A tool that lists an
`outputSchema` must return structured content, checked against the schema read as an input
type; an object's fields are checked only when its schema sets `additionalProperties: false`
(see [Typed tools](#typed-tools)). A result without structured content, or one that fails
the check, is an `McpToolError`.

Changed (TS yian): arguments holding `Infinity` anywhere, and for an HTTP server an int
beyond ±(2^53−1), can't be carried as JSON; the call is refused before anything is sent,
with a `ToolCheckError`, which is never retried.

**Idempotency.** Each `tools/call` request carries
`_meta: {"yian/call_id": ctx.call_id, "yian/attempt": ctx.attempt}`. A server with
side effects can use `yian/call_id` as its idempotency key, as a Python tool uses
`ctx.call_id` (see [Tool context](#tool-context)).

**Errors.** A failed call raises a subclass of `McpError` (from `yian.tools`). The
class decides what an AGENT and a TOOL node do with it:

| Class | Raised when | AGENT: the model reads it | TOOL: `retries:` apply |
| --- | --- | --- | --- |
| `McpToolError` | the server returned an error result (`isError`), or the call failed locally, for example structured content that fails the tool's output schema | yes | yes |
| `McpRequestError` | the server answered with a JSON-RPC error | yes | yes |
| `McpInvalidRequestError` | the JSON-RPC error code is -32700, -32600, -32601, or -32602 (a malformed or unknown request), or the server redirected | yes | no |
| `McpTimeoutError` | the call outlasted the call timeout | yes | yes |
| `McpTransportError` | an HTTP request to the server failed at the connection level | yes | yes |
| `McpServerClosed` | the server is gone: it was closed, its process or connection ended, or it forgot the HTTP session | no, the node fails | no |
| `McpCallCancelled` | the calling run stopped while the call waited | no, the node ends with the stop | no |

"The model reads it" means the class is a `ReportedToolError`; "no" under `retries:`
means it is a `PermanentToolError`. `McpTimeoutError` and `McpTransportError` are
also `TransientToolError`s. `McpInvalidRequestError` is an `McpRequestError`. After
`McpServerClosed`, later calls to that server fail at once; the host still closes a
server it opened. `McpConnectError` (a `RuntimeError`) is raised only when a server
opens, so the run doesn't start.

Changed (TS yian): Python yian refuses a redirect to another origin; TS yian follows no
redirect at all. An HTTP server has forgotten the session when it answers 404 while the
client holds a session id, or 400 "Server not initialized".

The messages are fixed text naming the server's label and the tool. Only an error
result's content and a JSON-RPC error's message and code come from the server; no
URL, header, or exception text reaches the model, since those may hold a token.
The label does reach the model: for an HTTP server built without `label=`, it is
the URL's host, so set `label=` when the host name shouldn't appear.
For a stdio server, `McpConnectError` ends with the last 20 lines the server wrote
to stderr; for an HTTP server it names only the error type and HTTP status.

Changed (TS yian): every error text is scrubbed of the server's `headers` values, the bare
token after a scheme word in one (`Bearer <token>`), and its `env` and URL query values,
each when it has 8 or more characters. An HTTP server's `McpConnectError` has the
underlying error as its `cause`.

**Timeouts.** A server must start and list its tools within
`YIAN_MCP_CONNECT_TIMEOUT_SECONDS` (30 s), and each call must finish within
`YIAN_MCP_CALL_TIMEOUT_SECONDS` (300 s). A server's `connect_timeout=` and
`timeout=` override them. A server takes the values from `yian.defaults.current()`
when it opens (see [Defaults and limits](#defaults-and-limits)).

Changed (TS yian): on Node, one HTTP response slower than 300 s can fail as
`McpTransportError`: that is the runtime's own header timeout.

**Stops.** A run's stop (Ctrl-C, a failing sibling node, or a host stop) cancels
that run's waiting calls; each raises `McpCallCancelled`. The cancel is per call,
so a server shared by several runs keeps serving the others. `close()` ends every
waiting call with `McpServerClosed`.

**HTTP.** A connection failure before the response starts (refused, reset, or a
DNS failure) fails only that request, with a retryable `McpTransportError`; the
server stays open and serves later calls once it is reachable again. A reset while
a JSON response body is being read ends the connection instead, and the server is
then gone (`McpServerClosed`). The proxy environment variables
(`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY`) apply.

Changed (TS yian): each request is its own exchange, so the server stays open when a
response breaks mid-way. A JSON response body that fails while it is read fails that call
as an `McpToolError`; a streamed (SSE) response that ends before its answer leaves the call
waiting until the call timeout (`McpTimeoutError`), unless the server supports resumption.
`retries:` apply to both. Proxies, CA files, and TLS checks are the runtime's settings (see
[Proxies and certificates](installation.md#proxies-and-certificates)).

**Protocol.** yian talks to servers that use the 2026-07-28 protocol and to older
ones that use the 2025 initialize handshake. It only lists and calls tools:
elicitation, sampling, prompts, and resources are not supported. A stdio server's
stderr goes to a temporary file, not the host's terminal.

Changed (TS yian): a stdio server always gets the 2025 `initialize` handshake; an HTTP
server negotiates the protocol version. Only the tail of a stdio server's stderr is kept,
in memory. `YIAN_MAX_MCP_MESSAGE_BYTES` (16 MiB) caps one message from a stdio server (not
an HTTP one); a longer one closes the server.

**Trust.** A server runs with whatever access the host gives it: yian starts the
command the host names, with no sandbox, and sends requests to the URL the host
names. Choose servers as you would any other executable. A host with a permission
layer can wrap `server.tools` like any other `FunctionTool`.

### Typed tools

A TOOL node calls one host tool directly. When a run starts, before any record is
written, every static TOOL node is checked against this run's tools, including
nodes that may never run. `run_flow`, and `resume_flow` for an unfinished run,
first raise `ValueError` when the node's `tool_id:` is not one of the run's tools
(`TOOL node 'place' names tool 'place_ordr', which isn't in this run's tools; known tools: ['place_order']; did you mean 'place_order'?`).
They then check the node against the tool it names, and raise `ValueError` naming
the node and the tool when an input name, a missing input, or an input or output
type can't match; the rules and messages are in
[Typed TOOL nodes](syntax.md#typed-tool-nodes).

Input types come from the tool's public argument schema. Each pydantic v2 field
is translated through its JSON Schema, so a tool whose schema is JSON Schema
follows the same rules:

| JSON Schema | yian type |
| --- | --- |
| `string`, `integer`, `number`, `boolean`, `null` | `str`, `int`, `float`, `bool`, `None` |
| `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | a range: `int(a..b)` / `float(a..b)`, or one-sided `int(a..)` / `int(..b)` (likewise `float`) when only one end is kept; an exclusive end becomes the nearest inclusive `int`, or the same value made inclusive for `float` (wider); a `float` end that isn't an exact double is dropped |
| string `enum` or `const` | `Literal(...)` |
| `format` (`date`, `uri`, …) | `str` |
| `array` | `list(...)` from `items`; an `array` without `items`, and tuples, give `list(Optional(Any))` |
| `object` with `properties` | input: `object`, unless `additionalProperties: false` makes it a record; output: a record, unless it allows extra keys |
| `anyOf`/`oneOf` with `null` | `Optional(...)` |
| `anyOf`/`oneOf` of records | `Union(...)` |
| `anyOf`/`oneOf` of `integer`s (or of `number`s) with different limits | one range covering all of them |
| other unions, or anything untranslatable | not checked |

An input with a pydantic function validator (`mode="before"`, `"wrap"`, or
`"plain"`), `SkipValidation`, or a custom type whose Python validation differs from
its JSON Schema is not checked. A pydantic v1 schema gets only
the name and required-input checks. A `FunctionTool`'s inputs are read from its
`params_json_schema`, and a TOOL node can invoke it. Any other `BaseTool` with a
raw JSON Schema `args_schema` (a dict) is checked, but a TOOL node can't invoke it
yet.

Changed (TS yian): inputs are read from a `FunctionTool`'s `parametersJsonSchema`, and
its return type from its `outputJsonSchema` (from `tool({ output })`), translated closed:
an object with `properties` is a record unless it allows extra keys. A schema number
reads as an int when it is a `bigint` or a safe integer (`1.0` and `1` are one JS
number) and as a float otherwise. The model sees the schema, so one that isn't JSON
(holding `Infinity` or a `Date`) or nests deeper than `YIAN_MAX_VALUE_NESTING` throws
`TypeError` when the tool is built; Python left a too-deep input unchecked. An integral
float within ±(2^53−1) passes for an `integer` as in Python, but one beyond it (`1e20`)
reads as a float and is refused. The run-start return type is read closed, but the runtime
check of a raw JSON Schema output looks inside an object only when it sets
`additionalProperties: false`; the TOOL node's `output:` contract checks the recorded value
at its boundary.

Changed: a limit on only one end (pydantic `Field(ge=0)`, JSON Schema `minimum`
alone) and an exclusive `number` limit used to be dropped. They are now kept, so a
binding that can never fit, such as a literal `-1` into a `Field(ge=0)` input, raises
`ValueError` at run start instead of failing inside the tool call. A union of
`integer`s (or of `number`s) with different limits, such as pydantic
`Union[conint(ge=0, le=3), conint(ge=5, le=9)]`, used to be unchecked; it is now checked
against one range covering all its members.

The return type is read from the annotation of the function behind the tool: a
function passed in `tools=` or in a group, one registered with `register_tool`,
a `FunctionTool` built with `function_tool`, or a `StructuredTool` built from a
function. A `FunctionTool` built directly has no return type, so its output is
not checked at run start. Return a `TypedDict`, `dict`,
`list`, or scalar. The output check is skipped when there is no annotation, the
annotation is `Any`, the tool is a `BaseTool` that doesn't wrap a function, the
tool uses `response_format="content_and_artifact"`, or the return type is not
JSON-like (a pydantic model, dataclass, set, date, or enum; most such values fail
the `output:` gate anyway, and with no `output:` they fail the untyped output check,
`contract.output`; an enum that subclasses `str` or `int` passes as its value). On
Python 3.11, use `typing_extensions.TypedDict`; pydantic rejects
`typing.TypedDict` there.

```python
from typing import Literal, Optional, Union
from typing_extensions import TypedDict


class Filled(TypedDict):
    type: Literal["filled"]
    qty: int


class Interrupted(TypedDict):
    type: Literal["interrupted"]
    by: str
    partial: Optional[int]


def place_order(qty: int) -> Union[Filled, Interrupted]:
    ...


result = run_flow(flow, {}, tools={"place_order": place_order})
```

`flow` is the `OrderResult` example from [Typed TOOL nodes](syntax.md#typed-tool-nodes),
loaded with `load_flow`.
The run-start check doesn't replace the gate: the node's `output:` still validates
each returned value.

A node or flow-assert failure returns `RunResult(status="failed")` with one
immutable `result.failure`; a successful flow returns `status="succeeded"` with
`failure=None`. Inspect `failure.code` for program logic and `failure.message` for
display. `failure.node_id`, `failure.trace`, `failure.cause_type`, and
`failure.traceback` retain execution, authored-source, and debugging context.
Canonical presentation resolves `failure.trace` through `ComposedFlow.diagnose()`.
There are no `error`, `error_type`, or standalone `traceback` aliases.
Host-control, callback, log I/O, and internal reducer errors propagate after the
engine has drained active workers and released the boundary-owned log.

### Failure codes

`failure.code` is one of the codes below (`yian.diagnostics.FAILURE_CODES`). A failed
run is terminal: `resume_flow` on its log returns the same failure and runs nothing, so
"Retry helps" below means a new run. `failure.cause_type`, the exception's class name,
is for debugging only; don't branch on it.

| Code | Raised when | Typical host reaction | Retry helps |
|------|-------------|-----------------------|-------------|
| `binding.evaluate` | A node input's `${...}` binding or default can't be evaluated, or a required input is unbound. | Fix the flow or its inputs. | No |
| `binding.type` | A tagged binding (`!int ${...}`) holds a value that breaks its tag. | Fix the flow or its inputs. | No |
| `contract.input` | A value supplied to a node input breaks the input's declared type, or a MAP `over:`, WAIT `until:`, or AGENT `llm_config` value has the wrong shape. | Fix the flow or its inputs. | No |
| `assertion.failed` | A node's or the flow's `asserts:` condition is false. (TS yian: an `asserts:` condition that fails to evaluate is `condition.evaluate`.) | Treat the result as rejected; fix the flow or its inputs. | Only if the asserted values vary between runs |
| `node.execute` | A node's body raised and no finer code applies: a CODE node's exception, a tool's error, an AGENT's model error, an ungranted tool request. (TS yian: exhausted structured-output corrections and rate-limit, quota, outage, context-length, and refusal errors have their own codes below; other provider errors, such as a bad request, an unknown model, a redirect, or a malformed reply, stay `node.execute`.) | Read `failure.message`. | Sometimes: transient errors such as a timeout or a rate limit (TS yian: a tool's; a model call's have their own codes) |
| `contract.node_result` | A node returned something other than `Output`, `Route`, or `Grow`, or a node that isn't CALL, MAP, or LOOP returned `Grow`. | Report a bug in that node's implementation. | No |
| `contract.output` | A node's or the flow's output value doesn't match its declared type. | Fix the flow, or the code or tool that produced the value. | Only if the producer's output varies |
| `contract.route` | A node returned a route handle that isn't a non-empty string. | Report a bug in that node's implementation. | No |
| `contract.growth` | A CALL, MAP, or LOOP growth was rejected, typically because the graph would grow past `YIAN_MAX_TOTAL_NODES`. | Bound the fan-out, or raise the [limit](defaults.md). | Only with a larger limit |
| `graph.terminal_unreachable` | A route skipped every path to the flow's output. A flow from `load_flow` doesn't reach this through its outputs; a hand-built graph can. | Fix the graph. | No |
| `loop.max_exceeded` | A `while:`/`until:` LOOP with `max: M` would start body run M+1 (see below). | Treat it as the loop ending without meeting its condition, or raise `max:`. | Only if the body's results vary between runs |
| `agent.max_tool_iterations` | An AGENT's tool loop reached its `max_tool_iterations` and the last reply still asked for tools; those calls are not run. There is no cap by default. | Raise the cap, or simplify the task. | Sometimes: model replies vary |
| `provider.auth` | An AGENT's model build or model call failed on its credentials (see below). | Ask the user to fix the connection (`yian auth <provider>`, or the right key). | After the credentials are fixed; or at once if the token refresh request itself failed (a network error or a 5xx) |
| `agent.output_invalid` (TS yian) | An AGENT's structured reply still breaks its `output:` type after its `retries:` corrections. | Loosen the type or improve the prompt. | Sometimes: model replies vary |
| `provider.rate_limit` (TS yian) | The provider refused the model call for its rate limit (HTTP 429), and yian's model-call retries ran out or the provider forbade a retry. | Wait, then run again; lower concurrency. | Yes, later |
| `provider.quota` (TS yian) | The provider refused the call because the account's quota or credit is used up (OpenAI's `insufficient_quota`, or ChatGPT's `usage_limit_reached` or `usage_not_included`). | Ask the user to add credit or change plan. | After the quota is restored |
| `provider.unavailable` (TS yian) | The provider was unreachable or failing (a 5xx or overloaded reply, a timeout, a dropped connection, or a reply cut short), and yian's model-call retries ran out or the provider forbade a retry. | Run again later, or switch provider. | Yes, later |
| `provider.context_length` (TS yian) | The request is longer than the model's context window. | Shorten the inputs or history, or pick a model with a larger window. | No |
| `provider.refused` (TS yian) | The model declined to answer: a refusal or a content filter stopped the reply, or the provider's content filter rejected the request (an HTTP 4xx naming `content_filter`, `content_policy_violation` or `ResponsibleAIPolicyViolation`). | Read `failure.message`; change the input. | Rarely |
| `condition.evaluate` (TS yian) | A LOOP `while:`/`until:`, a CASE `when:`, or an `asserts:` condition can't be evaluated (an error, as opposed to evaluating false). | Fix the flow or its inputs. | No |

Changed (TS yian): the codes marked (TS yian) exist only in TS yian; Python yian
reports these failures as `node.execute`, except that an `asserts:` condition that
fails to evaluate counts as false (`assertion.failed`). For a node's `asserts:`, the
`condition.evaluate` message reads
``node 'n' pre-assert '<display>' (`<expression>`) could not be evaluated: <error>``
(`post-assert` after the body); the expression shows only when it differs from the display.

`loop.max_exceeded` is on the LOOP node, with a message such as `loop 'refine'
exceeded max (5)`. A failure inside the body keeps its own code, and a `times:` loop
never raises it.

`provider.auth` means the provider refused the AGENT's own model credentials, or yian
couldn't supply them:

- the OpenAI environment key (`OPENAI_API_KEY`) is missing or changed;
- a saved login is missing or changed, its subscription expired, or a token refresh
  was refused or its request failed (a network error or a 5xx). In that last case a
  new run can help without changing the credentials;
- the provider answered HTTP 401 or 403, or 400 naming `API_KEY_INVALID` (Gemini's
  invalid key). Another provider's missing key is `provider.auth` only when the
  provider answers 401 or 403. Changed (TS yian): a 4xx that yian reads as a
  content-filter refusal, such as OpenRouter's 403 content-policy reply, is
  `provider.refused`, not `provider.auth`.

Typed-output correction retries stop at once on it. A TOOL node's or a granted tool's
own 401 is `node.execute`. An agent that resolves its `llm_config` when it runs and
needs a host key the resume wasn't given raises `AuthError` out of `resume_flow`, a
host error, not a `Failure`; see [`resume_flow`](#durable-recovery-with-resume_flow).
Credential errors in memory learners and `RunResult.backward` raise their own
exceptions, not a `provider.auth` failure.

An older yian can't read a run log holding a code added after it.

## Explicit backward feedback

`RunResult.backward(feedback=None, *, api_key=None, headers=None) -> BackwardOutcome`
operates on a terminal run created with `learnable=True`. `api_key` is the provider
key of a run started with `llm_config.api_key`, and `headers` the header values it
sent beside the key (`llm_config.headers`); `None` uses the ones the result was
created with. Given headers must have the recorded names; a name, value, or count
that doesn't fit raises `ValueError` before the attempt starts. It leaves the
forward `input`, `status`, `output`, `failure`, and `events` unchanged. A fresh forward result starts with `backward_outcome=None`
and `learning=None`; an accepted attempt updates the separately observed
`backward_outcome` and its `learning` projection.

A string addresses the whole output. A dictionary selects existing output fields;
a list selects positions and must have the exact output length at every nested
level, including positions containing only `None`. Omitted fields and `None`
carry no effective feedback. An empty string is effective text. A failed forward
run accepts only a string or `None`, because it has no usable output shape.
Invalid feedback is rejected before a new attempt or memory session opens, and
preserves this result's previously observed outcome and learning objects. A message
that is not a string, object, list, or `None`, or an object key that is not a string,
raises `TypeError`, except an int longer than 4300 digits, which raises `ValueError`
naming where it sits (`feedback at note: integer with 5000 digits is larger than yian
allows (4300 digits)`); feedback that does not match the output's shape, or nests
deeper than 100 levels, raises `ValueError`.

Changed (TS yian): `await result.backward(feedback, { apiKey, headers, onEvent })` returns a Promise
of the `BackwardOutcome`, and its errors are rejections. `feedback` is taken through host intake
(see [Types → host intake](typing.md#two-constructors-infer-vs-check)), so a whole `number` arrives
as an int (`{n: 3}` holds `3n`), and a value that isn't a yian value, or nests deeper than
`YIAN_MAX_VALUE_NESTING` levels as the run's `env` sets it, is refused with `TypeCheckError`. A
message that is not a string, object, list or `null` raises `TypeError`; feedback that doesn't match
the output's shape, a result not retained with `learnable: true`, and headers that don't fit raise
`RangeError`, not `ValueError`. There is no int digit limit. A second call on the same result while
one is in progress rejects with an `Error` ("a backward call on this result is already in progress")
and appends nothing. The `learning` and `backwardOutcome` getters read the last accepted attempt
this result observed, so they change when a call settles, whether it resolves or rejects after its
attempt was accepted. `onEvent`, which Python lacks, receives each AGENT backward model call's retry
as a `ModelCallRetried` (consumer `"backward"`) and one `ReplayDegraded` per reason per call; a
throw from it fails that AGENT's backward, and the attempt finishes `failed`. Without it (the
default) nothing is published.

This complete example needs no model or memory backend:

```python
from yian import load_flow, run_flow

flow = load_flow('''
id: structured-feedback
name: structured_feedback
input:
  items: !list(int)
nodes: {}
output:
  items: ${input.items}
  label: batch
''')
result = run_flow(flow, {"items": [1, 2]}, learnable=True)
assert result.backward_outcome is None and result.learning is None

outcome = result.backward({"items": [None, "Check this value."]})
assert outcome.status == "succeeded"
assert outcome.feedback.input_feedback == {"items": ([None, "Check this value."],)}
assert result.output == {"items": [1, 2], "label": "batch"}
```

`BackwardOutcome` is exported from `yian.runtime`. Its `status` is
`succeeded` or `failed`; `feedback` is the accepted complete root `BackwardResult`,
or `None` when the accepted prefix has no completed result. `failure` identifies a backward,
learner, or commit failure. A `memory.invalid_context` learner failure, raised when a
memory's learn context cannot be built from run material, names the owner occurrence
(`node_id`, `node_started_seq`, `scope`, `run_id`) and the cause, such as
`memory learn context could not be constructed for node 'n': ... nests deeper than the
memory limit of 128 levels (derived from YIAN_MAX_VALUE_NESTING)`; only a failure in the
root flow's fallback feedback has no node. `learning` is `None` when the run had no
memory. The root result's `input_feedback` maps input names to ordered tuples of
ordinary messages. These tuples are batches; positional lists inside them remain single
structured messages. Node self/variable channels contain ordered text messages
and remain attributed to their own occurrences, rather than becoming root channels.

Changed (TS yian): there is no 128-level memory bound, so copying memory JSON never fails
for depth; values are bounded by value nesting where they are produced.

Changed (TS yian): `BackwardOutcome`, `LearningResult` and `PolicyCommitOutcome` hold
the frozen values they were given, where Python copies them; these classes,
`BackwardFailure` and `LearningFailure` are frozen. Feedback batches, `failures` and
`policies` are arrays, not tuples; `nodeStartedSeq` is a `number`. Captures, attributed
results and mutation payloads return their stored frozen values, not fresh copies; copy one
before you modify it.

`result.backward()` with no external feedback skips unreached AGENT backward
model calls. It still presents accepted history to every enabled memory key,
including keys with empty or unexecuted histories. With effective feedback,
each reached AGENT makes at most one tool-free model call using its retained
rendered prompt, inputs, local variables/domains, accepted evidence, and recorded
effective model configuration. The evidence includes the system prompt the model
received: the rendered [`system_prompt:`](syntax.md#system-prompt-system_prompt),
or the framework instruction when the node has none, with any skill listing.
Tool interactions in that evidence are inert data. No prompt/default/recall is evaluated again, and variable feedback does not
update variables, model configuration, or the graph.

Known callback/result failures return a failed outcome without partial root
feedback. A later learner or commit failure retains already completed feedback.
Host errors, cancellation, setup errors, and log faults propagate while leaving
the accepted prefix inspectable. They do not manufacture a successful or failed
completion record.

Changed (TS yian): a stop is a signal, not an error found in a cause chain, so a
backend's `MemoryOperationError` from learn or commit is a learning failure whatever its
`cause`. A backward call finishes only its own attempt: one overtaken by a later call
while it awaits (its memory opener, a handler, a learner or a commit) is refused after
that await and appends nothing further.

## Skills

An AGENT granted `skills:` in YAML lists each skill's name and description in its
system message and reads the full instructions on demand through the built-in
`load_skill` tool (see [Agent skills](syntax.md#agent-skills-skills)). The host
chooses which skills exist for a run:

```python
from yian.compose import load_flow, load_skills
from yian.run import resume_flow, run_flow

flow = load_flow(open("review.yaml").read())   # the AGENT says `skills: all`
shared = load_skills(["shared/skills"])
personal = load_skills(["people/ada/skills"])
grant = ["code-review"]                          # the host's own per-person grant
skills = {n: s for n, s in shared.items() if n in grant} | personal   # personal wins
result = run_flow(flow, {"diff": diff}, log="run.jsonl", skills=skills)

# After a crash, pass the same skills again.
resume_flow(flow, log="run.jsonl", skills=skills)
```

A flow that grants skills (a list or `all`) needs a `skills=` mapping on `run_flow`
and on `resume_flow` of an unfinished run. Pass `{}` to grant none. Without it, or
when a listed name is missing, the call raises `ValueError` before any record is
written.

- `load_skills(roots, *, skip_invalid=False)` reads every `<root>/<name>/SKILL.md`.
  Missing roots, hidden folders, and folders without `SKILL.md` are skipped. When
  two roots have a skill with the same name, the earlier root wins. Invalid
  frontmatter, a missing `description`, a bad name, an unreadable file, a file over
  256 KiB, or two skills with one name in the same root raise one `SkillError` that
  lists every problem; `skip_invalid=True` warns and skips them instead.
- `default_skill_roots(cwd=None, home=None)` returns the folders `yian run`
  uses: `~/.claude/skills`, `~/.agents/skills`, then `.claude/skills` and
  `.agents/skills` in each directory from `cwd` up to the git root.
- `yian.skills.Skill(name, description, instructions, directory=None)`
  builds a skill in code. `directory=None` means it has no bundled files.

Changed (TS yian): `loadSkills(roots, { skipInvalid, warn })` returns a `Map` sorted by
name; with `skipInvalid` it passes each `skipped skill: ...` message to `warn` (default
`process.emitWarning`). `new Skill({ name, description, instructions, directory })` takes a
path string or `null` for `directory`. `defaultSkillRoots({ cwd, home, env })` reads `HOME`
through `env`, and an empty `HOME` counts as unset. Frontmatter lines break only at LF,
CRLF, and CR, and a leading BOM is allowed. Every frontmatter key must be text (see
[Syntax → YAML values](syntax.md#yaml-values)). Frontmatter that is `false`, `0`, `""` or
`[]` is "frontmatter must be a mapping" (Python asks for a description).

A skill's text is read once, when `load_skills` runs, so every node in the run sees
the same instructions. The listing becomes part of the node's recorded
`base_system_prompt` evidence and each `load_skill` result is a recorded tool
result, so backward learning sees what the model saw. The run log does not store
skills: pass the same mapping to `resume_flow`, as with `memory=`.

Each `load_skill` result is capped at 256 KiB (`YIAN_MAX_SKILL_FILE_BYTES`; see
[Defaults and limits](defaults.md)). There is no cumulative cap: every
result stays in the loop's message history, the evidence, and any logs.

## Steering

An AGENT authored `steerable: true` (see
[Steering a running agent](syntax.md#steering-a-running-agent-steerable)) reads a
host inbox while its tool loop runs. The inbox is any object with two methods:

```python
import threading
from yian.run import run_flow

class Inbox:
    """Steers queued per node id, plus interrupt flags, shared with the UI thread."""

    def __init__(self):
        self.lock, self.steers, self.stops = threading.Lock(), {}, set()

    def steer(self, node_id: str, text: str) -> None:      # called by the UI
        with self.lock:
            self.steers.setdefault(node_id, []).append(text)

    def interrupt(self, node_id: str) -> None:              # called by the UI
        with self.lock:
            self.stops.add(node_id)

    def drain(self, node_id: str) -> list[str]:             # called by the engine
        with self.lock:
            return self.steers.pop(node_id, [])

    def interrupted(self, node_id: str) -> bool:            # called by the engine
        with self.lock:
            return node_id in self.stops

inbox = Inbox()
result = run_flow(flow, inputs, log="run.jsonl", steering=inbox)
```

- `drain(node_id)` returns the queued steer messages for that node and removes
  them. `interrupted(node_id)` says whether to stop that node now.
- Both are called from the worker thread running the node, so make them
  thread-safe. `node_id` is the expanded id, which is unique per LOOP iteration
  and MAP item (`loop#1/worker`); take ids from `NodeStarted` events.
- Only nodes with `steerable: true` call the inbox. A finished node never calls it
  again, so an interrupt for it is ignored.
- An object without both methods raises `TypeError` before any record is
  written. An exception from the inbox, or a wrong return type (`drain` must
  return a `list` of `str`, `interrupted` a `bool`), stops the run as a host
  error, not a node failure.
- An interrupted node succeeds with its text so far. The run log records its
  output like any other; it has no separate interrupted flag. A complete reply
  whose tool call holds a NaN or infinity, or arguments nested deeper than 100
  levels, still fails the node, interrupt or not.
- `on_event` shows the interrupt taking effect. A cut stream ends with its
  `final=True` chunk, and its `ModelCallFinished` has no `error` and the usage of
  the chunks already received. Each tool call the node skipped gets a
  `ToolCallStarted` then a `ToolCallFinished` with `error="interrupted"`, so a
  UI can mark that call interrupted.
- An interrupt never stops a tool that is already running. A tool that raises
  fails the node, so a host that unblocks a waiting tool (for example an
  `ask_user` prompt) on interrupt must make the tool return a result, not raise.
- Steers ride in the node's conversation. With session memory they are recorded
  with the model request that carried them, so a node retried after a crash or
  failure sees them in its history. Without session memory a steer read by a node
  that fails or crashes before it finishes is not delivered again when the node
  is retried. Pass the same inbox to `resume_flow(..., steering=inbox)`; finished
  nodes replay without reading it.
- `yian.run.Steering` is the matching `typing.Protocol`.

Changed (TS yian): the inbox is push-based. In place of `interrupted(node_id)`,
`interruptSignal(nodeId)` returns an `AbortSignal` the host aborts to interrupt that
node. `drain(nodeId)` returns the queued messages or a promise of them, and the node
awaits it between its effects. Only the main thread reads the inbox (CODE bodies run on
worker threads but never see it), so it needs no lock. The `Steering` interface lives in
`nodes/base.ts`, and `yian/run` (`run.ts`) re-exports it.

Also changed (TS yian): every model call streams, so the interrupt is read before each later
text chunk of every chat call, and a host without `onStream` can cut a reply too; a reply
whose text had all arrived is kept whole. So an interrupt during a reply's reasoning or
tool-call stream takes effect at its next text chunk, or once the reply is complete. An interrupt during the wait before retrying a failed model
call ends the wait at once, and the node succeeds with no new text (`""`, the cut call's
empty reply). Steers are awaited at the top of each iteration only, and an interrupt that lands while
an async `drain` settles stops the node before the next model call. A reply's tool calls
are checked for an interrupt before each call's check, before each group, and before the
node fails at a call whose check failed (see
[Tool context](#tool-context)), and a group that is running finishes.

## Injecting agent memory

YAML names each memory and its type without importing an implementation:

```yaml
# support.yaml
id: support
name: support
long_term_memory:
  notes: episodic                       # flow-level: every AGENT inside recalls it
  lessons:
    type: reflective
    config: {max_items: 100, recall_limit: 12}
input:
  question: !str
nodes:
  answer:
    kind: agent
    long_term_memory: {chat: session}   # AGENT-level: this node's conversation
    input: {question: "${input.question}"}
    output: str
    prompt: Answer ${question}.
output: ${answer.output}
```

Changed (TS yian): `session` is not a memory type, so this flow is a load error; carry a
chat's history as a LOOP field (see [SessionMemory foundation](#sessionmemory-foundation)).

The host owns backends, model configuration, credentials, storage, and
lifetime. `run_flow(..., memory=None)` leaves declarations dormant, and
`run_flow` never constructs memory. `yian run` builds every declared
built-in memory; see [CLI memory](cli.md#memory-and-storage). A Python host
builds them with `build_memories`, keeps the registry open for as long as it
wants continuity, including through `backward()`, and closes it:

```python
from pathlib import Path

from yian import build_memories, load_flow, run_flow

path = Path("support.yaml")
flow = load_flow(path.read_text(), search_paths=[path.parent], source_label=path.name)
model = {"provider": "ollama", "model": "qwen3:30b-a3b-instruct-2507-q4_K_M"}
with build_memories(flow, root="./mem") as memory:
    result = run_flow(
        flow,
        {"question": "What did we learn?"},
        llm_config=model,
        memory=memory,
        learnable=True,
    )
    print(result.status, result.output)  # Forward has finished; nothing has learned yet.
    outcome = result.backward("Explain the main point more directly.")
print(outcome.status, outcome.learning)
```

Passing the same registry to later fresh runs, or building one with the same
`root` in a later process, makes committed lessons available to recall.
`memory["lessons"].reset()` explicitly starts a fresh generation. Without
`root` or `storage`, every built-in memory is process-local; see
[optional durable memory](#optional-durable-memory).

For model-free execution history, the `episodic` type keeps execution records.
To configure one backend yourself, supply it under its key and let
`build_memories` build the rest:

```python
from yian.memory.episodic import EpisodicMemory

episodic = EpisodicMemory(max_episodes=100, recall_limit=10)
with build_memories(flow, supplied={"notes": episodic}) as memory:
    result = run_flow(flow, {"question": "What happened last time?"}, memory=memory, learnable=True)
    result.backward()  # Appends retained episodes after the reverse pass.
```

YAML can set the same limits instead:
`notes: {type: episodic, config: {max_episodes: 100, recall_limit: 10}}`.
`build_memories` passes `config` to the constructor as keywords; a supplied key
must not have YAML `config`. Each successful backward attempt appends the
execution records of every accepted occurrence whose stack holds the key, in
recorded order; `max_episodes` then evicts the oldest records. Recall without a
context returns the whole bounded history chronologically. AGENT recall ranks
records by lexical overlap with the system prompt, rendered prompt, and
canonical inputs, breaks ties by recency, and returns at most `recall_limit`
records. The returned text is newline-separated canonical JSON so each episode
keeps its run and execution attribution.

`EpisodicMemory` is process-local by default. Reuse the same object to share committed
history across fresh runs in one process. `reset()` clears history and starts a
new generation. A session snapshot is detached JSON suitable for the engine's
recorded recovery boundary or for restoring a pristine `EpisodicMemory`.
Use `storage=path` to retain complete backend state across restarts.

Changed (TS yian): `new EpisodicMemory({ maxEpisodes, recallLimit, storage,
createParent })` takes an options object whose limits are integral `number`s or
`bigint`s; snapshots are deep-frozen rather than detached copies; `recall` is
async and awaits `rewriteQuery`, which may return a promise; recall text and
entry ids use yian's run-log canonical JSON (`logJson`: ints as digits, floats
marked `.0`); an episode holding `±Infinity` fails `learn`
as `episodic.invalid_context`.

### Building memories with `build_memories`

`build_memories(flow, *, root=None, storage=None, supplied=None, factories=None)`
reads the flow's memory table and returns a `MemoryRegistry`, a read-only
mapping from memory key to backend with a `types` mapping and `close()`. For
each key:

1. `supplied[key]`, if given, is used as is and never closed by the registry.
   A supplied key whose declaration has non-empty `config` is an error.
2. Otherwise a custom type is built by `factories[type](**config)`.
3. Otherwise the built-in type is built with `config` as keywords. Its storage
   is `storage[key]` (parent must exist), else
   `<root>/<flow id>/<scope...>/<name>.sqlite` (`.md` for profile; parents
   created on first use), else RAM. A profile without a path gets an
   in-RAM `InMemoryMarkdownStore`.

These are `MemorySetupError`s:

- a custom type with no factory or supplied value;
- `factories` naming a built-in type (use `supplied` to replace one instance);
- `storage` for a supplied or custom key;
- a key in `storage` or `supplied` that the flow does not declare;
- a value that is not a `Memory`;
- a constructor rejecting its config;
- a `storage` value that is not a `str` or `os.PathLike[str]` (for example
  `None`, bytes, or an int); the error names the key;
- a `storage` value that is not a local file path (`:memory:`, a `file:`
  URI, an empty string), for a profile `.md` path as for a SQLite file;
- two built-in keys whose storage paths (from `storage` or `root`) resolve to
  the same file, compared after resolving links and ignoring case; the error
  names both keys;
- `root` with a flow whose `id:` is not one safe path component.

If a later key fails, everything already built is closed. `close()` closes the
backends the registry built, in reverse build order. A backend whose close
raised (for example a retryable `episodic.busy`) stays owned, so a later
`close()` retries it. Construction never
touches the filesystem; files open on a backend's first operation
(`open_run`, `reset`, or entering it with `with`). Learner models are not
configurable from YAML. A keyed or custom reflective or profile learner comes
through `supplied=` or `factories=`. A supplied key takes no YAML `config`, so
set its limits on the object:

```python
from yian.memory.reflective import ReflectiveMemory

# For a flow that declares `lessons: reflective` (no config) and `vector: {type: vector}`.
memory = build_memories(
    flow,
    root="./mem",
    supplied={"lessons": ReflectiveMemory({"provider": "ollama", "model": "qwen3"}, max_items=100)},
    factories={"vector": MyVectorMemory},
)
```

Changed (TS yian): `await buildMemories(flow, { root, storage, supplied, factories, env, models,
onEvent })` is async and resolves to a `MemoryRegistry`, a read-only `Map` subclass whose `types` is
a read-only `Map` too (`set`, `delete`, `clear` and the upserts throw `TypeError`). `storage`,
`supplied` and `factories` are plain records, and a `storage` value must be a string. `env` (default
`processEnv`) reaches the reflective and profile backends the call builds. Those learners build
their models through `models` (a `ModelAccess`, under the node id `__memory__` with consumer
`"memory:<key>"`; its `env` wins over `env`), and `onEvent` receives each learner retry as a
`ModelCallRetried`, its error scrubbed of the call's credentials, and each replay reason as a
`ReplayDegraded`; a throw from `onEvent` during a retry fails that learner call. A built-in gets its
YAML `config` keys in camelCase (`max_episodes` is `maxEpisodes`). A factory is called with
`toHost(config)` (safe ints as `number`) and may return a promise, which is awaited; a factory or
constructor that throws `TypeError` or `RangeError` "rejected its config", and the setup error keeps
it as `cause`. `await registry.close()` awaits each owned object's `close()`, which may return a
promise; a call made while a close is in flight returns that close's promise. Use `await using
registry = await buildMemories(flow)` where the runtime or compiler supports it (else
`try`/`finally` with `await registry.close()`): when the body throws and the close fails too, JS
throws a `SuppressedError` whose `error` is the close failure and whose `suppressed` is the body's
error, where Python notes the cleanup failure on the primary. Messages show lists and quoted keys in
the text form (`memory key(s) ["other"]`, `storage["notes"]`) and drop Python's `=` from keyword
names (`use supplied to replace one built-in instance`).

Changed (TS yian): `new ReflectiveMemory({ llmConfig, maxItems, recallLimit, storage, createParent,
env, createModel, onRetry })` takes an options object. `env` is where the learner's model defaults
and credentials are read (default `processEnv`). `createModel(config)` builds the learner's model
from the pinned config (default `chatModelFromConfig`), and `onRetry` sees each transport retry, its
error scrubbed of the call's credentials. `recall` and `learn` are async. The learner makes one
structured call through `callWithRetries`, so only transport errors retry, under the process's
`YIAN_MODEL_RETRIES`. yian checks the reply itself (exactly `actions`); an invalid reply fails
`reflective.contract` without a retry, and a model without native structured output fails
`reflective.structured_output`. The reflection prompt shows its materials in the text form (`1.0` is
`1`, bare `Infinity`, JS key order). Limits are integral `number`s or `bigint`s; snapshots are
deep-frozen; `recall` awaits `rewriteQuery`, which may return a promise, and the default query uses
`logJson`; blank checks use JS `trim()` whitespace. Messages keep Python's wording, with `null` for
`None`, `array` for `tuple` and `rewriteQuery()` for `rewrite_query()`.

### Profile memory

`ProfileMemory` recalls one markdown document into every AGENT whose stack holds
its key, and rewrites it on backward with a compare-and-set write. For a flow
that declares `long_term_memory: {about: profile}`:

```python
from yian.memory.profile import ProfileMemory
from yian.memory.storage import FileMarkdownStore

profile = ProfileMemory(FileMarkdownStore("people/ada.md"), max_chars=32_000)
result = run_flow(flow, inputs, memory={"about": profile}, learnable=True)
outcome = result.backward("Ada prefers short replies.")
if outcome.failure is not None and outcome.failure.code == "profile.version_mismatch":
    ...  # someone edited the profile; nothing was written; start a new attempt
```

Only `profile.version_mismatch` guarantees that nothing was written. After any
other commit failure, or an attempt killed mid-commit (commit status `unknown`),
look for the attempt's `commit_id` in the store's write provenance before
retrying. If it is missing, compare the current document with the proposed
`text` in the run log. The snapshot records the pinned version and text hash, never the text.
`FileMarkdownStore` takes the same local file path rule as SQLite storage.
`profile.close()` does no I/O; afterwards `open_run`, `reset`, `read`, `write`,
and its open sessions raise `MemoryOperationError` with code `profile.closed`.
A session used after its own `close()` raises `profile.session_closed`, as
episodic and reflective sessions raise `<type>.session_closed`.
A registry from `build_memories` closes the profiles it built.
See [profile memory](learning.md#profile-memory) for recall, learning, and store
rules.

Changed (TS yian): `new ProfileMemory(store, { llmConfig, maxChars, inputBudgetChars, env,
createModel, onRetry })`, exported from `@ngocbh/yian` (Python imports `yian.memory.profile`) and
passed in a `Map`. `openRun`, `read`, `write`, `learn` and `commit` are async; store errors map by
kind: a Node system error is `profile.store_unavailable`, a `RangeError` `profile.store_invalid`.
`env` is where the learner's model defaults and credentials are read (default `processEnv`).
`createModel(config)` gets the resolved config with the host credentials (default
`chatModelFromConfig`), and `onRetry` sees each transport retry, its error scrubbed of the call's
credentials. The learner makes one plain call through `callWithRetries`, so only transport errors
retry; a reply without `<profile>` tags fails `profile.contract` without a retry.

### Optional durable memory

`EpisodicMemory`, `ReflectiveMemory`, and `SessionMemory` accept the host-only
keyword `storage="path/to/memory.sqlite"`. `storage=None`, the default, opens no
database. Use a separate file for each backend; its parent directory must already
exist. Paths are not flow YAML, public snapshot fields, or run-log settings.

Construction opens nothing beyond checking the path's shape. The file opens
on the first operation (`open_run()`, `reset()`, or `with backend:`), under
the backend's lock. `create_parent=True` creates the parent
directory then; `build_memories` passes it only for paths derived from
`root`. `close()` before the first open does no I/O, and a closed backend
rejects every later `open_run()` with `<type>.closed`. A backend checks its
process at every operation once a path is configured, so a forked child
cannot use it even if the parent never opened it.

Changed (TS yian): `EpisodicMemory` and `ReflectiveMemory` take
`{ storage, createParent }` in their options object (TS yian has no `SessionMemory`); the file opens at the first operation
(`openRun()`, `reset()`, or episodic's and reflective's `rewriteQuery()`). There is no `with`
form (call `close()` in `finally`) and no fork check.

```python
from yian.memory.session import SessionMemory

# The flow's AGENT declares `long_term_memory: {chat: session}`.
with SessionMemory(storage="./session.sqlite") as memory:
    first = run_flow(flow, first_inputs, memory={"chat": memory})

# A later process can reopen the same file after the first owner closes it.
with SessionMemory(storage="./session.sqlite") as memory:
    second = run_flow(flow, next_inputs, memory={"chat": memory})
```

The local SQLite file has one owning backend instance at a time, including
between operations. A second owner fails immediately. This is a single-host,
single-owning-process backend, not shared-cluster storage. New files use mode
`0600`; contents are not encrypted and can include prompts, replies, tool output,
and learned notes. Use an access-controlled local filesystem with reliable SQLite
locking and durability; network/shared filesystem guarantees are not provided.

Use `close()` or the backend's `with` block to release ownership. Idle `open_run()`
handles do not block close, but become unusable afterward. Active operations,
including reflective model callbacks, and active or queued session turns reject
close immediately without interrupting work. Closing a run handle cannot release
the protection of an external call still finishing. Context cleanup preserves an
initiating exception; handles inherited across `fork()` are not usable.

Each accepted backend mutation is saved transactionally before it becomes visible
in memory. An I/O failure leaves the handle unusable until it is closed and
reopened; do not blindly retry effects. Reopening rejects incompatible, malformed,
or unsupported stored state instead of overwriting it. `reset()` clears the
memory's content and starts a new generation, while retaining commit deduplication so
replayed commits cannot resurrect cleared content. It does not remove the file or
external provider resources.

Private storage retains state omitted from public snapshots: commit deduplication,
pristine/adoption flags, and SessionMemory's derived summary cache. It never saves
live handles, queues, callbacks, or provider clients. Public run-opening snapshots
stay pinned, and reopening an older snapshot does not rewind newer live history.
The run log is not a substitute for this store, and repeating an interrupted
model request or tool effect remains possible.

Episodic and reflective limits are stored with the file. A limit left unset
adopts the stored value; a different explicit value, from YAML `config` or the
constructor, fails with `<type>.config_conflict`. Only removing the file or
using the stored limits clears it; `reset()` keeps the stored limits. Recorded run snapshots remain unchanged, and future reflective learning
uses the current host `llm_config`. SessionMemory uses the current history
budget and recent-turn count. Increasing the recent-turn count restores raw
context for newly protected turns instead of hiding them behind an older
summary. Storage ownership belongs to the host. `yian run` places files with
`--memory-root DIR` or `--memory-storage KEY=PATH` and passes each memory's YAML
`config` to its constructor.

Changed (TS yian): `SQLiteStorage` keeps the same file, table, and single-owner rules, but its
state envelope is version 3 (Python's is 2). A `.sqlite` file written by Python yian fails to
open with `storage.version` and is never read; there is no importer, so start with a fresh
file. The state may hold `Infinity` and `-Infinity`, stored with yian's JSON tag
`{"$float":"inf"}`. NaN fails with `storage.invalid_state`, `-0` is stored as `0`, and ints are
`bigint`. JS can't fork a live process, so there is no fork
check and no `storage.forked`. The path must be a string.

Application-specific implementations use the source-neutral `Memory` and
`MemorySession` protocols exported by `yian.memory`.
`Memory.open_run()` returns a run-scoped `MemorySession` that implements
snapshots, `recall`, `learn`, and idempotent `commit`. Build a custom type with
`build_memories(..., factories={"my_type": MyMemory})`.

`session.recall(context)` may return a string or `MemoryContent(text=..., transcript=...)`.
A string contributes notes/instructions. A transcript source implements
`TranscriptSource.open_turn(...)` and contributes structured message history.
SessionMemory returns this richer carrier; it never flattens history into a note.
Each key in the AGENT's stack is recalled once, outermost first, and contributes
one notes section. Only a session memory, whose one owner is the AGENT itself,
may contribute a transcript. A notes memory beside a session adds context
without hiding the session.

Changed (TS yian): every host memory method (`Memory.openRun({ snapshot })` and
`reset()`, the session's `recall`, `learn`, `commit` and `close`) may return a value or a
promise, and yian awaits it. A session has no `with` form: call `close()`. There are no
transcripts: `new MemoryContent({ text })` holds only notes, and `MemoryContext` only its
sections (no `TranscriptSource` or `ConversationTurn`).

The evaluator gives agents one `MemoryContext`, containing labelled text sections
and an optional transcript turn. This is analogous to using a session transcript
together with project instructions or saved memories. Both influence the model;
only the transcript preserves who said what and which result belongs to each tool
call. Context handles are transient and do not change public snapshot formats.

Recall works whether or not `learnable=True`. During an explicit backward attempt,
every memory key gets exactly one `learn` call, in sorted key order, even with an
empty batch. The batch holds every accepted occurrence whose recorded stack
contains the key, of any node kind, with its episode and attributed feedback.
Unattributed root feedback goes only to memories declared on the root file's
flow. Only a memory's own learner proposes mutations for it. All proposals must
succeed before any memory commit is requested. A known learner, context, or
proposal failure discards the temporary proposals and commits nothing.

After proposals are accepted, each memory key receives one ordered batch, in
sorted key order, including an empty batch. Its request is durable before the
backend call. A known commit failure stops later memories; already successful
memories are not rolled back. `LearningResult.status` is `succeeded`, `failed`,
or `incomplete`. Its ordered `policies` entries are `PolicyCommitOutcome` values
whose `.memory` names the key; they report:

| Commit status | Accepted evidence |
|---|---|
| `unrequested` | no commit request |
| `unknown` | a request without an accepted success/failure observation |
| `succeeded` | confirmed success |
| `failed` | confirmed backend failure |

Known learner/commit failures make learning `failed`, regardless of the backend's
retryability hint. An interrupted attempt remains `incomplete` unless it already
contains such a failure, even if every memory's success was recorded before the
final completion record. Failure entries (`LearningFailure`) name the memory key
and the operation, `learn` or `commit`.

A Started-only prefix has no completed feedback, even if process-local work ran
before an append failed. Ready retains completed feedback and proposals for
inspection. Requests and observations add the commit dispositions above; only
Finished settles the attempt.

Each `result.backward(...)` call is a new attempt, including after a successful,
failed, or committing attempt, and on a result from `resume_flow`. It opens fresh
sessions from the original snapshots and recomputes all callbacks and proposals.
An unfinished earlier attempt is first recorded as interrupted. Accepted proposals
are inspectable but are never resumed. There is no automatic retry, rebase, or
rollback. Every attempt's commit IDs (`ac-memory-v3:`) include its attempt
number, so a backend that deduplicates by commit ID still applies a later
attempt's identical batch. A new attempt after an `unknown` request may
therefore apply a second write; the host reconciles. An older
`EpisodicMemory` or `ReflectiveMemory` revision in the same namespace and reset
epoch restores successfully. Its later `learn` fails with
`episodic.stale_generation` or `reflective.stale_generation` without rebasing;
the backward outcome retains the completed feedback.

Changed (TS yian): that record is a failed `BackwardFinished` with failure code
`backward.unfinished` ("backward attempt did not finish"); Python yian records `backward.interrupted`.

Changed (TS yian): the commit-ID digest input is yian's run-log canonical JSON
(`logJson`: keys in UTF-16 code-unit order, ints of any size, integral floats
written with `.0`) of `[run_id, attempt_id, memory, [[mutation_id, payload], ...]]`, not
Python's `json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))`.
The two texts differ only for floats, for objects shaped like the codec's escapes
(`{"$record": ...}`, `{"$float": "inf"}`), and for the order of keys holding
characters outside the Basic Multilingual Plane (UTF-16 code units versus code
points); elsewhere the IDs match. A run log never spans both implementations, so
the prefix stays `ac-memory-v3`.

## SessionMemory foundation

Changed (TS yian): no session memory; carry history as a LOOP field. TS yian has no
`SessionMemory`, no conversation contracts (`Conversation*`, `TranscriptSource`,
`ConversationTurn`) and no history summaries: a `session` memory is a load error
("session memory was removed; carry the history as a loop field"), an AGENT's model reads
only its own start's messages, and the `history_summary` model-call purpose and the
`YIAN_SESSION_*` settings are gone. Chat history is a value a LOOP carries from one turn to
the next. The rest of this section is Python yian's.

`yian.memory.session.SessionMemory` retains AGENT
message conversations across calls to `run_flow`.
The backend owns history updates and
FIFO admission directly; runtime owns only the optional turn's lifetime and
authored scope. Explicit backward learning retains self-feedback as notes through
the same memory lifecycle. Older API context can be summarized without deleting
raw history. `yian run` builds every declared built-in memory, session included.

Declare a session memory on the AGENT, for example
`long_term_memory: {chat: session}`, and pass the same backend under that key to
each run. A session cannot be declared on a flow. For example:

```python
from yian import load_flow, run_flow
from yian.memory.session import SessionMemory

flow = load_flow("""
id: remembered-chat
name: remembered_chat
input:
  question: !str
nodes:
  answer:
    kind: agent
    long_term_memory: {chat: session}
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
output: ${answer.output}
""")
memory = SessionMemory()
model = {"provider": "ollama", "model": "llama3.2:3b"}
first = run_flow(flow, {"question": "My project is called Atlas."},
                 memory={"chat": memory}, llm_config=model)
second = run_flow(flow, {"question": "What is my project's name?"},
                  memory={"chat": memory}, llm_config=model)
```

The second invocation receives the first invocation's messages. Neither
`learnable=True` nor `backward()` is needed. Without a recalled transcript source,
API calls start with the current prompt and any recalled notes.

`SessionMemory` is the concrete backend; the existing `MemorySession` name still
means the run-scoped protocol returned by a memory backend. The contracts-only
`yian.memory` facade exports `MemoryContent`, `MemoryContextRequest`,
`MemoryContext`, immutable conversation carriers, and the `TranscriptSource` and
`ConversationTurn` protocols. Existing text-only recall implementations remain valid. Import the concrete backend from its own module:

```python
from yian.memory.session import SessionMemory

memory = SessionMemory()
with memory.open_run() as opened:
    snapshot = opened.snapshot  # Detached JSON, pinned when this run handle opened.

restored = SessionMemory()
with restored.open_run(snapshot=snapshot) as opened:
    assert opened.snapshot == snapshot

memory.reset()  # Backend-wide; allowed only while no turn is active or queued.
```

The host constructor accepts `history_budget_chars` (default 32,000,
`YIAN_SESSION_HISTORY_BUDGET_CHARS`), `recent_turns` (default 4,
`YIAN_SESSION_RECENT_TURNS`), and optional `storage=path`. YAML `config` may set
`history_budget_chars` and `recent_turns`; `storage` is host-only. A public snapshot has
exactly `version` (currently `3`), `namespace`, `generation`, `conversation`, and
`notes`. `conversation` is the instance's one conversation view, or `null`. A snapshot
also holds learned notes, never queues, leases, the private summary cache, or
the commit ledger. Version 1 and 2 snapshots are rejected with
`session.invalid_snapshot`.
A pristine backend can adopt a snapshot; an already
used backend accepts only proven history in its current namespace and generation,
without rewinding live state. Every snapshot getter returns a detached copy of
the original opening snapshot, even after turns advance live
history. Without `storage`, the backend writes no files or database. See
[optional durable memory](#optional-durable-memory) for restart persistence.

Changed (TS yian): `new SessionMemory({ historyBudgetChars, recentTurns, storage,
createParent })` takes an options object whose settings are integral `number`s or
`bigint`s; snapshots are deep-frozen rather than detached copies; there is no `with`
form (call `close()` in `finally`) and no fork check.

### Conversation ownership and updates

A session memory key holds exactly one conversation, owned by the single AGENT
that declares it. Every call site of that AGENT shares the conversation, and
calls always queue in FIFO order, as do concurrent runs on the same backend.
Another AGENT keeps a separate conversation only by declaring its own session
key; there is no flow-wide conversation or queue. MAP items and LOOP iterations
do not create new conversation identities;
there is no fork option. Queue order is admission order, not a promise of MAP
input-index order. Use non-session memory for independent MAP items.
Other memories in the AGENT's stack still provide ordinary recall; a
non-session memory never hides the session. Runtime supplies the consumer's
static path and the accepted invocation's run ID/start sequence; the backend
never inspects a graph.

Entering the recalled transcript's `open_turn(request, check_stopped=check)` reserves the queue position
before reading the latest live conversation. The turn's `request(...)` and
`observe(...)` methods validate and publish directly in SessionMemory. No runtime
checkpoint callback or acknowledgment is required. The caller executes a provider
or tool between those calls, outside the backend lock. Cooperative stop checks run
while waiting and between effects; actual observations are retained before honoring
a stop that arrived during the effect. A model or tool call that raises while a stop
is pending is observed `interrupted` and the run stops instead of failing (a host
callback fault still escapes); if the stop is honored only after the failure is
recorded, the effect stays `failed` and the run still stops.

The evaluator keeps the lease through the node body, stream cleanup,
post-assertions, and result-shape validation, then releases it before returning
to the dispatcher. A completed turn records completion of that execution, not
acceptance of a later typed-output validation or run-log append.

These updates advance automatically through the turn capability, independently of
`learnable` or backward. `SessionMemory` recall returns
`MemoryContent`: `.text` supplies the memory's committed learned notes,
and `.transcript` supplies the source for structured history. Context-free recall
returns empty `.text` with the same transcript source. Code calling `recall`
directly should read `.text` when it only needs notes.
The API adapter's compatibility descriptor contains the resolved provider and
versioned message format. Compatible model, prompt, and tool-grant changes keep
history; changing provider or message format requires explicit reset. Every call
uses current instructions (the node's rendered `system_prompt:` or the framework
instruction), recall, and tool grants. Historical system/developer
instructions are not reinstated as current instructions.

Changed (TS yian): the message format is `yian.message.v1`, and each stored item's payload is
`{exchange_id, message}` with `message` in that format; TS yian never reads Python memory
stores. `await source.openTurn(request, { signal })` waits for the lease and resolves to the
admitted turn; an aborted `signal` is the cooperative stop, and a waiter learns of a poisoned
store at its next wake-up. `request`, `observe` and `view` are synchronous. `turn.close(error?)`
replaces `__exit__`: it never rethrows `error`, and appends a stop or cleanup failure to
`error.notes`. `-0` equals `0` in a compatibility descriptor.

### API history summaries

```python
memory = SessionMemory(history_budget_chars=32_000, recent_turns=4)
```

`history_budget_chars` is a positive integer soft target for projected historical
context; `recent_turns` is a nonnegative integer. These are host settings, not a
provider token-window estimate. Current instructions, new input, tool schemas,
and learned-note recall are outside this history budget.
The count is the character length of the canonical JSON message-dictionary
projection of prior context, plus the cached summary text when present. Summary
labels and current-turn context are not included in that count.

When older context exceeds the budget, SessionMemory can request a summary using
the agent's current, unbound model. It always preserves the latest four prior
whole turns by default, plus the active turn. A turn includes all its model/tool
exchanges and structured correction attempts; failed and interrupted admissions
also count. `recent_turns=0` still preserves the active turn. Recent turns are
never truncated, even when they alone exceed the budget.

Summarization changes only active context. Raw messages, actual tool results,
and learned notes remain intact. The summary is labelled historical data, not a
new instruction or executable tool request. Summaries are cached privately and
extended only when more older turns need compaction; an oversized valid summary
is retained as-is, without truncation or an automatic retry. Reopening an older
run snapshot does not rewind live history or its cache. Reset clears the cache.

Summary work happens under the same conversation's FIFO lease, outside the
backend's condition lock. Other session memories proceed independently.
A failed, malformed, empty, or tool-request summary response fails that invocation
without deleting history or making the normal answer request. Summary work does
not run backward learning or enter structured-output correction retries.

Changed (TS yian): `await turn.prepareHistory({ projectionVersion, project, summarize })`
is async and awaits `project` and `summarize`, which may return promises; the budget
counts code points, as Python's `len()` does.

### Learned notes

With `learnable=True`, explicit `result.backward(feedback)` can produce textual
`self_feedback` in the existing API backward model call. SessionMemory's
learner turns that feedback into deterministic note mutations; the existing
memory commit publishes the batch atomically and idempotently. There is no
additional note-generation model call. For the flow and model above:

```python
reviewed = run_flow(flow, {"question": "Describe Atlas concisely."},
                    memory={"chat": memory}, llm_config=model, learnable=True)
outcome = reviewed.backward("State assumptions before presenting conclusions.")
```

Any self-feedback successfully committed by that backward attempt is available
to later runs through ordinary memory recall. Merely enabling `learnable` does
not add notes; backward without effective feedback adds none. Notes belong to
the session memory and record the AGENT path and run that produced them. Other
memories in the stack do not learn these notes, and
executions that failed before admission produce no session note.

Recall uses the run-opening pinned notes, while conversation history uses the
latest live view after FIFO admission. A run already open when notes commit keeps
its earlier recall view. Notes are separate from raw messages, so learning never
rewrites model/tool history and does not replay tools. They are also separate
from the transcript-summary history budget. Backend-wide `reset()` clears
notes, history, and derived summaries, retaining commit deduplication so replay cannot resurrect
cleared notes.

### Partial history, retry, and reset

Failed or interrupted turns retain completed observations. An unobserved request
stays pending, meaning its result is unknown. A new invocation may continue after
that incomplete turn; it receives its own identity and does not turn old unknown
results into successes. The memory backend does not execute tools or providers.
Restoring saved history is not an instruction to execute stored tool calls, but
ordinary node retry may send a request again and repeat actions. Exactly-once
execution and exact mid-request continuation are not promised.

API history retains actual model replies, tool results, and structured-output
corrections. Native structured replies are retained before local parsing, so a
parsing failure does not erase a received reply. A failed memory write fails the
invocation rather than triggering a provider retry. For provider input only, the
adapter makes incomplete tool exchanges explicit as unknown/error context and
renders schema-formatting calls as inert context. These projections never alter
raw history, invent a successful result, or execute a historical tool call.

Closing a run handle cancels its queued turns and prevents new requests. An active
turn retains its lease until its owner exits, so another run cannot overlap a
provider call that is still finishing. Actual observations can still be retained
by that owner before release. A released turn cannot append late results.

The run-opening snapshot stays immutable. Reopening a compatible older snapshot
against the same backend uses its latest live history without rewinding it. A new
process restored only from that old snapshot has only that snapshot's history:
later progress requires retained backend state or reopening its durable storage.
The run log does not rebuild SessionMemory.

`reset()` immediately raises `MemoryOperationError` with code `session.busy`
without changing state if **any** turn is active or queued. When idle, it clears
the conversation and advances the generation. Older run
handles then fail as stale. Reset does not delete
external provider history or files. See the [rendered contracts](reference.md#session-memory-foundation)
for the lower-level signatures.

## Diagnosing a failed run

```python
from yian.diagnostics import Diagnostic

if result.failure is not None:
    diagnostic: Diagnostic = loaded.diagnose(result.failure)
    print(diagnostic.message)
    for frame in diagnostic.frames:
        print(frame.label, frame.range.start_line)
```

`ComposedFlow.diagnose(failure) -> Diagnostic` is the Compose front end's source
resolution seam. It copies the failure's `code`, `message`, `node_id`,
`cause_type`, and `traceback`, then resolves every opaque trace origin in its
existing outermost-to-innermost order. Duplicate origins remain duplicate
frames. An empty trace produces an unlocated diagnostic without a warning. If a
non-empty trace cannot be resolved completely, diagnosis returns no frames and
the single note `Source presentation was unavailable.` rather than exposing a
partial source stack or replacing the semantic failure.

`Diagnostic` is defined in `yian.diagnostics`; it is not currently a
package-root `yian` export. The CLI also calls
`loaded.diagnose(result.failure)` and sends that value to the shared renderer;
library callers use the same method when they need presentation-ready frames.

## Authoritative run logs

Every execution is reduced from one ordered `RunLog`, including process-local runs.
Records use schema 9, and readers accept only schema 9. A log written with
schema 7 or 8 is rejected on read with "run log schema N predates schema 9 and
is not supported"; other versions are rejected too. Recovery preserves the
accepted prefix's canonical bytes and appends new records without rewriting it.

The closed durable record vocabulary is `RunStartedRecord`,
`RunResumedRecord`, `NodeStartedRecord`, `NodeStepRecord`, `NodeSucceededRecord`,
`NodeRoutedRecord`, `NodeExpandedRecord`, `NodeFailedRecord`,
`MemoryCommitRequestedRecord`,
`MemoryCommitSucceededRecord`, `MemoryCommitFailedRecord`,
`BackwardStartedRecord`, `BackwardReadyRecord`, and `BackwardFinishedRecord`.
`RunStartedRecord.defaults` holds the [`yian.defaults`](#defaults-and-limits) values the
run started with that differ from the built-in ones, keyed by entry name
(`{"MAX_TOTAL_NODES": 20000}`); it is omitted from the line when empty, so a run with
every built-in value writes the same bytes as before, and a log without it counts as
all built-in values. `RunStartedRecord.call_time_llm` lists the agents whose
`llm_config` resolves only when they run (see
[Values in `llm_config`](syntax.md#values-in-llm_config)), which
`effective_llm_by_static_path` leaves out: `agents` maps each one's static path to the
layers below its own (the outer flows and the host layer), and `selection` holds the
defaults, saved models, and connection identities they may resolve to, limited to
their providers when run start knows them. Neither holds a key or header value. It is
omitted when no agent resolves at call time, so such logs are unchanged; a yian
version from before the field can't read a log that holds it, although the schema
stays 9. Each such agent records the config it resolved, without keys or header
values, as a `NodeStepRecord` step `{"effective_llm_config": {...}}` before it opens
memory or calls its model. A `NodeStepRecord` holds one step an in-flight node recorded,
such as a [TOOL retry](#tool-retries); the node gives it meaning, and a resumed node
reads its steps back. A log holding one can't be read by a yian version from before
the record existed. The four node
outcome records carry canonical forward inputs and routes when the recorded
run has `learnable=True`; `learnable=False` omits that capture. Optional model/tool
evaluation evidence is separate. Terminal success
or failure is derived from node transitions. `NodeFailedRecord` stores
the nested `Failure` directly beside any evidence. A causal `NodeRoutedRecord`
stores the optional derived terminal-unreachable `Failure` that replay
revalidates before restoring; no separate forward terminal record exists.

Changed (TS yian): records use schema 15, and readers accept only schema 15. The schema
is checked before anything else in a line. A schema 9 log is rejected with "run log
schema 9 was written by Python yian; resume it with Python yian", a schema 10 to 14 log
(TS yian's, from before record times, before `RunStarted` held the flow files, before pool
writes were `pool_commit` records, before node steps were numbered, or before an unfinished
backward attempt was `backward.unfinished`) with "run log schema N predates schema 15 and is
not supported", and any other schema with "unsupported run-record schema: N" (a float token
shows as `15.0`); the "predates" message for schemas 7
and 8 is gone. All are a
`RunLogSchemaError` (from `yian/run`), a `RangeError` whose `schema` holds the schema the
line names (`null` when it isn't an integer), so a host tells a log it can't read from a
damaged one. yian isn't released yet, so there is no migration: finish or drop an older
run with the yian version that wrote it.

Changed (TS yian): `RunStartedRecord` records what the run needs to rebuild it, once:
`flow`, the flow file and every `uses:` file the loader read, `{root, dir, files}`, or
`null` for a flow built in code. `files` maps each file's key to `{text, refs}`: its text,
and `refs`, each `uses:` ref written in it (its path, with `@<version>` when pinned) mapped to
the key of the file it resolved to. `dir` is the real path of the folder the root was loaded
from, and a key is a file's real path relative to it (`..` for a file outside it, such as a
`system.paths` folder or a child a host resolver read from elsewhere), so one file reached
by two paths is recorded once; a file's CODE references resolve in its key's folder. The
root is keyed by the file name of its label, and a host resolver's child's files by their
real paths too. Two texts a host resolver gave that claim one path get keys of their own
(`<key>~1`). A YAML root whose host-resolver child was built in code records `flow: null`. `skills`, each skill an AGENT
grants, `{<name>: {description, instructions}}`; `tools`, each tool an AGENT grants, groups
expanded, `{<name>: {description, parameters}}`; `yian_version`, the version that started the
run; and `parent`, `{run_id, node}` for a run the host started from another run (a child run
or a learning pass, `runFlow`'s `parent` option; `node` is `null` when no one node did), else
`null`. A CALL, MAP or LOOP splices its child into the same run, so a subflow has no `parent`.
`flow_fingerprint` stays. A memory key table entry no longer holds `conversation_eligible`.
Codex's `session_meta` (the base instructions and tools, once, on the first line) and
DeepSeek Harness's request header are the counterparts.

Changed (TS yian): every record carries `at`, when the log appended it: a UTC RFC 3339
time with milliseconds, such as `"at":"2026-10-20T14:02:11.123Z"` (the form of
`Date.prototype.toISOString`, years 0000 to 9999), read from the log's clock as it appends.
`runFlow` and `resumeFlow` stamp from their `scheduler`'s clock (the wall clock by default),
and `MemoryRunLog` and `JsonlRunLog` take a `clock` option (epoch milliseconds, default
`Date.now`). `at` is metadata: replay, the reducer, the flow fingerprint and recovery ignore
it, and a resume keeps the recorded times while its own records carry the resume's times,
so the gap before `RunResumed` shows. Times within one log aren't guaranteed to increase,
since the clock can move; `seq` is the order. A node's duration is its outcome record's
`at` (`NodeSucceeded`, `NodeRouted`, `NodeExpanded` or `NodeFailed`) less its `NodeStarted`
record's `at`. `RecordEnvelope.at` is `null` only on an envelope no log has appended, such
as one the engine prepares; `encodeRecord` writes such an envelope without `at`, and
`decodeRecord` reads that text back with `at: null`, but every log reader (`resumeFlow`,
`JsonlRunLog.resume`, `MemoryRunLog.fromRecords`, `readRunLog`, `iterRunRecords`) refuses a
line without `at`. A clock reading that isn't a finite time in those years fails the append
with a `RangeError` before anything is written, without poisoning the log. Python yian's
schema 9 records hold no time.

Changed (TS yian): every write to the variable pool is its own record, `PoolCommitRecord`
(`pool_commit`), with the flat payload `{node_id, key, kind, value}`: `key` is the pool key the
value goes under (Python's `commit_target`: the node itself, or for a CALL's child END and a
stopping LOOP driver, their spawner), and `kind`/`value` are the typed value's wire form, so int
and float, `Any` and files stay apart. `NodeSucceededRecord` holds no value: its payload is
`{node_id, commit, evidence, forward_capture}`, where `commit` is the seq of the node's
`pool_commit` it accepts. `RunStartedRecord` holds no `initial_pool`: the run start is
`run_started`, then a `pool_commit` per system value (`system.now`, `system.run_id`,
`system.today`, in key order, keyed `system.<name>`, a name no node id can take), then the START
inputs (`key` = `node_id` = the START's id), all naming the START. These initial commits apply
at once, and START is ready only after the last.

- Write order: a node's `pool_commit`, then its `node_succeeded`. Any commit other than the run
  start's waits on its node until that node's success names its seq; then it is applied to the
  pool. A success naming another seq, and a `pool_commit` for a node that isn't running, is a
  `RangeError`, live and on replay.
- An orphan commit (a crash between the two records) is ignored: the resume's `RunResumed`
  retries its node, which runs again and writes a new commit at a new seq. The orphan line stays
  in the log, unread. LangGraph saves a step's writes before the checkpoint that applies them but
  keeps a crashed step's pending writes; yian drops them, since the success is what accepts the
  commit.
- A log that ends inside its run start (before the START inputs' commit) can't be resumed, since
  it doesn't hold the run's inputs yet: `resumeFlow` refuses it with "run log PATH ends inside its
  run start; start the run again" and leaves it as it is (`replayRunLog` too; `inspectRun`
  without the flow reads it as `crashed` with nothing in flight).
- `runRecordOffsets(path)` (from `yian/run`) returns the byte offset where each record's line
  starts, by seq: a header scan (`iterRunRecords` with `payloads: false`) that takes no lock.
  Read the record at seq `k`, such as the commit a success names, with
  `iterRunRecords(path, { afterOffset: offsets[k] })`. Nothing beside the log caches the pool.

Temporal stamps every history event with an `eventTime`, LangGraph every checkpoint with a
`ts`, and Codex and Claude Code each session record with a `timestamp` of the same form; yian's
`at` follows them. The vocabulary adds
`RunStoppedRecord`: `reason` is `signal` when the host's abort signal fired or
`callback` when a host callback threw or rejected, `message` is the reason's text, and
`node_ids` lists the nodes in flight. It is not a terminal record. Every payload field
is required: `RunStartedRecord` writes `defaults` as `{}` when every value is built-in
and `call_time_llm` as `null` when no agent resolves at call time. Route payloads have
no `count` or `content_index`, and there is no `repeat` route kind. Values keep ints of
any size and `±Infinity`, written `{"$float": "inf"}` or `{"$float": "-inf"}`; NaN is
refused. Counters (`seq`, `schema`, `num_workers`, `attempt_id`, indexes) are
safe-integer `number`s in the API and JSON ints in the line. A damaged or foreign line
fails with a `RangeError`, or a `TypeError` for a field of the wrong JSON type; text
that isn't JSON, including a BOM or invalid UTF-8, fails with "invalid run-record JSON".
A malformed `value_json` inside a route keeps the reader's `JSONDecodeError`.

Backward attempts use `BackwardStartedRecord`, then one `BackwardReadyRecord`
after the complete backward fold and every memory proposal, then ordered
per-memory commit requests/observations and `BackwardFinishedRecord`. A known
failure before Ready can finish the attempt directly. Commit records require the
exact positive keyword-only `attempt_id`; there are no per-memory learning records or legacy
unattributed commits. Replay validates the same full inventory, routes, batches,
and attempt rules without invoking callbacks or learners.

The forward dispatcher and explicit backward driver use the same authority order:

```
prepare and validate record -> append record -> apply reducer -> publish observers
```

An observer therefore never claims a transition that was not first accepted by
the log. `JsonlRunLog` writes one canonical JSON record per line and fsyncs every
append while holding a nonblocking exclusive file lock. `MemoryRunLog` preserves
the same ordering without a durable path.

Changed (TS yian): the single-owner lock is a SQLite exclusive lock on the sibling file
`<log>.lock`, not a lock on the log itself. `JsonlRunLog.create` and `JsonlRunLog.resume`
take it without waiting and hold it for the log's lifetime, until `close()`; the kernel
releases it when the process dies. The lock follows symlinks: it sits beside the log's
resolved path, so every symlink to a log shares one lock, but a hard link to the log is a
different lock (Python's flock on the file excluded hard links as well), so don't run two
writers through hard links. The lock file is kept after `close()` and never deleted, so
its presence alone doesn't mean a writer is active. A lock another writer
holds raises an `Error` "run log <path> is locked by another writer (is the run still
active?)" with the busy `FileLockError` as its `cause`. Each append writes its line at
the known end of the complete prefix and fsyncs before the log keeps the record; a
write that makes no progress fails with "short run-log write: wrote K of N bytes". When
a write or fsync fails, the log truncates the file back to the prefix, fsyncs it, and
rethrows that error; from then on the log refuses appends with "run log <path> refuses
appends after a failed append", the original error as its `cause`, and keeps the lock
until `close()`. When that rollback fails too, the append raises an `AggregateError`
"run log <path> append failed and its rollback also failed" holding the append's error
and the rollback's errors, and the file may still hold part or all of the failed line: a
later resume drops a partial line but keeps a complete one as a record, even though its
fsync failed.
A record the envelope or the codec refuses raises its `RangeError` or `TypeError` before
anything is written and doesn't poison the log. A damaged record in a log being read
raises a `RangeError` naming the path, the 1-based line and the seq.

Each record is fsynced before it counts, in Python yian and in TS yian on Node and Bun, so a
durable run pays one sync per record. yian calls the runtime's plain `fsync` and never asks for
`F_FULLFSYNC`, as Python does: the records are safe if yian crashes or is killed, but on a power
loss the last step or two may be lost, and a resume redoes them. Changed (TS yian): on macOS,
Node's `fsyncSync` and `fdatasyncSync` are themselves `fcntl(F_FULLFSYNC)` (libuv's choice),
which also flushes the drive's cache, so Node is stricter and slower there than Bun, whose
`fsyncSync` is the plain `fsync` (Python's `os.fsync` is the same plain call, not measured here).
On an Apple M4 Pro MacBook Pro (Mac16,8, APFS), measured 2026-10: a 10,000-item MAP (with
`YIAN_MAX_TOTAL_NODES` raised past its default of 10,000) writes 60,011 records; Node 22 pays
about 4.2 ms per record and spends about 250 s in sync, 254 s in all against 4 s with a memory
log, serial or with 8 workers, while Bun pays about 52 µs per record, about 3 s in sync. A bare
100-byte append plus sync costs 3.7 to 3.9 ms on Node 22 and 26 and 28 to 35 µs on Bun.

!!! warning "Run logs may contain sensitive data"
    Canonical records retain caller inputs, node outputs (including human, tool,
    and model data), resolved model configuration, captured prompts/variables,
    feedback, memory snapshots and learning proposals, and failure tracebacks. The
    runtime does not redact or encrypt them. Put durable logs on an
    access-controlled path; do not commit or publish them.

Changed (TS yian): a `NodeStepRecord` is `{node_id, n, step}`. `n` numbers the node's steps
from 1 across its starts (a resumed node's steps continue after its earlier starts'), so
`node_id` + `n` names a step; a record whose `n` isn't the node's next is refused on append and
on replay. Each `step` is a JSON object with a string `kind`:

- `tool_attempt`: a TOOL's retry, `{event: "failed" | "started", attempt, ...}` (see
  [Durable recovery](#durable-recovery-with-resume_flow)).
- `llm_config`: a call-time AGENT's resolved config, `{effective_llm_config}`.
- `model_call`: an AGENT's model call, written once the call returns and before anything acts
  on its reply (before its tool calls run, and before an invalid reply fails the node):
  `purpose` (`chat`, `structured_native` or `structured_fallback`), `model`, `request_sha256`,
  `started_at` and `first_token_at` (record times; `first_token_at` is `null` before the first
  event), `output` (the reply as a `yian.message.v1` message), `response_id`, `finish` (the
  provider's finish reason, or `interrupted` for a reply a host interrupt cut) and `usage`
  (`{input_tokens, output_tokens, cache_read_tokens, reasoning_tokens}`, a count `null` when
  unreported; `null` without usage). A failed attempt is a `model_call` too, written whether or
  not it is retried, with `output` (what arrived, or `null`), `usage` (what the provider
  reported before the failure, or `null`), `error` (`{kind, message}`: a provider error's kind,
  such as `rate_limit`, else the error's class; the message scrubbed of every credential the call
  may have sent) and `retry_in` (the wait in seconds before the retry, `null` when it isn't
  retried) in place of `response_id` and `finish`. A call a stop or abort cut writes nothing.
  `request_sha256` is the SHA-256 of the request's messages (each message's `author` included),
  tools, tool choice, parallel tool calls and output format, as `logJson` text, with the steers
  the AGENT read left out. Media
  in a reply is never inline: its bytes are stored as `media/<sha256>.<ext>` beside the run's
  log, `output` holds the file reference, and the request (and its digest) holds that form;
  only the request sent to the provider has the bytes inline.
- `tool_result`: one tool call's result, written as that call finishes (calls run together
  write in finish order; the next request still lists results in call order) and before the
  next model call: `call_id` (the id of the reply's `tool_call` part), `started_at`, `output`
  and `is_error` (exactly the text and flag the model reads), and `closed: "skipped"` for a call
  a host interrupt skipped (`started_at` `null`, `output` `"Not run: interrupted by the user."`,
  `is_error` true). A call that threw, or whose result has no text, writes no result. A resume
  writes `closed: "crashed"` for a replayed call with no result (`started_at` `null`, `output`
  `"The tool call was interrupted; its outcome is unknown. Check its effect before retrying."`,
  `is_error` true); see [Durable recovery](#durable-recovery-with-resume_flow).

A tool call's idempotency key (`ToolContext.callId`) is `<run id>/<node>/<n>/<index>`, `n` the
reply's `model_call` step.

## Durable recovery with `resume_flow`

Start a recoverable run by supplying a new log path:

```python
from pathlib import Path

from yian import load_flow, run_flow
from yian.run import resume_flow

run_log = Path("run.jsonl")
loaded = load_flow('''
id: delayed-feedback
name: delayed_feedback
input:
  text: !str
nodes: {}
output: ${input.text}
''')

# In the original process. `run.jsonl` must not already exist.
result = run_flow(
    loaded,
    {"text": "A long explanation."},
    log=run_log,
    learnable=True,
)

# Safe in a later process, whether the prefix is terminal or unfinished.
recovered = resume_flow(loaded, log=run_log)
assert recovered.events == [] and recovered.backward_outcome is None
outcome = recovered.backward("Use fewer words.")
assert outcome.feedback.input_feedback == {"text": ("Use fewer words.",)}
assert recovered.output == "A long explanation."
```

The exact recovery surface is:

```python
resume_flow(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: Callable[[Any], None] | None = None,
    on_stream: Callable[[StreamChunk], None] | None = None,
    memory: Mapping[str, Memory] | None = None,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    api_key: str | SecretStr | None = None,
    headers: Mapping[str, str | SecretStr] | None = None,
    context: Any = None,
) -> RunResult
```

`api_key` re-supplies the key of a run started with `llm_config.api_key`. It goes to
the agents the log marks for it, at the `base_url` the log recorded for them if the
run had one, is never recorded, and stays with the returned result for `backward`.
A log that marks no agent ignores it. `headers` re-supplies the header values that
run sent with its key (`llm_config.headers`) the same way. Their names must be the
ones the log recorded, compared without case: missing or extra names raise
`ValueError` naming them, before anything is appended, and so does `api_key`
alone on a log that recorded header names. An agent that resolves its `llm_config`
when it runs (`RunStartedRecord.call_time_llm`) gets the key when the config it
resolves is the key's provider; a log whose call-time agents may resolve to it keeps
the key, and checks the header names, as for a marked agent. Resuming such a run
without the key raises `yian.llm_clients.auth.AuthError` when such an agent resolves
to the key's provider, before its model call; it is a host error, not a failure of the node, so resume again
with the key. `context` re-supplies the host's app
object; the log does not hold it (see [Tool context](#tool-context)).

Changed (TS yian): `resumeFlow({ log, numWorkers, onEvent, onStream, memory, tools, skills,
steering, apiKey, headers, context, signal, env, codeRoot })` returns a Promise of the
`RunResult`, and the
errors are its rejections. `memory` and `skills` are `Map`s and `tools` a plain record, as for
`runFlow`. `signal` is the host's stop: aborting it stops the resumed segment, which rejects with
the signal's reason and records a `RunStoppedRecord`, and an already-aborted signal rejects before
the log is locked. `env` is the environment the resume reads connections, credentials and its
`YIAN_*` values from; the run's recorded values must match it. `scheduler`, `random` and `models`
are for tests, as for `runFlow`. Header names that aren't the recorded ones, a negative
`numWorkers` and an invalid log prefix raise `RangeError`, not `ValueError`.

Changed (TS yian): `resumeFlow({ log, flow, ... })` takes no flow argument. It compiles the
flow files `RunStarted` recorded, never the files on disk now, so an edited flow file means a
new run; the fingerprint is still checked, against the compiled recorded flow. The load
(`loadFlowSource(source)`, which takes a loaded flow's `source` or a run's recorded `flow`)
starts at the root and follows each file's `refs` inside the log: no disk, search path or
symlink. A CODE reference's file is not recorded and must still be there, in the folder of
the file's key below `dir`. CODE files are read from disk on resume, so an edited CODE file is picked up, as a
reference-mode CODE node's function is today. `resumeFlow({ log, codeRoot })` runs
`loadFlow`'s `codeRoot` check over every recorded file, a host child's included: a CODE
reference whose file's real path is now outside `codeRoot` (say, the file or its folder was
replaced by a symlink pointing out) raises `LoadError` before anything is appended, even for
a finished log. A run whose flow was built in code ignores `codeRoot`: the host confines that
flow when it loads it. `loadRecordedFlow(log, { env, codeRoot })` (from `yian/run`) compiles a
log's recorded flow
the same way without locking it, for a host that needs the flow before it resumes (the CLI
reads its skills and MCP needs). A run whose flow was built in code records `flow: null` and is
resumed with `resumeFlow({ log, flow })`, the flow fingerprint-checked as before; passing `flow`
for a run with recorded files raises `RangeError`. For an unfinished run, once the MCP servers
are open and before anything is appended, the description and parameters of each tool an AGENT
grants, and the description and instructions of each granted skill, must be the recorded ones:
a difference, a missing one or an added one raises `RangeError`
(`resume refused: tools.lookup description differs from the recorded one`); a finished log
isn't checked. A log another yian version started resumes with a warning naming both versions,
through `warn`.

Recovery locks the existing file before reading it and validates the complete
record prefix; a file whose complete lines are not a valid run log is rejected and
left unchanged. Only then does it durably truncate any bytes after its final
newline, and it checks that the supplied flow has the recorded fingerprint. Before the
fingerprint, it checks that this process's [defaults and limits](#defaults-and-limits)
equal the ones the run started with; a difference raises `DefaultsMismatchError` (a
`ValueError`) with one line per variable, such as `YIAN_MAX_TOTAL_NODES is 20000 here,
but this run started with 5000; set YIAN_MAX_TOTAL_NODES=5000 to resume it`, or
`... started with the default 10000; unset YIAN_MAX_TOTAL_NODES to resume it`. A log
that recorded an entry this version doesn't know can't be resumed by it. A log
another writer still holds raises `BlockingIOError` naming the path; a damaged
record raises `ValueError` naming the path, its 1-based line, and the expected seq.
A log holding an int longer than 4300 digits is rejected the same way and left
unchanged; the message ends with `run-record JSON: integer with N digits is larger
than yian allows (4300 digits); a log written while yian accepted such ints can't be
read`.
It reconstructs the typed pool,
dynamic expansions, original input, run identity, diagnostics, and snapshotted LLM
configuration without reading the current LLM defaults. An agent that resolves its
`llm_config` when it runs uses the defaults and saved logins the log recorded. One that
had started before the crash recorded the config it resolved; `resume_flow` checks
that config as it checks the recorded map, and the restarted node must resolve to the
same config. A different one means the flow or yian version changed, and
`resume_flow` raises `ValueError` naming the node (`node 'chat' llm_config: recorded
config differs from the recomputed one; ...`) before any model call; the node does
not fail. A run started before the
AGENT `mode:` field was removed cannot be resumed: the AGENT manifest changed, so
every AGENT flow has a new fingerprint.

Changed (TS yian): `JsonlRunLog.resume(path)` opens the log, locks `<log>.lock`, reads
the file through the descriptor it holds, and validates the complete prefix; only then
does it truncate and fsync an incomplete tail. A missing log throws the `fs` error with
`code` `ENOENT` and creates no lock file. A log another writer still holds raises an
`Error` naming the path, with a `FileLockError` as its `cause`, not `BlockingIOError`
(see [Authoritative run logs](#authoritative-run-logs)). A damaged record raises a
`RangeError`, not `ValueError`, reading `run log <path> line N (seq S): ...` with the
decoder's error as its `cause`; a file without a complete line raises `cannot resume an
empty run log: <path>`. There is no int digit limit, so no log is rejected for a long
int.

The fingerprint covers the flow as written: its nodes, wiring, types, prompts,
inline `code:` bodies, granted tool, group, and skill names, and static child flows.
It covers every value written in the flow, including literals under any name and URL
or connection text. Nothing is redacted, so editing any literal fails the fingerprint
check. Since a flow file is recorded in full, don't write a secret in it (see
[Secrets stay out of flow files](syntax.md#secrets-stay-out-of-flow-files)): pass a
model key as `run_flow(..., llm_config={"provider": ..., "api_key": ...})`, and
other credentials as input values (which the run log records, as the warning above
says) or through host tools.
It does not cover what the host supplies. A reference-mode CODE node
(`code: pkg.mod:fn`) contributes only its `module:function` text, not the
function's source, so editing that function between a crash and `resume_flow`
passes the check and the restarted nodes run the new code. Tool implementations,
skill contents, and model behavior are likewise outside it. Keep them unchanged
until the run finishes, or start a fresh run.

Changed (TS yian): the fingerprint is SHA-256 over the RFC 8785 (JCS) text of TS
yian's own manifest, so it differs from Python yian's for the same flow; a
reference-mode CODE node contributes its `./file.ts:fn` text, a tagged value
entry contributes the built-in identity operation, and the directory the flow
was loaded from isn't covered. A flow's CODE runs when the flow runs, never when it
is loaded or checked (`yian check`); run only flows you trust. It runs on worker threads
of the host's process with the host's privileges, not in a sandbox, and a thread imports
a referenced file when it first calls it, so an edited file may apply to later calls (on
a new thread) even within one run.

If the original run enabled memory and forward execution is unfinished,
recovery requires backends under the same memory keys. Each session
must reproduce the exact snapshot stored by `RunStarted`; this prevents a resumed
run from reading a different memory generation. A `MemoryRegistry` built for
another flow, with a different type for some key, fails with
`MemorySetupError`. Every terminal run is inspected
silently without opening or validating memory, even after an interrupted backward
attempt. A supplied mapping is retained for a later explicit backward call.

For later backward on a memory-enabled log, pass a mapping with the same memory
keys, for example `build_memories(flow, root=...)` with the original root, to
`resume_flow(..., memory=...)`; terminal inspection retains it without opening it.

The log does not record `tools` or `context`. To continue an unfinished run whose
nodes use per-run tools or ask the human, pass the same `tools` (including
`ask_user`) and `context` to `resume_flow` again. Tool group contents may differ
from the original run's: completed nodes keep their recorded results, and nodes
that restart see the groups passed to `resume_flow`. Every grant must still resolve, as in `run_flow`.
Pass an MCP server again the same way. An unopened `McpServer` is opened only when
the run is unfinished, after the log is replayed and before the tool checks, and is
closed when `resume_flow` returns or raises; terminal inspection opens none.
Opening raises `McpConnectError` or `ImportError` as in `run_flow` (see
[MCP servers](#mcp-servers)).

If primary execution is unfinished and eligible for retry, recovery appends one
`RunResumedRecord`, retries the exact ordered set of nodes that have a durable start but no durable
outcome, and continues the ready frontier. `num_workers=None` preserves the
recorded worker count; an integer selects and records the new segment's count.
Fresh inputs, `run_id`, and LLM configuration cannot be supplied during recovery.

A retried node may resend requests and repeat tool actions; recovery does not
provide exactly-once execution or resume an external request midway through. A
retried TOOL node passes its tool the same `ctx.call_id` as before, so a tool can
use it as an idempotency key; a retried AGENT's tool calls get new ids (see
[Tool context](#tool-context); TS yian replays an AGENT's recorded calls, below). A TOOL node passes the next `ctx.attempt`; one with
`retries:` keeps the retries it had left (see [TOOL retries](#tool-retries)). The new
accepted start receives a new sequence, and metadata for its old unfinished start is
rejected.
Runtime neither reconstructs a transcript from the run log nor fabricates results
for unknown effects. SessionMemory owns whatever conversation history was saved;
the run log is not a durable replacement for that memory backend.

Changed (TS yian): there is no session memory, so nothing keeps a conversation beside the
run log; a chat's history is the value a LOOP carries, recorded as node outputs. Steers aren't
recorded: a steer an AGENT read before a crash is lost (the replayed requests don't hold it, and
`request_sha256` leaves steers out so they still match), and one queued while the run was down
joins the first live model call.

Changed (TS yian): an AGENT records each model call and each tool result as a node step (see
[Authoritative run logs](#authoritative-run-logs)), and a resumed AGENT replays them instead of
repeating them. It rebuilds each request as before and, for each recorded call in order, takes
the recorded reply back without calling the model or publishing anything, once the recorded
`purpose` and `request_sha256` match the rebuilt ones. A tool call with a recorded result gets
that result without running. A replayed call with no result (a crash or a stop cut it off) is
checked as live and, when its check passes, closed with a `tool_result` written as
`closed: "crashed"`, which the model reads as "outcome unknown"; it isn't run again. A recorded
reply a host interrupt cut, or recorded `skipped` results, end the AGENT as the interrupt did; a
recorded call that failed for good fails the node again with its recorded error. Failed attempts
after the last recorded call belong to the first live call: it waits the last one's `retry_in`
again and its attempts continue their count toward `YIAN_MODEL_RETRIES`. The AGENT reads its
host inbox (steers and the interrupt) only once the replay ends. New steps are numbered after
the recorded ones; a call a crash cut before its `model_call` was written ran no tool, so no
idempotency key names two calls that ran.

A recorded call whose `purpose` or `request_sha256` differs from the rebuilt request (another
host, tool set or yian version builds a different request) is a host fault: the resume rejects
with a `RangeError` naming the node, the step's `n` and both values, and writes nothing for the
AGENT beyond its new start, so the run stays resumable with the host that started it.

After incomplete-tail repair, terminal inspection appends no records and invokes
no graph nodes, backward hooks, or memory learners/commits. It projects the last
accepted attempt, including an interrupted Started/Ready/request prefix.
`RunResult.events` contains only forward observers published in the current call;
backward records emit no observers. A resumed AGENT that re-executes publishes
fresh in-loop observers to `on_event`, and nothing is replayed. (TS yian: only for the calls it
makes live; replayed calls publish nothing.)

A retained result keeps private program/data copies and memory references, not
open forward sessions, workers, or a log lock. An explicit backward call reopens
and validates its authority, restores fresh memory sessions when needed, and
closes resources afterward. The durable locator is anchored when acquired, so
later working-directory changes cannot redirect the result. A nonblocking guard
rejects overlapping calls on one result, and durable log locks exclude other
handles. Invalid submissions preserve the caller's last observed outcome; a
prefix-refresh failure makes that handle unusable rather than permitting a stale
retry. Log/cleanup faults preserve the primary exception and accepted records.

!!! important "Recovery may retry an interrupted node"
    An eligible durable `NodeStarted` with no durable outcome is run again.
    Callers using the optional SessionMemory turn capability consume the backend's
    latest saved history, but may resend requests or repeat external effects.
    The log makes engine state deterministic, but cannot make an interrupted
    external effect exactly-once.

Changed (TS yian): right after `RunResumed`, a resume publishes one `NodeAttemptResumed(node_id,
crashed)` event per node it runs again, in retry order. `crashed` counts that node's
crashed attempts, including the one this resume retries: starts with no outcome before a
resume, unless a `RunStoppedRecord` covered them. A host stop or a failing host callback records
that stop before the run winds down, so only a crash (or a host fault that isn't a callback)
counts. Nodes running beside a crashing node are counted too: the crash interrupts every start in
flight, not only the one that was running when the process died. A stop adds nothing to the
count of the nodes it covered (a node that never crashed resumes with `crashed` 0). yian sets no
limit; a host that resumes on its own decides how many crashes to tolerate.
`RunResult.crashedAttempts` and `inspectRun` hold the same counts.

Changed (TS yian): a resume keeps the recorded times and stamps its own records, from its
`RunResumed` on, from its `scheduler`'s clock; replay ignores `at`, so a log replays the same
whatever times it holds (see [Authoritative run logs](#authoritative-run-logs)).

Changed (TS yian): the log also holds `RunStoppedRecord` (see [Authoritative run
logs](#authoritative-run-logs)): when the host's signal aborts a run or a resume, or a host
callback throws or rejects, the engine lets the nodes in flight wind down and appends one record
naming them, with `reason` `signal` or `callback`. It isn't terminal: the stopped nodes are still
in flight, a resume retries them after its `RunResumed`, and they aren't counted as crashed.
Between the stop and that resume only those nodes' steps are appended.

### Reading a run log with `inspectRun`

Changed (TS yian): `inspectRun(log, { flow, env })` reads where the run in a log stands without
changing the file or taking its lock, and returns `{ status, inFlight, crashedAttempts }`.
`status` is `running` while a writer holds the log's lock, `succeeded` or `failed` once the run
finished, `stopped` when the host stopped its last segment (a `RunStoppedRecord`), and otherwise
`crashed`, the process having ended without a stop; a run that is neither running nor finished
can be resumed. `inFlight` lists the nodes started and not settled, in start order (`[]` for a
finished run), and `crashedAttempts` the attempts a resume counted after a crash, plus, for a
run that is neither running nor finished, one for each node in flight that no stop covers: what
`resumeFlow` would count. With `flow`, the log is replayed as `resumeFlow` replays it, so the status
is exact, and `env`'s `YIAN_*` values must match the recorded ones. Without it, the status is read
from the records, which don't hold the ready nodes: a run whose process ended after the top-level
END succeeded (or after a route proved END unreachable) but before a node already ready started
reads as finished, though `resumeFlow` would still run that node. Pass the flow when you have it.
The result also holds `runId`, `schema`, `flowFingerprint` and `learnable` from the log's
`RunStarted`, `lastSeq`, the seq of its last complete record, and `failure`, the `Failure` that
ended a `failed` run (a node's failure, or the `graph.terminal_unreachable` failure of a route that
proved END unreachable; `null` for any other status), so a host can list and label runs without
reading records. It also holds `startedAt`, the `at` of `RunStarted`, and `endedAt`, the `at` of the
log's last record for a `succeeded` or `failed` run (`null` for any other status; a later backward
attempt's records move it). The first five and `startedAt` are `null` only for a running log with no record yet. Without
`flow`, the records are streamed with `iterRunRecords`, so memory is bounded by the largest
record. An empty log with no writer raises `RangeError`, as does a log whose first record isn't
`RunStarted`; a missing one raises the `fs` error with `code` `ENOENT`, and a schema this version
doesn't read a `RunLogSchemaError`. When a writer holding the lock rolls back a failed append under
the read, the log is read once more and reads as `running` (a failed append poisons the writer,
so it cuts nothing more); with no writer, the shrink raises `RunLogChangedError`.

### Following a run log with `iterRunRecords`

Changed (TS yian): `iterRunRecords(log, { afterOffset, runId, afterSeq, kinds, payloads })` (from `yian/run`)
streams the records of a run log another process may still be writing. It never takes the log's
lock or writes the file, so the writer is never blocked. It yields one entry per record:

| Field | Meaning |
|---|---|
| `seq` | the record's sequence number |
| `offset` | the byte offset where its line starts |
| `end` | the byte offset just past its newline: pass it as `afterOffset` to read on from there |
| `runId` | the run id the record names |
| `at` | when the log appended it, as `RecordEnvelope.at`, such as `2026-10-20T14:02:11.123Z` (also with `payloads: false`) |
| `kind` | its wire kind, such as `node_started` |
| `nodeId` | the node it is about (a node's start, step, pool commit, success or route, a growth's spawner, a failure's node), or `null` |
| `envelope` | the validated `RecordEnvelope`, or `null` with `payloads: false` |

```ts
// Saved between calls: the last record read, or null to read from the start.
let last: { end: number; runId: string; seq: number } | null = null;
const after = last === null ? {} : { afterOffset: last.end, runId: last.runId, afterSeq: last.seq };
for (const entry of iterRunRecords(log, { ...after, payloads: false })) {
  show(entry.seq, entry.kind, entry.nodeId);
  last = { end: entry.end, runId: entry.runId, seq: entry.seq };
}
// Later, the same loop from `last` reads only what was appended since. On a
// RunLogChangedError, set `last` to null and read the log again from the start.
```

- It reads the file in 64 KiB chunks, so memory is bounded by the largest line. It is a generator:
  it opens the file, and throws a bad option or a missing file (`ENOENT`), when the first entry is
  asked for, not at the call, and closes the file when the iteration ends.
- It stops at the end of the file. An unterminated last line, which the writer may still be
  finishing, isn't yielded; a later call from the last entry's `end` reads it once it's complete.
- `afterOffset` (default 0) must be the end of a line. From the start, each record's `seq` must be
  its line's index; after an offset, `afterSeq + 1` (or, without it, the first record read) sets
  the count. Every record must name the first one's run id. Hosts save `{ end, runId, seq }` of
  the last entry and pass them as `afterOffset`, `runId` and `afterSeq`; the two describe the
  record that ends at `afterOffset`, so they need one.
- A file shorter than where the reader got to (`afterOffset`, or the end of the last line read)
  raises `RunLogChangedError` (a `RangeError`) "run log PATH shrank to N bytes under its reader at
  byte M; it was truncated or replaced". With `runId` or `afterSeq`, an offset that no longer ends
  a line, or a first record that isn't seq `afterSeq + 1` of run `runId`, raises it too ("run log
  PATH doesn't continue the records read before byte M: ..."). Read the log again from 0 then.
- A record yielded while a writer runs can still be rolled back by that writer's failed append.
  The next call raises `RunLogChangedError` if the file hasn't since grown past the offset;
  otherwise the offset may no longer be a line end, or may fall among other records, which only
  `runId` and `afterSeq` catch. Another run's log renamed over the path is caught the same way.
- `payloads: false` checks each line's envelope (schema, run id, seq, time, kind) and node id without
  validating the payload, and yields `envelope: null`. `kinds` yields only records of those kinds,
  checking the others the same way; a kind no record has raises `RangeError`.
- It reads every schema `resumeFlow` reads (schema 15); a line of another schema raises
  `RunLogSchemaError` naming the line. A damaged line raises `RangeError` naming the log and the
  line (after an offset, the byte), with the underlying error as `cause`.

### Reading values from a run log

Changed (TS yian): for hosts that read run-log values, `yian/run` exports the codec the log writes
them with: `TypedValue` and `FileRef`, `logJson`, `parseJson` and `JSONDecodeError`, and the types
`StorageKind`, `Value`, `ValueRecord`, `JsonValue`, `JsonObject` and `JsonMode`. A `pool_commit`
payload is flat, `{node_id, key, kind, value}`, and its `kind` and `value` are the typed value's
wire form. `TypedValue.fromWire({ kind, value })` decodes it strictly: the object must hold exactly
`kind` and `value`, `kind` must be a storage kind (not `Any`) and `value` of that kind, or it throws
a `RangeError`. It checks the value's top level and a typed list's items, not nested contents, which
it trusts `parseJson` to have read. Kind `file` gives a `FileRef`, and a `list[float]` keeps its int
items as `bigint`s.
`typed.toWire()` gives `{kind, value}` back, each `FileRef` written as its `{uri, mime, name}`
mapping.

`logJson(value)` writes JSON as a run-log line does: compact, keys in UTF-16 code-unit order, a
`bigint` as an int of any size, a `number` as a float (`.0` added when it has no `.` or exponent),
`±Infinity` as `{"$float": "inf"}` or `{"$float": "-inf"}`, and a record shaped like a tag (exactly
`{"$float": "inf"}` or `{"$float": "-inf"}`, or with `$record` as its only key) wrapped as
`{"$record": …}`. NaN, `-0`, text holding a lone surrogate and anything else that isn't JSON throw
`TypeError`. `parseJson(text, "codec")` reads that text back: an int as a `bigint`, any other number
as a `number`, the tags as `±Infinity` and a `{"$record": …}` escape as the record inside it;
`parseJson(text, "plain")` decodes neither. Text yian doesn't read (not JSON, a `NaN` or `Infinity`
token, a float that overflows, a duplicate key, a lone surrogate) throws `JSONDecodeError`, whose
`reason`, `line`, `column` and `position` say where. This is the codec the log writer,
`VariablePool.dumps` and `yian run --json` use, so re-encoding a decoded `pool_commit` gives the
log's bytes: `logJson({ ...payload, ...typed.toWire() })` is the payload as the line holds it.

```ts
import { iterRunRecords, logJson, parseJson, TypedValue } from "@ngocbh/yian/run";

for (const { envelope } of iterRunRecords(log, { kinds: ["pool_commit"] })) {
  const { node_id, key, kind, value } = envelope.payload;
  const typed = TypedValue.fromWire({ kind, value }); // an int is a bigint, a file a FileRef
  show(node_id, key, typed.kind, typed.value);
  const text = logJson(typed.toWire()); // {"kind":…,"value":…}, the value as the log writes it
  const back = TypedValue.fromWire(parseJson(text, "codec")); // the same value again
}
```

The `json` conformance group (`conformance/cases/json/`) specifies the form, for example
`log_number_form`, `log_infinity_and_record_escape` and `read_codec_record_escape`.

- The `TypedValue` constructor doesn't check that the value fits the kind; `fromWire` is the checked
  way to build one.
- `toWire` throws `TypeCheckError` (a `RangeError`, not exported) for a value that isn't JSON, which
  the unchecked constructor, or `fromWire` given data `parseJson` didn't make, can hold.
- A `FileRef` appears only for kind `file`; it holds only `uri`, `mime` and `name`, and yian stores
  no file content.
- `logJson` writes every JS `number` as a float (`{a: 1}` gives `{"a":1.0}`), so pass ints as
  `bigint`s; it refuses `undefined` and a `Date`.
- Plain `JSON.parse` reads `logJson` text, but loses ints past 2^53 and reads `±Infinity` as the
  tag object and an escaped record with its `$record` wrapper.
- `yian/run` imports `node:` modules, so it is for server-side hosts, not browsers.

No Python counterpart: Python yian's run log refuses infinity and writes no tags, so Python hosts
read values with `json` and `yian.typesys.TypedValue`.

### Naming a run's nodes with `staticNodes`

Changed (TS yian): `staticNodes(flow)` (from `yian/run`) returns `[path, node]` for every node of a
loaded flow (a `ComposedFlow` or its compiled graph) and of its child flows, parents before their
children. A path joins node ids with `/`: `call/write` is node `write` of the child flow node
`call` runs. These are the paths `RunStarted` keys `effective_llm_by_static_path` by and the node
ids a log holds for every node the graph has from the start (a growth's nodes get ids of their
own), so a host reads the kind of a logged node as `node.kind`. The nodes are the flow's own; read
them, don't change them.

A node a run grows has an id of its own: MAP element `i` of node `each` runs its child's nodes as
`each#i/...`, and LOOP `node` runs iteration `k`'s body as `node#k/...` and its driver as `node~k`.
`staticPathOf(nodeId)` (from `yian/run`) maps such an id onto its `staticNodes` path by dropping
each `/` segment's `#<i>` and `~<k>` suffixes: `each#1/echo` is `each/echo`, `node~2` is `node`.
Authored node ids are identifiers, so neither suffix is ever part of one.

## Defaults and limits

Every predefined default and limit is an entry in `yian.defaults`, overridable by a
`YIAN_<NAME>` environment variable; [Defaults and limits](defaults.md) lists them all.

```python
from yian.defaults import DefaultsError, DefaultsMismatchError, current

current().max_total_nodes  # 10000 unless YIAN_MAX_TOTAL_NODES sets it
```

- `current()` returns this process's frozen `Defaults`, one lower-case attribute per
  entry. It reads and checks every variable the first time it is called and returns
  the same values afterwards, so set the variables before the first yian call;
  changing `os.environ` later has no effect. `load_flow`, `run_flow`, and
  `resume_flow` call it first.
- `DefaultsError` (a `ValueError`) is raised for a variable that is not a plain
  decimal integer in its range, naming the variable, the value, and the range. It is
  a host setup error, not a `LoadError`.
- `DefaultsMismatchError` (a `ValueError`) is raised by `resume_flow` and by
  `RunResult.backward`'s replay when the run's recorded values differ from this
  process's; see [Durable recovery](#durable-recovery-with-resume_flow).

Changed (TS yian): the `env` option of `loadFlow` and `runFlow` sets the `YIAN_*` values for
that load or run: `current()` returns the values read from it for all of its work, so concurrent
loads and runs with different environments each keep their own. `withEnv(env, fn)` (from
`ts/src/defaults.ts`) does the same for any code; `withEnv(processEnv, fn)` keeps the values
already in effect. Elsewhere `current()` returns the process values, read once.

Changed: these module constants were removed; set the variable instead.

| Removed | Variable (default) |
|---------|--------------------|
| `yian.typesys.values.MAX_VALUE_NESTING` | `YIAN_MAX_VALUE_NESTING` (100) |
| `yian.expr.MAX_EXPRESSION_NESTING` | `YIAN_MAX_VALUE_NESTING` (100); expressions share the value bound |
| `yian.compose.source.scalars.MAX_YAML_NESTING` | `YIAN_MAX_YAML_NESTING` (200) |
| `yian.compose.nodes.resolver.MAX_CALL_NESTING` | `YIAN_MAX_CALL_NESTING` (50) |
| `yian.runtime.state.MAX_TOTAL_NODES` | `YIAN_MAX_TOTAL_NODES` (10,000) |
| `yian.nodes.agent.loop.MAX_TOOL_ITERATIONS` | `YIAN_AGENT_MAX_TOOL_ITERATIONS` (-1, no cap) |
| `yian.skills.catalog.MAX_SKILL_FILE_BYTES` | `YIAN_MAX_SKILL_FILE_BYTES` (262,144) |
| `yian.skills.tool.MAX_LISTED_DESCRIPTION` | `YIAN_MAX_SKILL_DESCRIPTION_CHARS` (1,024) |
| `yian.cli.utils.MAX_KEY_PROMPTS` | `YIAN_MAX_KEY_PROMPTS` (3) |

## CLI path policy

The Python API remains path-neutral: `run_flow(..., log=None)` uses memory and a
caller chooses any durable path explicitly. The `yian` host adds a user-facing
policy on top:

- fresh `yian run FLOW` selects and reports
  `~/.yian/runs/<run_id>.jsonl` unless `--log PATH` selects another
  path; once runtime commits `RunStarted`, that file is authoritative;
- `yian run FLOW --resume PATH` calls the same recovery boundary without accepting
  fresh inputs, a second log path, or provider/model configuration, and checks
  the recorded models' provider keys before an unfinished run appends anything;
- omitting `-w` on resume preserves the recorded worker count, while an explicit
  value affects the new segment only;
- `yian run` builds every declared built-in memory with `build_memories` and
  closes the registry after execution drains. `--memory-root DIR` keeps files
  under `DIR/<flow id>/`, and repeated `--memory-storage KEY=PATH` places one
  memory; both apply on fresh runs and on resume, and other memories stay in RAM.
  A flow that declares a custom memory type exits 1 with a setup error, also on
  `--resume`. Terminal inspection opens no memory backend;
- each `yian chat` attempt, including a failed-turn restart, receives a separate
  reported default path.

The default CLI directories are mode `0700`; `JsonlRunLog` creates files with
mode `0600`. See [the CLI guide](cli.md#durable-logs-and-recovery) for commands
and the log-sensitivity warning.

Changed (TS yian): `JsonlRunLog` also creates the empty lock file `<log>.lock` beside
the log when it is missing, with mode `0600` narrowed by the umask, and leaves it in
place.

Changed (TS yian): a run gets a folder of its own. Fresh `yian run FLOW` writes
`~/.yian/runs/<run_id>/log.jsonl`, making the run's folder with mode `0700`. A library
host gets the same layout under its own runs folder from `runLogPath(runsDir, runId)`
(`@ngocbh/yian/run`), which makes the new folder `<runsDir>/<runId>` (an existing one
is refused) and returns its `log.jsonl` path. A run's folder is the folder that holds its log, and the files the run writes
beside the log (its lock file and media files) go only there. A host that passes a `log` path
of its own must keep one run per folder; Yian doesn't check it.

Changed (TS yian): a `yian.message.v1` media part may hold a file reference,
`{"type": "media", "mime": ..., "file": {"uri": ..., "mime": ..., "name": ...}}`, in
place of `data` or `url`. A run's media files live in `<run folder>/media/<sha256>.<ext>`
(mode `0600`, the `media` folder `0700`), and a reference names one by its path relative to
the run folder (`media/<sha256>.<ext>`), so a run folder copied elsewhere keeps its
references. A request sent to a provider gets the bytes inline again, read from the run
folder; a `uri` that names no path inside the run's `media` folder, as written or with
symlinks followed (a symlinked `media` folder included), is refused. Nothing writes media files yet: the AGENT will store the media in its replies this
way.

Changed (TS yian): a `yian.message.v1` message may carry `author`, a non-empty string naming who
wrote it when that isn't the flow's user or the model (a person or an agent). Any role may carry
it, but adapters send only a user message's author. On every route the message's parts are sent
inside a closed `<authored_message author="NAME">` … `</authored_message>` wrapper, `NAME`
XML-escaped; on Chat Completions the tags are folded into the first and last text, so a one-text
message is still a string. Chat Completions `name` is not sent yet. An unauthored user message
is not wrapped: text outside every wrapper is the flow's own user. An authored message with no
text goes unchanged. In every user and tool text, each opening of the tag (`<authored_message` or
`</authored_message`, in any case, and with whitespace, `\`, `/` or invisible format characters
after `<`, so `<\/authored_message>` too) has its `<` escaped as `&lt;`, and Gemini escapes the
tool JSON it decodes as well, so a message body or a tool's output can't forge a wrapper. Since
user text is always escaped, a flow that wants to explain the `<authored_message>` convention to
the model must do so in its system prompt. Not covered: lookalike tags (a fullwidth `＜`, a
character inside the name, `authored-message`), a model that reads `&lt;` as `<`, `\u003c`-style
escapes in tool output sent as text (no literal `<` reaches the model), and system prompts,
which are never escaped, so a flow must not put untrusted text there.

## Blocking human interaction

An authored `human_input` node and an AGENT node granted `tools: [ask_user]` call the
run's `ask_user` handler synchronously. A library host passes the handler with each
run as `tools={"ask_user": handler}`; the calling node remains active until the
handler returns or raises:

```python
from yian import load_flow, run_flow

approval = load_flow(
    open("examples/human-approval.yaml").read(),
    search_paths=["examples"],
)

def ask(**request):
    return input(request["question"] + " ")

result = run_flow(
    approval,
    {"task": "Plan a team offsite"},
    llm_config={"provider": "ollama", "model": "llama3.2:3b"},
    tools={"ask_user": ask},
)
```

The handler receives `question`, `questions`, `answer_schema`, `title`,
`node_id`, and `options` as keyword arguments, so define it as
`def handler(**request)`, or as
`def handler(ctx: ToolContext[App], **request)` to receive the call's
[tool context](#tool-context) first. A handler that cannot take all six is
rejected before the run starts. `human_input` uses the trusted metadata path;
model-requested `ask_user` calls can supply only the public `question` and
`options` arguments. `options` is `None` or a cleaned list of up to six short
answer labels the model suggested. Show them as choices, but still let the human
type another answer. Each run has its own handler, so concurrent runs in one
process never share one. Within one run, the default `num_workers=8` lets several
nodes (for example two MAP items) call the handler at the same time, so the
handler must be thread-safe.
If another node fails the run while a handler is waiting, `on_event` receives
`NodeFailed` and `RunFailed` right away, before the run waits for its in-flight
nodes. `run_flow` returns only after every pending handler returns or raises, and
it discards their answers. To stop waiting on a human, withdraw the prompt when
`on_event` sees the failure and make the handler return.
For a `human_input` node, `node_id` is the expanded node id (for example
`loop#1/ask` for a LOOP's second iteration), so a run id plus `node_id` identifies
one question; a handler that takes `ctx` reads the run id as `ctx.run_id`.
A model-requested call sets only `question` and `options`; `questions`,
`answer_schema`, `title`, and `node_id` are `None`. Only a handler that takes
`ctx` can tell which AGENT, or which LOOP or MAP occurrence of it, asked:
`ctx.node` is its expanded node id. The mapping is the whole tool set,
so a flow that also uses registered tools passes
`tools={**registered_tools(), "ask_user": ask}`. `ask_user` never comes from the
global registry: a run without its own handler fails any HUMAN_INPUT node that
asks, a flow whose AGENT grants `ask_user` fails before the run starts, and
`register_tool("ask_user")` is rejected.

Changed (TS yian): the handler is `(request, ctx) => answer`, always given the call's
`ToolContext` second, and may return a Promise. `request` is camelCase, `{ question,
options, questions, answerSchema, title, nodeId }`, with `null` for an absent field;
`invokeTrusted("ask_user", ...)` takes the same keys. Each item of `questions` is a parsed
spec `{ question, header, options: [{ label, description }], multiSelect }` with defaults
filled. There is no signature check: any function is accepted as the handler.

CODE nodes cannot ask. A CODE function does not see the run's tools, so
`invoke_tool("ask_user", ...)` inside it fails. Ask through a `human_input` node,
or pass a per-run tool and call it from a TOOL node.

An authored WAIT needs no host handler. It blocks one worker until its absolute
UTC `until:` deadline, returns `None`, and can be woken cooperatively when the host
stops the run. Changed (TS yian): the wait re-reads the wall clock at least hourly,
so a deadline survives a sleeping machine.

## `RunResult`

The terminal outcome of a fresh or recovered run exposes these public fields:

| Field | Meaning |
|-------|---------|
| `input` | canonical, coerced caller input retained from `RunStarted` |
| `status` | exactly `"succeeded"` or `"failed"` |
| `run_id` | durable identity shared by the state and log |
| `log` | durable JSONL `Path`, or `None` for a process-local fresh run |
| `output` | terminal flow value on success; otherwise `None`. Each position the flow's inferred output type marks `file` holds a `FileRef`, as `RunSucceeded.output` does; untyped files stay mappings |
| `failure` | canonical `Failure` on failure; otherwise `None` |
| `events` | post-durable observer events published during this call only; in-loop AGENT observers are excluded |
| `learning` | learning projection of the last observed backward outcome; `None` before an attempt is observed or when the run had no memory |
| `backward_outcome` | last observed accepted `BackwardOutcome`, or `None` before an attempt is observed |

Changed (TS yian): the fields are camelCase (`runId`, `backwardOutcome`), and `log` is the path
as given to `runFlow`, a string, not resolved. `crashedAttempts` maps each node id to the
number of its attempts a resume retried with no recorded stop (as `NodeAttemptResumed` counts
them); it is empty for a run never resumed.

Call `backward(feedback=None)` explicitly on a retained result. It changes only
the observed backward/learning fields; forward status, values, failure, and events
keep their original meaning.

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `yian` CLI](cli.md) — run flows from the terminal.
