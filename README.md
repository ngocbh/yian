# Yian - The Agent Composer

[![Documentation](https://readthedocs.org/projects/yian/badge/?version=latest)](https://yian.readthedocs.io)

Yian is a small language and runtime for readable agent flows. You
author a fixed graph in Docker-Compose-shaped YAML; the framework infers data
dependencies, applies explicit ordering constraints, validates types, and runs
through an append-only log.

## Goals

- Workflows that repeat, must be reliable, and should improve across runs, such as SOPs and
  trading routines.
- Not a replacement for interactive agent harnesses. Flexible or one-off work belongs in
  Codex, Claude Code, or OpenCode.

In this framework, an **agent** is a composable transformation from a named
input record to an output value. It can be represented by one node, a flow of
nodes, or a child flow used inside another flow. The YAML **AGENT node**
(`kind: agent`) is one concrete implementation: an LLM-backed node with a task
prompt and optional tool grants, model configuration, and long-term memory.

The author fixes the graph. LLMs generate values and may choose among explicitly
granted tools, but they do not rewrite the flow structure at run time.

```yaml
# hello.yaml — a compact one-node flow
id: hello
name: hello
input:
  name: !str
output: str
kind: agent
prompt: Write a warm one-sentence greeting to ${name}.
```

```console
yian run hello.yaml --input name=Ada --provider ollama --model llama3.2:3b
```

See [Agent](docs/agent.md) for the conceptual model and
[Flow syntax](docs/syntax.md) for the complete YAML surface.
AGENT nodes can declare [typed local variables](docs/syntax.md#agent-owned-variables)
with validated defaults and reference them in prompts as `${name}`.

## Why fixed flows?

- **Readable:** the flow is the plan. Humans and models can inspect the same
  artifact before it runs.
- **Composable:** typed nodes build flows; CALL, MAP, and LOOP apply child flows
  recursively.
- **Predictable:** data references determine the data edges, and the LLM never
  changes the authored call graph.
- **Recoverable:** every accepted transition is appended before it is applied,
  so an interrupted run can replay a validated record prefix.
- **Evolvable:** fixed flow and node identities give learning stable targets, so
  recorded outcomes from completed runs can improve later behavior without
  mutating the graph mid-run.

## Install

> **Early stage.** Yian is under active development. The API, YAML
> surface, and CLI may change quickly.

Yian needs Python 3.11+. Install from PyPI:

```console
python -m pip install "yian[all]"
```

To run the unreleased `main` branch or work on Yian itself, see the
[development install](docs/installation.md#development-install).

Provider extras can be installed separately: `anthropic`, `openai`, `google`,
or `ollama`. The core engine and CLI do not require a provider SDK; a flow needs
one only when it contains an AGENT node.

Yian has no built-in provider or model. Save a login with `yian auth <provider>`,
such as `yian auth anthropic --model claude-sonnet-4-5` or `yian auth openai`
(ChatGPT subscription or API key). It saves the credential and model, and becomes
the default provider when none is saved yet. `yian auth` lists saved logins.
`yian auth <provider> --default` switches the default; `--remove` deletes a login.
A flow's `llm_config`, `--provider`/`--model`, and
`YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL` all take precedence over the saved
login. See
[choosing a provider and model](docs/installation.md#choosing-a-provider-and-model).

## Run a flow

```console
yian run FLOW.yaml [--inputs inputs.json] [--input k=v]... \
  [--provider P] [--model M] [-w N] [--log PATH]
```

`--inputs` accepts a JSON object, including lists, records, and explicit nulls.
Repeatable `--input k=v` supplies scalar overrides. A name the flow does not
declare is an error. In a terminal, the CLI prompts for declared inputs that are
still missing.

New runs use up to eight workers for independent ready nodes. Set
`--num-workers N` / `-w N` to override the count, or `-w 0` for serial execution.
Resuming without a worker flag preserves the count recorded in the run log.

The CLI reports its selected JSONL path before execution. Once runtime
initialization commits `RunStarted`, that file is the authoritative log. The
default is `~/.yian/runs/<run_id>.jsonl`; choose another new path with
`--log`.
Resume an interrupted run against the same flow:

```console
yian run FLOW.yaml --resume PATH
```

Resume checks the run's provider keys first. In a terminal it prompts for a
missing key; otherwise it exits and appends nothing, so you can set the key and
resume again.

Run logs can contain inputs, model/tool/human data, outputs, memory snapshots,
configuration, and tracebacks. Treat them as sensitive. See [The `yian`
CLI](docs/cli.md) for input behavior, diagnostics, concurrency, and recovery.

For an interactive flow-backed chat:

```console
yian chat [FLOW] [--workspace DIR] [--provider P] [--model M]
```

The default yian chat can inspect, validate, run, and—after explicit
approval—write flows inside its workspace. A supplied chat flow owns its own
typed exit behavior.

## Use the Python API

```python
from pathlib import Path

from yian import load_flow, run_flow

path = Path("hello.yaml")
flow = load_flow(
    path.read_text(),
    search_paths=[path.parent],
    source_label=path.name,
)
result = run_flow(
    flow,
    {"name": "Ada"},
    llm_config={"provider": "ollama", "model": "llama3.2:3b"},
)
print(result.status, result.output)
```

`load_flow` parses, validates, and compiles the YAML into a reusable
`ComposedFlow`. `run_flow` makes a private graph for one run and returns a
`RunResult` with either the flow output or an expected `Failure`. Host, control,
log, and invariant faults propagate as exceptions.

A library host can inject per-run tools (including its `ask_user` handler and
named tool groups that an AGENT grants by name) with `tools=`, provider
configuration, and named memory implementations. Before a run starts, each TOOL
node's `tool_id:` must name one of the run's tools, and its inputs and `output:` are
checked against the argument and return types that tool declares; see
[Typed tools](docs/api.md#typed-tools). A tool whose first parameter is a
`ToolContext` receives its call id and the app object passed as `run_flow(..., context=obj)`;
a TOOL node's call id stays the same when `resume_flow` re-runs it, so it can serve as an
idempotency key (see [Tool context](docs/api.md#tool-context)). A TOOL node with `retries:`
calls its tool again after a failure, with backoff, and a resumed run continues the attempt
count (see [TOOL retries](docs/api.md#tool-retries)). A `FunctionTool` gives a
tool as a name, a description, a JSON Schema, and a call. To pass a provider key without an
environment variable, add it to the run's config, as in
`llm_config={"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": key}`;
the key is never recorded or echoed, so pass it again to `resume_flow(...,
api_key=key)` or `result.backward(..., api_key=key)`. A `base_url` beside the key,
such as an Azure OpenAI resource's `https://<resource>.openai.azure.com/openai/v1/`
with `"provider": "openai"`, is where that key is sent; flow files cannot set it,
except as the resource endpoint beside `provider: azure`.
Without one, the provider's default or environment endpoint applies.
Passing `learnable=True`
retains accepted execution for a later explicit `result.backward(feedback)` call,
with or without memory. Each call is a new
attempt, so feedback can arrive several times, days later. Forward
execution returns before any learning or commit; memory recall is independent.
The returned `BackwardOutcome` and `result.backward_outcome` keep feedback and
learning separate from the forward output and failure. See the
[Python API](docs/api.md#explicit-backward-feedback) for feedback shapes, memory,
and silent terminal recovery.

Pass `on_stream=` to receive an AGENT's reply text as it is generated. Pass
`on_event=` to receive node transitions plus, while an AGENT runs, per-call
`ModelCallFinished` token usage and tool call observers. Both are live callbacks
only; see [Python API](docs/api.md#run_flow) for payloads.

A host can steer or interrupt a running AGENT that declares
`steerable: true`. Pass `steering=inbox`, any object with `drain(node_id) ->
list[str]` and `interrupted(node_id) -> bool`. Queued steers reach the model as
user messages at the top of the next loop iteration. An interrupt ends the loop
at the next safe point, including mid-stream, and the node succeeds with its text
so far. See [Steering](docs/api.md#steering).

The built-in [`EpisodicMemory`](docs/learning.md#episodic-memory) keeps bounded,
model-free execution history for later runs. A flow declares a named memory,
`long_term_memory: {notes: episodic}`; its host builds the declared backends
with `build_memories` (in RAM unless given storage) and commits retained
episodes through explicit backward:

```python
from yian import build_memories

with build_memories(flow) as memory:
    result = run_flow(flow, inputs, memory=memory, learnable=True)
    result.backward()
```

[`ProfileMemory`](docs/learning.md#profile-memory) keeps one markdown document,
such as a person's profile, in a versioned file. Every AGENT under the declaring
flow sees the whole document; backward rewrites it with one model call and a
compare-and-set write. Declare `long_term_memory: {about: profile}` and supply
the backend under its key:

```python
from yian import build_memories
from yian.memory.profile import ProfileMemory
from yian.memory.storage import FileMarkdownStore

profile = ProfileMemory(FileMarkdownStore("people/ada.md"))
with build_memories(flow, supplied={"about": profile}) as memory:
    result = run_flow(flow, inputs, memory=memory, learnable=True)
    result.backward("Ada prefers short replies.")
```

[`SessionMemory`](docs/api.md#sessionmemory-foundation) supplies a structured session
transcript through the same memory context as notes from other memories. It keeps
conversation history across runs in the same process. Declare
`long_term_memory: {chat: session}` on the AGENT and reuse one registry:

```python
from yian import build_memories

with build_memories(flow) as memory:
    first = run_flow(flow, first_inputs, memory=memory)
    second = run_flow(flow, next_inputs, memory=memory)
    memory["chat"].reset()  # Starts fresh; fails while any turn is active or queued.
```

History advances automatically, without `learnable=True` or backward. Every call
site of the declaring AGENT shares its one conversation, and calls queue in
admission order. Two AGENTs cannot share a session key, so each keeps a separate
conversation. Explicit backward learning reuses API self-feedback
as committed notes for future runs, without another model call. API history uses a
soft 32,000-character context budget and preserves the latest four whole turns;
older context can be summarized without deleting raw history. Configure these
defaults on `SessionMemory(history_budget_chars=32_000, recent_turns=4)`, or
process-wide with `YIAN_SESSION_HISTORY_BUDGET_CHARS` and `YIAN_SESSION_RECENT_TURNS`.
Run logs do not reconstruct history. The episodic, reflective, and session
backends optionally retain their state across restarts in a SQLite file.
`build_memories` places one memory with `storage={key: path}`, or every memory
under `root=` (the flow `id:` must be usable as one directory name):

```python
with build_memories(flow, storage={"chat": "./session.sqlite"}) as memory:
    result = run_flow(flow, inputs, memory=memory)
```

An AGENT can also use Claude Code and Codex skills. `skills: [release-notes]` (or
`skills: all`) lists each granted skill's name and description in the system
message, and the model reads a skill's `SKILL.md` or bundled files through the
built-in `load_skill` tool when it needs them. `yian run` supplies the skills in
`~/.claude/skills`, `~/.agents/skills`, and the project's `.claude/skills` and
`.agents/skills`; a library host passes `skills=load_skills([...])` from
`yian.compose` (`{}` when a granting flow should get none). See [Skills](docs/api.md#skills).

Each local SQLite file has one owning backend until `close()` or context exit.
Idle run handles do not block closing; active work or queued turns do. Omit
`storage` to keep memory process-local. See [durable memory](docs/api.md#optional-durable-memory)
for ownership and recovery.

`yian run` builds every declared built-in memory (`episodic`, `reflective`,
`session`, `profile`). They stay in RAM unless a CLI option places them:

```bash
yian run flow.yaml --memory-root ./mem
yian run flow.yaml --resume ./run.jsonl --memory-root ./mem
```

`--memory-root DIR` keeps every memory under `DIR/<flow id>/` (the flow `id:`
must be usable as one directory name), and repeatable `--memory-storage KEY=PATH`
places one declared memory.
Use the same storage on resume; a run log is not a memory backup. The CLI does
not run backward learning. A flow that declares a custom memory type runs from
Python with `build_memories`; `yian run` exits 1 with a setup error for it.
See [CLI memory](docs/cli.md#memory-and-storage) for lifetime and examples.

## Documentation

- [Agent](docs/agent.md) — what “agent,” node, flow, child flow, memory, and run
  mean in this framework.
- [Architecture](docs/architecture.md) — how YAML becomes a compiled flow and a
  `RunResult`.
- [Flow syntax](docs/syntax.md) — the complete authoring reference.
- [Examples](docs/examples.md) — annotated flows from `examples/`.
- [Compose](docs/compose.md) — parsing, lowering, validation, and source origins.
- [The engine](docs/engine.md) — scheduling, records, recovery, and learning.
- [Runtime nodes](docs/nodes.md) — the `Node` / `NodeResult` contract.
- [Defaults and limits](docs/defaults.md) — every built-in default and limit, and
  the `YIAN_<NAME>` environment variable that changes it.
- [Engine design](src/yian/README.md) — contributor invariants and
  package ownership.

## Develop and test

```console
pip install -e ".[all,dev]"
PYTHONPATH=src python -m pytest
```

See [Contributing](CONTRIBUTING.md) for the issue/branch/PR workflow and
[Installation](docs/installation.md) for local environment setup.

## License

Apache 2.0 — see [LICENSE](LICENSE).
