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

Yian is the npm package `@ngocbh/yian`, written in TypeScript, in `ts/`. It runs on Node 22.18+
or Bun 1.4+. The first release isn't published yet; until then, run it from a clone:

```console
git clone https://github.com/ngocbh/yian
cd yian/ts
bun install --frozen-lockfile
bun src/cli/index.ts --help
```

See [Install](ts/README.md#install) for the package, the binaries and the optional MCP SDK.

Yian has no built-in provider or model. Save a login with `yian auth <provider>`,
such as `yian auth anthropic --model claude-sonnet-4-5` or `yian auth openai`
(ChatGPT subscription or API key). It saves the credential and model, and becomes
the default provider when none is saved yet. `yian auth` lists saved logins.
`yian auth <provider> --default` switches the default; `--remove` deletes a login.
A flow's `llm_config`, `--provider`/`--model`, and
`YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL` all take precedence over the saved
login. A field of a flow's `llm_config` may come from a value, such as
`model: ${input.model}`; one that resolves to null leaves the field to the next
layer. See
[choosing a provider and model](docs/installation.md#choosing-a-provider-and-model).

## Run a flow

```console
yian run FLOW.yaml [--inputs inputs.json] [--input k=v]... \
  [--provider P] [--model M] [-w N] [--log PATH] [--mcp-config FILE]...
```

`--inputs` accepts a JSON object, including lists, records, and explicit nulls.
Repeatable `--input k=v` supplies scalar overrides. A name the flow does not
declare is an error. In a terminal, the CLI prompts for declared inputs that are
still missing.

New runs use up to eight workers for independent ready nodes. Set
`--num-workers N` / `-w N` to override the count, or `-w 0` for serial execution.
Resuming without a worker flag preserves the count recorded in the run log.

To give a flow the tools of MCP servers, list them in `~/.yian/mcp.json` or a file
passed with `--mcp-config FILE` (the common `mcpServers` format). An AGENT grants a
server by its name, and only the servers a flow needs start; see
[MCP servers](docs/cli.md#mcp-servers).

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

`yian check FLOW...`, `yian skills check DIR...`, and `yian config check` check a
setup without running a flow; `yian providers` and `yian models` list the providers
and the models a login or a key offers. With `--json`, these commands,
`yian auth --remove`, and `yian run` write one `yian.cli.v1` JSON document on stdout,
and device sign-in writes JSON lines; see
[Machine-readable output](docs/cli.md#machine-readable-output).

For an interactive flow-backed chat:

```console
yian chat [FLOW] [--workspace DIR] [--provider P] [--model M]
```

The default yian chat can inspect, validate, run, and—after explicit
approval—write flows inside its workspace. A supplied chat flow owns its own
typed exit behavior.

TS yian: `yian chat FLOW` takes a flow with an `Inbox(str)` or `Inbox(Message)` input, such as
[`examples/chat.yaml`](examples/chat.yaml), sends each line you type to it, interrupts a
running reply on Ctrl-C, and ends with `/exit`; see [Chatting with a flow](docs/cli.md#chatting-with-a-flow-yian-chat).

## Use the library

The TypeScript API (`loadFlow`, `runFlow`, `advance`, host tools, memories, streaming and
events) is in [`ts/README.md`](ts/README.md).

An AGENT can also use Claude Code and Codex skills. `skills: [release-notes]` (or
`skills: all`) lists each granted skill's name and description in the system
message, and the model reads a skill's `SKILL.md` or bundled files through the
built-in `load_skill` tool when it needs them. `yian run` supplies the skills in
`~/.claude/skills`, `~/.agents/skills`, and the project's `.claude/skills` and
`.agents/skills`; a library host passes its own. See [Skills](docs/api.md#skills).

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
not run backward learning. A flow that declares a custom memory type runs from a
library host with `buildMemories`; `yian run` exits 1 with a setup error for it.
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

## Develop and test

```console
cd ts
bun run typecheck
bun run test
bun run test:node
bun tests/conformance.ts
```

See [Contributing](CONTRIBUTING.md) for the issue/branch/PR workflow and
[Installation](docs/installation.md) for local environment setup.

## License

Apache 2.0 — see [LICENSE](LICENSE).
