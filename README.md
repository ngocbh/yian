# The Agent Composer

[![Documentation](https://readthedocs.org/projects/agent-composer/badge/?version=latest)](https://agent-composer.readthedocs.io)

Agent Composer is a small language and runtime for readable agent flows. You
author a fixed graph in Docker-Compose-shaped YAML; the framework infers data
dependencies, applies explicit ordering constraints, validates types, and runs
through an append-only log.

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
  name: str
output: str
kind: agent
prompt: Write a warm one-sentence greeting to ${name}.
```

```console
ac run hello.yaml --input name=Ada --provider ollama --model llama3.2:3b
```

See [Agent](docs/agent.md) for the conceptual model and
[Flow syntax](docs/syntax.md) for the complete YAML surface.
AGENT nodes can declare [typed local variables](docs/syntax.md#agent-owned-variables)
with validated defaults and reference them in prompts as `${name}`.

## Why fixed flows?

- **Readable:** the flow is the plan. Humans and models can inspect the same
  artifact before it runs.
- **Composable:** typed nodes build flows; CALL, MAP, and LOOP apply child flows
  recursively within the runtime depth limit.
- **Predictable:** data references determine the data edges, and the LLM never
  changes the authored call graph.
- **Recoverable:** every accepted transition is appended before it is applied,
  so an interrupted run can replay a validated record prefix.
- **Evolvable:** fixed flow and node identities give learning stable targets, so
  recorded outcomes from completed runs can improve later behavior without
  mutating the graph mid-run.

## Install

> **Early stage.** Agent Composer is under active development. The API, YAML
> surface, and CLI may change quickly.

The current documentation describes the `main` branch. PyPI release `0.0.1`
contains an older, incompatible package, so install the current version from a
clone until the next release:

```console
git clone https://github.com/ngocbh/agent-composer.git
cd agent-composer
python -m pip install ".[all]"
```

For development, use `python -m pip install -e ".[all,dev]"` instead.

Provider extras can be installed separately: `anthropic`, `openai`, `google`,
or `ollama`. The core engine and CLI do not require a provider SDK; a flow needs
one only when it contains an AGENT node.

## Run a flow

```console
ac run FLOW.yaml [--inputs inputs.json] [--input k=v]... \
  [--provider P] [--model M] [-w N] [--log PATH]
```

`--inputs` accepts a JSON object, including lists, records, and explicit nulls.
Repeatable `--input k=v` supplies scalar overrides. In a terminal, the CLI
prompts for declared inputs that are still missing.

The CLI reports its selected JSONL path before execution. Once runtime
initialization commits `RunStarted`, that file is the authoritative log. The
default is `~/.agent_composer/runs/<run_id>.jsonl`; choose another new path with
`--log`.
Resume an interrupted run against the same flow:

```console
ac run FLOW.yaml --resume PATH
```

Run logs can contain inputs, model/tool/human data, outputs, memory snapshots,
configuration, and tracebacks. Treat them as sensitive. See [The `ac`
CLI](docs/cli.md) for input behavior, diagnostics, concurrency, and recovery.

For an interactive flow-backed chat:

```console
ac chat [FLOW] [--workspace DIR] [--provider P] [--model M]
```

The default composer chat can inspect, validate, run, and—after explicit
approval—write flows inside its workspace. A supplied chat flow owns its own
typed exit behavior.

## Use the Python API

```python
from pathlib import Path

from agent_composer import load_flow, run_flow

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

A library host can inject tool handlers, provider configuration, and named
memory implementations. Passing `learnable=True` retains accepted execution for a
later explicit `result.backward(feedback)` call, with or without memory. Forward
execution returns before any learning or commit; memory recall is independent.
The returned `BackwardOutcome` and `result.backward_outcome` keep feedback and
learning separate from the forward output and failure. See the
[Python API](docs/api.md#explicit-backward-feedback) for feedback shapes, memory,
and silent terminal recovery.

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
- [Engine design](src/agent_composer/README.md) — contributor invariants and
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
