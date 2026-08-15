# Python API

Use the engine as a library: load a flow, run it, and optionally retain an
authoritative append-only log for recovery. An AGENT's `ask_user` call,
HUMAN_INPUT, and WAIT stay inside live node calls; their enclosing nodes return
ordinary outputs.

```python
from agent_composer import load_flow, run_flow

loaded = load_flow(open("hello.yaml").read(), search_paths=["."])
result = run_flow(loaded, {"name": "Ada"})
print(result.status, result.output)
```

## Public surface

These names are exported from the top-level `agent_composer` package:

| Name | What it is |
|------|------------|
| `load_flow` | parse + compile + validate Compose-YAML into a `ComposedFlow` |
| `run_flow` | drive a `ComposedFlow` to a terminal, returning a `RunResult` |
| `ComposedFlow` | a compiled, validated flow ready to run |
| `CompiledFlow` | the compiled flow IR (`loaded.compiled`) |
| `VariablePool` | the typed value pool |
| `evaluate_when` | evaluate a boolean `when:`/assert expression |
| `LoadError` | raised by `load_flow` on a bad flow |
| `FlowValidationError` | raised on a failed validation check |
| `ExpressionError` | raised on a bad `${...}` expression |

Recovery and the result type live in `agent_composer.run`:

```python
from agent_composer.run import RunResult, resume_flow
```

The lower-level `FlowEngine`, `RunState`, `RunLog`, `MemoryRunLog`, and
`JsonlRunLog` interfaces are scoped to `agent_composer.runtime`. A direct engine
caller must first apply a `RunStarted` transition to a matching state/log pair;
the package-root functions below own that setup for normal callers.

## `load_flow`

```python
load_flow(text: str, *, search_paths: list[str | Path] = ...) -> ComposedFlow
```

Parses, compiles, and validates the flow `text`, including every authored input
default. Pass `search_paths` so that a flow's `call:` / `uses:` references resolve
relative to the flow file's directory — typically `search_paths=[flow_dir]`.
Raises `LoadError` / `FlowValidationError` on a bad flow (compile-time errors
surface here, not at run time).

## `run_flow`

```python
run_flow(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 0,
    on_event: Callable[[Any], None] | None = None,
    on_stream: Callable[[StreamChunk], None] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> RunResult
```

Coerces supplied `inputs` against the flow's declared types, snapshots the
complete LLM configuration, initializes the run log, and drives the flow to a
terminal. Stored defaults are not coerced.

- `log` — an optional JSONL path. `None` still uses the same record/reducer path
  with a process-local log. A path creates a new durable log exclusively; an
  existing file is never replaced.
- `run_id` — host-injected run identity (readable as `${system.run_id}`); minted
  when omitted and retained by every record.
- `num_workers` — `0` runs serially; a positive value bounds the worker pool.
- `on_event` — receives post-durable observer events as they occur.
- `on_stream` — receives transient `StreamChunk` values. Chunks are neither
  durable records nor members of `RunResult.events`.
- `llm_config` — the outer authored layer of the per-agent configuration cascade.
  The resolved map and process defaults are captured at initialization.

A node or flow-assert failure returns `RunResult(status="failed")`; a successful
flow returns `status="succeeded"`. Host-control, callback, log I/O, and internal
reducer errors propagate after the engine has drained active workers and released
the boundary-owned log.

## Authoritative run logs

Every execution is reduced from one ordered `RunLog`, including process-local
runs. The closed durable record vocabulary is `RunStartedRecord`,
`RunResumedRecord`, `NodeStartedRecord`, `NodeSucceededRecord`,
`NodeRoutedRecord`, `NodeExpandedRecord`, and `NodeFailedRecord`. Terminal
success or failure is derived from those node transitions.

For each transition, the dispatcher follows one authority order:

```
prepare and validate record -> append record -> apply reducer -> publish observers
```

An observer therefore never claims a transition that was not first accepted by
the log. `JsonlRunLog` writes one canonical JSON record per line and fsyncs every
append while holding a nonblocking exclusive file lock. `MemoryRunLog` preserves
the same ordering without a durable path.

!!! warning "Run logs may contain sensitive data"
    Canonical records retain caller inputs, node outputs (including human, tool,
    and model data), resolved model configuration, and failure tracebacks. The
    runtime does not redact or encrypt them. Put durable logs on an
    access-controlled path; do not commit or publish them.

## Durable recovery with `resume_flow`

Start a recoverable run by supplying a new log path:

```python
from pathlib import Path

from agent_composer import load_flow, run_flow
from agent_composer.run import resume_flow

run_log = Path("run.jsonl")
loaded = load_flow(open("flow.yaml").read(), search_paths=["."])

# In the original process. `run.jsonl` must not already exist.
result = run_flow(loaded, {"question": "..."}, log=run_log)

# Safe in a later process, whether the prefix is terminal or unfinished.
recovered = resume_flow(loaded, log=run_log)
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
) -> RunResult
```

Recovery locks the existing file before reading it, durably truncates any bytes
after its final newline, validates the complete record prefix, and checks that the
supplied flow has the recorded fingerprint. It reconstructs the typed pool,
dynamic expansions, original input, run identity, diagnostics, and snapshotted LLM
configuration without reading current process defaults.

If the prefix is unfinished, recovery appends one `RunResumedRecord` and retries
the exact ordered set of nodes that have a durable start but no durable outcome,
then continues the ready frontier. `num_workers=None` preserves the recorded worker
count; an integer selects and records the new segment's count. Fresh inputs,
`run_id`, and LLM configuration cannot be supplied during recovery.

If the prefix is already terminal, `resume_flow` is read-only and silent: it
returns the reconstructed result without appending a record or replaying historical
events. In either case, `RunResult.events` contains only observers published in
the current call.

!!! important "Recovery may retry an interrupted node"
    A durable `NodeStarted` with no durable outcome is intentionally run again.
    Code that performs external side effects must therefore tolerate retry; the
    log makes engine state deterministic, but it cannot make an interrupted
    external effect exactly-once.

## CLI path policy

The Python API remains path-neutral: `run_flow(..., log=None)` uses memory and a
caller chooses any durable path explicitly. The `ac` host adds a user-facing
policy on top:

- fresh `ac run FLOW` creates and reports
  `~/.agent_composer/runs/<run_id>.jsonl` unless `--log PATH` selects another new
  file;
- `ac run FLOW --resume PATH` calls the same recovery boundary without accepting
  fresh inputs, a second log path, or provider/model configuration;
- omitting `-w` on resume preserves the recorded worker count, while an explicit
  value affects the new segment only;
- each `ac chat` attempt, including a failed-turn restart, receives a separate
  reported default path.

The default CLI directories are mode `0700`; `JsonlRunLog` creates files with
mode `0600`. See [the CLI guide](cli.md#durable-logs-and-recovery) for commands
and the log-sensitivity warning.

## Blocking human interaction

An authored `human_input` node and an agent granted `tools: [ask_user]` call the
same process-owned tool synchronously. A library host installs one handler for the
whole live run; the calling node remains active until the handler returns or
raises:

```python
from agent_composer.tools import install_ask_user

approval = load_flow(
    open("examples/human-approval.yaml").read(),
    search_paths=["examples"],
)

def ask(**request):
    return input(request["question"] + " ")

with install_ask_user(ask):
    result = run_flow(approval, {"task": "Plan a team offsite"})
```

The handler receives `question`, `questions`, `answer_schema`, `title`, and
`node_id`. `human_input` uses the trusted metadata path; model-requested
`ask_user` calls can supply only the public `question` argument. Only one handler
may own the reserved tool in a process at a time.

An authored WAIT needs no host handler. It blocks one worker until its absolute
UTC `until:` deadline, returns `None`, and can be woken cooperatively when the host
stops the run.

## `RunResult`

The terminal outcome of a fresh or recovered run has exactly these fields:

| Field | Meaning |
|-------|---------|
| `input` | canonical, coerced caller input retained from `RunStarted` |
| `status` | exactly `"succeeded"` or `"failed"` |
| `run_id` | durable identity shared by the state and log |
| `log` | durable JSONL `Path`, or `None` for a process-local fresh run |
| `output` | terminal flow value on success; otherwise `None` |
| `error` | human-readable failure detail on failure; otherwise `None` |
| `locator` | symbolic source location for a failed flow, when available |
| `range` | resolved source range for a failed flow, when available |
| `traceback` | captured node traceback for exception-backed failures, when available |
| `events` | post-durable observer events published during this call only |

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `ac` CLI](cli.md) — run flows from the terminal.
