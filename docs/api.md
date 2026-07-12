# Python API

Use the engine as a library: load a flow, run it, and (if it pauses) resume it.

```python
from agent_composer import load_flow, run_flow

loaded = load_flow(open("hello.yaml").read(), search_paths=["."])
result = run_flow(loaded, {"name": "Ada"})
print(result.status, result.output)
```

## Public surface

These are exported from the top-level `agent_composer` package:

| Name | What it is |
|------|------------|
| `load_flow` | parse + compile + validate Compose-YAML into a `LoadedFlow` |
| `run_flow` | drive a `LoadedFlow` to a terminal, returning a `RunResult` |
| `LoadedFlow` | a compiled, validated flow ready to run |
| `CompiledFlow` | the compiled flow IR (`loaded.compiled`) |
| `FlowEngine` | the execution runtime (you rarely touch it directly) |
| `TypedVariablePool` | the typed value pool |
| `evaluate_when` | evaluate a boolean `when:`/assert expression |
| `LoadError` | raised by `load_flow` on a bad flow |
| `FlowValidationError` | raised on a failed validation check |
| `ExpressionError` | raised on a bad `${...}` expression |

The resume helpers live in `agent_composer.run`:

```python
from agent_composer.run import resume_flow, resume_command, RunResult
```

## `load_flow`

```python
load_flow(text: str, *, search_paths: list[str | Path] = ...) -> LoadedFlow
```

Parses, compiles, and validates the flow `text`. Pass `search_paths` so that a
flow's `call:` / `uses:` references resolve relative to the flow file's directory
— typically `search_paths=[flow_dir]`. Raises `LoadError` / `FlowValidationError`
on a bad flow (compile-time errors surface here, not at run time).

## `run_flow`

```python
run_flow(
    loaded: LoadedFlow,
    inputs: dict[str, Any],
    *,
    run_id: str | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> RunResult
```

Coerces `inputs` against the flow's declared types, seeds the pool, enforces
`asserts:`, and drives the flow to a terminal. It **never raises on a flow
failure** — a failed/aborted run, or a false assert, comes back as a `RunResult`
with `status != "succeeded"`.

- `run_id` — host-injected run id (readable as `${system.run_id}`); minted if
  omitted.
- `on_event` — a callback invoked with each engine event as it occurs (use it for
  progress). Event types include `NodeStarted`, `RunSucceeded`, `RunPaused`,
  `RunFailed`, `RunAborted`.

## `RunResult`

The outcome of one run (or resume):

| Field | Meaning |
|-------|---------|
| `status` | `"succeeded"`, `"failed"`, `"paused"`, or `"aborted"` |
| `output` | the flow's terminal value (a scalar or a multi-field object) — set on success |
| `error` | failure detail — set when `status == "failed"` |
| `input` | the coerced run-argument dict |
| `events` | the raw engine events, in order |
| `pause_reasons` | the `PauseReason`s when `status == "paused"` |
| `engine` | the live `FlowEngine` when paused (fast in-process resume) |
| `checkpoint` | a serializable snapshot when paused (durable, cross-process resume) |

## Resuming a paused run

A flow suspends on a `human_input` node, a timed `wait`, or an `ask_user`
control. When `result.status == "paused"`, deliver an answer per pause reason and
resume:

```python
from agent_composer.run import resume_flow, resume_command

result = run_flow(loaded, {"task": "Plan a team offsite"})
while result.status == "paused":
    commands = []
    for reason in result.pause_reasons:
        if reason.type == "human_input_required":
            answer = input(reason.prompt + " ")        # collect the typed answer
            commands.append(resume_command(loaded, reason, answer))
        elif reason.type == "scheduled_pause":
            commands.append(resume_command(loaded, reason, None))   # release a wait
    result = resume_flow(loaded, engine=result.engine, commands=commands)

print(result.status, result.output)
```

### `resume_flow`

```python
resume_flow(
    loaded: LoadedFlow,
    *,
    engine: FlowEngine | None = None,
    checkpoint: Any = None,
    commands: list[Any] | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> RunResult
```

Drives a suspended run to its next terminal via **exactly one** handle:
`engine=` (fast, in-process — from `result.engine`) or `checkpoint=` (durable,
cross-process — from `result.checkpoint`, e.g. deserialized after a restart).
`commands` are applied before the run continues. A re-pause returns a fresh
`RunResult` with new handles.

### `resume_command`

```python
resume_command(loaded: LoadedFlow, reason: Any, value: Any) -> Command
```

Maps a host-resumable `PauseReason` plus an answer `value` to the engine command
that delivers it. For a `human_input` the `value` is the typed answer; for a
`wait` release pass `value=None`. A reason with no `node_id` (a bare external
event) is not host-resumable and raises.

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `ac` CLI](cli.md) — the same run/resume loop from the terminal.
