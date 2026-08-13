# Python API

Use the engine as a library: load a flow, run it, and serve blocking human
interaction from the host. An AGENT's `ask_user` call, HUMAN_INPUT, and WAIT stay
inside live node calls; their enclosing nodes return ordinary outputs. A generic
artificial-pause API remains temporarily for compatibility.

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
| `load_flow` | parse + compile + validate Compose-YAML into a `ComposedFlow` |
| `run_flow` | drive a `ComposedFlow` to a terminal, returning a `RunResult` |
| `ComposedFlow` | a compiled, validated flow ready to run |
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
load_flow(text: str, *, search_paths: list[str | Path] = ...) -> ComposedFlow
```

Parses, compiles, and validates the flow `text`. Pass `search_paths` so that a
flow's `call:` / `uses:` references resolve relative to the flow file's directory
— typically `search_paths=[flow_dir]`. Raises `LoadError` / `FlowValidationError`
on a bad flow (compile-time errors surface here, not at run time).

## `run_flow`

```python
run_flow(
    loaded: ComposedFlow,
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

## Blocking human interaction

An authored `human_input` node and an agent granted `tools: [ask_user]` call the
same process-owned tool synchronously. A library host installs one handler for the
whole run lifecycle; the calling node remains live until the handler returns or
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

## Temporary generic pause compatibility

No authorable AGENT, HUMAN_INPUT, or WAIT node produces `RunPaused`. Only a
custom/artificial compiled node that directly returns the generic `Pause` result can
reach this dormant checkpoint path; issue #117 removes it. A host maintaining such
legacy code can deliver one compatibility value per host-resumable reason and resume:

```python
from agent_composer.run import resume_flow, resume_command

def compatibility_value_for(reason):
    """Return the host-specific legacy answer/event payload for this reason."""
    ...

# `legacy_flow` contains a custom/artificial node, not an authorable WAIT/HUMAN_INPUT.
result = run_flow(legacy_flow, {})
while result.status == "paused":
    commands = [
        resume_command(legacy_flow, reason, compatibility_value_for(reason))
        for reason in result.pause_reasons
        if reason.node_id is not None
    ]
    if not commands:  # a bare external-event reason needs its external watcher
        break
    result = resume_flow(legacy_flow, engine=result.engine, commands=commands)

print(result.status, result.output)
```

### `resume_flow`

```python
resume_flow(
    loaded: ComposedFlow,
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
resume_command(loaded: ComposedFlow, reason: Any, value: Any) -> Command
```

Maps a host-resumable compatibility `PauseReason` plus a `value` to the engine
command that delivers it as the artificial parked leaf's output. A reason with no
`node_id` (a bare external event) is not host-resumable and raises. This helper is
not used for authorable WAIT or human interaction.

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `ac` CLI](cli.md) — the same run/resume loop from the terminal.
