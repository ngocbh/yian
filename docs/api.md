# Python API

Use the engine as a library: load a flow, run it, and optionally retain an
authoritative append-only log for recovery. An AGENT node's `ask_user` call,
HUMAN_INPUT, and WAIT stay inside live node calls; their enclosing nodes return
ordinary outputs. For the node/flow/agent distinction, start with
[Agent](agent.md).

```python
from pathlib import Path

from agent_composer import load_flow, run_flow

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

These names are exported from the top-level `agent_composer` package:

| Name | What it is |
|------|------------|
| `load_flow` | parse + compile + validate Compose-YAML into a `ComposedFlow` |
| `run_flow` | drive a `ComposedFlow` to a terminal, returning a `RunResult` |
| `Failure` | immutable semantic failure (`code`, message, node, origin trace, debug cause) |
| `ComposedFlow` | a compiled, validated flow with an optional resolver and `diagnose(failure)` |
| `CompiledFlow` | the compiled flow IR (`loaded.compiled`) |
| `VariablePool` | the typed value pool |
| `evaluate_when` | evaluate a boolean `when:`/assert expression |
| `LoadError` | bad-flow exception containing one canonical `diagnostic` |
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
resolver takes precedence over local search paths. Pass the filename as
`source_label` when `text` came from a file. Raises
`LoadError` on a bad flow (compile-time errors surface here, not at run time);
its immutable `.diagnostic` owns the message, notes, and every ordered
`DiagnosticFrame`, including frames from imported files.

```python
from agent_composer import LoadError

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
    memory: dict[str, Memory] | None = None,
    learn: bool = False,
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
- `llm_config` — the outer authored layer of the configuration cascade for AGENT nodes.
  The resolved map and process defaults are captured at initialization.
- `memory` — host-created implementations keyed by authored `memory_policy`
  name. `None` leaves all policies dormant. A mapping opens one snapshot-isolated
  session per referenced policy, enables recall for AGENT nodes, and records the exact
  public snapshots needed by recovery.
- `learn` — when true, run durable learning after the graph reaches either
  success or failure. It requires `memory`; learning failures are reported
  separately and never replace the primary result.

A node or flow-assert failure returns `RunResult(status="failed")` with one
immutable `result.failure`; a successful flow returns `status="succeeded"` with
`failure=None`. Inspect `failure.code` for program logic and `failure.message` for
display. `failure.node_id`, `failure.trace`, `failure.cause_type`, and
`failure.traceback` retain execution, authored-source, and debugging context.
Canonical presentation resolves `failure.trace` through `ComposedFlow.diagnose()`.
There are no `error`, `error_type`, or standalone `traceback` aliases.
Host-control, callback, log I/O, and internal reducer errors propagate after the
engine has drained active workers and released the boundary-owned log.

## Injecting agent memory

YAML names a policy without importing its implementation:

```yaml
id: remembered-answer
name: remembered_answer
memory_policy: reflective
input:
  question: str
nodes:
  answer:
    kind: agent
    input: {question: "${input.question}"}
    output: str
    prompt: Answer ${question}.
output: ${answer.output}
```

The host owns the backend, model configuration, credentials, tenant identity,
and lifetime:

```python
from agent_composer.memory.reflective import ReflectiveMemory

model = {"provider": "ollama", "model": "qwen3:30b-a3b-instruct-2507-q4_K_M"}
reflective = ReflectiveMemory(model)
result = run_flow(
    loaded,
    {"question": "What did we learn?"},
    llm_config=model,
    memory={"reflective": reflective},
    learn=True,
)
```

Passing the same memory object to later fresh runs makes committed lessons
available to recall. `reflective.reset()` explicitly starts a fresh generation.
`ReflectiveMemory` is process-local; persistence and other policy families are
separate future backends.

Application-specific implementations use the source-neutral `Memory`,
`MemorySession`, and `MemoryFrame` protocols exported by
`agent_composer.memory`. The host implements session snapshots, frame recall and
learning, and idempotent policy commits behind those interfaces.

`RunResult.learning` is `None` when `learn=False`. Otherwise it is independent
of `RunResult.status` and has status `succeeded`, `failed`, or `incomplete`.
Failures identify the policy and the exact frame for `learn`, or the policy-wide
`commit`; `incomplete` means a retryable operation remains for `resume_flow`.
Every frame owned by a flow or AGENT node learns only from its own accepted
occurrences. Descendant AGENT nodes recall flow frames before an optional local
frame owned by that node.

## Diagnosing a failed run

```python
from agent_composer.diagnostics import Diagnostic

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

`Diagnostic` is defined in `agent_composer.diagnostics`; it is not currently a
package-root `agent_composer` export. The CLI also calls
`loaded.diagnose(result.failure)` and sends that value to the shared renderer;
library callers use the same method when they need presentation-ready frames.

## Authoritative run logs

Every execution is reduced from one ordered schema-4 `RunLog`, including
process-local runs. The closed durable record vocabulary is `RunStartedRecord`,
`RunResumedRecord`, `NodeStartedRecord`, `NodeSucceededRecord`,
`NodeRoutedRecord`, `NodeExpandedRecord`, `NodeFailedRecord`,
`MemoryFrameLearningStartedRecord`, `MemoryFrameLearnedRecord`,
`MemoryFrameLearningFailedRecord`, `MemoryCommitRequestedRecord`,
`MemoryCommitSucceededRecord`, and `MemoryCommitFailedRecord`. The four node
outcome records may retain optional source-neutral evaluation evidence. Terminal
success or failure is derived from node transitions. `NodeFailedRecord` stores
the nested `Failure` directly beside any evidence. A causal `NodeRoutedRecord`
stores the optional derived terminal-unreachable `Failure` that replay
revalidates before restoring; no separate terminal record exists. Earlier
schemas are deliberately rejected rather than migrated.

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
    and model data), resolved model configuration, memory snapshots and learning
    proposals, and failure tracebacks. The
    runtime does not redact or encrypt them. Put durable logs on an
    access-controlled path; do not commit or publish them.

## Durable recovery with `resume_flow`

Start a recoverable run by supplying a new log path:

```python
from pathlib import Path

from agent_composer import load_flow, run_flow
from agent_composer.run import resume_flow

run_log = Path("run.jsonl")
flow_path = Path("flow.yaml")
loaded = load_flow(
    flow_path.read_text(),
    search_paths=[flow_path.parent],
    source_label=flow_path.name,
)

# In the original process. `run.jsonl` must not already exist.
result = run_flow(
    loaded,
    {"question": "..."},
    log=run_log,
    llm_config={"provider": "ollama", "model": "llama3.2:3b"},
)

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
    memory: dict[str, Memory] | None = None,
) -> RunResult
```

Recovery locks the existing file before reading it, durably truncates any bytes
after its final newline, validates the complete record prefix, and checks that the
supplied flow has the recorded fingerprint. It reconstructs the typed pool,
dynamic expansions, original input, run identity, diagnostics, and snapshotted LLM
configuration without reading current process defaults.

If the original run enabled memory and either primary execution or learning is
unfinished, recovery requires implementations under the same names. Each session
must reproduce the exact snapshot stored by `RunStarted`; this prevents a resumed
run from reading a different memory generation. A fully settled terminal run is
replayed silently and does not open or validate a supplied memory mapping.

If primary execution is unfinished, recovery appends one `RunResumedRecord`,
retries the exact ordered set of nodes that have a durable start but no durable
outcome, and continues the ready frontier. `num_workers=None` preserves the
recorded worker count; an integer selects and records the new segment's count.
Fresh inputs, `run_id`, and LLM configuration cannot be supplied during recovery.

If the primary result is terminal but learning has a retryable failure,
`resume_flow` appends only the next learning intent/result records; it does not
append `RunResumedRecord` or execute graph nodes. After any incomplete-tail
repair, recovery appends no records and is silent when both the primary result
and learning are settled. In every case, `RunResult.events` contains only primary
observers published in the current call; learning records do not emit observers.

!!! important "Recovery may retry an interrupted node"
    A durable `NodeStarted` with no durable outcome is intentionally run again.
    Code that performs external side effects must therefore tolerate retry; the
    log makes engine state deterministic, but it cannot make an interrupted
    external effect exactly-once.

## CLI path policy

The Python API remains path-neutral: `run_flow(..., log=None)` uses memory and a
caller chooses any durable path explicitly. The `ac` host adds a user-facing
policy on top:

- fresh `ac run FLOW` selects and reports
  `~/.agent_composer/runs/<run_id>.jsonl` unless `--log PATH` selects another
  path; once runtime commits `RunStarted`, that file is authoritative;
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

An authored `human_input` node and an AGENT node granted `tools: [ask_user]` call the
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
    result = run_flow(
        approval,
        {"task": "Plan a team offsite"},
        llm_config={"provider": "ollama", "model": "llama3.2:3b"},
    )
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
| `failure` | canonical `Failure` on failure; otherwise `None` |
| `events` | post-durable observer events published during this call only |
| `learning` | independent post-terminal learning result, or `None` when learning is disabled |

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `ac` CLI](cli.md) — run flows from the terminal.
