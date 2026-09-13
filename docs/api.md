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
- `memory` — host-created implementations keyed by the policy name selected by
  each authored `long_term_memory` declaration. `None` leaves all declarations
  dormant. A mapping opens one snapshot-isolated session per referenced policy,
  enables recall for AGENT nodes, and records the exact public snapshots needed
  by recovery.
- `learn` — retain accepted inputs, outputs, routes, and invocation context for
  a later explicit `result.backward(feedback)` call. It defaults to `False` and
  needs no memory policy. Forward execution never calls backward hooks, frame
  learners, or policy commits.

A node or flow-assert failure returns `RunResult(status="failed")` with one
immutable `result.failure`; a successful flow returns `status="succeeded"` with
`failure=None`. Inspect `failure.code` for program logic and `failure.message` for
display. `failure.node_id`, `failure.trace`, `failure.cause_type`, and
`failure.traceback` retain execution, authored-source, and debugging context.
Canonical presentation resolves `failure.trace` through `ComposedFlow.diagnose()`.
There are no `error`, `error_type`, or standalone `traceback` aliases.
Host-control, callback, log I/O, and internal reducer errors propagate after the
engine has drained active workers and released the boundary-owned log.

## Explicit backward feedback

`RunResult.backward(feedback=None) -> BackwardOutcome` operates on a terminal run
created with `learn=True`. It leaves the forward `input`, `status`, `output`,
`failure`, and `events` unchanged. A fresh forward result starts with `backward_outcome=None`
and `learning=None`; an accepted attempt updates the separately observed
`backward_outcome` and its `learning` projection.

A string addresses the whole output. A dictionary selects existing output fields;
a list selects positions and must have the exact output length at every nested
level, including positions containing only `None`. Omitted fields and `None`
carry no effective feedback. An empty string is effective text. A failed forward
run accepts only a string or `None`, because it has no usable output shape.
Invalid feedback is rejected before a new attempt or memory session opens, and
preserves this result's previously observed outcome and learning objects.

This complete example needs no model or memory backend:

```python
from agent_composer import load_flow, run_flow

flow = load_flow('''
id: structured-feedback
name: structured_feedback
input:
  items: list[int]
nodes: {}
output:
  items: ${input.items}
  label: batch
''')
result = run_flow(flow, {"items": [1, 2]}, learn=True)
assert result.backward_outcome is None and result.learning is None

outcome = result.backward({"items": [None, "Check this value."]})
assert outcome.status == "succeeded"
assert outcome.feedback.input_feedback == {"items": ([None, "Check this value."],)}
assert result.output == {"items": [1, 2], "label": "batch"}
```

`BackwardOutcome` is exported from `agent_composer.runtime`. Its `status` is
`succeeded` or `failed`; `feedback` is the accepted complete root `BackwardResult`,
or `None` when the accepted prefix has no completed result. `failure` identifies a backward,
learner, or commit failure. `learning` is `None` when no memory policy was enabled.
The root result's `input_feedback` maps input names to ordered tuples of ordinary
messages. These tuples are batches; positional lists inside them remain single
structured messages. Node self/variable channels contain ordered text messages
and remain attributed to their own occurrences, rather than becoming root channels.

`result.backward()` with no external feedback skips unreached AGENT backward
model calls. It still presents accepted history to every enabled memory frame,
including frames with empty or unexecuted histories. With effective feedback,
each reached AGENT makes at most one tool-free model call using its retained
rendered prompt, inputs, local variables/domains, accepted evidence, and recorded
effective model configuration. Tool interactions in that evidence are inert
data. No prompt/default/recall is evaluated again, and variable feedback does not
update variables, model configuration, or the graph.

Known callback/result failures return a failed outcome without partial root
feedback. A later learner or commit failure retains already completed feedback.
Host errors, cancellation, setup errors, and log faults propagate while leaving
the accepted prefix inspectable. They do not manufacture a successful or failed
completion record.

## Injecting agent memory

YAML declares long-term memory and selects a host policy without importing its
implementation:

```yaml
# memory.yaml
id: remembered-answer
name: remembered_answer
long_term_memory: reflective
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
from pathlib import Path

from agent_composer import load_flow, run_flow
from agent_composer.memory.reflective import ReflectiveMemory

path = Path("memory.yaml")
loaded = load_flow(path.read_text(), search_paths=[path.parent], source_label=path.name)
model = {"provider": "ollama", "model": "qwen3:30b-a3b-instruct-2507-q4_K_M"}
reflective = ReflectiveMemory(model)
result = run_flow(
    loaded,
    {"question": "What did we learn?"},
    llm_config=model,
    memory={"reflective": reflective},
    learn=True,
)
print(result.status, result.output)  # Forward has finished; nothing has learned yet.
outcome = result.backward("Explain the main point more directly.")
print(outcome.status, outcome.learning)
```

Passing the same memory object to later fresh runs makes committed lessons
available to recall. `reflective.reset()` explicitly starts a fresh generation.
`ReflectiveMemory` is process-local; persistence and other policy families are
separate future backends.

Application-specific implementations use the source-neutral `Memory`,
`MemorySession`, and `MemoryFrame` protocols exported by
`agent_composer.memory`. The host implements session snapshots, frame recall and
learning, and idempotent policy commits behind those interfaces.

Recall works whether or not `learn=True`. During an explicit backward attempt,
each enabled static frame receives its owned history and descendant evidence in
recorded layout order. An inherited frame is read-only to the descendant's own
learner: only the receiver frame's callback can propose mutations for that frame.
All frame proposals must succeed before any policy commit is requested. A known
learner, context, or proposal failure discards the temporary proposals and commits
nothing.

After proposals are accepted, each enabled policy receives one ordered batch,
including an empty batch. Its request is durable before the backend call. A known
commit failure stops later policies; already successful policies are not rolled
back. `LearningResult.status` is `succeeded`, `failed`, or `incomplete`, and its
ordered `policies` entries report:

| Policy status | Accepted evidence |
|---|---|
| `unrequested` | no commit request |
| `unknown` | a request without an accepted success/failure observation |
| `succeeded` | confirmed success |
| `failed` | confirmed backend failure |

Known learner/commit failures make learning `failed`, regardless of the backend's
retryability hint. An interrupted attempt remains `incomplete` unless it already
contains such a failure, even if every policy success was recorded before the
final completion record. Failure entries identify the policy and, for learning,
the exact receiving frame.

A Started-only prefix has no completed feedback, even if process-local work ran
before an append failed. Ready retains completed feedback and proposals for
inspection. Requests and observations add the policy dispositions above; only
Finished settles the attempt.

A successful backward attempt or any accepted commit request permanently closes
retry eligibility. Before that boundary, a caller may explicitly retry the whole
attempt with `result.backward(...)`: it opens fresh sessions from the original
snapshots and recomputes all callbacks and proposals. Accepted proposals are
inspectable but are never resumed. There is no automatic retry, rebase, or
rollback. An older `ReflectiveMemory` revision in the same namespace and reset
epoch restores successfully. Its later frame `learn` fails with
`reflective.stale_generation` without rebasing; the backward outcome retains
the completed feedback.

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

Every execution is reduced from one ordered schema-6 `RunLog`, including
process-local runs. The closed durable record vocabulary is `RunStartedRecord`,
`RunResumedRecord`, `NodeStartedRecord`, `NodeSucceededRecord`,
`NodeRoutedRecord`, `NodeExpandedRecord`, `NodeFailedRecord`,
`MemoryCommitRequestedRecord`,
`MemoryCommitSucceededRecord`, `MemoryCommitFailedRecord`,
`BackwardStartedRecord`, `BackwardReadyRecord`, and `BackwardFinishedRecord`. The four node
outcome records carry canonical forward inputs and routes when the recorded
run has `learn=True`; `learn=False` omits that capture. Optional model/tool
evaluation evidence is separate. Terminal success
or failure is derived from node transitions. `NodeFailedRecord` stores
the nested `Failure` directly beside any evidence. A causal `NodeRoutedRecord`
stores the optional derived terminal-unreachable `Failure` that replay
revalidates before restoring; no separate forward terminal record exists. Earlier
schemas are deliberately rejected rather than migrated.

Backward attempts use `BackwardStartedRecord`, then one `BackwardReadyRecord`
after the complete backward fold and every frame proposal, then ordered policy
requests/observations and `BackwardFinishedRecord`. A known failure before Ready
can finish the attempt directly. Commit records require the exact positive
keyword-only `attempt_id`; there are no per-frame learning records or legacy
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

!!! warning "Run logs may contain sensitive data"
    Canonical records retain caller inputs, node outputs (including human, tool,
    and model data), resolved model configuration, captured prompts/variables,
    feedback, memory snapshots and learning proposals, and failure tracebacks. The
    runtime does not redact or encrypt them. Put durable logs on an
    access-controlled path; do not commit or publish them.

## Durable recovery with `resume_flow`

Start a recoverable run by supplying a new log path:

```python
from pathlib import Path

from agent_composer import load_flow, run_flow
from agent_composer.run import resume_flow

run_log = Path("run.jsonl")
loaded = load_flow('''
id: delayed-feedback
name: delayed_feedback
input:
  text: str
nodes: {}
output: ${input.text}
''')

# In the original process. `run.jsonl` must not already exist.
result = run_flow(
    loaded,
    {"text": "A long explanation."},
    log=run_log,
    learn=True,
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
    memory: dict[str, Memory] | None = None,
) -> RunResult
```

Recovery locks the existing file before reading it, durably truncates any bytes
after its final newline, validates the complete record prefix, and checks that the
supplied flow has the recorded fingerprint. It reconstructs the typed pool,
dynamic expansions, original input, run identity, diagnostics, and snapshotted LLM
configuration without reading current process defaults.

If the original run enabled memory and forward execution is unfinished,
recovery requires implementations under the same names. Each session
must reproduce the exact snapshot stored by `RunStarted`; this prevents a resumed
run from reading a different memory generation. Every terminal run is inspected
silently without opening or validating memory, even after an interrupted backward
attempt. A supplied mapping retains policy references for a later eligible
explicit backward call.

For later backward on a memory-enabled log, pass the original policy mapping to
`resume_flow(..., memory=...)`; terminal inspection retains it without opening it.

If primary execution is unfinished, recovery appends one `RunResumedRecord`,
retries the exact ordered set of nodes that have a durable start but no durable
outcome, and continues the ready frontier. `num_workers=None` preserves the
recorded worker count; an integer selects and records the new segment's count.
Fresh inputs, `run_id`, and LLM configuration cannot be supplied during recovery.

After incomplete-tail repair, terminal inspection appends no records and invokes
no graph nodes, backward hooks, or memory learners/commits. It projects the last
accepted attempt, including an interrupted Started/Ready/request prefix.
`RunResult.events` contains only forward observers published in the current call;
backward records emit no observers.

A retained result keeps private program/data copies and policy references, not
open forward sessions, workers, or a log lock. An explicit backward call reopens
and validates its authority, restores fresh memory sessions when needed, and
closes resources afterward. The durable locator is anchored when acquired, so
later working-directory changes cannot redirect the result. A nonblocking guard
rejects overlapping calls on one result, and durable log locks exclude other
handles. Invalid submissions preserve the caller's last observed outcome; a
prefix-refresh failure makes that handle unusable rather than permitting a stale
retry. Log/cleanup faults preserve the primary exception and accepted records.

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

The terminal outcome of a fresh or recovered run exposes these public fields:

| Field | Meaning |
|-------|---------|
| `input` | canonical, coerced caller input retained from `RunStarted` |
| `status` | exactly `"succeeded"` or `"failed"` |
| `run_id` | durable identity shared by the state and log |
| `log` | durable JSONL `Path`, or `None` for a process-local fresh run |
| `output` | terminal flow value on success; otherwise `None` |
| `failure` | canonical `Failure` on failure; otherwise `None` |
| `events` | post-durable observer events published during this call only |
| `learning` | learning projection of the last observed backward outcome; `None` before an attempt is observed or when no policies were enabled |
| `backward_outcome` | last observed accepted `BackwardOutcome`, or `None` before an attempt is observed |

Call `backward(feedback=None)` explicitly on a retained result. It changes only
the observed backward/learning fields; forward status, values, failure, and events
keep their original meaning.

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `ac` CLI](cli.md) — run flows from the terminal.
