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
    num_workers: int = 8,
    on_event: Callable[[Any], None] | None = None,
    on_stream: Callable[[StreamChunk], None] | None = None,
    llm_config: dict[str, Any] | None = None,
    memory: dict[str, Memory] | None = None,
    learnable: bool = False,
    tools: Mapping[str, BaseTool | Callable[..., Any] | list | tuple] | None = None,
    skills: Mapping[str, Skill] | None = None,
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
- `num_workers` — defaults to `8`; `0` runs serially, and a positive value bounds
  the worker pool.
- `on_event` — receives post-durable observer events as they occur. While an
  AGENT node runs it also receives transient in-loop observers
  (`ModelCallStarted`, `ModelCallFinished` with `TokenUsage`, `ToolCallStarted`,
  `ToolCallFinished`), strictly between that node's `NodeStarted` and its outcome.
  In-loop observers are never durable, never members of `RunResult.events`, and
  never replayed on resume; filter them with
  `isinstance(event, agent_composer.events.IN_LOOP_EVENTS)`. Observers from
  concurrently running nodes interleave (runs are pooled by default); correlate
  them by `node_id` and `exchange_id`. `usage` is the provider's own count
  (`None` when unreported) and `error` is an exception type name only.
- `on_stream` — receives transient `StreamChunk` values. Chunks are neither
  durable records nor members of `RunResult.events`. With `on_stream` set, AGENT
  chat replies stream visible text as `StreamChunk(node_id, "text", chunk,
  exchange_id=...)`; structured replies do not stream. A completed streamed call
  ends with one empty `final=True` chunk. A failed call ends without one and its
  `ModelCallFinished.error` is set; a host discards that exchange's partial text.
  The node value is authoritative: streamed text can differ from it in the
  whitespace between content blocks. Without `on_stream`, AGENT calls `invoke`
  exactly as before.
- `llm_config` — the outer authored layer of the configuration cascade for AGENT nodes.
  The resolved map and process defaults are captured at initialization.
- `memory` — host-created implementations keyed by the policy name selected by
  each authored `long_term_memory` declaration. `None` leaves all declarations
  dormant. A mapping opens one run-scoped session per referenced policy, enables
  recall and optional conversation history for AGENT nodes, and records exact
  opening snapshots for recovery. Those snapshots stay pinned; SessionMemory
  reads latest live history after turn admission.
- `learnable` — retain accepted inputs, outputs, routes, and invocation context for
  a later explicit `result.backward(feedback)` call. It defaults to `False` and
  needs no memory policy. Forward execution never calls backward hooks, frame
  learners, or policy commits.
- `tools` — the tools this run sees, keyed by tool id. TOOL nodes, AGENT
  `tools:` grants, and HUMAN_INPUT look ids up here. `None` uses the tools
  registered with `register_tool`. A mapping, even an empty one, is the whole
  set: registered tools are hidden from the run. To keep them, pass
  `{**registered_tools(), "my_tool": fn}`. `registered_tools()` returns a copy
  taken when you call it, so register every tool before the run starts. A
  function or method becomes a tool named by its key and described by its
  docstring. To bind values such as a session id, use a closure or a bound
  method; `functools.partial` and other callable objects are rejected. A
  LangChain `BaseTool` must already use its key as its name. The `ask_user`
  entry is the run's human handler (see
  [Blocking human interaction](#blocking-human-interaction)). Tools are host
  callables, so the run log does not record them. An AGENT that grants skills
  also gets the built-in `load_skill` tool, which is not part of this mapping.
  A list or tuple value is a tool group (see [Tool groups](#tool-groups)).
  Every static AGENT grant must name a tool or group in this set; otherwise
  `run_flow` raises `ValueError` before the log is created.
- `skills` — the skills this run may use, keyed by name (see [Skills](#skills)).
  An AGENT's `skills:` grant selects from this mapping. A listed skill that is
  missing raises `ValueError` before the run starts.

### Tool groups

A host can fill a named group of tools per run. The flow grants the group by
name, next to plain tool ids:

```yaml
nodes:
  worker:
    kind: agent
    mode: tool_calling
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
Each group item is a LangChain `BaseTool` named by its `name`, or a function or
bound method named by its `__name__` (a lambda has no usable name and is
rejected). Groups expand before the grant check, so the model calls the tools by
their own names, never by the group name. A group is not a tool: a TOOL node
cannot call it. Before the log is created, `run_flow` raises:

- `ValueError` when an AGENT grants a group (or tool) the mapping lacks;
- `ValueError` when a tool name appears in two groups, twice in one group, or
  both in a group and as a top-level key, since tools are called by name;
- `ValueError` when a group tool is named `ask_user`. The handler stays the
  top-level `tools["ask_user"]` entry, and an AGENT that grants `ask_user` next to
  groups gets that handler.

Groups are host capabilities, like every other entry in `tools`: the log records
neither their contents nor their tools. Only the group names in the flow are part
of its fingerprint. Each run gets its own groups, so concurrent runs never see each
other's tools, and `resume_flow` uses the groups passed to it (see
[Durable recovery](#durable-recovery-with-resume_flow)).

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
created with `learnable=True`. It leaves the forward `input`, `status`, `output`,
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
result = run_flow(flow, {"items": [1, 2]}, learnable=True)
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

## Skills

An AGENT granted `skills:` in YAML lists each skill's name and description in its
system message and reads the full instructions on demand through the built-in
`load_skill` tool (see [Agent skills](syntax.md#agent-skills-skills)). The host
chooses which skills exist for a run:

```python
from agent_composer.compose import load_flow, load_skills
from agent_composer.run import resume_flow, run_flow

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
- `default_skill_roots(cwd=None, home=None)` returns the folders `composer run`
  uses: `~/.claude/skills`, `~/.agents/skills`, then `.claude/skills` and
  `.agents/skills` in each directory from `cwd` up to the git root.
- `agent_composer.skills.Skill(name, description, instructions, directory=None)`
  builds a skill in code. `directory=None` means it has no bundled files.

A skill's text is read once, when `load_skills` runs, so every node in the run sees
the same instructions. The listing becomes part of the node's recorded
`base_system_prompt` evidence and each `load_skill` result is a recorded tool
result, so backward learning sees what the model saw. The run log does not store
skills: pass the same mapping to `resume_flow`, as with `memory=`.

Each `load_skill` result is capped at 256 KiB. There is no cumulative cap: every
result stays in the loop's message history, the evidence, and any logs.

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

The Python host owns the backend, model configuration, credentials, tenant
identity, and lifetime. `run_flow(..., memory=None)` still leaves declarations
dormant. The `composer run` host supplies declared built-in policies automatically;
see [CLI memory](cli.md#memory-policies-and-storage). Python injection supports
custom policies and constructor settings:

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
    learnable=True,
)
print(result.status, result.output)  # Forward has finished; nothing has learned yet.
outcome = result.backward("Explain the main point more directly.")
print(outcome.status, outcome.learning)
```

Passing the same memory object to later fresh runs makes committed lessons
available to recall. `reflective.reset()` explicitly starts a fresh generation.
`ReflectiveMemory` is process-local by default; `storage=path` enables the optional
SQLite backend described below.

For model-free execution history, use the built-in episodic policy:

```python
from agent_composer.memory.episodic import EpisodicMemory

episodic = EpisodicMemory(max_episodes=100, recall_limit=10)
result = run_flow(
    loaded,
    {"question": "What happened last time?"},
    memory={"episodic": episodic},
    learnable=True,
)
result.backward()  # Appends retained episodes after the reverse pass.
```

The corresponding declaration is `long_term_memory: episodic`. Its mapping form
accepts `config.max_episodes` and `config.recall_limit`, which override the
constructor defaults for that static frame. Each successful backward attempt
appends the complete owner-scoped execution records in their recorded order;
`max_episodes` then evicts the oldest records. Recall without a context returns
the whole bounded history chronologically. AGENT recall ranks records by lexical
overlap with the system prompt, rendered prompt, and canonical inputs, breaks
ties by recency, and returns at most `recall_limit` records. The returned text is
newline-separated canonical JSON so each episode keeps its run and execution
attribution.

`EpisodicMemory` is process-local by default. Reuse the same object to share committed
history across fresh runs in one process. `reset()` clears history and starts a
new generation. A session snapshot is detached JSON suitable for the engine's
recorded recovery boundary or for restoring a pristine `EpisodicMemory`.
Use `storage=path` to retain complete backend state across restarts.

### Profile memory

`ProfileMemory` recalls one markdown document into every AGENT under its outermost
frame and rewrites it on backward with a compare-and-set write:

```python
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.storage import FileMarkdownStore

profile = ProfileMemory(FileMarkdownStore("people/ada.md"), max_chars=32_000)
result = run_flow(loaded, inputs, memory={"profile": profile}, learnable=True)
outcome = result.backward("Ada prefers short replies.")
if outcome.failure is not None and outcome.failure.code == "profile.version_mismatch":
    ...  # someone edited the profile; nothing was written; start a new attempt
```

Only `profile.version_mismatch` guarantees that nothing was written. After any
other commit failure, or an attempt killed mid-commit (policy status `unknown`),
look for the attempt's `commit_id` in the store's write provenance before
retrying. If it is missing, compare the current document with the proposed
`text` in the run log. The snapshot records the pinned version and text hash, never the text.
See [profile memory](learning.md#profile-memory) for recall, learning, and store
rules.

### Optional durable memory

`EpisodicMemory`, `ReflectiveMemory`, and `SessionMemory` accept the host-only
keyword `storage="path/to/memory.sqlite"`. `storage=None`, the default, opens no
database. Use a separate file for each backend; its parent directory must already
exist. Paths are not flow YAML, public snapshot fields, or run-log settings.

```python
from agent_composer.memory.session import SessionMemory

with SessionMemory(storage="./session.sqlite") as memory:
    first = run_flow(flow, first_inputs, memory={"session": memory})

# A later process can reopen the same file after the first owner closes it.
with SessionMemory(storage="./session.sqlite") as memory:
    second = run_flow(flow, next_inputs, memory={"session": memory})
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
or unsupported stored state instead of overwriting it. `reset()` clears policy
content and starts a new generation, while retaining commit deduplication so
replayed commits cannot resurrect cleared content. It does not remove the file or
external provider resources.

Private storage retains state omitted from public snapshots: commit deduplication,
pristine/adoption flags, and SessionMemory's derived summary cache. It never saves
live handles, queues, callbacks, or provider clients. Public run-opening snapshots
stay pinned, and reopening an older snapshot does not rewind newer live history.
The run log is not a substitute for this store, and repeating an interrupted
model request or tool effect remains possible.

Current constructor settings control future work after reopening. Existing frame
limits and recorded run snapshots remain unchanged; new frames use current
defaults, and future reflective learning uses the current host `llm_config`.
SessionMemory uses the current history budget and recent-turn count. Increasing
the recent-turn count restores raw context for newly protected turns instead of
hiding them behind an older summary. These settings and storage ownership belong
to the host. `composer run` exposes storage selection through
`--memory-storage POLICY=PATH` and otherwise uses constructor defaults.

Application-specific implementations use the source-neutral `Memory`,
`MemorySession`, and `MemoryFrame` protocols exported by
`agent_composer.memory`. The host implements session snapshots, frame recall and
learning, and idempotent policy commits behind those interfaces.

`frame.recall(context)` may return a string or `MemoryContent(text=..., transcript=...)`.
A string contributes notes/instructions. A transcript source implements
`TranscriptSource.open_turn(...)` and contributes structured message history.
SessionMemory returns this richer carrier; it never flattens history into a note.
All visible frames are recalled once, their notes remain ordered outer-to-local,
and the innermost contributed transcript is admitted. An ordinary notes frame
inside a session frame adds context without hiding the session.

The evaluator gives agents one `MemoryContext`, containing labelled text sections
and an optional transcript turn. This is analogous to using a session transcript
together with project instructions or saved memories. Both influence the model;
only the transcript preserves who said what and which result belongs to each tool
call. Context handles are transient and do not change public snapshot formats.

Recall works whether or not `learnable=True`. During an explicit backward attempt,
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

Each `result.backward(...)` call is a new attempt, including after a successful,
failed, or committing attempt, and on a result from `resume_flow`. It opens fresh
sessions from the original snapshots and recomputes all callbacks and proposals.
An unfinished earlier attempt is first recorded as interrupted. Accepted proposals
are inspectable but are never resumed. There is no automatic retry, rebase, or
rollback. After the first attempt that sends commits, later attempts use
attempt-scoped commit IDs (`ac-memory-v2:`), so a backend that deduplicates by
commit ID still applies a later attempt's identical batch. A new attempt after an
`unknown` request may therefore apply a second write; the host reconciles. An older
`EpisodicMemory` or `ReflectiveMemory` revision in the same namespace and reset
epoch restores successfully. Its later frame `learn` fails with
`episodic.stale_generation` or `reflective.stale_generation` without rebasing;
the backward outcome retains the completed feedback.

## SessionMemory foundation

`agent_composer.memory.session.SessionMemory` retains AGENT
message conversations across calls to `run_flow`.
The backend owns history updates and
FIFO admission directly; runtime owns only the optional turn's lifetime and
authored scope. Explicit backward learning retains self-feedback as notes through
the same memory lifecycle. Older API context can be summarized without deleting
raw history. `composer run` activates declared built-in policies automatically.

Declare `long_term_memory: session` on a flow or AGENT node and reuse the same
backend under that policy name. For example:

```python
from agent_composer import load_flow, run_flow
from agent_composer.memory.session import SessionMemory

flow = load_flow("""
id: remembered-chat
name: remembered_chat
long_term_memory: session
input: {question: str}
nodes:
  answer:
    kind: agent
    mode: plain
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
output: ${answer.output}
""")
memory = SessionMemory()
model = {"provider": "ollama", "model": "llama3.2:3b"}
first = run_flow(flow, {"question": "My project is called Atlas."},
                 memory={"session": memory}, llm_config=model)
second = run_flow(flow, {"question": "What is my project's name?"},
                  memory={"session": memory}, llm_config=model)
```

The second invocation receives the first invocation's messages. Neither
`learnable=True` nor `backward()` is needed. Without a recalled transcript source,
API calls start with the current prompt and any recalled notes.

`SessionMemory` is the concrete backend; the existing `MemorySession` name still
means the run-scoped protocol returned by a memory backend. The contracts-only
`agent_composer.memory` facade exports `MemoryContent`, `MemoryContextRequest`,
`MemoryContext`, immutable conversation carriers, and the `TranscriptSource` and
`ConversationTurn` protocols. Existing text-only recall implementations remain valid. Import the concrete backend from its own module:

```python
from agent_composer.memory.session import SessionMemory

memory = SessionMemory()
with memory.open_run() as opened:
    snapshot = opened.snapshot  # Detached JSON, pinned when this run handle opened.

restored = SessionMemory()
with restored.open_run(snapshot=snapshot) as opened:
    assert opened.snapshot == snapshot

memory.reset()  # Backend-wide; allowed only while no turn is active or queued.
```

The host constructor accepts `history_budget_chars=32_000`, `recent_turns=4`, and
optional `storage=path`;
these settings are not flow YAML. Frame configuration must be `{}`. A public
snapshot has exactly `version` (currently `2`), `namespace`, `generation`,
`conversations`, and `notes`. It contains complete conversation views and learned
notes, not queues, leases, the private summary cache, or the commit ledger.
Version-1 snapshots remain
readable as history without notes; an opened version-1 snapshot stays exactly
version 1 when returned by that run handle.
A pristine backend can adopt a snapshot; an already
used backend accepts only proven history in its current namespace and generation,
without rewinding live state. Every snapshot getter returns a detached copy of
the original opening snapshot, even after turns advance live
history. Without `storage`, the backend writes no files or database. See
[optional durable memory](#optional-durable-memory) for restart persistence.

### Conversation ownership and updates

A conversation key combines the existing authored memory frame with the leaf's
`consumer_static_path`. Repeated invocations of that authored leaf share one
conversation and always queue in FIFO order. Distinct leaf/callsite keys remain
independent, including under a flow-owned frame: there is no flow-wide conversation
or queue. MAP items and LOOP iterations do not create new conversation identities;
there is no fork option. Queue order is admission order, not a promise of MAP
input-index order. Use non-session memory for independent MAP items.
The innermost frame contributing a transcript owns the turn; all applicable
frames still provide ordinary recall. A local non-session frame does not hide
an enclosing SessionMemory frame. Runtime supplies authored identity and the
accepted invocation's run ID/start sequence; the backend never inspects a graph.

Entering `frame.open_turn(request, check_stopped=check)` reserves the queue position
before reading the latest live conversation. The turn's `request(...)` and
`observe(...)` methods validate and publish directly in SessionMemory. No runtime
checkpoint callback or acknowledgment is required. The caller executes a provider
or tool between those calls, outside the backend lock. Cooperative stop checks run
while waiting and between effects; actual observations are retained before honoring
a stop that arrived during the effect.

The evaluator keeps the lease through the node body, stream cleanup,
post-assertions, and result-shape validation, then releases it before returning
to the dispatcher. A completed turn records completion of that execution, not
acceptance of a later typed-output validation or run-log append.

These updates advance automatically through the turn capability, independently of
`learnable` or backward. `SessionMemory` recall returns
`MemoryContent`: `.text` supplies committed learned notes for the requested leaf,
and `.transcript` supplies the source for structured history. Context-free recall
returns empty `.text` with the same transcript source. Code calling frame recall
directly should read `.text` when it only needs notes.
The API adapter's compatibility descriptor contains the resolved provider and
versioned message format. Compatible model, prompt, and tool-grant changes keep
history; changing provider or message format requires explicit reset. Every call
uses current instructions, recall, and tool grants. Historical system/developer
instructions are not reinstated as current instructions.

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
backend's condition lock. Other authored conversations can proceed independently.
A failed, malformed, empty, or tool-request summary response fails that invocation
without deleting history or making the normal answer request. Summary work does
not run backward learning or enter structured-output correction retries.

### Learned notes

With `learnable=True`, explicit `result.backward(feedback)` can produce textual
`self_feedback` in the existing API backward model call. SessionMemory's frame
learner turns that feedback into deterministic note mutations; the existing
memory commit publishes the batch atomically and idempotently. There is no
additional note-generation model call. For the flow and model above:

```python
reviewed = run_flow(flow, {"question": "Describe Atlas concisely."},
                    memory={"session": memory}, llm_config=model, learnable=True)
outcome = reviewed.backward("State assumptions before presenting conclusions.")
```

Any self-feedback successfully committed by that backward attempt is available
to later runs through ordinary memory recall. Merely enabling `learnable` does
not add notes; backward without effective feedback adds none. Notes belong to
the frame and authored leaf that actually admitted the conversation. Sibling
leaves and enclosing recall-only frames do not learn duplicate notes, and
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

The four schema-8 conversation RunLog wrappers remain readable as inert historical
metadata. Runtime validates canonical payloads, run identity, unfinished accepted
start, enabled frame/policy, compiled consumer scope, and outer lifecycle. It
advances only record/trace sequencing, with no transcript reconstruction, output
binding, frontier changes, execution episodes, observers, or backend calls.
After primary failure, only observations and close for still-unfinished starts
are accepted; no wrapper is accepted after primary success or any backward start,
or after its originating start completes or is replaced by retry.

`reset()` immediately raises `MemoryOperationError` with code `session.busy`
without changing state if **any** turn is active or queued. When idle, it clears
all local conversations and advances the generation. Older run/frame
handles then fail as stale. Reset does not delete
external provider history or files. See the [rendered contracts](reference.md#session-memory-foundation)
for the lower-level signatures.

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

Every execution is reduced from one ordered `RunLog`, including process-local runs.
New records use schema 8; readers accept exactly schemas 7 and 8. The thirteen
pre-existing kinds have the same strict payload validation in both versions; the
four conversation kinds require schema 8. Other versions are rejected. Recovery
preserves each accepted schema-7 record's canonical bytes, schema, and meaning,
then appends schema-8 records without rewriting or relabelling the prefix.
Mixed-version logs can be reopened.

The closed durable record vocabulary is `RunStartedRecord`,
`RunResumedRecord`, `NodeStartedRecord`, `NodeSucceededRecord`,
`NodeRoutedRecord`, `NodeExpandedRecord`, `NodeFailedRecord`,
`MemoryCommitRequestedRecord`,
`MemoryCommitSucceededRecord`, `MemoryCommitFailedRecord`,
`BackwardStartedRecord`, `BackwardReadyRecord`, `BackwardFinishedRecord`,
`ConversationAdmittedRecord`, `ConversationEffectRequestedRecord`,
`ConversationEffectObservedRecord`, and `ConversationClosedRecord`. The four node
outcome records carry canonical forward inputs and routes when the recorded
run has `learnable=True`; `learnable=False` omits that capture. Optional model/tool
evaluation evidence is separate. Terminal success
or failure is derived from node transitions. `NodeFailedRecord` stores
the nested `Failure` directly beside any evidence. A causal `NodeRoutedRecord`
stores the optional derived terminal-unreachable `Failure` that replay
revalidates before restoring; no separate forward terminal record exists.
Historical schema-7 runs require no conversation state in the runtime.

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
    memory: dict[str, Memory] | None = None,
    tools: Mapping[str, BaseTool | Callable[..., Any] | list | tuple] | None = None,
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
attempt. A supplied mapping retains policy references for a later explicit
backward call.

For later backward on a memory-enabled log, pass the original policy mapping to
`resume_flow(..., memory=...)`; terminal inspection retains it without opening it.

The log does not record `tools`. To continue an unfinished run whose nodes use
per-run tools or ask the human, pass the same `tools` (including `ask_user`) to
`resume_flow` again. Tool group contents may differ from the original run's:
completed nodes keep their recorded results, and nodes that restart see the
groups passed to `resume_flow`. Every grant must still resolve, as in `run_flow`.

If primary execution is unfinished and eligible for retry, recovery appends one
`RunResumedRecord`, retries the exact ordered set of nodes that have a durable start but no durable
outcome, and continues the ready frontier. `num_workers=None` preserves the
recorded worker count; an integer selects and records the new segment's count.
Fresh inputs, `run_id`, and LLM configuration cannot be supplied during recovery.

Historical conversation records do not block this retry. A retried node may
resend requests and repeat tool actions; recovery does not provide exactly-once
execution or resume an external request midway through. The new accepted start
receives a new sequence, and metadata for its old unfinished start is rejected.
Runtime neither reconstructs a transcript from the run log nor fabricates results
for unknown effects. SessionMemory owns whatever conversation history was saved;
the run log is not a durable replacement for that memory backend.

After incomplete-tail repair, terminal inspection appends no records and invokes
no graph nodes, backward hooks, or memory learners/commits. It projects the last
accepted attempt, including an interrupted Started/Ready/request prefix.
`RunResult.events` contains only forward observers published in the current call;
backward records emit no observers. A resumed AGENT that re-executes publishes
fresh in-loop observers to `on_event`, and nothing is replayed.

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
    An eligible durable `NodeStarted` with no durable outcome is run again.
    Callers using the optional SessionMemory turn capability consume the backend's
    latest saved history, but may resend requests or repeat external effects.
    The log makes engine state deterministic, but cannot make an interrupted
    external effect exactly-once.

## CLI path policy

The Python API remains path-neutral: `run_flow(..., log=None)` uses memory and a
caller chooses any durable path explicitly. The `composer` host adds a user-facing
policy on top:

- fresh `composer run FLOW` selects and reports
  `~/.agent_composer/runs/<run_id>.jsonl` unless `--log PATH` selects another
  path; once runtime commits `RunStarted`, that file is authoritative;
- `composer run FLOW --resume PATH` calls the same recovery boundary without accepting
  fresh inputs, a second log path, or provider/model configuration;
- omitting `-w` on resume preserves the recorded worker count, while an explicit
  value affects the new segment only;
- `composer run` supplies declared built-in memory policies and closes their backends
  after execution drains. Repeated `--memory-storage POLICY=PATH` selects optional
  durable files on fresh runs or resume; omitted policies remain process-local.
  Terminal inspection and old dormant logs open no memory backend;
- each `composer chat` attempt, including a failed-turn restart, receives a separate
  reported default path.

The default CLI directories are mode `0700`; `JsonlRunLog` creates files with
mode `0600`. See [the CLI guide](cli.md#durable-logs-and-recovery) for commands
and the log-sensitivity warning.

## Blocking human interaction

An authored `human_input` node and an AGENT node granted `tools: [ask_user]` call the
run's `ask_user` handler synchronously. A library host passes the handler with each
run as `tools={"ask_user": handler}`; the calling node remains active until the
handler returns or raises:

```python
from agent_composer import load_flow, run_flow

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
`def handler(**request)`. A handler that cannot take all six is rejected before
the run starts. `human_input` uses the trusted metadata path; model-requested
`ask_user` calls can supply only the public `question` and `options` arguments.
`options` is `None` or a cleaned list of up to six short answer labels the model
suggested. Show them as choices, but still let the human type another answer. Each run has its own handler, so
concurrent runs in one process never share one. Within one run, the default
`num_workers=8` lets several nodes (for example two MAP items) call the handler at
the same time, so the handler must be thread-safe.
`node_id` is the expanded node id (for example `loop#1/ask` for a LOOP's second
iteration), so a run id plus `node_id` identifies one question; bind the run id in
the handler's closure when the host needs it. The mapping is the whole tool set,
so a flow that also uses registered tools passes
`tools={**registered_tools(), "ask_user": ask}`. `ask_user` never comes from the
global registry: a run without its own handler fails any HUMAN_INPUT node that
asks, a flow whose AGENT grants `ask_user` fails before the run starts, and
`register_tool("ask_user")` is rejected.

CODE nodes cannot ask. A CODE function does not see the run's tools, so
`invoke_tool("ask_user", ...)` inside it fails. Ask through a `human_input` node,
or pass a per-run tool and call it from a TOOL node.

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
| `events` | post-durable observer events published during this call only; in-loop AGENT observers are excluded |
| `learning` | learning projection of the last observed backward outcome; `None` before an attempt is observed or when no policies were enabled |
| `backward_outcome` | last observed accepted `BackwardOutcome`, or `None` before an attempt is observed |

Call `backward(feedback=None)` explicitly on a retained result. It changes only
the observed backward/learning fields; forward status, values, failure, and events
keep their original meaning.

## Next

- [Flow syntax](syntax.md) — the YAML these functions load.
- [The `composer` CLI](cli.md) — run flows from the terminal.
