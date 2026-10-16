# Agent

In Yian, an **agent** is a composable transformation from a named,
typed input record to an output. A flow exposes an output value, which may be a
scalar, list, record, or `None`. A runtime node exposes a node outcome: a value,
a route, or a request to expand a child flow.

“Agent” is a conceptual umbrella, not one Python class or YAML kind. The
transformation can be represented by one node, a flow of nodes, or a child flow
used inside another flow.

## What an agent is composed of

Every agent has two parts:

1. A **boundary** says what may enter and what must leave. A flow boundary is a
   named, typed `input:` record plus one `output:` value. A node boundary is its
   bound local input record plus one node outcome.
2. **Behavior** performs the transformation behind that boundary.

The shape of the behavior depends on the representation:

| Representation | Shape of its behavior |
| --- | --- |
| Node | One operation, atomic at the graph-state boundary. |
| Flow | A workflow graph whose structure is fixed by the author and whose nodes are connected by data and ordering dependencies. |
| Child flow | A flow graph entered from another flow through CALL, MAP, or LOOP. |

Atomic at the graph-state boundary does not mean “one machine instruction” or
even “one external call.” For example, one AGENT node may make several model and
tool calls before returning. The scheduler still sees one node evaluation with
one outcome. For example, other node kinds provide different atomic behavior:
CODE runs a Python function, TOOL invokes a registered capability, CASE chooses
a route, HUMAN_INPUT asks a person, and WAIT blocks until a deadline.

The concrete **AGENT node** (`kind: agent`) is the LLM-backed atomic behavior.
Its model request deserves a closer look because prompt text, memory, tools, and
the output contract enter that request in different ways.

## Terms

| Term | Meaning |
| --- | --- |
| **Agent** | The conceptual transformation from a named input record to an output. |
| **Node** | One schedulable step in a flow. It receives a bound local input record and returns a node outcome. |
| **AGENT node** | The LLM-backed node kind authored as `kind: agent`. `AgentNode` is its Python class. |
| **Flow** | A graph of nodes packaged behind one input and output boundary. |
| **Root flow** | The flow passed to `run_flow`. |
| **Child flow** | A flow applied inside another by CALL, MAP, or LOOP. “Subflow” is a common synonym; these docs use **child flow**. |
| **Run** | One invocation of a root flow, including its child-flow expansions, under one run log. |

A node outcome is more general than a value: most nodes produce a value, CASE
routes execution, and CALL/MAP/LOOP expand an authored child graph whose terminal
value becomes their output. The engine, not the node, records and applies that
outcome.

## Inside an AGENT node

An AGENT node may make one or more model requests before returning. Each request
has ordered messages and may have turn-dependent fields alongside them.

<figure class="ac-request-visual" aria-labelledby="ac-request-visual-title">
  <figcaption id="ac-request-visual-title">
    <strong>What one model request contains</strong>
    <span>Only messages have a Yian-defined order.</span>
  </figcaption>

  <div class="ac-request-frame">
    <section class="ac-request-messages" aria-labelledby="ac-request-messages-title">
      <header class="ac-request-section-title">
        <h3 id="ac-request-messages-title">Messages</h3>
        <span class="ac-request-badge">ordered</span>
      </header>

      <ol class="ac-request-message-stack">
        <li class="ac-request-message" data-message="system">
          <div class="ac-request-message-heading">
            <span class="ac-request-step" aria-hidden="true">1</span>
            <code>SystemMessage</code>
          </div>
          <div class="ac-request-system-content">
            <div class="ac-request-content-block">
              <strong>System instruction</strong>
              <span>Rendered <code>system_prompt:</code>, or the framework instruction</span>
            </div>
            <div class="ac-request-memory">
              <div class="ac-request-memory-title">
                <strong>Recalled long-term memory</strong>
                <span class="ac-request-badge">outer → local</span>
              </div>
              <ol class="ac-request-memory-frames" aria-label="Long-term memory recall order">
                <li>Root-flow memories</li>
                <li>Enclosing child-flow memories</li>
                <li>AGENT memories</li>
              </ol>
              <small>Only non-empty recalled memories appear.</small>
            </div>
          </div>
        </li>

        <li class="ac-request-message" data-message="human">
          <div class="ac-request-message-heading">
            <span class="ac-request-step" aria-hidden="true">2</span>
            <code>HumanMessage</code>
          </div>
          <div class="ac-request-content-block">
            <strong>Current task</strong>
            <span>Rendered <code>prompt:</code></span>
          </div>
        </li>

        <li class="ac-request-message ac-request-message--optional" data-message="additional">
          <div class="ac-request-message-heading">
            <span class="ac-request-step" aria-hidden="true">3</span>
            <strong>Additional messages</strong>
            <span class="ac-request-badge">when needed</span>
          </div>
          <div class="ac-request-additions">
            <div class="ac-request-content-block">
              <strong>In-node interaction history</strong>
              <span>Assistant tool call + <code>ToolMessage</code> result</span>
            </div>
            <div class="ac-request-content-block">
              <strong>Structured-output guidance</strong>
              <span>Fallback schema instruction or retry correction</span>
            </div>
          </div>
        </li>
      </ol>
    </section>

    <aside class="ac-request-fields" aria-labelledby="ac-request-fields-title">
      <header class="ac-request-section-title">
        <h3 id="ac-request-fields-title">Alongside messages</h3>
        <span class="ac-request-badge">turn-dependent</span>
      </header>
      <div class="ac-request-field">
        <strong>Tool/function schemas</strong>
        <span>Name · description · input schema, bound on tool-calling turns</span>
      </div>
      <div class="ac-request-field">
        <strong>Structured-output binding</strong>
        <span>Provider-supported schema/control; otherwise guidance becomes a message</span>
      </div>
    </aside>
  </div>

  <div class="ac-request-route" aria-label="The provider adapter serializes the request for the model">
    <span class="ac-request-route-arrow" aria-hidden="true">↓</span>
    <div class="ac-request-adapter">
      <strong>Provider adapter</strong>
      <span>Chooses the API and wire representation</span>
    </div>
    <span class="ac-request-route-arrow" aria-hidden="true">↓</span>
    <div class="ac-request-model"><strong>Model invocation</strong></div>
  </div>

  <p class="ac-request-note">
    <strong>Long-term</strong> means memory that can be recalled across runs.
    Storage lifetime and persistence come from the host memory implementation;
    built-in <code>ReflectiveMemory</code> is process-local.
  </p>
</figure>

Only the messages have a Yian-defined sequence. The provider adapter owns
the eventual token and control representation, so Yian does not claim
that tool descriptions occur before or after recalled memory in a flattened
token stream.

<figure class="ac-turn-visual" aria-labelledby="ac-turn-visual-title">
  <figcaption id="ac-turn-visual-title">
    <strong>One AGENT node may make several model requests</strong>
    <span>The scheduler still observes one node evaluation.</span>
  </figcaption>

  <ol class="ac-turn-timeline">
    <li>
      <strong>Recall and build the initial request</strong>
      <p>Recall long-term memory once, then create the system and human messages.</p>
    </li>
    <li>
      <div class="ac-turn-title">
        <strong>Tool rounds</strong>
        <span class="ac-request-badge">optional · repeatable</span>
      </div>
      <p>Assistant tool call → execute a granted tool → append the call and result → request again.</p>
    </li>
    <li>
      <div class="ac-turn-title">
        <strong>Structured generation and correction</strong>
        <span class="ac-request-badge">optional</span>
      </div>
      <p>Use a provider binding or add fallback schema guidance. Invalid output adds a correction and retries. Without a tool loop, this may be the initial model request.</p>
    </li>
    <li>
      <strong>Return one AGENT node outcome</strong>
      <p>The validated value crosses the graph-state boundary.</p>
    </li>
  </ol>
</figure>

With both tools and structured output, structured generation starts after the
tool loop and uses the accumulated message history.

A host can watch these requests live. With `on_event`, every request —
including structured attempts and memory summary preparation — is bracketed by
`ModelCallStarted`/`ModelCallFinished` (the latter with provider `TokenUsage`),
and every granted tool by `ToolCallStarted`/`ToolCallFinished`. These are live
callbacks only; the node value, the run log, and retained memory are unchanged.

With `on_stream`, each chat request's visible text arrives as `StreamChunk`s
sharing that request's `exchange_id`. A completed request ends with one empty
`final=True` chunk. A failed request ends without one; its `ModelCallFinished.error`
is set, and a host should discard that request's partial text. The node value is
authoritative: streamed text can differ from it in the whitespace between content
blocks.

A [steering](api.md#steering) interrupt that cuts a streamed request still ends it
with the `final=True` chunk; its `ModelCallFinished` has no `error` and carries
the usage received so far. Each tool call the interrupt skipped is reported as a
`ToolCallStarted` followed by `ToolCallFinished(error="interrupted")`.

### How YAML becomes the request

| YAML or host value | Role in execution |
| --- | --- |
| Node `input:` | Binds the local values available to prompt rendering and memory recall. It is not independently dumped into a message. |
| AGENT `variables:` | Supplies typed local defaults to prompt rendering. Each invocation receives a fresh copy; see [agent-owned variables](syntax.md#agent-owned-variables). |
| `system_prompt:` | Renders local `${name}` values into the system instruction, replacing the framework instruction. See [system prompt](syntax.md#system-prompt-system_prompt). |
| `prompt:` | Renders local `${name}` values into the initial `HumanMessage`. |
| `long_term_memory:` plus host `memory=build_memories(flow)` | Supplies one memory context: recalled notes become labelled system sections (`{"index", "memory", "text"}`); a session memory contributes structured prior messages. |
| `tools: [name, ...]` | Resolves the granted ids from the run's tools (registered tools, or the host's `tools=` mapping) and binds their schemas beside the messages; an AGENT with `tools:` or `skills:` runs the tool loop, and one without them makes one model call. A name may be a host-filled tool group, which expands to its tools. |
| `skills: [name, ...]` or `skills: all` plus host `skills=` | Lists each granted skill's name and description in the system message and binds the built-in `load_skill` tool, which returns a skill's instructions or a bundled file on demand. See [agent skills](syntax.md#agent-skills-skills). |
| Structured `output:` | Selects a provider-supported structured-output binding or a fallback schema instruction, then validates the returned value. A bare `str` remains free text. |
| `steerable: true` plus host `steering=` | Sends the host's queued steer messages as `HumanMessage`s with the next model request, and stops the node on a host interrupt (a streamed reply is cut); the node succeeds with its text so far. See [steering a running agent](syntax.md#steering-a-running-agent-steerable). |
| `llm_config:`, `retries:`, `env:` | Controls execution; these values are not additional prompt messages. |

The system instruction is the rendered `system_prompt:` when the node has one,
and the framework instruction otherwise. An authored prompt replaces the framework
instruction rather than extending it. The skill listing that granted `skills:` add
follows it, then recalled memory sections. Every model request of the node starts
with this same instruction: the one model call, each tool-loop round, and
structured generation. The memory context is opened with the instruction without
the listing, so skill grants do not change recall. A session transcript keeps past `load_skill` calls and
results, like any tool call, even if a later run grants no skills.

### Memory is an outer-to-local stack

An AGENT node recalls a stack of memories, in order:

1. each enclosing flow's memories, outermost flow first, through CALL, MAP,
   and LOOP;
2. the AGENT node's own memories.

A memory key that appears twice is recalled once, at its outermost position.
Each memory key contributes at most one labelled section; empty recalls are
left out. A session memory contributes a structured transcript instead,
preserving message roles and tool-result links. Only a key with exactly one
declaring AGENT may supply a transcript, and one context holds at most one.
Each memory is written only by its own learner. MAP items and LOOP iterations
reuse the same memories rather than creating new ones per occurrence. A
`ProfileMemory` is one key like any other: every AGENT whose stack holds it
sees the whole document.

An AGENT inside an imported child recalls the importing flow's memories by
position, plus its own keys such as `lib.notes`. A child's declaration never
binds to a parent's memory.

```yaml
id: layered_memory
name: layered_memory
long_term_memory: {notes: episodic}
defs:
  research:
    long_term_memory: {lessons: reflective}
    input:
      topic: !str
    nodes:
      investigate:
        kind: agent
        long_term_memory: {chat: session}
        input: {topic: "${input.topic}"}
        output: str
        prompt: Investigate ${topic}.
    output: ${investigate.output}
input:
  topic: !str
nodes:
  run: {kind: call, call: research, input: {topic: "${input.topic}"}}
output: ${run.output}
```

For `investigate`, the stack is `notes` (root flow), then `lessons`
(`research`), then `chat` (the AGENT). No `${memory}` reference is required. A
Python host passes `memory=build_memories(flow)` to `run_flow`. `learnable=True`
independently retains accepted execution; forward returns before memory learning
or learned-mutation commits. The host calls `result.backward(feedback)`
explicitly afterward. Every memory key then gets one `learn` call with its own
batch, including empty ones, and can propose writes only for itself. New
committed lessons can affect later fresh runs when the host reuses the same
memories. See [Injecting agent memory](api.md#injecting-agent-memory) for the
host API.

For API conversation continuity, declare a
[`SessionMemory`](api.md#sessionmemory-foundation) on the AGENT, as `chat` is
above. A session key has exactly one declaring AGENT, and an AGENT declares at
most one session. That key holds one conversation, shared by every call site of
the AGENT; other memories in the stack still provide recall. History advances
during forward without learning. Repeated invocations take turns in admission
order, including LOOP iterations and MAP items; MAP does not fork
conversations. Each new call applies current instructions and tool grants, and
restoring historical tool calls never executes them.

### AGENT backward uses the recorded invocation

A reached AGENT makes at most one tool-free model call using its accepted
authored/rendered prompt, delivered inputs, resolved variables and type domains,
output or failure, effective model configuration, and supplied feedback/context.
The base system prompt, ordered recalled sections, and visible interactions are
retained separately. Tool interactions are inert evidence in this call; tools are
not granted or executed. Defaults, prompt rendering, recall, and model selection
are not repeated. Partial established locals survive later forward failures,
with unavailable values distinguished from empty ones.

The model returns validated input, self, and local-variable feedback. This is
advice, not an automatic variable update, graph rewrite, or rerun. With no
effective feedback, the AGENT backward call is skipped, while every memory key
still learns from accepted history. Memory learners are separate
callbacks from this AGENT model call. Terminal resume invokes neither; see
[explicit backward feedback](api.md#explicit-backward-feedback).

## Representing agents in YAML

At a full-flow boundary, top-level `input:` declares parameter types and
top-level `output:` binds the returned value. Within the graph, a node's
`input:` binds values from flow inputs or earlier nodes; value-producing AGENT
nodes and CODE nodes use `output:` to declare the value they produce. References such
as `${draft.output}` both move data and define graph edges.

### One node: compact AGENT node

When one AGENT node is enough, compact mode makes the flow and that node share
one definition:

```yaml
id: classify
name: classify
input:
  text: !str
output: Literal(positive, neutral, negative)
kind: agent
prompt: |-
  Classify ${text} as positive, neutral, or negative.
  Return exactly one label.
```

Here top-level `input:` is the shared flow/node signature, and `output:` is the
AGENT node's output type re-exported as the flow output.

### Several nodes: a flow

Use a full flow when the transformation needs multiple steps:

```yaml
id: summarize
name: summarize
input:
  article: !str
nodes:
  draft:
    kind: agent
    input:
      article: ${input.article}
    output: str
    prompt: "Summarize this article in one sentence: ${article}"

  count_words:
    kind: code
    input:
      text: ${draft.output}
    output: int
    code: |
      return len(inputs["text"].split())

output:
  summary: ${draft.output}
  word_count: ${count_words.output}
```

The flow itself is the agent at the public boundary. `draft` and `count_words`
are its internal nodes, and their references determine the dependency graph.

### Reuse: a child flow

Put a reusable agent in `defs:` and apply it with a CALL node:

```yaml
id: review_draft
name: review_draft
input:
  draft: !str

defs:
  review:
    input:
      text: !str
    nodes:
      reviewer:
        kind: agent
        input:
          text: ${input.text}
        output: str
        prompt: "Review this draft and suggest one improvement: ${text}"
    output: ${reviewer.output}

nodes:
  run_review:
    kind: call
    call: review
    input:
      text: ${input.draft}

output: ${run_review.output}
```

`review` is a child flow with its own boundary and graph. CALL runs it once;
MAP applies a child flow to a list, and LOOP repeatedly applies one while
threading a carried record.

## Planned extensions

An AGENT grants tools by id (registered, or passed per run with `tools=`) with
`tools:` and host-supplied `SKILL.md` packages with `skills:`. A host tool can
take a `ToolContext` first to receive its call id and the host's app object (see
[Tool context](api.md#tool-context)). Host-controlled discovery of project-local
tools is tracked separately.

## The authored structure stays fixed

The author defines the graph. An LLM may generate values and choose among tools
explicitly granted to its AGENT node, but it does not rewrite the flow. Runtime
CALL/MAP/LOOP expansion instantiates child flows already defined by the author.

Nodes are **pure at the graph-state boundary**. A node may call an LLM, run
Python, invoke a tool, wait, or ask a human, but it receives only its bound local
input and capabilities. It cannot mutate the variable pool, scheduler, or graph.
It returns an outcome; the engine records that outcome and updates run state.

Next, read [Architecture](architecture.md) for the path from YAML to a `RunResult`,
or use [Flow syntax](syntax.md) as the complete authoring reference.
