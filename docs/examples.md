# Examples

The [`examples/`](https://github.com/ngocbh/yian/tree/main/examples)
directory in the repository contains a set of generic flows, smallest first.
After cloning the repository, run them from its root with
`yian run examples/<file> --input ...` (set a provider/model first — see
[Installation](installation.md)). The directory is not bundled in the current
Python package.

## `hello.yaml` — the smallest flow

One AGENT node, a string in and a string out.

```yaml
id: hello
name: hello
description: Greet someone with a short, friendly one-liner.

input:
  name: !str
output: str
kind: agent
prompt: |-
  Write a short, warm one-sentence greeting addressed to ${name}.
  Just the greeting, no preamble.
```

```console
yian run examples/hello.yaml --input name=Ada
```

Shows compact mode: the flow and its single AGENT node share the typed `input:`
and `output:` boundary, and the prompt reads the auto-wired local `${name}`.

## `summarize.yaml` — one transform

A single AGENT node that condenses a block of text into one sentence. Unlike compact
`hello`, it shows the equivalent full `nodes:` form with explicit input and output wiring.

```console
yian run examples/summarize.yaml --input text="...long text..."
```

## `classify.yaml` — a constrained output

An AGENT node whose `output:` is a `Literal(...)` enum, so the model must answer with
exactly one tag, and the flow returns a one-field object.

```yaml
id: classify
name: classify
input:
  text: !str

nodes:
  label:
    kind: agent
    input:
      text: ${input.text}
    output: Literal(positive, neutral, negative)
    prompt: "Classify the sentiment of: ${text}"
output:
  sentiment: ${label.output}
```

```console
yian run examples/classify.yaml --input text="I love this!"
```

## `human-approval.yaml` — a deterministic human gate

Exercises the **`ask`** built-in. An AGENT node drafts a plan, a TOOL node with
`tool_id: ask` always asks for an `approve` / `revise` answer (a `Literal` typedef),
`case ... on` routes on the typed answer, and a coalesce joins whichever branch ran. The
host answers at once, or its handler returns `pending()` and the host answers later with
`complete()`; either way the answer is the node's ordinary output.

```yaml
approve:
  kind: tool
  tool_id: ask
  input:
    prompt: |-
      Here is the proposed plan:

      ${draft.output}

      Approve it as-is, or send it back to be revised? (approve / revise)
  output: Approval
```

The `prompt:` is an input, so it reads `${draft.output}` directly.

```console
yian run examples/human-approval.yaml --input task="Plan a team offsite"
#   → prompts you to approve or revise
```

`yian run` asks at the terminal and answers inline.

Changed (TS yian): the gate was a `human_input` node; TS yian removed that kind, so this
file needs TS yian.

## `ask-user.yaml` — a model-chosen clarifying question

Exercises the **`ask_user` tool** on an AGENT node. The node calls the
blocking tool **only if** it judges the request ambiguous — otherwise it answers
outright. Whether the call happens depends on the prompt and the model.

```console
yian run examples/ask-user.yaml --input request="Book me a table for dinner"
```

Contrast with `human-approval.yaml`: an `ask` node always asks; `ask_user` asks
only if the model decides to. When it does, the run's ask handler answers, and the
answer returns to the same live model/tool loop.

## `approve-plan.yaml` (escalation)

Asks the lead to approve a plan, and the manager when the lead doesn't answer in a day.
`to:` is a host extension of `ask`: yian's `ask` has no recipient, so the host's ask slot
declares the extra input (`AskTool.inputs`), and yian checks it and passes it through in
the request.

```yaml
id: approve_plan
name: approve_plan
typedefs:
  Approval:
    decision: Literal(approve, reject)
    note: Optional(str)
input:
  plan: !str
nodes:
  lead:
    kind: tool
    tool_id: ask
    input:
      to: lead                           # host extension: who answers
      prompt: |-
        Approve this plan?

        ${input.plan}
      timeout: 24h
    output: Optional(Approval)           # null: the lead didn't answer in time
  escalate:
    kind: case
    cases:
      - when: ${lead.output == null}
        then: manager
  manager:
    kind: tool
    tool_id: ask
    input:
      to: manager
      prompt: |-
        No answer from the lead in 24h. Approve this plan?

        ${input.plan}
      timeout: 48h
    output: Optional(Approval)
output: !Optional(Approval) ${lead.output | manager.output}
```

- A node's `output:` is a type, not a tag (`output: Optional(Approval)`; a tag there is a
  load error, see [Where](syntax.md#where)). The flow's single output takes a tag
  (`output: !T ${...}`).
- `escalate` is a routing-only searched `case` with no `else:`: when the lead answered, no
  branch runs and `manager` is skipped (see [`case`](syntax.md#case-branching) and
  [Exhaustiveness](syntax.md#exhaustiveness)).
- `${lead.output | manager.output}` is the first non-null operand (a coalesce). A skipped
  `manager` reads as null in a flow output (see
  [Skipped branches and flow outputs](syntax.md#skipped-branches-and-flow-outputs)). So the
  output is the lead's answer, else the manager's, else null when neither answered.
- With strict outputs (`output: Approval`), a timeout fails the node with `tool.timeout`
  instead (see [Failure codes](api.md#failure-codes)).

The host's handler returns `pending()` and answers later with `complete()`; the run ends
`waiting` in between, and each `ask`'s `timeout` is a deadline on its call.

## `chat.yaml` — a turn-by-turn chat

A conversational assistant on an inbox that folds older turns into a rolling summary. The
flow takes `messages: !Inbox(str)`: the host sends each line the user types to it, and closes
it to end the chat. A LOOP runs one turn per batch of messages and carries
`{summary, history, closed}`, where `history` is a list of turns, each a list of messages.
Each turn starts two nodes side by side, a TOOL `take` and a CALL of
[`compact_history.yaml`](#compact_historyyaml-a-rolling-summary):

```yaml
compacted:                      # beside `heard`: compacts the last reply's history
  kind: call
  call: compact_history
  input: {summary: "${input.summary}", history: "${input.history}"}
heard:
  kind: tool
  tool_id: take                 # waits until a message arrives or the inbox closes
  input:
    from: ${input.messages}
    take: all
```

`take: all` returns everything sent since the last turn as `{events, closed}`. While no
message is waiting the run ends `waiting`, and `compacted` runs before it does, so a fold
happens while the run waits for the next message. A routing-only `case` sends a non-empty
`events` to `answered`, a CALL of the `respond` def whose AGENT replies. `answered` gets the
summary and the kept turns flattened into one list of messages
(`${[m for t in compacted.output.history for m in t]}`), and the reply's system prompt ends
with `Conversation so far: ${summary}` (empty before the first fold). The turn's output
appends the new turn: `${compacted.output.history + (answered.output :- [])}`. `closed` comes
from the take, so a take that returns events with `closed: true` is answered and then ends the
loop, and a closed take with no events skips `answered`. The flow's output is
`{summary, history}`, with the history flat.

The AGENT records only its reply, and the history is built in derived bindings (CALL and END
commits are pointers), so the run log grows linearly: per turn a constant, plus one summary
per fold, plus the flow's output once. Every reply request holds at most the budget, the new
message, the summary and the system text.

Chat with it at the terminal (TS yian; see [Chatting with a flow](cli.md#chatting-with-a-flow-yian-chat)):

```console
yian chat examples/chat.yaml
#   → type a line; the reply prints
#   → /exit (or Ctrl-D) closes the inbox and ends the chat
```

On a turn that folds, the reply prints, and the next prompt appears once the summary is
written: `yian chat` shows it after the run step returns (#1091). The summary itself is never
printed.

Changed (TS yian): the chat asked through a `human_input` node each turn and ended on
`/exit`; TS yian removed that kind, so this file needs TS yian. It also uses list
comprehensions, `len()` and slices, which are TS-only.

## `compact_history.yaml` — a rolling summary

Keeps a chat history's text under a budget by folding older turns into a rolling summary;
`chat.yaml` calls it each turn, and any flow that carries a history can. Its inputs:

| Input | Default | Meaning |
|---|---|---|
| `summary` | (required) | the summary so far (`""` before the first fold) |
| `history` | (required) | `list(list(Msg))`: one list of messages per turn |
| `budget` | `32000` | characters of message text before compacting |
| `keep` | `4` | whole turns kept verbatim |

It returns `{summary, history}`. A routing-only `case` checks
`len(input.history) > input.keep` and the history's text length
(`len(join([m.text for t in input.history for m in t], ""))`) against `budget`:

- **Under budget, or at most `keep` turns:** the case takes no branch, `folded` is skipped
  and the output's `:-` defaults pass both inputs through, with no model call.
- **Over budget:** `folded` calls the `fold` def, whose AGENT `summarize` writes a new summary
  from the old one and the older turns, and the last `keep` turns stay verbatim.

`keep` counts whole turns, so the cut always falls between turns:
`history[:len(history) - keep]` is folded and `history[len(history) - keep:]` is kept. The
slices sit in the `folded` CALL's input binding, not the AGENT's: a CALL's binding is rebuilt,
never recorded, and the AGENT binds plain references, so a learnable run records constant-size
routes. `keep: 0` (or less) folds every turn.

The summarizer is an ordinary AGENT: after the normal model retries a failed summary fails the
run, and a failed run is terminal, so the chat ends. A node fallback (`on_failure:`, #15) will
let it keep the history and retry on the next turn. The AGENT is never named `reply`, because
`yian chat` prints any node named `reply`.

`Msg` is text-only (`{role, text}`) for now. When chats carry tools (#1025), `Msg` becomes the
built-in `Message`, and the budget and the prompt read text parts, so this flow's interface
changes then.

Changed (TS yian): list comprehensions, `len()` and slices are TS-only, so this file needs TS
yian.

## `decision-brief.yaml` — a realistic multi-stage flow

The fullest example. Given a decision question it:

1. **fans out** to three independent angle AGENT nodes (the case for, against, the
   risks) — no edges between them, so a worker pool can run them in parallel;
2. **fans in** to a synth AGENT node that picks a typed `Decision` enum;
3. **routes** on that decision with `case ... on` to a decision-specific brief;
4. **finalizes** by combining the decision with whichever brief ran (an n-ary
   `|` coalesce consumed downstream);
5. returns a **multi-field object**, guarded by an `assert`.

```console
yian run examples/decision-brief.yaml --input question="Should a small team adopt a monorepo?" --num-workers 4
```

It uses a `Literal` typedef, an `Optional` input with an expression fallback, fan-out +
fan-in, a typed enum output, `case ... on` with `else`, a coalesce, a multi-field output,
and asserts — all with AGENT nodes, so it runs against any provider with no extra
Python.

!!! note "Why the example uses an enum"
    The synth node's verdict is a `Decision` enum because the following `case`
    routes directly on those tags. The enum itself uses structured generation,
    as do record, number, boolean, and list outputs from AGENT nodes; only `str`
    and `Optional(str)` stay on the free-text path. See
    [Flow syntax → `agent`](syntax.md#agent-the-llm-backed-agent-node).

## Next

- [Flow syntax](syntax.md) — the full reference for these constructs.
- [Python API](api.md) — drive these flows from code.
