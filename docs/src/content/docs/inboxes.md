---
title: "Inboxes and waiting"
description: "Wait for messages, time or people: take, sleep, ask, and durable suspend and resume."
---

This page explains how a flow waits for messages, time or people. Three built-in tools wait, each
called from a `kind: call` node:

| `call` | Waits for | Output |
|---|---|---|
| `take` | messages sent to an inbox input, or to any of several | `{events: list(T), closed: bool}`; from several, `{events: list({inbox, event: T}), closed: bool, closed_inboxes: list(...)}` |
| `sleep` | a time | `null` |
| `ask` | a person's answer | the answer, typed by `output:` |

A waiting run holds no worker and no process. It stops with the status `waiting`, its run log on
disk, and moves on later, in the same process or another one.

## Inbox inputs

An inbox is a flow input whose values arrive while the run goes on. Declare it with an `Inbox(T)`
tag, where `T` is the type of each message:

```yaml
id: collect
name: collect
input:
  notes: !Inbox(str)
nodes:
  gather:
    kind: loop
    input:
      notes: ${input.notes}
      seen: !list(str) []
      closed: false
    while: not closed
    nodes:
      heard:
        kind: call
        call: take
        input:
          from: ${input.notes}
          take: all
    output:
      seen: ${input.seen + heard.output.events}
      closed: ${heard.output.closed}
output: ${gather.output.seen}
```

This flow collects notes until its inbox closes, then returns them all.

- An inbox input is required and has no default. A run takes no value for it: messages come
  through `send` (see [Sending messages](#sending-messages)). Passing one with `--input` or
  `--inputs` is a usage error.
- Read an inbox only whole, as `${input.notes}`, in one of three places: a `take`'s `from:` (or an
  item of its list), an agent's `steer_inbox:` (see [Agents](/agents/)), or an `Inbox` input of a
  `kind: call` node's or a `loop`'s child.
- Anything else is a load error: text around it, a field read, a code or agent input, a `case`,
  a `map`, an inline `call(...)` of a flow, a flow output. An inline `call(take, from=...)` is a
  `take`.
- A loop passes an inbox to every run of its body and leaves it out of its output.
- A child that a `map` runs can't declare an inbox.
- An inbox nothing reads is a load error:

```text
inbox input `notes` is never read; take from it, or pass it to a call
```

## Take messages

A `take` reads messages from an inbox:

```yaml check=fragment
heard:
  kind: call
  call: take
  input:
    from: ${input.notes}        # required: an inbox input
    take: all                   # one (the default) or all
    timeout: 30m                # optional
```

| Input | Meaning |
|---|---|
| `from:` | The inbox to read, as `${input.<name>}`, or a list of them (see [Take from several inboxes](#take-from-several-inboxes)) |
| `take:` | `one` takes the oldest unread message; `all` takes every unread one |
| `timeout:` | How long to wait for a message; see [Durations](#durations) |

| When the take runs | Its output |
|---|---|
| Messages are unread | One (`take: one`) or all of them, at once; `closed: true` when the inbox is closed and this take read the last one |
| Nothing is unread and the inbox is open | It waits for the next message |
| The timeout passes first | `{events: [], closed: false}` |
| The inbox is closed and nothing is left | `{events: [], closed: true}` |

A take on a closed inbox still gets the messages sent before the close. `take`'s output is fixed, so `output:`, `asserts:` and `retries:`
are load errors.

### Take from several inboxes

`from:` can list inboxes of one message type. The take reads them together and wakes for a
message to any of them:

```yaml check=fragment
heard:
  kind: call
  call: take
  input:
    from:
      - ${input.messages}
      - ${input.queued}
    take: all
```

In flow style, quote each item: `from: ["${input.messages}", "${input.queued}"]`.

- `take: one` takes the oldest unread message of the listed inboxes, by when it was sent;
  `take: all` takes every unread one, in the order they were sent.
- Its output names each message's inbox by its input name, and the closed inboxes:
  `{events: list({inbox, event: T}), closed: bool, closed_inboxes: list(...)}`.
- The names are those of the flow the take is in. A def or loop body that takes from inboxes passed
  in as `chat` and `later` names them `chat` and `later`, so a caller comparing `inbox` with its
  own names (`messages`, `queued`) matches nothing.
- `closed_inboxes` lists the inboxes that are closed with nothing left to read. `closed` is true
  when that is all of them; until then the take waits on the open ones.
- `timeout:` covers the whole take. When it passes: `events: []`, `closed: false`.
- An empty list, an inbox listed twice, an item that isn't an `Inbox` input, or inboxes of two
  message types are load errors. An inbox a caller passes in under two names fails the take with
  `contract.input`.

### A queue the steers don't read

An agent reads its `steer_inbox:` before each model call, so a message sent there joins the turn
that is running. A second inbox, read only by the take that starts a turn, holds messages for the
next turn:

```yaml
id: queue
name: queue
input:
  messages: !Inbox(str)         # steers the running reply
  queued: !Inbox(str)           # waits for the next turn
nodes:
  chat:
    kind: loop
    input:
      messages: ${input.messages}
      queued: ${input.queued}
      closed: false
    while: not closed
    nodes:
      heard:
        kind: call
        call: take
        input:
          from:
            - ${input.messages}
            - ${input.queued}
          take: all
      route:
        kind: case
        cases:
          - when: ${heard.output.events != []}
            then: reply
      reply:
        kind: agent
        steer_inbox: ${input.messages}
        input:
          said: ${[e.event for e in heard.output.events]}
        prompt: "Answer: ${said}"
        output: !Optional(str)
    output:
      closed: ${heard.output.closed}
output: null
```

The host sends a message to `messages` to change the running turn, or to `queued` to keep it for
the next one. To move a queued message into the running turn, it takes it back from `queued` with
`withdraw(log, "queued", { events: [seq] })` and sends it to `messages`. The `seq` is what `send`
returned, or an entry of `inspectRun(log).unread`. To end the chat, close both inboxes: while one
is open, the take waits on it.

## Sleep

A `sleep` waits until a time. Give exactly one of `until:` and `for:`:

```yaml check=fragment
cool_off:
  kind: call
  call: sleep
  input:
    for: 1h                     # or until: 2026-05-21T09:00:00Z
```

- `until:` is a date or datetime, or a reference to one. A value without an offset is read in
  UTC. A time already past returns at once.
- `for:` is a duration.
- Its output is `null`. `output:`, `asserts:` and `retries:` are load errors.

## Ask a person

An `ask` sends a question to a person and waits for the answer. It needs one of `prompt:`,
`questions:` and `adaptive_questions:`:

```yaml
id: plan_trip
name: plan_trip
typedefs:
  Trip: {city: str, nights: str}
nodes:
  plan:
    kind: call
    call: ask
    input:
      questions:
        - question: Which city?
          header: city
        - question: How many nights?
          header: nights
          options:
            - label: "2"
            - label: "5"
      timeout: 24h
    output: !Optional(Trip)     # null: nobody answered in time
output: ${plan.output}
```

| Input | Meaning |
|---|---|
| `prompt:` | One free-text question |
| `questions:` | A list of 1 to 4 questions (`YIAN_MAX_QUESTIONS` changes the 4), each with `question:` and a unique `header:`; optional `options:` (each a `label:`, unique within the question, and an optional `description:`) and `multi_select:` |
| `adaptive_questions:` | A block whose `prompt:` asks a model to write the questions from the node's other inputs |
| `timeout:` | How long to wait; see below |
| anything else | Refused before the run, unless the host's `ask` handler declares it; then it reaches the handler, such as `to: lead` |

- `output:` is the answer's type. A `questions:` ask answers a record keyed by `header`. Without
  `output:`, a `questions:` ask answers an `object`, and a `prompt:` ask takes whatever the host
  answers (text at the terminal).
- With a `timeout:`, an unanswered ask is `null` when `output:` is `Optional(...)`. Otherwise it
  fails with `tool.timeout`.
- `asserts:` apply to the answer. `retries:` is a load error.
- `call: ask_user` is the same as `ask`.

`yian run` asks at the terminal and waits for your answer there, so `timeout:` has no effect in
it. A host's handler may hold the question instead; then the run waits until the host
answers it (see [Embedding yian](/embedding/)).

An agent can also ask a person, only when its model chooses to, through the `ask_user` tool (see
[Agents](/agents/)).

## Durations

`timeout:` and `for:` take a positive integer and a unit:

| Unit | Example | Seconds |
|---|---|---|
| `s` | `90s` | 90 |
| `m` | `30m` | 1,800 |
| `h` | `24h` | 86,400 |
| `d` | `2d` | 172,800 |

A plain number is a load error: `timeout: 86400` fails with an error that ends
``a duration is text such as 24h, got 86400; write 86400s``. A duration read from a reference
must be typed `str` (`timeout: !str ${input.wait}`) and is checked when the node runs. A literal
duration takes no tag.

## Sending messages

A host program moves messages and answers into a run through its log:

| Call | What it does |
|---|---|
| `send` | Adds a message to an inbox |
| `withdraw` | Takes back the messages no take or agent has read yet: everyone's, one sender's, or the ones you name |
| `close` | Closes an inbox: later sends are refused, and takes still read what is queued |
| `complete` | Answers a call a host's tool handler held, such as an `ask` |
| `release` | Runs an agent's tool call the host's [`beforeToolCall`](/embedding/#gate-tool-calls) held |

A `withdraw` by one sender (`by`), or of the messages you name (`events`), leaves the other
messages: the messages kept go to the takes that lost theirs, oldest first, as new messages would;
a take that lost nothing keeps what it holds. A named message already read, or taken back by an
earlier `withdraw`, stays as it is, and `withdraw` says which of the two it was.

None of these moves the run; the host then calls `advance`. A message may nest two levels less than
`YIAN_MAX_VALUE_NESTING` (98 by default), since a take reads it inside `{events: [...]}`; `send`
refuses a deeper one. [Embedding yian](/embedding/) covers the calls in full.

On a terminal, `yian run` does this for you. It runs a flow with one `Inbox(str)` or
`Inbox(Message)` input (pick one with `--inbox NAME` when there are several), sends each line you
type, and prints the output of each node named `reply`:

```sh
yian run collect.yaml
```

| You do | The chat calls |
|---|---|
| Type a line | `send`, then `advance` |
| Type `/exit` or `/quit`, or press Ctrl-D | `close`, then `advance` until the run ends |
| Press Esc | `withdraw`, puts the unread lines back in the prompt, then stops the reply |
| Press Ctrl-C | during a turn, stops the reply; at the prompt, or a second time within 2 s, cancels the run |

A message larger than `YIAN_MAX_EVENT_BYTES` (1 MiB by default) is refused, and the chat prints
`not sent: <reason>`. While the prompt is up, the chat moves the run on when a `sleep` or a
`timeout` comes due. `yian run --resume LOG` picks up a chat that stopped. See
[Chatting on a terminal](/cli/#chatting-on-a-terminal).

## Suspend and resume

When every node that can still run is waiting, the run stops with the status `waiting`. Its log
holds everything it needs to go on. Off a terminal, or whenever it doesn't chat, `yian run` exits 0
and prints, on stderr, what it waits on and how to continue:

```text
run waiting: gather#0/heard (take)
continue with: yian run --resume /home/you/.yian/runs/<run id>/log.jsonl
```

With `--json`, stdout holds the same as one document:

```console
$ yian run collect.yaml --json
{"format":"yian.cli.v1","status":"waiting","waits":[{"call_id":"…/gather#0/heard","deadline":null,"inbox":"notes","node":"gather#0/heard","settled":false,"tool":"take"}],"wake_at":null}
```

| Field | Meaning |
|---|---|
| `waits[].node` | The waiting node; `#0` is the first run of a loop body |
| `waits[].tool` | `take`, `sleep`, `ask` or a host tool |
| `waits[].inbox` | The inbox a take reads (a list, for a take of several), else `null` |
| `waits[].deadline` | When the wait times out, else `null` |
| `waits[].settled` | Whether it already has its answer, timed out or was closed, so the node wakes on the next move |
| `wake_at` | The earliest deadline of a wait that isn't settled, else `null` |

To go on, resume the log (this run was started with `--log ./run.jsonl`):

```console
$ yian run --resume ./run.jsonl
$ yian status ./run.jsonl
status: waiting
```

- `--resume` wakes every node whose wait is settled or past its deadline, and runs on from there.
  When nothing can wake, it appends nothing and prints the waiting result again.
- `--resume` doesn't wait for a deadline itself. Run it again after `wake_at`, or let a host do
  it.
- `--resume` runs the flow the log recorded, not the file on disk, so an edited flow needs a new
  run.
- `yian status` reads a log without changing it. With `--json` it adds `ready` (every call a
  waiting node waits on is settled, a held call it waits on is answered, closed or released, or an
  interrupt stops it) and `wake_at` (its earliest deadline). A run is due for `--resume` once
  `wake_at` has passed, though `ready` stays `false`.

A run whose process dies resumes the same way: nodes that started and didn't finish run again,
before any waiting node wakes.
[How yian works](/how-it-works/) explains how the log makes this safe.
