# Examples

The [`examples/`](https://github.com/ngocbh/agent-composer/tree/main/examples)
directory ships a set of generic flows, smallest first. Run any of them with
`ac run examples/<file> --input ...` (set a provider/model first — see
[Installation](installation.md)).

## `hello.yaml` — the smallest flow

One AGENT, a string in and a string out.

```yaml
input:
  name: str
nodes:
  greet:
    kind: agent
    input:
      name: ${input.name}
    output: str
    prompt: |-
      Write a short, warm one-sentence greeting addressed to ${name}.
output: ${greet.output}
```

```console
ac run examples/hello.yaml --input name=Ada
```

Shows the core shape: typed `input:`, a single AGENT leaf, a local prompt
reference (`${name}`), and the flow `output:`.

## `summarize.yaml` — one transform

A single AGENT that condenses a block of text into one sentence. Same shape as
`hello`, with a longer prompt.

```console
ac run examples/summarize.yaml --input text="...long text..."
```

## `classify.yaml` — a constrained output

An AGENT whose `output:` is a `Literal[...]` enum, so the model must answer with
exactly one tag, and the flow returns a multi-field object.

```yaml
nodes:
  label:
    kind: agent
    input:
      text: ${input.text}
    output: Literal[positive, neutral, negative]
    prompt: "Classify the sentiment of: ${text}"
output:
  sentiment: ${label.output}
```

```console
ac run examples/classify.yaml --input text="I love this!"
```

## `human-approval.yaml` — a deterministic human gate

Exercises the **`human_input`** node. An agent drafts a plan, a `human_input`
node always blocks for an `approve` / `revise` answer (a `Literal` typedef),
`case ... on` routes on the typed answer, and a coalesce joins whichever branch
ran. The gate returns the answer as an ordinary node output; it does not pause or
checkpoint the flow.

```console
ac run examples/human-approval.yaml --input task="Plan a team offsite"
#   → prompts you to approve or revise while the flow remains live
```

Run it from the CLI to see the guaranteed blocking gate.

## `ask-user.yaml` — a model-chosen clarifying question

Exercises the **`ask_user` tool** on a `tool_calling` agent. The agent calls the
blocking tool **only if** it judges the request ambiguous — otherwise it answers
outright. Whether the call happens depends on the prompt and the model.

```console
ac run examples/ask-user.yaml --input request="Book me a table for dinner"
```

Contrast with `human-approval.yaml`: `human_input` always asks; `ask_user` asks
only if the model decides to. When it does, the answer returns to the same live
agent tool loop.

## `decision-brief.yaml` — a realistic multi-stage flow

The fullest example. Given a decision question it:

1. **fans out** to three independent angle agents (the case for, against, the
   risks) — no edges between them, so they run in parallel;
2. **fans in** to a synth agent that picks a typed `Decision` enum;
3. **routes** on that decision with `case ... on` to a decision-specific brief;
4. **finalizes** by combining the decision with whichever brief ran (an n-ary
   `|` coalesce consumed downstream);
5. returns a **multi-field object**, guarded by an `assert`.

```console
ac run examples/decision-brief.yaml --input question="Should a small team adopt a monorepo?"
```

It uses a `Literal` typedef, an `Optional` input with a default, fan-out +
fan-in, a typed enum output, `case ... on` with `else`, a coalesce, multi-output,
and asserts — all with AGENT nodes, so it runs against any provider with no extra
Python.

!!! note "Why the example uses an enum"
    The synth node's verdict is a `Decision` enum because the following `case`
    routes directly on those tags. AGENT nodes may also declare a record, number,
    boolean, or list output; those shapes use structured generation. See
    [Flow syntax → `agent`](syntax.md#agent-an-llm-leaf).

## Next

- [Flow syntax](syntax.md) — the full reference for these constructs.
- [Python API](api.md) — drive these flows from code.
