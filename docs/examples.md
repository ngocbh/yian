# Examples

The [`examples/`](https://github.com/ngocbh/agent-composer/tree/main/examples)
directory in the repository contains a set of generic flows, smallest first.
After cloning the repository, run them from its root with
`composer run examples/<file> --input ...` (set a provider/model first — see
[Installation](installation.md)). The directory is not bundled in the current
Python package.

## `hello.yaml` — the smallest flow

One AGENT node, a string in and a string out.

```yaml
id: hello
name: hello
description: Greet someone with a short, friendly one-liner.

input:
  name: str
output: str
kind: agent
prompt: |-
  Write a short, warm one-sentence greeting addressed to ${name}.
  Just the greeting, no preamble.
```

```console
composer run examples/hello.yaml --input name=Ada
```

Shows compact mode: the flow and its single AGENT node share the typed `input:`
and `output:` boundary, and the prompt reads the auto-wired local `${name}`.

## `summarize.yaml` — one transform

A single AGENT node that condenses a block of text into one sentence. Unlike compact
`hello`, it shows the equivalent full `nodes:` form with explicit input and output wiring.

```console
composer run examples/summarize.yaml --input text="...long text..."
```

## `classify.yaml` — a constrained output

An AGENT node whose `output:` is a `Literal[...]` enum, so the model must answer with
exactly one tag, and the flow returns a one-field object.

```yaml
id: classify
name: classify
input:
  text: str

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
composer run examples/classify.yaml --input text="I love this!"
```

## `human-approval.yaml` — a deterministic human gate

Exercises the **`human_input`** node. An AGENT node drafts a plan, a `human_input`
node always blocks for an `approve` / `revise` answer (a `Literal` typedef),
`case ... on` routes on the typed answer, and a coalesce joins whichever branch
ran. The gate returns the answer as an ordinary node output from the same live
invocation.

```console
composer run examples/human-approval.yaml --input task="Plan a team offsite"
#   → prompts you to approve or revise while the flow remains live
```

Run it from the CLI to see the guaranteed blocking gate.

## `ask-user.yaml` — a model-chosen clarifying question

Exercises the **`ask_user` tool** on a `tool_calling` AGENT node. The node calls the
blocking tool **only if** it judges the request ambiguous — otherwise it answers
outright. Whether the call happens depends on the prompt and the model.

```console
composer run examples/ask-user.yaml --input request="Book me a table for dinner"
```

Contrast with `human-approval.yaml`: `human_input` always asks; `ask_user` asks
only if the model decides to. When it does, the answer returns to the same live
model/tool loop.

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
composer run examples/decision-brief.yaml --input question="Should a small team adopt a monorepo?" --num-workers 4
```

It uses a `Literal` typedef, an `Optional` input with an expression fallback, fan-out +
fan-in, a typed enum output, `case ... on` with `else`, a coalesce, a multi-field output,
and asserts — all with AGENT nodes, so it runs against any provider with no extra
Python.

!!! note "Why the example uses an enum"
    The synth node's verdict is a `Decision` enum because the following `case`
    routes directly on those tags. The enum itself uses structured generation,
    as do record, number, boolean, and list outputs from AGENT nodes; only a bare `str` stays
    on the free-text path. See
    [Flow syntax → `agent`](syntax.md#agent-the-llm-backed-agent-node).

## Next

- [Flow syntax](syntax.md) — the full reference for these constructs.
- [Python API](api.md) — drive these flows from code.
