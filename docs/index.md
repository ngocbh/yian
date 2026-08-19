# The Agent Composer

**Bridging the trust gap between humans and agents.**

Hand a free-form LLM agent a complex task and it improvises a plan on the fly—calling tools,
branching, looping—in whatever shape the context happens to produce. That flexibility
is also the problem: the plan is *opaque*. You don't see the structure the model chose,
you can't tell whether it has a bug, and the next run might quietly do something else.
When the stakes are real, "it usually works" isn't trust.

Agent Composer makes the flow a **first-class artifact that both you and the
model can read**. Instead of the agent inventing its plan at runtime, the flow is
written out as a small Docker-Compose-shaped YAML file — by you, by an LLM, or by the
two of you together. You can see exactly what runs, inspect it for bugs, and refine it
after an error; so can the model. The human owns the graph; the LLMs only fill the leaf
boxes — they never rewrite the structure at runtime.

A flow is a function: it has a typed `input:`, a graph of `nodes:`, and an `output:`.
The graph between nodes is *inferred* from the `${...}` references — you never draw
edges by hand.

```yaml
# debate.yaml — frame a question, argue both sides in parallel, then decide
id: debate
name: debate
input:
  question: str
nodes:
  frame:
    kind: agent
    input:
      question: ${input.question}
    output: str
    prompt: "Restate '${question}' and list the 2-3 criteria that should drive it."
  for_case:
    kind: agent
    input:
      brief: ${frame.output}            # edge: frame -> for_case
    output: str
    prompt: "Make the strongest case FOR, against these criteria: ${brief}"
  against_case:
    kind: agent
    input:
      brief: ${frame.output}            # frame -> against_case (runs parallel to for_case)
    output: str
    prompt: "Make the strongest case AGAINST, against these criteria: ${brief}"
  verdict:
    kind: agent
    input:
      for_case: ${for_case.output}      # fan-in: verdict waits for BOTH sides
      against_case: ${against_case.output}
    output: str
    prompt: |-
      Weigh both sides and recommend in 2-3 sentences, with the key reason.
      For: ${for_case}
      Against: ${against_case}
output: ${verdict.output}
```

The four nodes form a **diamond**, inferred entirely from the `${...}` references —
no edges are drawn by hand:

```
        ┌─> for_case ────┐
frame ──┤                ├──> verdict
        └─> against_case ┘
```

`for_case` and `against_case` both read `${frame.output}` but never reference each
other, so the engine runs them **in parallel**; `verdict` reads both, so it **waits
for both** before it runs. The structure is fixed by the author: every run argues
both sides before deciding — you can read that guarantee straight off the file.

```console
$ ac run debate.yaml --input question="Should a small team adopt a monorepo?"
Adopt the monorepo. For a small team the simpler cross-project refactors and single ...
```

## Why this shape

- **The flow is readable** — it *is* the plan, in plain YAML. You can review
  it before it runs, spot a bug in the structure, and refine it after an error — and an
  LLM can do the same, because the surface is small and explicit.
- **The structure is fixed by the author** — the LLM fills leaf boxes; it does not
  rewrite the graph. The same flow runs the same way every time, so a fix stays fixed.
- **An agent is a typed transformation** — it can be represented by one node, a
  flow, or a child flow. The LLM-backed `kind: agent` is one concrete node kind.
- **Flows compose** — CALL, MAP, and LOOP apply child flows nested to any depth.
- **Pure at the boundary** — a node *returns* its output and the engine *binds* it; a
  node never mutates shared state. Outputs are immutable, typed, serializable values.
  Together with the append-only run log, that referential transparency makes runs
  reproducible and recoverable from a validated record prefix.
- **Effects stay inside explicit leaves** — an AGENT node's granted `ask_user` tool and a
  `human_input` gate block in their live node call; `wait` blocks until an absolute UTC
  deadline. `ask_user` returns its answer into the same model/tool loop; the graph sees each
  leaf's eventual ordinary output instead of a graph rewrite or continuation.

## Where to go next

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — `pip install agent-composer`, provider extras, and picking a model.
- :material-console: **[The `ac` CLI](cli.md)** — run a flow from the terminal, supply inputs, and serve blocking human requests.
- :material-file-code: **[Flow syntax](syntax.md)** — the full Compose-YAML reference: types, `${...}` refs, node kinds, `case`, coalesce, asserts.
- :material-robot: **[Agent](agent.md)** — the framework's agent model, vocabulary, and YAML representations.
- :material-lightbulb: **[Examples](examples.md)** — walk through the flows that ship in `examples/`.
- :material-language-python: **[Python API](api.md)** — use the engine as a library (`load_flow` / `run_flow` / `resume_flow`).

</div>
