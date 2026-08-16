# The Agent Composer

[![Documentation](https://readthedocs.org/projects/agent-composer/badge/?version=latest)](https://agent-composer.readthedocs.io)

Hand an agent a complex task and it improvises a plan on the fly — calling tools,
branching, looping — in whatever shape the context happens to produce. That flexibility
is also the problem: the workflow is *opaque*. You don't see the plan the agent chose,
you can't tell whether it has a bug, and the next run might quietly do something else.
When the stakes are real, "it usually works" isn't trust.

The Agent Composer makes the workflow a **first-class artifact that both you and the
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

## Install

> **Early stage.** Agent Composer is under active development and not yet on PyPI.
> The API, YAML surface, and CLI may move or change quickly between commits.

Install directly from the repository:

```console
pip install "git+https://github.com/ngocbh/agent-composer.git"
```

Or clone and install in editable mode:

```console
git clone https://github.com/ngocbh/agent-composer.git
cd agent-composer
pip install -e .
```

Provider SDKs are optional extras — install the one(s) you use:

```console
pip install "agent-composer[anthropic] @ git+https://github.com/ngocbh/agent-composer.git"   # Claude
pip install "agent-composer[openai]    @ git+https://github.com/ngocbh/agent-composer.git"   # GPT
pip install "agent-composer[google]    @ git+https://github.com/ngocbh/agent-composer.git"   # Gemini
pip install "agent-composer[ollama]    @ git+https://github.com/ngocbh/agent-composer.git"   # local models
pip install "agent-composer[all]       @ git+https://github.com/ngocbh/agent-composer.git"   # everything
```

From a clone, the extras are simply `pip install -e ".[anthropic]"`, `".[all]"`, etc.

The core (engine + CLI) installs with no provider SDK; importing a provider you
haven't installed raises a clear `pip install agent-composer[...]` hint.

## The `ac` CLI

```console
ac run FLOW.yaml [--input k=v]... [--inputs inputs.json] [--log PATH] [-w N] [--provider P] [--model M] [--quiet] [--verbose] [--engine-trace]
ac run FLOW.yaml --resume PATH [-w N] [--quiet] [--verbose] [--engine-trace]
```

- Every fresh CLI run is durable. By default its reported log is
  `~/.agent_composer/runs/<run_id>.jsonl`; `--log PATH` selects a different new
  file. Existing files are never overwritten. The default directories use mode
  `0700` and JSONL files use `0600`; logs can contain inputs, model/tool/human
  data, outputs, configuration, and tracebacks, so do not publish them.
- `--resume PATH` recovers that log with the same flow. It rejects `--input`,
  `--inputs`, `--provider`, `--model`, and `--log`, so recovery cannot replace
  recorded inputs or model configuration.
- `--input k=v` — set one input (repeatable). Supplied strings are coerced, where
  supported, against the input's declared type; stored defaults are not coerced.
- `--inputs file.json` — load inputs from a JSON object. `--input` flags override
  individual keys.
- At run start a boxed **banner** (to stderr) names the flow being run — its
  `name`, `version`, and `description` — unless `--quiet`.
- Each declared input still missing is **prompted interactively** — each prompt is
  labelled with its canonical resolved type, a required (`*`)/optional mark, and
  any default. Optional/defaulted boolean and `Literal[...]` prompts offer an
  explicit omit/use-declared-default action; blank optional/defaulted free text does
  the same. Omitted keys stay absent so START applies an exact authored default when
  present, while an optional input without one remains absent.
- `HUMAN_INPUT` and an authored agent `ask_user` tool block inside the live node
  call while the CLI's main-thread broker collects the answer. `WAIT` likewise
  blocks a worker until its absolute UTC deadline and returns `None`. Each
  interaction completes inside the same live node invocation.
- Per-node progress prints to **stderr**: each running node shows a spinner that
  becomes a green `✓` on success or a red `✗` (plus the error) on failure.
  `--quiet`/`-q` suppresses it; `--verbose`/`-v` also prints each node's output.
- `--num-workers`/`-w N` — engine worker pool size. A fresh run defaults to `0`,
  the single-threaded deterministic drain. Resume without this flag preserves
  the latest recorded count; an explicit value changes only future scheduling.
  `>=1` runs independent ready nodes (a fan-out) concurrently. The output is the
  same either way.
- The **current working directory is added to `sys.path`** for the run, so a flow's
  `code: pkg.mod:fn` refs to packages under the dir you invoke `ac` from resolve
  without exporting `PYTHONPATH` (like `python script.py` makes its dir importable).
- A flow that fails to **compile** prints a **located** error — a boxed `.yaml`
  source frame (line numbers, the offending line(s) highlighted; a multi-node error
  like a cycle highlights every implicated node and prints a legend of the dependency
  edges that close the loop) titled `file:line`, with the message below — instead of an
  engine traceback; `--engine-trace` adds the Python traceback for debugging the engine
  itself.
- A **runtime** failure resolves its opaque origin trace through the loaded flow and uses
  the same diagnostic renderer as load failures. Nested `defs:`, external `uses:`, CALL,
  MAP, and LOOP failures show source-owned frames in outermost-to-innermost order, down to
  the exact operation origin. If source presentation is missing or broken, the original
  failure remains unlocated instead of the CLI inferring a range from runtime ids or graphs.

### Choosing a provider/model

The default provider and model are read from the environment:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=anthropic        # or openai / google / ollama
export AGENT_COMPOSER_DEFAULT_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=...                            # provider's own key var
```

For a local Ollama endpoint:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=ollama
export AGENT_COMPOSER_DEFAULT_MODEL=llama3.2:3b
export OLLAMA_BASE_URL=http://localhost:11434
ac run examples/hello.yaml --input name=Ada
```

### `ac chat` — an interactive REPL, built as a flow

```console
ac chat [FLOW] [--workspace/-C DIR] [--provider P] [--model M]
```

An interactive chat session. The REPL is itself an Agent Composer flow — a
`loop` node runs one turn per iteration. Its blocking `human_input` feeds a
`case`: exact `/exit` and `/quit` values route to an exit record before the
reply model, while ordinary text runs `agent` and deterministic transcript code.
The assistant's reply prints before the next `You:` prompt.

- `FLOW` (optional) — a chat flow `.yaml`. The **default** is the bundled
  *composer chat*: a `tool_calling` agent with flow-op tools
  (`list_flows`, `read_flow`, `validate_flow`, `run_flow`, `write_flow`) so you
  can build and run flows conversationally.
- `--workspace`/`-C DIR` — the directory the flow-op tools are confined to
  (default: the current directory).
- `--provider P` / `--model M` — fill-the-gap overrides for agents that set no
  provider/model of their own (the outermost cascade layer, same as `ac run`).
- The bundled flow ends on exact `/exit` or `/quit`; the broker normalizes EOF
  (Ctrl-D) to `/exit`. A user-supplied flow must author the same route — otherwise
  typed or EOF-normalized `/exit` is ordinary input. `ac chat` never intercepts
  typed text.
- A single bad turn does not end the session: if the reply agent errors mid-turn
  (a raise, or hitting its tool-iteration cap), the CLI prints the failure and
  continues from your conversation so far. The initial attempt and every restart
  receive separate reported logs under the default runs directory.
- Ctrl-C propagates into a flow running through the bundled `run_flow` tool (including
  a blocking WAIT); a nested flow failure remains a tool result and does not cancel chat.

## Examples

The [`examples/`](examples/) directory ships a few generic flows:

- `hello.yaml` — the smallest agent flow (one AGENT, string in/out), in compact form.
- `debate.yaml` — frame → argue for/against in parallel → verdict (a fan-out/fan-in diamond).
- `summarize.yaml` — condense a block of text into one sentence.
- `classify.yaml` — label text with a constrained `Literal[...]` output.
- `triage-ticket.yaml` — extract a structured record from a support message, then draft a reply.
- `decision-brief.yaml` — fan-out to three angles, pick a verdict, route, and finalize.
- `ask-user.yaml` / `human-approval.yaml` — model-chosen vs. always-on blocking human interaction.
- `chat.yaml` — a turn-by-turn conversational REPL (a `loop` per turn; deterministic transcript) — the shape `ac chat` runs.

## Use it as a library

```python
from pathlib import Path

from agent_composer import load_flow, run_flow

path = Path("hello.yaml")
loaded = load_flow(
    path.read_text(),
    search_paths=[path.parent],
    source_label=path.name,
)
result = run_flow(loaded, {"name": "Ada"})
print(result.status, result.output)
```

`source_label` gives load-time diagnostics ownership of the text they frame.
`LoadError.diagnostic` is a canonical `Diagnostic`, including ordered source
frames for definitions and imported files; callers never need to combine an
error range with separately retained YAML text.

Failed runs return `result.failure`, an immutable `Failure` also exported from
`agent_composer`. Use its stable `code` for control flow and `message` for
display; node identity, opaque origin trace, and optional debugging cause stay
on that same value rather than parallel error fields.

Library callers can resolve that opaque trace with
`loaded.diagnose(result.failure)`, which returns the canonical presentation-only
`Diagnostic` defined in `agent_composer.diagnostics` (it is not a package-root
export). The method copies `code`, `message`, `node_id`, `cause_type`, and
`traceback`, then resolves the complete trace atomically. The CLI uses this same
method for runtime failures before passing the result to the shared diagnostic
renderer; it does not resolve runtime ids or graph topology itself.

`loaded.origin_resolver` is the source-neutral authority for every origin reachable
from the compiled Compose graph. It is `None` only when that graph is source-free;
otherwise it resolves local, nested, and external-file origins to diagnostic frames.
Resolver metadata does not affect flow equality, fingerprints, or growth digests.

Lower-level Python graph builders and tooling can import the source-neutral
`Binding`, `Parameter`, `Assertion`, `ValueContract`, `ContractOrigins`, and
`ValuePath` vocabulary from `agent_composer.nodes`. Origins are optional; these
definitions are being adopted by engine consumers incrementally.

Pass `log="run.jsonl"` to create an authoritative append-only schema-3 JSONL history for
the run. If the process is interrupted, load the same flow in a later process and
recover the durable prefix with `resume_flow(loaded, log="run.jsonl")` from
`agent_composer.run`. A fresh durable run never overwrites an existing path.
Earlier schemas are not migrated. Run logs retain canonical inputs, node outputs
(including human/tool/model data), resolved model configuration, and failure
tracebacks. Treat them as sensitive:
choose an access-controlled path and do not commit or publish them.

## Develop & test

```console
pip install -e ".[all,dev]"
pytest
```

For a per-directory dev environment — a `.env` you can `source` (or auto-load
with direnv) plus venv activation — copy `.env.example` to `.env` and see
[Local development environment](docs/installation.md#local-development-environment).
Note that `ac` does not read `.env` automatically; you source it or let direnv load it.

## License

Apache 2.0 — see [LICENSE](LICENSE).
