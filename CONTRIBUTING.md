# Contributing

Thanks for your interest in Agent Composer. This guide covers local setup and the
conventions we hold contributions to — most importantly, the **docstring style**.

## Development setup

```bash
git clone https://github.com/ngocbh/agent-composer
cd agent-composer
pip install -e ".[all,dev]"
```

The `all` extra pulls in every provider client (Anthropic / OpenAI / Google /
Ollama); `dev` adds the test and build tooling.

## Contribution workflow

All work is tracked in **GitHub issues** — there is no in-repo backlog. Every change,
from a typo fix to a new node kind, follows the same issue-driven workflow.

**Before you write any code:**

1. **Find or create the issue.** Search the [issue tracker](https://github.com/ngocbh/agent-composer/issues)
   first. If nothing covers the work, open one using the matching template under
   `.github/ISSUE_TEMPLATE/` (bug / feature / setup) and apply sensible labels.

**To address an issue:**

2. **Claim it.** Assign yourself (`gh issue edit <n> --add-assignee @me`) and/or add the
   `in progress` label so nobody duplicates your effort.
3. **Read it — but don't trust it.** Read the whole issue, treating every claim as
   unverified. Issues go stale as the code moves.
4. **Verify against the current codebase.** Reproduce the bug, or for a feature decide
   whether it is still necessary and not already implemented. Quote `file:line` evidence.
   If the issue is already addressed or no longer needed, **close it with that evidence**
   instead of writing code.
5. **Treat any proposed solution as one option, not the plan.** A fix proposed earlier may
   be stale — verify it still holds against today's code before adopting it.
6. **List the options.** Enumerate candidate approaches with their trade-offs, weighed
   against the project's design intent (general combinators, a kind-blind engine core,
   node purity, structural determinism — see [`src/agent_composer/README.md`](src/agent_composer/README.md)).
7. **Split if it's too big.** Break a large issue into linked sub-issues and tackle them
   one at a time.
8. **Finalize, branch, and post a plan.** Choose the approach, cut a branch (below), and
   post an **implementation plan** as a comment on the issue (see
   [Implementation plan structure](#implementation-plan-structure)).
9. **Implement** in small, tested steps — code + test + green, one logical change per commit.
10. **Run an adversarial self-review** before opening the PR: try to break your own code —
    edge cases, boundary inputs, resume/concurrency paths, and the engine invariants (layer
    ladder, kind census, node purity). Fix what you find.
11. **Open the PR** using `.github/PULL_REQUEST_TEMPLATE.md`, linking the issue with
    `Fixes #<n>`.

### Branch naming

Never commit a feature straight to `main`. Cut a branch first:

- **Branch off** `main` for independent work, or off another `dev/...` branch when the
  work builds on an unmerged feature.
- **Name it** `dev/<domain>/<feature>` — `<domain>` is the area it touches (`engine`,
  `cli`, `compose`, `docs`, ...) and `<feature>` is a short kebab-case slug, e.g.
  `dev/engine/loop-until-times` or `dev/cli/chat-repl`.
- **A feature is finished only once it is merged to `main`** — not when the code is
  written or tests pass on the branch. Git history and merged PRs are the record.

```bash
git switch -c dev/engine/loop-until-times   # cut the branch off main
# ... implement, test, commit ...
# open a PR; a feature is done only after it merges to main
```

### Implementation plan structure

Post the plan as a comment on the issue *before* implementing, so the approach can be
reviewed while it is still cheap to change. A plan should contain:

- **Summary** — one or two sentences: what will change and why. Link the issue.
- **Problem verification** — how you confirmed the issue is real against the *current*
  code: `file:line` evidence, reproduction steps for a bug, or the justification that a
  feature is still needed and not already covered.
- **Options considered** — the candidate approaches, each with its trade-offs, and which
  one you chose and why. Note explicitly if you rejected the issue's original proposed
  solution and the reason.
- **Chosen approach** — the design in enough detail to review before any code exists:
  new/changed node kinds, `${...}` syntax, seams, data shapes.
- **Changes** — the files/modules you expect to touch, and the tests you will add.
- **Invariants respected** — how the change keeps the layer ladder acyclic, the engine
  core kind-blind (census stays at 0), nodes pure, and the graph structurally
  deterministic. Call out any tension.
- **Test plan** — the specific tests that will prove the change works.
- **Risks / out of scope** — what could break, and anything deliberately deferred (file a
  follow-up issue for it).


## Running tests

```bash
pytest
```

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI / Read the Docs runs
```

The API reference is generated from docstrings by
[mkdocstrings](https://mkdocstrings.github.io/), so a clear docstring is what
makes the reference useful — see the next section.

## Docstring style

We follow a **HuggingFace-flavored, Google-section** style. It is Google-compatible
(so mkdocstrings renders `Args:` / `Returns:` / `Raises:` into tables) but uses
backtick-quoted types and `*optional*, defaults to ...` markers like the
🤗 Transformers codebase.

**Every argument must be documented**, and the description must explain the
argument's *meaning, shape, and constraints* — not merely restate its type.

### Template

```python
def fn(required, optional=None):
    """
    One-line summary in the imperative mood, ending with a period.

    Optional extended description: the *why*, important behavior, invariants,
    or edge cases a caller must know. Omit if the summary says everything.

    Args:
        required (`type`):
            What it is and what it's *for* — not just its type. State the shape
            (e.g. `dict[node_id -> list[Edge]]`), the allowed values, and any
            constraint the caller must satisfy.
        optional (`type`, *optional*, defaults to `None`):
            Same, plus what the default means when omitted.

    Returns:
        `ReturnType`:
            What the value represents and how to interpret its fields/states.
            (Omit this section entirely for functions that return `None`.)

    Raises:
        `SomeError`:
            The exact condition that triggers it. (Omit if it never raises.)

    Example:
        ```python
        >>> result = fn(required=...)
        >>> result.status
        'succeeded'
        ```
    """
```

### Worked example

```python
def run_flow(loaded, inputs, *, run_id=None, on_event=None):
    """
    Coerce inputs, seed the variable pool, enforce asserts, and drive the flow to a terminal.

    Never raises on a flow failure: a failed, paused, or aborted run is returned as a
    `RunResult` with a non-`"succeeded"` status. A false boundary assert returns a
    `status="failed"` result *before* any node runs.

    Args:
        loaded (`ComposedFlow`):
            A compiled, validated flow from [`load_flow`][agent_composer.load_flow].
            Carries the IR, the declared input schema, and the assert sets.
        inputs (`dict[str, Any]`):
            Run arguments keyed by declared input name. Each value is coerced to its
            declared type; names omitted here fall back to their declared defaults.
        run_id (`str`, *optional*, defaults to `None`):
            Host-injected run id, readable in the flow as `${system.run_id}`. When
            `None`, a fresh id is minted per run.
        on_event (`Callable[[Any], None]`, *optional*, defaults to `None`):
            Called with each engine event as it occurs (`NodeStarted`, `RunSucceeded`,
            `RunPaused`, `RunFailed`, `RunAborted`). Use it for progress reporting.

    Returns:
        `RunResult`:
            Outcome of the run. `status` is one of `"succeeded"`, `"failed"`,
            `"paused"`, or `"aborted"`; `output` is set on success, `pause_reasons`
            and resume handles on a pause.

    Example:
        ```python
        from agent_composer import load_flow, run_flow

        loaded = load_flow(open("hello.yaml").read(), search_paths=["."])
        result = run_flow(loaded, {"name": "Ada"})
        print(result.status, result.output)  # succeeded ...
        ```
    """
```

### Conventions

- **Summary** — one imperative line. Add an extended description only when it
  conveys non-obvious information (the *why*, an invariant, an edge case).
- **Args** — document every argument, always. Type in backticks; for keyword
  arguments add `*optional*, defaults to X`. The description explains *meaning +
  shape + constraints*, not a restatement of the type.
- **Returns** — describe what the value *means* and how to read its fields/states.
  Omit the section for functions that return `None`.
- **Raises** — list only exceptions the function actually raises, with the exact
  triggering condition.
- **Example** — include for public-API symbols; optional for internal helpers.
- **Cross-references** — link other symbols with
  `[name][agent_composer.path.name]`; mkdocstrings turns them into links.
- **Comments vs. docstrings** — docstrings explain *what* a thing is and how to
  use it; inline comments explain *why* a non-obvious line exists.
