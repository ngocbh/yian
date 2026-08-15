<!--
Follow the issue-driven workflow in CONTRIBUTING.md. A PR should map to an issue
that already has a verified problem statement and an implementation-plan comment.
-->

## Summary

<!-- One or two sentences: what changed and why. -->

Fixes #<!-- issue number -->

## What changed

<!-- The concrete changes, grouped by area (engine / compose / cli / docs). -->

-

## Implementation plan

<!-- Link the plan comment you posted on the issue before implementing. Note any
     deviations from that plan and why. -->

- Plan: <!-- Paste the URL of the plan comment. -->
- Deviations from the plan:

## Invariants respected

- [ ] Layer ladder stays acyclic (`diagnostics` below `events`; nodes combine
      `diagnostics` / `typesys` / `expr` before `compile <- compose`); runtime
      consumes contracts below compose, and the run boundary stays above both.
- [ ] Engine core stays **kind-blind** — no new `match node.kind` in `runtime/`; `tests/engine/test_kind_census.py` still passes.
- [ ] Nodes stay **pure** — a node returns `Output | Route | Grow` and never writes the pool.
- [ ] Runtime state stays **record-authoritative** — prepare, append to `RunLog`, then
      apply through the reducer; recovery replays that same reducer.
- [ ] No self-rewriting graph / agentic routing (structural determinism preserved).
- [ ] No new heavy dependency in the core.
- [ ] Docs + skills updated in this PR for any surface change (syntax / node kind / CLI / invariant).

## Test plan

<!-- The specific tests that prove this works. Paste the command you ran. -->

```bash
PYTHONPATH=src python -m pytest tests/engine
```

- [ ] New/updated tests cover the change and would catch a regression.
- [ ] Full suite passes locally.
