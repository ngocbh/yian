# Contributing

This is the main guide for contributing to Yian. Keep changes small,
clear, and tied to a GitHub issue.

## Development setup

Yian is TypeScript. It needs Bun 1.4 or newer, and Node 22.18 or newer for the Node
checks.

```bash
git clone https://github.com/ngocbh/yian
cd yian
bun install --frozen-lockfile
```

## GitHub workflow

Every change starts with a GitHub issue, including small fixes.

1. Search for an existing issue. If none fits, use the right issue template.
2. Assign the issue to yourself and add the `in progress` label.
3. Check the issue against the current code. Reproduce bugs and cite
   `file:line` evidence.
4. Close a stale or finished issue with evidence instead of changing code.
5. Split work that is too large for one focused pull request into linked issues.
6. Create a `dev/<domain>/<short-name>` branch.
7. Post a short plan as an issue comment before implementation.
8. Work in small steps: change, test, pass, commit.
9. Add an issue comment only for useful progress, a blocker, or a plan change.
10. Open a pull request from the template and link the issue with
    `Fixes #<number>`.

Work is complete only after the pull request merges into `main`.

Useful commands:

```bash
gh issue list --state all --search "short search terms"
gh issue create --template bug_report.yml
gh issue create --template feature_request.yml
gh issue edit 123 --add-assignee @me --add-label "in progress"
```

## Write clear GitHub text

Use short sentences and simple words. Put facts before opinions. Include only
evidence, decisions, changes, and results that help the next reader.

### Issues

Use the issue template and keep each answer brief. A useful issue says:

```text
Problem
<What happens now, and when?>

Goal
<What should happen?>

Evidence
<A command, error, reproduction, or file:line.>

Done when
- <One result that can be checked.>

Out of scope
- <Related work this issue will not do.>
```

A proposed fix is a suggestion, not a fact. Check it against the current code.
Only list alternatives when there is a real design choice.

### Plan comments

Post the plan on the issue. Do not create a tracked plan file for routine work.
Use only the lines that add value:

```text
## Plan

- Verified: <why the work is still needed, with evidence>
- Approach: <the small change to make>
- Decision: <a non-obvious choice and why>
- Changes: <files or areas to edit>
- Checks: <tests or commands to run>
- Docs: <guides or skills to update>
- Out of scope: <work deliberately left out>
```

Post a temporary plan file with:

```bash
gh issue comment 123 --body-file /tmp/issue-123-plan.md
```

If the approach or scope changes, add a short comment that explains why.

### Progress comments

Comment only when the update helps someone follow the work:

```text
## Progress

- Done: <finished result>
- Next: <next useful step>
- Blocker or plan change: <what changed and why, if needed>
```

Do not post comments that only say “still working.”

### Pull requests

Use `.github/PULL_REQUEST_TEMPLATE.md`. Keep the body short and complete:

- Summarize what changed and why in one or two sentences.
- Add `Fixes #123`.
- List the main code, test, and doc changes.
- Link the plan comment and explain any deviation.
- Check the project rules that the change respects.
- Paste the exact test commands and state their result.

## Branches, tests, and commits

Create a branch from current `main`, unless the work depends on another unmerged
branch.

```bash
git switch main
git pull --ff-only
git switch -c dev/engine/short-name
```

Stay inside the issue scope. Ask when a contract, requirement, name, or
cross-cutting design choice is unclear.

Run focused tests while working, then the relevant suite before opening a pull
request:

```bash
bun run typecheck
bun test ./tests/engine        # focused tests
bun run test                   # the whole suite on Bun
bun run test:node              # the same tests on Node
bun tests/conformance.ts       # the conformance cases
```

Add or update a test for every behavior change. Make one logical change per
commit, and commit only when its tests pass. Never claim a check passed unless
you ran it.

## Engine rules

Read [`docs/engine.md`](docs/engine.md) before an
engine change.

- Keep `runtime/` kind-blind.
- Keep nodes pure at the graph-state boundary: they return `Output`, `Route`, or
  `Grow` and do not write the variable pool.
- Keep runtime transitions record-authoritative: prepare and validate, append to
  `RunLog`, then apply through the reducer. Recovery folds the same reducer.
- Keep the authored structure fixed; runtime growth only instantiates authored
  child flows.
- Keep package imports acyclic.
- Prefer general, domain-neutral nodes and combinators over one-off behavior.

## Docs change with code

Update docs and project skills in the same pull request as the code they explain.

| Change | Update |
| --- | --- |
| YAML, expressions, or node authoring | `docs/syntax.md` and the `composing-agents` skill |
| Engine behavior or rules | `docs/engine.md` |
| CLI or public API | `README.md` and the docstrings |

Check edited flow templates still load. Build the docs when they change:

```bash
bun test ./tests/docs
(cd docs && bun install && bun run build)
```

## Docstrings

Write TSDoc: a first sentence that says what the symbol does, then `@param`, `@returns` and
`@throws` where they add something the signature doesn't say. Document an argument's meaning,
shape, limits and default. List only errors the function throws.

```ts
/**
 * Return the result for the input.
 *
 * @param items Items keyed by node id. Unknown ids are not allowed.
 * @param name The name used in messages; the default name when absent.
 * @throws RangeError when `items` holds an unknown node id.
 */
export function fn(items: Record<string, Item>, name?: string): Result {
```

Add examples to public APIs. Docstrings explain what code does; comments explain
why non-obvious code exists. Do not put task numbers, plan labels, or
AI-attribution footers in source code.
