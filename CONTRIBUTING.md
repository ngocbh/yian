# Contributing

yian is a TypeScript package. You need Bun 1.4 or newer, and Node 22.18 or newer for the Node
checks. [`ARCHITECTURE.md`](ARCHITECTURE.md) explains how the code fits together.

## Development setup

```bash
git clone https://github.com/ngocbh/yian && cd yian
bun install --frozen-lockfile
bun src/cli/index.ts --help        # the yian CLI, run from source
```

## Tests

Run focused tests while you work, then the full checks before you open a pull request:

```bash
bun run typecheck
bun test ./tests/engine/kind_census.test.ts             # one file
bun run build                                           # some tests run the built dist/ on Node
bun run test                                            # every test, on Bun
bun run test:node                                       # the same tests, on Node
bun tests/conformance.ts                                # every conformance case
bun tests/conformance.ts --case run-core/code_greeting  # one case
```

When you change the docs site (`docs/`), also run:

```bash
bun test ./tests/docs
(cd docs && bun install --frozen-lockfile && bun run build)
```

[`docs/STYLE.md`](docs/STYLE.md) says how to write a docs page, and
[`conformance/README.md`](conformance/README.md) how to add a conformance case.

## Pull requests

1. Every change starts with a GitHub issue. Check its claims against the current code first.
2. Post a short plan as a comment on the issue.
3. Branch from `main` as `dev/<area>/<short-name>`.
4. Make small commits, each with its tests passing. Add a test for every behavior change.
5. Update the docs page that describes the behavior you changed, in the same pull request.
6. Open a pull request from the template, with `Fixes #<issue>` and the exact checks you ran.

The work is done when the pull request merges into `main`.

## Code style

- Keep code direct and scoped to the issue.
- Docstrings (TSDoc) say what a symbol does; comments say why non-obvious code exists.
- No issue numbers, plan labels or AI-attribution footers in source.
