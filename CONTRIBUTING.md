# Contributing

yian is a TypeScript package. You need Bun 1.4 or newer, and Node 22.18 or newer for the Node
checks. [`ARCHITECTURE.md`](ARCHITECTURE.md) explains how the code fits together.

## Development setup

```bash
git clone https://github.com/ngocbh/yian && cd yian
bun install --frozen-lockfile
bun src/cli/index.ts --help        # the yian CLI, run from source
```

`./install.sh` puts this checkout's `yian` on your PATH, so `yian` runs your working tree. Set
`YIAN_BIN_DIR` to choose the folder (default `~/.local/bin`) and `YIAN_BUN` to choose the Bun.
`./uninstall.sh` removes the wrapper and the PATH line.

## Tests

Run focused tests while you work, then the full checks before you open a pull request:

```bash
bun run typecheck
bun test ./tests/repo/kind_census.test.ts                      # one file
bun run build                                                  # some tests run the built dist/ on Node
bun run test                                                   # every test, on Bun
bun run test:node                                              # the same tests, on Node
bun tests/conformance/runner.ts                                # every conformance case
bun tests/conformance/runner.ts --case run-core/code_greeting  # one case
```

A new test goes in the `tests/` folder that mirrors the `src/` folder it tests (the root for
`src/*.ts`), or `tests/e2e/` for a whole flow; see "Where tests live" in `ARCHITECTURE.md`.

When you change the docs site (`docs/`), also run:

```bash
bun test ./tests/docs
(cd docs && bun install --frozen-lockfile && bun run build)
YIAN_UPDATE_BUNDLE=1 bun test tests/repo/bundled.test.ts   # the docs, skill and flows yian chat carries
```

The same regeneration follows a change to the `composing-agents` skill or to `src/cli/flows/`.
`src/cli/bundled.ts` is generated but committed, so yian runs from a checkout.

[`docs/STYLE.md`](docs/STYLE.md) says how to write a docs page, and
[`tests/conformance/README.md`](tests/conformance/README.md) how to add a conformance case.

## Pull requests

1. Every change starts with a GitHub issue. Check its claims against the current code first.
2. Post a short plan as a comment on the issue.
3. Branch from `main` as `dev/<area>/<short-name>`.
4. Make small commits, each with its tests passing. Add a test for every behavior change.
5. When behavior changes, update the docs page that describes it, concisely, in the same pull
   request, and regenerate the bundle `yian chat` carries
   (`YIAN_UPDATE_BUNDLE=1 bun test tests/repo/bundled.test.ts`).
6. Open a pull request from the template, with `Fixes #<issue>` and the exact checks you ran.

The work is done when the pull request merges into `main`.

## Code style

- Keep code direct and scoped to the issue.
- Docstrings (TSDoc) say what a symbol does; comments say why non-obvious code exists.
- No issue numbers, plan labels or AI-attribution footers in source.
