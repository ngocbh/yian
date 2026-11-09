# Writing the yian docs

This guide is for anyone who writes or reviews a page in `docs/src/content/docs/`. It isn't
published.

## Readers

- Flow authors write YAML flows and run them with the `yian` CLI.
- Host developers run flows from their own TypeScript program.
- Contributor material (how the engine is built and tested) lives in the repo, not on the site.

## Voice

- Use plain words and short sentences. Write to "you".
- Be concise and direct. Say each thing once, in as few words as it needs; cut what a reader
  doesn't need to use yian.
- Open each page with a sentence or two on what it is for.
- Give one example per concept. Put anything readers look up (flags, defaults, codes) in a table.
- Describe yian as it is. No history: no "used to", "now", "new", "changed" or "no longer".
- No internal names, issue numbers or plan labels. No Python: yian is a TypeScript package.
- Use sentence case for titles and headings.

## The code is the source of truth

Check each statement against `src/` and `tests/`. When the old page you are replacing and the
code disagree, follow the code and say so in your PR description.

## Pages and links

- One `.md` page per outline entry, with `title` and `description` in its front matter.
- Leave the sidebar in `docs/astro.config.mjs` alone; ask in your PR if a page should move.
- Link by absolute path with a trailing slash: `[types](/types/)`, `[Optional](/types/#optional)`.
  The build fails on a link or anchor that doesn't exist.
- Don't link into the yian repository (github.com/ngocbh/yian): it's private, so readers get a
  404. Don't link source files or `examples/`; show examples in full.
- Keep within your page's line budget (`wc -l`). The PR description gives the budget.

## Examples

`tests/docs/examples.test.ts` checks every code block on every page. It also checks the code
blocks in the root `README.md`.

- Tag YAML `yaml`. Each block must load as a flow unless you mark it with `check=`:

  | Mark | Use it for | The test |
  |---|---|---|
  | (none) | a whole flow | loads it with the real loader |
  | `check=runnable` | a flow whose output you show | loads it, runs it, compares the output |
  | `check=fails` | a flow that shows a load error | expects the load to fail with the error in the `text` block after it |
  | `check=fragment` | part of a flow, or other YAML | checks only the YAML syntax |
  | `source=examples/<file>` | a flow from `examples/`, shown whole | checks the block is identical to the file, then loads it |

- A titled block that isn't a fragment is written at its `title=` path, beside the page's other
  titled blocks, so a `uses:` ref can import another block of the same page
  (`title="app/report.yaml"` importing `../common/clean`), and an `!include` can read a titled
  `md` block. Two blocks of a page can't share a title.
- A runnable block has `title="<name>.yaml"`. Follow it with a `console` block whose first line
  is `$ yian run <name>.yaml --input k=v ...` and whose next lines are what the run prints.
- When a runnable flow calls a model, give the model's replies, in order, on the line before the
  YAML block: `<!-- yian-test {"replies": ["Hello, Ada!"]} -->`.
- Show commands in `console` blocks, each line starting with `$ `, followed only by what the
  command prints on stdout. Every `yian` command and flag is checked against the CLI, with or
  without the `$ `, and so are `npx @ngocbh/yian` and `bunx @ngocbh/yian` lines; don't put them in
  `sh` or `bash` blocks.
- Draw diagrams in `mermaid` blocks.

## Before you open a PR

Run `bun test ./tests/docs` and `(cd docs && bun run build)`. The build checks links, YAML
highlighting and diagrams. Don't delete old pages under `docs/`; the final docs PR deletes them
all.
