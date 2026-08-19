# Plan: Agent concepts documentation

**Goal**: Make the documentation concise, self-contained, and consistent from
the conceptual definition of an agent through YAML authoring and execution.

**Architecture**: Add one canonical Agent page, make Architecture an end-to-end
map, and let each deeper page own its details. Use “agent” as a conceptual
umbrella while reserving “AGENT node” for `kind: agent`.

**Tech stack**: Markdown, MkDocs Material, Mermaid, Agent Composer YAML.

## Dependency order

| Group | Steps | Can parallelize |
| --- | --- | --- |
| 1 | 1-2 | Yes; new Agent page and Architecture own different files |
| 2 | 3-4 | Yes; public docs and project guides own different files |
| 3 | 5 | No; verifies all prior edits |

## Step 1: Add the conceptual Agent page

**Files**: `docs/agent.md`, `mkdocs.yml`

1. Write the canonical definition, anatomy, terminology table, and composition
   model.
2. Add compact-node, multi-node, and child-flow YAML examples.
3. Put Agent first under Internals and link onward to Architecture and Syntax.
4. Load each example with `load_flow` to verify the authored shapes.
5. Commit the page and navigation as one logical change.

## Step 2: Rewrite Architecture as the lifecycle map

**File**: `docs/architecture.md`

1. Replace repeated Compose and Engine detail with the YAML-to-`RunResult`
   lifecycle.
2. Retain the major data/ownership boundaries and package dependency map.
3. Include primary success/failure, propagated host faults, recovery, and
   optional post-terminal learning.
4. Link detailed stages to `compose.md`, `engine.md`, and `nodes.md`.
5. Build the page and commit after its links and diagrams render.

## Step 3: Normalize public documentation

**Files**: `docs/index.md`, `docs/syntax.md`, `docs/nodes.md`, `docs/engine.md`,
`docs/api.md`, `docs/typing.md`, `README.md`

1. Replace conflicting definitions with the canonical short definition or a
   link to Agent.
2. Standardize child flow, AGENT node, task prompt, variable pool, node
   outcome, flow output, and `RunResult`.
3. Remove repeated internal explanations when an owning page can be linked.
4. Preserve all current syntax and API behavior.
5. Run targeted phrase/link searches and commit the public consistency pass.

## Step 4: Synchronize contributor guidance

**Files**: `src/agent_composer/README.md`,
`.claude/skills/engine/SKILL.md`,
`.claude/skills/composing-agents/SKILL.md`

1. Replace the narrow agent definition with the conceptual/runtime distinction.
2. Use child flow consistently and reserve memory for agent memory.
3. Keep detailed engine invariants intact; change only wording needed for the
   shared model.
4. Compare the guides against current code contracts and commit the guide pass.

## Step 5: End-to-end verification and review

**Files**: all changed documentation

1. Run `git diff --check`.
2. Run terminology searches for stale conflicting phrases.
3. Run `conda run -n agentcomposer mkdocs build --strict`.
4. Review the built navigation and rendered pages.
5. Request independent review against issue #257 and this design.
6. Fix all critical/important findings, rerun checks, open the PR, wait for CI,
   and merge.
