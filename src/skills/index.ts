/**
 * Host-supplied skills: the `Skill` value a host passes to a run.
 *
 * `catalog` holds the value and confined bundled-file reads; `tool` holds the AGENT's built-in
 * `load_skill` tool and the system-message listing. Reading `SKILL.md` directories from disk
 * lives in `compose/skills.ts`, and its loaders are never re-exported here.
 *
 * Knows about: `defaults`, `diagnostics`, `typesys` and `llm`.
 * Never imports: nodes, compile, compose, runtime, run, or cli.
 */

export { Skill, SkillError } from "./catalog.ts";
