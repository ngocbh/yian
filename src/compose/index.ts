/**
 * The Compose-YAML loader: reads Compose-shaped YAML flows into the
 * engine's runtime model.
 *
 * `loader.ts` is the front door (`loadFlow` to a `ComposedFlow`, or a `LoadError`); the phases it
 * runs live in `parse/`, `desugar/`, `edges/`, `validate/` and `build/`, the per-kind readers in
 * `nodes/`, and `skills.ts` reads `SKILL.md` folders for a run's `skills` (`loadSkills`,
 * `defaultSkillRoots`). This index re-exports the loader's public names and nothing else;
 * callers of any other name import the module that defines it. `ChildResolver` is the loader's
 * `uses:` resolver, which may return a promise, not the phases' internal call resolver of the
 * same name.
 *
 * Knows about: yaml (through `source/`), defaults, diagnostics, typesys, expr, nodes, memory
 * contracts, LLM-client config, skills, compile and platform.
 * Never imports: runtime, the package-root run boundary, the CLI; nothing in the engine imports
 * compose.
 */

export { synthesizeBoundaryGraph } from "./build/boundary.ts";
export { ComposedFlow } from "./build/pipeline.ts";
export { expandCaseOutputs } from "./desugar/case_outputs.ts";
export { desugarInlineCalls } from "./desugar/inline_calls.ts";
export { inferDataEdges } from "./edges/data.ts";
export { LoadError } from "./helpers/errors.ts";
export { readFlowInputs, readType } from "./helpers/types.ts";
export { loadFlow } from "./loader.ts";
export type { ChildResolver } from "./loader.ts";
export { AgentDescriptor } from "./nodes/agent.ts";
export { NodeDescriptor } from "./nodes/base.ts";
export { CallDescriptor } from "./nodes/call.ts";
export { CaseDescriptor } from "./nodes/case.ts";
export { CodeDescriptor } from "./nodes/code.ts";
export { ChildSignature, childSignature } from "./nodes/resolver.ts";
export { ToolDescriptor } from "./nodes/tool.ts";
export { ComposeFile, parseFile } from "./parse/file.ts";
export { parseNodes } from "./parse/nodes.ts";
export { defaultSkillRoots, loadSkills } from "./skills.ts";
export { rejectCycles } from "./validate/cycles.ts";
export { checkCaseHandles } from "./validate/handles.ts";
export { validateReferences } from "./validate/references.ts";
