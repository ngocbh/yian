/**
 * Graph-stage: an `ask` node's `adaptive_questions:` block desugar — block to synth agent + ask.
 *
 * A TOOL `ask` node may carry an `adaptive_questions:` nested block instead of a static
 * `questions:` input or a free-text `prompt:`. The block IS the spec of an LLM that AUTHORS the
 * questions: `{prompt: <LLM brief, required>, llm_config?, retries?}`. This pass lowers that single
 * descriptor into TWO ordinary descriptors:
 *
 * 1. a synth **AgentDescriptor** (id `<ask>__compose`) that composes a `list[Question]`: its
 *    codomain is the code-built `questionListType()` set via the `outputContractOverride` seam (no
 *    surface type-string), its prompt is the block's brief, it inherits the ask's context inputs
 *    (every input but `timeout`, `questions` and `__questions`) with their declared types, and it
 *    records the ask as its `host`, so the host-edge step gives it the ask's CASE routes,
 *    `dependsOn`, `runsAfter`, and data gates on the ask's other inputs (no LLM call when the ask
 *    can't run);
 * 2. the original **ToolDescriptor** ask (same id), rewritten to read the composer's output through
 *    its reserved `__questions` input.
 *
 * WHY a descriptor-level pass (modeled on `desugarInlineCalls`): it runs BEFORE the build loop, so
 * the synth agent flows through the kind-blind `desc.build(ctx)` + prompt-scope validation + ref
 * wiring exactly like any author-written agent. A brief that references an undeclared `${...}` is
 * therefore caught by the same prompt-scope check that guards every agent. A post-build pass would
 * skip that validation entirely.
 *
 * `__questions` is a reserved input name: the wire from the composer's output into the ask's
 * questions. Authors do not write it; the ask's parse refuses it.
 *
 * Located parse metadata: the synth composer agent inherits the ask descriptor's
 * `range`/`fieldRanges` (so its downstream errors locate at the ask), except that its
 * `siteKey("field", "prompt")` range is the brief's own range and its `siteKey("field",
 * "llm_config")` range is the block's `llm_config:` value range. Its `fieldLabels` names its
 * `prompt` and `llm_config` as the ask's `adaptive_questions.prompt` and
 * `adaptive_questions.llm_config`, so a bad brief or config is reported on the ask at the authored
 * value, and names each copied `input 'p'` as the ask's `input 'p'`, so a bad binding reads
 * exactly like the ask's own error at the same range and is reported once. The rewritten ask
 * keeps its own ranges via `replace`.
 *
 * Knows about: diagnostics, node contracts, the ask's question type and tool id, and the AGENT
 * and TOOL descriptors.
 * Never imports: compose/parse, edges, validate, build, the loader, or the runtime; the pipeline
 * imports this back.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { ValueContract } from "../../nodes/contracts.ts";
import { questionListType } from "../../nodes/tool/questions.ts";
import { ASK } from "../../tools/context.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { AgentDescriptor, type AgentDescriptorFields } from "../nodes/agent.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { QUESTIONS_INPUT, ToolDescriptor } from "../nodes/tool.ts";

/**
 * Reserved ask input name: the wire from the synth composer's output into the ask's questions.
 * Authors never write it; the desugar owns the namespace.
 */
export const QPARAM = QUESTIONS_INPUT;

/**
 * Lower each TOOL `ask` with `adaptive_questions:` into a synth agent + rewritten ask.
 *
 * Returns a NEW descriptor map. A non-adaptive descriptor passes through untouched; an adaptive
 * ask is replaced IN PLACE by two entries emitted in order — the synth composer agent first (id
 * `<ask>__compose`), then the rewritten ask — so the agent precedes the ask in insertion order
 * (build/edge order).
 *
 * @param descriptors Parsed descriptors keyed by node id in authored graph order.
 * @param options.diagnostics Factory for located load errors.
 * @returns New ordered mapping with each adaptive ask lowered to two nodes.
 * @throws LoadError for a synthesized node-id collision.
 */
export function desugarAdaptiveQuestions(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Map<string, NodeDescriptor> {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const newDescriptors = new Map<string, NodeDescriptor>();
  for (const [gid, desc] of descriptors) {
    if (!(desc instanceof ToolDescriptor && desc.toolId === ASK && desc.adaptiveQuestions !== null)) {
      newDescriptors.set(gid, desc);
      continue;
    }
    const synthId = `${gid}__compose`;
    if (descriptors.has(synthId)) {
      throw diagnostics.error(
        `node ${reprText(gid)} (tool_id: ${ASK}): synthesized composer id ${reprText(synthId)} ` +
          "collides with an existing node id",
        { range: desc.range },
      );
    }
    // The composer reads the ask's context: every input but its own question surface and
    // timeout, so a context input reaches the host as an extra too. The ask's parse refused an
    // authored `__questions`.
    const { timeout: _timeout, questions: _questions, [QPARAM]: _reserved, ...context } = desc.inputs;
    newDescriptors.set(synthId, composerFor(gid, desc, context));
    const wire = `\${${synthId}.output}`;
    newDescriptors.set(gid, desc.replace({ adaptiveQuestions: null, inputs: { ...desc.inputs, [QPARAM]: wire } }));
  }
  return newDescriptors;
}

/**
 * Return the synth composer agent for the asking node `desc` (id `gid`) that carries an
 * `adaptive_questions:` block: id `<gid>__compose`, the block's brief as its prompt, `inputs` as its
 * context inputs with their declared types, the question list as its output, and `gid` as its host.
 */
function composerFor(
  gid: string,
  desc: NodeDescriptor & { readonly adaptiveQuestions: unknown },
  inputs: Readonly<Record<string, unknown>>,
): AgentDescriptor {
  // The asking node's parse checked the block is a mapping with a `prompt:`.
  const auto = desc.adaptiveQuestions as Readonly<Record<string, unknown>>;
  // The block IS the agent spec. Pass optional fields only when present so the AgentDescriptor
  // defaults (e.g. `current().agentRetries`, llmConfig {}) apply otherwise. The synth agent
  // inherits the ask's ranges so its downstream errors locate there, but its prompt and
  // llm_config are the block's, so their errors locate at the block's values and name the ask's
  // fields. Its copied inputs are named as the ask's, so their errors read as the ask's own.
  // Its internal contract has no authorable type string.
  const fieldRanges = new Map<string, SourceRange>(desc.fieldRanges);
  fieldRanges.delete(siteKey("field", "prompt"));
  const blockRange = desc.fieldRanges.get(siteKey("field", "adaptive_questions"));
  const briefRange = desc.fieldRanges.get(siteKey("text", "adaptive_questions.prompt")) ?? blockRange;
  if (briefRange !== undefined) fieldRanges.set(siteKey("field", "prompt"), briefRange);
  const llmConfigRange = desc.fieldRanges.get(siteKey("field", "adaptive_questions.llm_config")) ?? blockRange;
  if (llmConfigRange !== undefined) fieldRanges.set(siteKey("field", "llm_config"), llmConfigRange);
  const fieldLabels = new Map([
    ["prompt", `node ${reprText(gid)} adaptive_questions.prompt`],
    ["llm_config", `node ${reprText(gid)} adaptive_questions.llm_config`],
  ]);
  // Keyed like `Node.wiringRefLabel`'s default, which validators look up.
  for (const p of Object.keys(inputs)) {
    fieldLabels.set(`input ${reprText(p)}`, `node ${reprText(gid)} input ${reprText(p)}`);
  }
  const agentFields: AgentDescriptorFields = {
    id: `${gid}__compose`,
    prompt: auto["prompt"] as string,
    // The composer reads the asking node's context inputs, so it checks their declared types too.
    inputs: { ...inputs },
    inputTypes: new Map([...desc.inputTypes].filter(([name]) => Object.hasOwn(inputs, name))),
    outputContractOverride: new ValueContract(questionListType()),
    outputs: null,
    // `host` gives the composer the ask's routes and ordering; its copied inputs already make
    // it wait like the ask.
    host: gid,
    range: desc.range,
    fieldRanges,
    fieldLabels,
  };
  if (Object.hasOwn(auto, "llm_config")) agentFields.llmConfig = auto["llm_config"];
  // The ask's parse checked the YAML integer; an AGENT counts retries in a safe integer.
  if (Object.hasOwn(auto, "retries")) agentFields.retries = Number(auto["retries"]);
  return new AgentDescriptor(agentFields);
}
