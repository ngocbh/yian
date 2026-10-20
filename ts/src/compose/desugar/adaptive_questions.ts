/**
 * Graph-stage: `human_input` `adaptive_questions:` block desugar — block to synth agent + gate.
 *
 * A `human_input` node may carry an `adaptive_questions:` nested block instead of a static
 * `questions:` list or a free-text `prompt:`. The block IS the spec of an LLM that AUTHORS the
 * questions: `{prompt: <LLM brief, required>, llm_config?, retries?}`. This pass lowers that single
 * descriptor into TWO ordinary descriptors:
 *
 * 1. a synth **AgentDescriptor** (id `<gate>__compose`) that composes a `list[Question]`: its
 *    codomain is the code-built `questionListType()` set via the `outputContractOverride` seam (no
 *    surface type-string), its prompt is the block's brief, it inherits the gate's context `input:`
 *    with its declared types, and it records the gate as its `host`, so the host-edge step gives it
 *    the gate's CASE routes, `dependsOn`, `runsAfter`, and data gates on the gate's other inputs
 *    (no LLM call when the gate can't run);
 * 2. the original **HumanInputDescriptor** gate (same id), rewritten to read its questions from the
 *    composer: `questions: "${__questions}"`, with the composer's output wired into the reserved
 *    `__questions` input.
 *
 * WHY a descriptor-level pass (modeled on `desugarInlineCalls`): it runs BEFORE the build loop, so
 * the synth agent flows through the kind-blind `desc.build(ctx)` + prompt-scope validation + ref
 * wiring exactly like any author-written agent. A brief that references an undeclared `${...}` is
 * therefore caught by the same prompt-scope check that guards every agent. A post-build pass would
 * skip that validation entirely.
 *
 * `__questions` is a reserved input name: the wire from the composer's output into the gate's
 * questions input. Authors do not write it.
 *
 * Located parse metadata: the synth composer agent inherits the gate descriptor's
 * `range`/`fieldRanges` (so its downstream errors locate at the gate), except that its
 * `siteKey("field", "prompt")` range is the brief's own range and its `siteKey("field",
 * "llm_config")` range is the block's `llm_config:` value range. Its `fieldLabels` names its
 * `prompt` and `llm_config` as the gate's `adaptive_questions.prompt` and
 * `adaptive_questions.llm_config`, so a bad brief or config is reported on the gate at the authored
 * value, and names each copied `input 'p'` as the gate's `input 'p'`, so a bad binding reads
 * exactly like the gate's own error at the same range and is reported once. The rewritten gate
 * keeps its own ranges via `replace`.
 *
 * Knows about: diagnostics, node contracts, the human_input question type, and the AGENT and
 * HUMAN_INPUT descriptors.
 * Never imports: compose/parse, edges, validate, build, the loader, or the runtime; the pipeline
 * imports this back.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { ValueContract } from "../../nodes/contracts.ts";
import { questionListType } from "../../nodes/human_input/questions.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { AgentDescriptor, type AgentDescriptorFields } from "../nodes/agent.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { HumanInputDescriptor } from "../nodes/human_input.ts";

/**
 * Reserved gate input name: the wire from the synth composer's output into the gate's questions
 * input. Authors never write it; the desugar owns the namespace.
 */
export const QPARAM = "__questions";

/**
 * Lower each `adaptive_questions:` human_input into a synth agent + rewritten gate.
 *
 * Returns a NEW descriptor map. A non-adaptive descriptor passes through untouched; an adaptive
 * `human_input` is replaced IN PLACE by two entries emitted in order — the synth composer agent
 * first (id `<gate>__compose`), then the rewritten gate — so the agent precedes the gate in
 * insertion order (build/edge order).
 *
 * @param descriptors Parsed descriptors keyed by node id in authored graph order.
 * @param options.diagnostics Factory for located load errors.
 * @returns New ordered mapping with each adaptive gate lowered to two nodes.
 * @throws LoadError for a synthesized node-id collision or an authored reserved `__questions`
 *   input on an adaptive gate.
 */
export function desugarAdaptiveQuestions(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Map<string, NodeDescriptor> {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const newDescriptors = new Map<string, NodeDescriptor>();
  for (const [gid, desc] of descriptors) {
    if (!(desc instanceof HumanInputDescriptor && desc.adaptiveQuestions !== null)) {
      newDescriptors.set(gid, desc);
      continue;
    }

    // The gate's parse checked the block is a mapping with a `prompt:`.
    const auto = desc.adaptiveQuestions as Readonly<Record<string, unknown>>;
    const synthId = `${gid}__compose`;
    if (descriptors.has(synthId)) {
      throw diagnostics.error(
        `node ${reprText(gid)} (kind=human_input): synthesized composer id ${reprText(synthId)} ` +
          "collides with an existing node id",
        { range: desc.range },
      );
    }
    if (Object.hasOwn(desc.inputs, QPARAM)) {
      throw diagnostics.error(
        `node ${reprText(gid)} (kind=human_input): input ${reprText(QPARAM)} is reserved for adaptive_questions`,
        { range: desc.fieldRanges.get(siteKey("input", QPARAM)) ?? desc.range },
      );
    }

    // The block IS the agent spec. Pass optional fields only when present so the AgentDescriptor
    // defaults (e.g. `current().agentRetries`, llmConfig {}) apply otherwise. The synth agent
    // inherits the gate's ranges so its downstream errors locate there, but its prompt and
    // llm_config are the block's, so their errors locate at the block's values and name the gate's
    // fields. Its copied inputs are named as the gate's, so their errors read as the gate's own.
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
    for (const p of Object.keys(desc.inputs)) {
      fieldLabels.set(`input ${reprText(p)}`, `node ${reprText(gid)} input ${reprText(p)}`);
    }
    const agentFields: AgentDescriptorFields = {
      id: synthId,
      prompt: auto["prompt"] as string,
      // The composer reads the gate's context inputs, so it checks their declared types too.
      inputs: { ...desc.inputs },
      inputTypes: new Map(desc.inputTypes),
      outputContractOverride: new ValueContract(questionListType()),
      outputs: null,
      // `host` gives the composer the gate's routes and ordering; its copied inputs already make
      // it wait like the gate.
      host: gid,
      range: desc.range,
      fieldRanges,
      fieldLabels,
    };
    if (Object.hasOwn(auto, "llm_config")) agentFields.llmConfig = auto["llm_config"];
    // The gate's parse checked the YAML integer; an AGENT counts retries in a safe integer.
    if (Object.hasOwn(auto, "retries")) agentFields.retries = Number(auto["retries"]);
    newDescriptors.set(synthId, new AgentDescriptor(agentFields));

    // Rewrite the gate: read questions from the composer via the reserved wire.
    newDescriptors.set(
      gid,
      desc.replace({
        adaptiveQuestions: null,
        questions: `\${${QPARAM}}`,
        inputs: { ...desc.inputs, [QPARAM]: `\${${synthId}.output}` },
      }),
    );
  }
  return newDescriptors;
}
