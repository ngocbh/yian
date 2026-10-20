/**
 * Validate: each built human_input gate's `questions:` surface at load time.
 *
 * A loader post-build pass over the built node map. A literal `questions:` list is structurally
 * checked through `parseQuestions` (it rejects a bad count, a bad shape or duplicate headers); a
 * ref-form `questionsInput` must name one of the node's declared inputs (mirrors the AGENT
 * prompt-scope rule). Its range anchor comes from the gate's transient Compose
 * `BuildResult.fieldRanges` entry for the `questions` field (the coarse `questions:` span), else
 * the authored node range.
 *
 * Knows about: nodes (`HumanInputNode` and its questions parser), compose nodes (`BuildResult`,
 * `siteKey`), and load diagnostics.
 * Never imports: build, the loader, or the runtime.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { HumanInputNode } from "../../nodes/human_input/node.ts";
import { parseQuestions, QuestionSpecError } from "../../nodes/human_input/questions.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";

/**
 * Validate each built human_input gate's questions surface at load time.
 *
 * A literal questions list must pass `parseQuestions`; a ref-form `questionsInput` must name one
 * of the node's declared inputs. Gates without questions pass through. Every problem is
 * collected into one error, anchored at the first located gate's `questions:` field range, else
 * its node range.
 *
 * @param options.diagnostics Builds the error (default: an unlocated one).
 * @throws LoadError listing every invalid gate, one per line.
 */
export function validateHumanQuestions(
  buildResults: ReadonlyMap<string, BuildResult>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): void {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const errors: [string, SourceRange | null][] = [];
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    if (!(node instanceof HumanInputNode)) continue;
    const range = result.fieldRanges.get(siteKey("field", "questions")) ?? result.nodeRange;
    if (node.questions !== null) {
      // A literal list baked into the node: check it at load, so a bad count, shape or duplicate
      // header fails here rather than at run.
      try {
        parseQuestions(node.questions);
      } catch (error) {
        if (!(error instanceof QuestionSpecError)) throw error;
        errors.push([`node ${reprText(nodeId)} (kind=human_input): ${error.message}`, range]);
      }
    } else if (node.questionsInput !== null) {
      // A ref-form gate reads its questions from a declared input, so the name must be wired in
      // through `input:` (mirrors the AGENT prompt-scope rule).
      const declared = new Set((node.params ?? []).map((p) => p.name));
      if (!declared.has(node.questionsInput)) {
        errors.push([
          `node ${reprText(nodeId)} (kind=human_input): questions reference ` +
            `\${${node.questionsInput}} is not a declared input — feed dynamic ` +
            "questions through `input:` wiring and reference the declared name " +
            "(questions: ${name}), or use adaptive_questions:",
          range,
        ]);
      }
    }
  }
  if (errors.length > 0) {
    const firstRange = errors.find(([, range]) => range !== null)?.[1] ?? null;
    throw diagnostics.error(
      "flow has invalid human_input questions:\n  " + errors.map(([message]) => message).join("\n  "),
      { range: firstRange },
    );
  }
}
