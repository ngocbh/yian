/**
 * kind=human_input: the blocking human gate's descriptor.
 *
 * Owns the `human_input` node's `HumanInputDescriptor`. It wraps the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) to reject the removed
 * `adaptive_questions.mode:` key and any `adaptive_questions:` key the adaptive desugar does not
 * read, and to record one `siteKey("text", "questions[i]...")` range per question string, a
 * `siteKey("text", "adaptive_questions.prompt")` range for the adaptive brief, and a
 * `siteKey("field", "adaptive_questions.llm_config")` range for its `llm_config:` value. It
 * overrides `_validateFields` to enforce the one-of-three question-surface legality. A `prompt:`,
 * an `adaptive_questions.prompt:` brief, or a literal question's `question`/`header`/option
 * `label`/`description` YAML read as non-text is rejected at that value. Registered in `KINDS`
 * and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * The `adaptive_questions:` expansion (LLM-authored questions to a synthesized agent feeding a
 * plain questions gate) is a cross-node desugar and stays a graph stage; this descriptor only
 * checks the block's shape and carries it as written.
 *
 * Knows about: diagnostics, compose source and helpers, the AGENT reader (`MODE_REMOVED_FIX`,
 *              `retriesError`), nodes (human_input) and typesys.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { siteLabel, templateSites } from "../../nodes/human_input/node.ts";
import { isList, isRecord } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
import { MODE_REMOVED_FIX, retriesError } from "./agent.ts";
import { NodeDescriptor, type NodeDescriptorFields, notTextError, rejectNonText, siteKey } from "./base.ts";

/** The keys the adaptive desugar reads from an `adaptive_questions:` block. */
const ADAPTIVE_KEYS: ReadonlySet<string> = new Set(["prompt", "llm_config", "retries"]);

/** The fields `HumanInputDescriptor`'s constructor takes: the common ones and the gate's own. */
export interface HumanInputDescriptorFields extends NodeDescriptorFields {
  prompt?: string | null;
  questions?: unknown;
  adaptiveQuestions?: unknown;
  asserts?: readonly string[] | null;
}

/**
 * kind=human_input: block for a person; the typed answer is the node's output.
 *
 * Three author surfaces, exactly one of which carries the question(s):
 * - `questions`: a literal list (`[{question, header, options, multi_select}, ...]`), or a
 *   `${...}` reference text naming a declared input;
 * - `adaptive_questions`: a block `{prompt: <LLM brief>, llm_config?, retries?}`; the engine asks
 *   an LLM to author the questions;
 * - `prompt`: the free-text approve or answer surface.
 *
 * `allowed` adds `prompt`/`questions`/`adaptive_questions`, `inputs`/`outputs` and `asserts` to
 * the commons; nothing is required, since the question surface is one of three. `inputs` is the
 * one binding field.
 */
export class HumanInputDescriptor extends NodeDescriptor {
  static override readonly kind = "human_input";
  static override readonly allowed = this.allow(
    "prompt",
    "questions",
    "adaptive_questions",
    "inputs",
    "outputs",
    "asserts",
  );
  static override readonly required = this.require();
  static override readonly bindingFields: readonly string[] = ["inputs"];

  /** The free-text prompt template; `null` when absent. */
  readonly prompt: string | null;
  /** The authored `questions:` as YAML read it: a list, reference text, or `null`. */
  readonly questions: unknown;
  /** The authored `adaptive_questions:` block as YAML read it, or `null`. */
  readonly adaptiveQuestions: unknown;
  /** The node asserts. */
  readonly asserts: readonly string[];

  /**
   * Create a descriptor from `fields`; absent fields are `null`, and `asserts` reads `null` as
   * empty.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: HumanInputDescriptorFields) {
    super(fields);
    this.prompt = fields.prompt ?? null;
    this.questions = fields.questions ?? null;
    this.adaptiveQuestions = fields.adaptiveQuestions ?? null;
    this.asserts = fields.asserts ?? [];
  }

  /**
   * Reject unknown `adaptive_questions:` keys and a non-text brief or question; range the texts.
   *
   * After the template `parse`, rejects the first `question`/`header`/option `label`/
   * `description` in a literal `questions:` list that is present but not text, at that value.
   * On top of the template `parse`'s ranges, records a `siteKey("text",
   * "adaptive_questions.prompt")` range for the brief, a `siteKey("field",
   * "adaptive_questions.llm_config")` range for its `llm_config:` value, and one
   * `siteKey("text", siteLabel(path))` range for each string `templateSites` finds in a literal
   * `questions:` list, so an error points at the string or config rather than the whole block.
   *
   * @throws LoadError when the `adaptive_questions:` block authors `mode:` or a key other than
   *   `prompt`/`llm_config`/`retries` (at the key), its brief is not text or holds a lone
   *   surrogate, the template `parse` rejects the node, or a literal question's text field is
   *   present but not text, or holds a lone surrogate.
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const id = reprText(nodeId);
    const adaptive = cursor.get("adaptive_questions");
    if (isRecord(adaptive)) {
      if (Object.hasOwn(adaptive, "mode")) {
        throw diagnostics.error(
          `node ${id} (kind=human_input): \`adaptive_questions.mode:\` was removed: ${MODE_REMOVED_FIX}`,
          { range: keyRange(cursor.child("adaptive_questions"), "mode") ?? cursor.keyRangeOf("adaptive_questions") },
        );
      }
      for (const key of Object.keys(adaptive)) {
        if (ADAPTIVE_KEYS.has(key)) continue;
        throw diagnostics.error(
          `node ${id} (kind=human_input): \`adaptive_questions:\` field ${reprText(key)} is not allowed ` +
            `(allowed: ${[...ADAPTIVE_KEYS].sort(compareCodeUnits).join(", ")})`,
          { range: keyRange(cursor.child("adaptive_questions"), key) ?? cursor.keyRangeOf("adaptive_questions") },
        );
      }
    }
    const brief = isRecord(adaptive) ? (adaptive["prompt"] ?? null) : null;
    if (brief !== null) {
      const briefRange = rangeAt(cursor, ["adaptive_questions", "prompt"]) ?? cursor.keyRangeOf("adaptive_questions");
      if (typeof brief !== "string") {
        throw diagnostics.error(
          `node ${id} (kind=human_input): ` + notTextError("adaptive_questions.prompt", brief),
          { range: briefRange },
        );
      }
      checkWrittenValue(brief, `node ${id} field 'adaptive_questions.prompt'`, briefRange, diagnostics);
    }
    const desc = super.parse(nodeId, cursor) as HumanInputDescriptor;
    if (isRecord(adaptive)) {
      for (const [key, site] of [
        ["prompt", siteKey("text", "adaptive_questions.prompt")],
        ["llm_config", siteKey("field", "adaptive_questions.llm_config")],
      ] as const) {
        if (!Object.hasOwn(adaptive, key)) continue;
        const range = rangeAt(cursor, ["adaptive_questions", key]);
        if (range !== null) desc.fieldRanges.set(site, range);
      }
    }
    if (!isList(desc.questions)) return desc as InstanceType<T>;
    const questionsRange = desc.fieldRanges.get(siteKey("field", "questions")) ?? desc.range;
    const found = nonTextQuestionField(desc.questions);
    if (found !== null) {
      const [path, value] = found;
      throw diagnostics.error(`node ${id} (kind=human_input): ` + notTextError(siteLabel(path), value), {
        range: rangeAt(cursor, path) ?? questionsRange,
      });
    }
    checkWrittenValue(desc.questions, `node ${id} field 'questions'`, questionsRange, diagnostics);
    for (const [path] of templateSites(null, desc.questions)) {
      const range = rangeAt(cursor, path);
      if (range !== null) desc.fieldRanges.set(siteKey("text", siteLabel(path)), range);
    }
    return desc as InstanceType<T>;
  }

  /**
   * Enforce the one-of-three question-surface legality, then check `prompt:` and run the base
   * checks.
   *
   * Exactly one of `prompt`/`questions`/`adaptive_questions` is needed (an error at the node
   * body, as no question field carries a range), `questions` and `adaptive_questions` are
   * mutually exclusive, and an adaptive block must be a mapping carrying its own `prompt:` (the
   * LLM brief) and a valid `retries:`; those errors point at `adaptive_questions:`. Then a present
   * `prompt:` must be text without a lone surrogate, at its field.
   *
   * @throws LoadError for the first rule broken.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const id = reprText(nodeId);
    const prompt = fields["prompt"] ?? null;
    const hasQuestions = (fields["questions"] ?? null) !== null;
    const auto = fields["adaptiveQuestions"] ?? null;
    const autoRange = ranges.get(siteKey("field", "adaptive_questions")) ?? null;
    if (prompt === null && !hasQuestions && auto === null) {
      throw diagnostics.error(
        `node ${id} (kind=${this.kind}): needs \`prompt:\`, \`questions:\`, or \`adaptive_questions:\``,
        { range: ownRange },
      );
    }
    if (hasQuestions && auto !== null) {
      throw diagnostics.error(
        `node ${id} (kind=${this.kind}): \`questions:\` and \`adaptive_questions:\` are mutually exclusive`,
        { range: autoRange },
      );
    }
    if (auto !== null) {
      if (!isRecord(auto) || !auto["prompt"]) {
        throw diagnostics.error(
          `node ${id} (kind=${this.kind}): \`adaptive_questions:\` requires a \`prompt:\` (the LLM brief)`,
          { range: autoRange },
        );
      }
      const problem = Object.hasOwn(auto, "retries") ? retriesError(auto["retries"]) : null;
      if (problem !== null) {
        throw diagnostics.error(`node ${id} (kind=${this.kind}): \`adaptive_questions.retries:\` ${problem}`, {
          range: autoRange,
        });
      }
    }
    const promptRange = ranges.get(siteKey("field", "prompt")) ?? ownRange;
    rejectNonText(this.kind, nodeId, "prompt", prompt, promptRange, diagnostics);
    if (prompt !== null) checkWrittenValue(prompt, `node ${id} field 'prompt'`, promptRange, diagnostics);
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }
}

/**
 * Return the first question text field that is present and not text, as `[path, value]`.
 *
 * Checks each record question's `question` and `header`, then the `label` and `description` of
 * each record option in a list `options`; path steps are as in `templateSites`. Other shapes are
 * left to the question validator.
 *
 * @param questions The node's literal `questions:` list.
 * @returns The offending field's key path from the node body and its value, or `null` when every
 *   present text field is text.
 */
function nonTextQuestionField(questions: readonly unknown[]): [(string | number)[], unknown] | null {
  for (const [i, question] of questions.entries()) {
    if (!isRecord(question)) continue;
    for (const key of ["question", "header"]) {
      if (Object.hasOwn(question, key) && typeof question[key] !== "string") {
        return [["questions", i, key], question[key]];
      }
    }
    const options = question["options"];
    if (!isList(options)) continue;
    for (const [j, option] of options.entries()) {
      if (!isRecord(option)) continue;
      for (const key of ["label", "description"]) {
        if (Object.hasOwn(option, key) && typeof option[key] !== "string") {
          return [["questions", i, "options", j, key], option[key]];
        }
      }
    }
  }
  return null;
}

/**
 * Return the range of the value at `path` under `cursor`, or `null` when it has none.
 *
 * `null` also covers a path the cursor can't walk (such as past a recursive alias cut), so the
 * caller's error falls back to an enclosing range.
 */
function rangeAt(cursor: SourceCursor, path: readonly (string | number)[]): SourceRange | null {
  try {
    for (const step of path) cursor = cursor.child(step);
  } catch (error) {
    if (error instanceof SourceAccessError) return null;
    throw error;
  }
  return cursor.range;
}

/**
 * Return the range of `key` in the mapping under `cursor`, or `null` when it has none.
 *
 * `null` also covers a mapping past a recursive alias cut, so the caller's error falls back to
 * an enclosing range.
 */
function keyRange(cursor: SourceCursor, key: string): SourceRange | null {
  try {
    return cursor.keyRangeOf(key);
  } catch (error) {
    if (error instanceof SourceAccessError) return null;
    throw error;
  }
}
