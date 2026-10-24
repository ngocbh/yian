/**
 * kind=human_input: the blocking human gate's descriptor.
 *
 * Owns the `human_input` node's `HumanInputDescriptor`. It wraps the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) to reject the removed
 * `adaptive_questions.mode:` key and any `adaptive_questions:` key the adaptive desugar does not
 * read, and to record one `siteKey("text", "questions[i]...")` range per question string, a
 * `siteKey("text", "adaptive_questions.prompt")` range for the adaptive brief, and a
 * `siteKey("field", "adaptive_questions.llm_config")` range for its `llm_config:` value. It
 * overrides `_validateFields` to enforce the one-of-three question-surface legality, and `build`,
 * which constructs the runtime `HumanInputNode` and derives its answer schema. A `prompt:`,
 * an `adaptive_questions.prompt:` brief, or a literal question's `question`/`header`/option
 * `label`/`description` YAML read as non-text is rejected at that value. Registered in `KINDS`
 * and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * The `adaptive_questions:` expansion (LLM-authored questions to a synthesized agent feeding a
 * plain questions gate) is a cross-node desugar and stays a graph stage; this descriptor only
 * checks the block's shape and carries it as written.
 *
 * Knows about: diagnostics, compose source and helpers, the AGENT reader
 *              (`MODE_REMOVED_FIX`, `retriesError`), nodes (contracts and human_input) and typesys.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { isExactly, ValueContract } from "../../nodes/contracts.ts";
import { HumanInputNode, siteLabel, templateSites } from "../../nodes/human_input/node.ts";
import { pythonTypeName } from "../../typesys/types.ts";
import {
  isList,
  isRecord,
  rejectLoneSurrogates,
  Type,
  TypeCheckError,
  valueRepr,
  ValueKind,
} from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  notTextError,
  rangeAt,
  rejectNonText,
  siteKey,
  wholeRefName,
} from "./base.ts";
import {
  answerSchema,
  checkAdaptiveBlock,
  checkAdaptiveFields,
  nonTextQuestionField,
  recordAdaptiveRanges,
} from "./tool.ts";

export { answerSchema };

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
   *   present but not text, or text in the list holds a lone surrogate (at that text; for a
   *   key, at its record).
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const id = reprText(nodeId);
    checkAdaptiveBlock(nodeId, `kind=${this.kind}`, cursor, diagnostics);
    const desc = super.parse(nodeId, cursor) as HumanInputDescriptor;
    recordAdaptiveRanges(desc, cursor);
    if (!isList(desc.questions)) return desc as InstanceType<T>;
    const questionsRange = desc.fieldRanges.get(siteKey("field", "questions")) ?? desc.range;
    const found = nonTextQuestionField(desc.questions);
    if (found !== null) {
      const [path, value] = found;
      throw diagnostics.error(`node ${id} (kind=${this.kind}): ` + notTextError(siteLabel(path), value), {
        range: rangeAt(cursor, path) ?? questionsRange,
      });
    }
    try {
      rejectLoneSurrogates(desc.questions, `node ${id} field 'questions'`);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw diagnostics.error(error.message, {
        range: rangeAt(cursor, ["questions", ...error.path]) ?? questionsRange,
        cause: error,
      });
    }
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
    if (auto !== null) checkAdaptiveFields(nodeId, `kind=${this.kind}`, auto, autoRange, diagnostics);
    const promptRange = ranges.get(siteKey("field", "prompt")) ?? ownRange;
    rejectNonText(this.kind, nodeId, "prompt", prompt, promptRange, diagnostics);
    if (prompt !== null) checkWrittenValue(prompt, `node ${id} field 'prompt'`, promptRange, diagnostics);
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /**
   * Build the runtime `HumanInputNode` and its flow-owned wiring.
   *
   * Builds the gate in one of its three question forms, sets its output contract from the
   * authored `output:` (a questions gate with none gets a bare open OBJECT, since the host returns
   * a free-form answer record), derives `answerSchema` from that contract's type, and sets the
   * node's `params` from the authored `input:`; `_finish` stamps the descriptor origin.
   *
   * @throws LoadError when `questions:` is neither a list nor text, the text is not exactly one
   *   whole `${name}`, or `output:` isn't a valid type expression.
   */
  override build(ctx: BuildContext): BuildResult {
    const node = this.buildGate(ctx.diagnostics);
    node.outputContract = this.outputContract(ctx);
    if (node.outputContract === null && (node.questions !== null || node.questionsInput !== null)) {
      node.outputContract = new ValueContract(Type.scalar(ValueKind.OBJECT));
    }
    node.answerSchema = answerSchema(isExactly(node.outputContract, ValueContract) ? node.outputContract.type : null);
    node.params = this.sinkParams();
    return this._finish(node, this.sinkWiring());
  }

  /**
   * Build the gate, carrying its `questions:` in one of three forms.
   *
   * - A list (the literal form) is carried as `questions` (`questionsInput` is `null`).
   * - Text, exactly one whole `${name}` span, names the input the gate reads the list from at run
   *   time, recorded as `questionsInput` (`questions` is `null`). No surrounding text, dots, calls
   *   or operators; whether the name is a declared input is checked later, with the other
   *   question checks.
   * - `null` builds a plain prompt gate with no questions.
   *
   * @throws LoadError at `questions:` when it is neither a list nor text, or the text is not
   *   exactly one whole `${name}`.
   */
  private buildGate(diagnostics: LoadDiagnostics): HumanInputNode {
    const id = reprText(this.id);
    const prompt = this.prompt ?? "";
    const questions = this.questions;
    if (isList(questions)) {
      return new HumanInputNode(this.id, { prompt, questions, questionsInput: null, title: this.nodeName });
    }
    const range = this.fieldRanges.get(siteKey("field", "questions")) ?? this.range;
    if (typeof questions === "string") {
      const name = wholeRefName(questions);
      if (name === null) {
        throw diagnostics.error(
          `node ${id} (kind=${this.kind}): \`questions:\` as a string must be exactly one \`\${name}\` naming a ` +
            `declared input (e.g. \`questions: \${qs}\`), got ${valueRepr(questions)}; put any text in ` +
            "`prompt:` instead",
          { range },
        );
      }
      return new HumanInputNode(this.id, { prompt, questions: null, questionsInput: name, title: this.nodeName });
    }
    if (questions !== null) {
      throw diagnostics.error(
        `node ${id} (kind=${this.kind}): \`questions:\` must be a list of questions (each item ` +
          "`- question: ...`) or exactly `${name}` naming a declared input, got " +
          `${pythonTypeName(questions)} ${valueRepr(questions)}`,
        { range },
      );
    }
    return new HumanInputNode(this.id, { prompt, title: this.nodeName });
  }
}
