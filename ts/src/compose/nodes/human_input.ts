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
 * Knows about: diagnostics, compose source and helpers, expr (template scanning), the AGENT reader
 *              (`MODE_REMOVED_FIX`, `retriesError`), nodes (contracts and human_input) and typesys.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { isBareName, type Tree } from "../../expr/grammar.ts";
import { scanTemplate, Span, type TemplateSegment } from "../../expr/template.ts";
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
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
import { MODE_REMOVED_FIX, retriesError } from "./agent.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  notTextError,
  rejectNonText,
  siteKey,
} from "./base.ts";

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
    const adaptive = cursor.get("adaptive_questions");
    if (isRecord(adaptive)) {
      if (Object.hasOwn(adaptive, "mode")) {
        throw diagnostics.error(
          `node ${id} (kind=${this.kind}): \`adaptive_questions.mode:\` was removed: ${MODE_REMOVED_FIX}`,
          { range: keyRange(cursor.child("adaptive_questions"), "mode") ?? cursor.keyRangeOf("adaptive_questions") },
        );
      }
      for (const key of Object.keys(adaptive)) {
        if (ADAPTIVE_KEYS.has(key)) continue;
        throw diagnostics.error(
          `node ${id} (kind=${this.kind}): \`adaptive_questions:\` field ${reprText(key)} is not allowed ` +
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
          `node ${id} (kind=${this.kind}): ` + notTextError("adaptive_questions.prompt", brief),
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

/**
 * Return the input name when `text` is exactly one `${name}` span, else `null`.
 *
 * `null` also covers surrounding text, a dotted path, a call, an operator, and a span that does
 * not parse.
 */
function wholeRefName(text: string): string | null {
  let segments: TemplateSegment[];
  try {
    segments = scanTemplate(text);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  const [segment] = segments;
  if (segments.length !== 1 || !(segment instanceof Span) || !isBareName(segment.tree)) return null;
  return String((segment.tree as Tree).children[0]);
}

/** Return the answer-schema entry for field `name` of type `typ`. */
function fieldSchema(name: string, typ: Type, required: boolean): Readonly<Record<string, unknown>> {
  const entry: Record<string, unknown> = { name, type: typ.kind, required };
  if (typ.tags !== undefined && typ.tags.size > 0) {
    // A Literal enum: the allowed values.
    entry["enum"] = [...typ.tags].sort(compareCodeUnits);
  } else if (typ.element?.tags !== undefined && typ.element.tags.size > 0) {
    // A list of a Literal enum: the element's values.
    entry["enum"] = [...typ.element.tags].sort(compareCodeUnits);
  }
  if (typ.kind === ValueKind.OBJECT && typ.fields !== undefined && typ.fields.size > 0) {
    const requiredKeys = typ.required ?? new Set<string>();
    entry["fields"] = [...typ.fields].map(([key, field]) => fieldSchema(key, field, requiredKeys.has(key)));
  }
  return entry;
}

/**
 * Return a light answer-field schema for a HUMAN_INPUT, derived from its output type.
 *
 * A record output gives one entry per field (recursively); any other type (a scalar, an enum, a
 * list, a union) gives a single `answer` entry. Each entry is `{name, type, required}`, `type`
 * being the type's `ValueKind` text, plus a sorted `enum` for a `Literal` (or a list of one) and
 * `fields` for a nested record. A host renderer prompts against it.
 *
 * Example: `answerSchema(Type.scalar(ValueKind.STRING))` is
 * `[{ name: "answer", type: "str", required: true }]`; `answerSchema(null)` is `[]`.
 */
export function answerSchema(typ: Type | null): Readonly<Record<string, unknown>>[] {
  if (typ === null) return [];
  if (typ.kind === ValueKind.OBJECT && typ.fields !== undefined && typ.fields.size > 0) {
    const requiredKeys = typ.required ?? new Set<string>();
    return [...typ.fields].map(([key, field]) => fieldSchema(key, field, requiredKeys.has(key)));
  }
  return [fieldSchema("answer", typ, !typ.nullable)];
}
