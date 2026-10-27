/**
 * Typed specs for an `ask`'s questions and a validating parser.
 *
 * A question is one prompt the host shows a person: free text by default, or a choice over
 * `options` (single- or multi-select). `parseQuestions` turns the raw list authored in a flow
 * into frozen `QuestionSpec` objects, applying the engine's authoring constraints (1 to
 * `current().maxQuestions` questions, unique headers).
 *
 * Knows about:   defaults, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { current } from "../../defaults.ts";
import { compareCodeUnits } from "../../diagnostics.ts";
import { typeName } from "../../typesys/types.ts";
import { isList, isRecord, Type, ValueKind, valueRepr, writeAsText } from "../../typesys/values.ts";

/** One selectable choice within a question. */
export interface OptionSpec {
  /** The choice text shown to and returned by the person. */
  readonly label: string;
  /** Helper text explaining the choice; empty when omitted. */
  readonly description: string;
}

/**
 * A single question posed to a person.
 *
 * With no `options` the question is free text; with `options` it is a choice, single-select
 * unless `multiSelect` is set.
 */
export interface QuestionSpec {
  /** The prompt text. */
  readonly question: string;
  /** A short key, unique within the set, that keys the collected answer. */
  readonly header: string;
  /** The selectable choices; empty means the question is free text. */
  readonly options: readonly OptionSpec[];
  /** Whether the person may pick more than one option; ignored for free-text questions. */
  readonly multiSelect: boolean;
}

/** Thrown when a raw question list violates the authoring constraints. */
export class QuestionSpecError extends RangeError {
  override name = "QuestionSpecError";
}

const QUESTION_KEYS: readonly string[] = ["question", "header", "options", "multi_select"];
const OPTION_KEYS: readonly string[] = ["label", "description"];

/**
 * Validate a raw question list and return the parsed questions, in input order.
 *
 * The list must hold 1 to `current().maxQuestions` items and every `header` must be unique.
 * Each item is checked in this order, and the first problem is reported as
 * `question[<i>] is invalid: <problem>`: it is a record; it has no key but `question`, `header`,
 * `options` and `multi_select`; it has `question` and `header`; `question`, then `header`, is
 * text; `options`, when present, is a list whose every entry is a record with no key but
 * `label` and `description`, has `label`, and holds text in both; `multi_select`, when present,
 * is `true` or `false`. A key given as `null` counts as present. Omitted `options` read `[]`, an
 * omitted `description` `""`, an omitted `multi_select` `false`. Each spec, its options and each
 * option are frozen.
 *
 * @throws QuestionSpecError when `raw` is not a list, has too few or too many items, an item is
 *   invalid, or two items share a `header`.
 */
export function parseQuestions(raw: unknown): QuestionSpec[] {
  if (!isList(raw)) {
    throw new QuestionSpecError(`questions must be a list, got ${typeName(raw)}`);
  }
  const most = current().maxQuestions;
  if (raw.length < 1 || raw.length > most) {
    throw new QuestionSpecError(`expected 1..${most} questions, got ${raw.length}`);
  }
  const questions = raw.map((item, i) => {
    try {
      return parseQuestion(item);
    } catch (error) {
      if (!(error instanceof QuestionSpecError)) throw error;
      throw new QuestionSpecError(`question[${i}] is invalid: ${error.message}`, { cause: error });
    }
  });
  const headers = questions.map((q) => q.header);
  if (new Set(headers).size !== headers.length) {
    throw new QuestionSpecError(`question headers must be unique, got ${writeAsText(headers)}`);
  }
  return questions;
}

/** Read one question item; the message names its first problem, without the item prefix. */
function parseQuestion(item: unknown): QuestionSpec {
  if (!isRecord(item)) throw new QuestionSpecError(`must be a record, got ${valueRepr(item)}`);
  checkKeys(item, QUESTION_KEYS, ["question", "header"], "");
  const question = readText(item, "question", "");
  const header = readText(item, "header", "");
  let options: readonly OptionSpec[] = Object.freeze([]);
  if (Object.hasOwn(item, "options")) {
    const raw = item["options"];
    if (!isList(raw)) throw new QuestionSpecError(`options must be a list, got ${valueRepr(raw)}`);
    options = Object.freeze(raw.map(parseOption));
  }
  let multiSelect = false;
  if (Object.hasOwn(item, "multi_select")) {
    const raw = item["multi_select"];
    if (typeof raw !== "boolean") {
      throw new QuestionSpecError(`multi_select must be true or false, got ${valueRepr(raw)}`);
    }
    multiSelect = raw;
  }
  return Object.freeze({ question, header, options, multiSelect });
}

/** Read option `j` of a question. */
function parseOption(raw: unknown, j: number): OptionSpec {
  if (!isRecord(raw)) throw new QuestionSpecError(`options[${j}] must be a record, got ${valueRepr(raw)}`);
  checkKeys(raw, OPTION_KEYS, ["label"], `options[${j}]: `);
  const label = readText(raw, "label", `options[${j}].`);
  const description = Object.hasOwn(raw, "description") ? readText(raw, "description", `options[${j}].`) : "";
  return Object.freeze({ label, description });
}

/** Refuse keys outside `allowed` (sorted), then absent `required` keys (in order). */
function checkKeys(
  record: Readonly<Record<string, unknown>>,
  allowed: readonly string[],
  required: readonly string[],
  prefix: string,
): void {
  const unknown = Object.keys(record)
    .filter((key) => !allowed.includes(key))
    .sort(compareCodeUnits);
  if (unknown.length > 0) throw new QuestionSpecError(`${prefix}unknown key(s) ${writeAsText(unknown)}`);
  const missing = required.filter((key) => !Object.hasOwn(record, key));
  if (missing.length > 0) throw new QuestionSpecError(`${prefix}missing key(s) ${writeAsText(missing)}`);
}

/** The text at `key`; `prefix` leads the message when it is not text. */
function readText(record: Readonly<Record<string, unknown>>, key: string, prefix: string): string {
  const value = record[key];
  if (typeof value !== "string") throw new QuestionSpecError(`${prefix}${key} must be text, got ${valueRepr(value)}`);
  return value;
}

/**
 * The type a synthesized compose agent generates questions against.
 *
 * It mirrors `QuestionSpec`/`OptionSpec` so the structured-output model emits a list of question
 * records: `question` and `header` text (required), `options` a list of `{label, description}`
 * records, `multi_select` a boolean. `description` is required on the option record so the model
 * always emits the helper text.
 */
export function questionListType(): Type {
  const option = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["label", Type.scalar(ValueKind.STRING)],
      ["description", Type.scalar(ValueKind.STRING)],
    ]),
    required: new Set(["label", "description"]),
  });
  const question = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["question", Type.scalar(ValueKind.STRING)],
      ["header", Type.scalar(ValueKind.STRING)],
      ["options", new Type({ kind: ValueKind.LIST_OBJECT, element: option })],
      ["multi_select", Type.scalar(ValueKind.BOOLEAN)],
    ]),
    required: new Set(["question", "header"]),
  });
  return new Type({ kind: ValueKind.LIST_OBJECT, element: question });
}
