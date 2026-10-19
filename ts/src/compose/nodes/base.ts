/**
 * The base descriptor contract shared by every node kind: the parse-time analogue of
 * `nodes/base.ts`.
 *
 * It owns the shared messages for a field whose value YAML read as the wrong type (text or
 * otherwise), for a list or record holding a `${...}` span where the field binds whole, and for
 * an `input:` that is not a mapping, so every kind reports those mistakes in the same words.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { callSuffixOf, type ExprNode, Tree, wholeSpanInterior } from "../../expr/grammar.ts";
import { scanTemplate, Span, templateRefs } from "../../expr/template.ts";
import { bindingSource } from "../../nodes/contracts.ts";
import { isRecord, valueRepr } from "../../typesys/values.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { hasSpan } from "../helpers/tags.ts";
import { yamlReadAs } from "../helpers/yaml_words.ts";

/**
 * Return the key of a descriptor's `fieldRanges` entry for the site named `name`: a field, an
 * `input:` binding or an `llm_config` field.
 *
 * Example: `siteKey("field", "inputs")` is `"field:inputs"`.
 */
export function siteKey(kind: "field" | "input" | "llm_config", name: string): string {
  // One text key, since a Map compares a [kind, name] pair by identity, not by value.
  return `${kind}:${name}`;
}

/**
 * Describe a non-mapping `input:` value and how to write it as a binding mapping.
 *
 * @param kind The node kind; it picks the message's binding noun and fix verb.
 * @param value The authored `input:` value, which is not a mapping.
 * @returns What `input:` must be, what it got, and a fix after a `;`. A lone reference `${R}`
 *   gets the example `<field>: ${R.<field>}`.
 */
export function inputShapeProblem(kind: string, value: unknown): string {
  let noun: string;
  let verb: string;
  if (kind === "loop") {
    [noun, verb] = ["carried fields", "list each carried field"];
  } else if (kind === "call" || kind === "map") {
    [noun, verb] = ["the child's arguments", "bind each argument"];
  } else {
    [noun, verb] = ["parameter names to bindings", "bind each parameter"];
  }
  const path = directReferencePath(value);
  let got: string;
  let fix: string;
  if (path !== null) {
    got = `the reference '\${${path}}'`;
    fix = `${verb}, such as \`<field>: \${${path}.<field>}\``;
  } else if (typeof value === "string" && wholeSpanInterior(value) !== null) {
    got = `the expression '${value}'`;
    fix = `${verb}, such as \`<field>: \${...}\``;
  } else {
    got = valueRepr(value);
    if ((Array.isArray(value) || typeof value === "string") && value.length === 0) {
      fix = "write `input: {}` or remove `input:`";
    } else if (typeof value === "string" && hasSpan(value)) {
      fix = `${verb}, such as \`<field>: \${...}\``;
    } else {
      fix = `${verb}, such as \`<field>: ...\``;
    }
  }
  return `\`input:\` must be a mapping of ${noun} (got ${got}); ${fix}`;
}

/**
 * Reject a list or record holding a `${...}` span in a field that binds whole.
 *
 * Such a field's container is a literal, so a span inside it would stay text.
 *
 * @param kind The node kind, for the message.
 * @param nodeId The node id, for the message.
 * @param key The authored field name.
 * @param value The field's authored value.
 * @param range Where the error points.
 * @param diagnostics Factory for located load errors.
 * @throws LoadError when `value` is a list or record with a span inside it.
 */
export function rejectContainerSpans(
  kind: string,
  nodeId: string,
  key: string,
  value: unknown,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  if ((Array.isArray(value) || isRecord(value)) && holdsSpan(value)) {
    throw diagnostics.error(
      `${kind} node ${reprText(nodeId)} ${key}: a value holding \${...} can't be a list or ` +
        "mapping; write the list as an expression, e.g. `${[a.output, b.output]}`",
      { range },
    );
  }
}

/**
 * Return the load error message for a field whose value YAML read as the wrong type.
 *
 * @param field The field's path as authored, such as `over` or `questions[0].header`.
 * @param needed What the field takes, with its article, such as `text` or `a list`.
 * @param value The field's value as YAML read it; `null` is reported as empty.
 * @param fix What the author should write instead.
 * @returns The message, naming what YAML read the value as and the fix.
 *
 * Example: `valueTypeError("over", "a list", 5n, "write a list")` is
 * ``"`over:` must be a list, but YAML read it as an integer; write a list"``;
 * `valueTypeError("over", "a list", null, "write a list")` is
 * ``"`over:` must be a list, but it is empty; write a list"``.
 */
export function valueTypeError(field: string, needed: string, value: unknown, fix: string): string {
  if (value === null) return `\`${field}:\` must be ${needed}, but it is empty; ${fix}`;
  return `\`${field}:\` must be ${needed}, but YAML read it as ${yamlReadAs(value)}; ${fix}`;
}

/**
 * Return the load error message for a text field whose value YAML read as non-text.
 *
 * @param field The field's path as authored, such as `prompt` or `questions[0].header`.
 * @param value The field's value as YAML read it.
 * @param fix What the author should write instead. `null` uses "write the text" for an empty
 *   value and "put it in quotes" otherwise.
 * @returns The message, naming what YAML read the value as and the fix.
 *
 * Example: `notTextError("prompt", 5n)` is
 * ``"`prompt:` must be text, but YAML read it as an integer; put it in quotes"``;
 * `notTextError("questions[0].header", null)` is
 * ``"`questions[0].header:` must be text, but it is empty; write the text"``.
 */
export function notTextError(field: string, value: unknown, fix: string | null = null): string {
  fix ??= value === null ? "write the text" : "put it in quotes";
  return valueTypeError(field, "text", value, fix);
}

/**
 * Reject a text or expression field whose value YAML read as non-text.
 *
 * @param kind The node kind, for the message.
 * @param nodeId The node id, for the message.
 * @param key The authored field name.
 * @param value The field's authored value.
 * @param range Where the error points.
 * @param diagnostics Factory for located load errors.
 * @param options.required Whether the field needs a value: `false` (the default) lets `null`
 *   through as an absent field; `true` rejects it as empty.
 * @param options.fix What the author should write instead; `null` uses `notTextError`'s default.
 * @throws LoadError when `value` is neither text nor `null`, or is `null` while `required` is set.
 */
export function rejectNonText(
  kind: string,
  nodeId: string,
  key: string,
  value: unknown,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
  options: { required?: boolean; fix?: string | null } = {},
): void {
  const { required = false, fix = null } = options;
  if (typeof value !== "string" && (value !== null || required)) {
    throw diagnostics.error(
      `node ${reprText(nodeId)} (kind=${kind}): ` + notTextError(key, value, fix),
      { range },
    );
  }
}

/** Whether any text inside `value`, or `value` itself, holds a `${...}` span. */
function holdsSpan(value: unknown): boolean {
  if (typeof value === "string") return hasSpan(value);
  if (isRecord(value)) return Object.values(value).some(holdsSpan);
  if (Array.isArray(value)) return value.some(holdsSpan);
  return false;
}

/**
 * Return the path a binding reads when it is one lone whole-value reference.
 *
 * Python keeps this in `compose/helpers/output_contracts.py`.
 *
 * @param binding A raw binding value or a `Binding` carrier.
 * @returns The dotted reference path (`item`, `item.name`, `a.output`), or `null` for a
 *   literal, an embedded or computed template, a call, or malformed text.
 */
function directReferencePath(binding: unknown): string | null {
  const source = bindingSource(binding);
  if (typeof source !== "string") return null;
  let segments;
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  const [segment] = segments;
  if (!(segments.length === 1 && segment instanceof Span)) return null;
  if (!isDirectReference(segment.tree)) return null;
  const refs = templateRefs(segments);
  return refs.length === 1 ? refs[0]! : null;
}

/** Whether a whole-value span's tree is a bare or dotted reference with no call suffix. */
function isDirectReference(tree: ExprNode): boolean {
  return tree instanceof Tree && tree.data === "refcall" && callSuffixOf(tree) === undefined;
}
