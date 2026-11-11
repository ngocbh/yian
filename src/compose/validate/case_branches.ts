/**
 * Graph-stage: a `case` branch names a node or calls a flow, and a `case` has no `output:`.
 *
 * A `case` only routes. Each `then:`/`else:` is classified by its YAML tag and scalar style:
 *
 * - an unquoted string starting with `call(` calls a flow. A whole-value directive is left for
 *   `desugarCaseCallTargets`; any other text starting with `call(` is a load error naming the
 *   problem. Any other untagged unquoted string without `${` that names a node names it; one that
 *   names no node is a load error that suggests the nearest node and the value entry to write for
 *   the text. An unquoted date is text, so it is such a word too;
 * - anything else is a value: a tagged value (even a word naming a node, as in a value entry),
 *   quoted text (`"..."`, `'...'`, or a block `|`/`>`), `null` or an empty `else:`, a number
 *   (infinity too), a bool, a mapping, a list, and unquoted text holding `${`. A value branch is a
 *   load error that shows the rewrite: a value entry in `nodes:` (`route_then_0: !str approve`)
 *   and the branch naming it. The entry keeps the branch's own tag; else it takes the case's
 *   former `output:` type when a tag can spell it and the value fits it (a lone `${input.x}` or
 *   `${node.output}` reference fits when its type does); else the reference's type, or a tag
 *   guessed from the YAML value; else a `!<type>` placeholder (a lone span of another type).
 *
 * A value no entry can hold is its own load error instead of a rewrite: a value that contains
 * itself, a written-out `!object` mapping with an empty key, a `call(...)` nested in it, a flow
 * call or a malformed span inside its `${...}`, a span reading the case itself, and a quoted
 * `call(...)` holding a span (quoting a call makes it text).
 *
 * A case `output:` (`output: str`, or tagged `output: !str`) is a load error at the `output:` that
 * says to tag the result where it is read (`x: !str ${route.output}`). A case with both reports
 * its first value branch first.
 *
 * Runs in `assemble` before `desugarInlineCalls`, so the branches are still as authored.
 *
 * Knows about: `compile/validation` (type relations), `diagnostics`, `expr`, `typesys`, and the
 * compose peers `desugar/inline_calls` (the `call(...)` check), `desugar/value_entries` (value
 * entry descriptors and the values they can't hold), `helpers` (errors, the `call(` head,
 * positional metadata, YAML quoting, tag spelling, type reading), `nodes/case` (the descriptor,
 * number spelling), `parse/nodes` (identifiers, the example type of a value) and `source`.
 */

import { typeRelation } from "../../compile/validation.ts";
import { getCloseMatches, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError, exprRefsOf } from "../../expr/index.ts";
import { wholeSpanInterior } from "../../expr/grammar.ts";
import { flowCallCalleesInSpans } from "../../expr/template.ts";
import { BUILTIN_TYPE_NAMES, type TypeRegistry } from "../../typesys/types.ts";
import { buildValueAs, isList, isRecord, rejectDeepNesting, type Type, TypeCheckError } from "../../typesys/values.ts";
import { ArgumentValueError, desugarCallDirectives } from "../desugar/inline_calls.ts";
import { EMPTY_RECORD_KEY, infersRecord, ValueEntryDescriptor } from "../desugar/value_entries.ts";
import { CALL_HEAD, spanCallFlowId } from "../helpers/directives.ts";
import { type LoadDiagnostics, LoadError, sourceText } from "../helpers/errors.ts";
import { nth } from "../helpers/source.ts";
import {
  CLOSE_MATCH_CUTOFF,
  hasSpan,
  oldFormSuggestion,
  oldTypedText,
  QUOTED_STYLES,
  resolveTypeText,
  suggestedLiteral,
  suggestTag,
  tagText,
  yamlQuoted,
} from "../helpers/tags.ts";
import { readType } from "../helpers/types.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { CaseDescriptor, numberText } from "../nodes/case.ts";
import { containsItself, exampleType, isIdentifier } from "../parse/nodes.ts";
import type { ScalarStyle } from "../source/common.ts";
import { parseYamlLiteral, type SourceCursor, SourceSyntaxError } from "../source/index.ts";

// A value longer than this many code points is cut in a message; the author writes it as a block.
const SHOWN_LIMIT = 60;
// Characters that end or nest a plain scalar inside a YAML flow mapping or list.
const FLOW_SPECIAL: ReadonlySet<string> = new Set(",[]{}:#&*!|>'\"%@`");
// The tag a branch's authored text starts with, such as `!int ` in `!int 3`.
const TAG_PREFIX = /!\S*\s*/;
// A lone reference whose type is known before the build: `input.x` or `node.output`.
const REFERENCE = /^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$/;
// Shown after an entry tagged `!<type>` for a lone span whose type isn't known yet.
const SPAN_TYPE_NOTE = "name its type for `<type>`";

/** Returns the authored `output:` of a case, or `null` when it has none. */
export type OutputReader = (nodeId: string) => SourceCursor | null;

/** One authored `then:`/`else:` of a case. */
interface Branch {
  /** `"then:"` or `"else:"`. */
  readonly label: string;
  /** The authored value. */
  readonly value: unknown;
  /** Its YAML scalar style; `null` (unknown) reads as plain. */
  readonly style: ScalarStyle | null;
  /** Its local YAML tag, such as `"int"`, or `null`. */
  readonly tag: string | null;
  /** The `!include(T) PATH` it was read with, as written, or `null`. */
  readonly include: string | null;
  /** The branch's range, else the case's range. */
  readonly range: SourceRange | null;
  /**
   * The value entry id a rewrite suggests before it is made unique among the node ids, such as
   * `route_then_0`.
   */
  readonly entryBase: string;
}

/** The `output:` a case declares, read for its migration message. */
interface CaseOutput {
  /** The declared type, or `null` when it does not resolve. */
  readonly type: Type | null;
  /**
   * The type as written (a tag, a type name, or a record in flow style); `null` when `output:` is
   * empty or contains itself.
   */
  readonly text: string | null;
  /** The `output:` range, else the case's range. */
  readonly range: SourceRange | null;
}

/**
 * Reject every `case` branch that is not a node or a call, and every case `output:`.
 *
 * @param descriptors The flow's parsed descriptors, keyed by node id; branches are as authored.
 * @param options.registry The flow's typedefs, for spelling a type as a tag.
 * @param options.inputTypes The flow's input types, which tag the value entry of a lone
 *   `${input.x}` branch.
 * @param options.outputAt Returns a case's authored `output:` by node id, or `null` when it has
 *   none. It sees a tagged `output: !int`, which leaves the descriptor's `outputs` empty.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError when a branch holds a value, is an unquoted word that names no node, or is
 *   text starting with `call(` that is not one well-formed directive; a branch's value can't be
 *   held by a value entry (a value that contains itself, a written-out `!object` mapping with an
 *   empty key, a nested `call(...)`, a malformed span, a flow call in a span, a span reading the
 *   case, a quoted `call(...)` holding a span); or a case declares `output:`. Cases are checked in
 *   order, each one's `then:` branches, its `else:`, then its `output:`.
 *
 * Example: a case `route` with `then: "approve"` fails with ``case node 'route' then: a branch
 * names a node or calls a flow; put the value in its own entry in `nodes:` and name it:
 * `route_then_0: !str approve` and `then: route_then_0` ``.
 */
export function checkCaseBranches(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: {
    registry: TypeRegistry;
    inputTypes: ReadonlyMap<string, Type>;
    outputAt: OutputReader;
    diagnostics: LoadDiagnostics;
  },
): void {
  const { registry, inputTypes, outputAt, diagnostics } = options;
  for (const [nid, desc] of descriptors) {
    if (!(desc instanceof CaseDescriptor)) continue;
    const output = readOutput(desc, outputAt(desc.id), registry);
    for (const branch of branches(desc)) {
      checkBranch(branch, { caseId: nid, output, descriptors, registry, inputTypes, diagnostics });
    }
    if (output !== null) {
      throw diagnostics.error(
        `case node ${reprText(nid)}: a \`case\` routes and has no \`output:\`; tag its result ` +
          `where you read it${readExample(nid, output, registry)}`,
        { range: output.range },
      );
    }
  }
}

/** Return the case's `then:` branches in order, then its `else:` when written. */
function branches(desc: CaseDescriptor): Branch[] {
  const found: Branch[] = [];
  for (const [index, spec] of desc.cases.entries()) {
    if (!isRecord(spec) || !Object.hasOwn(spec, "then")) continue;
    found.push({
      label: "then:",
      value: spec["then"],
      style: nth(desc.thenStyles, index),
      tag: nth(desc.thenTags, index),
      include: nth(desc.thenIncludes, index),
      range: nth(desc.thenRanges, index) ?? desc.range,
      entryBase: `${desc.id}_then_${index}`,
    });
  }
  if (desc.hasElse) {
    found.push({
      label: "else:",
      value: desc.else_,
      style: desc.elseStyle,
      tag: desc.elseTag,
      include: desc.elseInclude,
      range: desc.fieldRanges.get(siteKey("field", "else")) ?? desc.range,
      entryBase: `${desc.id}_else`,
    });
  }
  return found;
}

/**
 * Accept a node name or a `call(...)` branch; reject anything else with its rewrite.
 *
 * @param branch The branch.
 * @param options.caseId The case that owns it.
 * @param options.output The case's former `output:`, which tags the value entry a rewrite shows
 *   when the value fits it.
 * @param options.descriptors The flow's descriptors; a bare word must name one, and a suggested
 *   entry id must not.
 * @param options.registry The flow's typedefs.
 * @param options.inputTypes The flow's input types.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError as `checkCaseBranches` says.
 */
function checkBranch(
  branch: Branch,
  options: {
    caseId: string;
    output: CaseOutput | null;
    descriptors: ReadonlyMap<string, NodeDescriptor>;
    registry: TypeRegistry;
    inputTypes: ReadonlyMap<string, Type>;
    diagnostics: LoadDiagnostics;
  },
): void {
  const { caseId, output, descriptors, registry, inputTypes, diagnostics } = options;
  const host = `case node ${reprText(caseId)} ${branch.label}`;
  if (branch.include !== null) {
    // A value a file holds goes in its own entry read the same way, never spelled out.
    const entry = freeId(branch.entryBase, descriptors);
    throw valueError(host, branch, entry, `\`${entry}: ${branch.include}\``, diagnostics);
  }
  const value = branch.value;
  if (typeof value === "string" && !QUOTED_STYLES.has(branch.style)) {
    if (CALL_HEAD.test(value.trim())) {
      checkCall(branch, host, { descriptors, diagnostics });
      return;
    }
    const old = branch.tag === null ? oldTypedText(value, registry) : null;
    if (old !== null && old[2]) {
      // The old `Type = value` text: the type it names tags the entry.
      const entry = freeId(branch.entryBase, descriptors);
      const shown = oldFormSuggestion(entry, old[0], old[1], old[2], { registry, placeholder: null });
      throw valueError(host, branch, entry, shown, diagnostics);
    }
    // A tag makes the branch a value, as it makes a `nodes:` entry one.
    if (branch.tag === null && !value.includes("${")) {
      if (descriptors.has(value)) return;
      throw unknownWordError(host, branch, { descriptors, diagnostics });
    }
  } else if (typeof value === "string" && CALL_HEAD.test(value.trim()) && value.includes("${")) {
    throw diagnostics.error(
      `${host} a quoted \`call(...)\` is text, so it can't hold \`\${...}\` here; ` +
        "drop the quotes to call the flow, or reword the text",
      { range: branch.range },
    );
  }
  rejectUnholdable(value, { host, branch, caseId, typeNames: new Set(registry.keys()), diagnostics });
  const entry = freeId(branch.entryBase, descriptors);
  const shown = entryText(entry, branch, { host, output, descriptors, registry, inputTypes, diagnostics });
  throw valueError(host, branch, entry, shown, diagnostics);
}

/**
 * Reject unquoted text starting with `call(` that is not one well-formed directive.
 *
 * @param branch The branch; its value is unquoted text starting with `call(`.
 * @param host Message prefix naming the branch, such as `"case node 'r' then:"`.
 * @param options.descriptors The flow's descriptors; a suggested entry id must not name one.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError when the directive has an unreadable argument or is not exactly one call.
 *   Unless the text holds a span or only one argument is wrong, the message shows the value entry
 *   to write for the text.
 */
function checkCall(
  branch: Branch,
  host: string,
  options: { descriptors: ReadonlyMap<string, NodeDescriptor>; diagnostics: LoadDiagnostics },
): void {
  const { descriptors, diagnostics } = options;
  const value = branch.value as string;
  try {
    desugarCallDirectives(value, () => "__check");
  } catch (error) {
    if (error instanceof ArgumentValueError) {
      // The target is a single call; only one of its arguments is wrong.
      throw diagnostics.error(`${host} ${error.message}`, { range: branch.range, cause: error });
    }
    if (!(error instanceof ExpressionError)) throw error;
    // A value entry can't hold quoted call text with a span, so it gets no rewrite.
    let hint = "";
    if (!value.includes("${")) {
      const entry = freeId(branch.entryBase, descriptors);
      hint = `. For the text, add a value entry and name it: \`${entry}: !str ${scalarText(value.trim(), false)}\``;
    }
    throw diagnostics.error(`${host} a branch target must be a node id or a single call: ${error.message}${hint}`, {
      range: branch.range,
      cause: error,
    });
  }
}

/**
 * Build the error for an unquoted word that names no node.
 *
 * @param host Message prefix naming the branch.
 * @param branch The branch; its value is unquoted text without a span.
 * @param options.descriptors The flow's descriptors, offered as close matches; a suggested entry
 *   id must not name one.
 * @param options.diagnostics Source ownership for located errors.
 * @returns Located at the branch; it names the nearest node when one is close and shows the `!str`
 *   value entry for the text, named `<word>_text` when that is an identifier, else after the
 *   branch.
 */
function unknownWordError(
  host: string,
  branch: Branch,
  options: { descriptors: ReadonlyMap<string, NodeDescriptor>; diagnostics: LoadDiagnostics },
): LoadError {
  const { descriptors, diagnostics } = options;
  const word = branch.value as string;
  const matches = getCloseMatches(word, descriptors.keys(), 1, CLOSE_MATCH_CUTOFF);
  const suggestion = matches.length > 0 ? ` Did you mean \`${matches[0]}\`?` : "";
  const base = isIdentifier(`${word}_text`) ? `${word}_text` : branch.entryBase;
  const entry = freeId(base, descriptors);
  return diagnostics.error(
    `${host} \`${word}\` names no node.${suggestion} For the text, add a value entry and ` +
      `name it: \`${entry}: !str ${scalarText(word, false)}\``,
    { range: branch.range },
  );
}

/**
 * Reject a branch value, or a part of one, that no value entry can hold.
 *
 * Infinity is a float value a value entry holds, and the YAML reader refuses NaN, so a number
 * always passes.
 *
 * @param value The branch value or a part of one.
 * @param options.host Message prefix naming the branch.
 * @param options.branch The branch, for its label and range.
 * @param options.caseId The case that owns the branch.
 * @param options.typeNames The flow's typedef names; a call to one inside `${...}` builds a record
 *   and is no flow call.
 * @param options.diagnostics Source ownership for located errors.
 * @param options.nested Whether `value` sits inside a mapping or list (default `false`).
 * @throws LoadError when the value contains itself, or a nested string starts with `call(`, or a
 *   span is malformed, calls a flow, or reads the case.
 */
function rejectUnholdable(
  value: unknown,
  options: {
    host: string;
    branch: Branch;
    caseId: string;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
    nested?: boolean;
  },
): void {
  const { host, branch, diagnostics, nested = false } = options;
  if (!nested && containsItself(value)) {
    throw diagnostics.error(`${host} a branch value can't contain itself`, { range: branch.range });
  }
  if (typeof value === "string") {
    if (nested && CALL_HEAD.test(value.trim())) {
      throw diagnostics.error(
        `${host} a \`call(...)\` nested in a value is not supported; write the call ` +
          `as the branch (\`${branch.label} call(...)\`) or in a node, or reword the text`,
        { range: branch.range },
      );
    }
    if (value.includes("${")) rejectSpan(value, options);
    return;
  }
  if (!isRecord(value) && !isList(value)) return;
  for (const item of isRecord(value) ? Object.values(value) : value) {
    rejectUnholdable(item, { ...options, nested: true });
  }
}

/**
 * Reject text whose `${...}` is malformed, calls a flow, or reads the case.
 *
 * @param text A string of the branch value holding `${`.
 * @param options.host Message prefix naming the branch.
 * @param options.branch The branch, for its label and range.
 * @param options.caseId The case that owns the branch.
 * @param options.typeNames The flow's typedef names.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError as `rejectUnholdable` says for a span.
 */
function rejectSpan(
  text: string,
  options: {
    host: string;
    branch: Branch;
    caseId: string;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
  },
): void {
  const { host, branch, caseId, typeNames, diagnostics } = options;
  let refs: string[];
  let callees: string[];
  try {
    refs = exprRefsOf(text);
    // A builtin type called like a function gets its own error at the entry.
    callees = flowCallCalleesInSpans(text, typeNames).filter((callee) => !BUILTIN_TYPE_NAMES.has(callee));
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw diagnostics.error(`${host} ${error.message}`, { range: branch.range, cause: error });
  }
  if (callees[0] === "call") {
    throw diagnostics.error(
      `${host} call(...) can't go inside \`\${...}\`. Write it unquoted as the branch, ` +
        `\`${branch.label} call(${spanCallFlowId(text)}, ...)\`.`,
      { range: branch.range },
    );
  }
  if (callees.length > 0) {
    throw diagnostics.error(
      `${host} a flow call ${reprText(callees[0]!)} inside \`\${...}\` is not supported; call ` +
        `the flow as the branch: \`${branch.label} call(${callees[0]}, ...)\``,
      { range: branch.range },
    );
  }
  if (refs.some((ref) => ref.split(".", 1)[0] === caseId)) {
    throw diagnostics.error(`${host} the value reads ${reprText(caseId)} itself`, { range: branch.range });
  }
}

/**
 * Build the error for a value branch, showing its value entry and the branch.
 *
 * @param host Message prefix naming the branch.
 * @param branch The branch.
 * @param entry The suggested value entry id.
 * @param shown The value entry as the message shows it.
 * @param diagnostics Source ownership for located errors.
 * @returns Located at the branch.
 */
function valueError(host: string, branch: Branch, entry: string, shown: string, diagnostics: LoadDiagnostics): LoadError {
  return diagnostics.error(
    `${host} a branch names a node or calls a flow; put the value in its own entry in ` +
      `\`nodes:\` and name it: ${shown} and \`${branch.label} ${entry}\``,
    { range: branch.range },
  );
}

/**
 * Spell the value entry a value branch becomes, for its error message.
 *
 * @param entry The entry id.
 * @param branch The branch whose value the entry holds.
 * @param options.host Message prefix naming the branch.
 * @param options.output The case's former `output:`; its type tags the entry when a tag can spell
 *   it and the value fits it.
 * @param options.descriptors The flow's descriptors, which type a lone `${node.output}`.
 * @param options.registry The flow's typedefs.
 * @param options.inputTypes The flow's input types, which type a lone `${input.x}`.
 * @param options.diagnostics Source ownership, for a scalar's authored text.
 * @returns Such as `` `route_then_0: !str approve` ``, then any note in parentheses. A value past
 *   `SHOWN_LIMIT` code points is cut with `…` and followed by `(write it as a block)`.
 * @throws LoadError when the entry would be a written-out `!object` over a mapping with an empty
 *   key.
 */
function entryText(
  entry: string,
  branch: Branch,
  options: {
    host: string;
    output: CaseOutput | null;
    descriptors: ReadonlyMap<string, NodeDescriptor>;
    registry: TypeRegistry;
    inputTypes: ReadonlyMap<string, Type>;
    diagnostics: LoadDiagnostics;
  },
): string {
  const { host, diagnostics } = options;
  const [tag, typ, text, note] = entryTag(branch, valueText(branch, diagnostics), options);
  if (typ !== null && infersRecord(typ, branch.value) && holdsEmptyKey(branch.value)) {
    throw diagnostics.error(`${host} ${EMPTY_RECORD_KEY}`, { range: branch.range });
  }
  return shown(entry, tag, text, note);
}

/**
 * Choose the tag of the value entry a value branch becomes.
 *
 * @param branch The branch whose value the entry holds.
 * @param authored The value's YAML, without the branch's own tag.
 * @param options.output The case's former `output:`.
 * @param options.descriptors The flow's descriptors.
 * @param options.registry The flow's typedefs.
 * @param options.inputTypes The flow's input types.
 * @returns The tag text (`<type>` when no tag is known), the type it names when known, the
 *   value's YAML after it, and any note to show after the entry.
 */
function entryTag(
  branch: Branch,
  authored: string,
  options: {
    output: CaseOutput | null;
    descriptors: ReadonlyMap<string, NodeDescriptor>;
    registry: TypeRegistry;
    inputTypes: ReadonlyMap<string, Type>;
  },
): [tag: string, typ: Type | null, text: string, note: string | null] {
  const { output, registry } = options;
  if (branch.tag !== null) {
    const resolved = resolveTypeText(branch.tag, registry);
    return [branch.tag, resolved === null ? null : resolved[0], authored, null];
  }
  const interior = typeof branch.value === "string" ? wholeSpanInterior(branch.value) : null;
  const reference = interior === null ? null : referenceType(interior, options);
  if (output !== null) {
    const tag = tagText(output.type, { registry, typeText: output.text });
    if (output.type !== null && tag !== null) {
      let text = authored;
      let note: string | null = null;
      // Spelled to fit the type where that is cheap: `3` for `str` is `"3"`.
      if (branch.value !== null) [text, note] = suggestedLiteral(text, output.type);
      if (fits(text, output.type, reference)) return [tag, output.type, text, note];
    }
  }
  const typ = interior !== null ? reference : exampleType(branch.value);
  const tag = tagText(typ, { registry });
  if (tag === null) return ["<type>", null, authored, SPAN_TYPE_NOTE];
  return [tag, typ, authored, null];
}

/**
 * Spell a value entry in backticks, then the note and the advice for a cut value. The cut counts
 * code points, so it never splits an astral character.
 */
function shown(entry: string, tag: string, text: string, note: string | null = null): string {
  const chars = Array.from(text);
  const cut = chars.length <= SHOWN_LIMIT ? text : chars.slice(0, SHOWN_LIMIT).join("") + "…";
  let result = `\`${entry}: !${tag} ${cut}\``;
  if (note !== null) result += ` (${note})`;
  if (chars.length > SHOWN_LIMIT) result += " (write it as a block)";
  return result;
}

/**
 * Report whether a tag of type `typ` reads the YAML `text`.
 *
 * @param text The value's YAML.
 * @param typ The tag's type.
 * @param reference The known type of a lone reference `text` is, else `null`.
 * @returns For a lone reference of known type, whether that type fits `typ`; for other text
 *   holding a span, true, since its type isn't known yet; else whether `typ` reads the value.
 */
function fits(text: string, typ: Type, reference: Type | null): boolean {
  if (reference !== null) return typeRelation(reference, typ) === "fits";
  if (hasSpan(text)) return true;
  try {
    buildValueAs(typ, parseYamlLiteral(text));
  } catch (error) {
    if (error instanceof SourceSyntaxError || error instanceof TypeCheckError) return false;
    throw error;
  }
  return true;
}

/**
 * Return the type a lone span reads when it is known before the build.
 *
 * @param interior The text inside a branch's lone `${...}`.
 * @param options.descriptors The flow's descriptors.
 * @param options.registry The flow's typedefs.
 * @param options.inputTypes The flow's input types.
 * @returns The type of the flow input `input.x`, or the declared output of the node `node.output`,
 *   reads; `null` for any other span.
 */
function referenceType(
  interior: string,
  options: {
    descriptors: ReadonlyMap<string, NodeDescriptor>;
    registry: TypeRegistry;
    inputTypes: ReadonlyMap<string, Type>;
  },
): Type | null {
  const { descriptors, registry, inputTypes } = options;
  const found = REFERENCE.exec(interior.trim());
  if (found === null) return null;
  const head = found[1]!;
  const name = found[2]!;
  if (head === "input") return inputTypes.get(name) ?? null;
  const desc = descriptors.get(head);
  if (name !== "output" || desc === undefined) return null;
  if (desc instanceof ValueEntryDescriptor) return desc.tag.type;
  const outputs = desc.outputs;
  const resolved = typeof outputs === "string" ? resolveTypeText(outputs, registry) : null;
  return resolved === null ? null : resolved[0];
}

/** Report whether a value that doesn't contain itself holds a mapping keyed `""`. */
function holdsEmptyKey(value: unknown): boolean {
  if (isRecord(value)) return Object.hasOwn(value, "") || Object.values(value).some(holdsEmptyKey);
  if (isList(value)) return value.some(holdsEmptyKey);
  return false;
}

/**
 * Spell a branch value as YAML to follow a tag on one line.
 *
 * Text is plain when plain YAML reads it back unchanged, else quoted; a number keeps its authored
 * text; a mapping or list is written in flow style.
 *
 * @param branch The branch; its value does not contain itself.
 * @param diagnostics Source ownership, for a scalar's authored text.
 * @returns The value's YAML, without the branch's own tag.
 */
function valueText(branch: Branch, diagnostics: LoadDiagnostics): string {
  const value = branch.value;
  if (typeof value === "string") return scalarText(value, false);
  if (isRecord(value) || isList(value)) return flowText(value);
  let authored = sourceText(diagnostics, branch.range);
  if (authored && branch.tag !== null) authored = authored.replace(TAG_PREFIX, "");
  if ((typeof value === "bigint" || typeof value === "number") && authored) return authored;
  return scalarText(value, true);
}

/**
 * Spell a mapping, list, or scalar that doesn't contain itself in YAML flow style.
 *
 * @param value The value or a part of one.
 * @returns Flow-style YAML, such as `{reply: "${input.text}", exited: true}`.
 */
function flowText(value: unknown): string {
  if (isRecord(value)) {
    const items = Object.entries(value).map(([key, item]) => `${scalarText(key, true)}: ${flowText(item)}`);
    return "{" + items.join(", ") + "}";
  }
  if (isList(value)) return "[" + value.map(flowText).join(", ") + "]";
  return scalarText(value, true);
}

/**
 * Spell a scalar as YAML that reads back as the same value.
 *
 * @param value The scalar.
 * @param flow Whether it sits inside a flow mapping or list, where `,`, `{`, `}`, `[`, `]`, and
 *   `:` would end or nest it, so text holding them is quoted.
 * @returns `null`, `true`/`false`, a number (an int's digits, `.inf` or `-.inf` for infinity, any
 *   other float positionally with a `.`, by `numberText`), or text: plain when plain YAML reads it
 *   back unchanged and it is not a `call(...)`, else quoted.
 */
function scalarText(value: unknown, flow: boolean): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "bigint") return String(value);
  if (typeof value === "number") {
    if (value === Infinity) return ".inf";
    if (value === -Infinity) return "-.inf";
    return numberText(value);
  }
  const text = String(value);
  if (CALL_HEAD.test(text.trim()) || (flow && Array.from(text).some((ch) => FLOW_SPECIAL.has(ch)))) {
    return yamlQuoted(text);
  }
  let plain: boolean;
  try {
    plain = parseYamlLiteral(text) === text;
  } catch (error) {
    if (!(error instanceof SourceSyntaxError)) throw error;
    plain = false;
  }
  return plain ? text : yamlQuoted(text);
}

/** Return `base`, or `base_2`, `base_3`, ... when it already names a node. */
function freeId(base: string, descriptors: ReadonlyMap<string, NodeDescriptor>): string {
  if (!descriptors.has(base)) return base;
  let suffix = 2;
  while (descriptors.has(`${base}_${suffix}`)) suffix += 1;
  return `${base}_${suffix}`;
}

/**
 * Read the `output:` a case declares, or return `null` when it has none.
 *
 * @param desc The case.
 * @param cursor The case's authored `output:`, or `null`.
 * @param registry The flow's typedefs the type resolves against.
 * @returns The declared type, as far as it resolves, and the `output:` range. An empty `output:`,
 *   one that contains itself, or one nested past the value limit has no text.
 */
function readOutput(desc: CaseDescriptor, cursor: SourceCursor | null, registry: TypeRegistry): CaseOutput | null {
  if (cursor === null) return null;
  const value = cursor.hasValue ? cursor.value : null;
  const spec = value !== null && value !== "" ? value : cursor.tag;
  // The key line, which a block mapping's value range starts below.
  const range = cursor.keyRange ?? cursor.range ?? desc.range;
  if (spec === null) return { type: null, text: null, range };
  if (isRecord(spec) || isList(spec)) {
    // The self-containment check and `flowText` recurse through the raw value, so an alias chain
    // could overflow them. This only decides whether the rejected `output:` can be shown, so the
    // value limit bounds it rather than the type rule.
    try {
      rejectDeepNesting(spec, "case output");
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      return { type: null, text: null, range };
    }
  }
  if (containsItself(spec)) return { type: null, text: null, range };
  if (typeof spec === "string") {
    const resolved = resolveTypeText(spec, registry);
    if (resolved === null) return { type: null, text: spec.trim(), range };
    return { type: resolved[0], text: resolved[1], range };
  }
  // A record spelled as written (`{reply: str}`) reads better than its canonical label.
  const text = flowText(spec);
  try {
    return { type: readType(spec, registry), text, range };
  } catch (error) {
    if (error instanceof LoadError || error instanceof TypeCheckError) return { type: null, text, range };
    throw error;
  }
}

/**
 * Return the example of reading a case's result with its former type as a tag.
 *
 * @param caseId The case.
 * @param output Its former `output:`.
 * @param registry The flow's typedefs.
 * @returns Such as `` , e.g. `x: !str ${route.output}` ``, or, for a type a tag can't spell, the
 *   advice to name it in `typedefs:`.
 */
function readExample(caseId: string, output: CaseOutput, registry: TypeRegistry): string {
  const read = `\${${caseId}.output}`;
  if (output.text === null) return `, e.g. \`x: !<type> ${read}\``;
  const suggestion = suggestTag("x", output.type, read, { registry, typeText: output.text });
  if (tagText(output.type, { registry, typeText: output.text }) === null) return `; ${suggestion}`;
  return `, e.g. ${suggestion}`;
}
