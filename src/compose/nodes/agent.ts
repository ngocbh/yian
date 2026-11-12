/**
 * kind=agent: the LLM-reasoner node's descriptor.
 *
 * Owns the `agent` node's `AgentDescriptor`, which wraps the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) to reject the removed `mode:` key
 * and a variable named `inf` and to read a list `prompt:` into its items (`readPromptList`),
 * checks the shapes of its own fields, and builds the runtime `AgentNode`. Registered in `KINDS`
 * and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 * `readLlmConfig` reads an authored `llm_config:` layer with the `${...}` values and tags in it,
 * `validateLlmConfig` checks a flow, def or AGENT layer, and `checkLlmConfigValue` one holding
 * values.
 *
 * Knows about: diagnostics, compose source and helpers, expr (span reading and the binder),
 *              llm_clients (config and provider blocks), nodes (agent and contracts), skills,
 *              tools (grants), typesys and defaults.
 */

import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import { current } from "../../defaults.ts";
import { MISSING_ROOT } from "../../expr/expressions.ts";
import { wholeSpanInterior } from "../../expr/grammar.ts";
import { evalBinding, Verbatim } from "../../expr/template.ts";
import { hostOnlyFieldError, LLMConfig, responsesBlockError } from "../../llm_clients/config.ts";
import { locatedBlockProblems, PROVIDER_BLOCKS, ValidationError } from "../../llm_clients/provider_blocks.ts";
import {
  BASE_URL_FROM_VALUE,
  declaredHostOnlyError,
  isValue,
  LLM_CONFIG_KEY,
  valueLeaves,
} from "../../nodes/agent/llm_values.ts";
import {
  AgentNode,
  isMaxToolIterations,
  type PromptItem,
  type PromptText,
  STEER_INBOX_KEY,
} from "../../nodes/agent/node.ts";
import { memoryKey } from "../../nodes/base.ts";
import { typeToSchema } from "../../nodes/agent/structured.ts";
import {
  Binding,
  isExactly,
  Parameter,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { SkillError } from "../../skills/index.ts";
import { GrantError, parseToolGrants, type ToolGrant } from "../../tools/grants.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isList, isMemoryType, isRecord, type Type, typeLevels, valueRepr } from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError, rejectRemovedMemoryKey } from "../helpers/errors.ts";
import { checkMemoryGrants, grantErrorText, memoryBinding } from "../helpers/memory.ts";
import { includeFile, readTaggedValue } from "../helpers/tags.ts";
import { readVariables } from "../helpers/types.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { formerBooleanFix } from "../helpers/yaml_words.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  type BuildContext,
  BuildResult,
  cursorAt,
  NodeDescriptor,
  type NodeDescriptorFields,
  notTextDetail,
  notTextError,
  rangeAt,
  rejectInfinityName,
  rejectNonText,
  siteKey,
  wholeRefName,
} from "./base.ts";

/** Shared by the AGENT and `adaptive_questions:` errors for the removed `mode:` key. */
export const MODE_REMOVED_FIX =
  "an AGENT with tools runs the tool loop, one without tools makes a single call; delete this line";

/**
 * Return why `value` is not a valid `retries:` count, or `null` when it is.
 *
 * A count is a YAML int (`bigint`) from 0 to `Number.MAX_SAFE_INTEGER`, so it converts to a
 * `number` exactly; a boolean is refused, so `retries: true` doesn't mean 1.
 *
 * Example: `retriesError(-1n)` is `"must be a non-negative integer, got -1"`.
 */
export function retriesError(value: unknown): string | null {
  if (typeof value === "bigint" && value >= 0n) {
    if (value <= BigInt(Number.MAX_SAFE_INTEGER)) return null;
    return `must be at most ${Number.MAX_SAFE_INTEGER}, got ${valueRepr(value)}`;
  }
  return `must be a non-negative integer, got ${valueRepr(value)}`;
}

/**
 * Return `"; "` and the fix for a YAML 1.1 boolean spelling in a boolean field, else `""`.
 *
 * Example: `formerBooleanHint("yes")` is ``"; `yes` is text; write `true`"``.
 */
function formerBooleanHint(value: unknown): string {
  const fix = formerBooleanFix(value);
  return fix === null ? "" : `; ${fix[1]}`;
}

/**
 * Read one authored `llm_config` layer with the `${...}` values and tags in it.
 *
 * A layer without values is a plain record whose strings are `$$`-unescaped; a tagged literal in
 * it stands for its value. One holding values is a `ValueTemplate`. A whole record written as
 * exactly one `${...}` (optionally tagged) is a `Binding`; any other top-level text or scalar is
 * returned as written for the shape check to refuse.
 *
 * @param cursor The authored `llm_config:` value.
 * @param registry The flow's typedefs the tags resolve against.
 * @param options.prefix Message prefix naming the layer, such as `node 'a': llm_config`.
 * @param options.sitePath Catalog site prefix for the layer.
 * @param options.catalogBuilder File-owned origin catalog builder.
 * @param options.diagnostics Factory for located load errors.
 * @returns The layer, and the value range of each key written in it, keyed by its dotted path
 *   (`anthropic.thinking`); empty for a whole-record value.
 * @throws LoadError when a tag is malformed or names no type, a tag sits on the whole mapping, a
 *   tag has no value, or the tag reader refuses the value.
 */
export function readLlmConfig(
  cursor: SourceCursor,
  registry: TypeRegistry,
  options: {
    prefix: string;
    sitePath: readonly (string | number)[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): [layer: unknown, fieldRanges: Map<string, SourceRange>] {
  const { prefix, sitePath, catalogBuilder, diagnostics } = options;
  const slot = readTaggedValue(cursor, registry, {
    label: "llm_config",
    sitePath,
    catalogBuilder,
    diagnostics,
    ledger: catalogBuilder.consumedTags,
    nestedCallFix: "compute it in a node and read that node's output with `${...}`, or reword the text",
  });
  if (!slot.hasValue) {
    throw diagnostics.error(`${prefix}: a tag needs a value: \`llm_config: !${cursor.tag} \${...}\``, {
      range: cursor.range,
    });
  }
  const value = slot.value;
  if (typeof value === "string") {
    if (wholeSpanInterior(value) === null) return [value, new Map()];
    const origin = catalogBuilder.register([...sitePath, "path"], cursor);
    return [new Binding(value, { origin, contract: slot.contract }), new Map()];
  }
  if (slot.contract !== null) {
    throw diagnostics.error(
      `${prefix}: a tag on the whole mapping types nothing; tag a field's value, or write the whole record ` +
        "as one value: `llm_config: !T ${...}`",
      { range: cursor.range },
    );
  }
  const ranges = new Map<string, SourceRange>();
  collectKeyRanges(cursor, "", ranges);
  if (!(value instanceof ValueTemplate)) return [value, ranges];
  if (valueLeaves(value).length === 0) {
    // Only tagged literals, already checked against their tags: a literal layer.
    return [literalPart(value.value), ranges];
  }
  return [withKeyOrigins(value, cursor, sitePath, catalogBuilder), ranges];
}

/**
 * Give every written key of a template layer an origin, literal keys included.
 *
 * A call-time check of the resolved config can refuse a literal field (a provider block the
 * connection does not take), and its failure points at that field.
 */
function withKeyOrigins(
  template: ValueTemplate,
  cursor: SourceCursor,
  sitePath: readonly (string | number)[],
  catalogBuilder: SourceCatalogBuilder,
): ValueTemplate {
  const known = new Set(template.origins.map(([path]) => JSON.stringify(path)));
  const added: (readonly [ValuePath, OriginRef])[] = [];
  const visit = (at: SourceCursor, path: readonly string[]): void => {
    if (!isRecord(at.value)) return;
    for (const key of Object.keys(at.value)) {
      const child = at.child(key);
      const here = [...path, key];
      if (!known.has(JSON.stringify(here))) {
        const origin = catalogBuilder.register([...sitePath, "key", ...here], child);
        if (origin !== null) added.push([here, origin]);
      }
      visit(child, here);
    }
  };
  visit(cursor, []);
  return new ValueTemplate(template.value, [...template.origins, ...added]);
}

/** Record the value range of every mapping key under `cursor` by its dotted path. */
function collectKeyRanges(cursor: SourceCursor, prefix: string, ranges: Map<string, SourceRange>): void {
  if (!isRecord(cursor.value)) return;
  for (const key of Object.keys(cursor.value)) {
    const child = cursor.child(key);
    const path = `${prefix}${key}`;
    if (child.range !== null) ranges.set(path, child.range);
    collectKeyRanges(child, `${path}.`, ranges);
  }
}

/** Return whether a layer part is, or holds, a value resolved later; a tagged literal is not. */
function holdsValue(tree: unknown): boolean {
  if (isValue(tree)) return true;
  if (isRecord(tree)) return Object.values(tree).some(holdsValue);
  if (Array.isArray(tree)) return tree.some(holdsValue);
  return false;
}

/** Return `tree` with its values in place and its literals as the values they bind. */
function unescapeLiterals(tree: unknown): unknown {
  if (isValue(tree)) return tree;
  if (isRecord(tree)) {
    return Object.fromEntries(Object.entries(tree).map(([key, item]) => [key, unescapeLiterals(item)]));
  }
  if (Array.isArray(tree)) return tree.map(unescapeLiterals);
  return literalPart(tree);
}

/**
 * Return the range of the deepest written mapping key on a block problem's key path.
 *
 * @param keys The block's key, then the problem's key path in it.
 * @param fieldRanges Value range of each key written in the layer, by dotted path; list items
 *   have none.
 * @returns The range, or `null` when no key on the path has one.
 */
function blockProblemRange(
  keys: readonly (string | number)[],
  fieldRanges: ReadonlyMap<string, SourceRange>,
): SourceRange | null {
  for (let end = keys.length; end > 0; end--) {
    const head = keys.slice(0, end);
    if (head.every((part) => typeof part === "string")) {
      const found = fieldRanges.get(head.join("."));
      if (found !== undefined) return found;
    }
  }
  return null;
}

/** Return the value a literal template part binds: tags dropped, strings `$$`-unescaped. */
function literalPart(tree: unknown): unknown {
  if (tree instanceof Binding) return literalPart(tree.source);
  if (tree instanceof Verbatim) return tree.text;
  if (typeof tree === "string") return evalBinding(tree, MISSING_ROOT);
  if (isRecord(tree)) return Object.fromEntries(Object.entries(tree).map(([key, item]) => [key, literalPart(item)]));
  if (Array.isArray(tree)) return tree.map(literalPart);
  return tree;
}

/** Put `prefix` in place of the leading `llm_config` of a host-only field message. */
function hostOnlyMessage(prefix: string, message: string): string {
  // A message that does not start with `llm_config` is kept whole after the prefix.
  return message.startsWith("llm_config") ? prefix + message.slice("llm_config".length) : `${prefix}: ${message}`;
}

/** Return the field names a record type, or any member of a record union, declares. */
function declaredFields(typ: Type): Set<string> {
  const names = new Set(typ.fields?.keys() ?? []);
  for (const member of typ.members ?? []) {
    for (const name of member.fields?.keys() ?? []) names.add(name);
  }
  return names;
}

/**
 * Check an `llm_config` layer holding values, and split off its literal `inherit`.
 *
 * Fields that only the host sets may not appear, from a value or not, and a whole-record type may
 * not declare them; `base_url` beside a literal `provider: azure` must itself be literal.
 * `inherit` must be a literal boolean. The literal top-level fields are checked as `LLMConfig`
 * fields, the literal parts of a provider block holding values against that block, and a literal
 * provider and model against the block (`checkResponsesBlock`).
 *
 * @param cfg The layer, or the whole-record value.
 * @param prefix Message prefix naming the layer.
 * @param sourceRange The layer's location, when retained.
 * @param diagnostics Factory for located load errors.
 * @param fieldRanges Value range of each key written in the layer, by dotted path.
 * @param options.allowInherit Whether the layer may set `inherit` (an AGENT's own layer; the
 *   default); elsewhere it is an unknown field.
 * @returns The layer without `inherit`, then `inherit`.
 * @throws LoadError located at the offending field when its range is known.
 */
export function checkLlmConfigValue(
  cfg: ValueTemplate | Binding,
  prefix: string,
  sourceRange: SourceRange | null,
  diagnostics: LoadDiagnostics,
  fieldRanges: ReadonlyMap<string, SourceRange>,
  options: { allowInherit?: boolean } = {},
): [ValueTemplate | Binding, boolean] {
  const { allowInherit = true } = options;
  if (cfg instanceof Binding) {
    const message = cfg.contract === null ? null : declaredHostOnlyError(declaredFields(cfg.contract.type));
    if (message !== null) throw diagnostics.error(`${prefix}: ${message}`, { range: sourceRange });
    return [cfg, true];
  }
  let tree = cfg.value;
  if (!isRecord(tree)) throw diagnostics.error(`${prefix} must be a mapping`, { range: sourceRange });
  const at = (path: string) => fieldRanges.get(path) ?? sourceRange;

  let inherit = true;
  if (allowInherit && Object.hasOwn(tree, "inherit")) {
    if (holdsValue(tree["inherit"])) {
      throw diagnostics.error(
        `${prefix} \`inherit:\` must be written as true or false; it decides which layers apply before any ` +
          "value is known",
        { range: at("inherit") },
      );
    }
    const written = literalPart(tree["inherit"]);
    if (typeof written !== "boolean") {
      throw diagnostics.error(
        `${prefix} \`inherit:\` must be true or false, got ${valueRepr(written)}${formerBooleanHint(written)}`,
        { range: at("inherit") },
      );
    }
    inherit = written;
    const { inherit: _inherit, ...rest } = tree;
    tree = rest;
    cfg = new ValueTemplate(rest, cfg.origins.filter(([path]) => path[0] !== "inherit"));
  }
  // Each literal field as the value it binds, so a tagged `provider: !str azure` counts as
  // `azure`; a field holding a value is null here.
  const literal = Object.fromEntries(
    Object.entries(tree).map(([key, item]) => [key, holdsValue(item) ? null : literalPart(item)]),
  );
  const hostOnly = hostOnlyFieldError(literal);
  if (hostOnly !== null) {
    const [field, message] = hostOnly;
    throw diagnostics.error(hostOnlyMessage(prefix, message), { range: at(field) });
  }
  if (Object.hasOwn(tree, "base_url") && holdsValue(tree["base_url"])) {
    throw diagnostics.error(`${prefix}.${BASE_URL_FROM_VALUE}`, { range: at("base_url") });
  }
  checkLiteralFields(literal, prefix, sourceRange, diagnostics, at);
  for (const [key, block] of Object.entries(tree)) {
    if (!Object.hasOwn(PROVIDER_BLOCKS, key) || !holdsValue(block)) continue;
    const problems = locatedBlockProblems(key, unescapeLiterals(block), isValue);
    if (problems.length > 0) {
      throw diagnostics.error(`${prefix}: ${problems[0]!.message}`, {
        range: blockProblemRange([key, ...problems[0]!.keys], fieldRanges) ?? sourceRange,
      });
    }
  }
  // An `api` from a value is known only when it resolves, as is the route it picks.
  if (!(Object.hasOwn(tree, "api") && holdsValue(tree["api"]))) {
    checkResponsesBlock(literal["provider"], literal["model"], literal["api"], tree, prefix, diagnostics, at);
  }
  return [cfg, inherit];
}

/**
 * Refuse a layer whose literal `provider`, `model` and `api` send the call to the Responses API
 * while it writes that provider's block (`responsesBlockError`); a provider or model from a value
 * is checked when it resolves.
 *
 * @param api The literal `api`, which `LLMConfig` checked.
 * @param tree The layer as written; the block may hold values.
 * @param at The location of a top-level field by its key.
 * @throws LoadError at the block.
 */
function checkResponsesBlock(
  provider: unknown,
  model: unknown,
  api: unknown,
  tree: Readonly<Record<string, unknown>>,
  prefix: string,
  diagnostics: LoadDiagnostics,
  at: (key: string) => SourceRange | null,
): void {
  if (typeof provider !== "string" || typeof model !== "string") return;
  const block = Object.hasOwn(tree, provider) ? tree[provider] : null;
  const message = responsesBlockError(provider, model, api === "responses" || api === "chat" ? api : null, block);
  if (message !== null) throw diagnostics.error(`${prefix}: ${message}`, { range: at(provider) });
}

/**
 * Check literal `llm_config` fields as `LLMConfig` fields.
 *
 * @param cfg The literal fields; a field holding a value is `null`.
 * @param prefix Message prefix naming the layer.
 * @param sourceRange The layer's location, when retained.
 * @param fieldRange The location of a top-level field by its key.
 * @throws LoadError when a field violates `LLMConfig`: a text field YAML read as non-text is named
 *   by `notTextError`, any other violation keeps the `LLMConfig` message.
 */
function checkLiteralFields(
  cfg: Readonly<Record<string, unknown>>,
  prefix: string,
  sourceRange: SourceRange | null,
  diagnostics: LoadDiagnostics,
  fieldRange: (key: string) => SourceRange | null,
): void {
  try {
    new LLMConfig(cfg);
  } catch (error) {
    if (!(error instanceof ValidationError)) throw error;
    const detail = notTextDetail(error, cfg);
    const key = error.issues[0]?.loc[0];
    const range = typeof key === "string" ? fieldRange(key) : sourceRange;
    throw diagnostics.error(`${prefix}: ` + (detail !== null ? notTextError(...detail) : error.message), {
      range,
      cause: error,
    });
  }
}

/**
 * Validate an authored `llm_config:` layer at load and split off `inherit`.
 *
 * `inherit` (the whole-node cascade opt-out, default `true`) is not an `LLMConfig` field, so it is
 * removed before the check that refuses unknown keys (such as `temparature`). A layer holding
 * `${...}` values (a `ValueTemplate`, or a `Binding` for the whole record) is checked by
 * `checkLlmConfigValue`; its values are checked when they resolve.
 *
 * @param cfg The authored layer, or `null` for defaults.
 * @param prefix Message prefix naming the layer, such as `node 'a': llm_config`.
 * @param sourceRange The layer's location, when retained.
 * @param diagnostics Factory for located load errors.
 * @param fieldRanges The value range of each key written in the layer, by dotted path.
 * @param options.allowInherit Whether the layer may set `inherit` (an AGENT's own layer; the
 *   default); elsewhere it is an unknown field.
 * @returns The layer without `inherit`, then `inherit`.
 * @throws LoadError, located at the offending field when its range is known, when the layer is not
 *   a mapping, sets a non-boolean `inherit`, sets a host-only field (`api_key`, `organization`,
 *   `project`, `headers`, or `base_url` other than beside `provider: azure`), violates
 *   `LLMConfig`, or writes a block its model can't take (`checkResponsesBlock`).
 */
export function validateLlmConfig(
  cfg: unknown,
  prefix: string,
  sourceRange: SourceRange | null,
  diagnostics: LoadDiagnostics,
  fieldRanges: ReadonlyMap<string, SourceRange>,
  options: { allowInherit?: boolean } = {},
): [Readonly<Record<string, unknown>> | ValueTemplate | Binding, boolean] {
  const { allowInherit = true } = options;
  if (cfg === null) return [{}, true];
  if (cfg instanceof ValueTemplate || cfg instanceof Binding) {
    return checkLlmConfigValue(cfg, prefix, sourceRange, diagnostics, fieldRanges, { allowInherit });
  }
  if (!isRecord(cfg)) throw diagnostics.error(`${prefix} must be a mapping`, { range: sourceRange });
  const at = (path: string) => fieldRanges.get(path) ?? sourceRange;
  let rest: Readonly<Record<string, unknown>> = cfg;
  let inherit: unknown = true;
  if (allowInherit) ({ inherit = true, ...rest } = cfg);
  if (typeof inherit !== "boolean") {
    throw diagnostics.error(
      `${prefix} \`inherit:\` must be true or false, got ${valueRepr(inherit)}${formerBooleanHint(inherit)}`,
      { range: at("inherit") },
    );
  }
  const hostOnly = hostOnlyFieldError(rest);
  if (hostOnly !== null) throw diagnostics.error(hostOnlyMessage(prefix, hostOnly[1]), { range: at(hostOnly[0]) });
  checkLiteralFields(rest, prefix, sourceRange, diagnostics, at);
  checkResponsesBlock(rest["provider"], rest["model"], rest["api"], rest, prefix, diagnostics, at);
  return [rest, inherit];
}

/**
 * Return why an `env:` mapping's `max_tool_iterations` isn't one an AGENT accepts, or `null` when
 * it is valid or absent.
 *
 * Example: `{max_tool_iterations: 0}` gives ``"`env.max_tool_iterations` must be a positive int or
 * -1 (no cap), got 0"``.
 */
export function maxToolIterationsError(env: Readonly<Record<string, unknown>>): string | null {
  if (!Object.hasOwn(env, "max_tool_iterations")) return null;
  const value = env["max_tool_iterations"];
  if (isMaxToolIterations(value)) return null;
  return `\`env.max_tool_iterations\` must be a positive int or -1 (no cap), got ${valueRepr(value)}`;
}

/** The roles a prompt list message may take; `system` is `system_prompt:`. */
const PROMPT_ROLES: ReadonlySet<string> = new Set(["user", "assistant"]);

/**
 * Return the values of an agent body's prompt text: a text `prompt:`, each `user:` or
 * `assistant:` message of a list `prompt:`, and `system_prompt:`. Each may be text
 * `!include(str)` read, which is sent as written.
 */
function promptTextCursors(cursor: SourceCursor): SourceCursor[] {
  const found: SourceCursor[] = [];
  for (const key of ["prompt", "system_prompt"]) {
    if (!cursor.has(key)) continue;
    const field = cursor.child(key);
    const items = field._valueView();
    if (key !== "prompt" || !isList(items) || field.typedInclude !== null) {
      found.push(field);
      continue;
    }
    items.forEach((item, index) => {
      if (!isRecord(item)) return;
      for (const role of PROMPT_ROLES) if (Object.hasOwn(item, role)) found.push(field.child(index).child(role));
    });
  }
  return found;
}

/**
 * Refuse a value `!include(T)` read as YAML, for a `T` that isn't text, as prompt text.
 *
 * @throws LoadError at the first one.
 */
function rejectTypedPromptText(cursor: SourceCursor, diagnostics: LoadDiagnostics): void {
  for (const field of promptTextCursors(cursor)) {
    const include = field.typedInclude;
    if (include === null || include.data === null) continue;
    throw diagnostics.error(
      `${includeFile(include.path, include.type)}: \`${field.keyText}:\` takes text; read the file with ` +
        "`!include(str)`",
      { range: field.range },
    );
  }
}

/** Return `text` as `Verbatim` text when `!include(str)` read it, at `cursor`, else as a template. */
function promptText(text: string, cursor: SourceCursor | null): PromptText {
  return cursor?.verbatim === true ? new Verbatim(text) : text;
}

/**
 * Read an authored list `prompt:` into its items, and range each at
 * `siteKey("field", "prompt[<i>]")` in `desc.fieldRanges`.
 *
 * An item is a lone `${name}` (a splice) or a one-key `user:`/`assistant:` mapping whose value is
 * template text.
 *
 * @throws LoadError at the item for text that isn't a lone `${name}`, any other shape, a
 *   `system:` item, non-text message text or text holding a lone surrogate, or a last item that
 *   is an `assistant:` message; at the field for an empty list.
 */
function readPromptList(
  nodeId: string,
  desc: AgentDescriptor,
  cursor: SourceCursor,
  diagnostics: LoadDiagnostics,
): PromptItem[] {
  const where = `node ${reprText(nodeId)} (kind=agent)`;
  const list = desc.prompt as readonly unknown[];
  const fieldRange = desc.fieldRanges.get(siteKey("field", "prompt")) ?? desc.range;
  if (list.length === 0) {
    throw diagnostics.error(`${where}: \`prompt:\` must hold at least one message`, { range: fieldRange });
  }
  const items = list.map((item, index): PromptItem => {
    const range = rangeAt(cursor, ["prompt", index]) ?? fieldRange;
    if (range !== null) desc.fieldRanges.set(siteKey("field", `prompt[${index}]`), range);
    if (typeof item === "string") {
      // Quoting the item would quote the whole file.
      if (cursorAt(cursor, ["prompt", index])?.verbatim === true) {
        throw diagnostics.error(
          `${where}: a prompt list item is \`\${name}\` or \`user:\`/\`assistant:\` text; put the ` +
            "`!include(str)` under `user:`",
          { range },
        );
      }
      const name = wholeRefName(item);
      if (name !== null) return { splice: name };
      throw diagnostics.error(
        `${where}: prompt list item \`${item}\` isn't a splice: a splice is a lone \`\${name}\` naming one ` +
          "of this node's inputs or variables; write a message as `user: <text>`",
        { range },
      );
    }
    const keys = isRecord(item) ? Object.keys(item) : [];
    const role = keys.length === 1 ? keys[0]! : undefined;
    if (role === "system") {
      throw diagnostics.error(`${where}: a prompt list item can't be \`system:\`; use \`system_prompt:\``, { range });
    }
    if (role === undefined || !PROMPT_ROLES.has(role)) {
      throw diagnostics.error(`${where}: a prompt list item is \`\${name}\` or \`user:\`/\`assistant:\` text`, {
        range,
      });
    }
    const text = (item as Record<string, unknown>)[role];
    const textRange = rangeAt(cursor, ["prompt", index, role]) ?? range;
    if (typeof text !== "string") {
      throw diagnostics.error(`${where}: ${notTextError(`prompt[${index}].${role}`, text)}`, { range: textRange });
    }
    checkWrittenValue(text, `node ${reprText(nodeId)} field 'prompt[${index}].${role}'`, textRange, diagnostics);
    return { role: role as "user" | "assistant", text: promptText(text, cursorAt(cursor, ["prompt", index, role])) };
  });
  const last = items.at(-1)!;
  if ("role" in last && last.role === "assistant") {
    throw diagnostics.error(`${where}: the last message of \`prompt:\` must be from the user`, {
      range: desc.fieldRanges.get(siteKey("field", `prompt[${items.length - 1}]`)) ?? fieldRange,
    });
  }
  return items;
}

/** The fields `AgentDescriptor`'s constructor takes: the common ones and the AGENT's own. */
export interface AgentDescriptorFields extends NodeDescriptorFields {
  prompt?: PromptText | readonly PromptItem[] | null;
  systemPrompt?: PromptText | null;
  variables?: unknown;
  tools?: readonly ToolGrant[];
  skills?: readonly string[] | "all" | null;
  llmConfig?: unknown;
  retries?: number;
  asserts?: readonly string[] | null;
  steerInbox?: string | null;
  steerRender?: string | null;
  outputContractOverride?: ValueContract | null;
}

/**
 * kind=agent: an LLM reasoner (a prompt, an optional system prompt, optional tools, skills and
 * model config).
 *
 * `allowed` adds `inputs`/`outputs` and the AGENT's own keys to the commons; nothing is required.
 * Only `inputs` is a binding field: the `${...}` references of `prompt:` and `system_prompt:` are
 * read by template scanning, not the inline-call desugar.
 */
export class AgentDescriptor extends NodeDescriptor {
  static override readonly kind = "agent";
  static override readonly allowed = this.allow(
    "inputs",
    "outputs",
    "prompt",
    "system_prompt",
    "variables",
    "tools",
    "skills",
    "llm_config",
    "retries",
    "asserts",
    "steer_inbox",
  );
  static override readonly required = this.require();
  static override readonly bindingFields: readonly string[] = ["inputs"];
  static override readonly writtenFields = [
    ...NodeDescriptor.writtenFields,
    ["steer_inbox.render", "a template over each steer event"],
    ["steer_inbox", "an inbox reference"],
  ] as const;

  /** The prompt template, `Verbatim` text or list prompt; `null` when absent. */
  readonly prompt: PromptText | readonly PromptItem[] | null;
  /** The system prompt template or `Verbatim` text; `null` when absent. */
  readonly systemPrompt: PromptText | null;
  /** The authored `variables:` as YAML read it; `build` reads the declarations. */
  readonly variables: unknown;
  /** The authored tool grants. */
  readonly tools: readonly ToolGrant[];
  /** The granted skill names, or `"all"`. */
  readonly skills: readonly string[] | "all";
  /** The authored `llm_config:` as YAML read it; `build` reads the values in it. */
  readonly llmConfig: unknown;
  /** The effective retry count, so the manifest holds it and a default flow keeps its identity. */
  readonly retries: number;
  /** The node asserts. */
  readonly asserts: readonly string[];
  /** The authored `steer_inbox:` source (`${input.<name>}`, or the mapping's `from:`); `null` when absent. */
  readonly steerInbox: string | null;
  /** The `steer_inbox:` mapping's `render:` template over `${event}`; `null` when absent. */
  readonly steerRender: string | null;
  /**
   * Never parsed from YAML: set only by the `adaptive_questions` desugar, for a synthesized AGENT
   * whose output type is built in code and has no type text; it overrides the declared `output:`.
   */
  readonly outputContractOverride: ValueContract | null;

  /**
   * Create a descriptor from `fields`. `retries` defaults to `current().agentRetries`;
   * `variables` and `llmConfig` to `{}` (a `null` is kept as written for `build`, which refuses a
   * null `variables:` and reads a null `llm_config:` as the defaults); `skills` and `asserts` read
   * `null` as empty.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: AgentDescriptorFields) {
    super(fields);
    this.prompt = fields.prompt ?? null;
    this.systemPrompt = fields.systemPrompt ?? null;
    this.variables = fields.variables === undefined ? {} : fields.variables;
    this.tools = fields.tools ?? [];
    this.skills = fields.skills ?? [];
    this.llmConfig = fields.llmConfig === undefined ? {} : fields.llmConfig;
    this.retries = fields.retries ?? current().agentRetries;
    this.asserts = fields.asserts ?? [];
    this.steerInbox = fields.steerInbox ?? null;
    this.steerRender = fields.steerRender ?? null;
    this.outputContractOverride = fields.outputContractOverride ?? null;
  }

  /**
   * Reject the removed `mode:` and memory keys (`rejectRemovedMemoryKey`), parse as usual, then
   * reject a variable named `inf`, since `${inf}` reads infinity, and read a list `prompt:` into
   * its items (`readPromptList`).
   *
   * Text `!include(str)` read as a prompt, a system prompt or a list prompt's message is
   * `Verbatim`, sent as written.
   *
   * @throws LoadError when the node authors `mode:`, `long_term_memory:` or `memory_policy:` (at
   *   its key) or a variable named `inf` (at its key), a list `prompt:` is malformed, prompt text
   *   is a value `!include(T)` read for a `T` other than `str`, or the template `parse` rejects
   *   it.
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    if (cursor.has("mode")) {
      let message = `node ${reprText(nodeId)}: \`mode:\` was removed: ${MODE_REMOVED_FIX}`;
      const tools = cursor.get("tools");
      if (cursor.get("mode") === "plain" && isList(tools) && tools.length > 0) {
        message += " (its `tools:` will now be used)";
      }
      throw diagnostics.error(message, { range: cursor.keyRangeOf("mode") });
    }
    rejectRemovedMemoryKey((key) => cursor.has(key), cursor, diagnostics, `node ${reprText(nodeId)}: `);
    rejectTypedPromptText(cursor, diagnostics);
    let desc = super.parse(nodeId, cursor) as AgentDescriptor;
    // Text `!include(str)` read is sent as written.
    const { prompt, systemPrompt } = desc;
    if (typeof prompt === "string" || typeof systemPrompt === "string") {
      desc = desc.replace({
        prompt: typeof prompt === "string" ? promptText(prompt, cursorAt(cursor, ["prompt"])) : prompt,
        systemPrompt:
          typeof systemPrompt === "string" ? promptText(systemPrompt, cursorAt(cursor, ["system_prompt"])) : systemPrompt,
      });
    }
    // A `steer_inbox:` mapping's errors and reads point at its `from:` and `render:` values.
    if (isRecord(cursor.get("steer_inbox"))) {
      for (const [key, site] of [
        ["from", siteKey("field", "steer_inbox")],
        ["render", siteKey("field", "steer_inbox.render")],
      ] as const) {
        const range = rangeAt(cursor, ["steer_inbox", key]);
        if (range !== null) desc.fieldRanges.set(site, range);
      }
    }
    if (isRecord(desc.variables)) {
      for (const name of Object.keys(desc.variables)) {
        rejectInfinityName(
          `node ${reprText(nodeId)} variable ${reprText(name)}`,
          name,
          cursor.child("variables").keyRangeOf(name) ?? cursor.rangeOf("variables") ?? cursor.range,
          diagnostics,
        );
      }
    }
    if (!isList(desc.prompt)) return desc as InstanceType<T>;
    return desc.replace({ prompt: readPromptList(nodeId, desc, cursor, diagnostics) }) as InstanceType<T>;
  }

  /**
   * Reject the retired `steerable:` key, at its key, and a `steer_inbox:` mapping with a key other
   * than `from:` and `render:`, no `from:`, or a `render:` that isn't non-empty text, before the
   * generic key check.
   */
  static override _normalizeCursor(cursor: SourceCursor): SourceCursor {
    const steerInbox = cursor.has("steer_inbox") ? cursor.get("steer_inbox") : null;
    if (isRecord(steerInbox)) {
      const diagnostics = LoadDiagnostics.fromCursor(cursor);
      const unknown = Object.keys(steerInbox).find((key) => key !== "from" && key !== "render");
      if (unknown !== undefined) {
        throw diagnostics.error("`steer_inbox:` takes only `from:` and `render:`", {
          range: rangeAt(cursor, ["steer_inbox", unknown]) ?? cursor.keyRangeOf("steer_inbox"),
        });
      }
      if (!Object.hasOwn(steerInbox, "from")) {
        throw diagnostics.error("`steer_inbox:` needs `from:` naming an Inbox input", {
          range: cursor.keyRangeOf("steer_inbox"),
        });
      }
      const render = Object.hasOwn(steerInbox, "render") ? steerInbox["render"] : null;
      if (render !== null && (typeof render !== "string" || render === "")) {
        throw diagnostics.error("`steer_inbox.render:` must be non-empty text", {
          range: rangeAt(cursor, ["steer_inbox", "render"]) ?? cursor.keyRangeOf("steer_inbox"),
        });
      }
    }
    if (cursor.has("steerable")) {
      throw LoadDiagnostics.fromCursor(cursor).error(
        "node field `steerable:` was replaced by `steer_inbox:`: declare an `Inbox(...)` input and write " +
          "`steer_inbox: ${input.<name>}`; the host sends steers with `send`",
        { range: cursor.keyRangeOf("steerable") },
      );
    }
    return cursor;
  }

  /**
   * Check the `prompt:`, `system_prompt:`, `tools:`, `retries:`, `skills:` and `steer_inbox:`
   * shapes, store `retries` as a `number`, split a `steer_inbox:` mapping into its source and
   * `steerRender`.
   *
   * @throws LoadError at the field for a bad shape or a lone surrogate in a prompt.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
    cursor: SourceCursor,
  ): void {
    const id = reprText(nodeId);
    const rangeOf = (key: string) => ranges.get(siteKey("field", key)) ?? ownRange;
    const prompt = fields["prompt"] ?? null;
    // `parse` reads a list prompt's items, where each item's range is known.
    if (!isList(prompt)) {
      rejectNonText(this.kind, nodeId, "prompt", prompt, rangeOf("prompt"), diagnostics);
      if (prompt !== null) checkWrittenValue(prompt, `node ${id} field 'prompt'`, rangeOf("prompt"), diagnostics);
    }
    if (Object.hasOwn(fields, "systemPrompt")) {
      const systemPrompt = fields["systemPrompt"];
      if (typeof systemPrompt !== "string" || !systemPrompt) {
        throw diagnostics.error(`node ${id} (kind=${this.kind}): \`system_prompt:\` must be a non-empty string`, {
          range: rangeOf("system_prompt"),
        });
      }
      checkWrittenValue(systemPrompt, `node ${id} field 'system_prompt'`, rangeOf("system_prompt"), diagnostics);
    }
    if (Object.hasOwn(fields, "tools")) {
      const tools = fields["tools"];
      if (!isList(tools)) {
        const example = typeof tools === "string" && tools ? tools : "search";
        throw diagnostics.error(
          `node ${id} (kind=${this.kind}): \`tools:\` must be a list of tool or group grants; ` +
            `write \`tools: [${example}]\``,
          { range: rangeOf("tools") },
        );
      }
      try {
        fields["tools"] = parseToolGrants(tools, `node ${id} (kind=${this.kind})`);
      } catch (error) {
        if (!(error instanceof GrantError)) throw error;
        const at = error.index === null ? null : rangeAt(cursor, ["tools", error.index]);
        throw diagnostics.error(error.message, { range: at ?? rangeOf("tools") });
      }
    }
    if (Object.hasOwn(fields, "retries")) {
      const problem = retriesError(fields["retries"]);
      if (problem !== null) {
        throw diagnostics.error(`node ${id} (kind=${this.kind}): \`retries:\` ${problem}`, {
          range: rangeOf("retries"),
        });
      }
      fields["retries"] = Number(fields["retries"]);
    }
    const skills = fields["skills"] ?? null;
    if (
      skills !== null &&
      skills !== "all" &&
      !(isList(skills) && skills.every((name) => typeof name === "string"))
    ) {
      throw diagnostics.error(
        `node ${id} (kind=${this.kind}): \`skills:\` must be a list of skill names or \`all\``,
        { range: rangeOf("skills") },
      );
    }
    if (isRecord(fields["steerInbox"])) {
      // `_normalizeCursor` checked the mapping's keys; it splits into the source and the render.
      const { from, render = null } = fields["steerInbox"];
      fields["steerInbox"] = from;
      fields["steerRender"] = render;
    }
    if (Object.hasOwn(fields, "steerInbox")) {
      const steerInbox = fields["steerInbox"];
      if (typeof steerInbox !== "string" || steerInbox === "") {
        throw diagnostics.error("`steer_inbox:` must be `${input.<name>}` naming an Inbox input", {
          range: rangeOf("steer_inbox"),
        });
      }
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics, cursor);
  }

  /**
   * Build the runtime `AgentNode` and its flow-owned wiring.
   *
   * Reads the `llm_config:` layer and the local variables, and sets the node's `params` from the
   * authored `input:` and its output contract from the override or the authored `output:`.
   * `_finish` stamps the descriptor origin; the result also ranges `__llm_config` at the authored
   * `llm_config:`, where errors about its values point.
   *
   * @throws LoadError for a bad `env.max_tool_iterations`, a bad variable declaration, a tool or
   *   skill grant the node refuses, an `output:` nesting deeper than
   *   `current().maxAgentOutputNesting` or holding a type a model can't produce, a variable named
   *   like an input, or a memory grant naming a tool its type lacks or showing a tool under a
   *   reserved name.
   * @throws TypeError when `ctx.variableDeclarationReader` returns anything but an array of exact
   *   `Parameter`s or `null`.
   */
  override build(ctx: BuildContext): BuildResult {
    const id = reprText(this.id);
    const llmConfigRange = this.fieldRanges.get(siteKey("field", "llm_config")) ?? this.range;
    const llmFieldRanges = new Map<string, SourceRange>();
    const llmPrefix = siteKey("llm_config", "");
    for (const [key, range] of this.fieldRanges) {
      if (key.startsWith(llmPrefix)) llmFieldRanges.set(key.slice(llmPrefix.length), range);
    }
    const [ownCfg, inherit] = validateLlmConfig(
      this.llmConfig,
      this.fieldLabels.get("llm_config") || `node ${id}: llm_config`,
      llmConfigRange,
      ctx.diagnostics,
      llmFieldRanges,
    );
    const steerInboxRange = this.fieldRanges.get(siteKey("field", "steer_inbox")) ?? this.range;
    const maxToolIterations = maxToolIterationsError(this.env);
    if (maxToolIterations !== null) {
      throw ctx.diagnostics.error(`node ${id}: ${maxToolIterations}`, {
        range: this.fieldRanges.get(siteKey("field", "env")) ?? this.range,
      });
    }
    const variables = this.readVariables(ctx);
    // A grant naming a memory input of this node's own flow is a memory grant.
    const memoryInputs = [
      ...new Set(
        this.tools
          .map((grant) => (grant.kind === "name" ? grant.name : grant.group))
          .filter((name) => isMemoryType(ctx.flowInputs.get(name))),
      ),
    ];
    const toolsRange = this.fieldRanges.get(siteKey("field", "tools")) ?? this.range;
    try {
      checkMemoryGrants(
        this.tools.filter((grant) => memoryInputs.includes(grant.kind === "name" ? grant.name : grant.group)),
        ctx.flowInputs,
        ctx.memoryTypes,
      );
    } catch (error) {
      if (!(error instanceof GrantError)) throw error;
      throw ctx.diagnostics.error(grantErrorText(`node ${id} (kind=${this.kind})`, error), {
        range: toolsRange,
        cause: error,
      });
    }
    let node: AgentNode;
    try {
      node = new AgentNode(this.id, {
        prompt: this.prompt ?? "",
        systemPrompt: this.systemPrompt,
        tools: [...this.tools],
        llmConfig: ownCfg,
        llmInherit: inherit,
        retries: this.retries,
        title: this.nodeName,
        variables,
        skills: this.skills,
        steerInbox: this.steerInbox !== null,
        steerRender: this.steerRender,
        memoryInputs,
      });
    } catch (error) {
      // AgentNode checks skill grants and tool names; a refusal is the loader's one error type.
      if (!(error instanceof RangeError)) throw error;
      const skillsRange = this.fieldRanges.get(siteKey("field", "skills")) ?? this.range;
      throw ctx.diagnostics.error(`node ${id}: ${error.message}`, {
        range: error instanceof SkillError ? skillsRange : toolsRange,
        cause: error,
      });
    }
    node.llmConfigLabel = this.fieldLabels.get("llm_config") ?? null;

    const override = this.outputContractOverride;
    node.outputContract = override ?? this.outputContract(ctx);
    if (isExactly(node.outputContract, ValueContract)) {
      const outputsRange = this.fieldRanges.get(siteKey("field", "outputs")) ?? this.range;
      // Schema generation, here and at run, recurses per level; deeper types aren't proven to
      // fit the stack. An engine-built output type is not the author's.
      const levels = typeLevels(node.outputContract.type);
      const limit = current().maxAgentOutputNesting;
      if (override === null && levels > limit) {
        throw ctx.diagnostics.error(
          `node ${id}: this AGENT's \`output:\` nests ${levels} levels, deeper than ${limit}, the limit ` +
            "for a model's structured reply (each record and list is a level); flatten it, or build the " +
            "deeper value in a CODE node",
          { range: outputsRange },
        );
      }
      try {
        typeToSchema(node.outputContract.type);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw ctx.diagnostics.error(
          `node ${id}: an AGENT can't produce this \`output:\`; a model generates only str, int, float, ` +
            "bool, date, datetime, Literal, record, union, object, Any, and list values, not `file` or " +
            "`None`. Produce the value in a CODE node.",
          { range: outputsRange, cause: error },
        );
      }
    }
    node.params = this.sinkParams();
    const inputNames = new Set(node.params.map((parameter) => parameter.name));
    // The reserved keys the run binds beside the inputs start with `__`, which no input or
    // variable may, so only an input's name can collide with a variable's.
    for (const variable of variables) {
      if (!inputNames.has(variable.name)) continue;
      const message = `node ${id}: variable ${reprText(variable.name)} collides with an input name`;
      const frame =
        ctx.resolveOrigin !== null && variable.origin !== null ? ctx.resolveOrigin(variable.origin) : null;
      if (frame !== null) throw ctx.diagnostics.error(message, { frames: [frame] });
      throw ctx.diagnostics.error(message, {
        range: this.fieldRanges.get(siteKey("field", "variables")) ?? this.range,
      });
    }
    const wiring = this.sinkWiring();
    if (this.steerInbox !== null) wiring.set(STEER_INBOX_KEY, this.steerInbox);
    for (const input of memoryInputs) wiring.set(memoryKey(input), memoryBinding(input));
    const result = this._finish(node, wiring);
    // Errors about `llm_config` values point at the authored `llm_config:`, and errors about the
    // steer inbox at `steer_inbox:`.
    const fieldRanges = new Map(result.fieldRanges);
    if (llmConfigRange !== null) fieldRanges.set(siteKey("field", LLM_CONFIG_KEY), llmConfigRange);
    if (this.steerInbox !== null && steerInboxRange !== null) {
      fieldRanges.set(siteKey("field", STEER_INBOX_KEY), steerInboxRange);
    }
    return new BuildResult({ ...result, fieldRanges });
  }

  /**
   * Compile this node's variable declarations at their authored site, through
   * `ctx.variableDeclarationReader`, or source-free when it declines or is unset.
   *
   * @throws LoadError for a bad declaration, ranged at `variables:` when the error has no frame.
   * @throws TypeError when the reader returns anything but an array of exact `Parameter`s or
   *   `null`.
   */
  private readVariables(ctx: BuildContext): Parameter[] {
    try {
      if (ctx.variableDeclarationReader !== null) {
        const variables: unknown = ctx.variableDeclarationReader(this.id, this.variables);
        if (variables !== null) {
          if (!isList(variables) || !variables.every((variable) => isExactly(variable, Parameter))) {
            throw new TypeError("variable declaration reader must return a list of exact Parameters or null");
          }
          return variables;
        }
      }
      // Only a node with no authored source gets here; a variable needs a default, which only a
      // tagged YAML declaration carries, so any declaration fails.
      return readVariables(this.variables, ctx.registry, { diagnostics: ctx.diagnostics });
    } catch (error) {
      if (!(error instanceof LoadError) || error.diagnostic.frames.length > 0) throw error;
      throw ctx.diagnostics.prepend(error, {
        range: this.fieldRanges.get(siteKey("field", "variables")) ?? this.range,
        cause: error,
      });
    }
  }
}
