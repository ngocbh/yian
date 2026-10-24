/**
 * kind=tool: the deterministic-tool node's descriptor.
 *
 * Owns the `tool` node's `ToolDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`), checks its own fields, and builds
 * the runtime `ToolNode`. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 * Also owns the question helpers an asking node shares: the answer schema derived from an
 * `output:` type, the `adaptive_questions:` keys, and the non-text question check.
 *
 * Knows about: diagnostics, compose source and helpers, the AGENT reader (`retriesError`), nodes
 *              (tool), typesys and defaults.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { typeRelation } from "../../compile/validation.ts";
import { current } from "../../defaults.ts";
import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { bindingSource, isExactly, ValueContract } from "../../nodes/contracts.ts";
import { ToolNode } from "../../nodes/tool/node.ts";
import { LOAD_SKILL } from "../../skills/tool.ts";
import { ASK, BUILTIN_TOOL_IDS, type BuiltinToolId, SLEEP, TAKE } from "../../tools/context.ts";
import { parseDateOrDatetime, parseDuration } from "../../typesys/temporal.ts";
import {
  canonicalTypeLabel,
  isInboxType,
  isList,
  isRecord,
  listKindForElement,
  Type,
  ValueKind,
  valueRepr,
} from "../../typesys/values.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { literalValue, NOT_LITERAL } from "../helpers/output_contracts.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
import { retriesError } from "./agent.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  rejectNonText,
  siteKey,
} from "./base.ts";

/** The fields `ToolDescriptor`'s constructor takes: the common ones and the TOOL's own. */
export interface ToolDescriptorFields extends NodeDescriptorFields {
  toolId: string;
  retries?: number;
  asserts?: readonly string[] | null;
}

/**
 * kind=tool: a direct, deterministic call to one tool in the run's `ToolSet`.
 *
 * `allowed` adds `tool_id`, `inputs`/`outputs`, `retries` and `asserts` to the commons; `tool_id`
 * is required. A tool's authored `input:` is its sink-binding surface, so `inputs` is its one
 * binding field. `output:` is optional; when present it is the node's output contract, which the
 * engine checks the tool's return value against on every call.
 */
export class ToolDescriptor extends NodeDescriptor {
  static override readonly kind = "tool";
  static override readonly allowed = this.allow("tool_id", "inputs", "outputs", "retries", "asserts");
  static override readonly required = this.require("tool_id");
  static override readonly bindingFields: readonly string[] = ["inputs"];

  /** The tool id, looked up in the run's `ToolSet`. */
  readonly toolId: string;
  /** The effective retry count, so the manifest holds it; zero is left out of the manifest. */
  readonly retries: number;
  /** The node asserts. */
  readonly asserts: readonly string[];

  /**
   * Create a descriptor from `fields`. `retries` defaults to `current().toolRetries`; `asserts`
   * reads `null` as empty.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: ToolDescriptorFields) {
    super(fields);
    this.toolId = fields.toolId;
    this.retries = fields.retries ?? current().toolRetries;
    this.asserts = fields.asserts ?? [];
  }

  /**
   * Reject a non-text `tool_id:` (`null` included), the reserved `load_skill`, the fields a
   * built-in refuses, and a bad `retries:`; store `retries` as a `number`, then run the base checks.
   *
   * `take` and `sleep` refuse `retries:`, `asserts:` and `output:` (they make no tool call, and
   * their output is fixed); `ask` refuses `retries:`. `tool_id: ask_user` is a plain tool call.
   *
   * @throws LoadError at the field for a bad shape or a refused field.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const rangeOf = (key: string) => ranges.get(siteKey("field", key)) ?? ownRange;
    rejectNonText(this.kind, nodeId, "tool_id", fields["toolId"] ?? null, rangeOf("tool_id"), diagnostics, {
      required: true,
    });
    const toolId = fields["toolId"] as string;
    if (toolId === LOAD_SKILL) {
      throw diagnostics.error(`node ${reprText(nodeId)} (kind=${this.kind}): tool id \`${LOAD_SKILL}\` is reserved by yian`, {
        range: rangeOf("tool_id"),
      });
    }
    if (BUILTIN_TOOL_IDS.has(toolId as BuiltinToolId)) {
      const refuse = (key: string, problem: string) => {
        if (!Object.hasOwn(fields, key)) return;
        throw diagnostics.error(`node ${reprText(nodeId)} (tool_id: ${toolId}): ${problem}`, { range: rangeOf(key) });
      };
      if (toolId === ASK) {
        refuse("retries", "`retries:` doesn't apply to ask: the host owns retries of a call it holds");
      } else {
        refuse("retries", `\`retries:\` doesn't apply to ${toolId}: there is no tool call to retry`);
        refuse("asserts", `\`asserts:\` doesn't apply to ${toolId}`);
        const output = toolId === TAKE ? "inferred ({events: list(T), closed: bool})" : "None";
        refuse("outputs", `${toolId}'s output is ${output}; remove \`output:\``);
      }
    }
    if (Object.hasOwn(fields, "retries")) {
      const problem = retriesError(fields["retries"]);
      if (problem !== null) {
        throw diagnostics.error(`node ${reprText(nodeId)} (kind=${this.kind}): \`retries:\` ${problem}`, {
          range: rangeOf("retries"),
        });
      }
      fields["retries"] = Number(fields["retries"]);
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /**
   * Build the runtime `ToolNode` and its flow-owned wiring.
   *
   * Sets the node's `params` from the authored `input:` and its output contract from the
   * optional `output:` (`null` when absent); `_finish` stamps the descriptor origin. A built-in
   * checks its inputs first and infers its output: `take` gives `{events: list(T), closed: bool}`
   * for its `Inbox(T)`, `sleep` gives `None`. A literal duration (`timeout`, `for`) is wired as
   * its seconds.
   *
   * @throws LoadError when `output:` isn't a valid type expression, or a built-in's input is
   *   unknown, missing or of the wrong type (at the input).
   */
  override build(ctx: BuildContext): BuildResult {
    const builtin = BUILTIN_TOOL_IDS.has(this.toolId as BuiltinToolId);
    // A built-in is never retried by the node: take and sleep make no call, and the host owns an ask.
    const retries = builtin ? 0 : this.retries;
    const node = new ToolNode(this.id, { toolId: this.toolId, title: this.nodeName, retries });
    const wiring = this.sinkWiring();
    if (this.toolId === TAKE) {
      node.outputContract = this.buildTake(ctx);
      this.wireDuration(ctx, wiring, "timeout");
    } else if (this.toolId === SLEEP) {
      this.checkSleep(ctx);
      this.wireDuration(ctx, wiring, "for");
      node.outputContract = new ValueContract(Type.scalar(ValueKind.NONE));
    } else {
      node.outputContract = this.outputContract(ctx);
    }
    node.params = this.sinkParams();
    return this._finish(node, wiring);
  }

  /**
   * Check a take's inputs and return its output contract, `{events: list(T), closed: bool}` for
   * the `Inbox(T)` its `from` reads.
   *
   * @throws LoadError for an input other than `from`, `take` and `timeout`, a missing `from`, a
   *   `from` that isn't an inbox, or a `take` other than `one` or `all`.
   */
  private buildTake(ctx: BuildContext): ValueContract {
    const id = reprText(this.id);
    for (const name of Object.keys(this.inputs)) {
      if (TAKE_INPUTS.has(name)) continue;
      throw ctx.diagnostics.error(`node ${id} (tool_id: take): take takes from, take and timeout; got ${reprText(name)}`, {
        range: this.inputRange(name),
      });
    }
    if (!Object.hasOwn(this.inputs, "from")) {
      throw ctx.diagnostics.error(`node ${id} (tool_id: take): take needs from: \${input.<inbox>}`, {
        range: this.fieldRanges.get(siteKey("field", "inputs")) ?? this.range,
      });
    }
    const fromType = this.inputType(ctx, "from");
    if (fromType === null || !isInboxType(fromType)) {
      throw ctx.diagnostics.error(
        `node ${id} (tool_id: take): take's from must be an Inbox input of this flow, as \${input.<name>}`,
        { range: this.inputRange("from") },
      );
    }
    if (Object.hasOwn(this.inputs, "take")) this.checkTakeMode(ctx);
    const element = fromType.element!;
    return new ValueContract(
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([
          ["events", new Type({ kind: listKindForElement(element), element })],
          ["closed", Type.scalar(ValueKind.BOOLEAN)],
        ]),
        required: new Set(["events", "closed"]),
      }),
    );
  }

  /**
   * Check a take's `take:`: a literal `one` or `all`, or a reference whose type may be one.
   *
   * @throws LoadError at the input otherwise.
   */
  private checkTakeMode(ctx: BuildContext): void {
    const subject = this.inputSubject("take");
    const literal = literalValue(bindingSource(this.inputs["take"]));
    if (literal !== NOT_LITERAL) {
      if (literal === "one" || literal === "all") return;
      throw ctx.diagnostics.error(`${subject}: must be one or all, got ${valueRepr(literal)}`, {
        range: this.inputRange("take"),
      });
    }
    const typ = this.inputType(ctx, "take");
    if (typ === null || typeRelation(typ, TAKE_MODE) !== "disjoint") return;
    throw ctx.diagnostics.error(
      `${subject}: must be one or all, but the value is ${canonicalTypeLabel(typ)}`,
      { range: this.inputRange("take") },
    );
  }

  /**
   * Check a sleep's inputs: exactly one of `until` and `for`. A literal `until` must be a date or
   * datetime, and a reference to one must be able to hold a date, datetime or text.
   *
   * @throws LoadError for any other input, both or neither, or a bad `until`.
   */
  private checkSleep(ctx: BuildContext): void {
    const id = reprText(this.id);
    const names = Object.keys(this.inputs);
    const extra = names.find((name) => name !== "until" && name !== "for");
    if (extra !== undefined || names.length !== 1) {
      const got = extra === undefined ? "" : `; got ${reprText(extra)}`;
      throw ctx.diagnostics.error(`node ${id} (tool_id: sleep): sleep takes until: or for:, exactly one${got}`, {
        range: extra === undefined ? (this.fieldRanges.get(siteKey("field", "inputs")) ?? this.range) : this.inputRange(extra),
      });
    }
    if (names[0] !== "until") return;
    const subject = this.inputSubject("until");
    const range = this.inputRange("until");
    const literal = literalValue(bindingSource(this.inputs["until"]));
    if (literal !== NOT_LITERAL) {
      if (typeof literal === "string") {
        try {
          parseDateOrDatetime(literal);
          return;
        } catch (error) {
          if (!(error instanceof RangeError)) throw error;
        }
      }
      throw ctx.diagnostics.error(`${subject}: must be a date or datetime, got ${valueRepr(literal)}; ${UNTIL_FIX}`, {
        range,
      });
    }
    const typ = this.inputType(ctx, "until");
    if (typ === null || typeRelation(typ, DATE) !== "disjoint" || typeRelation(typ, DATETIME) !== "disjoint") return;
    throw ctx.diagnostics.error(
      `${subject}: must be a date or datetime, but the value is ${canonicalTypeLabel(typ)}; ${UNTIL_FIX}`,
      { range },
    );
  }

  /**
   * Rewrite duration input `name`'s wiring source, when present: a literal to its seconds (a
   * bigint); a reference stays as written once its type is `str`, and is parsed when the node runs.
   *
   * @throws LoadError at the input for a malformed duration, a plain number, or a reference not
   *   typed `str` (an unproven one included: tag it `!str`).
   */
  private wireDuration(ctx: BuildContext, wiring: Map<string, unknown>, name: string): void {
    if (!wiring.has(name)) return;
    const subject = this.inputSubject(name);
    const range = this.inputRange(name);
    const literal = literalValue(bindingSource(wiring.get(name)));
    if (literal !== NOT_LITERAL) {
      if (typeof literal !== "string") {
        const hint = typeof literal === "bigint" ? `; write ${literal}s` : "";
        throw ctx.diagnostics.error(`${subject}: a duration is text such as 24h, got ${valueRepr(literal)}${hint}`, {
          range,
        });
      }
      try {
        wiring.set(name, BigInt(parseDuration(literal)));
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw ctx.diagnostics.error(`${subject}: ${error.message}`, { range, cause: error });
      }
      return;
    }
    if (this.inputType(ctx, name)?.kind === ValueKind.STRING) return;
    throw ctx.diagnostics.error(
      `${subject}: a duration reference must be typed str (tag it: \`${name}: !str \${...}\`)`,
      { range },
    );
  }

  /** Return the message prefix naming this node's input `name`. */
  private inputSubject(name: string): string {
    return `node ${reprText(this.id)} (tool_id: ${this.toolId}) input ${reprText(name)}`;
  }

  /** Return the range of input `name`, else of `input:`, else of the node. */
  private inputRange(name: string): SourceRange | null {
    return this.fieldRanges.get(siteKey("input", name)) ?? this.fieldRanges.get(siteKey("field", "inputs")) ?? this.range;
  }

  /**
   * Return the type input `name`'s binding is proven to have: a tagged binding's declared type,
   * else the one `ctx.bindingContractReader` reads; `null` when unproven or not one exact type.
   */
  private inputType(ctx: BuildContext, name: string): Type | null {
    const contract =
      this.inputTypes.get(name) ??
      ctx.bindingContractReader?.(this.inputSubject(name), this.inputs[name], this.inputRange(name)) ??
      null;
    return isExactly(contract, ValueContract) ? contract.type : null;
  }
}

/** The inputs a `take` reads. */
const TAKE_INPUTS: ReadonlySet<string> = new Set(["from", "take", "timeout"]);
const DATE = Type.scalar(ValueKind.DATE);
const DATETIME = Type.scalar(ValueKind.DATETIME);
const UNTIL_FIX = "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`";
/** The values of a take's `take:`. */
const TAKE_MODE = new Type({ kind: ValueKind.STRING, tags: new Set(["one", "all"]) });

/** The keys the adaptive desugar reads from an `adaptive_questions:` block. */
export const ADAPTIVE_KEYS: ReadonlySet<string> = new Set(["prompt", "llm_config", "retries"]);

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
export function nonTextQuestionField(questions: readonly unknown[]): [(string | number)[], unknown] | null {
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
 * Return the range of `key` in the mapping under `cursor`, or `null` when it has none.
 *
 * `null` also covers a mapping past a recursive alias cut, so the caller's error falls back to
 * an enclosing range.
 */
export function keyRange(cursor: SourceCursor, key: string): SourceRange | null {
  try {
    return cursor.keyRangeOf(key);
  } catch (error) {
    if (error instanceof SourceAccessError) return null;
    throw error;
  }
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
 * Return a light answer-field schema for an asking node, derived from its output type.
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
