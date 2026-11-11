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
import { bindingSource, isExactly, ValueContract, ValueTemplate } from "../../nodes/contracts.ts";
import { parseQuestions, QuestionSpecError } from "../../nodes/tool/questions.ts";
import { memoryKey } from "../../nodes/base.ts";
import { ToolNode } from "../../nodes/tool/node.ts";
import { LOAD_SKILL } from "../../skills/tool.ts";
import { ASK, ASK_USER, BUILTIN_TOOL_IDS, type BuiltinToolId, SLEEP, TAKE } from "../../tools/context.ts";
import { GrantError, visibleToolName } from "../../tools/grants.ts";
import { RESERVED_TOOL_NAMES, type ToolRef } from "../../tools/registry.ts";
import { parseDateOrDatetime, parseDuration } from "../../typesys/temporal.ts";
import {
  canonicalTypeLabel,
  isInboxType,
  isMemoryType,
  isList,
  isRecord,
  listKindForElement,
  Type,
  ValueKind,
  valueRepr,
} from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { checkMemoryGrants, grantErrorText, memoryBinding } from "../helpers/memory.ts";
import { literalValue, NOT_LITERAL } from "../../compile/output_contracts.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
import { MODE_REMOVED_FIX, retriesError } from "./agent.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  notTextError,
  rangeAt,
  siteKey,
  siteLabel,
  valueTypeError,
} from "./base.ts";

/** The fields `ToolDescriptor`'s constructor takes: the common ones and the TOOL's own. */
export interface ToolDescriptorFields extends NodeDescriptorFields {
  toolRef: ToolRef;
  retries?: number;
  asserts?: readonly string[] | null;
  adaptiveQuestions?: unknown;
}

/**
 * kind=tool: a direct, deterministic call to one tool in the run's `ToolSet`.
 *
 * `allowed` adds `tool_id`, `inputs`/`outputs`, `retries`, `asserts` and `adaptive_questions` to
 * the commons; `tool_id` is required. A tool's authored `input:` is its sink-binding surface, so
 * `inputs` is its one binding field. `output:` is optional; when present it is the node's output
 * contract, which the engine checks the tool's return value against on every call.
 *
 * `tool_id:` is a tool's id or a group's tool, `{group: tool}`. `tool_id: take | sleep | ask` names
 * a yian built-in, whose inputs are checked at load and whose output is inferred (`take`, `sleep`)
 * or is the answer type (`ask`, which alone takes `adaptive_questions:`).
 */
export class ToolDescriptor extends NodeDescriptor {
  static override readonly kind = "tool";
  static override readonly allowed = this.allow(
    "tool_id",
    "inputs",
    "outputs",
    "retries",
    "asserts",
    "adaptive_questions",
  );
  static override readonly required = this.require("tool_id");
  static override readonly bindingFields: readonly string[] = ["inputs"];
  static override readonly writtenFields = [
    ...NodeDescriptor.writtenFields,
    ["adaptive_questions.prompt", "a template"],
  ] as const;

  /** The tool, looked up in the run's `ToolSet`: a bare id, or a group's tool. */
  readonly toolRef: ToolRef;
  /** The effective retry count, so the manifest holds it; zero is left out of the manifest. */
  readonly retries: number;
  /** The node asserts. */
  readonly asserts: readonly string[];
  /** `ask`: the authored `adaptive_questions:` block as YAML read it, or `null`. */
  readonly adaptiveQuestions: unknown;

  /**
   * Create a descriptor from `fields`. `retries` defaults to `current().toolRetries`; `asserts`
   * reads `null` as empty.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: ToolDescriptorFields) {
    super(fields);
    this.toolRef = fields.toolRef;
    this.retries = fields.retries ?? current().toolRetries;
    this.asserts = fields.asserts ?? [];
    this.adaptiveQuestions = fields.adaptiveQuestions ?? null;
  }

  /** The tool's name as messages show it: its id, or `<group>_<tool>` for a group's tool. */
  get toolId(): string {
    return visibleToolName(this.toolRef.group, this.toolRef.name);
  }

  /**
   * Check an `adaptive_questions:` block and an `ask`'s reserved input around the template
   * `parse`, and range the block's brief and `llm_config:`.
   *
   * @throws LoadError when `adaptive_questions:` is on a tool id other than `ask` (at the block),
   *   authors a key other than `prompt`/`llm_config`/`retries` or a non-text brief, an `ask`
   *   authors input `__questions`, or the template `parse` rejects the node.
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const toolId = cursor.get("tool_id");
    if (typeof toolId === "string" && cursor.has("adaptive_questions")) {
      if (toolId !== ASK && toolId !== ASK_USER) {
        throw diagnostics.error(
          `node ${reprText(nodeId)} (tool_id: ${toolId}): \`adaptive_questions:\` applies only to tool_id: ask`,
          { range: cursor.keyRangeOf("adaptive_questions") },
        );
      }
      checkAdaptiveBlock(nodeId, `tool_id: ${ASK}`, cursor, diagnostics);
    }
    const desc = super.parse(nodeId, cursor) as ToolDescriptor;
    if (desc.toolId === ASK && Object.hasOwn(desc.inputs, QUESTIONS_INPUT)) {
      throw diagnostics.error(
        `node ${reprText(nodeId)} (tool_id: ask): input '${QUESTIONS_INPUT}' is reserved for adaptive_questions`,
        { range: rangeAt(cursor, ["inputs", QUESTIONS_INPUT]) ?? desc.range },
      );
    }
    recordAdaptiveRanges(desc, cursor);
    return desc as InstanceType<T>;
  }

  /**
   * Read `tool_id:` into `toolRef`, rejecting anything but a tool id or a one-key `{group: tool}`
   * (`null` included), a built-in or `load_skill` named in a group, the reserved `load_skill`, the
   * fields a built-in refuses, and a bad `retries:`; store `retries` as a `number`, then run the
   * base checks.
   *
   * `take` and `sleep` refuse `retries:`, `asserts:` and `output:` (they make no tool call, and
   * their output is fixed); `ask` refuses `retries:`. `tool_id: ask_user`, `ask`'s old name, is
   * rewritten to `ask` first, so the node and its manifest are an `ask`'s.
   *
   * @throws LoadError at the field for a bad shape or a refused field.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
    cursor: SourceCursor,
  ): void {
    const rangeOf = (key: string) => ranges.get(siteKey("field", key)) ?? ownRange;
    const raw = fields["toolId"] ?? null;
    const ref = ToolDescriptor.readToolRef(nodeId, raw, rangeOf("tool_id"), diagnostics);
    delete fields["toolId"];
    fields["toolRef"] = ref;
    // A group's tool showing as a built-in's name is refused at build, which knows whether the
    // group is a memory input, whose fix is to rename the input.
    const toolId = visibleToolName(ref.group, ref.name);
    if (toolId === LOAD_SKILL && ref.group === null) {
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
    if (toolId === ASK) ToolDescriptor.checkAskSurface(nodeId, fields, ranges, ownRange, diagnostics);
    if (Object.hasOwn(fields, "retries")) {
      const problem = retriesError(fields["retries"]);
      if (problem !== null) {
        throw diagnostics.error(`node ${reprText(nodeId)} (kind=${this.kind}): \`retries:\` ${problem}`, {
          range: rangeOf("retries"),
        });
      }
      fields["retries"] = Number(fields["retries"]);
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics, cursor);
  }

  /**
   * Read an authored `tool_id:`: a tool id (`ask_user` read as `ask`), or a one-key mapping
   * `{group: tool}` naming one tool of a group.
   *
   * @throws LoadError at `tool_id:` for any other value, `null` included, or a built-in or
   *   `load_skill` named in a group.
   */
  private static readToolRef(
    nodeId: string,
    raw: unknown,
    range: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): ToolRef {
    const fail = (problem: string): never => {
      const where = `node ${reprText(nodeId)} (kind=${this.kind})`;
      throw diagnostics.error(`${where}: ${problem}`, { range });
    };
    if (typeof raw === "string") {
      return Object.freeze({ group: null, name: raw === ASK_USER ? ASK : raw });
    }
    if (!isRecord(raw)) {
      const fix = raw === null ? "write a tool name" : "put it in quotes";
      return fail(valueTypeError("tool_id", "a tool name or `{group: tool}`", raw, fix));
    }
    const keys = Object.keys(raw);
    if (keys.length !== 1 || keys[0] === "") {
      return fail("`tool_id:` names one tool: `{cases: search}` or `search`");
    }
    const group = keys[0]!;
    const name = raw[group];
    if (typeof name !== "string" || name === "") {
      const first = Array.isArray(name) ? name[0] : undefined;
      const example = typeof first === "string" && first !== "" ? first : "search";
      return fail(`\`tool_id: {${group}: ...}\` names one tool; write \`{${group}: ${example}}\``);
    }
    if (BUILTIN_TOOL_IDS.has(name as BuiltinToolId) || name === ASK_USER || name === LOAD_SKILL) {
      const bare = name === ASK_USER ? ASK : name;
      const reserved = `\`${name}\` is reserved by yian`;
      const fix = name === LOAD_SKILL ? reserved : `write \`tool_id: ${bare}\``;
      return fail(
        `\`tool_id: {${group}: ${name}}\`: ${name} is a built-in, not a group's tool; ${fix}`,
      );
    }
    return Object.freeze({ group, name });
  }

  /**
   * Check an `ask`'s question surface: one of input `prompt`, input `questions` or
   * `adaptive_questions:` (an error at the node), `questions` and `adaptive_questions:` not both,
   * and an adaptive block with a brief and a valid `retries:` (errors at the block). A non-mapping
   * `input:` is left to the base shape check.
   *
   * @throws LoadError for the first rule broken.
   */
  private static checkAskSurface(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const id = reprText(nodeId);
    const given = fields["inputs"] ?? null;
    // The base check reports a malformed `input:`; reading it as empty would blame the questions.
    if (given !== null && !isRecord(given)) return;
    const inputs = isRecord(given) ? given : {};
    const hasQuestions = (inputs["questions"] ?? null) !== null;
    const auto = fields["adaptiveQuestions"] ?? null;
    const autoRange = ranges.get(siteKey("field", "adaptive_questions")) ?? ownRange;
    if ((inputs["prompt"] ?? null) === null && !hasQuestions && auto === null) {
      throw diagnostics.error(
        `node ${id} (tool_id: ask): ask needs input prompt:, input questions:, or adaptive_questions:`,
        { range: ownRange },
      );
    }
    if (hasQuestions && auto !== null) {
      throw diagnostics.error(
        `node ${id} (tool_id: ask): input questions: and adaptive_questions: are mutually exclusive`,
        { range: autoRange },
      );
    }
    if (auto !== null) checkAdaptiveFields(nodeId, `tool_id: ${ASK}`, auto, autoRange, diagnostics);
  }

  /**
   * Build the runtime `ToolNode` and its flow-owned wiring.
   *
   * Sets the node's `params` from the authored `input:` and its output contract from the
   * optional `output:` (`null` when absent); `_finish` stamps the descriptor origin. A built-in
   * checks its inputs first and infers its output: `take` gives `{events: list(T), closed: bool}`
   * for its `Inbox(T)`, `sleep` gives `None`. An `ask` keeps its declared `output:` (a questions
   * ask with none answers an `object`, a prompt-only one is free form) and derives its
   * `answerSchema` from it. A literal duration (`timeout`, `for`) is wired as its seconds.
   *
   * A `tool_id: {<memory>: <tool>}` naming a memory input of the node's own flow calls that
   * memory's tool, routed through the reserved `__memory_<input>` wiring, never the run's tools.
   *
   * @throws LoadError when `output:` isn't a valid type expression, a built-in's input is unknown,
   *   missing or of the wrong type (at the input), or a memory's tool names an Optional memory or a tool its type lacks.
   */
  override build(ctx: BuildContext): BuildResult {
    const { group, name } = this.toolRef;
    const memory = group === null ? undefined : ctx.flowInputs.get(group);
    if (isMemoryType(memory)) return this.buildMemoryTool(ctx, memory);
    // A group's tool shows as `<group>_<tool>`, which is never a built-in's name.
    if (group !== null && RESERVED_TOOL_NAMES.has(this.toolId)) {
      throw ctx.diagnostics.error(
        `node ${reprText(this.id)} (kind=${this.kind}): \`tool_id: {${group}: ${name}}\` ` +
          `would show as \`${this.toolId}\`, a name yian reserves; rename the group or the tool`,
        { range: this.fieldRanges.get(siteKey("field", "tool_id")) ?? this.range },
      );
    }
    const builtin = BUILTIN_TOOL_IDS.has(this.toolId as BuiltinToolId);
    // A built-in is never retried by the node: take and sleep make no call, and the host owns an ask.
    const retries = builtin ? 0 : this.retries;
    const node = new ToolNode(this.id, { toolId: this.toolRef, title: this.nodeName, retries });
    const wiring = this.sinkWiring();
    if (this.toolId === TAKE) {
      node.outputContract = this.buildTake(ctx);
      this.wireDuration(ctx, wiring, "timeout");
    } else if (this.toolId === SLEEP) {
      this.checkSleep(ctx);
      this.wireDuration(ctx, wiring, "for");
      node.outputContract = new ValueContract(Type.scalar(ValueKind.NONE));
    } else if (this.toolId === ASK) {
      this.checkAskInputs(ctx);
      this.wireDuration(ctx, wiring, "timeout");
      node.outputContract = this.outputContract(ctx);
      // A questions ask with no declared answer gets a free-form record.
      const asksQuestions = [QUESTIONS_INPUT, "questions"].some((name) => Object.hasOwn(this.inputs, name));
      if (node.outputContract === null && (asksQuestions || this.adaptiveQuestions !== null)) {
        node.outputContract = new ValueContract(Type.scalar(ValueKind.OBJECT));
      }
      // The host answers `T` for `output: Optional(T)`: null means nobody answered in time.
      const answer = isExactly(node.outputContract, ValueContract) ? node.outputContract.type : null;
      node.answerSchema = answerSchema(answer === null ? null : answer.replace({ nullable: false }));
    } else {
      node.outputContract = this.outputContract(ctx);
    }
    node.params = this.sinkParams();
    return this._finish(node, wiring);
  }

  /**
   * Build a TOOL node calling the tool `toolRef.name` of the memory input `toolRef.group`, whose
   * type is `memory`.
   *
   * @throws LoadError when the memory is Optional, which a run may leave unbound, or its type lacks
   *   the tool.
   */
  private buildMemoryTool(ctx: BuildContext, memory: Type): BuildResult {
    const { group, name } = this.toolRef;
    const input = group!;
    const range = this.fieldRanges.get(siteKey("field", "tool_id")) ?? this.range;
    if (memory.nullable) {
      throw ctx.diagnostics.error(
        `node ${reprText(this.id)}: \`tool_id: {${input}: ${name}}\` names \`${input}\`, an Optional memory; ` +
          "a TOOL node needs one that is always bound",
        { range },
      );
    }
    try {
      checkMemoryGrants([{ kind: "group", group: input, tools: [name] }], ctx.flowInputs, ctx.memoryTypes);
    } catch (error) {
      if (!(error instanceof GrantError)) throw error;
      throw ctx.diagnostics.error(grantErrorText(`node ${reprText(this.id)} (kind=${this.kind})`, error), {
        range,
        cause: error,
      });
    }
    const node = new ToolNode(this.id, {
      toolId: this.toolRef,
      title: this.nodeName,
      retries: this.retries,
      memoryTool: true,
    });
    node.outputContract = this.outputContract(ctx);
    node.params = this.sinkParams();
    const wiring = this.sinkWiring();
    wiring.set(memoryKey(input), memoryBinding(input));
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
   * datetime, a reference's type must not be disjoint from both `date` and `datetime`, and a list
   * or mapping holding `${...}` is refused.
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
    if (typ === null) {
      // A list or mapping holding `${...}` is unproven, yet it can never be a date or datetime.
      const source = this.inputs["until"];
      if (!(source instanceof ValueTemplate)) return;
      const shape = isList(source.value) ? "a list" : "a mapping";
      throw ctx.diagnostics.error(`${subject}: must be a date or datetime, but the value is ${shape}; ${UNTIL_FIX}`, {
        range,
      });
    }
    if (typeRelation(typ, DATE) !== "disjoint" || typeRelation(typ, DATETIME) !== "disjoint") return;
    throw ctx.diagnostics.error(
      `${subject}: must be a date or datetime, but the value is ${canonicalTypeLabel(typ)}; ${UNTIL_FIX}`,
      { range },
    );
  }

  /**
   * Check an ask's base surface: `prompt` is text, and `questions` (or the `__questions` the
   * adaptive desugar wires) is a valid question list. A reference passes when its type may be
   * one. Any other input is a host extra, checked against the host's declaration at run start.
   *
   * @throws LoadError at the input for a non-text `prompt`, a literal question list
   *   `parseQuestions` refuses, or a reference whose type can't be what the input needs.
   */
  private checkAskInputs(ctx: BuildContext): void {
    if (Object.hasOwn(this.inputs, "prompt")) {
      const subject = this.inputSubject("prompt");
      const literal = literalValue(bindingSource(this.inputs["prompt"]));
      if (literal !== NOT_LITERAL) {
        if (literal !== null && typeof literal !== "string") {
          throw ctx.diagnostics.error(`${subject}: ${notTextError("prompt", literal)}`, { range: this.inputRange("prompt") });
        }
      } else {
        const typ = this.inputType(ctx, "prompt");
        if (typ !== null && typeRelation(typ, Type.scalar(ValueKind.STRING)) === "disjoint") {
          throw ctx.diagnostics.error(`${subject}: must be text, but the value is ${canonicalTypeLabel(typ)}`, {
            range: this.inputRange("prompt"),
          });
        }
      }
    }
    for (const name of ["questions", QUESTIONS_INPUT]) {
      if (Object.hasOwn(this.inputs, name)) this.checkQuestions(ctx, name);
    }
  }

  /**
   * Check ask input `name`'s question list: a literal list must pass `parseQuestions`, and a
   * reference's type must be able to hold a list of records.
   *
   * @throws LoadError at the input otherwise.
   */
  private checkQuestions(ctx: BuildContext, name: string): void {
    const subject = this.inputSubject(name);
    const range = this.inputRange(name);
    const literal = literalValue(bindingSource(this.inputs[name]));
    if (literal === NOT_LITERAL) {
      const typ = this.inputType(ctx, name);
      if (typ === null || typeRelation(typ, QUESTION_LIST) !== "disjoint") return;
      throw ctx.diagnostics.error(
        `${subject}: must be a list of questions, but the value is ${canonicalTypeLabel(typ)}`,
        { range },
      );
    }
    if (!isList(literal)) {
      throw ctx.diagnostics.error(
        `${subject}: must be a list of questions (each item \`- question: ...\`), got ${valueRepr(literal)}`,
        { range },
      );
    }
    const found = nonTextQuestionField(literal);
    if (found !== null) {
      const [path, value] = found;
      throw ctx.diagnostics.error(`${subject}: ${notTextError(siteLabel(path), value)}`, { range });
    }
    try {
      parseQuestions(literal);
    } catch (error) {
      if (!(error instanceof QuestionSpecError)) throw error;
      throw ctx.diagnostics.error(`${subject}: ${error.message}`, { range, cause: error });
    }
  }

  /**
   * Rewrite duration input `name`'s wiring source, when present: a literal to its seconds (a
   * bigint); a reference stays as written once its type is `str`, and is parsed when the node runs.
   *
   * @throws LoadError at the input for a malformed duration, a plain number, a tagged literal, or
   *   a reference not typed `str` (an unproven one included: tag it `!str`).
   */
  private wireDuration(ctx: BuildContext, wiring: Map<string, unknown>, name: string): void {
    if (!wiring.has(name)) return;
    const subject = this.inputSubject(name);
    const range = this.inputRange(name);
    const literal = literalValue(bindingSource(wiring.get(name)));
    if (literal !== NOT_LITERAL) {
      // A tag would check the expanded seconds against it, so refuse it here, where it reads clearly.
      if (this.inputTypes.has(name)) {
        const written = typeof literal === "string" ? literal : "24h";
        throw ctx.diagnostics.error(`${subject}: a literal duration takes no tag; write \`${name}: ${written}\``, {
          range,
        });
      }
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

/** The `ask` input `adaptive_questions:` writes its questions to; no author writes it. */
export const QUESTIONS_INPUT = "__questions";
/** The inputs a `take` reads. */
const TAKE_INPUTS: ReadonlySet<string> = new Set(["from", "take", "timeout"]);
const DATE = Type.scalar(ValueKind.DATE);
const DATETIME = Type.scalar(ValueKind.DATETIME);
const UNTIL_FIX = "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`";
/** What an ask's question list reference must be able to hold. */
const QUESTION_LIST = new Type({ kind: ValueKind.LIST_OBJECT, element: Type.scalar(ValueKind.OBJECT) });
/** The values of a take's `take:`. */
const TAKE_MODE = new Type({ kind: ValueKind.STRING, tags: new Set(["one", "all"]) });

/** The keys the adaptive desugar reads from an `adaptive_questions:` block. */
const ADAPTIVE_KEYS: ReadonlySet<string> = new Set(["prompt", "llm_config", "retries"]);

/**
 * Check an authored `adaptive_questions:` block's keys and brief, before the template parse.
 *
 * @param label The node's kind or tool id as messages name it, such as `tool_id: ask`.
 * @throws LoadError when the block authors `mode:` or a key other than `prompt`/`llm_config`/
 *   `retries` (at the key), or its brief is not text or holds a lone surrogate (at the brief).
 */
function checkAdaptiveBlock(
  nodeId: string,
  label: string,
  cursor: SourceCursor,
  diagnostics: LoadDiagnostics,
): void {
  const subject = `node ${reprText(nodeId)} (${label})`;
  const adaptive = cursor.get("adaptive_questions");
  if (!isRecord(adaptive)) return;
  if (Object.hasOwn(adaptive, "mode")) {
    throw diagnostics.error(`${subject}: \`adaptive_questions.mode:\` was removed: ${MODE_REMOVED_FIX}`, {
      range: keyRange(cursor.child("adaptive_questions"), "mode") ?? cursor.keyRangeOf("adaptive_questions"),
    });
  }
  for (const key of Object.keys(adaptive)) {
    if (ADAPTIVE_KEYS.has(key)) continue;
    throw diagnostics.error(
      `${subject}: \`adaptive_questions:\` field ${reprText(key)} is not allowed ` +
        `(allowed: ${[...ADAPTIVE_KEYS].sort(compareCodeUnits).join(", ")})`,
      { range: keyRange(cursor.child("adaptive_questions"), key) ?? cursor.keyRangeOf("adaptive_questions") },
    );
  }
  const brief = adaptive["prompt"] ?? null;
  if (brief === null) return;
  const briefRange = rangeAt(cursor, ["adaptive_questions", "prompt"]) ?? cursor.keyRangeOf("adaptive_questions");
  if (typeof brief !== "string") {
    throw diagnostics.error(`${subject}: ${notTextError("adaptive_questions.prompt", brief)}`, { range: briefRange });
  }
  checkWrittenValue(brief, `node ${reprText(nodeId)} field 'adaptive_questions.prompt'`, briefRange, diagnostics);
}

/**
 * Record the ranges of an `adaptive_questions:` block's brief, as `siteKey("text",
 * "adaptive_questions.prompt")`, and `llm_config:` value, as `siteKey("field",
 * "adaptive_questions.llm_config")`, so an error points at them rather than the whole block.
 */
function recordAdaptiveRanges(desc: NodeDescriptor, cursor: SourceCursor): void {
  const adaptive = cursor.get("adaptive_questions");
  if (!isRecord(adaptive)) return;
  for (const [key, site] of [
    ["prompt", siteKey("text", "adaptive_questions.prompt")],
    ["llm_config", siteKey("field", "adaptive_questions.llm_config")],
  ] as const) {
    if (!Object.hasOwn(adaptive, key)) continue;
    const range = rangeAt(cursor, ["adaptive_questions", key]);
    if (range !== null) desc.fieldRanges.set(site, range);
  }
}

/**
 * Check a parsed `adaptive_questions:` block: a mapping with its own `prompt:` (the LLM brief)
 * and a valid `retries:`.
 *
 * @param label The node's kind or tool id as messages name it, such as `tool_id: ask`.
 * @throws LoadError at `range` for the first rule broken.
 */
function checkAdaptiveFields(
  nodeId: string,
  label: string,
  adaptive: unknown,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  const subject = `node ${reprText(nodeId)} (${label})`;
  if (!isRecord(adaptive) || !adaptive["prompt"]) {
    throw diagnostics.error(`${subject}: \`adaptive_questions:\` requires a \`prompt:\` (the LLM brief)`, { range });
  }
  const problem = Object.hasOwn(adaptive, "retries") ? retriesError(adaptive["retries"]) : null;
  if (problem !== null) {
    throw diagnostics.error(`${subject}: \`adaptive_questions.retries:\` ${problem}`, { range });
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
  const fields = recordFieldSchemas(typ);
  if (fields !== null) entry["fields"] = fields;
  return entry;
}

/** Return one answer-schema entry per field of a non-empty record type, else `null`. */
function recordFieldSchemas(typ: Type): Readonly<Record<string, unknown>>[] | null {
  if (typ.kind !== ValueKind.OBJECT || typ.fields === undefined || typ.fields.size === 0) return null;
  const requiredKeys = typ.required ?? new Set<string>();
  return [...typ.fields].map(([key, field]) => fieldSchema(key, field, requiredKeys.has(key)));
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
  return recordFieldSchemas(typ) ?? [fieldSchema("answer", typ, !typ.nullable)];
}
