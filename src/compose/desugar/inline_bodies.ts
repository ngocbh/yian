/**
 * Occurrence-local extraction for inline composite child-flow bodies.
 *
 * Extraction projects immutable source cursors. Each alias occurrence therefore gets its own
 * masked view while the document graph stays unchanged, and retained child cursors keep their
 * original paths and ranges. A tagged `nodes:` entry is a value, never an inline body, even when
 * its mapping has a `kind:` key. A `map` or `loop` whose `call:` names a tool gets a one-node body
 * generated the same way, as a projection over the author's own fields.
 *
 * Knows about: diagnostics, typesys values, the compose error and template helpers, the node
 *              kinds (`KINDS`, the CALL node's `call:` shape check and rule), the node parse's
 *              identifier and signature helpers, and source cursors.
 */

import { createHash } from "node:crypto";

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { isList, isRecord, type Value, type ValueRecord } from "../../typesys/values.ts";
import { CALL_HEAD } from "../helpers/directives.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { hasSpan } from "../helpers/tags.ts";
import { readCallTarget } from "../nodes/base.ts";
import type { CallsTool } from "../nodes/call.ts";
import { KINDS } from "../nodes/index.ts";
import {
  identifierSuggestion,
  isIdentifier,
  normalizeNodeSignatureCursor,
  rejectNonIdentifierInput,
} from "../parse/nodes.ts";
import { setRecordKey } from "../source/common.ts";
import { SourceCursor } from "../source/index.ts";

/** Callable-name prefix reserved for compiler-generated inline child flows. */
export const INLINE_FLOW_PREFIX = "__inline_flow_";

const INLINE_COMPOSITE_KINDS: ReadonlySet<string> = new Set(["call", "map", "loop"]);

// A tool call's own fields, which mean nothing on the map or loop running it: they move to the
// generated node, so they apply to each item's call or each turn's. A loop keeps `asserts:`, which
// check its final record.
const LOOP_TOOL_CALL_FIELDS: readonly string[] = ["retries", "output", "adaptive_questions"];
const MAP_TOOL_CALL_FIELDS: readonly string[] = [...LOOP_TOOL_CALL_FIELDS, "asserts"];

/** The fields that move from a `kind` node naming a tool to the node generated for it. */
function toolCallFields(kind: string): readonly string[] {
  return kind === "loop" ? LOOP_TOOL_CALL_FIELDS : MAP_TOOL_CALL_FIELDS;
}

/** One retained inline child and its occurrence-local private identity; frozen. */
export class InlineBody {
  /** The enclosing authored CALL/MAP/LOOP node id. */
  readonly nodeId: string;
  /** The complete lexical owner path for this anonymous child. */
  readonly owner: readonly string[];
  /** The stable private callable id derived from `owner`. */
  readonly flowId: string;
  /** The original located cursor for the child's `nodes:` value. */
  readonly nodesCursor: SourceCursor;
  /** The original located cursor for the child's `output:` value. */
  readonly outputCursor: SourceCursor;
  /** Outer input name to authored binding-value range. */
  readonly inputRanges: ReadonlyMap<string, SourceRange>;
  /** The tool a generated body calls, as `call:` writes it (`{crm: lookup}`); `null` for an authored body. */
  readonly tool: string | null;
  /** The memory input whose tool a generated body calls; `null` for none. */
  readonly memory: string | null;
  /**
   * A generated body's inputs, each with the `input:` key it passes through, the memory's with its
   * own name; empty for an authored body.
   */
  readonly toolInputs: ReadonlyMap<string, string>;

  /** Record one retained body; `owner` is copied and frozen. */
  constructor(fields: {
    nodeId: string;
    owner: readonly string[];
    flowId: string;
    nodesCursor: SourceCursor;
    outputCursor: SourceCursor;
    inputRanges: ReadonlyMap<string, SourceRange>;
    tool?: string | null;
    memory?: string | null;
    toolInputs?: ReadonlyMap<string, string>;
  }) {
    this.nodeId = fields.nodeId;
    this.owner = Object.freeze([...fields.owner]);
    this.flowId = fields.flowId;
    this.nodesCursor = fields.nodesCursor;
    this.outputCursor = fields.outputCursor;
    this.inputRanges = fields.inputRanges;
    this.tool = fields.tool ?? null;
    this.memory = fields.memory ?? null;
    this.toolInputs = fields.toolInputs ?? new Map();
    Object.freeze(this);
  }

  /** A fresh copy of the authored child output value. */
  get output(): Value {
    return this.outputCursor.value;
  }

  /** The authored child output value range when source-backed. */
  get outputRange(): SourceRange | null {
    return this.outputCursor.range;
  }
}

/**
 * Return the stable private callable id for one complete lexical owner path.
 *
 * The id hashes the owner's compact JSON text (UTF-8): no spaces after `,` or `:`, non-ASCII
 * characters written as themselves, and a lone surrogate escaped as `\uXXXX`.
 */
export function inlineFlowId(owner: readonly string[]): string {
  return INLINE_FLOW_PREFIX + createHash("sha256").update(JSON.stringify(owner), "utf8").digest("hex");
}

/**
 * Validate, retain, and mask inline CALL/MAP/LOOP bodies.
 *
 * The returned cursor is an occurrence-local projection. Named composites remain unchanged. An
 * inline composite becomes an ordinary private `call:` in that overlay, while the returned body
 * record retains its original child/output values and source locations. The supplied cursor and
 * its YAML anchors remain read-only.
 *
 * A `map` or `loop` whose `call:` names a tool gets a generated body (`toolBody`): the node calls
 * it, and its one node calls the tool once per item or turn. `output:` beside a named `call:` is
 * kept on a `call` node naming a tool, where it types the tool's result, moves to a map's or loop's
 * generated node, and is refused beside a flow.
 *
 * @param nodesCursor The located `nodes:` value.
 * @param options.owner The lexical owner path of the flow these nodes belong to.
 * @param options.callsTool Says whether a `call`, `map` or `loop` node's `call:` names a tool;
 *   without it every one names a flow.
 * @param options.memories The flow's memory inputs, whose tools a generated body calls.
 * @returns The masked `nodes:` cursor and the retained bodies by node id, in source order.
 * @throws LoadError for an authored call into the private namespace, a malformed inline body,
 *   `output:` beside a named `call:` that isn't a tool call's, a map's or loop's `input:` key naming
 *   the memory it calls, or a loop's carried field that isn't an identifier; as `callsTool` throws.
 * @throws SourceAccessError when a needed field is behind a recursive alias.
 */
export function extractInlineBodies(
  nodesCursor: SourceCursor,
  options: { owner: readonly string[]; callsTool?: CallsTool; memories?: ReadonlySet<string> },
): [SourceCursor, Map<string, InlineBody>] {
  const { owner, callsTool, memories = new Set<string>() } = options;
  const diagnostics = LoadDiagnostics.fromCursor(nodesCursor);
  const nodes = nodesCursor._valueView();
  if (!isRecord(nodes)) return [nodesCursor, new Map()];
  const nodesSnapshot = nodesCursor.value as ValueRecord;

  rejectAuthoredPrivateCallTargets(nodesCursor);
  const bodies = new Map<string, InlineBody>();
  const projected = new Map<string, readonly [Value, SourceCursor | null]>();

  for (const nodeId of Object.keys(nodes)) {
    const bodyCursor = nodesCursor.child(nodeId);
    const rawBody = bodyCursor._valueView();
    // A tagged entry is a value, even a mapping with `kind: call`.
    if (!isRecord(rawBody) || bodyCursor.tag !== null) {
      projected.set(nodeId, [nodesSnapshot[nodeId] as Value, bodyCursor]);
      continue;
    }
    const kind = rawBody["kind"];
    // A non-text kind is left for node dispatch, which reports it at the value.
    if (!(typeof kind === "string" && INLINE_COMPOSITE_KINDS.has(kind))) {
      projected.set(nodeId, [nodesSnapshot[nodeId] as Value, bodyCursor]);
      continue;
    }
    const subject = `node ${reprText(nodeId)} (kind=${kind})`;

    const hasCall = Object.hasOwn(rawBody, "call");
    const hasNodes = Object.hasOwn(rawBody, "nodes");
    if (hasCall === hasNodes) {
      throw diagnostics.error(`${subject}: exactly one of \`call:\` or inline \`nodes:\` is required`, {
        range: hasNodes ? bodyCursor.rangeOf("nodes") : bodyCursor.range,
      });
    }

    if (hasCall) {
      const call = rawBody["call"];
      const runsBody = kind === "map" || kind === "loop";
      // The node's own check reports a malformed `call:` first, as for a flow.
      if (runsBody) readCallTarget(kind, nodeId, { call }, bodyCursor.rangeOf("call"), diagnostics);
      if (runsBody && callsTool?.(nodeId, call, bodyCursor.rangeOf("call"), { kind }) === true) {
        // The node's parse sees it without the tool call's fields, so it couldn't list them.
        const moved = toolCallFields(kind).map((key) => (key === "output" ? "outputs" : key));
        const cls = KINDS.get(kind)!;
        cls._checkKeys(nodeId, cls._normalizeCursor(normalizeNodeSignatureCursor(bodyCursor)), moved);
        const group = isRecord(call) ? Object.keys(call)[0]! : null;
        const memory = group !== null && memories.has(group) ? group : null;
        const value = nodesSnapshot[nodeId] as ValueRecord;
        const body = toolBody(kind, nodeId, bodyCursor, value, owner, memory);
        bodies.set(nodeId, body);
        const [masked, maskedValue] = maskToolBody(kind, bodyCursor, body, value);
        projected.set(nodeId, [maskedValue, masked]);
        continue;
      }
      if (Object.hasOwn(rawBody, "output")) {
        // A `call` node naming a tool keeps it: the type of the tool's result.
        if (!(kind === "call" && callsTool?.(nodeId, call, bodyCursor.rangeOf("call")) === true)) {
          throw diagnostics.error(
            `${subject}: \`output:\` doesn't apply to a flow call: a flow's result type comes from the flow`,
            { range: bodyCursor.rangeOf("output") },
          );
        }
      }
      projected.set(nodeId, [nodesSnapshot[nodeId] as Value, bodyCursor]);
      continue;
    }

    const childNodes = rawBody["nodes"];
    if (!isRecord(childNodes) || Object.keys(childNodes).length === 0) {
      throw diagnostics.error(`${subject}: inline \`nodes:\` must be a non-empty mapping`, {
        range: bodyCursor.rangeOf("nodes"),
      });
    }
    if (!Object.hasOwn(rawBody, "output")) {
      throw diagnostics.error(`${subject}: inline \`nodes:\` requires \`output:\``, { range: bodyCursor.range });
    }

    // Mapping keys are always text here, so only the mapping check can fail.
    const outerInput = Object.hasOwn(rawBody, "input") ? rawBody["input"] : {};
    if (!isRecord(outerInput)) {
      throw diagnostics.error(`${subject}: inline \`input:\` must be a mapping with string keys`, {
        range: bodyCursor.rangeOf("input"),
      });
    }

    const childOwner = [...owner, "node", nodeId, "inline"];
    const flowId = inlineFlowId(childOwner);
    const body = new InlineBody({
      nodeId,
      owner: childOwner,
      flowId,
      nodesCursor: bodyCursor.child("nodes"),
      outputCursor: bodyCursor.child("output"),
      inputRanges: bodyCursor.mappingValueRanges("input"),
    });
    bodies.set(nodeId, body);
    const [masked, maskedValue] = maskInlineBody(bodyCursor, flowId, nodesSnapshot[nodeId] as ValueRecord);
    projected.set(nodeId, [maskedValue, masked]);
  }

  return [nodesCursor.projectMapping(projected), bodies];
}

/**
 * Reject authored composite calls into the compiler-private namespace.
 *
 * This preflight deliberately completes before any inline body is retained or masked. Otherwise
 * an authored call later in source order could target the predictable id of a sibling that
 * extraction had already synthesized.
 */
function rejectAuthoredPrivateCallTargets(nodesCursor: SourceCursor): void {
  const diagnostics = LoadDiagnostics.fromCursor(nodesCursor);
  const nodes = nodesCursor._valueView();
  if (!isRecord(nodes)) return;
  for (const nodeId of Object.keys(nodes)) {
    const bodyCursor = nodesCursor.child(nodeId);
    const rawBody = bodyCursor._valueView();
    if (!isRecord(rawBody) || bodyCursor.tag !== null) continue;
    const kind = rawBody["kind"];
    // A non-text kind is left for node dispatch, which reports it at the value.
    if (!(typeof kind === "string" && INLINE_COMPOSITE_KINDS.has(kind))) continue;
    // `call: {alias: def}` names its callable by its key.
    const callTarget = isRecord(rawBody["call"]) ? Object.keys(rawBody["call"])[0] : rawBody["call"];
    if (typeof callTarget !== "string" || !callTarget.startsWith(INLINE_FLOW_PREFIX)) continue;
    throw diagnostics.error(
      `node ${reprText(nodeId)} (kind=${kind}): call targets beginning with ` +
        `${reprText(INLINE_FLOW_PREFIX)} are reserved for compiler-generated inline child flows`,
      { range: bodyCursor.rangeOf("call") },
    );
  }
}

/**
 * Generate the body a `kind` node's tool call runs each time. Its one node has the node's id and
 * `kind:`, calls the tool and takes the node's `toolCallFields`; its output is that node's. The
 * body's assembly parses that node as the tool call it is.
 *
 * On a map, a literal argument (`limit: 5`) is written straight into the node's `input:`, so the
 * node checks it as a `call` node would. Each other argument is a body input, which the node passes
 * through as `k: ${input.k}`; an argument whose name a body input can't take (`max-results`)
 * passes through one named `arg_max_results`. A loop's arguments are its carried fields, each a
 * body input of its own name, since a literal seed holds only the first turn's value. The `memory`
 * whose tool it calls, if any, is a body input too, which the node's memory routing reads. Each
 * field keeps the author's source, and the memory's binding the source of `call:`, so the body's
 * errors point at what the author wrote.
 *
 * @param memory The memory input whose tool the node calls; `null` for none.
 * @throws LoadError at the key for an `input:` key named `memory`, or a loop's that isn't an
 *   identifier.
 */
function toolBody(
  kind: string,
  nodeId: string,
  cursor: SourceCursor,
  value: ValueRecord,
  owner: readonly string[],
  memory: string | null,
): InlineBody {
  const callCursor = cursor.child("call");
  const diagnostics = LoadDiagnostics.fromCursor(cursor);
  // A malformed `input:` is left to the node's own check.
  const authored = isRecord(value["input"]) ? value["input"] : {};
  const inputCursor = Object.hasOwn(value, "input") ? cursor.child("input") : callCursor;
  if (memory !== null && Object.hasOwn(authored, memory)) {
    throw diagnostics.error(
      `${kind} node ${reprText(nodeId)}: \`${memory}\` is the memory this node calls; yian passes it, so remove the entry`,
      { range: inputCursor.keyRangeOf(memory) },
    );
  }
  const authoredRanges = cursor.mappingValueRanges("input");
  const passed = new Map<string, readonly [Value, SourceCursor | null]>();
  const toolInputs = new Map<string, string>();
  const inputRanges = new Map<string, SourceRange>();
  const taken = new Set([...Object.keys(authored), ...(memory === null ? [] : [memory])]);
  for (const [key, argument] of Object.entries(authored)) {
    if (kind === "loop") {
      // The tool's result names the carried fields that change, so it can't rename one.
      const what = `loop node ${reprText(nodeId)}: carried field`;
      rejectNonIdentifierInput(what, key, inputCursor.keyRangeOf(key), diagnostics);
    } else if (isLiteral(argument)) {
      passed.set(key, [argument, inputCursor.child(key)]);
      continue;
    }
    // A body input is read as `${input.<name>}`, and a map binds `over` itself, so an argument
    // named otherwise gets a name of its own; the tool still gets it as `key`.
    let name = key;
    if (!isIdentifier(key) || (kind === "map" && key === "over")) {
      const base = `arg_${identifierSuggestion(key)}`;
      name = base;
      for (let n = 2; taken.has(name); n++) name = `${base}_${n}`;
      taken.add(name);
    }
    passed.set(key, [`\${input.${name}}`, inputCursor.child(key)]);
    toolInputs.set(name, key);
    const range = authoredRanges.get(key);
    if (range !== undefined) inputRanges.set(name, range);
  }
  if (memory !== null) {
    toolInputs.set(memory, memory);
    if (callCursor.range !== null) inputRanges.set(memory, callCursor.range);
  }

  const input = inputCursor.projectMapping(passed);
  const fields = new Map<string, readonly [Value, SourceCursor | null]>([
    ["kind", [kind, cursor.child("kind")]],
    ["call", [value["call"]!, callCursor]],
    ["input", [input.value, input]],
  ]);
  for (const key of toolCallFields(kind)) {
    if (Object.hasOwn(value, key)) fields.set(key, [value[key]!, cursor.child(key)]);
  }
  const node = cursor.projectMapping(fields);
  const call = value["call"];
  const [group, name] = isRecord(call) ? Object.entries(call)[0]! : [null, call];
  const childOwner = [...owner, "node", nodeId, "inline"];
  return new InlineBody({
    nodeId,
    owner: childOwner,
    flowId: inlineFlowId(childOwner),
    nodesCursor: cursor.projectMapping(new Map([[nodeId, [node.value, node]]])),
    outputCursor: SourceCursor.virtual(`\${${nodeId}.output}`),
    inputRanges,
    tool: group === null ? String(name) : `{${group}: ${String(name)}}`,
    memory,
    toolInputs,
  });
}

/**
 * Whether `value` holds no `${...}` span and no `call(...)` at any depth, so it means the same
 * written on the generated node as on the node the author wrote.
 */
function isLiteral(value: Value, seen: Set<object> = new Set()): boolean {
  if (typeof value === "string") return !hasSpan(value) && !CALL_HEAD.test(value.trim());
  if (!isList(value) && !isRecord(value)) return true;
  // A YAML alias may share or nest a value; each is walked once.
  if (seen.has(value)) return true;
  seen.add(value);
  return (isList(value) ? value : Object.values(value)).every((item) => isLiteral(item, seen));
}

/**
 * Project a `kind` node naming a tool onto a private `call:` of `body`, its generated body: the
 * node keeps its other fields, and its `input:` binds each of the body's inputs, the memory's as
 * `${input.<memory>}` at `call:`.
 */
function maskToolBody(
  kind: string,
  cursor: SourceCursor,
  body: InlineBody,
  bodyValue: ValueRecord,
): [SourceCursor, ValueRecord] {
  const masked: ValueRecord = {};
  const entries = new Map<string, readonly [Value, SourceCursor | null]>();
  for (const [key, value] of Object.entries(bodyValue)) {
    if (toolCallFields(kind).includes(key) || key === "input") continue;
    setRecordKey(masked, key, key === "call" ? body.flowId : value);
    entries.set(key, [null, cursor.child(key)]);
  }
  const authored = bodyValue["input"];
  if (authored !== undefined && !isRecord(authored)) {
    // Left for the node's own check.
    masked["input"] = authored;
    entries.set("input", [null, cursor.child("input")]);
  } else if (body.toolInputs.size > 0) {
    const base = authored === undefined ? cursor.child("call") : cursor.child("input");
    const input = new Map<string, readonly [Value, SourceCursor | null]>(
      [...body.toolInputs].map(([name, key]) =>
        key === body.memory
          ? [name, [`\${input.${key}}`, cursor.child("call")]]
          : [name, [authored![key]!, base.child(key)]],
      ),
    );
    const inputCursor = base.projectMapping(input);
    masked["input"] = inputCursor.value;
    entries.set("input", [null, inputCursor]);
  }
  return [cursor.projectMapping(entries), masked];
}

/** Project one `nodes:`/`output:` pair into a private `call:`. */
function maskInlineBody(
  cursor: SourceCursor,
  flowId: string,
  bodyValue: ValueRecord,
): [SourceCursor, ValueRecord] {
  const masked: ValueRecord = {};
  const entries = new Map<string, readonly [Value, SourceCursor | null]>();
  for (const [key, value] of Object.entries(bodyValue)) {
    if (key === "nodes") {
      masked["call"] = flowId;
      entries.set("call", [null, cursor.child("nodes")]);
    } else if (key !== "output") {
      setRecordKey(masked, key, value);
      entries.set(key, [null, cursor.child(key)]);
    }
  }
  return [cursor.projectMapping(entries), masked];
}
