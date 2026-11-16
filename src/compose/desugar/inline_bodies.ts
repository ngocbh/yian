/**
 * Occurrence-local extraction for inline composite child-flow bodies.
 *
 * Extraction projects immutable source cursors. Each alias occurrence therefore gets its own
 * masked view while the document graph stays unchanged, and retained child cursors keep their
 * original paths and ranges. A tagged `nodes:` entry is a value, never an inline body, even when
 * its mapping has a `kind:` key. A `map` whose `call:` names a tool gets a one-node body generated
 * the same way, as a projection over the author's own fields.
 *
 * Knows about: diagnostics, typesys values, the compose error helpers, the CALL node's
 *              `call:` shape check and rule, and source cursors.
 */

import { createHash } from "node:crypto";

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { isRecord, type Value, type ValueRecord } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { readCallTarget } from "../nodes/base.ts";
import type { CallsTool } from "../nodes/call.ts";
import { setRecordKey } from "../source/common.ts";
import { SourceCursor } from "../source/index.ts";

/** Callable-name prefix reserved for compiler-generated inline child flows. */
export const INLINE_FLOW_PREFIX = "__inline_flow_";

const INLINE_COMPOSITE_KINDS: ReadonlySet<string> = new Set(["call", "map", "loop"]);

// A tool call's own fields, which mean nothing on the map running it: they move to the generated
// node, so they apply to each item's call.
const TOOL_CALL_FIELDS: ReadonlySet<string> = new Set(["retries", "output", "adaptive_questions", "asserts"]);

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

  /** Record one retained body; `owner` is copied and frozen. */
  constructor(fields: {
    nodeId: string;
    owner: readonly string[];
    flowId: string;
    nodesCursor: SourceCursor;
    outputCursor: SourceCursor;
    inputRanges: ReadonlyMap<string, SourceRange>;
    tool?: string | null;
  }) {
    this.nodeId = fields.nodeId;
    this.owner = Object.freeze([...fields.owner]);
    this.flowId = fields.flowId;
    this.nodesCursor = fields.nodesCursor;
    this.outputCursor = fields.outputCursor;
    this.inputRanges = fields.inputRanges;
    this.tool = fields.tool ?? null;
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
 * A `map` whose `call:` names a tool gets a generated body (`toolBody`): the map calls it, and its
 * one node calls the tool once per item. `output:` beside a named `call:` is kept on a `call` node
 * naming a tool, where it types the tool's result, moves to a map's generated node, and is refused
 * otherwise.
 *
 * @param nodesCursor The located `nodes:` value.
 * @param options.owner The lexical owner path of the flow these nodes belong to.
 * @param options.callsTool Says whether a `call` or `map` node's `call:` names a tool; without it
 *   every one names a flow.
 * @param options.memories The flow's memory inputs, whose tools a generated body calls.
 * @returns The masked `nodes:` cursor and the retained bodies by node id, in source order.
 * @throws LoadError for an authored call into the private namespace, a malformed inline body,
 *   `output:` beside a named `call:` that isn't a tool call's, or a map's `input:` key naming the
 *   memory it calls; as `callsTool` throws.
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
      // The map's own check reports a malformed `call:` first, as for a flow.
      if (kind === "map") readCallTarget(kind, nodeId, { call }, bodyCursor.rangeOf("call"), diagnostics);
      if (kind === "map" && callsTool?.(nodeId, call, bodyCursor.rangeOf("call"), { kind }) === true) {
        const group = isRecord(call) ? Object.keys(call)[0]! : null;
        const memory = group !== null && memories.has(group) ? group : null;
        const value = nodesSnapshot[nodeId] as ValueRecord;
        const body = toolBody(kind, nodeId, bodyCursor, value, owner, memory);
        bodies.set(nodeId, body);
        const [masked, maskedValue] = maskToolBody(bodyCursor, body.flowId, value, memory);
        projected.set(nodeId, [maskedValue, masked]);
        continue;
      }
      if (Object.hasOwn(rawBody, "output")) {
        // A `call` node naming a tool keeps it: the type of the tool's result.
        if (!(kind === "call" && callsTool?.(nodeId, call, bodyCursor.rangeOf("call")) === true)) {
          const ruled = kind !== "loop" && callsTool !== undefined;
          const problem = ruled
            ? "`output:` doesn't apply to a flow call: a flow's result type comes from the flow"
            : "`output:` is only valid with inline `nodes:`";
          throw diagnostics.error(`${subject}: ${problem}`, { range: bodyCursor.rangeOf("output") });
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
 * Generate the body a `kind` node's tool call runs each time. Its inputs are the node's `input:`
 * keys and the `memory` whose tool it calls, if any; its one node has the node's id, calls the tool
 * with `k: ${input.k}` for each input, and takes the node's `TOOL_CALL_FIELDS`; its output is that
 * node's. Each field keeps the author's source, and the memory's binding the source of `call:`, so
 * the body's errors point at what the author wrote.
 *
 * @param memory The memory input whose tool the node calls; `null` for none.
 * @throws LoadError at the key for an `input:` key named `memory`.
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
  // A malformed `input:` is left to the map's own check.
  const authored = isRecord(value["input"]) ? value["input"] : {};
  const inputCursor = Object.hasOwn(value, "input") ? cursor.child("input") : callCursor;
  if (memory !== null && Object.hasOwn(authored, memory)) {
    throw LoadDiagnostics.fromCursor(cursor).error(
      `${kind} node ${reprText(nodeId)}: \`${memory}\` is the memory this node calls; yian passes it, so remove the entry`,
      { range: inputCursor.keyRangeOf(memory) },
    );
  }
  const passed = new Map<string, readonly [Value, SourceCursor | null]>(
    Object.keys(authored).map((key) => [key, [`\${input.${key}}`, inputCursor.child(key)]]),
  );
  if (memory !== null) passed.set(memory, [`\${input.${memory}}`, callCursor]);
  const inputRanges = new Map(cursor.mappingValueRanges("input"));
  if (memory !== null && callCursor.range !== null) inputRanges.set(memory, callCursor.range);

  const input = inputCursor.projectMapping(passed);
  const fields = new Map<string, readonly [Value, SourceCursor | null]>([
    ["kind", ["call", cursor.child("kind")]],
    ["call", [value["call"]!, callCursor]],
    ["input", [input.value, input]],
  ]);
  for (const key of TOOL_CALL_FIELDS) {
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
  });
}

/**
 * Project a node naming a tool onto a private `call:` of its generated body, `flowId`: the node
 * keeps its other fields and binds the `memory` whose tool it calls as `${input.<memory>}`, at
 * `call:`.
 */
function maskToolBody(
  cursor: SourceCursor,
  flowId: string,
  bodyValue: ValueRecord,
  memory: string | null,
): [SourceCursor, ValueRecord] {
  const masked: ValueRecord = {};
  const entries = new Map<string, readonly [Value, SourceCursor | null]>();
  for (const [key, value] of Object.entries(bodyValue)) {
    if (TOOL_CALL_FIELDS.has(key)) continue;
    setRecordKey(masked, key, key === "call" ? flowId : value);
    entries.set(key, [null, cursor.child(key)]);
  }
  const authored = Object.hasOwn(bodyValue, "input") ? bodyValue["input"]! : {};
  if (memory !== null && isRecord(authored)) {
    const base = Object.hasOwn(bodyValue, "input") ? cursor.child("input") : cursor.child("call");
    const input = new Map<string, readonly [Value, SourceCursor | null]>(
      Object.keys(authored).map((key) => [key, [authored[key]!, base.child(key)]]),
    );
    input.set(memory, [`\${input.${memory}}`, cursor.child("call")]);
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
