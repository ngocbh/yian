/**
 * Occurrence-local extraction for inline composite child-flow bodies.
 *
 * Extraction projects immutable source cursors. Each alias occurrence therefore gets its own
 * masked view while the document graph stays unchanged, and retained child cursors keep their
 * original paths and ranges. A tagged `nodes:` entry is a value, never an inline body, even when
 * its mapping has a `kind:` key.
 *
 * Knows about: diagnostics, typesys values, the compose error helpers and source cursors.
 * Never imports: compose/parse, edges, validate, build, the loader, or the runtime.
 */

import { createHash } from "node:crypto";

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { isRecord, type Value, type ValueRecord } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { setRecordKey } from "../source/common.ts";
import type { SourceCursor } from "../source/index.ts";

/** Callable-name prefix reserved for compiler-generated inline child flows. */
export const INLINE_FLOW_PREFIX = "__inline_flow_";

const INLINE_COMPOSITE_KINDS: ReadonlySet<string> = new Set(["call", "map", "loop"]);

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

  /** Record one retained body; `owner` is copied and frozen. */
  constructor(fields: {
    nodeId: string;
    owner: readonly string[];
    flowId: string;
    nodesCursor: SourceCursor;
    outputCursor: SourceCursor;
    inputRanges: ReadonlyMap<string, SourceRange>;
  }) {
    this.nodeId = fields.nodeId;
    this.owner = Object.freeze([...fields.owner]);
    this.flowId = fields.flowId;
    this.nodesCursor = fields.nodesCursor;
    this.outputCursor = fields.outputCursor;
    this.inputRanges = fields.inputRanges;
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
 * The id hashes the owner's compact JSON text (UTF-8), the same bytes Python's
 * `json.dumps(..., ensure_ascii=False, separators=(",", ":"))` gives for a list of strings. A
 * lone surrogate differs: JavaScript escapes it, where Python fails to encode it.
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
 * @param nodesCursor The located `nodes:` value.
 * @param options.owner The lexical owner path of the flow these nodes belong to.
 * @returns The masked `nodes:` cursor and the retained bodies by node id, in source order.
 * @throws LoadError for an authored call into the private namespace or a malformed inline body.
 * @throws SourceAccessError when a needed field is behind a recursive alias.
 */
export function extractInlineBodies(
  nodesCursor: SourceCursor,
  options: { owner: readonly string[] },
): [SourceCursor, Map<string, InlineBody>] {
  const { owner } = options;
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
      if (Object.hasOwn(rawBody, "output")) {
        throw diagnostics.error(`${subject}: \`output:\` is only valid with inline \`nodes:\``, {
          range: bodyCursor.rangeOf("output"),
        });
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
    const callTarget = rawBody["call"];
    if (typeof callTarget !== "string" || !callTarget.startsWith(INLINE_FLOW_PREFIX)) continue;
    throw diagnostics.error(
      `node ${reprText(nodeId)} (kind=${kind}): call targets beginning with ` +
        `${reprText(INLINE_FLOW_PREFIX)} are reserved for compiler-generated inline child flows`,
      { range: bodyCursor.rangeOf("call") },
    );
  }
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
