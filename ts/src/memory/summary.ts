/**
 * Pure whole-turn selection and private summary-cache lineage evidence.
 *
 * Projections and summaries are opaque text. This module owns no model, mutable history, lease, or
 * public snapshot representation: `admissions` reads turn order from a view's ledger,
 * `prefixDigest` proves the raw ledger prefix before one admission (SHA-256 hex of `logJson` text,
 * equal to Python yian's digest because the input is lists of strings), and `Summary` binds derived
 * text to that prefix and a projection identity.
 *
 * Knows about: `memory/contracts` and `typesys/json`.
 * Never imports: any other yian module.
 */

// The default import keeps calls patchable by test spies.
import crypto from "node:crypto";

import { logJson } from "../typesys/json.ts";
import { ConversationInvocation, type ConversationView, sameConversationInvocation } from "./contracts.ts";

// The characters Python's `str.strip()` removes (`str.isspace()`), copied from
// nodes/agent/structured.ts because memory never imports nodes.
const PY_SPACE =
  "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_BLANK = new RegExp(`^[${PY_SPACE}]*$`, "u");

// `ConversationApplied` holds each operation id as canonical JSON `[run_id, start_sequence, ordinal]`.
type OperationId = [string, number, number];

/**
 * Read turn order from admissions, including turns with no effect or item.
 *
 * @param view - A conversation view; its ledger isn't revalidated.
 * @returns Each admitted invocation, in ledger order.
 * @example `admissions(view).length` counts every admitted turn, closed or not.
 */
export function admissions(view: ConversationView): readonly ConversationInvocation[] {
  const result: ConversationInvocation[] = [];
  for (const entry of view.applied) {
    const [runId, sequence, ordinal] = JSON.parse(entry.operationId) as OperationId;
    if (ordinal === 0) result.push(new ConversationInvocation(runId, sequence));
  }
  return Object.freeze(result);
}

/**
 * Prove the complete raw prefix preceding the next admitted invocation.
 *
 * @param view - A conversation view; its ledger isn't revalidated.
 * @param count - The number of admitted turns the prefix covers.
 * @returns The lowercase SHA-256 of the `logJson` text of every entry before admission `count`
 *   (zero-based), each as `[operation_id, event_digest, before_digest, after_digest]`.
 * @throws RangeError when the view has no admission after the first `count`.
 * @example `prefixDigest(view, 0)` hashes `[]` when the view has an admission.
 */
export function prefixDigest(view: ConversationView, count: number): string {
  let seen = 0;
  const prefix: string[][] = [];
  for (const entry of view.applied) {
    if ((JSON.parse(entry.operationId) as OperationId)[2] === 0) {
      if (seen === count) return crypto.createHash("sha256").update(logJson(prefix)).digest("hex");
      seen += 1;
    }
    prefix.push([entry.operationId, entry.eventDigest, entry.beforeDigest, entry.afterDigest]);
  }
  throw new RangeError("a summary prefix must precede the current admission");
}

/** Derived text bound to an immutable raw prefix and projection identity. Frozen. */
export class Summary {
  readonly namespace: string;
  readonly generation: number;
  readonly projectionVersion: string;
  /** The covered invocations, in admission order. */
  readonly covered: readonly ConversationInvocation[];
  /** The `prefixDigest` of the view over `covered.length` turns. */
  readonly digest: string;
  readonly text: string;

  /**
   * Keep the fields as given, checking only the text, as Python's dataclass does.
   *
   * @throws TypeError when `text` isn't a string.
   * @throws RangeError when `text` is empty after Python's `strip()`.
   */
  constructor(
    namespace: string,
    generation: number,
    projectionVersion: string,
    covered: readonly ConversationInvocation[],
    digest: string,
    text: string,
  ) {
    if (typeof text !== "string") throw new TypeError("summary must be a string");
    if (PY_BLANK.test(text)) throw new RangeError("summary must contain nonempty text");
    this.namespace = namespace;
    this.generation = generation;
    this.projectionVersion = projectionVersion;
    this.covered = Object.freeze([...covered]);
    this.digest = digest;
    this.text = text;
    Object.freeze(this);
  }

  /**
   * Check ownership and complete prefix evidence before reusing text.
   *
   * @param view - The current conversation view.
   * @param prior - The invocations before the current one, in admission order.
   * @param version - The caller's projection version.
   * @returns Whether the namespace, generation and version match, `covered` is non-empty and
   *   leads `prior`, and the digest proves the view's prefix over `covered`.
   * @throws RangeError when `covered` leads `prior` but the view has no admission after it.
   */
  matches(view: ConversationView, prior: readonly ConversationInvocation[], version: string): boolean {
    return (
      this.namespace === view.namespace &&
      this.generation === view.generation &&
      this.projectionVersion === version &&
      this.covered.length > 0 &&
      // Python compares `prior[:len(covered)] == covered`: prior is at least as long, then each pair matches.
      prior.length >= this.covered.length &&
      this.covered.every((invocation, index) => sameConversationInvocation(prior[index]!, invocation)) &&
      this.digest === prefixDigest(view, this.covered.length)
    );
  }
}
