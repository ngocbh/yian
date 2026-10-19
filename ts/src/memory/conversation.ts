/**
 * Pure conversation transitions, exact JSON codecs and content evidence.
 *
 * This leaf owns no backend state or external effects. Both live application and recovery can
 * fold the same closed checkpoint vocabulary through these rules: `advanceConversation` applies one
 * checkpoint to a validated view; the codecs (`conversationViewToJson`, `conversationViewFromJson`,
 * `conversationCheckpointToJson`, `conversationCheckpointFromJson`) write and read frozen wire JSON
 * whose counters are `bigint`s; and the digests (`conversationContentDigest`, `checkpointDigest`)
 * are SHA-256 hex of `logJson` text. They differ from Python yian's digests only for floats, astral
 * key order and records shaped like the codec's `$float`/`$record` tags, which is fine because TS
 * stores are read only by TS.
 *
 * Knows about: `memory/contracts` and `typesys/json`.
 * Never imports: any other yian module.
 */

// The default import keeps calls patchable by test spies.
import crypto from "node:crypto";

import { logJson } from "../typesys/json.ts";
import {
  ConversationAdmitted,
  ConversationApplied,
  type ConversationCheckpoint,
  ConversationClosed,
  ConversationCompletion,
  ConversationEffectObserved,
  ConversationEffectRequested,
  ConversationEffectState,
  ConversationInvocation,
  ConversationItem,
  ConversationStamp,
  ConversationView,
  freezeJsonObject,
  type JsonObject,
  type JsonValue,
  MemoryOperationError,
  MemoryOperationFailure,
  sameConversationInvocation,
} from "./contracts.ts";

// A local copy of contracts.ts's isExactly, which that module doesn't export.
/** Whether `value` is an instance of exactly `cls`, not of a subclass. */
function isExactly<T>(value: unknown, cls: abstract new (...args: never[]) => T): value is T {
  return typeof value === "object" && value !== null && Object.getPrototypeOf(value) === cls.prototype;
}

/** Return `value` when it's exactly a `ConversationView` (Python's `_copy_conversation_view`). */
function requireView(value: unknown): ConversationView {
  if (!isExactly(value, ConversationView)) throw new TypeError("baseline must be a ConversationView");
  return value;
}

/** Return `value` when it's exactly one checkpoint (Python's `_copy_conversation_checkpoint`). */
function requireCheckpoint(value: unknown): ConversationCheckpoint {
  if (
    isExactly(value, ConversationAdmitted) ||
    isExactly(value, ConversationEffectRequested) ||
    isExactly(value, ConversationEffectObserved) ||
    isExactly(value, ConversationClosed)
  ) {
    return value;
  }
  throw new TypeError("checkpoints must contain exact ConversationCheckpoint values");
}

/** Return the lowercase SHA-256 hex digest of `value`'s `logJson` text. */
function digest(value: unknown): string {
  return crypto.createHash("sha256").update(logJson(value)).digest("hex");
}

function invocationJson(value: ConversationInvocation): JsonObject {
  return { run_id: value.runId, start_sequence: BigInt(value.startSequence) };
}

function itemJson(value: ConversationItem): JsonObject {
  return { item_id: value.itemId, format: value.format, payload: value.payload };
}

function failureJson(value: MemoryOperationFailure | null): JsonObject | null {
  if (value === null) return null;
  return { stage: value.stage, code: value.code, message: value.message, retryable: value.retryable };
}

function effectJson(value: ConversationEffectState): JsonObject {
  return {
    invocation: invocationJson(value.invocation),
    effect_ordinal: BigInt(value.effectOrdinal),
    kind: value.kind,
    request: value.request,
    status: value.status,
    observation_ordinals: value.observationOrdinals.map(BigInt),
    failure: failureJson(value.failure),
  };
}

function completionJson(value: ConversationCompletion): JsonObject {
  return {
    invocation: invocationJson(value.invocation),
    closed_ordinal: BigInt(value.closedOrdinal),
    status: value.status,
  };
}

function viewJson(value: ConversationView, ledger = true): JsonObject {
  const result: Record<string, JsonValue> = {
    namespace: value.namespace,
    generation: BigInt(value.generation),
    revision: BigInt(value.revision),
    compatibility: value.compatibility,
    items: value.items.map(itemJson),
    // Retain this inert field so message-only snapshot digests stay valid.
    reference: null,
    effects: value.effects.map(effectJson),
    closed_invocations: value.closedInvocations.map(completionJson),
  };
  if (ledger) {
    result.applied = value.applied.map((entry) => ({
      operation_id: entry.operationId,
      event_digest: entry.eventDigest,
      before_digest: entry.beforeDigest,
      after_digest: entry.afterDigest,
    }));
  }
  return result;
}

/** Return a detached frozen copy of the complete wire, at any depth, under the value number rule. */
function wire(value: unknown): JsonObject {
  return freezeJsonObject(value, "conversation wire", { limitNesting: false, valueNumbers: true });
}

/** Return a detached frozen copy of one view wire. */
function viewWire(value: unknown): JsonObject {
  const proto: unknown = typeof value === "object" && value !== null ? Object.getPrototypeOf(value) : undefined;
  if (proto !== Object.prototype && proto !== null) {
    throw new TypeError("conversation view wire must be an exact JSON object");
  }
  return wire(value);
}

/**
 * Return `value` when it's a wire record with exactly the space-separated `names`.
 *
 * @throws TypeError when `value` isn't a JSON object.
 * @throws RangeError when its keys aren't exactly `names`.
 */
function object(value: unknown, names: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("conversation wire record must be an exact JSON object");
  }
  const keys = Object.keys(value);
  const expected = names.split(" ");
  if (keys.length !== expected.length || !expected.every((name) => Object.hasOwn(value, name))) {
    throw new RangeError(`conversation wire record requires exactly: ${names}`);
  }
  return value as JsonObject;
}

function array(value: unknown): readonly JsonValue[] {
  if (!Array.isArray(value)) throw new TypeError("conversation wire array must be an exact list");
  return value;
}

/**
 * Return one wire counter as a number; a carrier then checks its range.
 *
 * @throws TypeError when `value` isn't a `bigint` (a JSON int).
 * @throws RangeError when it's beyond the safe integer range.
 */
function counter(value: unknown, name: string): number {
  // Same rule as runtime/log.ts's wireInt, which this leaf can't import.
  if (typeof value !== "bigint") throw new TypeError(`${name} must be an integer`);
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new RangeError(`${name} must be a safe integer`);
  return number;
}

function invocationFromJson(value: unknown): ConversationInvocation {
  const data = object(value, "run_id start_sequence");
  return new ConversationInvocation(data.run_id as string, counter(data.start_sequence, "start_sequence"));
}

function itemsFromJson(value: unknown): readonly ConversationItem[] {
  return array(value).map((item) => {
    const data = object(item, "item_id format payload");
    return new ConversationItem(data.item_id as string, data.format as string, data.payload);
  });
}

/** Reject retired wire state without adopting or rewriting its history. */
function rejectNativeHistory(data: JsonObject): void {
  if ((data.reference !== undefined && data.reference !== null) || data.kind === "native_session") {
    throw new MemoryOperationError(
      "snapshot",
      "session.unsupported_history",
      "Native session history is unsupported; use fresh SessionMemory.",
      false,
    );
  }
}

function failureFromJson(value: unknown): MemoryOperationFailure | null {
  if (value === null) return null;
  const data = object(value, "stage code message retryable");
  return new MemoryOperationFailure(
    data.stage as string,
    data.code as string,
    data.message as string,
    data.retryable as boolean,
  );
}

function effectFromJson(value: unknown): ConversationEffectState {
  const data = object(value, "invocation effect_ordinal kind request status observation_ordinals failure");
  rejectNativeHistory(data);
  return new ConversationEffectState(
    invocationFromJson(data.invocation),
    counter(data.effect_ordinal, "effect_ordinal"),
    data.kind as ConversationEffectState["kind"],
    data.request,
    data.status as ConversationEffectState["status"],
    array(data.observation_ordinals).map((ordinal) => counter(ordinal, "observation ordinal")),
    failureFromJson(data.failure),
  );
}

function completionFromJson(value: unknown): ConversationCompletion {
  const data = object(value, "invocation closed_ordinal status");
  return new ConversationCompletion(
    invocationFromJson(data.invocation),
    counter(data.closed_ordinal, "closed_ordinal"),
    data.status as ConversationCompletion["status"],
  );
}

/**
 * Encode a fully validated conversation as detached exact JSON.
 *
 * @param view - Exact view with complete valid ledger evidence.
 * @returns The frozen closed wire object, counters as `bigint`s, payloads at any depth.
 * @throws TypeError when a carrier isn't exactly its class.
 * @throws RangeError when history or digest evidence is invalid.
 * @example `conversationViewFromJson(conversationViewToJson(view))` equals `view`.
 */
export function conversationViewToJson(view: ConversationView): JsonObject {
  requireView(view);
  validateView(view);
  return wire(viewJson(view));
}

/**
 * Decode exact JSON and validate all retained conversation evidence.
 *
 * @param value - Complete closed view wire, as `parseJson` reads it (ints are `bigint`s).
 * @returns The validated frozen conversation state.
 * @throws TypeError when a wire value has the wrong type.
 * @throws RangeError when fields, identity, history or digest evidence is invalid.
 * @throws MemoryOperationError when retired provider-native history requires fresh memory.
 * @example `const view = conversationViewFromJson(savedView)` validates before use.
 */
export function conversationViewFromJson(value: JsonObject): ConversationView {
  const data = object(
    viewWire(value),
    "namespace generation revision compatibility items reference effects closed_invocations applied",
  );
  rejectNativeHistory(data);
  const view = new ConversationView(
    data.namespace as string,
    counter(data.generation, "generation"),
    counter(data.revision, "revision"),
    data.compatibility,
    itemsFromJson(data.items),
    array(data.effects).map(effectFromJson),
    array(data.closed_invocations).map(completionFromJson),
    array(data.applied).map((item) => {
      const entry = object(item, "operation_id event_digest before_digest after_digest");
      return new ConversationApplied(
        entry.operation_id as string,
        entry.event_digest as string,
        entry.before_digest as string,
        entry.after_digest as string,
      );
    }),
  );
  validateView(view);
  return view;
}

/**
 * Hash complete conversation content except its application ledger.
 *
 * @param view - Exact shaped carrier, including its revision; the ledger's digest evidence isn't
 *   interpreted.
 * @returns The lowercase SHA-256 of the `logJson` text without the `applied` field.
 * @throws TypeError when `view` isn't exactly a `ConversationView`.
 * @example `conversationContentDigest(view)` is unchanged when only a ledger digest changes.
 */
export function conversationContentDigest(view: ConversationView): string {
  return digest(viewJson(requireView(view), false));
}

function stampJson(value: ConversationStamp): JsonObject {
  return {
    invocation: invocationJson(value.invocation),
    namespace: value.namespace,
    generation: BigInt(value.generation),
    expected_revision: BigInt(value.expectedRevision),
    ordinal: BigInt(value.ordinal),
  };
}

function checkpointJson(value: ConversationCheckpoint): JsonObject {
  if (value instanceof ConversationAdmitted) {
    validateView(value.baseline);
    return { type: "admitted", stamp: stampJson(value.stamp), baseline: viewJson(value.baseline) };
  }
  if (value instanceof ConversationEffectRequested) {
    return {
      type: "effect_requested",
      stamp: stampJson(value.stamp),
      kind: value.kind,
      payload: value.payload,
      items: value.items.map(itemJson),
    };
  }
  if (value instanceof ConversationEffectObserved) {
    return {
      type: "effect_observed",
      stamp: stampJson(value.stamp),
      effect_ordinal: BigInt(value.effectOrdinal),
      status: value.status,
      items: value.items.map(itemJson),
      // The original checkpoint digest includes this inert wire field.
      reference: null,
      failure: failureJson(value.failure),
    };
  }
  return { type: "closed", stamp: stampJson(value.stamp), status: value.status };
}

function stampFromJson(value: unknown): ConversationStamp {
  const data = object(value, "invocation namespace generation expected_revision ordinal");
  return new ConversationStamp(
    invocationFromJson(data.invocation),
    data.namespace as string,
    counter(data.generation, "generation"),
    counter(data.expected_revision, "expected_revision"),
    counter(data.ordinal, "ordinal"),
  );
}

/**
 * Encode a checkpoint with its closed discriminator and complete payload.
 *
 * @param checkpoint - One exact checkpoint carrier.
 * @returns The frozen closed wire; an admission includes its complete baseline.
 * @throws TypeError when `checkpoint` isn't exactly a checkpoint carrier.
 * @throws RangeError when an admission's baseline evidence is invalid.
 * @example `const wire = conversationCheckpointToJson(admission)` retains baseline evidence.
 */
export function conversationCheckpointToJson(checkpoint: ConversationCheckpoint): JsonObject {
  return wire(checkpointJson(requireCheckpoint(checkpoint)));
}

const CHECKPOINT_NAMES: ReadonlyMap<string, string> = new Map([
  ["admitted", "type stamp baseline"],
  ["effect_requested", "type stamp kind payload items"],
  ["effect_observed", "type stamp effect_ordinal status items reference failure"],
  ["closed", "type stamp status"],
]);

/**
 * Decode one strict member of the closed conversation checkpoint vocabulary.
 *
 * @param value - Exact checkpoint wire with a supported `type` field.
 * @returns The frozen checkpoint carrier.
 * @throws TypeError when a wire value has the wrong type.
 * @throws RangeError when fields, the discriminator, shape or baseline evidence is invalid.
 * @throws MemoryOperationError when retired provider-native history requires fresh memory.
 * @example `const checkpoint = conversationCheckpointFromJson(savedCheckpoint)`.
 */
export function conversationCheckpointFromJson(value: JsonObject): ConversationCheckpoint {
  const data = wire(value);
  const kind = data.type;
  if (typeof kind !== "string") throw new TypeError("checkpoint type must be a string");
  if (!kind) throw new RangeError("checkpoint type must be non-empty");
  const names = CHECKPOINT_NAMES.get(kind);
  if (names === undefined) throw new RangeError("unsupported conversation checkpoint type");
  object(data, names);
  rejectNativeHistory(data);
  const stamp = stampFromJson(data.stamp);
  if (kind === "admitted") {
    return new ConversationAdmitted(stamp, conversationViewFromJson(data.baseline as JsonObject));
  }
  if (kind === "effect_requested") {
    return new ConversationEffectRequested(
      stamp,
      data.kind as ConversationEffectRequested["kind"],
      data.payload,
      itemsFromJson(data.items),
    );
  }
  if (kind === "effect_observed") {
    return new ConversationEffectObserved(
      stamp,
      counter(data.effect_ordinal, "effect_ordinal"),
      data.status as ConversationEffectObserved["status"],
      itemsFromJson(data.items),
      failureFromJson(data.failure),
    );
  }
  return new ConversationClosed(stamp, data.status as ConversationClosed["status"]);
}

/**
 * Hash one complete checkpoint, including an admission's complete baseline.
 *
 * @param checkpoint - One exact closed checkpoint carrier.
 * @returns The lowercase SHA-256 of the checkpoint wire's `logJson` text.
 * @throws TypeError when `checkpoint` isn't exactly a checkpoint carrier.
 * @throws RangeError when retained baseline evidence is invalid.
 * @example `checkpointDigest(admission)` includes the baseline's application ledger.
 */
export function checkpointDigest(checkpoint: ConversationCheckpoint): string {
  return digest(conversationCheckpointToJson(checkpoint));
}

function zeroView(view: ConversationView): ConversationView {
  return new ConversationView(view.namespace, view.generation, 0, view.compatibility, [], [], [], []);
}

/** Return an entry's invocation and ordinal, from its canonical `[run_id, start_sequence, ordinal]`. */
function operation(entry: ConversationApplied): [ConversationInvocation, number] {
  const [runId, startSequence, ordinal] = JSON.parse(entry.operationId) as [string, number, number];
  return [new ConversationInvocation(runId, startSequence), ordinal];
}

// Python keys its dicts by invocation and ordinal tuples; JSON text gives equal values equal keys,
// and an operation key is its canonical `operationId`.
function invocationKey(invocation: ConversationInvocation): string {
  return JSON.stringify([invocation.runId, invocation.startSequence]);
}

function operationKey(invocation: ConversationInvocation, ordinal: number): string {
  return JSON.stringify([invocation.runId, invocation.startSequence, ordinal]);
}

/** Prove the supplied ledger's ordering, anchor, adjacency and final content. */
function validateView(view: ConversationView): void {
  let expectedDigest = conversationContentDigest(zeroView(view));
  let previousInvocation: ConversationInvocation | null = null;
  let previousOrdinal = -1;
  const seen = new Set<string>();
  const operations = new Map<string, number>();
  const positive: string[] = [];
  const lastOrdinals = new Map<string, number>();
  for (const [index, entry] of view.applied.entries()) {
    const [invocation, ordinal] = operation(entry);
    if (previousInvocation === null || !sameConversationInvocation(invocation, previousInvocation)) {
      if (seen.has(invocationKey(invocation)) || ordinal !== 0) {
        throw new RangeError("ledger must have contiguous admission-led invocations");
      }
      seen.add(invocationKey(invocation));
    } else if (ordinal !== previousOrdinal + 1) {
      throw new RangeError("ledger operation ordinals must be contiguous");
    }
    if (entry.beforeDigest !== expectedDigest) {
      throw new RangeError("ledger digest chain does not match its predecessor");
    }
    expectedDigest = entry.afterDigest;
    previousInvocation = invocation;
    previousOrdinal = ordinal;
    operations.set(entry.operationId, index);
    if (ordinal > 0) positive.push(entry.operationId);
    lastOrdinals.set(invocationKey(invocation), ordinal);
  }
  if (expectedDigest !== conversationContentDigest(view)) {
    throw new RangeError("ledger final digest does not match conversation content");
  }
  validateHistory(view, operations, positive, lastOrdinals);
}

/** Account for every non-admission operation using retained actual evidence. */
function validateHistory(
  view: ConversationView,
  operations: ReadonlyMap<string, number>,
  positive: readonly string[],
  lastOrdinals: ReadonlyMap<string, number>,
): void {
  const claimed = new Set<string>();

  const claim = (invocation: ConversationInvocation, ordinal: number): number => {
    const identity = operationKey(invocation, ordinal);
    const position = operations.get(identity);
    if (position === undefined || claimed.has(identity)) {
      throw new RangeError("history must name unique accepted operation ordinals");
    }
    claimed.add(identity);
    return position;
  };

  const effectsByInvocation = new Map<string, ConversationEffectState[]>();
  let previous = -1;
  for (const effect of view.effects) {
    const position = claim(effect.invocation, effect.effectOrdinal);
    if (position <= previous) throw new RangeError("effects must retain accepted request order");
    previous = position;
    const key = invocationKey(effect.invocation);
    const list = effectsByInvocation.get(key);
    if (list) list.push(effect);
    else effectsByInvocation.set(key, [effect]);
    for (const ordinal of effect.observationOrdinals) claim(effect.invocation, ordinal);
  }

  previous = -1;
  for (const completion of view.closedInvocations) {
    const position = claim(completion.invocation, completion.closedOrdinal);
    const key = invocationKey(completion.invocation);
    if (position <= previous || lastOrdinals.get(key) !== completion.closedOrdinal) {
      throw new RangeError("closure must be the invocation's last accepted operation");
    }
    previous = position;
    if (
      completion.status === "completed" &&
      (effectsByInvocation.get(key) ?? []).some((effect) => effect.status === "pending")
    ) {
      throw new RangeError("successful closure cannot retain pending effects");
    }
  }

  if (claimed.size !== positive.length || !positive.every((identity) => claimed.has(identity))) {
    throw new RangeError("every positive operation must have retained effect or closure evidence");
  }
  if (view.items.length > 0 && view.effects.length === 0) {
    throw new RangeError("retained items require an accepted effect request");
  }
}

function conflict(message: string): MemoryOperationError {
  return new MemoryOperationError("conversation", "session.conflict", message, false);
}

/**
 * Apply exactly one new checkpoint; both inputs are frozen and stay unchanged.
 *
 * @param before - Complete validated predecessor content and ledger.
 * @param checkpoint - New checkpoint naming that predecessor.
 * @returns The successor with one new revision and ledger entry.
 * @throws TypeError when an input isn't exactly its carrier class.
 * @throws RangeError when an input's retained history evidence is invalid.
 * @throws MemoryOperationError (`session.conflict`) when the operation conflicts with its
 *   predecessor.
 * @example `const after = advanceConversation(before, admission)` advances the revision once;
 *   folding the same admission again is a conflict, not another revision.
 */
export function advanceConversation(before: ConversationView, checkpoint: ConversationCheckpoint): ConversationView {
  requireView(before);
  requireCheckpoint(checkpoint);
  validateView(before);
  const stamp = checkpoint.stamp;
  if (
    stamp.namespace !== before.namespace ||
    stamp.generation !== before.generation ||
    stamp.expectedRevision !== before.revision
  ) {
    throw conflict("checkpoint does not name the current predecessor");
  }
  const operationId = operationKey(stamp.invocation, stamp.ordinal);
  if (before.applied.some((entry) => entry.operationId === operationId)) {
    throw conflict("checkpoint operation has already been applied");
  }
  let items = before.items;
  let effects = before.effects;
  let completions = before.closedInvocations;
  if (checkpoint instanceof ConversationAdmitted) {
    validateView(checkpoint.baseline);
    if (logJson(viewJson(checkpoint.baseline)) !== logJson(viewJson(before))) {
      throw conflict("admission baseline does not match the complete predecessor");
    }
    if (before.applied.some((entry) => sameConversationInvocation(operation(entry)[0], stamp.invocation))) {
      throw conflict("invocation was already admitted");
    }
  } else {
    if (before.applied.length === 0) throw conflict("checkpoint requires an admitted invocation");
    const [invocation, ordinal] = operation(before.applied.at(-1)!);
    if (!sameConversationInvocation(invocation, stamp.invocation) || stamp.ordinal !== ordinal + 1) {
      throw conflict("checkpoint must be the current invocation's next ordinal");
    }
    if (completions.some((value) => sameConversationInvocation(value.invocation, invocation))) {
      throw conflict("invocation is already closed");
    }
    if (checkpoint instanceof ConversationEffectRequested || checkpoint instanceof ConversationEffectObserved) {
      const existingIds = new Set(items.map((item) => item.itemId));
      if (checkpoint.items.some((item) => existingIds.has(item.itemId))) {
        throw conflict("checkpoint repeats an existing conversation item ID");
      }
      items = [...items, ...checkpoint.items];
    }
    if (checkpoint instanceof ConversationEffectRequested) {
      effects = [
        ...effects,
        new ConversationEffectState(invocation, stamp.ordinal, checkpoint.kind, checkpoint.payload, "pending", [], null),
      ];
    } else if (checkpoint instanceof ConversationEffectObserved) {
      const index = effects.findIndex(
        (effect) =>
          sameConversationInvocation(effect.invocation, invocation) && effect.effectOrdinal === checkpoint.effectOrdinal,
      );
      if (index === -1) throw conflict("observation has no matching requested effect");
      const effect = effects[index]!;
      if (effect.status !== "pending") throw conflict("a terminal effect cannot receive more observations");
      const observed = new ConversationEffectState(
        invocation,
        effect.effectOrdinal,
        effect.kind,
        effect.request,
        checkpoint.status,
        [...effect.observationOrdinals, stamp.ordinal],
        checkpoint.failure,
      );
      effects = effects.with(index, observed);
    } else {
      if (
        checkpoint.status === "completed" &&
        effects.some((effect) => sameConversationInvocation(effect.invocation, invocation) && effect.status === "pending")
      ) {
        throw conflict("successful closure cannot leave pending effects");
      }
      completions = [...completions, new ConversationCompletion(invocation, stamp.ordinal, checkpoint.status)];
    }
  }
  const entry = new ConversationApplied(
    operationId,
    checkpointDigest(checkpoint),
    conversationContentDigest(before),
    "0".repeat(64),
  );
  const after = new ConversationView(
    before.namespace,
    before.generation,
    before.revision + 1,
    before.compatibility,
    items,
    effects,
    completions,
    [...before.applied, entry],
  );
  const sealed = new ConversationApplied(
    entry.operationId,
    entry.eventDigest,
    entry.beforeDigest,
    conversationContentDigest(after),
  );
  const result = new ConversationView(
    after.namespace,
    after.generation,
    after.revision,
    after.compatibility,
    after.items,
    after.effects,
    after.closedInvocations,
    [...before.applied, sealed],
  );
  // Prove the successor encodes, as Python does, before returning it.
  conversationViewToJson(result);
  return result;
}
