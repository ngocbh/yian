/**
 * Pure backward transfers and accepted backward/memory reducer projections.
 *
 * This module shares exact route-transfer validation with the backward callback driver and owns
 * immutable attempt state. It projects owner-local contexts, the only episode projection, for
 * every enabled memory key from complete accepted history and backward results. Contracts,
 * records, and retained traces sit below it; state and callback drivers consume its pure
 * projections. It invokes no callbacks or backend operations.
 *
 * Knows about:   diagnostics, expr/lineage, memory, nodes/base, runtime/contracts, runtime/log,
 *                runtime/trace, typesys.
 * Never imports: runtime state or backward, engine or run, compile, compose, or the CLI.
 */

import { createHash } from "node:crypto";

import { Failure } from "../diagnostics.ts";
import { MemoryEpisodeStatus, MemoryMutation } from "../memory/index.ts";
import { isExactly } from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson } from "../typesys/json.ts";
import { isList, isRecord } from "../typesys/values.ts";
import { canonicalSeed, NodeExpandedRecord, NodeFailedRecord, NodeRoutedRecord, NodeSucceededRecord } from "./log.ts";
import type { OutcomeRecord } from "./trace.ts";

/**
 * Project one canonical durable `Failure` into source-neutral JSON.
 *
 * @throws TypeError when `failure` isn't exactly a `Failure`, or a field has the wrong type.
 * @throws RangeError when its code is unknown.
 */
export function failureJson(failure: Failure): JsonObject {
  if (!isExactly(failure, Failure)) throw new TypeError("failure projection requires a Failure");
  // A Failure isn't frozen, so its fields are checked again before they enter durable JSON.
  new Failure(failure);
  return failure.toJSON();
}

/**
 * Project the actual accepted outcome without assuming model evidence.
 *
 * Returns the episode status, the success output (`null` on failure) and the failure JSON (`null`
 * on success).
 *
 * @throws RangeError when a success has no canonical typed value.
 * @throws TypeError when `record` isn't exactly one of the outcome record classes.
 */
export function episodeOutcome(record: OutcomeRecord): [MemoryEpisodeStatus, JsonValue, JsonObject | null] {
  if (isExactly(record, NodeSucceededRecord)) {
    const wire: unknown = record.value;
    if (!isRecord(wire) || !Object.hasOwn(wire, "value")) {
      throw new RangeError("successful record has no canonical typed value");
    }
    return [MemoryEpisodeStatus.SUCCEEDED, canonicalSeed(wire.value)[1], null];
  }
  if (isExactly(record, NodeRoutedRecord)) {
    if (record.terminalFailure === null) {
      return [MemoryEpisodeStatus.SUCCEEDED, { kind: "route", handle: record.handle }, null];
    }
    return [MemoryEpisodeStatus.FAILED, null, failureJson(record.terminalFailure)];
  }
  if (isExactly(record, NodeExpandedRecord)) {
    const output = { kind: "grow", seed: canonicalSeed(record.seed)[1], prune_ids: [...record.pruneIds] };
    return [MemoryEpisodeStatus.SUCCEEDED, output, null];
  }
  if (isExactly(record, NodeFailedRecord)) return [MemoryEpisodeStatus.FAILED, null, failureJson(record.failure)];
  throw new TypeError("unsupported graph outcome for episode projection");
}

/**
 * Return the sole deterministic ID for one memory's mutation batch in one attempt.
 *
 * The SHA-256 input is the UTF-8 run-log text (`logJson`: keys in code-unit order, ints of any
 * size, floats marked by `.0`) of `[runId, attemptId, memory, [[mutationId, payload], ...]]`, the
 * attempt id written as an int. Batch order is semantic. The result is
 * `ac-memory-v3:<hex digest>`.
 *
 * @throws RangeError when `attemptId` isn't a positive safe integer, or `runId` or `memory` isn't
 *   a non-empty string.
 * @throws TypeError when `mutations` isn't an array or an item isn't exactly a `MemoryMutation`.
 */
export function memoryCommitId(
  runId: string,
  memory: string,
  mutations: readonly MemoryMutation[],
  attemptId: number,
): string {
  if (typeof attemptId !== "number" || !Number.isSafeInteger(attemptId) || attemptId < 1) {
    throw new RangeError("attempt_id must be a positive integer");
  }
  if (typeof runId !== "string" || runId === "") throw new RangeError("run_id must be a non-empty string");
  if (typeof memory !== "string" || memory === "") throw new RangeError("memory must be a non-empty string");
  if (!isList(mutations)) throw new TypeError("mutations must be an array");
  const batch = mutations.map((mutation) => {
    if (!isExactly(mutation, MemoryMutation)) throw new TypeError("mutations items must be MemoryMutation values");
    return [mutation.mutationId, mutation.payload];
  });
  // A plain number would be written as a float (`1.0`), so the attempt id is a bigint.
  const text = logJson([runId, BigInt(attemptId), memory, batch]);
  return `ac-memory-v3:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}
