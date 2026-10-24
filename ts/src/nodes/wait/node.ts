/**
 * WAIT: block one worker until an absolute UTC deadline.
 *
 * WAIT accepts one resolved date or datetime string (the grammar in `typesys/temporal`) through
 * its reserved `until` wiring. A date and a datetime without an offset are read in UTC; a
 * datetime with an offset is normalized to UTC. The author source stays in flow wiring, and the
 * node owns no run-log or scheduler state. The node sleeps on its scheduler with its signal, so a
 * run stop (engine failure or host shutdown) wakes it.
 *
 * Knows about:   nodes/base, nodes/binding, nodes/contracts, diagnostics, platform and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { reprText } from "../../diagnostics.ts";
import { realScheduler, SLEEP_CHUNK_MS } from "../../platform.ts";
import type { VariablePool } from "../../typesys/pool.ts";
import { deadlineUtc } from "../../typesys/temporal.ts";
import { valueRepr } from "../../typesys/values.ts";
import { Node, type NodeCaps, NodeKind, Output } from "../base.ts";
import { BindingError, type ReservedBinder, resolveSource, sourceOrigin } from "../binding.ts";
import { bindingSource } from "../contracts.ts";

/**
 * Resolve a raw or carrier-wrapped WAIT source to a date/datetime string.
 *
 * Exactly one semantic `Binding` layer is removed before a `${...}` expression or literal is
 * evaluated against `pool`. The resolved string is checked as an absolute deadline against the
 * `typesys/temporal` grammar; the WAIT body reparses it only to compute the duration. An optional
 * `bind` capability retains the normalized value at the read boundary.
 *
 * @throws BindingError classified `contract.input` when the source resolves to a non-string, a
 *   string outside the grammar, an impossible date or datetime, or a deadline whose UTC year is
 *   outside 1 to 9999; other binding failures as `resolveSource` throws them.
 */
export function resolveUntil(
  untilSrc: unknown,
  pool: VariablePool,
  options: { bind?: ReservedBinder | null } = {},
): string {
  const { bind = null } = options;
  if (bind !== null) return bind("until", untilSrc, (value) => normalizeUntil(untilSrc, value)) as string;
  return normalizeUntil(untilSrc, resolveSource(untilSrc, pool));
}

/** Validate one resolved WAIT value and return the string delivered to it. */
function normalizeUntil(untilSrc: unknown, value: unknown): string {
  const origin = sourceOrigin(untilSrc);
  if (typeof value !== "string") {
    const source = bindingSource(untilSrc);
    const shown = typeof source === "string" ? reprText(source) : valueRepr(source);
    throw new BindingError(
      `wait \`until\` (${shown}) did not resolve to a date/datetime: ${valueRepr(value)}`,
      { failureCode: "contract.input", origin },
    );
  }
  try {
    deadlineUtc(value);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    // The resolved deadline is a run value, so the message writes it in the text form.
    throw new BindingError(`wait \`until\` ${valueRepr(value)}: ${error.message}`, {
      failureCode: "contract.input",
      origin,
      cause: error,
    });
  }
  return value;
}

/**
 * Block until the resolved absolute `until` deadline.
 *
 * The `until` source lives in the flow wiring under `until` and is pre-resolved by
 * `bindReserved`; `forward` receives that concrete date/datetime string and returns `Output(null)`
 * once the deadline is reached. A stop rejects with `caps.signal.reason`, so the engine discards
 * the outcome while draining.
 */
export class WaitNode extends Node {
  readonly kind = NodeKind.WAIT;

  /**
   * Build a WAIT node.
   *
   * @param nodeId The node's unique id.
   * @param options.title A display title.
   */
  constructor(nodeId: string, options: { title?: string | null } = {}) {
    const { title = null } = options;
    super(nodeId, { title });
  }

  /**
   * Resolve the required `until` wiring into the node's input record: `{until: <deadline>}`.
   *
   * @throws Error when the wiring has no `until` source.
   * @throws BindingError as `resolveUntil` throws it.
   */
  override bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    const { bind = null } = options;
    if (!nodeWiring.has("until")) throw new Error(`WAIT node ${reprText(this.id)} has no \`until\` wiring`);
    return { until: resolveUntil(nodeWiring.get("until"), pool, { bind }) };
  }

  /** A WAIT reserves `until`, its required author-wiring key. */
  override reservedWiringKeys(): ReadonlySet<string> {
    return new Set(["until"]);
  }

  /**
   * Wait for the remaining absolute duration on `caps.scheduler` (`realScheduler` when absent).
   *
   * A deadline at or before now returns at once, without checking the signal.
   *
   * @throws The reason of `caps.signal` once it aborts during the wait.
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<Output> {
    const scheduler = caps.scheduler ?? realScheduler;
    const deadline = deadlineUtc(inputs.until as string);
    // Re-read the wall clock after each chunk, so the deadline survives a sleeping laptop.
    for (let remaining = deadline - scheduler.now(); remaining > 0; remaining = deadline - scheduler.now()) {
      await scheduler.sleep(Math.min(remaining, SLEEP_CHUNK_MS), caps.signal);
    }
    return new Output(null);
  }
}
