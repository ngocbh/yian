/**
 * Memory proposals shared by runtime consumers.
 *
 * Knows about:   memory/contracts.
 * Never imports: runtime log, state, engine or run, compile, compose, or the CLI.
 */

import { type MemoryMutation, validateMutations } from "../memory/contracts.ts";

/** The ordered mutations one memory proposed in a backward attempt. Frozen. */
export class MemoryProposal {
  /** The recorded memory key. */
  readonly memory: string;
  /** Mutations with unique ids, in proposal order, kept by reference; an empty batch is valid. */
  readonly mutations: readonly MemoryMutation[];

  /**
   * Validate the key and keep a frozen copy of the mutation batch.
   *
   * @throws TypeError when `memory` isn't a string, `mutations` isn't an array, or an item isn't
   *   exactly a `MemoryMutation`.
   * @throws RangeError for an empty `memory` or two mutations with one id.
   */
  constructor(memory: string, mutations: readonly MemoryMutation[]) {
    if (typeof memory !== "string") throw new TypeError("memory must be a string");
    if (!memory) throw new RangeError("memory must be non-empty");
    this.memory = memory;
    this.mutations = validateMutations(mutations);
    Object.freeze(this);
  }
}
