/**
 * Backward proposal and record contracts: a port of `tests/engine/test_backward_records.py`.
 * Ported: `test_memory_proposal_owns_detached_mutations_in_supplied_order` and the
 * `MutationSubclass` row of `test_backward_records_reject_subclasses_of_exact_nested_carriers`.
 *
 * Changed: a `MemoryProposal` keeps the frozen mutations it was given, where Python copies them:
 * the test asserts the same mutation (`is not` becomes `toBe`), and a reader's payload edit throws
 * `TypeError` where Python changes a private copy. A mutation batch is an array, so the rejected
 * container is a `Set` (Python passes a list); assigning a field throws `TypeError` where Python
 * raises `FrozenInstanceError`, and Python's `ValueError` is `RangeError`.
 *
 * Later (T4-2): every record test, with the runtime log and state: the other rows of
 * `test_backward_records_reject_subclasses_of_exact_nested_carriers`, the record constructors,
 * commit attempt ids, the wire round trips and rejections, the codec revalidation, and the
 * prepare, append and replay tests.
 */

import { expect, test } from "bun:test";

import { MemoryMutation } from "../../src/memory/contracts.ts";
import { MemoryProposal } from "../../src/runtime/contracts.ts";

test("memory proposal owns detached mutations in supplied order", () => {
  const payload = { values: [1n] };
  const mutations = [new MemoryMutation("second", payload), new MemoryMutation("first", {})];
  const proposal = new MemoryProposal("agent", mutations);
  payload.values.push(2n);
  expect(() => {
    ((proposal.mutations[0]!.payload as { values: bigint[] }).values as bigint[]).length = 0;
  }).toThrow(TypeError);
  expect(proposal.mutations.map((mutation) => mutation.mutationId)).toEqual(["second", "first"]);
  expect(proposal.mutations[0]!.payload).toEqual({ values: [1n] });
  expect(proposal.mutations[0]).toBe(mutations[0]!);
  expect(proposal.memory).toBe("agent");
  expect(Object.isFrozen(proposal)).toBe(true);
  expect(Object.isFrozen(proposal.mutations)).toBe(true);
  expect(() => {
    (proposal as { mutations: readonly MemoryMutation[] }).mutations = [];
  }).toThrow(TypeError);
  expect(() => new MemoryProposal("agent", new Set(mutations) as never)).toThrow(TypeError);
  expect(() => new MemoryProposal("agent", [mutations[0]!, mutations[0]!])).toThrow(RangeError);
  expect(() => new MemoryProposal("agent", [mutations[0]!, mutations[0]!])).toThrow(/unique/);
  expect(new MemoryProposal("agent", []).mutations).toEqual([]);
});

// test_backward_records_reject_subclasses_of_exact_nested_carriers (the MutationSubclass row)
test("a memory proposal rejects a MemoryMutation subclass", () => {
  class MutationSubclass extends MemoryMutation {}
  expect(() => new MemoryProposal("agent", [new MutationSubclass("m1", {})])).toThrow(TypeError);
});
