/** A spliced child END's re-namespaced post-assert fires record-scoped, through `evalNode`. */

import { expect, test } from "bun:test";

import { NodeKind, Output } from "../../src/nodes/base.ts";
import { Assertion, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { EvaluationFailure, type WorkerOutcome } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { drive, nodeFlow } from "../helpers/run_state_fixtures.ts";

/**
 * A spliced child END whose post-assert reads the synth param `__assert_tail`, bound into the
 * record from `${each/tail.output}`.
 */
function endWithTailAssert(): EndNode {
  const end = EndNode.value("each/__end__", { outputName: "out" });
  end.postAsserts = [new Assertion("__assert_tail != ''", "tail must not be empty")];
  end.params = [...end.params!, new Parameter("__assert_tail")];
  return end;
}

/** Evaluate `end` wired from `${each/tail.output}` against `pool`. */
function run(end: EndNode, pool: VariablePool): Promise<WorkerOutcome> {
  const flow = nodeFlow(end, { out: "${each/tail.output}", __assert_tail: "${each/tail.output:-null}" });
  return drive(end, { pool, flow });
}

test("child end record scoped post assert fires via synth param", async () => {
  // The ref is lowered to a synth param bound into the record from the pool; nothing reads the
  // pool around the record.
  expect(endWithTailAssert().kind).toBe(NodeKind.END);
  const pool = new VariablePool();

  pool.set("each/tail", "ok");
  expect(await run(endWithTailAssert(), pool)).toBeInstanceOf(Output);

  pool.set("each/tail", "");
  const failed = await run(endWithTailAssert(), pool);
  expect(failed).toBeInstanceOf(EvaluationFailure);
  expect((failed as EvaluationFailure).failure.message).toContain("post-assert");
});
