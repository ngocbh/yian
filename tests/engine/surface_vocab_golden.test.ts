/**
 * The singular surface vocabulary at run time: section keywords, the node-first CASE value
 * shorthand, and an END post-assert reading `output`. Integers are bigints.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { RunFailed, RunSucceeded } from "../../src/events.ts";
import { Assertion } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { engine, FuncNode, graph } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-surface-vocab-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test("section keywords load", async () => {
  const flow = await loadFlow(`
id: f
name: f
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      v: \${input.x}
    output: str
    code: |
      return inputs["v"]
output: \${n.output}
`);

  const result = await runFlow(flow, { x: "hi" }, { env: hostEnv(dir) });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("hi");
});

test("case value shorthand node first", async () => {
  const flow = await loadFlow(`
id: f
name: f
input:
  x: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "input.x > 0"
        then: pos
    else: neg
  pos:
    kind: code
    input:
      v: \${input.x}
    output:
      tag: str
      val: int
    code: |
      return {tag: "pos", val: inputs["v"]}
  neg:
    kind: code
    input:
      v: \${input.x}
    output:
      tag: str
      val: int
    code: |
      return {tag: "neg", val: inputs["v"]}
output: \${gate.output.tag}
`);

  expect((await runFlow(flow, { x: 5 }, { env: hostEnv(dir) })).output).toBe("pos");
  expect((await runFlow(flow, { x: -5 }, { env: hostEnv(dir) })).output).toBe("neg");
});

test("END post-assert reads output", async () => {
  /** START -> n (5) -> END, END asserting `postAssert` over its value. */
  const flowWith = (postAssert: string) => {
    const end = EndNode.value(END_ID, { outputName: "result" });
    end.postAsserts = [new Assertion(postAssert, postAssert)];
    return graph([new FuncNode("n", () => 5n), end], [[START_ID, "n"]], {
      outputs: [new FlowOutput("result", "${n.output}")],
    });
  };

  // A single output is the bare value, so `output > 0` holds.
  expect((await engine(flowWith("output > 0")).run()).at(-1)).toEqual(new RunSucceeded({ output: 5n }));

  const last = (await engine(flowWith("output > 100")).run()).at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.message).toContain("post-assert failed");
});
