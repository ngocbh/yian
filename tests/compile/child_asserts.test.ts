/**
 * A def's `asserts:` enforced through the child's boundary nodes.
 *
 * At build a def's asserts are lowered onto its synthesized START/END nodes (boundary asserts
 * onto `START.postAsserts` with each `${input.X}` rewritten to `${output.X}`, post asserts onto
 * `END.postAsserts`). When the def runs through a CALL (once) or a MAP (per element), the cloned
 * child START/END fire them like any node's post asserts, so a failed child assert fails the
 * child node and therefore the parent run, exactly as at the top-level boundaries.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { RunFailed } from "../../src/events.ts";
import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { engine } from "../helpers/run_state_fixtures.ts";

// `checked` def: doubles its int input; boundary asserts n>=0, post asserts the output < 100.
const CHECKED_DEF = `
defs:
  checked:
    input:
      n: !int
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:double
        input:
          n: \${input.n}
        output: int
    output: \${x.output}
    asserts:
      - "input.n >= 0"
      - "x.output < 100"
`;

const CALL_FLOW = `
id: d
name: d
input:
  v: !int
${CHECKED_DEF}
nodes:
  call_it:
    kind: call
    call: checked
    input:
      n: \${input.v}
output: \${call_it.output}
`;

const MAP_FLOW = `
id: dm
name: dm
input:
  vs: !list(int)
${CHECKED_DEF}
nodes:
  map_it:
    kind: map
    call: checked
    over: \${input.vs}
    input:
      n: \${item}
output: \${map_it.output}
`;

// --- CALL: non-vacuous boundary + post pairs, run through expansion. The same parent flow
// succeeds with a satisfying value and fails with a violating one (the cloned child START/END
// own the boundary/post asserts). //
test("call child boundary assert satisfying value succeeds", async () => {
  const out = await runFlow(await loadConformanceFlow(CALL_FLOW), { v: 5n }); // 5 >= 0 -> boundary holds
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(10n);
});

test("call child boundary assert violating value fails", async () => {
  const res = await runFlow(await loadConformanceFlow(CALL_FLOW), { v: -1n }); // -1 violates input.n >= 0
  expect(res.status).toBe("failed");
  expect(res.failure).not.toBeNull();
  expect(res.failure!.message).toContain("assert failed");
});

test("call child post assert satisfying value succeeds", async () => {
  const out = await runFlow(await loadConformanceFlow(CALL_FLOW), { v: 5n }); // x=10 < 100 -> post holds
  expect(out.status).toBe("succeeded");
});

test("call child post assert violating value fails", async () => {
  const res = await runFlow(await loadConformanceFlow(CALL_FLOW), { v: 60n }); // x=120 violates x.output < 100
  expect(res.status).toBe("failed"); // the child END's pool-scoped post assert
  expect(res.failure).not.toBeNull();
  expect(res.failure!.message).toContain("assert failed");
});

// --- MAP: the boundary asserts fire per element, so the same flow succeeds when every element
// satisfies them. //
test("map child asserts all elements satisfy succeeds", async () => {
  const res = await runFlow(await loadConformanceFlow(MAP_FLOW), { vs: [1n, 2n, 3n] }); // every element >= 0
  expect(res.status).toBe("succeeded");
  expect(res.output).toEqual([2n, 4n, 6n]); // over-order join
});

// --- A failing child boundary assert fails the run through the cloned child START: an ordinary
// node failure on the child START, after the subgraph is spliced. Driven through the engine with
// `inputs` (the parent flow declares `v`/`vs`), so the top-level START is an ordinary scheduled
// node. //
for (const numWorkers of [0, 4]) {
  test(`call boundary failure fails run (numWorkers ${numWorkers})`, async () => {
    const loaded = await loadConformanceFlow(CALL_FLOW);
    // -1 violates n >= 0 on the cloned child START.
    const eng = engine(loaded.compiled, { inputs: { v: -1n }, numWorkers, growBuilder: buildGrow });
    const events = await eng.run();
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    // Fired by the cloned child START's post asserts.
    expect((events.at(-1) as RunFailed).failure.message).toContain("post-assert failed");
  });
}

for (const numWorkers of [0, 4]) {
  test(`map boundary failure fails run (numWorkers ${numWorkers})`, async () => {
    const loaded = await loadConformanceFlow(MAP_FLOW);
    // Element 1 violates n >= 0 on its child START.
    const eng = engine(loaded.compiled, { inputs: { vs: [1n, -1n] }, numWorkers, growBuilder: buildGrow });
    const events = await eng.run();
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    expect((events.at(-1) as RunFailed).failure.message).toContain("post-assert failed");
  });
}
