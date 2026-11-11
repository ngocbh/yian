// A runtime failure in a nested CALL, MAP or LOOP is framed outermost first at each authored
// site, against the source the flow was loaded from.

import { expect, test } from "bun:test";

import { renderRunError } from "../../src/cli/run.ts";
import type { Writer } from "../../src/cli/utils.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { type RunResult, runFlow } from "../../src/run.ts";

// tests/seeds/errors/e24-nested-code-raise.yaml, its `boom` raising in TS.
const E24 = `# e24 — a CODE callable that raises INSIDE a called child (a \`defs:\` sub-flow).
#
# RUNTIME: the inner \`boom\` node fails with runtime id \`run/boom\`. Its canonical
# origin trace contains the outer CALL site followed by the inner \`code:\` scalar,
# independent of that runtime namespace. Also exercises \`--engine-trace\`.
# The resolver supplies both frames; the CLI neither parses the id nor walks the
# baked child graph.
id: err-nested-code-raise
name: err_nested_code_raise
input:
  topic: !str

defs:
  inner:
    input:
      topic: !str
    nodes:
      boom:
        kind: code
        code: throw new Error("always fails")
        input:
          topic: \${input.topic}
        output: !str
    output: \${boom.output}

nodes:
  run:
    kind: call
    call: inner
    input:
      topic: \${input.topic}

output: \${run.output}
`;

const NESTED_CALL_LOOP_MAXED = `
id: nested-call-loop-maxed
name: nested_call_loop_maxed
defs:
  keep:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: 'return {n: inputs.n}'
        input:
          n: \${input.n}
        output:
          n: !int
    output: \${step.output}
  runaway:
    nodes:
      repeat:
        kind: loop
        call: keep
        input:
          n: 0
        while: "true"
        max: 1
    output: \${repeat.output}
nodes:
  outer:
    kind: call
    call: runaway
output: \${outer.output}
`;

const CALL_INPUT_FAILURE = `id: nested-binding
name: nested_binding
defs:
  inner:
    input:
      maybe: !Optional(str)
    nodes:
      leaf:
        kind: code
        input:
          okay: literal
          bad: \${input.maybe:?"missing nested value"}
        output: !str
        code: return inputs.okay
    output: \${leaf.output}
nodes:
  invoke:
    kind: call
    call: inner
    input: {}
output: \${invoke.output}
`;

const MAP_INPUT_FAILURE = `id: nested-map-binding
name: nested_map_binding
defs:
  inner:
    input:
      maybe: !Optional(str)
    nodes:
      leaf:
        kind: code
        input:
          okay: literal
          bad: \${input.maybe:?"missing mapped value"}
        output: !str
        code: return inputs.okay
    output: \${leaf.output}
nodes:
  invoke:
    kind: map
    call: inner
    over: \${[1]}
    input:
      maybe: null
output: \${invoke.output}
`;

const EXTERNAL_CHILD = `id: echo-child
name: echo_child
input:
  v: !int
nodes:
  x:
    kind: code
    code: return inputs.n * 2
    input:
      n: \${input.v}
    output: !int
output: \${x.output}
`;

const CALLER_POST_ASSERT = `id: out-parent
name: out_parent
uses:
  echo-child: echo-child
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: 60
    asserts:
      - output < 100
output: \${callee.output}
`;

const NESTED_CALLER_POST_ASSERT = `id: nested-assert-parent
name: nested_assert_parent
defs:
  deep:
    input:
      v: !int
    nodes:
      x:
        kind: code
        code: return inputs.n * 2
        input:
          n: \${input.v}
        output: !int
    output: \${x.output}
  middle:
    input:
      v: !int
    nodes:
      via:
        kind: call
        call: deep
        input:
          v: \${input.v}
        asserts:
          - output < 100
    output: \${via.output}
nodes:
  outer:
    kind: call
    call: middle
    input:
      v: 60
output: \${outer.output}
`;

/** Return the text `renderRunError` writes for `result` against `loaded`. */
async function rendered(result: RunResult, loaded: ComposedFlow): Promise<string> {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  await renderRunError(result, loaded, false, out, () => undefined);
  return chunks.join("");
}

/** Run and render one in-file nested failure against its retained source. */
async function render(source: string): Promise<[string, RunResult]> {
  const loaded = await loadFlow(source, { sourceLabel: "nested.yaml" });
  const result = await runFlow(loaded, {});
  expect(result.status).toBe("failed");
  return [await rendered(result, loaded), result];
}

test("nested call error frames at child range", async () => {
  // e24 throws in `run/boom`: canonical diagnosis resolves the call and its CODE invocation
  // directly from the ordered origin trace.
  const label = "e24-nested-code-raise.yaml";
  const loaded = await loadFlow(E24, { sourceLabel: label });
  const result = await runFlow(loaded, { topic: "X" });
  expect(result.status).toBe("failed");

  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.map((frame) => frame.range.startLine)).toEqual([27, 20]);

  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain(`${label}:27`);
  expect(out).toContain(`${label}:20`);
});

test("nested call input failure uses resolved child range", async () => {
  const [out, result] = await render(CALL_INPUT_FAILURE);
  expect(result.failure!.nodeId).toBe("invoke/leaf");
  expect(out).toContain("nested.yaml:17");
  expect(out).toContain("nested.yaml:12");
  expect(out).not.toContain("nested.yaml:14");
});

test("nested map input failure uses resolved child range", async () => {
  const [out, result] = await render(MAP_INPUT_FAILURE);
  expect(result.failure!.nodeId).toBe("invoke#0/leaf");
  expect(out).toContain("nested.yaml:17");
  expect(out).toContain("nested.yaml:12");
  expect(out).not.toContain("nested.yaml:14");
});

test("caller assert range is not attached to external child", async () => {
  const child = await loadFlow(EXTERNAL_CHILD, { sourceLabel: "echo_child.yaml" });
  const loaded = await loadFlow(CALLER_POST_ASSERT, { childResolver: () => child, sourceLabel: "parent.yaml" });
  const result = await runFlow(loaded, {});

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("callee/__end__");
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.length).toBeGreaterThan(0);
  expect(new Set(diagnostic.frames.map((frame) => frame.label))).toEqual(new Set(["parent.yaml"]));
  const assertLine = diagnostic.frames.at(-1)!.range.startLine;

  const out = await rendered(result, loaded);
  expect(out).toContain(`parent.yaml:${assertLine}`);
  expect(out).not.toContain("echo_child.yaml:");
});

test("diagnosis re-resolves origins against current source", async () => {
  // Rendering uses the current source's origins, not the old runtime coordinate.
  const oldLoaded = await loadFlow(CALL_INPUT_FAILURE, { sourceLabel: "nested.yaml" });
  const result = await runFlow(oldLoaded, {});
  expect(oldLoaded.diagnose(result.failure!).frames.at(-1)!.range.startLine).toBe(12);

  const shiftedLoaded = await loadFlow(`\n${CALL_INPUT_FAILURE}`, { sourceLabel: "nested.yaml" });
  const out = await rendered(result, shiftedLoaded);
  expect(out).toContain("nested.yaml:18");
  expect(out).toContain("nested.yaml:13");
  expect(out).not.toContain("nested.yaml:12");
});

test("nested caller assert range stays with owning call frame", async () => {
  const loaded = await loadFlow(NESTED_CALLER_POST_ASSERT, { sourceLabel: "nested-assert.yaml" });
  const result = await runFlow(loaded, {});

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("outer/via/__end__");
  const diagnostic = loaded.diagnose(result.failure!);
  const assertLine = diagnostic.frames.at(-1)!.range.startLine;

  const out = await rendered(result, loaded);
  expect(diagnostic.frames.map((frame) => frame.range.startLine)).toEqual([28, 19, 25]);
  expect(out).toContain(`nested-assert.yaml:${assertLine}`);
});

test("nested loop failure renders ordered catalog frames", async () => {
  // A respawned LOOP id needs no clone parsing to locate its authored frames.
  const loaded = await loadFlow(NESTED_CALL_LOOP_MAXED, { sourceLabel: "nested-loop.yaml" });
  const result = await runFlow(loaded, {});
  expect(result.status).toBe("failed");

  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.map((frame) => frame.range.startLine)).toEqual([28, 19]);

  const out = await rendered(result, loaded);
  expect(out.indexOf("nested-loop.yaml:28")).toBeLessThan(out.indexOf("nested-loop.yaml:19"));
  expect(out).toContain("exceeded max (1)");
});
