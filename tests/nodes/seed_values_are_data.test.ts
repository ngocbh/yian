/**
 * CALL / MAP / LOOP hand their child resolved values, never templates. A spawner binds its inputs
 * before it grows, so a string that reaches the child START is data: text that contains `${...}`
 * (a shell script, a JS template literal, an LLM answer) arrives unchanged and is never evaluated
 * a second time one level down.
 */

import { expect, test } from "bun:test";

import { cloneChild } from "../../src/compile/expand.ts";
import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";

function flow(source: string) {
  return loadConformanceFlow(source, { sourceLabel: "flow.yaml" });
}

const CALL = `
id: seed-call
name: seed_call
input:
  text: !str
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  echo:
    kind: call
    input:
      text: \${input.text}
    nodes:
      inner:
        kind: code
        input:
          text: \${input.text}
        output: str
        code: |
          return inputs["text"]
    output: \${inner.output}
output: \${echo.output}
`;

for (const text of [
  "${probe.output}",
  "hi ${probe.output}",
  "run=${system.run_id}",
  "costs ${5",
  "${input.text}",
  "${nope.output}",
  "$$ and $",
]) {
  test(`call passes dollar brace text through unchanged (${text})`, async () => {
    const result = await runFlow(await flow(CALL), { text });
    expect(result.status, JSON.stringify(result.failure)).toBe("succeeded");
    expect(result.output).toBe(text);
  });
}

test("map passes dollar brace items through unchanged", async () => {
  const result = await runFlow(
    await flow(`
id: seed-map
name: seed_map
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  mk:
    kind: code
    output: list(str)
    code: |
      return ["a \${probe.output}", "\${probe.output}", "plain"]
  each:
    kind: map
    over: \${mk.output}
    input:
      text: \${item}
    nodes:
      inner:
        kind: code
        input:
          text: \${input.text}
        output: str
        code: |
          return inputs["text"]
    output: \${inner.output}
output: \${each.output}
`),
    {},
  );
  expect(result.status, JSON.stringify(result.failure)).toBe("succeeded");
  expect(result.output).toEqual(["a ${probe.output}", "${probe.output}", "plain"]);
});

test("loop carries dollar brace text through unchanged", async () => {
  const result = await runFlow(
    await flow(`
id: seed-loop
name: seed_loop
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  mk:
    kind: code
    output: str
    code: |
      return "a \${probe.output}"
  lp:
    kind: loop
    times: 2
    input:
      text: \${mk.output}
    nodes:
      inner:
        kind: code
        input:
          text: \${input.text}
        output: str
        code: |
          return inputs["text"]
    output:
      text: \${inner.output}
output: \${lp.output}
`),
    {},
  );
  expect(result.status, JSON.stringify(result.failure)).toBe("succeeded");
  expect(result.output).toEqual({ text: "a ${probe.output}" });
});

test("clone child mints no edge for a dollar brace seed", async () => {
  const child = (
    await flow(`
id: seed-clone
name: seed_clone
input:
  text: !str
  other: !str
nodes:
  inner:
    kind: code
    input:
      text: \${input.text}
      other: \${input.other}
    output: str
    code: |
      return inputs["text"]
output: \${inner.output}
`)
  ).compiled;
  const cloned = cloneChild(child, "echo", { text: "${probe.output}", other: "${probe.output}" });
  expect(cloned.edges.some((edge) => edge.from === "probe")).toBe(false);
});
