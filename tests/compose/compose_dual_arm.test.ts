/**
 * The parser rejects the retired `inputs:`/`outputs:` section keys. `inputs:`/`outputs:` at the
 * top level, in a node body or in a `defs:` entry is a located load error naming the retirement;
 * `input:`/`output:` is the only spelling.
 */

import { afterEach, expect, test } from "bun:test";

import { parseFile } from "../../src/compose/parse/file.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { fileOf, loadError, runTextCase } from "../helpers/declarations.ts";

afterEach(() => forget());

const NEW_ONLY = `
id: f
name: f
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      v: \${input.x}
    output: !str
    code: tests.engine._fakes:passthrough
output:
  result: \${n.output}
`;

const LEGACY_INPUT = `
id: f
name: f
inputs:
  x: !str
nodes: {}
`;

const LEGACY_OUTPUT = `
id: f
name: f
input:
  x: !str
nodes: {}
outputs:
  r: \${n.output}
`;

test("a flow with only the new spellings parses", () => {
  const cf = parseFile(NEW_ONLY);

  expect(Object.keys(cf.inputs)).toEqual(["x"]);
  expect(Object.keys(cf.nodes)).toContain("n");
});

test("a top-level inputs: is rejected", () => {
  expect(loadError(() => parseFile(LEGACY_INPUT)).message).toContain("rename the section");
});

test("a top-level outputs: is rejected", () => {
  expect(loadError(() => parseFile(LEGACY_OUTPUT)).message).toContain("rename the section");
});

/** A flow whose node `n` has signature keys `first` and `second`. */
function nodePlural(first: string, second: string): string {
  return `id: f
name: f
input:
  x: !int
nodes:
  n:
    kind: code
    ${first}:
      x: \${input.x}
    ${second}: int
    code: return inputs["x"]
output: \${n.output}
`;
}

test.each([
  ["inputs", "output", "inputs"],
  ["input", "outputs", "outputs"],
])("a node's legacy %s/%s key is rejected at its key", (first, second, legacy) => {
  const text = nodePlural(first, second);
  const error = loadError(() => parseFile(text));

  expect(error.message).toMatch(new RegExp(`node 'n'.*'${legacy}'.*rename the key`));
  const keyRange = SourceDocument.parse(text).root.child("nodes").child("n").keyRangeOf(legacy);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(keyRange!);
});

test.each([
  ["input", "inputs"],
  ["inputs", "input"],
])("a node with both signature spellings is rejected: %s first", (first, second) => {
  const text = `id: f
name: f
input:
  x: !int
nodes:
  n:
    kind: code
    ${first}:
      x: \${input.x}
    ${second}:
      y: \${input.x}
    output: !int
    code: return inputs["x"]
output: \${n.output}
`;

  expect(loadError(() => parseFile(text)).message).toMatch(/'inputs'.*rename the key/);
});

test("a def's legacy signature key is rejected at its key", () => {
  const text = `id: f
name: f
input:
  x: !int
defs:
  child:
    inputs:
      x: !int
    nodes:
      n:
        kind: code
        input:
          x: \${input.x}
        output: !int
        code: return inputs["x"]
    output: \${n.output}
nodes:
  c:
    kind: call
    call: child
    input:
      x: \${input.x}
output: \${c.output}
`;
  const error = loadError(() => parseFile(text));

  expect(error.message).toMatch(/defs entry 'child'.*'inputs'.*rename the key/);
  const child = SourceDocument.parse(text).root.child("defs").child("child");
  const keyRange = child.keyRangeOf("inputs");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(keyRange!);
});

test("a single-node def's legacy signature key is rejected", () => {
  const text = `id: f
name: f
input:
  x: !int
defs:
  child:
    kind: code
    input:
      x: !int
    outputs: int
    code: return inputs["x"]
nodes:
  c:
    kind: call
    call: child
    input:
      x: \${input.x}
output: \${c.output}
`;

  expect(loadError(() => parseFile(text)).message).toMatch(
    /defs entry 'child'.*'outputs'.*rename the key/,
  );
});

test.each([
  "load-case/engine_compose_dual_arm__legacy_def_inputs_key_is_rejected",
  "load-case/engine_compose_dual_arm__legacy_node_inputs_key_is_rejected",
])("parseFile: %s", (id) => runTextCase({ id, call: fileOf }));
