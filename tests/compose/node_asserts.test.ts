/**
 * Node-level `asserts:`, the per-node contract: node-local validation, pre/post classification and
 * stamping onto the node.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { type NodeDescriptor, siteKey } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { POOL_HEADS, validateNodeAsserts } from "../../src/compose/validate/asserts.ts";
import { type FailureCode, OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import {
  Assertion,
  ContractOrigins,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { advance, runFlow } from "../../src/run.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  loadConformanceFlow,
  loadError,
  loadErrorAsync,
  runCase,
} from "../compose/declarations.ts";
import { expectFailsBeforeTheRun, rejected, stoppedRunLog } from "./run_state_fixtures.ts";

const DOUBLE = "./fixtures/compose_codefns.ts:double";

/** A flow whose CODE node `x` carries `asserts`. */
function flow(asserts: readonly string[], options: { inputDecl?: string; outputs?: string } = {}): string {
  const { inputDecl = "n: ${input.v}", outputs = "int" } = options;
  const lines = asserts.map((q) => `      - "${q}"`).join("\n");
  return `
id: na
name: na
input:
  v: !int
nodes:
  x:
    kind: code
    code: ${DOUBLE}
    input:
      ${inputDecl}
    output: ${outputs}
    asserts:
${lines}
output: \${x.output}
`;
}

/** The node `x` of `text`'s assembled flow. */
const nodeX = (text: string): Node => assembleText(text).flow.compiled.nodes.get("x")!;

/** Project carriers onto the executable and authored text under test here. */
const assertionText = (assertions: readonly Assertion[]) =>
  assertions.map((assertion) => [assertion.expression, assertion.display]);

/** Return source text large enough for direct validator ranges. */
const rangeDiagnostics = () => new LoadDiagnostics("flow.yaml", ("x".repeat(80) + "\n").repeat(30));

/** A minimal concrete node; the validator reads only its params and output contract. */
class TestNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** A descriptor carrying `asserts` and its ranges, standing in for an authored node. */
function descriptor(
  id: string,
  asserts: readonly string[],
  options: {
    assertRanges?: (SourceRange | null)[];
    fieldRanges?: Map<string, SourceRange>;
    range?: SourceRange | null;
  } = {},
): NodeDescriptor {
  return new CodeDescriptor({ id, code: "return null", asserts, ...options });
}

// ---------- classify + stamp ----------

test("node asserts classify pre and post", () => {
  const x = nodeX(flow(["output >= 0", "n >= 0"]));
  expect(assertionText(x.postAsserts)).toEqual([["output >= 0", "output >= 0"]]);
  expect(assertionText(x.preAsserts)).toEqual([["n >= 0", "n >= 0"]]);
});

test("the assertion site lookup is an option, never a positional argument", () => {
  const node = new TestNode("x");
  node.params = [new Parameter("n")];
  let called = false;
  const lookup = () => {
    called = true;
    return null;
  };
  // @ts-expect-error A site lookup passed in place of the options object is a type error.
  validateNodeAsserts(new Map([["x", node]]), new Map([["x", descriptor("x", ["n >= 0"])]]), lookup);
  expect(called).toBe(false);
});

test("a node assert's field walk uses the canonical output contract", () => {
  // Node-local validation projects the type without replacing provenance.
  const contractType = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["score", Type.scalar(ValueKind.NUMBER)]]),
    required: new Set(["score"]),
  });
  const origins = new ContractOrigins({ default: new OriginRef("node-output") });
  const node = new TestNode("x", { outputContract: new ValueContract(contractType, origins) });

  validateNodeAsserts(new Map([["x", node]]), new Map([["x", descriptor("x", ["output.score >= 0"])]]));

  expect(node.postAsserts).toStrictEqual([new Assertion("output.score >= 0", "output.score >= 0")]);
  expect((node.outputContract as ValueContract).type).toBe(contractType);
  expect((node.outputContract as ValueContract).origins).toBe(origins);
});

test("a node assert rejects an unknown field on a partial output contract", () => {
  // Node-local asserts use the partial contract's closed root field set.
  const node = new TestNode("x", {
    outputContract: new PartialRecordContract([
      ["score", new ValueContract(Type.scalar(ValueKind.NUMBER))],
      ["note", null],
    ]),
  });
  const nodes = new Map([["x", node]]);
  const descriptors = new Map([["x", descriptor("x", ["output.missing >= 0"])]]);
  expect(() => validateNodeAsserts(nodes, descriptors)).toThrow(/unknown field 'missing'/);
});

test.each([
  ["n >= 0", "output >= 0"],
  ["n >= 0", "${output >= 0}"],
  ["${n >= 0}", "output >= 0"],
  ["${n >= 0}", "${output >= 0}"],
])("node asserts classify the same across spellings: %s, %s", (pre, post) => {
  // A node assert's refs come from parsing the whole value, so the bare and whole-span spellings
  // classify alike: an `n`-only (input) assert is pre, an `output`-reading assert is post.
  const x = nodeX(flow([post, pre]));
  expect(assertionText(x.postAsserts)).toEqual([[post, post]]);
  expect(assertionText(x.preAsserts)).toEqual([[pre, pre]]);
});

test("duplicate node asserts are distinct ordered occurrences", () => {
  // Equal authored text must not collapse occurrence identity or provenance.
  const text = "n >= 0";
  const { flow: loaded, builder } = assembleText(flow([text, text]));
  const x = loaded.compiled.nodes.get("x")!;

  expect(assertionText(x.preAsserts)).toEqual([
    [text, text],
    [text, text],
  ]);
  const [first, second] = x.preAsserts as [Assertion, Assertion];
  expect(first).not.toBe(second);
  expect(first.origin).not.toBeNull();
  expect(second.origin).not.toBeNull();
  expect(first.origin!.key).not.toBe(second.origin!.key);
  const frames = [first, second].map((assertion) => builder.resolveRegistered(assertion.origin!));
  expect(frames.every((frame) => frame !== null)).toBe(true);
  expect(frames[0]!.range).not.toEqual(frames[1]!.range);
});

test.each(["nope > 0", "${nope > 0}"])("an unknown input ref is rejected in every spelling: %s", (bad) => {
  expect(() => assembleText(flow([bad]))).toThrow(/not a declared input/);
});

// ---------- rejections (located LoadError) ----------

test("an unknown input ref is rejected at its assert", () => {
  const source = flow(["nope > 0", "input.v > 0"]);
  const expectedLine = source.split("\n").findIndex((line) => line.includes("nope > 0")) + 1;
  const error = loadError(() => assembleText(source));
  expect(error.message).toMatch(/not a declared input/);
  expect(error.diagnostic.frames[0]!.range!.startLine).toBe(expectedLine);
});

test("a node assert field error prefers the exact input range", () => {
  // A pre-loop field error uses its field before the asserts block or node.
  const inputRange = new SourceRange(5, 3, 5, 9);
  const desc = descriptor("x", ["output >= 0"], {
    assertRanges: [new SourceRange(8, 5, 8, 20)],
    fieldRanges: new Map([
      [siteKey("input", "output"), inputRange],
      [siteKey("field", "asserts"), new SourceRange(7, 3, 8, 20)],
    ]),
    range: new SourceRange(2, 1, 2, 2),
  });
  const node = new TestNode("x");
  node.params = [new Parameter("output")];

  const error = loadError(() =>
    validateNodeAsserts(new Map([["x", node]]), new Map([["x", desc]]), { diagnostics: rangeDiagnostics() }),
  );
  expect(error.message).toMatch(/collides/);
  expect(error.diagnostic.frames[0]!.range).toEqual(inputRange);
});

test("a node assert error falls back to the descriptor range", () => {
  // A source-backed node still locates an error when its item range is absent.
  const nodeRange = new SourceRange(4, 3, 4, 4);
  const desc = descriptor("x", ["missing > 0"], { range: nodeRange });
  const node = new TestNode("x");
  node.params = [];

  const error = loadError(() =>
    validateNodeAsserts(new Map([["x", node]]), new Map([["x", desc]]), { diagnostics: rangeDiagnostics() }),
  );
  expect(error.message).toMatch(/not a declared input/);
  expect(error.diagnostic.frames[0]!.range).toEqual(nodeRange);
});

test("a node assert aggregate uses the first available range", () => {
  // A later authored range locates an aggregate whose first error is unlocated.
  const laterRange = new SourceRange(20, 3, 20, 19);
  const descriptors = new Map([
    ["first", descriptor("first", ["missing_first > 0"])],
    ["later", descriptor("later", ["missing_later > 0"], { assertRanges: [laterRange], range: laterRange })],
  ]);
  const nodes = new Map([
    ["first", new TestNode("first")],
    ["later", new TestNode("later")],
  ]);

  const error = loadError(() => validateNodeAsserts(nodes, descriptors, { diagnostics: rangeDiagnostics() }));
  expect(error.message).toMatch(/missing_first/);
  expect(error.diagnostic.frames[0]!.range).toEqual(laterRange);
});

test("a pool head ref is rejected", () => {
  expect(() => assembleText(flow(["input.v > 0"]))).toThrow(/node-local/);
});

// An in-file `dbl` def; the inline-call rows reject the call before anything resolves it.
const DBL_DEF = `
defs:
  dbl:
    input:
      n: !int
    nodes:
      y:
        kind: code
        code: ${DOUBLE}
        input:
          n: \${input.n}
        output: int
    output: \${y.output}
`;

/** A flow whose node `x` carries `assertion`, beside the `dbl` def. */
function inlineCallFlow(assertion: string): string {
  return `
id: na
name: na
input:
  v: !int
${DBL_DEF}
nodes:
  x:
    kind: code
    code: ${DOUBLE}
    input:
      n: \${input.v}
    output: int
    asserts:
      - ${assertion}
output: \${x.output}
`;
}

test("an inline call in a node assert is rejected", () => {
  // The retired `${flow(args)}` span form of an inline call is rejected in a node assert.
  expect(() => assembleText(inlineCallFlow('"${ dbl(n=input.v) >= 0 }"'))).toThrow(/inline/);
});

test("a call directive in a node assert is rejected", () => {
  // The whole-value `call(...)` directive form is rejected too: inline calls belong in a
  // flow-level assert, never node-local.
  expect(() => assembleText(inlineCallFlow("call(dbl, n=1)"))).toThrow(/inline/);
});

test("an input named output is rejected", () => {
  expect(() => assembleText(flow(["output >= 0"], { inputDecl: "output: ${input.v}" }))).toThrow(/collides/);
});

test("a dotted field on a scalar output is rejected", () => {
  // The output is an int (a scalar).
  expect(() => assembleText(flow(["output.field >= 0"]))).toThrow(
    "flow has invalid node asserts:\n" +
      "  node 'x' assert: ${output.field} — the node output is not a record, so it has no field 'field'",
  );
});

test("node asserts on a mapped call are rejected", async () => {
  const text = `
id: nam
name: nam
input:
  vs: !list(int)
${DBL_DEF}
nodes:
  m:
    kind: map
    call: dbl
    over: \${input.vs}
    input:
      n: \${item}
    asserts:
      - "output >= 0"
output: \${m.output}
`;
  expect((await loadErrorAsync(loadConformanceFlow(text))).message).toMatch(/mapped/);
});

test("a node assert holding a lone surrogate is a located load error", () => {
  // Only a double-quoted YAML escape writes a lone surrogate; the other errors still aggregate.
  const text = flow(["n == '\\uD800'", "nope > 0"]);
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'x' field 'asserts[0]': text holding a lone surrogate is not a yian value\n" +
      "  node 'x' assert: ${nope} is not a declared input or `${output}`",
  );
  const line = text.split("\n").findIndex((row) => row.includes("uD800")) + 1;
  expect(error.diagnostic.frames[0]!.range!.startLine).toBe(line);
});

// ---------- per-node enforcement, through runFlow ----------

/** Load `text` (its CODE references resolved) and run it on `inputs`. */
const run = async (text: string, inputs: Readonly<Record<string, unknown>>) =>
  runFlow(await loadConformanceFlow(text), inputs);

test("a pre-assert passes and runs", async () => {
  const result = await run(flow(["n >= 0"]), { v: 5 });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(10n);
});

test("a violated post-assert fails", async () => {
  // double(-1) = -2
  const result = await run(flow(["output >= 0"]), { v: -1 });
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("assertion.failed");
  expect(result.failure?.message).toContain("post-assert");
});

const CHECKED_DEF = `
defs:
  checked:
    input:
      n: !int
    nodes:
      y:
        kind: code
        code: ${DOUBLE}
        input:
          n: \${input.n}
        output: int
        asserts:
          - "output >= 0"
    output: \${y.output}
`;

test("a def-internal node assert runs through a call", async () => {
  const loaded = await loadConformanceFlow(`
id: dr
name: dr
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
`);
  // y = 10 holds `output >= 0`; y = -2 fails the def-internal post-assert.
  expect((await runFlow(loaded, { v: 5 })).status).toBe("succeeded");
  const bad = await runFlow(loaded, { v: -1 });
  expect(bad.status).toBe("failed");
  expect(bad.failure?.code).toBe("assertion.failed");
});

test("a def-internal node assert runs through a map", async () => {
  const loaded = await loadConformanceFlow(`
id: dm
name: dm
input:
  vs: !list(int)
${CHECKED_DEF}
nodes:
  m:
    kind: map
    call: checked
    over: \${input.vs}
    input:
      n: \${item}
output: \${m.output}
`);
  expect((await runFlow(loaded, { vs: [1, 2, 3] })).status).toBe("succeeded");
  // The element -1 gives y = -2, which fails.
  const bad = await runFlow(loaded, { vs: [1, -1, 3] });
  expect(bad.status).toBe("failed");
  expect(bad.failure?.code).toBe("assertion.failed");
});

const TWO_NODES = `
id: two
name: two
input:
  v: !int
nodes:
  a:
    kind: code
    code: ${DOUBLE}
    input:
      n: \${input.v}
    output: int
  b:
    kind: code
    code: ${DOUBLE}
    input:
      n: \${a.output}
    output: int
output: \${b.output}
`;

/** Give the loaded node `b` of `flow` an unparseable pre-assert. */
const breakAssert = (flow: ComposedFlow) => {
  flow.compiled.nodes.get("b")!.preAsserts = [new Assertion("${n} >= 0", "n non-negative")];
};

test("a code-built unparseable assert fails before the run starts", async () => {
  // A downstream node's bad assert throws before any node runs or any record is written.
  const flow = await loadConformanceFlow(TWO_NODES);
  breakAssert(flow);
  await expectFailsBeforeTheRun(
    flow,
    { v: 1 },
    /^node 'b' pre-assert 'n non-negative' \(`\$\{n\} >= 0`\) is not a valid boolean expression: .*write `n >= 0`/s,
  );
});

test("a code-built unparseable assert fails before resume appends", async () => {
  // Built in code, so resume runs this flow, broken while the run is stopped.
  const flow = new ComposedFlow((await loadConformanceFlow(TWO_NODES)).compiled);
  const dir = mkdtempSync(join(tmpdir(), "node-asserts-"));
  try {
    const path = join(dir, "run.jsonl");
    const lines = await stoppedRunLog(flow, { v: 1 }, path, "a", () => breakAssert(flow));

    const error = await rejected(advance({ log: path, flow }));
    expect(error).toBeInstanceOf(ExpressionError);
    expect((error as Error).message).toMatch(/^node 'b' pre-assert 'n non-negative'/);
    expect(readFileSync(path, "utf8")).toBe(lines);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------- the pool heads ----------

test("the pool heads are singular", () => {
  // `inputs`/`outputs` are not pool heads; the typo hint catches them.
  expect(POOL_HEADS).toEqual(new Set(["input", "system", "item"]));
  expect(POOL_HEADS.has("inputs")).toBe(false);
  expect(POOL_HEADS.has("outputs")).toBe(false);
});

// ---------- harvested conformance cases ----------

test.each([
  "field_of_a_scalar_output",
  "input_named_output",
  "unknown_and_pool_refs_in_node_asserts",
])("load-leaves/engine_node_asserts__%s", (name) => {
  runCase({ id: `load-leaves/engine_node_asserts__${name}`, call: assembled });
});

// ---------- nullable field reads ----------

const NULLABLE_HEAD =
  "id: na\nname: na\n" +
  "typedefs:\n  Rec: {a: str}\n  T: {f: Optional(str), g: str}\n" +
  "input:\n  go: !bool\n  rec: !Rec\n  maybe: !Optional(Rec)\n" +
  "nodes:\n";

/** A flow where `gate` may skip `maybe_skipped` (`Rec`), and `use` binds `r: <source>` and asserts `assertion`. */
const localAssert = (source: string, assertion: string) =>
  NULLABLE_HEAD +
  "  gate:\n    kind: case\n    cases:\n      - when: ${input.go}\n        then: maybe_skipped\n" +
  "  maybe_skipped:\n    kind: code\n    output: Rec\n    code: \"return {'a': 'x'}\"\n" +
  `  use:\n    kind: code\n    input:\n      r: ${source}\n    output: int\n    code: return 1\n` +
  `    asserts:\n      - "${assertion}"\n` +
  "output: ${use.output}\n";

/** A flow where `isinstance(mk.output, T)` routes `go`, which binds `u: ${mk.output}` and asserts `assertion`. */
const routedAssert = (assertion: string) =>
  NULLABLE_HEAD +
  "  mk:\n    kind: code\n    output: Any\n    code: \"return {'g': 'x'}\"\n" +
  "  route:\n    kind: case\n    cases:\n      - when: isinstance(mk.output, T)\n        then: go\n    else: other\n" +
  "  go:\n    kind: code\n    input:\n      u: ${mk.output}\n    output: str\n    code: return 'go'\n" +
  `    asserts:\n      - "${assertion}"\n` +
  "  other:\n    kind: code\n    output: str\n    code: return 'other'\n" +
  "output: ${go.output | other.output}\n";

test("a node assert on a local bound from a producer that may skip needs a guard", () => {
  const error = loadError(() => assembleText(localAssert("${maybe_skipped.output :- null}", "r.a == 'x'")));
  expect(error.diagnostic.code).toBe("expr.nullable_field_read");
  expect(error.message).toContain(
    "node 'use' assert: ${r.a} reads field 'a' of r, which may be null (node 'maybe_skipped' may be skipped while 'use' runs)",
  );
  assembleText(localAssert("${maybe_skipped.output :- null}", "r != null and r.a == 'x'"));
  assembleText(localAssert("${input.rec}", "r.a == 'x'"));
});

test("a node assert on an untyped local a CASE narrowed needs ?. for an optional field", () => {
  expect(loadError(() => assembleText(routedAssert("u.f == 'a'"))).message).toContain(
    "only narrowed to T; write u?.f",
  );
  assembleText(routedAssert("u.g == 'a'"));
});

test.each([
  ["!= null and", "r != null and r.a == 'x'"],
  ["== null or", "r == null or r.a == 'x'"],
  ["isinstance", "isinstance(r, Rec) and r.a == 'x'"],
])("a node assert narrowed by %s loads", (_name, assertion) => {
  assembleText(localAssert("${input.maybe}", assertion));
});

// ---------- declared fields at run time (#1149) ----------

/** A flow whose CODE node `x` binds `input` and returns `{k: "x"}` under `output`, carrying `asserts`. */
function declaredFlow(asserts: readonly string[], input: string, output = "Rec"): string {
  const lines = asserts.map((q) => `      - "${q}"`).join("\n");
  return `
id: na
name: na
typedefs:
  Rec: {k: str, opt: Optional(str)}
nodes:
  a:
    kind: code
    code: |
      return {}
  b:
    kind: code
    code: |
      return {k: "a"}
  x:
    kind: code
    code: |
      return {k: "x"}
    input:
      ${input}
    output: ${output}
    asserts:
${lines}
output: \${x.output.k}
`;
}

test.each<[string, string[], string, FailureCode | null]>([
  ["a pre assert over a declared absent field holds", ["r.opt == null"], "r: !Rec ${b.output}", null],
  ["a post assert over a declared absent field holds", ["output.opt == null"], "n: 1", null],
  ["a pre assert over an untyped absent key fails", ["u.k == 1"], "u: ${a.output}", "condition.evaluate"],
])("%s", async (_name, asserts, input, code) => {
  const result = await runFlow(await loadFlow(declaredFlow(asserts, input)), {});
  if (code === null) {
    expect(result.failure?.message).toBeUndefined();
    expect(result.output).toBe("x");
  } else {
    expect(result.failure?.code).toBe(code);
  }
});
