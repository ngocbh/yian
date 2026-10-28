/**
 * In-file `defs:` callables and defs-first `call` resolution: loading, alias resolution, the
 * awaited CODE check of defs, and runs.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, mock, spyOn, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { CallableScope } from "../../src/compose/loader.ts";
import { parseFile } from "../../src/compose/parse/file.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { CodeNode } from "../../src/nodes/code/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";

import { loadConformanceFlow, loadError, loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { childResolver } from "../helpers/run_state_fixtures.ts";

/** The baked child of `flow`'s CALL node `nodeId`. */
const childOf = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as CallNode).child as CompiledFlow;

// A multi-node def, called as a plain application (no `over:`):
// research_pair = data (make_report -> {report, n}) -> summary (echo the report).
const DEFS_PARENT = `
id: defs-parent
name: defs_parent
input:
  topic: !str
defs:
  research_pair:
    input:
      topic: !str
    nodes:
      data:
        kind: code
        code: ./fixtures/compose_codefns.ts:make_report
        input:
          topic: \${input.topic}
        output:
          report: str
          n: int
      summary:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${data.output.report}
        output: str
    output: \${summary.output}
nodes:
  research:
    kind: call
    call: research_pair
    input:
      topic: \${input.topic}
output: \${research.output}
`;

test("the defs: section parses", () => {
  const f = parseFile(DEFS_PARENT);
  expect(Object.hasOwn(f.defs, "research_pair")).toBe(true);
  expect(Object.hasOwn(f.defs["research_pair"] as object, "nodes")).toBe(true);
});

test("the callable scope eagerly builds defs and memoizes resolution", async () => {
  // One scope eagerly builds local defs and resolves each external alias once.
  const seen: [string, string | null][] = [];
  const externalChild = await loadConformanceFlow("id: external\nname: external\nnodes: {}\n");
  const document = SourceDocument.parse("defs:\n  local:\n    nodes: {}\n");
  const external = (path: string, version: string | null) => {
    seen.push([path, version]);
    return externalChild;
  };
  const uses = { remote: "library/child@declared-v1" };
  const scope = new CallableScope({ local: { nodes: {} } }, new Map(), external, uses, {
    defsCursor: document.root.child("defs"),
  });

  const resolve = spyOn(scope, "resolve");
  scope.eagerLoadDefs();
  expect(resolve.mock.calls).toEqual([["local"]]);
  resolve.mockRestore();
  const local = scope.resolve("local");
  expect(local.compiled.flowId).toBe("local");
  expect(scope.resolve("local")).toBe(local);

  const first = await scope.resolveAlias("remote");
  const second = scope.resolve("remote", "another-caller-version");
  expect(seen).toEqual([["library/child", "declared-v1"]]);
  expect(first).not.toBe(externalChild);
  expect(first.compiled).not.toBe(externalChild.compiled);
  expect(second).toBe(first);
});

test("the sync resolve refuses a uses: alias that wasn't resolved first", () => {
  const resolver = mock(() => loadConformanceFlow("id: child\nname: child\nnodes: {}\n"));
  const scope = new CallableScope({}, new Map(), resolver, { remote: "library/child" });
  const error = loadError(() => scope.resolve("remote"));
  expect(error.message).toBe("uses: alias 'remote' was not resolved before assembly");
  expect(resolver).not.toHaveBeenCalled();
});

test("several defs share one parse of the parent document", async () => {
  const parse = spyOn(SourceDocument, "parse");
  try {
    await loadConformanceFlow(`id: two-defs
name: two_defs
defs:
  first:
    nodes:
      one:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {value: first}
        output: str
    output: \${one.output}
  second:
    nodes:
      two:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {value: second}
        output: str
    output: \${two.output}
nodes:
  root:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input: {value: root}
    output: str
output: \${root.output}
`);
    expect(parse).toHaveBeenCalledTimes(1);
  } finally {
    parse.mockRestore();
  }
});

// Recursive and mutual defs are loud: the baked call graph is finite.
const SELF_RECURSIVE = `
id: rec
name: rec
input:
  topic: !str
defs:
  a:
    input:
      topic: !str
    nodes:
      step:
        kind: call
        call: a
        input:
          topic: \${input.topic}
    output: \${step.output}
nodes:
  go:
    kind: call
    call: a
    input:
      topic: \${input.topic}
output: \${go.output}
`;

const MUTUAL_RECURSIVE = `
id: mutual
name: mutual
input:
  topic: !str
defs:
  a:
    input:
      topic: !str
    nodes:
      to_b:
        kind: call
        call: b
        input:
          topic: \${input.topic}
    output: \${to_b.output}
  b:
    input:
      topic: !str
    nodes:
      to_a:
        kind: call
        call: a
        input:
          topic: \${input.topic}
    output: \${to_a.output}
nodes:
  go:
    kind: call
    call: a
    input:
      topic: \${input.topic}
output: \${go.output}
`;

test("a self-recursive def is loud", async () => {
  const error = await loadErrorAsync(loadConformanceFlow(SELF_RECURSIVE));
  expect(error.message.toLowerCase()).toContain("recursive");
  expect(error.message).toContain("'a'");
});

test("mutually recursive defs are loud", async () => {
  const error = await loadErrorAsync(loadConformanceFlow(MUTUAL_RECURSIVE));
  expect(error.message.toLowerCase()).toContain("recursive");
});

// A single-node def: a flat `kind:` plus logic body whose `input:` entries are parameters
// (name -> TYPE), auto-wired by name into the one node; `output:` is the codomain type.
const SINGLE_NODE_DEF = `
id: single
name: single
input:
  t: !str
defs:
  passthru:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input: { topic: !str }
    output: str
nodes:
  go:
    kind: call
    call: passthru
    input:
      topic: \${input.t}
output: \${go.output}
`;

test("a single-node def's projected header and logic keep their origins", async () => {
  const loaded = await loadConformanceFlow(SINGLE_NODE_DEF);
  const node = childOf(loaded.compiled, "go").nodes.get("passthru") as CodeNode;
  const resolver = loaded.originResolver!;

  expect(node.origin).not.toBeNull();
  expect(node.codeOrigin).not.toBeNull();
  expect(resolver.resolve(node.origin!)!.range.startLine).toBe(7);
  expect(resolver.resolve(node.codeOrigin!)!.range.startLine).toBe(9);
});

test("a single-node def's generated inputs keep the binding origin", async () => {
  const loaded = await loadConformanceFlow(SINGLE_NODE_DEF);
  const binding = childOf(loaded.compiled, "go").wiring.get("passthru")!.get("topic")!;
  const resolver = loaded.originResolver!;

  expect(binding.origin).not.toBeNull();
  expect(resolver.resolve(binding.origin!)!.range).toEqual(new SourceRange(10, 21, 10, 25));
});

test("a single-node def's generated output keeps the contract origin", async () => {
  const loaded = await loadConformanceFlow(SINGLE_NODE_DEF);
  const node = childOf(loaded.compiled, "go").nodes.get("passthru")!;
  const origin = (node.outputContract as ValueContract).origins.default;
  const resolver = loaded.originResolver!;

  expect(origin).not.toBeNull();
  expect(resolver.resolve(origin!)!.range).toEqual(new SourceRange(11, 13, 11, 16));
});

test("a single-node def's call is type-checked across flows", async () => {
  // The parent binds a str to a def param typed int.
  const text = `
id: single
name: single
input:
  t: !str
defs:
  passthru:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: !int
    output: str
nodes:
  go:
    kind: call
    call: passthru
    input:
      topic: \${input.t}
output: \${go.output}
`;
  await loadErrorAsync(loadConformanceFlow(text));
});

test("a call to an unknown callable is loud", async () => {
  const text = `
id: unk
name: unk
input:
  topic: !str
defs:
  known:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${x.output}
nodes:
  go:
    kind: call
    call: missing
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("missing");
  // A bare call to a non-def, non-alias callable is loud: external flows only through uses:.
  expect(error.message).toContain("uses:");
});

test("a malformed def without nodes: is loud", async () => {
  const text = `
id: bad
name: bad
input:
  topic: !str
defs:
  oops:
    input:
      topic: !str
    output: \${nope.output}
nodes:
  go:
    kind: call
    call: oops
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("nodes");
});

test("a def-internal error names the def", async () => {
  const text = `
id: dd
name: dd
input:
  topic: !str
defs:
  enrich:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${nonexistent.output}
        output: str
    output: \${x.output}
nodes:
  go:
    kind: call
    call: enrich
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("defs entry 'enrich'");
  expect(error.message).toContain("nonexistent");
});

test("a def body that isn't a mapping is loud", async () => {
  const text = `
id: nm
name: nm
input:
  topic: !str
defs:
  d: just a string
nodes:
  go:
    kind: call
    call: d
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("must be a mapping");
  expect(error.message).toContain("'d'");
});

test("an unknown def field is loud", async () => {
  const text = `
id: uf
name: uf
input:
  topic: !str
defs:
  d:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${x.output}
    bogus: 1
nodes:
  go:
    kind: call
    call: d
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("unknown field");
  expect(error.message).toContain("bogus");
});

test("an uncalled broken def is loud: every def builds eagerly", async () => {
  const text = `
id: ub
name: ub
input:
  topic: !str
defs:
  used:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${x.output}
  unused_broken:
    input:
      topic: !str
    nodes:
      y:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${nonexistent.output}
        output: str
    output: \${y.output}
nodes:
  go:
    kind: call
    call: used
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("defs entry 'unused_broken'");
});

test("an untagged str source can't fill a Literal child input", async () => {
  // A plain `str` can't fill a `Literal(...)` input at load, as `int` can't fill `int(1..12)`.
  const text = `id: str-to-literal
name: str_to_literal
typedefs:
  Mode: Literal(a, b)
input:
  s: !str
defs:
  child:
    kind: code
    code: |
      return inputs["mode"]
    input:
      mode: !Mode
    output: str
nodes:
  go:
    kind: call
    call: child
    input:
      mode: \${input.s}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("call node 'go': binding 'mode'");
  expect(error.message).toContain("child expects 'Literal(a, b)', source is 'str'");
});

test("a nullable source bound to a defaulted child input is strict", async () => {
  // The child's `topic` has a default but isn't nullable; the parent's null would shadow the
  // default, so the child default doesn't cover it.
  const text = `
id: nstrict
name: nstrict
input:
  x: !Optional(str)
defs:
  child:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: !str "fallback"
    output: str
nodes:
  go:
    kind: call
    call: child
    input:
      topic: \${input.x}
output: \${go.output}
`;
  await loadErrorAsync(loadConformanceFlow(text));
});

test("a nullable list element bound to a strict child is refused", async () => {
  // Element nullability is refused at load, before an actual null can run.
  const text = `
id: nullable-list-strict-child
name: nullable_list_strict_child
input:
  items: !list(Optional(int))
defs:
  child:
    kind: code
    code: |
      return inputs["items"]
    input:
      items: !list(int)
    output: list(int)
nodes:
  go:
    kind: call
    call: child
    input:
      items: \${input.items}
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("call node 'go': binding 'items'");
  expect(error.message).toContain("child expects 'list(int)', source is 'list(Optional(int))'");
});

test("an omitted child input takes its explicit null default", async () => {
  const loaded = await loadConformanceFlow(`id: omitted-null-default
name: omitted_null_default
defs:
  child:
    kind: code
    code: |
      return inputs["maybe"]
    input:
      maybe: !Optional(int) null
    output: Optional(int)
nodes:
  go:
    kind: call
    call: child
output: \${go.output}
`);
  const parameter = (loaded.compiled.nodes.get("go") as CallNode).childInputs[0]!;

  expect(parameter.constructor).toBe(Parameter);
  expect(parameter.contract!.constructor).toBe(ValueContract);
  expect((parameter.contract as ValueContract).type).toEqual(new Type({ kind: ValueKind.INTEGER, nullable: true }));
  expect(parameter.hasDefault).toBe(true);
  expect(parameter.default).toBeNull();
  expect(parameter.coerceStrings).toBe(true);
});

test("an invalid def default keeps its absolute range and the def's context", async () => {
  const text = `id: invalid-def-default
name: invalid_def_default
defs:
  child:
    kind: code
    code: |
      return inputs["value"]
    input:
      value: !int null
    output: int
nodes:
  go:
    kind: call
    call: child
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toContain("defs entry 'child'");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(9);
});

test("a single-node def's parameter named inf is refused at its key", async () => {
  const text = `id: f
name: f
defs:
  d:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      inf: !str
    output: str
nodes:
  go:
    kind: call
    call: d
    input:
      inf: x
output: \${go.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toBe(
    "defs entry 'd': input 'inf': `inf` is infinity in an expression, so ${inf} can't read it; rename it",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(
    SourceDocument.parse(text).root.child("defs").child("d").child("input").keyRangeOf("inf")!,
  );
});

test("defs without a source cursor build from their normalized bodies", () => {
  const scope = new CallableScope(
    {
      multi: {
        inputs: { x: "int" },
        nodes: { step: { kind: "code", code: "return 1", inputs: { x: "${input.x}" }, outputs: "int" } },
        outputs: "${step.output}",
      },
      single: { kind: "code", code: "return 1", inputs: { x: "int" }, outputs: "int" },
    },
    new Map(),
    null,
    {},
  );

  scope.eagerLoadDefs();
  const multi = scope.resolve("multi");
  const single = scope.resolve("single");
  expect(multi.compiled.wiring.get("step")!.get("x")!.source).toBe("${input.x}");
  expect(single.compiled.wiring.get("single")!.get("x")!.source).toBe("${input.x}");
  expect(multi.originResolver).toBeNull();
  expect(single.originResolver).toBeNull();
});

/** A multi-node def `name` over `x`, its nodes written as given (each indented six spaces). */
const defText = (name: string, nodes: string, output: string) => `  ${name}:
    input:
      x: !str
    nodes:
${nodes}    output: \${${output}.output}
`;

/** A CODE node `id` whose inline body fails the syntax check. */
const badCode = (id: string) => `      ${id}:
        kind: code
        code: "return ("
        input:
          x: \${input.x}
        output: str
`;

/** A CALL node `id` of `call`. */
const callOf = (id: string, call: string) => `      ${id}:
        kind: call
        call: ${call}
        input:
          x: \${input.x}
`;

/** A flow over `x` with `defs` and one root CODE node no def is called from. */
const flowWithDefs = (defs: string) => `id: f
name: f
input:
  x: !str
defs:
${defs}nodes:
  r:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      x: \${input.x}
    output: str
output: \${r.output}
`;

/** Assert `error`'s frames: the given defs' bodies in order, then node `nodeId`'s `code:` value. */
function expectDefFrames(error: LoadError, text: string, defs: string[], [def, nodeId]: [string, string]): void {
  const defsCursor = SourceDocument.parse(text).root.child("defs");
  const frames = error.diagnostic.frames;
  expect(frames.map((frame) => frame.note)).toEqual([...defs.map((name) => `definition '${name}'`), null]);
  const ranges = frames.map((frame): SourceRange | null => frame.range);
  expect(ranges).toEqual([
    ...defs.map((name) => defsCursor.child(name).range),
    defsCursor.child(def).child("nodes").child(nodeId).child("code").range,
  ]);
}

test("an uncalled def's bad CODE body is reported in the def's frame", async () => {
  const text = flowWithDefs(defText("d", badCode("k"), "k"));
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message.startsWith("defs entry 'd': node 'k': ")).toBe(true);
  expectDefFrames(error, text, ["d"], ["d", "k"]);
});

test("a called def's bad CODE body is reported in the caller's frame, then its own", async () => {
  const text = flowWithDefs(defText("b2", callOf("c", "b1"), "c") + defText("b1", badCode("k"), "k"));
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message.startsWith("defs entry 'b2': defs entry 'b1': node 'k': ")).toBe(true);
  expectDefFrames(error, text, ["b2", "b1"], ["b1", "k"]);
});

test("defs' CODE bodies are checked in written order, each def's nodes in order", async () => {
  const text = flowWithDefs(
    defText("b2", badCode("own") + callOf("c", "b1"), "own") + defText("b1", badCode("k"), "k"),
  );
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message.startsWith("defs entry 'b2': node 'own': ")).toBe(true);
  expectDefFrames(error, text, ["b2"], ["b2", "own"]);
});

// ---------- runs ----------

const dir = mkdtempSync(join(tmpdir(), "yian-defs-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Load `text` with the conformance fixtures (and `options`), run it on `inputs`, and return its output. */
async function runOutput(
  text: string,
  inputs: Record<string, unknown>,
  options: Parameters<typeof loadConformanceFlow>[1] = {},
): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text, options), inputs, { env: hostEnv(dir) });
  expect(result.failure).toBeNull();
  return result.output;
}

test("a defs callable runs end to end without an external resolver", async () => {
  // research_pair: make_report -> {report: "report for ACME", ...} -> echo the report.
  expect(await runOutput(DEFS_PARENT, { topic: "ACME" })).toBe("report for ACME");
});

test("defs shadow an external flow of the same name", async () => {
  const parent = `
id: shadow-parent
name: shadow_parent
input:
  topic: !str
defs:
  shared:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:make_report
        input:
          topic: \${input.topic}
        output:
          report: str
          n: int
    output: \${x.output.report}
nodes:
  call_shared:
    kind: call
    call: shared
    input:
      topic: \${input.topic}
output: \${call_shared.output}
`;
  const external = `
id: shared
name: shared
input:
  topic: !str
nodes:
  x:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: str
output: \${x.output}
`;
  // The in-file def (make_report) wins, not the external echo ("ACME").
  expect(await runOutput(parent, { topic: "ACME" }, { childResolver: childResolver({ shared: external }) })).toBe(
    "report for ACME",
  );
});

test("a single-node def runs", async () => {
  expect(await runOutput(SINGLE_NODE_DEF, { t: "ACME" })).toBe("ACME");
});

test("def asserts are enforced at the child seam", async () => {
  const text = `
id: da
name: da
input:
  topic: !str
defs:
  d:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${x.output}
    asserts:
      - x.output != ""
nodes:
  go:
    kind: call
    call: d
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  expect(await runOutput(text, { topic: "ACME" })).toBe("ACME");
});

/** The str-to-literal child, with the input type `source` and the call binding `binding`. */
const strToLiteralChild = (source: string, binding: string) => `
id: str-to-literal
name: str_to_literal
typedefs:
  Mode: Literal(a, b)
input:
  s: ${source}
defs:
  child:
    kind: code
    code: |
      return inputs["mode"]
    input:
      mode: !Mode
    output: str
nodes:
  go:
    kind: call
    call: child
    input:
      mode: ${binding}
output: \${go.output}
`;

test.each([
  ["!Literal(a)", "${input.s}"],
  ["!Mode", "${input.s}"],
  ["!str", "a"],
  ["!str", "!Mode ${input.s}"],
])("a tag subset or literal binding fills a literal child input: %s %s", async (source, binding) => {
  expect(await runOutput(strToLiteralChild(source, binding), { s: "a" })).toBe("a");
});

test.each([
  ["a", "a"],
  ["c", "other"],
])("an isinstance guard lets a str fill a literal child input: %s", async (s, expected) => {
  const text = `
id: str-to-literal-guarded
name: str_to_literal_guarded
typedefs:
  Mode: Literal(a, b)
input:
  s: !str
defs:
  child:
    kind: code
    code: |
      return inputs["mode"]
    input:
      mode: !Mode
    output: str
nodes:
  route:
    kind: case
    cases:
      - when: isinstance(input.s, Mode)
        then: go
    else: other
  go:
    kind: call
    call: child
    input:
      mode: \${input.s}
  other:
    kind: code
    code: return "other"
output: \${go.output | other.output}
`;
  expect(await runOutput(text, { s })).toBe(expected);
});

test("a nullable source with a literal escape is ok", async () => {
  // `:-safe` guarantees a non-null value (it fires on null, a present null included).
  const text = `
id: nok
name: nok
input:
  x: !Optional(str)
defs:
  child:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: !str
    output: str
nodes:
  go:
    kind: call
    call: child
    input:
      topic: \${input.x:-"safe"}
output: \${go.output}
`;
  expect(await runOutput(text, { x: null })).toBe("safe");
});

test("a def YAML merge uses the local nodes override", async () => {
  const text = `id: merged-def
name: merged_def
x-def-base: &def_base
  nodes:
    old:
      kind: code
      code: |
        return "old"
      output: str
  output: \${old.output}
defs:
  target:
    <<: *def_base
    nodes:
      new:
        kind: code
        code: |
          return "new"
        output: str
    output: \${new.output}
nodes:
  invoke:
    kind: call
    call: target
output: \${invoke.output}
`;
  expect(await runOutput(text, {})).toBe("new");
});
