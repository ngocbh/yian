/**
 * The CODE node's source classification, its inline line-origin selector and its inline and
 * built-in bodies: a port of `tests/engine/test_code_inline.py`.
 *
 * Ported: `test_inline_reads_inputs_dict_and_returns`,
 * `test_inline_calling_convention_matches_reference` (`Object.keys(inputs).sort()`),
 * `test_inline_multi_statement_body_with_helpers`, `test_inline_body_that_prints_still_returns_value`
 * (a spy on `console.log`) and `test_inline_non_owned_failure_uses_whole_code_origin` (also with a
 * `CodeError`-wrapped outside error).
 *
 * Ported (adapted): `test_classify_reference_inline_reject` and
 * `test_wrapped_line_selector_uses_exact_whole_then_node_fallback` (its `True` line is `true`);
 * `test_inline_stdlib_import_in_body_works_in_process` (a dynamic `import("node:path")`);
 * `test_reference_mode_still_dispatches_in_process` and
 * `test_reference_failure_uses_code_then_node_fallback` (`./compose_codefns.ts:echo` and `:boom`
 * with `baseDir` the repository's `conformance/fixtures`, and an inline `throw` for the source-free
 * node); `test_inline_exception_selects_deepest_owned_code_line` (JS `1 / 0` doesn't throw, so the lambda
 * and comprehension forms become a throwing arrow and a `map` callback; lines are 1-based);
 * `test_syntax_error_rejected_at_construction` becomes `syntax error rejected by the load check`
 * (construction succeeds; the awaited `check()` rejects with a `TypeStripError` at line 1).
 *
 * Changed: a reference is a relative path to a `.ts`, `.js` or `.mjs` file plus `:fn`
 * (`./mod.ts:fn`); any other value is an inline body, so there is no reject bucket and
 * `./helpers.ts`, `pkg.mod:fn` and `./x.py:fn` classify as inline. The selector takes the line of
 * the authored body (line 1 is the body's first line), not the line of Python's
 * `def main(inputs):` wrapper. `test_inline_return_comes_back_unchecked` becomes `inline return is
 * checked by intake`: a CODE result goes through host intake (design 3.2.1), so a `Date` fails
 * `contract.output` at the contract origin.
 *
 * TS-only: the constructor compiles nothing, so a body that doesn't parse constructs; the node's
 * `mode`; a non-integer line selects no line; the selector's origin shape checks; the built-in
 * identity form a value entry lowers to (its `"builtin"` mode, its frozen copy, and the shape and
 * type checks on `code`); TypeScript annotations and `await` run; ints arrive as `number` (big
 * ones as `bigint`) and an integral result becomes an int; the `inputs` proxy (an undeclared
 * input rejects, `in`, `JSON.stringify`, `return inputs`, strict assignment); a body returning
 * `undefined` rejects; a thrown string's and a thrown record's text; a thrown string falls back to
 * the whole code origin; a body error's code is `node.execute`; a built-in's origin is the node's;
 * the built-in identity runs on the bound record; the runtime's function header is 2 lines; a
 * failing body's stack names its frames after the node's `sourceFile` (`<inline-code>` by
 * default) and the body's own line and column, its raw copy keeps them, and the throwing line is
 * still located; a reference's frames name its own file; through
 * the evaluator and the engine, a success commits its int, a throwing body fails `node.execute`
 * with `Error: boom` and `causeType` `CodeError` at the line origin, a no-value body fails at the
 * whole `code:` origin, and an intake refusal fails `contract.output` at the contract origin; a
 * reference returning an int commits an int; a reference returning `undefined` rejects with the
 * no-value error; a missing export rejects with a `TypeError`, not a `CodeError`; a module that
 * throws while loading rejects with a `CodeError`; a `../` reference resolves against `baseDir`;
 * the load check refuses a wrapper escape without running it (at stripping, a `TypeStripError`, and
 * one that strips cleanly by the constructor, a `SyntaxError`), a body redeclaring `inputs` (the
 * constructor's `SyntaxError`), a Python-style `module:function` value and a reference to a
 * missing file, and passes an existing file (`../` included, never imported), a body that would
 * throw (never called) and a built-in. The body runs in a worker thread: a file input arrives as
 * a `FileRef` and a returned one commits a `FileRef`; a class instance, a getter and a function
 * in a result fail `contract.output` with intake's messages; a custom error class keeps its name
 * and its line; `process.exit(3)` in a body and a body awaiting a promise nothing resolves reject
 * with the worker's exit code, and the next call still works; a late timer throw rejects as a
 * `CodeError`; the worker uses the caller's `withEnv` values (a refused result's path is cut to a
 * small shown-value cap); two serial calls reuse one thread (a global set by the first is seen by
 * the second); a worker shares the host's live environment both ways (a host change after its
 * thread started reaches a body, and a body's change reaches the host); a call after an aborted
 * call still works; an input the clone refuses rejects, leaves no abort listener on the signal,
 * and the next call works; a late error from an earlier body while its worker is idle prints a
 * console error line and doesn't fail the next call; aborting the signal stops a body that never
 * yields, rejecting with its reason within 5 s; an already-aborted signal rejects before a worker
 * starts; a process whose spinning body was stopped exits on its own, on Bun and, with
 * `YIAN_TEST_NODE`, built on Node; and in a pooled run a busy CODE node doesn't block a sibling.
 *
 * Not ported: `test_no_return_body_rejected_at_construction` and `test_return_only_in_nested_def_still_rejected`
 * (Python `ast`; TS has no load-time `return` check);
 * `test_structurally_equal_foreign_code_cannot_impersonate_owned_code` (Python code objects);
 * `test_dynamic_same_filename_code_falls_back_to_owned_call_site` (Python `compile` and `exec`);
 * `test_reference_failure_does_not_inspect_origin_truthiness` (Python `__bool__`).
 *
 * Ported through `loadFlow` and `runFlow`, in `hostEnv`: `test_inline_flow_loads_and_runs` and
 * `test_inline_raise_becomes_failed_run` (the body JavaScript, `toFixed(1)` for Python's `:.1f`).
 */

import { afterAll, beforeAll, describe, expect, spyOn, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { getEventListeners } from "node:events";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { withEnv } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { NodeFailed, NodeSucceeded, type RunEvent } from "../../src/events.ts";
import { HostOutputError, Output } from "../../src/nodes/base.ts";
import { ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { classifyCodeSource, CodeError, CodeNode, selectInlineCodeOrigin } from "../../src/nodes/code/node.ts";
import { runtime, TypeStripError } from "../../src/platform.ts";
import { runFlow } from "../../src/run.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { FileRef, Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { drive, engine, graph, rejected } from "./run_state_fixtures.ts";

/** Run a CODE node `n` with `code` on the bound `inputs` and return its result's value. */
async function run(code: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await new CodeNode("n", { code }).run(inputs);
  expect(result).toBeInstanceOf(Output);
  return (result as Output).value;
}

/** Run `node` on `inputs`, which must fail, and return the origin it gives the error. */
async function failureOrigin(node: CodeNode, inputs: Record<string, unknown> = {}): Promise<OriginRef | null> {
  return node.exceptionOrigin(await rejected(node.run(inputs)));
}

/** A CODE node `n` with `code`, a whole-code origin, one origin per body line and `sourceFile`. */
function locatedNode(code: string, sourceFile?: string): CodeNode {
  return new CodeNode("n", {
    code,
    sourceFile,
    codeOrigin: new OriginRef("code"),
    codeLineOrigins: code.split("\n").map((_, index) => new OriginRef(`line:${index}`)),
  });
}

/** Return what a run of a CODE node `n` with `code` rejects with. */
function failure(code: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  return rejected(new CodeNode("n", { code }).run(inputs));
}

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

test("classify reference inline reject", () => {
  expect(classifyCodeSource("./mod.ts:fn")).toBe("reference");
  expect(classifyCodeSource("../lib/x.mjs:run")).toBe("reference");
  expect(classifyCodeSource("./a/b.js:f")).toBe("reference");
  expect(classifyCodeSource("return 1")).toBe("inline");
  expect(classifyCodeSource("const x = 1\nreturn x")).toBe("inline");
  // No `:fn`, a Python-style token and a non-TS file are bodies, not references.
  expect(classifyCodeSource("./helpers.ts")).toBe("inline");
  expect(classifyCodeSource("pkg.mod:fn")).toBe("inline");
  expect(classifyCodeSource("./x.py:fn")).toBe("inline");
});

test("wrapped line selector uses exact, whole, then node fallback", () => {
  const whole = new OriginRef("code");
  const lines = [new OriginRef("line:0"), new OriginRef("line:1")];
  const node = new OriginRef("node");

  expect(selectInlineCodeOrigin(2, { codeOrigin: whole, codeLineOrigins: lines, nodeOrigin: node })).toBe(lines[1]!);
  expect(selectInlineCodeOrigin(0, { codeOrigin: whole, codeLineOrigins: lines, nodeOrigin: node })).toBe(whole);
  expect(selectInlineCodeOrigin(3, { codeOrigin: whole, codeLineOrigins: lines.slice(0, 1), nodeOrigin: node })).toBe(
    whole,
  );
  expect(selectInlineCodeOrigin(1, { codeOrigin: null, codeLineOrigins: [], nodeOrigin: node })).toBe(node);
  expect(selectInlineCodeOrigin(true, { codeOrigin: null, codeLineOrigins: [], nodeOrigin: null })).toBeNull();
});

// --- the in-process entrypoint (one-record `inputs`) --- //

test("inline reads inputs and returns", async () => {
  expect(await run("return `${inputs.label}: ${inputs.rating.toFixed(1)}`", { label: "BUY", rating: 4.26 })).toBe(
    "BUY: 4.3",
  );
});

test("inline calling convention matches reference", async () => {
  // Called with ONE record (like a reference `fn(inputs)`), not spread arguments.
  expect(await run("return Object.keys(inputs).sort()", { b: 1n, a: 2n })).toEqual(["a", "b"]);
});

test("inline multi-statement body with helpers", async () => {
  const body = 'function label(r) {\n  return r >= 0 ? "pos" : "neg"\n}\nreturn label(inputs.r)';
  expect(await run(body, { r: -1n })).toBe("neg");
});

test("inline dynamic import works in process", async () => {
  expect(await run('const p = await import("node:path")\nreturn p.join("a", inputs.x)', { x: "b" })).toBe("a/b");
});

test("inline body that logs still returns its value", async () => {
  const log = spyOn(console, "log").mockImplementation(() => {});
  try {
    expect(await run('console.log("side effect")\nreturn inputs.x * 2', { x: 21n })).toBe(42n);
    expect(log).toHaveBeenCalledWith("side effect");
  } finally {
    log.mockRestore();
  }
});

test("syntax error rejected by the load check", async () => {
  const node = new CodeNode("n", { code: "return (" });

  const error = await rejected(node.check());

  expect(error).toBeInstanceOf(TypeStripError);
  expect((error as TypeStripError).line).toBe(1);
});

// --- source-neutral exception attribution --- //

for (const [name, code, expectedLine] of [
  ["nested-helper", 'function helper() {\n  throw new Error("boom")\n}\nhelper()\nreturn 0', 2],
  ["arrow", 'const fail = () => { throw new RangeError("boom") }\nreturn fail()', 1],
  ["map callback", "return inputs.values.map((v) => v.x.y)", 1],
] as const) {
  test(`inline exception selects the deepest owned code line (${name})`, async () => {
    const node = locatedNode(code);

    const origin = await failureOrigin(node, { values: [0n] });

    expect(origin).toBe(node.codeLineOrigins[expectedLine - 1]!);
  });
}

// --- a failing body's stack names its source file --- //

const HELPER_BODY = 'function helper() {\n  throw new Error("boom")\n}\nhelper()\nreturn 0';

test("a body error's stack names the source file and the authored line", async () => {
  const error = (await rejected(
    new CodeNode("n", { code: HELPER_BODY, sourceFile: "flows/my flow.yaml" }).run({}),
  )) as CodeError;

  expect(error).toBeInstanceOf(CodeError);
  expect(error.stack).toContain("(flows/my flow.yaml:2:");
  expect(error.stack).toContain("(flows/my flow.yaml:4:");
  expect(error.stack).not.toContain("yian-code:");
  // The copy of what the body threw keeps the runtime's own frames.
  expect((error.original as Error).stack).toContain("yian-code:n:4:");
});

test("a body's frames default to <inline-code>, a first-line column counted from the body", async () => {
  const error = (await failure('throw new Error("boom")')) as CodeError;

  const frame = /<inline-code>:1:(\d+)/.exec(error.stack ?? "");
  expect(frame).not.toBeNull();
  // Within the 23-character line, not shifted by the strict prologue sharing it.
  expect(Number(frame![1])).toBeLessThanOrEqual(23);
});

test("a renamed stack still locates the throwing line", async () => {
  const node = locatedNode(HELPER_BODY, "flow.yaml");

  expect(await failureOrigin(node)).toBe(node.codeLineOrigins[1]!);
});

test("inline non-owned failure uses the whole code origin", () => {
  const node = locatedNode("return 0");

  // Raised outside the node: no frame of the authored code.
  expect(node.exceptionOrigin(new Error("outside"))).toBe(node.codeOrigin);
  expect(node.exceptionOrigin(new CodeError(new Error("outside")))).toBe(node.codeOrigin);
});

// --- host intake checks the returned value --- //

test("inline return is checked by intake", async () => {
  const node = new CodeNode("n", { code: "return { ok: 1, bad: new Date(0) }" });
  const contractOrigin = new OriginRef("contract");
  node.outputContract = new ValueContract(Type.scalar(ValueKind.ANY), new ContractOrigins({ default: contractOrigin }));

  const error = await rejected(node.run({}));

  expect(error).toBeInstanceOf(HostOutputError);
  expect((error as Error).message).toBe("output at bad: Date is not a yian value");
  expect(node.exceptionCode(error)).toBe("contract.output");
  expect(node.exceptionOrigin(error)).toBe(contractOrigin);
});

test("a TypeCheckError the code throws stays node.execute", () => {
  const node = new CodeNode("n", { code: "return 1" });
  // What the code throws is wrapped, so only intake's `HostOutputError` is `contract.output`.
  expect(node.exceptionCode(new CodeError(new TypeCheckError("bad")))).toBe("node.execute");
  expect(node.exceptionCode(new TypeCheckError("bad"))).toBe("node.execute");
});

// --- reference mode --- //

/** The repository's conformance fixtures, where `compose_codefns.ts` lives. */
const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));

/** A temporary directory of helper modules the reference rows load. */
let helpers: string;

beforeAll(() => {
  helpers = mkdtempSync(join(tmpdir(), "yian-code-"));
  writeFileSync(join(helpers, "nothing.mjs"), "export function nothing(inputs) {}\n");
  writeFileSync(join(helpers, "broken.mjs"), 'throw new RangeError("cannot load")\n');
});

afterAll(() => {
  rmSync(helpers, { recursive: true, force: true });
});

test("reference mode dispatches in process", async () => {
  const node = new CodeNode("r", { code: "./compose_codefns.ts:echo", baseDir: FIXTURES });

  expect(node.mode).toBe("reference");
  expect(await node.run({ topic: "hi" })).toEqual(new Output("hi"));
});

test("reference failure uses code then node fallback", async () => {
  const codeOrigin = new OriginRef("code:reference");
  const reference = new CodeNode("reference", { code: "./compose_codefns.ts:boom", baseDir: FIXTURES, codeOrigin });
  reference.origin = new OriginRef("node:reference");
  const withoutCodeOrigin = new CodeNode("reference-fallback", {
    code: "./compose_codefns.ts:boom",
    baseDir: FIXTURES,
  });
  withoutCodeOrigin.origin = new OriginRef("node:reference-fallback");
  const sourceFree = new CodeNode("inline", { code: 'throw new Error("boom")\nreturn 0' });

  expect(await failureOrigin(reference)).toBe(codeOrigin);
  expect(await failureOrigin(withoutCodeOrigin)).toBe(withoutCodeOrigin.origin);
  expect(await failureOrigin(sourceFree)).toBeNull();
});

test("a reference's frames name its own file", async () => {
  const error = (await rejected(
    new CodeNode("r", { code: "./compose_codefns.ts:boom", baseDir: FIXTURES, sourceFile: "flow.yaml" }).run({}),
  )) as CodeError;

  expect(error).toBeInstanceOf(CodeError);
  expect(error.stack).toContain("compose_codefns.ts:");
  expect(error.stack).not.toContain("yian-code:");
  expect(error.stack).not.toContain("flow.yaml");
});

test("a reference returning an int commits an int", async () => {
  // `double` takes and returns ints as numbers.
  const node = new CodeNode("d", { code: "./compose_codefns.ts:double", baseDir: FIXTURES });

  expect(await node.run({ n: 21n })).toEqual(new Output(42n));
});

test("a reference returning undefined rejects with the no-value error", async () => {
  const error = await rejected(new CodeNode("u", { code: "./nothing.mjs:nothing", baseDir: helpers }).run({}));

  expect((error as Error).constructor).toBe(Error);
  expect((error as Error).message).toBe("CODE body returned no value; return null for no value");
});

test("a missing export rejects with a TypeError, not a CodeError", async () => {
  const error = await rejected(new CodeNode("m", { code: "./compose_codefns.ts:nope", baseDir: FIXTURES }).run({}));

  expect((error as Error).constructor).toBe(TypeError);
  expect((error as Error).message).toBe("'./compose_codefns.ts' exports no function 'nope'");
});

test("a module that throws while loading rejects with a CodeError", async () => {
  const error = await rejected(new CodeNode("b", { code: "./broken.mjs:run", baseDir: helpers }).run({}));

  expect(error).toBeInstanceOf(CodeError);
  expect((error as Error).message).toBe("RangeError: cannot load");
  expect((error as CodeError).original).toBeInstanceOf(RangeError);
});

test("a ../ reference resolves against baseDir", async () => {
  const node = new CodeNode("p", {
    code: "../fixtures/compose_codefns.ts:echo",
    baseDir: join(FIXTURES, "..", "cases"),
  });

  expect(await node.run({ topic: "up" })).toEqual(new Output("up"));
});

// --- through the evaluator and the engine --- //

/** Run `START -> n -> END`, `n` reading `x` from the run input, and return the events. */
function runEvents(node: CodeNode, x: unknown): Promise<readonly RunEvent[]> {
  node.params = [new Parameter("x")];
  const flow = graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ], { wiring: { [node.id]: { x: "${input.x}" } } });
  return engine(flow, { inputs: { x } }).run();
}

/** Return the one `NodeFailed` among `events`. */
function nodeFailed(events: readonly RunEvent[]): NodeFailed {
  const failed = events.filter((event) => event instanceof NodeFailed);
  expect(failed).toHaveLength(1);
  return failed[0] as NodeFailed;
}

test("the evaluator returns a body's value and fails a throwing body at its line", async () => {
  expect(await drive(new CodeNode("n", { code: "return 6 * 7" }))).toEqual(new Output(42n));

  const node = locatedNode('const x = 1\nthrow new Error("boom")');
  const outcome = await drive(node);
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect(failure.code).toBe("node.execute");
  expect(failure.message).toBe("Error: boom");
  expect(failure.causeType).toBe("CodeError");
  expect(failure.trace.at(-1)).toBe(node.codeLineOrigins[1]!);
});

test("a run commits a body's value", async () => {
  const events = await runEvents(new CodeNode("n", { code: "return inputs.x * 2" }), 21n);

  const succeeded = events.filter((event) => event instanceof NodeSucceeded && event.nodeId === "n");
  expect(succeeded).toEqual([new NodeSucceeded({ nodeId: "n", output: 42n })]);
});

test("a run fails a throwing body at its line", async () => {
  const node = locatedNode('if (inputs.x > 0) {\n  throw new Error("boom")\n}\nreturn 0');

  const { failure } = nodeFailed(await runEvents(node, 1n));

  expect(failure.code).toBe("node.execute");
  expect(failure.message).toBe("Error: boom");
  expect(failure.trace.at(-1)).toEqual(node.codeLineOrigins[1]!);
});

test("a run fails a body with no value at the whole code origin", async () => {
  const node = locatedNode("const y = inputs.x");

  const { failure } = nodeFailed(await runEvents(node, 1n));

  expect(failure.code).toBe("node.execute");
  expect(failure.message).toBe("CODE body returned no value; return null for no value");
  expect(failure.trace.at(-1)).toEqual(node.codeOrigin!);
});

test("a run fails a result intake refuses with contract.output at the contract origin", async () => {
  const node = locatedNode("return { handle: new Date(0) }");
  const contractOrigin = new OriginRef("contract");
  node.outputContract = new ValueContract(Type.scalar(ValueKind.ANY), new ContractOrigins({ default: contractOrigin }));

  const { failure } = nodeFailed(await runEvents(node, 1n));

  expect(failure.code).toBe("contract.output");
  expect(failure.message).toBe("output at handle: Date is not a yian value");
  expect(failure.trace.at(-1)).toEqual(contractOrigin);
});

// --- TS-only rows --- //

test("a non-integer line selects no line", () => {
  const whole = new OriginRef("code");
  const lines = [new OriginRef("line:0"), new OriginRef("line:1")];

  for (const line of [1.5, "1", null]) {
    expect(selectInlineCodeOrigin(line, { codeOrigin: whole, codeLineOrigins: lines, nodeOrigin: null })).toBe(whole);
  }
});

test("the selector rejects origin carriers of the wrong shape", () => {
  const line = new OriginRef("line:0");
  const bad = "not-an-origin" as unknown as OriginRef;

  expect(() => selectInlineCodeOrigin(1, { codeOrigin: bad, codeLineOrigins: [], nodeOrigin: null })).toThrow(
    new TypeError("CODE source origin must be an OriginRef or null"),
  );
  expect(() => selectInlineCodeOrigin(1, { codeOrigin: null, codeLineOrigins: [], nodeOrigin: bad })).toThrow(
    new TypeError("node origin must be an OriginRef or null"),
  );
  expect(() =>
    selectInlineCodeOrigin(1, {
      codeOrigin: null,
      codeLineOrigins: line as unknown as OriginRef[],
      nodeOrigin: null,
    }),
  ).toThrow(new TypeError("CODE line origins must be an array"));
  expect(() => selectInlineCodeOrigin(1, { codeOrigin: null, codeLineOrigins: [line, bad], nodeOrigin: null })).toThrow(
    new TypeError("CODE line origins must contain only OriginRef values"),
  );
});

test("the constructor compiles nothing: a body that doesn't parse constructs", () => {
  const node = new CodeNode("n", { code: "return (" });

  expect(node.code).toBe("return (");
  expect(node.mode).toBe("inline");
});

test("the node's mode follows the classification", () => {
  expect(new CodeNode("r", { code: "./mod.ts:fn" }).mode).toBe("reference");
  expect(new CodeNode("i", { code: "return inputs.v" }).mode).toBe("inline");
  expect(new CodeNode("p", { code: "./helpers.ts" }).mode).toBe("inline");
});

test("the built-in identity form has builtin mode and keeps a frozen copy", () => {
  const form = { builtin: "identity" as const, field: "x" };
  const node = new CodeNode("x", { code: form });

  expect(node.mode).toBe("builtin");
  expect(node.code).not.toBe(form);
  expect(node.code).toEqual(form);
  expect(Object.isFrozen(node.code)).toBe(true);
});

test("a CODE built-in of the wrong shape is rejected", () => {
  const message = 'CODE built-in must be {builtin: "identity", field: <input name>}';
  for (const code of [
    { builtin: "other", field: "x" },
    { builtin: "identity" },
    { builtin: "identity", field: "x", extra: 1 },
    { builtin: "identity", field: 1 },
  ]) {
    expectError(() => new CodeNode("x", { code: code as never }), TypeError, message);
  }
  expectError(
    () => new CodeNode("x", { code: { builtin: "identity", field: "" } }),
    RangeError,
    "CODE built-in field must be non-empty",
  );
});

test("a CODE code that is neither text nor a built-in is rejected", () => {
  expectError(
    () => new CodeNode("x", { code: 42 as never }),
    TypeError,
    "CODE code must be text or a built-in operation, got number",
  );
  expectError(
    () => new CodeNode("x", { code: [] as never }),
    TypeError,
    "CODE code must be text or a built-in operation, got Array",
  );
  expectError(
    () => new CodeNode("x", { code: null as never }),
    TypeError,
    "CODE code must be text or a built-in operation, got null",
  );
});

test("a thrown string falls back to the whole code origin", async () => {
  const node = locatedNode('throw "boom"');

  expect(await failureOrigin(node)).toBe(node.codeOrigin);
});

test("a body error's code is node.execute", async () => {
  const node = locatedNode('throw new Error("boom")');

  expect(node.exceptionCode(await rejected(node.run({})))).toBe("node.execute");
});

test("a built-in's failure origin is the node's", async () => {
  const builtin = new CodeNode("v", { code: { builtin: "identity", field: "v" }, codeOrigin: new OriginRef("code") });
  builtin.origin = new OriginRef("node");

  expect(await failureOrigin(builtin)).toBe(builtin.origin);
});

test("TypeScript annotations and await run", async () => {
  const body = "const n: number = await Promise.resolve(inputs.n as number)\nreturn n + 1";
  expect(await run(body, { n: 1n })).toBe(2n);
});

test("ints arrive as numbers, big ones as bigints, and an integral result becomes an int", async () => {
  expect(await run("return [typeof inputs.a, typeof inputs.b]", { a: 1n, b: 2n ** 60n })).toEqual([
    "number",
    "bigint",
  ]);
  expect(await run("return 42")).toBe(42n);
  expect(await run("return 2.5")).toBe(2.5);
});

test("an undeclared input rejects with a CodeError", async () => {
  const error = await failure("return inputs.missing", { x: 1n });

  expect(error).toBeInstanceOf(CodeError);
  expect((error as Error).message).toBe("ReferenceError: CODE input 'missing' is not declared");
  expect((error as Error).name).toBe("CodeError");
  expect((error as CodeError).original).toBeInstanceOf(ReferenceError);
});

test("the inputs proxy answers in, JSON.stringify and a spread", async () => {
  expect(await run('return ["missing" in inputs, "x" in inputs]', { x: 1n })).toEqual([false, true]);
  expect(await run("return JSON.stringify(inputs)", { x: 1n })).toBe('{"x":1}');
  expect(await run("return { ...inputs, y: 2 }", { x: 1n })).toEqual({ x: 1n, y: 2n });
});

test("return inputs returns the record", async () => {
  expect(await run("return inputs", { x: 1n, s: "a" })).toEqual({ x: 1n, s: "a" });
});

test("assigning to inputs rejects (strict mode)", async () => {
  const error = await failure("inputs.x = 1\nreturn null", { x: 0n });

  expect(error).toBeInstanceOf(CodeError);
  expect((error as CodeError).original).toBeInstanceOf(TypeError);
});

test("a body returning undefined rejects with the no-value error", async () => {
  const error = await failure("const x = 1");

  expect(error).not.toBeInstanceOf(CodeError);
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).constructor).toBe(Error);
  expect((error as Error).message).toBe("CODE body returned no value; return null for no value");
  expect(await run("return null")).toBeNull();
});

test("a thrown string's text is the string, a thrown record's its text form", async () => {
  const text = await failure('throw "boom"');
  expect(text).toBeInstanceOf(CodeError);
  expect((text as Error).message).toBe("boom");
  expect((text as CodeError).original).toBe("boom");

  const record = await failure('throw { a: 1, b: ["x"] }');
  expect((record as Error).message).toBe('{"a":1,"b":["x"]}');

  const error = await failure('throw new RangeError("bad")');
  expect((error as Error).message).toBe("RangeError: bad");
});

test("the built-in identity returns the bound field unchanged", async () => {
  const node = new CodeNode("v", { code: { builtin: "identity", field: "v" } });
  const record = { k: [1n] };

  expect(await node.run({ v: 2n ** 70n })).toEqual(new Output(2n ** 70n));
  // No host round trip: a float 2 stays the number 2, not the int 2n.
  expect(await node.run({ v: 2 })).toEqual(new Output(2));
  expect(((await node.run({ v: record })) as Output).value).toBe(record);
});

test("the built-in identity rejects a missing field", async () => {
  const node = new CodeNode("v", { code: { builtin: "identity", field: "v" } });

  const error = await rejected(node.run({ other: 1n }));

  expect(error).toBeInstanceOf(ReferenceError);
  expect((error as Error).message).toBe("CODE input 'v' is not declared");
});

test("the runtime's function header is 2 lines", () => {
  const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => object;
  const text = new AsyncFunction("inputs", "/*body*/").toString();

  expect(text.slice(0, text.indexOf("/*body*/")).split("\n").length - 1).toBe(2);
});

// --- the worker thread --- //

test("a file input arrives as a FileRef and a returned FileRef commits a FileRef", async () => {
  const file = new FileRef({ uri: "file:///tmp/a.txt", mime: "text/plain" });

  const value = await run("return { f: inputs.f, uri: inputs.f.uri, mime: inputs.f.mime, name: inputs.f.name }", {
    f: file,
  });

  const { f, ...fields } = value as Record<string, unknown>;
  expect(fields).toEqual({ uri: "file:///tmp/a.txt", mime: "text/plain", name: null });
  expect(FileRef.isFileRef(f)).toBe(true);
  expect(f).toEqual(file);
});

test("a class instance, a getter and a function in a result fail contract.output", async () => {
  for (const [code, message] of [
    ["class Thing {}\nreturn { x: new Thing() }", "output at x: Thing is not a yian value"],
    ["return { get x() { return 1 } }", "output at x: accessor property is not a yian value"],
    ["return { x: () => 1 }", "output at x: function is not a yian value"],
  ] as const) {
    const node = new CodeNode("n", { code });

    const error = await rejected(node.run({}));

    expect(error).toBeInstanceOf(HostOutputError);
    expect((error as HostOutputError).message).toBe(message);
    expect((error as HostOutputError).path).toEqual(["x"]);
    expect(node.exceptionCode(error)).toBe("contract.output");
  }
});

test("a custom error class keeps its name and its line", async () => {
  const node = locatedNode('class MyError extends Error {\n  name = "MyError"\n}\nthrow new MyError("m")');

  const error = await rejected(node.run({}));

  expect(error).toBeInstanceOf(CodeError);
  expect((error as Error).message).toBe("MyError: m");
  expect(node.exceptionOrigin(error)).toBe(node.codeLineOrigins[3]!);
});

test("process.exit in a body rejects with the worker's exit code and this process lives on", async () => {
  const error = await failure("process.exit(3)\nreturn 1");

  expect((error as Error).constructor).toBe(Error);
  expect((error as Error).message).toBe("CODE worker exited with code 3 before the body returned");
  expect(await run("return 1")).toBe(1n);
});

test("a body awaiting a promise nothing resolves rejects rather than hangs", async () => {
  const error = await failure("await new Promise(() => {})\nreturn 1");

  expect((error as Error).message).toBe("CODE worker exited with code 0 before the body returned");
  expect(await run("return 1")).toBe(1n);
});

test("a late timer throw rejects as a CodeError", async () => {
  const error = await failure('setTimeout(() => { throw new RangeError("late") }, 1)\nawait new Promise(() => {})');

  expect(error).toBeInstanceOf(CodeError);
  expect((error as Error).message).toBe("RangeError: late");
  expect((error as CodeError).original).toBeInstanceOf(RangeError);
});

test("two serial calls reuse one thread", async () => {
  const body = "globalThis.yianCount = (globalThis.yianCount ?? 0) + 1\nreturn globalThis.yianCount";

  // An earlier row may have run this body on the popped thread, so only the step is fixed.
  const first = (await run(body)) as bigint;
  expect(await run(body)).toBe(first + 1n);
});

test("a worker shares the host's live environment both ways", async () => {
  try {
    // A call first, so the next one lands on a thread that started before the host's change.
    expect(await run("return 1")).toBe(1n);
    process.env.YIAN_TEST_D47_ENV = "later";
    expect(await run("return process.env.YIAN_TEST_D47_ENV ?? null")).toBe("later");

    expect(await run('process.env.YIAN_TEST_D47_FROM_BODY = "body"\nreturn 1')).toBe(1n);
    expect(process.env.YIAN_TEST_D47_FROM_BODY).toBe("body");
  } finally {
    delete process.env.YIAN_TEST_D47_ENV;
    delete process.env.YIAN_TEST_D47_FROM_BODY;
  }
});

test("a call after an aborted call still works", async () => {
  const controller = new AbortController();
  const reason = new Error("stop");
  const node = new CodeNode("n", { code: "await new Promise((r) => setTimeout(r, 60_000))\nreturn 1" });

  const pending = rejected(node.run({}, { signal: controller.signal }));
  setTimeout(() => controller.abort(reason), 50);

  expect(await pending).toBe(reason);
  expect(await run("return 2")).toBe(2n);
});

test("an input the clone refuses rejects, leaves no abort listener, and the next call works", async () => {
  const controller = new AbortController();
  const node = new CodeNode("n", { code: "return 1" });

  const error = await rejected(node.run({ f: () => 1 }, { signal: controller.signal }));

  expect((error as Error).name).toBe("DataCloneError");
  expect(getEventListeners(controller.signal, "abort")).toHaveLength(0);
  expect(await run("return 2")).toBe(2n);
});

test("a late error while the worker is idle prints a console line and doesn't fail the next call", async () => {
  const logged = spyOn(console, "error").mockImplementation(() => {});
  try {
    expect(await run('setTimeout(() => { throw new RangeError("later") }, 20)\nreturn 1')).toBe(1n);
    const line = "CODE error after its body returned: RangeError: later";
    for (let waited = 0; waited < 2000 && !logged.mock.calls.some(([text]) => text === line); waited += 10) {
      await sleep(10);
    }

    expect(logged).toHaveBeenCalledWith(line);
    expect(await run("return 2")).toBe(2n);
  } finally {
    logged.mockRestore();
  }
});

test("the worker uses the caller's withEnv values", async () => {
  const env = (name: string) => (name === "YIAN_MAX_SHOWN_VALUE_CHARS" ? "10" : undefined);
  const node = new CodeNode("n", { code: 'return { ["k".repeat(50)]: new Date(0) }' });

  const error = await rejected(withEnv(env, () => node.run({})));

  // In-process, intake cuts the path's key to 9 characters and `…` under this cap.
  expect((error as Error).message).toBe("output at kkkkkkkkk…: Date is not a yian value");
});

// --- stopping a body, and bodies that overlap --- //

test(
  "aborting the signal stops a body that never yields",
  async () => {
    const log = spyOn(console, "log").mockImplementation(() => {});
    try {
      const controller = new AbortController();
      const reason = new Error("stop");
      const node = new CodeNode("n", { code: 'console.log("busy")\nwhile (true) {}' });

      const pending = rejected(node.run({}, { signal: controller.signal }));
      for (let waited = 0; waited < 10_000 && !log.mock.calls.some(([text]) => text === "busy"); waited += 10) {
        await sleep(10);
      }
      const aborted = performance.now();
      controller.abort(reason);

      expect(log).toHaveBeenCalledWith("busy");
      expect(await pending).toBe(reason);
      expect(performance.now() - aborted).toBeLessThan(5000);
    } finally {
      log.mockRestore();
    }
  },
  30_000,
);

test("an already-aborted signal rejects before a worker starts", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped before");
  controller.abort(reason);

  expect(await rejected(new CodeNode("n", { code: "return 1" }).run({}, { signal: controller.signal }))).toBe(reason);
});

const BUILT_NODE = join(import.meta.dirname, "..", "..", "dist", "src", "nodes", "code", "node.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set value that
 * can't run, a Node older than the `engines` floor, or a missing build fails the whole file, so a
 * typo can't pass as a run with the Node case skipped. A copy of `terminal.test.ts`'s: importing a
 * test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT_NODE)) throw new Error(`${BUILT_NODE} is missing; run bun run build first`);
  return node;
})();

/**
 * Run a script with `command` that loads the CODE node from `module`, aborts a spinning body
 * when its forwarded `console.info` arrives and prints `{same, ms}`; expect the reason, a prompt
 * stop and a process that exits on its own.
 */
function expectStoppedProcessExits(command: string, args: readonly string[], module: string): void {
  const script = join(helpers, `stop-${command === process.execPath ? "bun" : "node"}.mjs`);
  writeFileSync(
    script,
    `const { CodeNode } = await import(${JSON.stringify(pathToFileURL(module).href)});
const controller = new AbortController();
const reason = new Error("stop");
let aborted = 0;
// The body's forwarded \`console.info\` says it is spinning: stop it then.
console.info = () => {
  aborted = performance.now();
  controller.abort(reason);
};
const code = 'console.info("busy"); while (true) {}';
const pending = new CodeNode("n", { code }).run({}, { signal: controller.signal });
let error;
try {
  await pending;
} catch (caught) {
  error = caught;
}
console.log(JSON.stringify({ same: error === reason, ms: performance.now() - aborted }));
`,
  );
  const result = spawnSync(command, [...args, script], { encoding: "utf8", timeout: 50_000 });
  const context = `signal: ${result.signal}\nstderr: ${result.stderr}\nstdout: ${result.stdout}`;
  expect(result.status, context).toBe(0);
  const { same, ms } = JSON.parse(result.stdout) as { same: boolean; ms: number };
  expect(same).toBe(true);
  expect(ms).toBeLessThan(5000);
}

describe("a stopped body leaves no worker: the process exits", () => {
  test.skipIf(runtime !== "bun")(
    "on Bun",
    () => {
      const source = join(import.meta.dirname, "..", "..", "src", "nodes", "code", "node.ts");
      expectStoppedProcessExits(process.execPath, ["--no-install"], source);
    },
    60_000,
  );

  test.skipIf(!NODE)("on Node, built", () => expectStoppedProcessExits(NODE!, [], BUILT_NODE), 60_000);
});

test(
  "pooled: a busy CODE node doesn't block a sibling",
  async () => {
    const flag = join(helpers, "flag");
    rmSync(flag, { force: true });
    const a = new CodeNode("a", {
      code: `const fs = await import("node:fs")
const end = Date.now() + 10000
while (!fs.existsSync(${JSON.stringify(flag)}) && Date.now() < end) {}
return fs.existsSync(${JSON.stringify(flag)})`,
    });
    const b = new CodeNode("b", {
      code: `await new Promise((resolve) => setTimeout(resolve, 50))
const fs = await import("node:fs")
fs.writeFileSync(${JSON.stringify(flag)}, "")
return true`,
    });
    const flow = graph([a, b], [
      [START_ID, "a"],
      [START_ID, "b"],
      ["a", END_ID],
      ["b", END_ID],
    ]);

    const events = await engine(flow, { numWorkers: 2 }).run();

    const succeeded = events.filter((event) => event instanceof NodeSucceeded && ["a", "b"].includes(event.nodeId));
    expect(succeeded.map((event) => (event as NodeSucceeded).output)).toEqual([true, true]);
  },
  30_000,
);

// --- the awaited load check --- //

test("the load check refuses a wrapper escape at stripping and runs none of it", async () => {
  const node = new CodeNode("n", { code: "}); globalThis.yianEscaped = true; (function () {" });

  expect(await rejected(node.check())).toBeInstanceOf(TypeStripError);
  expect("yianEscaped" in globalThis).toBe(false);
});

test("the load check refuses a wrapper escape that strips cleanly, by the constructor", async () => {
  // Closes the strip wrapper cleanly, so only the function constructor stands in its way.
  const node = new CodeNode("n", { code: "}\nglobalThis.yianEscaped = true\nfunction f() {" });

  const error = await rejected(node.check());

  expect((error as Error).constructor).toBe(SyntaxError);
  expect("yianEscaped" in globalThis).toBe(false);
});

test("the load check refuses a body redeclaring inputs, by the constructor", async () => {
  const error = await rejected(new CodeNode("n", { code: "let inputs = 1\nreturn inputs" }).check());

  expect((error as Error).constructor).toBe(SyntaxError);
});

test("the load check refuses a Python-style reference", async () => {
  for (const code of ["pkg.mod:fn", "mymod:myfunc"]) {
    const error = await rejected(new CodeNode("p", { code }).check());

    expect((error as Error).constructor).toBe(RangeError);
    expect((error as Error).message).toBe(
      `\`code:\` value '${code}' looks like a Python \`module:function\` reference; use \`./file.ts:fn\``,
    );
  }
});

test("the load check refuses a reference to a missing file", async () => {
  const error = await rejected(new CodeNode("m", { code: "./missing.ts:fn", baseDir: helpers }).check());

  expect((error as Error).constructor).toBe(RangeError);
  expect((error as Error).message).toBe(`\`code:\` file './missing.ts' does not exist in '${helpers}'`);
});

test("the load check passes a reference to an existing file, ../ included, and imports nothing", async () => {
  await new CodeNode("r", { code: "./compose_codefns.ts:echo", baseDir: FIXTURES }).check();
  const cases = join(FIXTURES, "..", "cases");
  await new CodeNode("p", { code: "../fixtures/compose_codefns.ts:echo", baseDir: cases }).check();
  // The module throws when it loads, so passing shows the check never imports it.
  await new CodeNode("b", { code: "./broken.mjs:run", baseDir: helpers }).check();
});

test("the load check never calls the body", async () => {
  await new CodeNode("t", { code: 'throw new Error("never")' }).check();
});

test("the load check passes a built-in", async () => {
  await new CodeNode("v", { code: { builtin: "identity", field: "v" } }).check();
});

// --- integration through the loader (framing and run) --- //

const INLINE_FLOW = `\
id: t
name: t
input:
  rating: !float
nodes:
  verdict:
    kind: code
    input:
      rating: \${input.rating}
    output: str
    code: |
      return "r=" + inputs['rating'].toFixed(1)
output: \${verdict.output}
`;

test("inline flow loads and runs", async () => {
  const result = await runFlow(await loadConformanceFlow(INLINE_FLOW), { rating: 3.0 }, { env: hostEnv(helpers) });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("r=3.0");
});

test("inline raise becomes failed run", async () => {
  const flow = INLINE_FLOW.replace(`return "r=" + inputs['rating'].toFixed(1)`, 'throw new Error("boom")');
  const result = await runFlow(await loadConformanceFlow(flow), { rating: 3.0 }, { env: hostEnv(helpers) });
  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.message).toContain("boom");
});
