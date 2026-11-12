/**
 * `yian run`'s structured input parsing, its prompt cancellation, and run-error rendering
 * (`renderRunError` over the seeds' conformance cases). The banner, input labels and prompt widgets
 * are tested in `./run.test.ts`, `./terminal.test.ts` and
 * `./terminal_both_runtimes.ts`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { parseKv, renderRunError } from "../../src/cli/run.ts";
import { Terminal, parseStructuredInput } from "../../src/cli/terminal.ts";
import type { Writer } from "../../src/cli/utils.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Diagnostic, Failure, OriginRef } from "../../src/diagnostics.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { RunResult, runFlow } from "../../src/run.ts";
import { YianRangeError } from "../../src/typesys/errors.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";
import { caseText, linkConformanceFixtures, loadConformanceFlow } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-cli-prompt-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

// Seed e25's `uses:` child, on the search path beside the conformance fixtures its CODE names.
const libBoom = caseText("run-map-call/seed_e25_external_raise", { file: "lib_boom.yaml" }).text;
writeFileSync(join(dir, "lib_boom.yaml"), libBoom);
linkConformanceFixtures(dir);

/** The env the AGENT seeds run under: a default model, since `hostEnv` drops the shell's. */
const AGENT_ENV = hostEnv(dir, { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" });

/** Load conformance case `id`'s flow labelled `label`, with this file's directory on the search path. */
function loadSeed(id: string, label: string, runEnv = env): Promise<ComposedFlow> {
  return loadConformanceFlow(caseText(id).text, { searchPaths: [dir], sourceLabel: label, env: runEnv });
}

/** What `renderRunError` writes for `result` against `flow`. */
async function rendered(result: RunResult, flow: ComposedFlow, engineTrace = false): Promise<string> {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  await renderRunError(result, flow, engineTrace, out, env);
  return chunks.join("");
}

test("parse structured input does not hide validator faults", () => {
  // `new Type` doesn't validate its bounds, so the fault surfaces in `buildValueAs`.
  const malformed = new Type({
    kind: ValueKind.LIST_INTEGER,
    element: new Type({ kind: ValueKind.INTEGER, bounds: [5n, 1n] }),
  });

  let captured: unknown;
  try {
    parseStructuredInput("[3]", malformed, "x");
  } catch (error) {
    captured = error;
  }

  // yian's own RangeError, not the StructuredInputError (a subclass of it) a verdict would be.
  expect(captured).toBeInstanceOf(RangeError);
  expect((captured as RangeError).constructor).toBe(YianRangeError);
  expect((captured as RangeError).name).toBe("RangeError");
  expect((captured as RangeError).message).toBe("type bounds must be ordered from minimum to maximum");
});

test("parse kv keeps structured looking values as strings", () => {
  const parsed = parseKv(["items=[1, 2]", 'profile={"count": 2}']);

  expect(parsed).toEqual({ items: "[1, 2]", profile: '{"count": 2}' });
  expect(Object.values(parsed).every((value) => typeof value === "string")).toBe(true);
});

test("structured prompt cancellation bypasses validation", async () => {
  const input = new FakeInput();
  const output = new FakeOutput();
  const terminal = new Terminal({ input, output });
  const items = new Parameter("items", {
    contract: new ValueContract(Type.scalar(ValueKind.LIST_STRING)),
    required: true,
    coerceStrings: true,
  });
  const gathered = terminal.promptInputs([items], {});

  // An unparseable draft, then Ctrl-C: the draft is never validated.
  input.write("[\x03");

  expect(await gathered).toBeNull();
  expect(output.text).not.toContain("Invalid JSON");
});

// --- renderRunError: a runtime failure boxes the .yaml at the precise line --- //

test("run error uses loaded canonical diagnosis", async () => {
  const failure = new Failure({
    code: "node.execute",
    message: "semantic failure",
    nodeId: "work",
    trace: [new OriginRef("site")],
  });
  const result = new RunResult({ input: {}, status: "failed", runId: "cli-test", log: null, failure });
  const seen: Failure[] = [];
  const loaded = {
    diagnose(candidate: Failure) {
      seen.push(candidate);
      return new Diagnostic({
        code: candidate.code,
        message: candidate.message,
        notes: ["canonical presentation"],
        nodeId: candidate.nodeId,
      });
    },
  };

  const out = await rendered(result, loaded as unknown as ComposedFlow);

  expect(seen).toEqual([failure]);
  expect(seen[0]).toBe(failure);
  expect(out).toContain("semantic failure");
  expect(out).toContain("canonical presentation");
});

test("run error boxes failing node", async () => {
  // e07 omits `as_of`, so the canonical failure trace points at the failing `:?` binding.
  const loaded = await loadSeed("run-core/seed_e07_required_missing", "e07-required-missing.yaml", AGENT_ENV);
  const result = await runFlow(loaded, { topic: "X" }, { env: AGENT_ENV });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e07-required-missing.yaml:23"); // the as_of binding, not the node header
  expect(out).toContain("report"); // still inside the source window
  expect(out).toContain("as_of is required for the report");
});

test("run error uses reference code origin", async () => {
  // A referenced CODE failure carries its authored `code:` scalar in the canonical origin trace.
  const loaded = await loadSeed("run-leaves/seed_e20_code_raises", "e20-code-raises.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e20-code-raises.yaml:12");
  expect(out).toContain("╭");
});

test("run error boxes boundary assert line", async () => {
  // e18's boundary assert fails before any node runs, at the exact `asserts:` expression.
  const loaded = await loadSeed("run-core/seed_e18_false_boundary_assert", "e18-false-boundary-assert.yaml");
  const result = await runFlow(loaded, { window: -5n }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e18-false-boundary-assert.yaml:15"); // the `input.window > 0` line
  expect(out).toContain("assert failed");
});

test("run error boxes code wrong type output", async () => {
  // e21's CODE returns a value its declared `output: int` refuses, at the `output:` declaration.
  const loaded = await loadSeed("run-leaves/seed_e21_code_wrong_type", "e21-code-wrong-type.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e21-code-wrong-type.yaml:12"); // the `output: int` field line
  expect(out).toContain("╭");
});

test("run error boxes flow input line", async () => {
  // e08's `window` can't coerce to int, so its origin selects the declaration.
  const loaded = await loadSeed("run-core/seed_e08_input_type_mismatch", "e08-input-type-mismatch.yaml", AGENT_ENV);
  const result = await runFlow(loaded, { topic: "X", window: "soon" }, { env: AGENT_ENV });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e08-input-type-mismatch.yaml:20"); // the `window: int` declaration line
  expect(out).toContain("╭");
});

test("run error plain when unlocatable", async () => {
  // A failure with no authored origin stays a semantic message with no frame.
  const result = new RunResult({
    input: {},
    status: "failed",
    runId: "cli-test",
    log: null,
    failure: new Failure({ code: "graph.terminal_unreachable", message: "boom", nodeId: null }),
    events: [],
  });
  const loaded = await loadSeed("run-core/seed_e18_false_boundary_assert", "e18-false-boundary-assert.yaml");
  const out = await rendered(result, loaded);
  expect(out.trim()).toBe("boom");
  expect(out).not.toContain("╭"); // no boxed frame
});

test("run error renders namespaced failure origin trace", async () => {
  // The failure trace carries the owning CALL site despite the namespaced node id (`run/boom`).
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e24-nested-code-raise.yaml:27"); // the owning `run:` call node line
  expect(out).toContain("╭"); // boxed, not a plain `run failed:` line
});

test("run error boxes loop max exceeded at loop node", async () => {
  // The LOOP origin stays put even when the runtime driver id carries a respawn suffix (`polish~5`).
  const loaded = await loadFlow(caseText("load-loop/seed_28_refine_loop").text, {
    sourceLabel: "28-refine-loop.yaml",
    env,
  });
  const loopOrigin = loaded.compiled.nodes.get("polish")!.origin;
  expect(loopOrigin).not.toBeNull();
  const result = new RunResult({
    input: {},
    status: "failed",
    runId: "cli-test",
    log: null,
    failure: new Failure({
      code: "node.execute",
      message: "loop 'polish' exceeded max (5)",
      nodeId: "polish~5",
      trace: [loopOrigin!],
      causeType: "LoopMaxExceeded",
    }),
    events: [],
  });
  const out = await rendered(result, loaded);
  expect(out).toContain("╭"); // boxed, not a plain `run failed:` line
  expect(out).toContain("28-refine-loop.yaml:45"); // the authored `polish:` loop node
  expect(out).toContain("exceeded max (5)");
});

test("run error traceback only under engine trace", async () => {
  // The captured stack of a CODE throw shows only under `engineTrace`.
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.failure).not.toBeNull();
  expect(result.failure!.traceback).toBeTruthy(); // captured at the node-failure boundary

  expect(await rendered(result, loaded, false)).not.toContain("─ traceback ");

  const out = await rendered(result, loaded, true);
  expect(out).toContain("─ traceback ");
  expect(out).toContain("intentional CODE failure"); // the originating error
});

// --- renderRunError: ordered canonical frames across child flows --- //

test("run error multi frame stack into def", async () => {
  // e24's trace resolves the top-level CALL site, then the failing CODE origin inside `defs:inner`.
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("e24-nested-code-raise.yaml:27"); // frame 0: the owning `run:` call node
  expect(out).toContain("e24-nested-code-raise.yaml:20"); // frame 1: the authored `code:` value
  expect(out).toContain("intentional CODE failure");
});

test("run error multi frame stack into external", async () => {
  // e25's origins cross the file boundary: the CALL site here, the failing CODE in lib_boom.yaml.
  const loaded = await loadSeed("run-map-call/seed_e25_external_raise", "e25-external-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("e25-external-raise.yaml:19"); // frame 0: the `go:` call node
  expect(out).toContain("lib_boom.yaml:18"); // frame 1: the external file's `code:` line
});

test("run error multi frame three levels", async () => {
  // e26's origin trace resolves all three levels, outermost to innermost.
  const loaded = await loadSeed("run-map-call/seed_e26_three_level_raise", "e26-three-level-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("e26-three-level-raise.yaml:37"); // frame 0: the top `outer:` call node
  expect(out).toContain("e26-three-level-raise.yaml:29"); // frame 1: `via:` call in `middle`
  expect(out).toContain("e26-three-level-raise.yaml:20"); // frame 2: `boom` code in `deep`
});

test("run error multi frame on nested human input failure", async () => {
  // Seed 25 asks from inside a called `defs:` child (`gate/approve`); an answer
  // outside `Approval` fails the typed write, framed at the `gate:` CALL, then the child's `output:`.
  const loaded = await loadSeed("run-map-call/seed_25_nested_suspension", "25-nested-suspension.yaml");
  const result = await runFlow(loaded, { action: "ship it" }, { env, tools: { ask: () => "maybe" } });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("25-nested-suspension.yaml:41"); // frame 0: the `gate:` call node
  expect(out).toContain("25-nested-suspension.yaml:31"); // frame 1: the precise `output:` line
  expect(out).toContain("is not one of Literal(");
});

test("run error unavailable presentation stays unlocated", async () => {
  // A missing resolver can't be bypassed by inferring locations from the graph.
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const unresolved = new ComposedFlow(loaded.compiled, {
    version: loaded.version,
    name: loaded.name,
    description: loaded.description,
    importDepth: loaded.importDepth,
  });
  const out = await rendered(result, unresolved);
  expect(out).not.toContain("╭");
  expect(out).toContain("Source presentation was unavailable.");
  expect(out).toContain("intentional CODE failure");
});

const ANY_INPUT_FLOW = `id: echo
name: echo
input:
  x: !Any
nodes:
  keep:
    kind: code
    input:
      x: \${input.x}
    code: |
      return inputs["x"]
    output: !Any
output: \${keep.output}
`;

test("any input from the command line reaches the flow as text", async () => {
  const result = await runFlow(await loadFlow(ANY_INPUT_FLOW, { env }), parseKv(["x=3"]), { env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("3");
});
