/**
 * A run's copy of its flow shares the flow's types, so a deep `output:` record runs: a port of
 * `tests/engine/test_deep_type_copies.py`.
 *
 * Copying a compiled flow gives fresh nodes that share their types. A 100-level `output:` record
 * on a CODE node runs, runs learnably, and resumes under 10 nested `defs:` calls, and runs under
 * 50; a union nested at every level runs; and a deeper record allowed by a raised type limit fails
 * its output contract instead of overflowing.
 *
 * Ported: the copy rows, against `copyGraph` (the per-run copy `runFlow` makes, Python's
 * `copy.deepcopy` of a compiled flow), and the run rows. `deep_tree` is the TS fixture
 * `./fixtures/compose_codefns.ts:deep_tree`; the union row's CODE body is written in JS. Python
 * pins the recursion limit to a few hundred frames past the current depth around each run and
 * copy (`_stack_headroom`); TS has no such limit to pin, so the rows run on the default stack,
 * each run row with an explicit timeout. Python's `set_defaults(MAX_TYPE_NESTING=125)` is
 * `YIAN_MAX_TYPE_NESTING=125` in the load's and run's environment.
 *
 * Changed: `test_copying_a_compiled_flow_does_not_recurse_through_a_deep_record` checks only that
 * the copy is a new flow: there is no recursion limit to pin, and `copyGraph` never walks a type,
 * which `test_a_compiled_flow_copy_has_its_own_nodes_and_shares_their_types` checks.
 *
 * Not ported: `test_a_type_copies_as_itself` (TS has no copy operation on a `Type`; `copyGraph`
 * shares types instead of copying them).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { copyGraph } from "../../src/compile/model.ts";
import { outputContractType } from "../../src/nodes/contracts.ts";
import { type RunResult, resumeFlow, runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-deep-type-copies-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** A deep flow takes a few hundred milliseconds; the default 5 s leaves little room on a loaded machine. */
const TIMEOUT = 60_000;

const DEEP_TREE = "./fixtures/compose_codefns.ts:deep_tree";

/** Return a `{tree: {a: ... "str"}}` record type nesting `levels` records in total. */
const treeRecord = (levels: number) => "{tree: " + "{a: ".repeat(levels - 1) + '"str"' + "}".repeat(levels);

/** Return the value `deep_tree` makes for `n = levels - 1`, matching `treeRecord`. */
function treeValue(levels: number): Record<string, unknown> {
  let tree: unknown = "leaf";
  for (let i = 0; i < levels - 1; i++) tree = { a: tree };
  return { tree };
}

/** Return the lines of a `u` CODE node passing `n` through, with `output:` type `output`. */
const codeNode = (indent: string, output: string) => [
  `${indent}u:`,
  `${indent}  kind: code`,
  `${indent}  code: ${DEEP_TREE}`,
  `${indent}  input:`,
  `${indent}    n: \${input.n}`,
  `${indent}  output: ${output}`,
];

/**
 * Return a flow whose CODE node has a `levels`-level `output:` record under `calls` calls.
 *
 * With `calls = 0` the CODE node sits at the root. Otherwise the root calls `d0`, each `di` calls
 * `d(i+1)`, and the last def holds the CODE node.
 */
function flow(levels: number, calls: number): string {
  const output = treeRecord(levels);
  const lines = ["id: deep", "name: deep", "input:", "  n: !int"];
  if (calls === 0) {
    lines.push("nodes:", ...codeNode("  ", output), "output: ${u.output}");
    return lines.join("\n") + "\n";
  }
  lines.push("defs:");
  for (let i = 0; i < calls; i++) {
    lines.push(`  d${i}:`, "    input:", "      n: !int", "    nodes:");
    if (i < calls - 1) {
      lines.push("      u:", "        kind: call", `        call: d${i + 1}`, "        input:", "          n: ${input.n}");
    } else {
      lines.push(...codeNode("      ", output));
    }
    lines.push("    output: ${u.output}");
  }
  lines.push("nodes:", "  use:", "    kind: call", "    call: d0", "    input:", "      n: ${input.n}", "output: ${use.output}");
  return lines.join("\n") + "\n";
}

/** Assert `result` succeeded with `output`. */
function expectSucceeded(result: RunResult, output: unknown): void {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(output);
}

test("copying a compiled flow does not recurse through a deep record", async () => {
  const compiled = (await loadConformanceFlow(flow(100, 0), { env })).compiled;

  const copied = copyGraph(compiled);

  expect(copied).not.toBe(compiled);
});

test("a compiled flow copy has its own nodes and shares their types", async () => {
  const original = (await loadConformanceFlow(flow(3, 0), { env })).compiled;

  const copied = copyGraph(original);

  expect(copied.nodes.get("u")).not.toBe(original.nodes.get("u"));
  const typ = outputContractType(original.nodes.get("u")!.outputContract);
  expect(typ).not.toBeNull();
  expect(outputContractType(copied.nodes.get("u")!.outputContract)).toBe(typ);
});

test(
  "a deep record under nested calls runs",
  async () => {
    const loaded = await loadConformanceFlow(flow(100, 10), { env });

    expectSucceeded(await runFlow(loaded, { n: 99n }, { env, numWorkers: 0 }), treeValue(100));
  },
  TIMEOUT,
);

test(
  "a deep record under nested calls runs learnably and resumes",
  async () => {
    const text = flow(100, 10);
    const path = join(dir, "deep.jsonl");

    const first = await runFlow(await loadConformanceFlow(text, { env }), { n: 99n }, {
      env,
      log: path,
      learnable: true,
      numWorkers: 0,
    });
    expectSucceeded(first, treeValue(100));
    // Cut right after the last expansion so resume replays every call and runs the leaf.
    const lines = readFileSync(path, "utf8").split(/(?<=\n)/);
    const expanded = lines.flatMap((line, i) => (JSON.parse(line).kind === "node_expanded" ? [i] : []));
    expect(expanded).toHaveLength(10);
    writeFileSync(path, lines.slice(0, expanded.at(-1)! + 1).join(""));

    const resumed = await resumeFlow(await loadConformanceFlow(text, { env }), { log: path, env });

    expectSucceeded(resumed, treeValue(100));
  },
  TIMEOUT,
);

test(
  "a deep record at the call nesting limit runs",
  async () => {
    const loaded = await loadConformanceFlow(flow(100, 50), { env });

    expectSucceeded(await runFlow(loaded, { n: 99n }, { env, numWorkers: 0 }), treeValue(100));
  },
  TIMEOUT,
);

test(
  "a union nested at every level runs",
  async () => {
    const typedefs = ["typedefs:", "  S: {k: Literal(s)}", "  T0: {k: Literal(r), a: int}"];
    for (let i = 1; i < 100; i++) typedefs.push(`  T${i}: {k: Literal(r), a: 'Optional(Union(T${i - 1}, S))'}`);
    const text =
      [
        "id: deep",
        "name: deep",
        "input:",
        "  n: !int",
        "nodes:",
        "  u:",
        "    kind: code",
        "    input:",
        "      n: ${input.n}",
        "    output: T99",
        "    code: |",
        "      let v = {'k': 's'}",
        "      for (let i = 0; i < inputs['n']; i++)",
        "        v = {'k': 'r', 'a': v}",
        "      return v",
        "output: ${u.output}",
        ...typedefs,
      ].join("\n") + "\n";
    let expected: Record<string, unknown> = { k: "s" };
    for (let i = 0; i < 99; i++) expected = { k: "r", a: expected };
    const loaded = await loadConformanceFlow(text, { env });

    expectSucceeded(await runFlow(loaded, { n: 99n }, { env, numWorkers: 0 }), expected);
  },
  TIMEOUT,
);

test(
  "a record past the value limit fails its output contract",
  async () => {
    const raised = hostEnv(dir, { YIAN_MAX_TYPE_NESTING: "125" });
    const loaded = await loadConformanceFlow(flow(125, 10), { env: raised });

    const result = await runFlow(loaded, { n: 124n }, { env: raised, numWorkers: 0 });

    expect(result.status).toBe("failed");
    expect(result.failure!.code).toBe("contract.output");
    expect(result.failure!.message).toContain("nests deeper than");
  },
  TIMEOUT,
);
