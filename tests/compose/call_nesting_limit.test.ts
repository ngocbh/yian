/** Call nesting and child resolution. */

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import {
  callNesting,
  callNestingScope,
  type CallableResolver,
  type ResolvedChild,
  resolveChild,
} from "../../src/compose/nodes/resolver.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget, reload } from "../../src/defaults.ts";
import { NodeStarted } from "../../src/events.ts";
import { advance, runFlow } from "../../src/run.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  linkConformanceFixtures,
  loadError,
  loadConformanceFlow,
  loadErrorAsync,
  runCase,
} from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

// The runs' isolated environment and logs.
const runDir = mkdtempSync(join(tmpdir(), "yian-call-nesting-"));
const env = hostEnv(runDir);
afterAll(() => rmSync(runDir, { recursive: true, force: true }));

/** Return the ids of the nodes `events` saw start. */
function started(events: readonly unknown[]): string[] {
  return events.flatMap((event) => (event instanceof NodeStarted ? [event.nodeId] : []));
}

/** A node whose only static child is `child`, counting how often it is asked. */
class FakeNode {
  calls = 0;
  readonly child: object | null;

  constructor(child: object | null) {
    this.child = child;
  }

  originChildren(): readonly unknown[] {
    this.calls += 1;
    return this.child !== null ? [this.child] : [];
  }
}

/** A flow with the given nodes keyed by position. */
class FakeFlow {
  readonly nodes: Map<string, FakeNode>;

  constructor(nodes: FakeNode[]) {
    this.nodes = new Map(nodes.map((node, i) => [`n${i}`, node]));
  }
}

/** Return a flow nesting `levels` bodies, one per level. */
function chain(levels: number): FakeFlow {
  let flow = new FakeFlow([]);
  for (let i = 0; i < levels; i++) flow = new FakeFlow([new FakeNode(flow)]);
  return flow;
}

/** Return a resolved child whose compiled flow nests `levels` bodies. */
function resolved(levels: number): ResolvedChild {
  return { compiled: chain(levels) as unknown as CompiledFlow, input: [] };
}

/** Return the error `fn` throws. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

/** Run `fn` inside `depth` nested calls of `resolveChild`, the innermost calling `fn`. */
function nestedCalls<T>(depth: number, fn: () => T): T {
  if (depth === 0) return fn();
  let result!: T;
  const resolver: CallableResolver = () => {
    result = nestedCalls(depth - 1, fn);
    return resolved(0);
  };
  resolveChild(`n${depth}`, `c${depth}`, resolver);
  return result;
}

/** Return the error resolving `n`'s callable `c` to `child` throws. */
function resolveError(child: ResolvedChild): unknown {
  return thrown(() => resolveChild("n", "c", () => child));
}

beforeEach(() => forget());
afterEach(() => forget());

test("call nesting counts levels, not paths", () => {
  // Each flow's two nodes share the next flow, so there are 2**40 paths but 40 levels.
  let flow = new FakeFlow([]);
  const nodes: FakeNode[] = [];
  for (let i = 0; i < 40; i++) {
    const pair = [new FakeNode(flow), new FakeNode(flow)];
    nodes.push(...pair);
    flow = new FakeFlow(pair);
  }

  expect(callNesting(flow)).toBe(40);
  expect(nodes.reduce((sum, node) => sum + node.calls, 0)).toBe(80);
  expect(nodes.every((node) => node.calls === 1)).toBe(true);
});

describe("resolveChild", () => {
  test("returns the resolved child", () => {
    const child = resolved(3);
    const asked: unknown[][] = [];
    const got = resolveChild("n", "c", (...args) => {
      asked.push(args);
      return child;
    });
    expect(got).toBe(child);
    expect(asked).toEqual([["c"]]);
  });

  test("a LoadError from the resolver passes through", () => {
    const inner = new LoadDiagnostics().error("inner failure");
    const error = thrown(() =>
      resolveChild("n", "c", () => {
        throw inner;
      }),
    );
    expect(error).toBe(inner);
  });

  test("an error that isn't a LoadError propagates unchanged, as an engine bug", () => {
    const inner = new TypeError("undefined is not an object");
    const error = thrown(() =>
      resolveChild("n", "c", () => {
        throw inner;
      }),
    );
    expect(error).toBe(inner);
  });

  test("a null resolver can't resolve the callable", () => {
    const error = thrown(() => resolveChild("c", "child", null));
    expect(error).toBeInstanceOf(LoadError);
    expect((error as LoadError).message).toBe(
      "call node 'c': could not resolve callable 'child': no child resolver was given",
    );
  });

  test("a child label names the callable", () => {
    const error = thrown(() => resolveChild("n", "c", null, "the def 'c'"));
    expect((error as LoadError).message).toBe(
      "call node 'n': could not resolve the def 'c': no child resolver was given",
    );
  });

  test("the count at the limit fails before resolving", () => {
    let asked = 0;
    const error = callNestingScope(() =>
      nestedCalls(50, () =>
        thrown(() =>
          resolveChild("deep", "c", () => {
            asked += 1;
            return resolved(0);
          }),
        ),
      ),
    );
    expect(asked).toBe(0);
    expect((error as LoadError).message).toBe(
      "call node 'deep': calling callable 'c' nests calls more than 50 levels deep, past the call " +
        "nesting limit of 50 (each call, map, or loop body is a level); flatten the chain",
    );
  });

  test("one below the limit still resolves", () => {
    const child = resolved(0);
    expect(callNestingScope(() => nestedCalls(49, () => resolveChild("n", "c", () => child)))).toBe(child);
  });

  test("a child built deeper than the limit fails after resolving", () => {
    expect(resolveChild("n", "c", () => resolved(49)).compiled).toBeDefined();
    const error = resolveError(resolved(50));
    expect((error as LoadError).message).toBe(
      "call node 'n': calling callable 'c' nests calls 51 levels deep, past the call nesting " +
        "limit of 50 (each call, map, or loop body is a level); flatten the chain",
    );
  });

  test("the post-check counts the active calls", () => {
    const error = callNestingScope(() => nestedCalls(10, () => resolveError(resolved(40))));
    expect((error as LoadError).message).toContain("nests calls 51 levels deep");
  });

  test("the limit follows YIAN_MAX_CALL_NESTING", () => {
    reload((name) => (name === "YIAN_MAX_CALL_NESTING" ? "3" : undefined));
    expect(resolveChild("n", "c", () => resolved(2)).compiled).toBeDefined();
    const after = resolveError(resolved(3));
    expect((after as LoadError).message).toContain(
      "nests calls 4 levels deep, past the call nesting limit of 3",
    );
    const before = callNestingScope(() => nestedCalls(3, () => resolveError(resolved(0))));
    expect((before as LoadError).message).toContain(
      "more than 3 levels deep, past the call nesting limit of 3",
    );
  });
});

test.each([
  ["map", new MapDescriptor({ id: "m", call: "c", over: "${input.xs}" })],
  ["loop", new LoopDescriptor({ id: "m", call: "c", times: 1n })],
])("a %s whose child already nests at the limit fails, named by its kind", (kind, desc) => {
  const ctx = new BuildContext({ registry: new Map(), resolver: () => resolved(50) });

  const error = thrown(() => desc.build(ctx));

  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).message).toBe(
    `${kind} node 'm': calling callable 'c' nests calls 51 levels deep, past the call nesting ` +
      "limit of 50 (each call, map, or loop body is a level); flatten the chain",
  );
});

/** Return a flow whose root `top` nests `n` inline call bodies around a CODE leaf. */
function inlineChain(n: number): string {
  let body = "kind: code\ncode: ./fixtures/compose_codefns.ts:echo\ninput:\n  topic: ${input.x}\noutput: str\n";
  for (let i = 0; i < n; i++) {
    body =
      "kind: call\ninput:\n  x: ${input.x}\nnodes:\n  step:\n" +
      body.replace(/^(?=.)/gm, "    ") +
      "output: ${step.output}\n";
  }
  const header = `id: inline${n}\nname: inline${n}\ninput:\n  x: !str\nnodes:\n  top:\n`;
  return `${header}${body.replace(/^(?=.)/gm, "    ")}output: \${top.output}\n`;
}

test("inline bodies count", async () => {
  const result = await runFlow(await loadConformanceFlow(inlineChain(50)), { x: "hi" }, { numWorkers: 0, env });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("hi");

  const text = inlineChain(51);
  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "call node 'step': calling inline body of node 'step' nests calls more than 50 levels deep, past the " +
      "call nesting limit of 50 (each call, map, or loop body is a level); flatten the chain",
  );
  // An inline body is the callable its node calls, so the error sits on the innermost call
  // node's `nodes:` body.
  let innermost = SourceDocument.parse(text).root.child("nodes").child("top");
  for (let i = 0; i < 50; i++) innermost = innermost.child("nodes").child("step");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(innermost.rangeOf("nodes")!);
});

test.each<CaseRow>([{ id: "load-map-call/compose_call_nesting_limit__inline_bodies_count", call: assembled }])(
  "assemble: $id",
  runCase,
);

describe("callNestingScope", () => {
  test("counts from zero and restores the count after a throw", () => {
    reload((name) => (name === "YIAN_MAX_CALL_NESTING" ? "2" : undefined));
    const outcome = nestedCalls(2, () => {
      // A fresh scope counts from zero, so two more levels fit inside the outer two.
      expect(callNestingScope(() => nestedCalls(2, () => "inner"))).toBe("inner");
      expect(() =>
        callNestingScope(() =>
          nestedCalls(1, () => {
            throw new Error("boom");
          }),
        ),
      ).toThrow("boom");
      // Back in the outer scope at the limit: a third level fails before resolving.
      return resolveError(resolved(0));
    });
    expect((outcome as LoadError).message).toContain("more than 2 levels deep");
    // Outside every call the count is zero again.
    expect(nestedCalls(2, () => "done")).toBe("done");
  });
});

// ---------- chains of uses: files, through the loader ----------

const IMPORT_51 = "makes a chain of uses: files 51 deep, past the call nesting limit of 50";

const ROOT_LEAF =
  "  step:\n" +
  "    kind: code\n" +
  "    code: ./fixtures/compose_codefns.ts:echo\n" +
  "    input:\n" +
  "      topic: ${input.x}\n" +
  "    output: str\n" +
  "output: ${step.output}\n";

/** Return a flow `name` that imports `uses` without calling them; its node is a CODE leaf. */
function importing(name: string, uses: Record<string, string>): string {
  let head = `id: ${name}\nname: ${name}\ninput:\n  x: !str\n`;
  const entries = Object.entries(uses);
  if (entries.length > 0) head += "uses:\n" + entries.map(([alias, ref]) => `  ${alias}: ${ref}\n`).join("");
  return head + "nodes:\n" + ROOT_LEAF;
}

/**
 * Return flows `u0`..`u(n-1)` by name, each importing the next as `nxt`. With `calls`, each file
 * but the last calls `nxt`; otherwise every node is a CODE leaf. The last file is a CODE leaf that
 * imports `lastImports` as `nxt` when given.
 */
function chainTexts(
  n: number,
  options: { prefix?: string; calls?: boolean; lastImports?: string | null } = {},
): Record<string, string> {
  const { prefix = "u", calls = true, lastImports = null } = options;
  const texts: Record<string, string> = {};
  for (let i = 0; i < n; i++) {
    const name = `${prefix}${i}`;
    const target = i < n - 1 ? `${prefix}${i + 1}` : lastImports;
    if (calls && i < n - 1) {
      texts[name] =
        `id: ${name}\nname: ${name}\ninput:\n  x: !str\n` +
        `uses:\n  nxt: ${target}\nnodes:\n` +
        "  step:\n    kind: call\n    call: nxt\n    input:\n      x: ${input.x}\n" +
        "output: ${step.output}\n";
    } else {
      texts[name] = importing(name, target !== null ? { nxt: target } : {});
    }
  }
  return texts;
}

/**
 * Write each flow text to `<directory>/<name>.yaml`, linking `<directory>/fixtures` to the
 * conformance fixtures: a file read through the search path resolves its CODE references against
 * its own directory.
 */
function writeFiles(directory: string, texts: Record<string, string>): void {
  linkConformanceFixtures(directory);
  for (const [name, text] of Object.entries(texts)) writeFileSync(join(directory, `${name}.yaml`), text);
}

/**
 * Write a chain of `n` flow files and return a root flow importing the first as `first`; with
 * `calls` (the default) each file and the root call their import.
 */
function usesFiles(directory: string, n: number, calls = true): string {
  writeFiles(directory, chainTexts(n, { calls }));
  if (!calls) return importing("root", { first: "u0" });
  return (
    "id: root\nname: root\ninput:\n  x: !str\nuses:\n  first: u0\nnodes:\n" +
    "  top:\n    kind: call\n    call: first\n    input:\n      x: ${input.x}\n" +
    "output: ${top.output}\n"
  );
}

/** Return the error for `u(n-2)`'s import of `u(n-1)`, making the chain `n` files deep. */
function importMessage(n: number, limit = 50): string {
  return (
    `callable 'nxt' (uses: 'u${n - 1}'): importing it makes a chain of uses: files ` +
    `${n} deep, past the call nesting limit of ${limit} (a chain of uses: imports is ` +
    "held to the same limit as nested call, map, and loop bodies); flatten the chain"
  );
}

/** Return the 1-based number of the first line whose trimmed text is `line`. */
function lineOf(text: string, line: string): number {
  return text.split("\n").findIndex((authored) => authored.trim() === line) + 1;
}

/** Expect a load error's innermost frame to be file `label`'s `nxt: <target>` alias. */
function expectRefusedAtAlias(error: LoadError, label: string, text: string, target: string): void {
  const frame = error.diagnostic.frames.at(-1)!;
  expect(frame.label).toBe(label);
  expect(frame.note).toBe("uses alias 'nxt'");
  expect(frame.range).toEqual(SourceDocument.parse(text).root.child("uses").child("nxt").range!);
  expect(frame.range.startLine).toBe(lineOf(text, `nxt: ${target}`));
}

/** Return a file `x` over an `underX`-file chain, and a 40-file chain ending in `x`. */
function sharedFileTexts(underX: number): Record<string, string> {
  return {
    ...chainTexts(underX, { prefix: "c", calls: false }),
    x: importing("x", { nxt: "c0" }),
    ...chainTexts(40, { prefix: "a", calls: false, lastImports: "x" }),
  };
}

const ORDERS: [string, Record<string, string>][] = [
  ["x-first", { x: "x", a: "a0" }],
  ["chain-first", { a: "a0", x: "x" }],
];

/** Return a resolver loading each named text once, through itself, and reusing it. */
function memoizingResolver(texts: Record<string, string>) {
  const memo = new Map<string, ComposedFlow>();
  const resolve = async (path: string): Promise<ComposedFlow> => {
    let flow = memo.get(path);
    if (flow === undefined) {
      flow = await loadConformanceFlow(texts[path]!, { childResolver: resolve, sourceLabel: `${path}.yaml` });
      memo.set(path, flow);
    }
    return flow;
  };
  return resolve;
}

describe("chains of uses: files", () => {
  let tmp: string;

  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), "yian-uses-chain-"));
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  /** Make and return the directory `<tmp>/<name>`. */
  const dir = (name: string) => {
    const path = join(tmp, name);
    mkdirSync(path);
    return path;
  };

  test("uses: files count", async () => {
    const legal = dir("legal");
    const flow = await loadConformanceFlow(usesFiles(legal, 50), { searchPaths: [legal] });
    const result = await runFlow(flow, { x: "hi" }, { numWorkers: 0, env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe("hi");

    const deep = dir("deep");
    const error = await loadErrorAsync(loadConformanceFlow(usesFiles(deep, 51), { searchPaths: [deep] }));
    // The import is refused before the call check runs on the built chain.
    expect(error.message).toBe(importMessage(51));
    expectRefusedAtAlias(error, "u49.yaml", readFileSync(join(deep, "u49.yaml"), "utf8"), "u50");
  });

  test("a long uses: chain fails at the import past the limit", async () => {
    const text = usesFiles(tmp, 200);
    const error = await loadErrorAsync(loadConformanceFlow(text, { searchPaths: [tmp] }));

    expect(error.message).toBe(importMessage(51));
    const frames = error.diagnostic.frames;
    expect(frames.length).toBe(51);
    expectRefusedAtAlias(error, "u49.yaml", readFileSync(join(tmp, "u49.yaml"), "utf8"), "u50");
    expect(frames[0]!.label).toBe("<flow>");
    expect(frames[0]!.range.startLine).toBe(lineOf(text, "first: u0"));
  });

  test("an import-only chain counts", async () => {
    const legal = dir("legal");
    await loadConformanceFlow(usesFiles(legal, 50, false), { searchPaths: [legal] });

    const deep = dir("deep");
    const error = await loadErrorAsync(loadConformanceFlow(usesFiles(deep, 60, false), { searchPaths: [deep] }));
    expect(error.message).toBe(importMessage(51));
    expectRefusedAtAlias(error, "u49.yaml", readFileSync(join(deep, "u49.yaml"), "utf8"), "u50");
  });

  test.each(ORDERS)("the import depth doesn't depend on load order (%s)", async (_order, uses) => {
    // Through the 40-file chain, x's 10-file chain ends 51 files deep; loading x first caches
    // it, so the chain then reaches x from the cache.
    const deep = dir("deep");
    writeFiles(deep, sharedFileTexts(10));
    const error = await loadErrorAsync(loadConformanceFlow(importing("root", uses), { searchPaths: [deep] }));
    expect(error.message).toContain(IMPORT_51);

    const legal = dir("legal");
    writeFiles(legal, sharedFileTexts(9));
    await loadConformanceFlow(importing("root", uses), { searchPaths: [legal] });
  });

  test("an import chain follows the configured limit", async () => {
    reload((name) => (name === "YIAN_MAX_CALL_NESTING" ? "5" : undefined));
    const legal = dir("legal");
    await loadConformanceFlow(usesFiles(legal, 5, false), { searchPaths: [legal] });

    const deep = dir("deep");
    const error = await loadErrorAsync(loadConformanceFlow(usesFiles(deep, 6, false), { searchPaths: [deep] }));
    expect(error.message).toBe(importMessage(6, 5));
    expectRefusedAtAlias(error, "u4.yaml", readFileSync(join(deep, "u4.yaml"), "utf8"), "u5");
  });

  test("a loaded flow records its import depth", async () => {
    const flow = await loadConformanceFlow(usesFiles(tmp, 3, false), { searchPaths: [tmp] });
    expect(flow.importDepth).toBe(3);
    expect((await loadConformanceFlow(importing("f", {}))).importDepth).toBe(0);
  });

  test("a cross-file cycle longer than the limit reports the depth limit", async () => {
    // The cycle is 51 files deep before it closes, so the author sees the depth error.
    writeFiles(tmp, chainTexts(60, { calls: false, lastImports: "u0" }));
    const error = await loadErrorAsync(loadConformanceFlow(importing("root", { first: "u0" }), { searchPaths: [tmp] }));
    expect(error.message).toBe(importMessage(51));
  });
});

describe("custom resolvers", () => {
  test.each(ORDERS)("the import depth holds through a memoizing resolver (%s)", async (_order, uses) => {
    const root = importing("root", uses);
    const error = await loadErrorAsync(
      loadConformanceFlow(root, { childResolver: memoizingResolver(sharedFileTexts(10)) }),
    );
    expect(error.message).toContain(IMPORT_51);

    await loadConformanceFlow(root, { childResolver: memoizingResolver(sharedFileTexts(9)) });
  });

  test("a resolver that calls loadFlow is bounded", async () => {
    const texts = chainTexts(200);
    const resolve = (path: string): Promise<ComposedFlow> =>
      loadConformanceFlow(texts[path]!, { childResolver: resolve, sourceLabel: `${path}.yaml` });

    const error = await loadErrorAsync(
      loadConformanceFlow(importing("root", { first: "u0" }), { childResolver: resolve }),
    );
    expect(error.message).toBe(importMessage(51));
    expectRefusedAtAlias(error, "u49.yaml", texts["u49"]!, "u50");
  });

  const CALLING_ROOT =
    "id: f\nname: f\ninput:\n  x: !str\nuses:\n  ext: child\nnodes:\n" +
    "  top:\n    kind: call\n    call: ext\n    input:\n      x: ${input.x}\n" +
    "output: ${top.output}\n";

  test.each([
    [
      "def-calls",
      "id: f\nname: f\ninput:\n  x: !str\nuses:\n  ext: child\n" +
        "defs:\n  d:\n    input:\n      x: !str\n    nodes:\n      step:\n        kind: call\n" +
        "        call: ext\n        input:\n          x: ${input.x}\n    output: ${step.output}\n" +
        "nodes:\n  top:\n    kind: call\n    call: d\n    input:\n      x: ${input.x}\n" +
        "output: ${top.output}\n",
    ],
    ["root-calls", CALLING_ROOT],
    ["uncalled", importing("f", { ext: "child" })],
  ])("resolver errors become load errors at the alias (%s)", async (_kind, text) => {
    const raising = (): ComposedFlow => {
      throw new Error("boom");
    };
    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: raising }));

    expect(error.message).toBe("callable 'ext' (uses: 'child'): could not resolve: boom");
    const frames = error.diagnostic.frames;
    expect(frames.length).toBe(1);
    expect(frames[0]!.note).toBe("uses alias 'ext'");
    expect(frames[0]!.range.startLine).toBe(lineOf(text, "ext: child"));
  });

  test.each([
    [-1, "-1"],
    ["1", '"1"'],
    [true, "true"],
    [1.5, "1.5"],
    [null, "null"],
  ])("a resolved flow with import depth %p is refused", async (depth, shown) => {
    const text = importing("f", { ext: "child" });
    const leaf = await loadConformanceFlow(importing("child", {}));
    const resolve = () =>
      new ComposedFlow(leaf.compiled, {
        version: leaf.version,
        name: leaf.name,
        description: leaf.description,
        originResolver: leaf.originResolver,
        importDepth: depth as number,
      });

    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: resolve }));
    expect(error.message).toBe(
      `callable 'ext' (uses: 'child'): resolved flow has import_depth ${shown}; expected a non-negative int`,
    );
    const frames = error.diagnostic.frames;
    expect(frames.length).toBe(1);
    expect(frames[0]!.note).toBe("uses alias 'ext'");
    expect(frames[0]!.range.startLine).toBe(lineOf(text, "ext: child"));
  });

  test("a resolver may return a promise", async () => {
    const flow = await loadConformanceFlow(CALLING_ROOT, {
      childResolver: async () => loadConformanceFlow(importing("child", {})),
    });
    expect(flow.importDepth).toBe(1);
  });

  test("a rejecting resolver fails at the alias as could not resolve", async () => {
    const error = await loadErrorAsync(
      loadConformanceFlow(CALLING_ROOT, { childResolver: () => Promise.reject(new Error("offline")) }),
    );
    expect(error.message).toBe("callable 'ext' (uses: 'child'): could not resolve: offline");
    expect(error.diagnostic.frames.map((frame) => frame.note)).toEqual(["uses alias 'ext'"]);
  });

  test("two concurrent loads don't share the import count", async () => {
    // Both loads wait inside their import of u0 until the other arrives; a shared count would
    // then see two imports active and refuse u0's own import at the limit of 2.
    reload((name) => (name === "YIAN_MAX_CALL_NESTING" ? "2" : undefined));
    const texts = chainTexts(2, { calls: false });
    let arrived = 0;
    let release!: () => void;
    const bothArrived = new Promise<void>((resolve) => (release = resolve));
    const resolve = async (path: string): Promise<ComposedFlow> => {
      if (path === "u0") {
        arrived += 1;
        if (arrived === 2) release();
        await bothArrived;
      }
      return loadConformanceFlow(texts[path]!, { childResolver: resolve, sourceLabel: `${path}.yaml` });
    };
    const root = importing("root", { first: "u0" });
    const flows = await Promise.all([
      loadConformanceFlow(root, { childResolver: resolve }),
      loadConformanceFlow(root, { childResolver: resolve }),
    ]);
    expect(flows.map((flow) => flow.importDepth)).toEqual([2, 2]);
  });
});

// ---------- chains of defs: entries, through the loader ----------

const PAST_51 = "nests calls 51 levels deep, past the call nesting limit of 50";
const PAST_DEEP = "more than 50 levels deep, past the call nesting limit of 50";
const LEAF =
  "    input:\n" +
  "      x: !str\n" +
  "    nodes:\n" +
  "      step:\n" +
  "        kind: code\n" +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  "        input:\n" +
  "          topic: ${input.x}\n" +
  "        output: str\n" +
  "    output: ${step.output}\n";

/** Return a def body whose one `step` node calls `target` with `x`. */
function calling(target: string): string {
  return (
    "    input:\n      x: !str\n    nodes:\n      step:\n        kind: call\n" +
    `        call: ${target}\n` +
    "        input:\n          x: ${input.x}\n    output: ${step.output}\n"
  );
}

/**
 * Return a flow whose root `top` calls `c0` and each `ci` calls `c(i+1)`. The last def is a CODE
 * leaf echoing `x`, or calls `lastCalls` when given; `reverse` writes the defs last-first.
 */
function defsChain(
  n: number,
  options: { reverse?: boolean; prefix?: string; lastCalls?: string | null; uses?: Record<string, string> } = {},
): string {
  const { reverse = false, prefix = "c", lastCalls = null, uses = {} } = options;
  const entries: string[] = [];
  for (let i = 0; i < n; i++) {
    const body = i < n - 1 ? calling(`${prefix}${i + 1}`) : lastCalls !== null ? calling(lastCalls) : LEAF;
    entries.push(`  ${prefix}${i}:\n${body}`);
  }
  if (reverse) entries.reverse();
  let head = `id: chain${n}\nname: chain${n}\ninput:\n  x: !str\n`;
  const aliases = Object.entries(uses);
  if (aliases.length > 0) head += "uses:\n" + aliases.map(([alias, ref]) => `  ${alias}: ${ref}\n`).join("");
  return (
    head +
    "defs:\n" +
    entries.join("") +
    "nodes:\n  top:\n    kind: call\n" +
    `    call: ${prefix}0\n` +
    "    input:\n      x: ${input.x}\noutput: ${top.output}\n"
  );
}

/**
 * Return a flow whose defs reach one another by CALL, MAP and LOOP in turn: root `top` calls
 * `c0`, each `ci` reaches `c(i+1)`, and the last def is a CODE leaf. A LOOP body returns the
 * record it carries, so each def right under a LOOP outputs `{v: ...}`.
 */
function mixedChain(n: number): string {
  const entries: string[] = [];
  for (let i = 0; i < n; i++) {
    const underLoop = i > 0 && i % 3 === 0; // the CALL levels sit right under a LOOP
    let node: string;
    let result: string;
    if (i === n - 1) {
      node =
        "        kind: code\n        code: ./fixtures/compose_codefns.ts:echo\n" +
        "        input:\n          topic: ${input.v}\n        output: str\n";
      result = "${step.output}";
    } else {
      const callee = `c${i + 1}`;
      node = [
        `        kind: call\n        call: ${callee}\n        input:\n          v: \${input.v}\n`,
        `        kind: map\n        over: \${[input.v]}\n        call: ${callee}\n` +
          "        input:\n          v: ${item}\n",
        `        kind: loop\n        times: 1\n        call: ${callee}\n        input:\n          v: \${input.v}\n`,
      ][i % 3]!;
      result = underLoop ? "${input.v}" : "${step.output}";
    }
    const output = underLoop ? `\n      v: ${result}\n` : ` ${result}\n`;
    entries.push(`  c${i}:\n    input:\n      v: !str\n    nodes:\n      step:\n${node}    output:${output}`);
  }
  return (
    `id: mixed${n}\nname: mixed${n}\ninput:\n  v: !str\ndefs:\n` +
    entries.join("") +
    "nodes:\n  top:\n    kind: call\n    call: c0\n    input:\n      v: ${input.v}\noutput: ${top.output}\n"
  );
}

/**
 * Return a flow whose `call(...)` directives chain through every def: the root `step` echoes
 * `call(c0, x=${input.x})`, each `ci` echoes `call(c(i+1), ...)`, and the last def is a CODE leaf.
 */
function directiveChain(n: number): string {
  const echoCall = (target: string, indent: string) =>
    `${indent}step:\n` +
    `${indent}  kind: code\n` +
    `${indent}  code: ./fixtures/compose_codefns.ts:echo\n` +
    `${indent}  input:\n` +
    `${indent}    topic: call(${target}, x=\${input.x})\n` +
    `${indent}  output: str\n`;
  const entries: string[] = [];
  for (let i = 0; i < n; i++) {
    const body =
      i < n - 1
        ? "    input:\n      x: !str\n    nodes:\n" + echoCall(`c${i + 1}`, "      ") + "    output: ${step.output}\n"
        : LEAF;
    entries.push(`  c${i}:\n${body}`);
  }
  return (
    `id: directive${n}\nname: directive${n}\ninput:\n  x: !str\ndefs:\n` +
    entries.join("") +
    "nodes:\n" +
    echoCall("c0", "  ") +
    "output: ${step.output}\n"
  );
}

/** Return the start line of a load error's innermost frame. */
const errorLine = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

/** A 51-def chain whose defs nobody calls; its root node is a CODE leaf. */
const UNCALLED_DEFS = defsChain(51, { prefix: "k" }).split("nodes:\n  top:\n")[0] + "nodes:\n" + ROOT_LEAF;

describe("chains of defs: entries", () => {
  test("a chain at the limit loads, runs and resumes", async () => {
    const text = defsChain(50);
    const path = join(runDir, "chain.jsonl");
    const leaf = ["top", ...Array<string>(50).fill("step")].join("/");

    const first = await runFlow(await loadConformanceFlow(text), { x: "hi" }, { log: path, numWorkers: 0, env });
    expect(first.status).toBe("succeeded");
    expect(first.output).toBe("hi");
    // Cut right after the last expansion so resume replays all 50 and runs the leaf.
    const lines = readFileSync(path, "utf8").split("\n").slice(0, -1);
    const expanded = lines.flatMap((line, index) => (JSON.parse(line).kind === "node_expanded" ? [index] : []));
    expect(expanded).toHaveLength(50);
    writeFileSync(path, `${lines.slice(0, expanded.at(-1)! + 1).join("\n")}\n`);

    const resumed = await advance({ log: path, env });

    expect(resumed.status).toBe("succeeded");
    expect(resumed.output).toBe("hi");
    expect(started(resumed.events)).toContain(leaf);
  });

  test("a chain past the limit is located at the root call", async () => {
    const text = defsChain(51);
    const error = await loadErrorAsync(loadConformanceFlow(text));

    expect(error.message).toContain(PAST_51);
    expect(error.message).toContain("call node 'top': calling callable 'c0'");
    expect(error.diagnostic.frames.length).toBe(1);
    expect(errorLine(error)).toBe(lineOf(text, "call: c0"));
  });

  test.each([97, 1000])("a long chain of %p fails at the first call past the limit", async (n) => {
    const text = defsChain(n);
    const error = await loadErrorAsync(loadConformanceFlow(text));

    expect(error.message).toContain(PAST_DEEP);
    expect(error.message).toContain("call node 'step': calling callable 'c51'");
    expect(errorLine(error)).toBe(lineOf(text, "call: c51"));
  });

  test("a chain written callee first is caught from the built callables", async () => {
    const text = defsChain(51, { reverse: true });
    const error = await loadErrorAsync(loadConformanceFlow(text));

    expect(error.message).toContain(PAST_51);
    expect(errorLine(error)).toBe(lineOf(text, "call: c0"));
    await loadConformanceFlow(defsChain(50, { reverse: true }));
  });

  test("map and loop bodies count", async () => {
    const result = await runFlow(await loadConformanceFlow(mixedChain(50)), { v: "hi" }, { numWorkers: 0, env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toEqual([{ v: "hi" }]);
    // The CODE leaf, 50 bodies down, ran.
    expect(started(result.events).some((id) => id.split("/").length === 51 && id.endsWith("/step"))).toBe(true);

    const text = mixedChain(51);
    const error = await loadErrorAsync(loadConformanceFlow(text));
    expect(error.message).toContain(PAST_51);
    expect(errorLine(error)).toBe(lineOf(text, "call: c0"));
  });

  test("call directives count", async () => {
    const result = await runFlow(await loadConformanceFlow(directiveChain(50)), { x: "hi" }, { numWorkers: 0, env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe("hi");

    const error = await loadErrorAsync(loadConformanceFlow(directiveChain(51)));
    expect(error.message).toContain("past the call nesting limit of 50");
  });

  test.each([
    [
      "a nodes: def",
      "    nodes:\n      invoke:\n        kind: call\n        call: external\n    output: ${invoke.output}\n",
    ],
    ["a single-node def", "    kind: call\n    call: external\n"],
  ])("a broken import is reported at its alias before defs (%s)", async (_form, definition) => {
    const text =
      "id: def-inner-line\nname: def_inner_line\nuses:\n  external: child.yaml\ndefs:\n  local:\n" +
      definition +
      "nodes:\n  invoke:\n    kind: call\n    call: local\noutput: ${invoke.output}\n";
    const failResolver = (): ComposedFlow => {
      throw new LoadDiagnostics().error("external failure");
    };

    const error = await loadErrorAsync(loadConformanceFlow(text, { childResolver: failResolver }));

    expect(error.message).toBe("external failure");
    const frames = error.diagnostic.frames;
    expect(frames.length).toBe(1);
    expect(frames[0]!.note).toBe("uses alias 'external'");
    expect(frames[0]!.range.startLine).toBe(lineOf(text, "external: child.yaml"));
  });

  test("an alias colliding with a node id wins over its broken target", async () => {
    const text = "id: f\nname: f\ninput:\n  x: !str\nuses:\n  step: hub:x/y@1\nnodes:\n" + ROOT_LEAF;
    const error = await loadErrorAsync(loadConformanceFlow(text));
    expect(error.message).toBe('uses: alias ["step"] collides with a node id');
  });
});

describe("chains of uses: files over defs: entries", () => {
  let tmp: string;

  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), "yian-defs-chain-"));
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  /** Make and return the directory `<tmp>/<name>`. */
  const dir = (name: string) => {
    const path = join(tmp, name);
    mkdirSync(path);
    return path;
  };

  test("imports load before defs", async () => {
    // A file at the limit called from 49 levels down is caught when the call returns.
    const deep = dir("deep");
    writeFiles(deep, { lib: defsChain(50, { prefix: "k" }) });
    const text = defsChain(49, { lastCalls: "lib", uses: { lib: "lib" } });
    const error = await loadErrorAsync(loadConformanceFlow(text, { searchPaths: [deep] }));
    expect(error.message).toContain("nests calls 99 levels deep, past the call nesting limit of 50");
    expect(errorLine(error)).toBe(lineOf(text, "call: lib"));

    // A one-level file's uncalled defs nest 50 on their own, not under the importer.
    const legal = dir("legal");
    writeFiles(legal, { lib: UNCALLED_DEFS });
    await loadConformanceFlow(defsChain(49, { lastCalls: "lib", uses: { lib: "lib" } }), { searchPaths: [legal] });
  });

  test("imports and a calling defs chain count apart", async () => {
    // The import chain and the call chain in its last file are each 30 deep, 60 together.
    writeFiles(tmp, { ...chainTexts(30, { calls: false }), u29: defsChain(30) });
    await loadConformanceFlow(importing("root", { first: "u0" }), { searchPaths: [tmp] });
  });

  // A last file that calls its chain is the deepest accepted load, since importing it rebases a
  // compiled graph 50 levels deep.
  test.each([
    ["called", defsChain(50)],
    ["uncalled", UNCALLED_DEFS],
  ])("imports at the limit over defs at the limit load (%s)", async (_kind, last) => {
    writeFiles(tmp, { ...chainTexts(50, { calls: false }), u49: last });
    await loadConformanceFlow(importing("root", { first: "u0" }), { searchPaths: [tmp] });
  });
});
