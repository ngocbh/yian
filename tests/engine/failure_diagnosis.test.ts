/**
 * `ComposedFlow.diagnose` resolves a semantic failure through the flow's source authority,
 * presentation-only.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import {
  Diagnostic,
  DiagnosticFrame,
  Failure,
  OriginRef,
  type OriginResolver,
  SourceRange,
} from "../../src/diagnostics.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-failure-diagnosis-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const UNAVAILABLE_NOTE = ["Source presentation was unavailable."];

/** A minimal Compose boundary around a test resolver. */
function composedFlow(resolver: OriginResolver | null, compiled: object = {}): ComposedFlow {
  return new ComposedFlow(compiled as CompiledFlow, { originResolver: resolver });
}

/** One fully populated semantic failure for copy assertions. */
function failure(...origins: OriginRef[]): Failure {
  return new Failure({
    code: "contract.input",
    message: "expected an integer",
    nodeId: "outer/inner",
    trace: origins,
    causeType: "TypeCheckError",
    traceback: "Traceback: original failure",
  });
}

/** A resolver that records each origin it is asked for and answers `answer(origin)`. */
function recording(answer: (origin: OriginRef) => unknown): OriginResolver & { calls: OriginRef[] } {
  const calls: OriginRef[] = [];
  return {
    calls,
    resolve(origin) {
      calls.push(origin);
      return answer(origin) as DiagnosticFrame | null;
    },
  };
}

/** A resolver answering every origin with `frame`. */
const always = (frame: unknown): OriginResolver => ({ resolve: () => frame as DiagnosticFrame });

test("diagnose copies semantics and resolves ordered duplicate frames", () => {
  const outer = new OriginRef("outer");
  const inner = new OriginRef("inner");
  const outerFrame = new DiagnosticFrame({
    label: "root.yaml",
    text: "outer:\n",
    range: new SourceRange(1, 1, 1, 6),
    note: "called child",
  });
  const innerFrame = new DiagnosticFrame({ label: "child.yaml", text: "inner:\n", range: new SourceRange(1, 1, 1, 6) });
  const resolver = recording((origin) => (origin === outer ? outerFrame : innerFrame));
  const failed = failure(outer, inner, inner);

  const diagnostic = composedFlow(resolver).diagnose(failed);

  expect(resolver.calls).toEqual([outer, inner, inner]);
  expect(resolver.calls[0]).toBe(outer);
  expect(diagnostic).toStrictEqual(
    new Diagnostic({
      code: failed.code,
      message: failed.message,
      frames: [outerFrame, innerFrame, innerFrame],
      notes: [],
      nodeId: failed.nodeId,
      causeType: failed.causeType,
      traceback: failed.traceback,
    }),
  );
});

test("diagnose empty trace is unlocated without touching resolver", () => {
  const untouched: OriginResolver = {
    resolve(origin) {
      throw new Error(`resolver unexpectedly received ${origin.key}`);
    },
  };
  const failed = failure();

  const diagnostic = composedFlow(untouched).diagnose(failed);

  expect(diagnostic).toStrictEqual(
    new Diagnostic({
      code: failed.code,
      message: failed.message,
      nodeId: failed.nodeId,
      causeType: failed.causeType,
      traceback: failed.traceback,
    }),
  );
});

test("diagnose missing resolver degrades nonempty trace atomically", () => {
  const failed = failure(new OriginRef("missing"));

  const diagnostic = composedFlow(null).diagnose(failed);

  expect(diagnostic).toStrictEqual(
    new Diagnostic({
      code: failed.code,
      message: failed.message,
      notes: UNAVAILABLE_NOTE,
      nodeId: failed.nodeId,
      causeType: failed.causeType,
      traceback: failed.traceback,
    }),
  );
});

test("diagnose unresolved later origin discards partial frame prefix", () => {
  const first = new OriginRef("first");
  const missing = new OriginRef("missing");
  const firstFrame = new DiagnosticFrame({ label: "flow.yaml", text: "first:\n", range: new SourceRange(1, 1, 1, 6) });
  const resolver = recording((origin) => (origin === first ? firstFrame : null));

  const diagnostic = composedFlow(resolver).diagnose(failure(first, missing));

  expect(resolver.calls).toEqual([first, missing]);
  expect(diagnostic.frames).toEqual([]);
  expect(diagnostic.notes).toEqual(UNAVAILABLE_NOTE);
});

test("diagnose resolver exception does not mask semantic failure", () => {
  const failed = failure(new OriginRef("broken"));
  const resolver: OriginResolver = {
    resolve(origin) {
      throw new Error(`secret resolver detail for ${origin.key}`);
    },
  };

  const diagnostic = composedFlow(resolver).diagnose(failed);

  expect(diagnostic.message).toBe(failed.message);
  expect(diagnostic.frames).toEqual([]);
  expect(diagnostic.notes).toEqual(UNAVAILABLE_NOTE);
  expect(diagnostic.notes.join(" ")).not.toContain("secret");
});

test("diagnose malformed resolver result degrades unlocated", () => {
  const diagnostic = composedFlow(always({})).diagnose(failure(new OriginRef("bad")));

  expect(diagnostic.frames).toEqual([]);
  expect(diagnostic.notes).toEqual(UNAVAILABLE_NOTE);
});

test("diagnose malformed base frame field degrades unlocated", () => {
  const malformed = new DiagnosticFrame({ label: "flow.yaml", text: "value\n", range: {} as SourceRange });

  const diagnostic = composedFlow(always(malformed)).diagnose(failure(new OriginRef("bad")));

  expect(diagnostic.frames).toEqual([]);
  expect(diagnostic.notes).toEqual(UNAVAILABLE_NOTE);
});

test("diagnose throwing frame subclass degrades unlocated", () => {
  const frame = new DiagnosticFrame({ label: "flow.yaml", text: "value\n", range: new SourceRange(1, 1, 1, 6) });
  Object.defineProperty(frame, "text", {
    get() {
      throw new Error("secret frame detail");
    },
  });

  const diagnostic = composedFlow(always(frame)).diagnose(failure(new OriginRef("bad")));

  expect(diagnostic.frames).toEqual([]);
  expect(diagnostic.notes).toEqual(UNAVAILABLE_NOTE);
  expect(diagnostic.notes.join(" ")).not.toContain("secret");
});

test("diagnose valid frame subclasses are canonicalized", () => {
  class DerivedRange extends SourceRange {}
  class DerivedFrame extends DiagnosticFrame {}
  const derived = new DerivedFrame({
    label: "flow.yaml",
    text: "value\n",
    range: new DerivedRange(1, 1, 1, 6),
    note: "context",
  });

  const diagnostic = composedFlow(always(derived)).diagnose(failure(new OriginRef("valid")));

  expect(diagnostic.frames).toStrictEqual([
    new DiagnosticFrame({ label: "flow.yaml", text: "value\n", range: new SourceRange(1, 1, 1, 6), note: "context" }),
  ]);
  expect(Object.getPrototypeOf(diagnostic.frames[0])).toBe(DiagnosticFrame.prototype);
  expect(Object.getPrototypeOf(diagnostic.frames[0]!.range)).toBe(SourceRange.prototype);
});

test.each([
  [true, 1, 1, 1],
  [1, true, 1, 1],
  [1, 1, true, 1],
  [1, 1, 1, true],
  [0, 1, 1, 1],
  [1, 0, 1, 1],
  [1, 1, 0, 1],
  [1, 1, 1, 0],
  [2, 1, 1, 1],
  [2, 3, 2, 2],
])("diagnose invalid frame coordinates discard partial prefix: (%p, %p, %p, %p)", (...coordinates) => {
  const first = new OriginRef("first");
  const goodFrame = new DiagnosticFrame({ label: "outer.yaml", text: "outer\n", range: new SourceRange(1, 1, 1, 6) });
  const [startLine, startCol, endLine, endCol] = coordinates as number[];
  const invalidFrame = new DiagnosticFrame({
    label: "inner.yaml",
    text: "inner\n",
    range: new SourceRange(startLine!, startCol!, endLine!, endCol!),
  });
  const resolver: OriginResolver = { resolve: (origin) => (origin === first ? goodFrame : invalidFrame) };

  const diagnostic = composedFlow(resolver).diagnose(failure(first, new OriginRef("invalid")));

  expect(diagnostic.frames).toEqual([]);
  expect(diagnostic.notes).toEqual(UNAVAILABLE_NOTE);
});

test("diagnose reads only failure and resolver", () => {
  const forbidden = new Proxy(
    {},
    {
      get(_target, name) {
        throw new Error(`diagnose inspected forbidden surface ${String(name)}`);
      },
      has(_target, name) {
        throw new Error(`diagnose inspected forbidden surface ${String(name)}`);
      },
      getPrototypeOf() {
        throw new Error("diagnose inspected forbidden surface's prototype");
      },
      ownKeys() {
        throw new Error("diagnose inspected forbidden surface's keys");
      },
    },
  );
  const parse = SourceDocument.parse;
  SourceDocument.parse = () => {
    throw new Error("diagnose reparsed YAML");
  };
  try {
    const origin = new OriginRef("only");
    const frame = new DiagnosticFrame({ label: "memory", text: "value\n", range: new SourceRange(1, 1, 1, 6) });
    const resolveOnly: OriginResolver = {
      resolve(candidate) {
        expect(candidate).toBe(origin);
        return frame;
      },
    };

    const diagnostic = composedFlow(resolveOnly, forbidden).diagnose(failure(origin));

    expect(diagnostic.frames).toEqual([frame]);
    expect(diagnostic.notes).toEqual([]);
  } finally {
    SourceDocument.parse = parse;
  }
});

test("load flow origin resolves through installed authority", async () => {
  const source = `id: located
name: located
nodes:
  work:
    kind: code
    code: |
      return 1
    output: int
output: \${work.output}
`;
  const loaded = await loadFlow(source, { env });
  const origin = loaded.compiled.nodes.get("work")!.origin;
  expect(origin).toBeInstanceOf(OriginRef);

  const diagnostic = loaded.diagnose(failure(origin!));

  expect(diagnostic.frames).toStrictEqual([
    new DiagnosticFrame({ label: "<flow>", text: source, range: new SourceRange(4, 3, 4, 7) }),
  ]);
  expect(diagnostic.notes).toEqual([]);
});

const INLINE_FAILURES: Readonly<Record<string, string>> = {
  map: `id: inline-map-failure
name: inline_map_failure
nodes:
  outer:
    kind: map
    over: [item]
    input:
      value: \${item}
    nodes:
      fail:
        kind: code
        code: |
          throw new Error("boom")
          return inputs["value"]
        input:
          value: \${input.value}
        output: str
    output: \${fail.output}
output: \${outer.output}
`,
  loop: `id: inline-loop-failure
name: inline_loop_failure
nodes:
  outer:
    kind: loop
    times: 1
    input:
      n: 0
    nodes:
      fail:
        kind: code
        code: |
          throw new Error("boom")
          return { n: inputs["n"] }
        input:
          n: \${input.n}
        output:
          n: int
    output: \${fail.output}
output: \${outer.output}
`,
};

test.each(["map", "loop"])("inline composite failure resolves catalog frames (%s)", async (kind) => {
  const label = `inline-${kind}.yaml`;
  const loaded = await loadFlow(INLINE_FAILURES[kind]!, { sourceLabel: label, env });

  const result = await runFlow(loaded, {}, { env });

  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.traceback).not.toBeNull();
  expect(result.failure!.traceback).toContain(label);
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames).toHaveLength(2);
  expect(diagnostic.frames.map((frame) => frame.label)).toEqual([label, label]);
  expect(diagnostic.frames[0]!.text).toContain("outer:");
  expect(diagnostic.frames[1]!.text).toContain('throw new Error("boom")');
});

const EXTERNAL_FAILING_CHILD = `id: failing-child
name: failing_child
input:
  n: !int
nodes:
  fail:
    kind: code
    code: |
      throw new Error("boom")
      return { n: inputs["n"] }
    input:
      n: \${input.n}
    output:
      n: int
output: \${fail.output}
`;

const EXTERNAL_PARENTS: Readonly<Record<string, string>> = {
  map: `id: external-map-failure
name: external_map_failure
uses:
  child: failing-child
nodes:
  outer:
    kind: map
    call: child
    over: [0]
    input:
      n: \${item}
output: \${outer.output}
`,
  loop: `id: external-loop-failure
name: external_loop_failure
uses:
  child: failing-child
nodes:
  outer:
    kind: loop
    call: child
    times: 1
    input:
      n: 0
output: \${outer.output}
`,
};

test.each(["map", "loop"])("external composite failure resolves each document (%s)", async (kind) => {
  const child = await loadFlow(EXTERNAL_FAILING_CHILD, { sourceLabel: "child.yaml", env });
  const parentLabel = `external-${kind}.yaml`;
  const loaded = await loadFlow(EXTERNAL_PARENTS[kind]!, {
    childResolver: () => child,
    sourceLabel: parentLabel,
    env,
  });

  const result = await runFlow(loaded, {}, { env });

  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.traceback).not.toBeNull();
  expect(result.failure!.traceback).toContain("child.yaml");
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames).toHaveLength(2);
  expect(diagnostic.frames.map((frame) => frame.label)).toEqual([parentLabel, "child.yaml"]);
  expect(diagnostic.frames[0]!.text).toBe(EXTERNAL_PARENTS[kind]!);
  expect(diagnostic.frames[1]!.text).toBe(EXTERNAL_FAILING_CHILD);
});
