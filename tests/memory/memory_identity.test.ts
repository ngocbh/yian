/** Memory declaration sites, declarations, specs and stacks. */

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  DeclSite,
  isMemory,
  iterMemoryDeclOrigins,
  LearnContext,
  LearningFailure,
  LearningResult,
  MemoryContent,
  MemoryContext,
  MemoryDecl,
  MemoryOperationFailure,
  MemorySection,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  PolicyCommitOutcome,
  type RecallContext,
  rewriteMemoryDeclOrigins,
} from "../../src/memory/contracts.ts";
import * as memoryPackage from "../../src/memory/index.ts";
// Unused on purpose: the typecheck fails if the facade stops exporting one of these types.
import type {
  JsonObject,
  JsonScalar,
  JsonValue,
  LearningOperation,
  LearningStatus,
  Memory,
  MemorySession,
  PolicyCommitStatus,
  VisibleInteraction,
} from "../../src/memory/index.ts";
import {
  decodeRecord,
  encodeRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { parseJson } from "../../src/typesys/json.ts";

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

test("a declaration site validates and freezes", () => {
  const site = new DeclSite([], ["root"], "ask");
  expect([site.scope, site.owner, site.node]).toEqual([[], ["root"], "ask"]);
  expect(new DeclSite([], ["def", "child"]).node).toBeNull();
  expect(new DeclSite([], ["def", "child"], null).node).toBeNull();
  expect(() => {
    (site as { node: string | null }).node = "other";
  }).toThrow(TypeError);
  expect(Object.isFrozen(site.owner)).toBe(true);
  expectError(() => new DeclSite("root" as never, ["root"]), TypeError, "scope must be an array");
  expectError(() => new DeclSite([], []), RangeError, "owner must be non-empty");
  expectError(() => new DeclSite([], [""]), RangeError, "owner components must be non-empty");
  expectError(() => new DeclSite([], [1n] as never), TypeError, "owner components must be a string");
  expectError(() => new DeclSite([], ["root"], ""), RangeError, "node must be non-empty");
  expectError(() => new DeclSite([], ["root"], 1n as never), TypeError, "node must be a string");
  expectError(() => new DeclSite([], ["root"], "\uD800"), RangeError, "Unicode surrogate is not valid canonical JSON");
});

test("a declaration site copies its paths", () => {
  const owner = ["root"];
  const site = new DeclSite([], owner);
  owner.push("other");
  expect(site.owner).toEqual(["root"]);
});

test("the site order is total and puts a flow site before AGENT sites", () => {
  const sites = [
    new DeclSite([], ["root"], "b"),
    new DeclSite(["lib"], ["root"], null),
    new DeclSite([], ["root"], null),
    new DeclSite([], ["def", "child"], "a"),
    new DeclSite([], ["root"], "a"),
  ];
  expect(sites.sort(DeclSite.compare)).toEqual([
    new DeclSite([], ["def", "child"], "a"),
    new DeclSite([], ["root"], null),
    new DeclSite([], ["root"], "a"),
    new DeclSite([], ["root"], "b"),
    new DeclSite(["lib"], ["root"], null),
  ]);
  expect(DeclSite.compare(new DeclSite(["lib"], ["root"], "a"), new DeclSite(["lib"], ["root"], "a"))).toBe(0);
  expect(DeclSite.compare(new DeclSite([], ["root"]), new DeclSite([], ["root", "x"]))).toBeLessThan(0);
});

/** Build one episodic declaration at a root site. */
function decl(name = "notes", node: string | null = null, origin: OriginRef | null = null): MemoryDecl {
  return new MemoryDecl(name, "episodic", { recall_limit: 3n }, new DeclSite([], ["root"], node), origin);
}

test("a declaration snapshots its config and rejects private keys", () => {
  const config: Record<string, unknown> = { recall_limit: 3n };
  const site = new DeclSite([], ["root"]);
  const declaration = new MemoryDecl("notes", "episodic", config, site);
  config.recall_limit = 9n;
  expect(declaration.config).toEqual({ recall_limit: 3n });
  expect(() => {
    (declaration.config as Record<string, unknown>).recall_limit = 7n;
  }).toThrow(TypeError);
  expect(declaration.config).toEqual({ recall_limit: 3n });
  expect(declaration.origin).toBeNull();
  expect(declaration.site).toBe(site);
  expect(declaration.key).toBe("notes");
  expect(new MemoryDecl("notes", "episodic", {}, new DeclSite(["lib"], ["root"])).key).toBe("lib.notes");
  expectError(
    () => new MemoryDecl("notes", "episodic", { api_key: "x" }, site),
    RangeError,
    "config contains forbidden private key 'api_key'",
  );
  expectError(
    () => new MemoryDecl("notes", "episodic", {}, { scope: [], owner: ["root"], node: null } as never),
    TypeError,
    "site must be a DeclSite",
  );
  expectError(
    () => new MemoryDecl("notes", "episodic", {}, site, "x" as never),
    TypeError,
    "origin must be an OriginRef or null",
  );
  expectError(
    () => new MemoryDecl("", "episodic", {}, site),
    RangeError,
    "memory name '' must match [A-Za-z_][A-Za-z0-9_-]*",
  );
  expectError(
    () => new MemoryDecl("notes", "", {}, site),
    RangeError,
    "memory type '' must match [A-Za-z_][A-Za-z0-9_-]*",
  );
});

test("declaration origins traverse and rewrite shared carriers once", () => {
  const a = new OriginRef("a");
  const b = new OriginRef("b");
  const shared = decl("notes", null, a);
  const other = decl("other", "ask", b);
  const plain = decl("plain");
  const decls = [shared, other, shared, plain];
  expect([...iterMemoryDeclOrigins(decls, new Set())]).toEqual([a, b]);

  const calls: string[] = [];
  const mapper = (origin: OriginRef): OriginRef => {
    calls.push(origin.key);
    return new OriginRef(`mapped/${origin.key}`);
  };
  const seen = new Map<object, object>();
  const rewritten = rewriteMemoryDeclOrigins(decls, mapper, seen);
  expect(calls).toEqual(["a", "b"]);
  expect(Object.isFrozen(rewritten)).toBe(true);
  expect(rewritten.map((d) => d.origin?.key ?? null)).toEqual(["mapped/a", "mapped/b", "mapped/a", null]);
  expect(rewritten[0]).toBe(rewritten[2]!);
  expect(rewritten[0]).not.toBe(shared);
  expect(rewritten[3]).toBe(plain);
  expect([rewritten[1]!.name, rewritten[1]!.site, rewritten[1]!.config]).toEqual([
    other.name,
    other.site,
    other.config,
  ]);
  expect(rewritten[1]!.site).toBe(other.site);
  expect([shared.origin, other.origin]).toEqual([a, b]);
  expect(shared.origin).toBe(a);
  expect(rewriteMemoryDeclOrigins(decls, mapper, seen)).toBe(rewritten);
  expect(calls).toEqual(["a", "b"]);
  expect(decls).toEqual([shared, other, shared, plain]);
  expect(decls[0]).toBe(shared);

  // A different array sharing a declaration reuses its rebuilt form through the same `seen`.
  const c = new OriginRef("c");
  const third = decl("third", null, c);
  const again = rewriteMemoryDeclOrigins([third, shared], mapper, seen);
  expect(calls).toEqual(["a", "b", "c"]);
  expect(again[1]).toBe(rewritten[0]!);
  expect(again[0]!.site).toBe(third.site);

  const visited = new Set<object>();
  expect([...iterMemoryDeclOrigins(decls, visited)]).toEqual([a, b]);
  expect([...iterMemoryDeclOrigins([third, shared, other], visited)]).toEqual([c]);
});

test("the origin rewrite keeps the mapper's origin and checks its result", () => {
  const mapped = new OriginRef("mapped");
  const [rebuilt] = rewriteMemoryDeclOrigins([decl("notes", null, new OriginRef("a"))], () => mapped, new Map());
  expect(rebuilt!.origin).toBe(mapped);
  expectError(
    () => rewriteMemoryDeclOrigins([decl("notes", null, new OriginRef("a"))], () => "x" as never, new Map()),
    TypeError,
    "origin mapper must return an OriginRef",
  );
});

test("origin traversal requires an array of exact declarations", () => {
  class SubDecl extends MemoryDecl {}
  const sub = new SubDecl("notes", "episodic", {}, new DeclSite([], ["root"]));
  for (const run of [
    (decls: never) => [...iterMemoryDeclOrigins(decls, new Set())],
    (decls: never) => rewriteMemoryDeclOrigins(decls, (o) => o, new Map()),
  ]) {
    expectError(() => run(new Set() as never), TypeError, "memories must be an array");
    expectError(() => run([1n] as never), TypeError, "memories items must be exact MemoryDecl values");
    expectError(() => run([sub] as never), TypeError, "memories items must be exact MemoryDecl values");
  }
});

test("a spec requires sorted owners", () => {
  const agent = new DeclSite([], ["root"], "ask");
  const flow = new DeclSite([], ["root"]);
  const spec = new MemorySpec("chat", "episodic", {}, [agent]);
  expect(Object.isFrozen(spec.owners)).toBe(true);
  expect(new MemorySpec("notes", "episodic", {}, [flow, agent]).owners).toEqual([flow, agent]);
  const order = "owners must be sorted and unique by DeclSite.compare";
  expectError(() => new MemorySpec("notes", "episodic", {}, [agent, flow]), RangeError, order);
  expectError(() => new MemorySpec("notes", "episodic", {}, [flow, flow]), RangeError, order);
  expectError(() => new MemorySpec("notes", "episodic", {}, []), RangeError, "owners must be non-empty");
  expectError(() => new MemorySpec("", "t", {}, [flow]), RangeError, "key must be non-empty");
  expectError(() => new MemorySpec("k", 1n as never, {}, [flow]), TypeError, "type must be a string");
  expectError(
    () => new MemorySpec("k", "t", { api_key: 1n }, [flow]),
    RangeError,
    "config contains forbidden private key 'api_key'",
  );
  expectError(
    () => new MemorySpec("k", "t", [] as never, [flow]),
    TypeError,
    "config must be an exact JSON object",
  );
  expectError(
    () => new MemorySpec("k", "t", {}, new Set([flow]) as never),
    TypeError,
    "owners must be an array",
  );
  expectError(
    () => new MemorySpec("k", "t", {}, [{}] as never),
    TypeError,
    "owners items must be DeclSite values",
  );
});

test("a compiled spec takes a scoped key and keys the declaration check refuses", () => {
  const owners = [new DeclSite(["lib"], ["root"])];
  const spec = new MemorySpec("lib.notes", "episodic", { storage: 1n, bogus: 2n }, owners);
  expect(spec.key).toBe("lib.notes");
  expect(spec.config).toEqual({ bogus: 2n, storage: 1n });
  expect(Object.keys(spec.config)).toEqual(["bogus", "storage"]);
});

test("a stack requires a locator and unique keys", () => {
  const stack = new MemoryStack(["child", "ask"], ["notes", "chat"]);
  expect(stack.memories).toEqual(["notes", "chat"]);
  expect(Object.isFrozen(stack.locator)).toBe(true);
  expect(new MemoryStack(["plain"], []).memories).toEqual([]);
  expectError(() => new MemoryStack([], ["notes"]), RangeError, "locator must be non-empty");
  expectError(() => new MemoryStack(["ask"], ["notes", "notes"]), RangeError, "memories must be unique");
  expectError(() => new MemoryStack(["ask"], [""]), RangeError, "memories components must be non-empty");
});

test("a snapshot envelope names its memory key", () => {
  const snapshot: Record<string, unknown> = { v: 1n };
  const envelope = new MemorySnapshotEnvelope("notes", snapshot);
  snapshot.v = 2n;
  expect(envelope.memory).toBe("notes");
  expect(envelope.snapshot).toEqual({ v: 1n });
  expect(Object.isFrozen(envelope)).toBe(true);
  expect(Object.isFrozen(envelope.snapshot)).toBe(true);
  expect(() => {
    (envelope.snapshot as Record<string, unknown>).v = 3n;
  }).toThrow(TypeError);
  expect("policyName" in envelope).toBe(false);
  expectError(() => new MemorySnapshotEnvelope("", {}), RangeError, "memory must be non-empty");
  expectError(() => new MemorySnapshotEnvelope(1n as never, {}), TypeError, "memory must be a string");
  expectError(
    () => new MemorySnapshotEnvelope("\uD800", {}),
    RangeError,
    "Unicode surrogate is not valid canonical JSON",
  );
  // The snapshot is checked first.
  expectError(() => new MemorySnapshotEnvelope("", []), TypeError, "snapshot must be an exact JSON object");
});

test("a snapshot envelope takes a snapshot deeper than the value nesting limit", () => {
  let deep: JsonObject = {};
  for (let i = 0; i < 150; i++) deep = { d: deep };
  expect(new MemorySnapshotEnvelope("notes", deep).snapshot).toEqual(deep);
});

test("learning carriers name their memory key", () => {
  const cause = new MemoryOperationFailure("commit", "declined", "Declined.", true);
  const failure = new LearningFailure("notes", "commit", cause);
  const outcome = new PolicyCommitOutcome("notes", "failed", { commitId: "c", failure: cause });
  expect([failure.memory, outcome.memory]).toEqual(["notes", "notes"]);
  const result = new LearningResult("failed", { failures: [failure], policies: [outcome] });
  expect(result.policies[0]!.memory).toBe("notes");
  expectError(
    () => new LearningResult("failed", { failures: [failure], policies: [outcome, outcome] }),
    RangeError,
    "learning memory keys must be unique",
  );
});

test("the commit records name their memory key", () => {
  const cause = new MemoryOperationFailure("commit", "declined", "Declined.", true);
  for (const record of [
    new MemoryCommitRequestedRecord("notes", "c1", [], 1),
    new MemoryCommitSucceededRecord("notes", "c1", 1),
    new MemoryCommitFailedRecord("notes", "c1", cause, 1),
  ]) {
    expect(record.memory).toBe("notes");
    const wire = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 0, record));
    expect((parseJson(wire, "codec") as { payload: { memory: string } }).payload.memory).toBe("notes");
    expect(decodeRecord(wire).record).toEqual(record);
  }
});

test("the package doc and exports describe keyed memories", () => {
  const source = readFileSync(new URL("../../src/memory/index.ts", import.meta.url), "utf8");
  expect(source.startsWith("/**")).toBe(true);
  const doc = source.slice(0, source.indexOf("*/"));
  expect(doc).not.toMatch(/\bframes?\b/i);
  for (const phrase of ["memory key", "DeclSite", "MemoryDecl", "MemorySpec", "MemoryStack"]) {
    expect(doc).toContain(phrase);
  }
  // Exact, so a leaked internal (the origin traversal or rewrite) fails.
  expect(Object.keys(memoryPackage)).toEqual([
    "BUILTIN_MEMORY_CONFIG_KEYS",
    "DeclSite",
    "LearnContext",
    "LearningFailure",
    "LearningResult",
    "MEMORY_IDENTIFIER_PATTERN",
    "MemoryContent",
    "MemoryContext",
    "MemoryDecl",
    "MemoryEpisode",
    "MemoryEpisodeStatus",
    "MemoryMutation",
    "MemoryOccurrenceKind",
    "MemoryOccurrenceSegment",
    "MemoryOperationError",
    "MemoryOperationFailure",
    "MemoryRecallError",
    "MemoryScope",
    "MemorySection",
    "MemorySetupError",
    "MemorySnapshotEnvelope",
    "MemorySpec",
    "MemoryStack",
    "ModelTextInteraction",
    "ModelValueInteraction",
    "PolicyCommitOutcome",
    "RecallContext",
    "ToolCallInteraction",
    "ToolResultInteraction",
    "checkMemoryDeclaration",
    "isMemory",
  ]);
});

test("the memory protocol has no frames", async () => {
  const recallContexts: (RecallContext | null)[] = [];
  const learnContexts: LearnContext[] = [];
  const memory: Memory = {
    openRun: () => ({
      snapshot: { namespace: "test" },
      close() {},
      recall: (context = null) => {
        recallContexts.push(context);
        return "notes";
      },
      learn: (context) => {
        learnContexts.push(context);
        return [];
      },
      commit() {},
    }),
    reset() {},
  };
  const session = await memory.openRun();
  expect(isMemory(memory)).toBe(true);
  // The typecheck fails if `MemorySession` gains or loses a member.
  const members: Record<keyof MemorySession, true> = {
    snapshot: true,
    close: true,
    recall: true,
    learn: true,
    commit: true,
  };
  expect(Object.keys(session).sort()).toEqual(Object.keys(members).sort());
  expect("openFrame" in session).toBe(false);
  expect(await session.recall()).toBe("notes");
  expect(await session.learn(new LearnContext("run", "notes", []))).toEqual([]);
  expect(recallContexts).toEqual([null]);
  expect(learnContexts.map((context) => context.memory)).toEqual(["notes"]);
  await session.close();
});

test("memory content validates its notes", () => {
  const content = new MemoryContent({ text: "notes" });
  expect(content.text).toBe("notes");
  expect(new MemoryContent().text).toBe("");
  for (const [text, cls, message] of [
    [null, TypeError, "text must be a string"],
    [7, TypeError, "text must be a string"],
    ["\ud800", RangeError, "Unicode surrogate is not valid canonical JSON"],
  ] as const) {
    expectError(() => new MemoryContent({ text: text as never }), cls, message);
  }
  expect(() => {
    (content as { text: string }).text = "changed";
  }).toThrow(TypeError);
});

test("a memory context requires exact sections", () => {
  expect(new MemoryContext().sections).toEqual([]);
  const section = new MemorySection("notes", "outer");
  const sections = [section];
  const context = new MemoryContext({ sections });
  sections.push(new MemorySection("chat", "inner"));
  expect(context.sections).toEqual([section]);
  expect(context.sections[0]).toBe(section);
  expect(Object.isFrozen(context)).toBe(true);
  expect(Object.isFrozen(context.sections)).toBe(true);
  expectError(() => new MemoryContext({ sections: new Set() as never }), TypeError, "sections must be an array");
  expectError(
    () => new MemoryContext({ sections: [{ memory: "notes", text: "" }] as never }),
    TypeError,
    "sections must contain MemorySection values",
  );
});
