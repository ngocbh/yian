/**
 * Declaration sites, declarations, specs and stacks, from `tests/engine/test_memory_identity.py`:
 * `test_decl_site_validates_and_freezes`,
 * `test_decl_site_sort_is_total_and_puts_flow_before_agent`,
 * `test_memory_decl_snapshots_config_and_rejects_private_keys`,
 * `test_memory_decl_origins_traverse_and_rewrite_shared_carriers_once`,
 * `test_memory_spec_requires_sorted_owners_and_derived_eligibility`,
 * `test_memory_stack_requires_locator_and_unique_keys`,
 * `test_snapshot_envelope_names_its_memory_key`, `test_learning_carriers_name_their_memory_key`
 * and `test_package_docstring_and_exports_describe_keyed_memories`; plus a lone-surrogate node
 * id, a compiled spec whose key and config the declaration check would refuse, and a snapshot
 * deeper than the value nesting limit.
 *
 * Changed: sites sort with `DeclSite.compare` (Python sorts by `DeclSite.sort_key()`; the order
 * message names `DeclSite.compare`), and equal sites compare 0. A declaration's config is its
 * deep-frozen snapshot, so assigning to it throws `TypeError` (Python returns a fresh copy). The
 * declarations are frozen, so the origin rewrite returns rebuilt declarations and leaves the inputs
 * unchanged (Python rewrites them in place); rewriting the same array with the same `seen` returns
 * the same array. An envelope's snapshot is its deep-frozen copy (Python returns a fresh copy).
 * The package test reads `memory/index.ts`'s leading TSDoc for the docstring and pins the exact
 * list of runtime exports for `__all__` (the type-only exports are checked by the typecheck).
 * Python's `ValueError` is `RangeError`; a frozen field throws `TypeError` where Python raises
 * `FrozenInstanceError`.
 *
 * Not ported: the list-for-tuple row (TS has only arrays); the origin re-check during traversal
 * and rewrite (`memory declaration origin must be an OriginRef or None`), since a frozen
 * `MemoryDecl` checked its origin when built.
 *
 * Left for later: `test_backward_failure_and_commit_records_name_their_memory_key`; its
 * `BackwardFailure` line lands with `BackwardFailure`, and its commit-record half with the runtime
 * log (Later (T4-2)).
 */

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  DeclSite,
  iterMemoryDeclOrigins,
  LearningFailure,
  LearningResult,
  MemoryDecl,
  MemoryOperationFailure,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  PolicyCommitOutcome,
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
} from "../../src/memory/index.ts";

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

// test_decl_site_validates_and_freezes
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

// test_decl_site_sort_is_total_and_puts_flow_before_agent
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

// test_memory_decl_snapshots_config_and_rejects_private_keys
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

// test_memory_decl_origins_traverse_and_rewrite_shared_carriers_once
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

// test_memory_spec_requires_sorted_owners_and_derived_eligibility
test("a spec requires sorted owners and derived eligibility", () => {
  const agent = new DeclSite([], ["root"], "ask");
  const flow = new DeclSite([], ["root"]);
  const spec = new MemorySpec("chat", "session", {}, [agent], true);
  expect(spec.conversationEligible).toBe(true);
  expect(Object.isFrozen(spec.owners)).toBe(true);
  expect(new MemorySpec("notes", "episodic", {}, [flow, agent], false).owners).toEqual([flow, agent]);
  const eligibility = "conversation_eligible must be true exactly for one AGENT owner";
  expectError(() => new MemorySpec("chat", "session", {}, [flow], true), RangeError, eligibility);
  expectError(() => new MemorySpec("chat", "session", {}, [agent], false), RangeError, eligibility);
  const order = "owners must be sorted and unique by DeclSite.compare";
  expectError(() => new MemorySpec("notes", "episodic", {}, [agent, flow], false), RangeError, order);
  expectError(() => new MemorySpec("notes", "episodic", {}, [flow, flow], false), RangeError, order);
  expectError(() => new MemorySpec("notes", "episodic", {}, [], false), RangeError, "owners must be non-empty");
  expectError(
    () => new MemorySpec("notes", "episodic", {}, [agent], 1 as never),
    TypeError,
    "conversation_eligible must be a boolean",
  );
  expectError(() => new MemorySpec("", "t", {}, [flow], false), RangeError, "key must be non-empty");
  expectError(() => new MemorySpec("k", 1n as never, {}, [flow], false), TypeError, "type must be a string");
  expectError(
    () => new MemorySpec("k", "t", { api_key: 1n }, [flow], false),
    RangeError,
    "config contains forbidden private key 'api_key'",
  );
  expectError(
    () => new MemorySpec("k", "t", [] as never, [flow], false),
    TypeError,
    "config must be an exact JSON object",
  );
  expectError(
    () => new MemorySpec("k", "t", {}, new Set([flow]) as never, false),
    TypeError,
    "owners must be an array",
  );
  expectError(
    () => new MemorySpec("k", "t", {}, [{}] as never, false),
    TypeError,
    "owners items must be DeclSite values",
  );
});

test("a compiled spec takes a scoped key and keys the declaration check refuses", () => {
  const owners = [new DeclSite(["lib"], ["root"])];
  const spec = new MemorySpec("lib.notes", "episodic", { storage: 1n, bogus: 2n }, owners, false);
  expect(spec.key).toBe("lib.notes");
  expect(spec.config).toEqual({ bogus: 2n, storage: 1n });
  expect(Object.keys(spec.config)).toEqual(["bogus", "storage"]);
});

// test_memory_stack_requires_locator_and_unique_keys
test("a stack requires a locator and unique keys", () => {
  const stack = new MemoryStack(["child", "ask"], ["notes", "chat"]);
  expect(stack.memories).toEqual(["notes", "chat"]);
  expect(Object.isFrozen(stack.locator)).toBe(true);
  expect(new MemoryStack(["plain"], []).memories).toEqual([]);
  expectError(() => new MemoryStack([], ["notes"]), RangeError, "locator must be non-empty");
  expectError(() => new MemoryStack(["ask"], ["notes", "notes"]), RangeError, "memories must be unique");
  expectError(() => new MemoryStack(["ask"], [""]), RangeError, "memories components must be non-empty");
});

// test_snapshot_envelope_names_its_memory_key
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
  // The snapshot is checked first, as Python does.
  expectError(() => new MemorySnapshotEnvelope("", []), TypeError, "snapshot must be an exact JSON object");
});

test("a snapshot envelope takes a snapshot deeper than the value nesting limit", () => {
  let deep: JsonObject = {};
  for (let i = 0; i < 150; i++) deep = { d: deep };
  expect(new MemorySnapshotEnvelope("notes", deep).snapshot).toEqual(deep);
});

// test_learning_carriers_name_their_memory_key
test("learning carriers name their memory key", () => {
  const cause = new MemoryOperationFailure("commit", "declined", "Declined.", true);
  const failure = new LearningFailure("notes", "commit", cause);
  const outcome = new PolicyCommitOutcome("notes", "failed", { commitId: "c", failure: cause });
  expect([failure.memory, outcome.memory]).toEqual(["notes", "notes"]);
  const result = new LearningResult("failed", { failures: [failure], policies: [outcome] });
  expect(result.policies[0]!.memory).toBe("notes");
  const repeated = () => new LearningResult("failed", { failures: [failure], policies: [outcome, outcome] });
  expect(repeated).toThrow(RangeError);
  expect(repeated).toThrow(/memory keys must be unique/);
});

// test_package_docstring_and_exports_describe_keyed_memories
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
    "LearningFailure",
    "LearningResult",
    "MEMORY_IDENTIFIER_PATTERN",
    "MemoryDecl",
    "MemoryMutation",
    "MemoryOccurrenceKind",
    "MemoryOccurrenceSegment",
    "MemoryOperationError",
    "MemoryOperationFailure",
    "MemoryScope",
    "MemorySetupError",
    "MemorySnapshotEnvelope",
    "MemorySpec",
    "MemoryStack",
    "PolicyCommitOutcome",
    "checkMemoryDeclaration",
    "isMemory",
  ]);
});
