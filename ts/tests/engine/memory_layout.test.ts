/**
 * Pure static memory-layout derivation and atomic scope application (`compile/memory_layout.ts`):
 * the rows of `tests/engine/test_memory_layout.py` that build flows by hand.
 *
 * Ported: `test_disabled_layout_needs_no_flow_id_and_leaves_every_scope_none`,
 * `test_memory_owning_python_flow_requires_exact_direct_boundaries` (every case),
 * `test_recursive_static_child_cycle_is_rejected`,
 * `test_memory_table_compares_configs_as_canonical_json` (every case),
 * `test_memory_table_merges_identical_declarations_at_one_site`,
 * `test_disabled_layout_has_an_empty_key_table`,
 * `test_config_clash_carries_every_other_site_as_a_related_origin`,
 * `test_config_clash_lists_a_repeated_declaration_once` (every case),
 * `test_memory_flow_without_an_id_resolves`, `test_one_agent_stacks_several_keys_in_authored_order`,
 * `test_agent_declaration_must_name_its_own_node` and `test_flow_declaration_must_not_name_a_node`.
 *
 * Changed:
 * - Python's `ValueError` is a `RangeError`; `MemorySetupError` extends `RangeError`.
 * - Python's private `_memory_table` is `memoryTable`, exported for tests.
 * - The configs are TS JSON: Python's int `1` is `1n`, its float `1.0` is the number `1`.
 * - The noncanonical boundary id (Python's `str` subclass) is `new String("__start__")`; the
 *   boundary ids are reassigned with `Object.assign` (they're `readonly` in TS's types only).
 * - `caught.value.__notes__` (Python's `add_note`) is `error.notes`.
 * - Python's `copy.deepcopy` of a declaration is a declaration rebuilt from equal parts.
 *
 * Ported too, the rows that load flows: `test_flow_start_and_end_scopes_carry_the_enclosing_keys`,
 * `test_composite_drivers_see_only_enclosing_keys`,
 * `test_repeated_definition_shares_one_owner_and_inline_children_own_their_keys`,
 * `test_apply_is_idempotent_and_does_not_change_authored_identity`,
 * `test_apply_scope_mismatch_is_atomic`,
 * `test_apply_revalidates_forged_nested_layout_scope_before_stamping`,
 * `test_apply_revalidates_a_forged_key_table_before_stamping`,
 * `test_apply_revalidates_forged_equal_existing_scope_atomically`,
 * `test_apply_stale_object_target_is_atomic`, `test_apply_rejects_shared_child_targets_before_stamping`,
 * `test_key_table_retains_config_and_origin_outside_node_scopes`,
 * `test_layout_records_key_table_with_sorted_owners_and_placements`,
 * `test_memory_stacks_dedupe_a_repeated_key_at_its_outermost_position`,
 * `test_single_agent_owner_is_conversation_eligible`,
 * `test_one_key_with_two_configs_is_a_load_error_at_the_first_site`,
 * `test_session_def_called_twice_has_one_owner_and_two_placements` (on a custom type),
 * `test_scopes_are_root_relative_and_carry_outermost_first_keys` and
 * `test_one_def_under_two_parents_has_one_stack_per_placement_and_one_owner`. Their CODE references
 * are TS's form (`./fixtures/compose_codefns.ts:echo`), resolved against the conformance directory.
 * Frozen values can't be changed in place, so Python's `object.__setattr__` forgeries are
 * unfrozen objects of the same class carrying the forged field (`forged`), and its
 * `copy.deepcopy` of a node is `copyGraph`. A key has no `placements` and no
 * `conversation_eligible` flag (session memory is gone), so those rows check the owners only.
 *
 * Not ported: the session rows (`test_agent_with_two_session_memories_is_a_setup_error`,
 * `test_flow_level_session_is_a_load_error`, `test_session_key_with_two_agent_owners_is_a_load_error`,
 * `test_flow_level_session_in_an_uncalled_def_is_not_an_error`,
 * `test_session_placements_must_resolve_one_compatibility` and
 * `test_session_compatibility_ignores_other_types_and_disabled_layouts`). TS-only instead: a
 * `session` memory, from YAML or built in code, is a load error.
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import {
  applyMemoryLayout,
  memoryStacks,
  memoryTable,
  type ResolvedMemoryLayout,
  resolveMemoryLayout,
} from "../../src/compile/memory_layout.ts";
import { CompiledFlow, copyGraph, END_ID, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl, MemoryScope, MemorySetupError, MemoryStack } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

import { CONFORMANCE_DIR, loadErrorAsync } from "../compose/declarations.ts";

/** Return the error `fn` throws, failing when it returns. */
function caught(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected an error");
}

/** Expect `fn` to throw a `MemorySetupError` whose message matches `pattern`, and return it. */
function setupError(fn: () => unknown, pattern?: RegExp): MemorySetupError {
  const error = caught(fn);
  expect(error).toBeInstanceOf(MemorySetupError);
  if (pattern !== undefined) expect((error as Error).message).toMatch(pattern);
  return error as MemorySetupError;
}

/** A CODE node returning 1. */
function code(id: string): CodeNode {
  return new CodeNode(id, { code: "return 1" });
}

/** Return one resolved target by its structural locator. */
function target(layout: ResolvedMemoryLayout, ...locator: string[]) {
  return layout.targets.find((item) => item.locator.join("\0") === locator.join("\0"))!;
}

test("disabled layout needs no flow id and leaves every scope null", () => {
  const flow = new CompiledFlow(new Map<string, Node>([["plain", code("plain")]]), []);
  flow.nodes.get("plain")!.memoryScope = new MemoryScope(["plain"]);

  const layout = resolveMemoryLayout(flow);

  expect(layout.enabled).toBe(false);
  expect(layout.memories).toEqual([]);
  expect(layout.targets.map((item) => [item.locator, item.scope])).toEqual([[["plain"], null]]);
  // Resolution is pure and ignores an existing derived scope.
  expect(flow.nodes.get("plain")!.memoryScope).toEqual(new MemoryScope(["plain"]));
  setupError(() => applyMemoryLayout(flow, layout), /different memory scope/);
});

test.each([
  ["missing-both", /missing direct START and END/],
  ["missing-start", /missing direct START/],
  ["missing-end", /missing direct END/],
  ["start-subclass", /start_id must be a non-empty exact string/],
  ["end-empty", /end_id must be a non-empty exact string/],
  ["same", /distinct START and END/],
  ["wrong-start-id", /start_id must be canonical '__start__'/],
  ["wrong-end-id", /end_id must be canonical '__end__'/],
  ["wrong-start-class", /START boundary must be an exact StartNode/],
  ["wrong-end-class", /END boundary must be an exact EndNode/],
] as const)("memory-owning flow requires exact direct boundaries (%s)", (kase, message) => {
  const origin = new OriginRef("flow/memories/x");
  let nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [] })],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  if (kase === "missing-both") nodes = new Map<string, Node>([["body", code("body")]]);
  else if (kase === "missing-start") nodes.delete(START_ID);
  else if (kase === "missing-end") nodes.delete(END_ID);
  else if (kase === "wrong-start-id") nodes.set("entry", new StartNode("entry", { params: [] }));
  else if (kase === "wrong-end-id") nodes.set("result", EndNode.value("result", { outputName: null }));
  else if (kase === "wrong-start-class") nodes.set(START_ID, new CodeNode(START_ID, { code: "return None" }));
  else if (kase === "wrong-end-class") nodes.set(END_ID, new CodeNode(END_ID, { code: "return None" }));
  const flow = new CompiledFlow(nodes, [], {
    flowId: "root",
    memories: [new MemoryDecl("reflective", "reflective", {}, new DeclSite([], ["root"], null), origin)],
  });
  if (kase === "start-subclass") Object.assign(flow, { startId: new String(START_ID) });
  else if (kase === "end-empty") Object.assign(flow, { endId: "" });
  else if (kase === "same") Object.assign(flow, { endId: START_ID });
  else if (kase === "wrong-start-id") Object.assign(flow, { startId: "entry" });
  else if (kase === "wrong-end-id") Object.assign(flow, { endId: "result" });

  const error = setupError(() => resolveMemoryLayout(flow), message);

  expect(error.origin).toBe(origin);
  expect([...nodes.values()].every((node) => node.memoryScope === null)).toBe(true);
});

test("recursive static child cycle is rejected", () => {
  const flow = new CompiledFlow(new Map(), [], { flowId: "root" });
  const cycle = new CallNode("cycle", { flowId: "root", child: flow });
  flow.nodes.set("cycle", cycle);

  setupError(() => resolveMemoryLayout(flow), /recursive child-flow cycle/);

  expect(cycle.memoryScope).toBeNull();
});

test.each([
  ["true", false, { limit: true }],
  ["1.0", false, { limit: 1 }],
  ["2", false, { limit: 2n }],
  ["true", true, { limit: true }],
  ["1.0", true, { limit: 1 }],
  ["2", true, { limit: 2n }],
] as const)("memory table compares configs as canonical JSON (other limit: %s, same site: %p)", (_label, sameSite, other) => {
  const root = new DeclSite([], ["root"], null);
  const first = new MemoryDecl("notes", "notes", { limit: 1n }, root);
  const clash = new MemoryDecl("notes", "notes", other, sameSite ? root : new DeclSite([], ["def", "x"], null));
  const error = setupError(
    () =>
      memoryTable([
        first,
        clash,
      ]),
    /memory 'notes' is declared with different type or config/,
  );
  expect(error.notes).toHaveLength(1);
});

test("memory table merges identical declarations at one site", () => {
  const site = new DeclSite([], ["root"], null);
  const first = new MemoryDecl("notes", "notes", { limit: 1n }, site);
  const again = new MemoryDecl("notes", "notes", { limit: 1n }, site);
  const table = memoryTable([
    first,
    again,
    first,
  ]);
  expect(table).toHaveLength(1);
  expect(table[0]!.owners).toEqual([site]);
});

test("disabled layout has an empty key table", () => {
  const flow = new CompiledFlow(new Map<string, Node>([["plain", code("plain")]]), []);
  const layout = resolveMemoryLayout(flow);
  expect(layout.memories).toEqual([]);
  expect(memoryStacks(layout)).toEqual([]);
});

test("config clash carries every other site as a related origin", () => {
  // DeclSite.compare order: ["def", "x"] < ["def", "y"] < ["root"].
  const first = new MemoryDecl("notes", "notes", { limit: 1n }, new DeclSite([], ["def", "x"], null), new OriginRef("a"));
  const second = new MemoryDecl("notes", "notes", { limit: 2n }, new DeclSite([], ["def", "y"], null), new OriginRef("b"));
  const third = new MemoryDecl("notes", "notes", { limit: 1n }, new DeclSite([], ["root"], null), new OriginRef("c"));
  const error = setupError(() =>
    memoryTable([
      third,
      second,
      first,
    ]),
  );
  expect(error.origin).toEqual(new OriginRef("a"));
  expect(error.related).toEqual([new OriginRef("b"), new OriginRef("c")]);
  expect(error.notes).toHaveLength(2);
});

test.each([
  ["first", false],
  ["first", true],
  ["other", false],
  ["other", true],
] as const)("config clash lists a repeated declaration once (repeated: %s, copied: %p)", (repeated, copied) => {
  // A def called from two nodes yields its declaration once per call: the same object, or an
  // equal copy when the run copies the child per call.
  const inDef = new MemoryDecl("notes", "notes", { limit: 3n }, new DeclSite([], ["def", "x"], null), new OriginRef("a"));
  const inRoot = new MemoryDecl("notes", "notes", { limit: 2n }, new DeclSite([], ["root"], null), new OriginRef("b"));
  const [twice, once] = repeated === "first" ? [inDef, inRoot] : [inRoot, inDef];
  const again = copied
    ? new MemoryDecl(
        twice.name,
        twice.type,
        { ...twice.config },
        new DeclSite([...twice.site.scope], [...twice.site.owner], twice.site.node),
        new OriginRef(twice.origin!.key),
      )
    : twice;
  expect(again === twice).toBe(!copied);
  const error = setupError(() =>
    memoryTable([
      twice,
      again,
      once,
    ]),
  );
  expect(error.origin).toEqual(new OriginRef("a"));
  expect(error.related).toEqual([new OriginRef("b")]);
  expect(error.notes).toEqual(["also declared at root"]);
});

test("memory flow without an id resolves", () => {
  const site = new DeclSite([], ["root"], "agent");
  const agent = new AgentNode("agent", { memories: [new MemoryDecl("reflective", "reflective", {}, site)] });
  const flow = new CompiledFlow(new Map<string, Node>([["agent", agent]]), []);
  expect(flow.flowId).toBeNull();

  const layout = resolveMemoryLayout(flow);

  expect(layout.enabled).toBe(true);
  expect(layout.memories.map((memory) => memory.key)).toEqual(["reflective"]);
  expect(target(layout, "agent").scope).toEqual(new MemoryScope(["agent"], { memories: ["reflective"] }));
});

test("one agent stacks several keys in authored order", () => {
  const site = new DeclSite([], ["root"], "agent");
  const agent = new AgentNode("agent", {
    memories: [new MemoryDecl("reflective", "reflective", {}, site), new MemoryDecl("episodic", "episodic", {}, site)],
  });
  const flow = new CompiledFlow(new Map<string, Node>([["agent", agent]]), []);

  const layout = resolveMemoryLayout(flow);

  expect(target(layout, "agent").scope!.memories).toEqual(["reflective", "episodic"]);
  expect(layout.memories.map((memory) => memory.key)).toEqual(["episodic", "reflective"]);
  agent.memories = [...agent.memories, new MemoryDecl("reflective", "reflective", {}, site)];
  setupError(() => resolveMemoryLayout(flow), /declares memory 'reflective' more than once/);
});

test("agent declaration must name its own node", () => {
  // A programmatic flow could otherwise make two AGENTs share one AGENT owner.
  const forged = new DeclSite([], ["root"], "a");
  const flow = new CompiledFlow(
    new Map<string, Node>([
      ["a", new AgentNode("a", { memories: [new MemoryDecl("chat", "episodic", {}, forged)] })],
      ["b", new AgentNode("b", { memories: [new MemoryDecl("chat", "episodic", {}, forged)] })],
    ]),
    [],
  );
  setupError(() => resolveMemoryLayout(flow), /AGENT node 'b' memory 'chat' names site node 'a'/);
});

test("flow declaration must not name a node", () => {
  const flow = new CompiledFlow(new Map<string, Node>([["a", new AgentNode("a")]]), []);
  flow.memories = [new MemoryDecl("notes", "episodic", {}, new DeclSite([], ["root"], "a"))];
  setupError(() => resolveMemoryLayout(flow), /flow memory 'notes' names site node 'a'/);
});

// ---------- loaded flows ----------

const NESTED = `id: root
name: root
long_term_memory: outer
defs:
  child:
    long_term_memory:
      inner: {type: inner, config: {limit: 3}}
    nodes:
      answer:
        kind: agent
        long_term_memory: local
        prompt: answer
        output: str
      normalize:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {topic: child}
        output: str
    output: \${answer.output}
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`;

/** Load `text` with the conformance directory as its base and return its compiled graph. */
async function compiled(text: string, sourceLabel?: string): Promise<CompiledFlow> {
  return (await loadFlow(text, { baseDir: CONFORMANCE_DIR, ...(sourceLabel === undefined ? {} : { sourceLabel }) }))
    .compiled;
}

/** Return an unfrozen object of `value`'s class carrying its fields with `changes` applied. */
function forged<T extends object>(value: T, changes: Record<string, unknown>): T {
  return Object.assign(Object.create(Object.getPrototypeOf(value)) as T, value, changes);
}

/** Return each key wrapped as a `String` object: equal as text, but not a string. */
function boxed(keys: readonly string[]): string[] {
  return keys.map((key) => new String(key) as unknown as string);
}

test("flow START and END scopes carry the enclosing keys", async () => {
  const withoutChildPolicy = await compiled(
    NESTED.replace("    long_term_memory:\n      inner: {type: inner, config: {limit: 3}}\n", ""),
  );
  const inherited = resolveMemoryLayout(withoutChildPolicy);

  for (const nodeId of [START_ID, END_ID]) {
    expect(target(inherited, "invoke", nodeId).scope).toEqual(
      new MemoryScope(["invoke", nodeId], { memories: ["outer"] }),
    );
  }
  expect(target(inherited, "invoke", "answer").scope!.memories).toEqual(["outer", "local"]);
  expect(inherited.memories.map((memory) => memory.key)).toEqual(["local", "outer"]);
});

test("composite drivers see only enclosing keys", async () => {
  const layout = resolveMemoryLayout(await compiled(NESTED));

  // The CALL sees only its enclosing flow's keys; the child flow's keys begin inside the called
  // child.
  expect(target(layout, "invoke").scope!.memories).toEqual(["outer"]);
  expect(layout.memories.map((memory) => memory.key)).toEqual(["inner", "local", "outer"]);
});

test("repeated definition shares one owner and inline children own their keys", async () => {
  const repeatedDef = await compiled(`id: root
name: root
defs:
  child:
    long_term_memory: reflective
    nodes:
      answer:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {topic: child}
        output: str
    output: \${answer.output}
nodes:
  left: {kind: call, call: child}
  right: {kind: call, call: child}
output: \${right.output}
`);
  const definitionLayout = resolveMemoryLayout(repeatedDef);
  expect(definitionLayout.memories).toHaveLength(1);
  expect(definitionLayout.memories[0]!.owners).toEqual([new DeclSite([], ["def", "child"], null)]);
  const stacks = new Map(memoryStacks(definitionLayout).map((stack) => [stack.locator.join("/"), stack.memories]));
  expect(stacks.get("left")).toEqual([]);
  expect(stacks.get("left/answer")).toEqual(["reflective"]);
  expect(stacks.get("right/answer")).toEqual(["reflective"]);

  const inline = await compiled(`id: root
name: root
nodes:
  left:
    kind: call
    nodes:
      answer:
        kind: agent
        long_term_memory: reflective
        prompt: left
        output: str
    output: \${answer.output}
  right:
    kind: call
    nodes:
      answer:
        kind: agent
        long_term_memory: reflective
        prompt: right
        output: str
    output: \${answer.output}
output: \${right.output}
`);
  const memories = resolveMemoryLayout(inline).memories;
  expect(memories).toHaveLength(1);
  const memory = memories[0]!;
  expect(memory.owners).toEqual([
    new DeclSite([], ["root", "node", "left", "inline"], "answer"),
    new DeclSite([], ["root", "node", "right", "inline"], "answer"),
  ]);
});

test("apply is idempotent and does not change authored identity", async () => {
  const flow = await compiled(NESTED);
  const manifest = flowManifest(flow);
  const fingerprint = flowFingerprint(flow);
  const layout = resolveMemoryLayout(flow);

  expect(resolveMemoryLayout(flow)).toEqual(layout);
  applyMemoryLayout(flow, layout);
  expect(resolveMemoryLayout(flow)).toEqual(layout);
  const firstScopes = layout.targets.map((item) => item.node.memoryScope);
  applyMemoryLayout(flow, layout);

  expect(layout.targets.map((item) => item.node.memoryScope)).toEqual(firstScopes);
  expect(flowManifest(flow)).toEqual(manifest);
  expect(flowFingerprint(flow)).toBe(fingerprint);
});

test("apply scope mismatch is atomic", async () => {
  const flow = await compiled(NESTED);
  const layout = resolveMemoryLayout(flow);
  const final = layout.targets.at(-1)!;
  final.node.memoryScope = new MemoryScope(["different"]);

  setupError(() => applyMemoryLayout(flow, layout), /different memory scope/);

  expect(layout.targets.slice(0, -1).every((item) => item.node.memoryScope === null)).toBe(true);
  expect(final.node.memoryScope).toEqual(new MemoryScope(["different"]));
});

test("apply revalidates a forged nested layout scope before stamping", async () => {
  const flow = await compiled(NESTED);
  const layout = resolveMemoryLayout(flow);
  const final = layout.targets.at(-1)!;
  const scope = forged(final.scope!, { memories: boxed(final.scope!.memories) });
  const forgedLayout = forged(layout, { targets: [...layout.targets.slice(0, -1), forged(final, { scope })] });

  const error = caught(() => applyMemoryLayout(flow, forgedLayout));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toMatch(/memories.*must be a string/);

  expect(layout.targets.every((item) => item.node.memoryScope === null)).toBe(true);
});

test("apply revalidates a forged key table before stamping", async () => {
  const flow = await compiled(NESTED);
  const layout = resolveMemoryLayout(flow);
  const memory = forged(layout.memories[0]!, { key: new String("inner") });
  const forgedLayout = forged(layout, { memories: [memory, ...layout.memories.slice(1)] });

  const error = caught(() => applyMemoryLayout(flow, forgedLayout));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toMatch(/key must be a string/);

  expect(layout.targets.every((item) => item.node.memoryScope === null)).toBe(true);
});

test("apply revalidates a forged equal existing scope atomically", async () => {
  const flow = await compiled(NESTED);
  const layout = resolveMemoryLayout(flow);
  const final = layout.targets.at(-1)!;
  const scope = forged(final.scope!, { memories: boxed(final.scope!.memories) });
  final.node.memoryScope = scope;

  setupError(() => applyMemoryLayout(flow, layout), /noncanonical memory scope/);

  expect(layout.targets.slice(0, -1).every((item) => item.node.memoryScope === null)).toBe(true);
  expect(final.node.memoryScope).toBe(scope);
});

test("apply stale object target is atomic", async () => {
  const flow = await compiled(NESTED);
  const layout = resolveMemoryLayout(flow);
  const replacement = copyGraph(flow.nodes.get("invoke")!);
  flow.nodes.set("invoke", replacement);

  setupError(() => applyMemoryLayout(flow, layout), /same node/);

  expect(replacement.memoryScope).toBeNull();
  expect(layout.targets.every((item) => item.node.memoryScope === null)).toBe(true);
});

test("apply rejects shared child targets before stamping", async () => {
  const child = await compiled(`id: child
name: child
long_term_memory: reflective
nodes:
  answer:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input: {topic: child}
    output: str
output: \${answer.output}
`);
  const flow = new CompiledFlow(
    new Map<string, Node>([
      ["left", new CallNode("left", { flowId: "child", child })],
      ["right", new CallNode("right", { flowId: "child", child })],
    ]),
    [],
    { flowId: "root" },
  );
  const layout = resolveMemoryLayout(flow);

  setupError(() => applyMemoryLayout(flow, layout), /multiple static paths/);

  expect(layout.targets.every((item) => item.node.memoryScope === null)).toBe(true);
});

test("key table retains config and origin outside node scopes", async () => {
  const flow = await compiled(NESTED, "memory.yaml");
  const layout = resolveMemoryLayout(flow);
  const inner = layout.memories.find((memory) => memory.key === "inner")!;

  expect(inner.config).toEqual({ limit: 3n });
  expect(inner.owners).toEqual([new DeclSite([], ["def", "child"], null)]);
  expect(inner.origin).toEqual(((flow.nodes.get("invoke") as CallNode).child as CompiledFlow).memories[0]!.origin);
  expect(inner.origin).not.toBeNull();
  expect("origin" in target(layout, "invoke", "answer").scope!).toBe(false);
});

test("layout records the key table with sorted owners", async () => {
  const source = NESTED.replace("long_term_memory: local", "long_term_memory: outer");
  const layout = resolveMemoryLayout(await compiled(source));

  expect(layout.memories.map((m) => [m.key, m.type, m.config, m.owners])).toEqual([
    ["inner", "inner", { limit: 3n }, [new DeclSite([], ["def", "child"], null)]],
    ["outer", "outer", {}, [new DeclSite([], ["def", "child"], "answer"), new DeclSite([], ["root"], null)]],
  ]);
  expect(layout.memories[1]!.spec.owners).toEqual(layout.memories[1]!.owners);
});

test("memory stacks dedupe a repeated key at its outermost position", async () => {
  const source = NESTED.replace("long_term_memory: local", "long_term_memory: outer");
  const layout = resolveMemoryLayout(await compiled(source));

  const stacks = memoryStacks(layout);
  const byLocator = new Map(stacks.map((stack) => [stack.locator.join("\0"), stack.memories]));
  const locators = stacks.map((stack) => stack.locator.join("\0"));
  expect(locators).toEqual([...byLocator.keys()].sort());
  expect(new Set(byLocator.keys())).toEqual(new Set(layout.targets.map((item) => item.locator.join("\0"))));
  expect(byLocator.get("invoke")).toEqual(["outer"]);
  expect(byLocator.get("invoke\0normalize")).toEqual(["outer", "inner"]);
  expect(byLocator.get("invoke\0answer")).toEqual(["outer", "inner"]);
});

test("a single agent owner owns its key", async () => {
  const flow = await compiled(
    "id: one\nname: one\nnodes:\n  ask:\n    kind: agent\n" +
      "    long_term_memory: chat\n    prompt: hi\n    output: str\n" +
      "output: ${ask.output}\n",
  );
  const memories = resolveMemoryLayout(flow).memories;
  expect(memories).toHaveLength(1);
  expect(memories[0]!.owners).toEqual([new DeclSite([], ["root"], "ask")]);
});

test("one key with two configs is a load error at the first site", async () => {
  const source = NESTED.replace(
    "long_term_memory: local",
    "long_term_memory: {inner: {type: inner, config: {limit: 4}}}",
  );
  const error = await loadErrorAsync(loadFlow(source, { baseDir: CONFORMANCE_DIR }));
  expect(error.message).toMatch(/memory 'inner'.*different type or config/);
  const notes = error.diagnostic.notes;
  expect(notes[0]).toBe("also declared at def/child AGENT 'answer'");
  expect(notes).toHaveLength(2);
  expect(notes[1]!).toStartWith("related declaration at <flow>:");
});

const AGENT_CALLS = `id: calls
name: calls
defs:
  body:
    nodes:
      ask:
        kind: agent
        long_term_memory: notes
        prompt: answer
        output: str
    output: \${ask.output}
nodes:
  a: {kind: call, call: body}
  b: {kind: call, call: body}
output: \${[a.output, b.output]}
`;

test("an agent memory def called twice has one owner", async () => {
  const memories = resolveMemoryLayout(await compiled(AGENT_CALLS)).memories;
  expect(memories).toHaveLength(1);
  expect(memories[0]!.owners).toEqual([new DeclSite([], ["def", "body"], "ask")]);
});

test("a session memory is a load error", async () => {
  const source =
    "id: chat\nname: chat\nnodes:\n" +
    "  ask:\n    kind: agent\n    long_term_memory: {history: session}\n    prompt: hi\n    output: str\n" +
    "output: ${ask.output}\n";
  const error = await loadErrorAsync(loadFlow(source));
  expect(error.message).toContain(
    "long_term_memory 'history': session memory was removed; carry the history as a loop field",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(6);
  const site = new DeclSite([], ["root"], "ask");
  expect(() => new MemoryDecl("history", "session", {}, site)).toThrow(
    new RangeError("session memory was removed; carry the history as a loop field"),
  );
});

test("scopes are root-relative and carry outermost-first keys", async () => {
  const layout = resolveMemoryLayout(await compiled(NESTED));

  expect(layout.enabled).toBe(true);
  expect("frames" in layout).toBe(false);
  expect("rootFlowId" in layout).toBe(false);
  expect(target(layout, START_ID).scope).toEqual(new MemoryScope([START_ID], { memories: ["outer"] }));
  expect(target(layout, "invoke").scope).toEqual(new MemoryScope(["invoke"], { memories: ["outer"] }));
  expect(target(layout, "invoke", "normalize").scope).toEqual(
    new MemoryScope(["invoke", "normalize"], { memories: ["outer", "inner"] }),
  );
  expect(target(layout, "invoke", "answer").scope).toEqual(
    new MemoryScope(["invoke", "answer"], { memories: ["outer", "inner", "local"] }),
  );
  for (const nodeId of [START_ID, END_ID]) {
    expect(target(layout, "invoke", nodeId).scope).toEqual(
      new MemoryScope(["invoke", nodeId], { memories: ["outer", "inner"] }),
    );
  }
  const sorted = [...layout.targets].sort((a, b) => (a.locator.join("\0") < b.locator.join("\0") ? -1 : 1));
  expect(memoryStacks(layout)).toEqual(sorted.map((item) => new MemoryStack(item.locator, item.scope!.memories)));
});

test("one def under two parents has one stack per placement and one owner", async () => {
  const flow = await compiled(`id: root
name: root
long_term_memory: outer
defs:
  leaf:
    long_term_memory: leaf_notes
    nodes:
      work:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {topic: leaf}
        output: str
    output: \${work.output}
  left_parent:
    long_term_memory: left_notes
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: \${leaf_call.output}
  right_parent:
    long_term_memory: right_notes
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: \${leaf_call.output}
nodes:
  left: {kind: call, call: left_parent}
  right: {kind: call, call: right_parent}
output: \${right.output}
`);

  const layout = resolveMemoryLayout(flow);
  const stacks = new Map(memoryStacks(layout).map((stack) => [stack.locator.join("/"), stack.memories]));

  expect(stacks.get("left/leaf_call/work")).toEqual(["outer", "left_notes", "leaf_notes"]);
  expect(stacks.get("right/leaf_call/work")).toEqual(["outer", "right_notes", "leaf_notes"]);
  expect(stacks.get("left")).toEqual(["outer"]);
  expect(stacks.get("right")).toEqual(["outer"]);
  const byKey = new Map(layout.memories.map((memory) => [memory.key, memory.spec]));
  expect([...byKey.keys()]).toEqual(["leaf_notes", "left_notes", "outer", "right_notes"]);
  expect(byKey.get("leaf_notes")!.owners).toEqual([new DeclSite([], ["def", "leaf"], null)]);
  expect(byKey.get("outer")!.owners).toEqual([new DeclSite([], ["root"], null)]);
});
