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
 * `test_agent_declaration_must_name_its_own_node`, `test_flow_declaration_must_not_name_a_node` and
 * `test_agent_with_two_session_memories_is_a_setup_error`.
 *
 * Changed:
 * - Python's `ValueError` is a `RangeError`; `MemorySetupError` extends `RangeError`.
 * - Python's private `_memory_table` is `memoryTable`, exported for tests.
 * - The configs are TS JSON: Python's int `1` is `1n`, its float `1.0` is the number `1`.
 * - The noncanonical boundary id (Python's `str` subclass) is `new String("__start__")`; the
 *   boundary ids are reassigned with `Object.assign` (they're `readonly` in TS's types only).
 * - `caught.value.__notes__` (Python's `add_note`) is `error.notes`.
 * - Python's `copy.deepcopy` of a declaration is a declaration rebuilt from equal parts.
 * - The "declares more than one session memory" message lists the keys as JSON strings
 *   (`"chat", "session"`), not Python's repr.
 *
 * Later (T3-22): `test_flow_start_and_end_scopes_carry_the_enclosing_keys`,
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
 * `test_session_def_called_twice_has_one_owner_and_two_placements`,
 * `test_flow_level_session_is_a_load_error`, `test_session_key_with_two_agent_owners_is_a_load_error`,
 * `test_flow_level_session_in_an_uncalled_def_is_not_an_error`,
 * `test_session_placements_must_resolve_one_compatibility`,
 * `test_session_compatibility_ignores_other_types_and_disabled_layouts`,
 * `test_scopes_are_root_relative_and_carry_outermost_first_keys` and
 * `test_one_def_under_two_parents_has_one_stack_per_placement_and_one_owner` (they load flows).
 */

import { expect, test } from "bun:test";

import {
  applyMemoryLayout,
  memoryStacks,
  memoryTable,
  type ResolvedMemoryLayout,
  resolveMemoryLayout,
} from "../../src/compile/memory_layout.ts";
import { CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl, MemoryScope, MemorySetupError } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

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
        [first, null],
        [clash, null],
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
    [first, null],
    [again, null],
    [first, null],
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
      [third, null],
      [second, null],
      [first, null],
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
      [twice, ["p1"]],
      [again, ["p2"]],
      [once, null],
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
  // A programmatic flow could otherwise make two AGENTs share one session owner.
  const forged = new DeclSite([], ["root"], "a");
  const flow = new CompiledFlow(
    new Map<string, Node>([
      ["a", new AgentNode("a", { memories: [new MemoryDecl("chat", "session", {}, forged)] })],
      ["b", new AgentNode("b", { memories: [new MemoryDecl("chat", "session", {}, forged)] })],
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

test("agent with two session memories is a setup error", () => {
  const site = new DeclSite([], ["root"], "agent");
  const agent = new AgentNode("agent", {
    memories: [new MemoryDecl("session", "session", {}, site), new MemoryDecl("chat", "session", {}, site)],
  });
  const flow = new CompiledFlow(new Map<string, Node>([["agent", agent]]), []);

  setupError(
    () => resolveMemoryLayout(flow),
    /AGENT 'agent' declares more than one session memory: "chat", "session"/,
  );
});
