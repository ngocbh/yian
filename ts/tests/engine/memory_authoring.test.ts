/**
 * Authored flow memory identity in the manifest (`compile/manifest.ts`): rows of
 * `tests/engine/test_memory_authoring.py`.
 *
 * Ported: `test_authored_identity_includes_flow_id_and_policy_but_excludes_origin` (flow
 * memories), `test_policy_config_keeps_public_behavior_options`,
 * `test_manifest_keeps_every_value_of_a_forged_policy_config` and
 * `test_expansion_digest_excludes_flow_id_and_agent_memory` (the flow-id half).
 *
 * Changed:
 * - `CompiledFlow({}, [])` is `new CompiledFlow(new Map(), [])`, Python's `copy.deepcopy` is
 *   `copyGraph`, and ints in memory configs are `bigint` (`4n`), since a built-in key refuses a
 *   `number`.
 * - `test_manifest_keeps_every_value_of_a_forged_policy_config`: a `MemoryDecl` is frozen and keeps
 *   no private `_config`, so the row forges a `MemoryDecl` (its prototype plus own fields) whose
 *   `config` holds `api_token`, a key the constructor refuses, then swaps that config.
 *
 * Later (T3-22, with AGENT): the `agent_policy` lines of
 * `test_authored_identity_includes_flow_id_and_policy_but_excludes_origin`, the AGENT-memory half
 * of `test_expansion_digest_excludes_flow_id_and_agent_memory`, and the AGENT half of
 * `test_python_constructors_default_and_validate_authored_memory_fields`.
 *
 * Later (T3-22): `test_full_root_retains_flow_owned_policy_with_one_origin`,
 * `test_full_nested_definition_retains_flow_owned_policy_and_semantic_id`,
 * `test_compact_root_projects_policy_to_its_agent`,
 * `test_flat_agent_definition_projects_policy_to_its_agent`,
 * `test_compose_stamps_decl_sites_for_root_def_and_inline_owners`,
 * `test_non_agent_node_rejects_long_term_memory_at_the_authored_value`,
 * `test_malformed_flow_declaration_is_located_at_its_authored_occurrence`,
 * `test_retired_memory_policy_is_a_located_migration_error`,
 * `test_inline_body_rejects_flow_policy_but_allows_child_agent_policy`,
 * `test_authored_policy_config_rejects_host_private_fields` and
 * `test_compose_loaded_flow_id_changes_the_previous_memoryless_identity` (they load flows).
 *
 * Not ported: the `CompiledFlow` half of
 * `test_python_constructors_default_and_validate_authored_memory_fields` (`graph.test.ts` checks a
 * compiled flow's `flowId` and `memories`) and `test_decl_config_rejects_host_private_fields`
 * (`memory_decl_shape.test.ts` and `memory_identity.test.ts` hold the private-key rules).
 */

import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";

const ROOT = new DeclSite([], ["root"], null);

/** Return a node-free compiled flow holding `memories`. */
function flowWith(...memories: MemoryDecl[]): CompiledFlow {
  return new CompiledFlow(new Map(), [], { memories });
}

test("authored identity includes flow id and policy but excludes origin", () => {
  const plain = new CompiledFlow(new Map(), []);
  const samePlain = copyGraph(plain);
  const flowNamed = new CompiledFlow(new Map(), [], { flowId: "named" });
  const flowPolicy = flowWith(new MemoryDecl("reflective", "reflective", { max_items: 4n }, ROOT));
  const flowPolicyRelocated = flowWith(
    new MemoryDecl("reflective", "reflective", { max_items: 4n }, ROOT, new OriginRef("other/source")),
  );
  const flowPolicyRenamed = flowWith(new MemoryDecl("custom", "custom", { max_items: 4n }, ROOT));
  const flowPolicyReconfigured = flowWith(new MemoryDecl("reflective", "reflective", { max_items: 5n }, ROOT));
  const flowPolicyResited = flowWith(
    new MemoryDecl("reflective", "reflective", { max_items: 4n }, new DeclSite([], ["def", "x"], null)),
  );

  const plainManifest = flowManifest(plain);
  expect("flow_id" in plainManifest).toBe(false);
  expect("memories" in plainManifest).toBe(false);
  expect(flowManifest(samePlain)).toEqual(plainManifest);
  expect(flowFingerprint(samePlain)).toBe(flowFingerprint(plain));
  expect(flowFingerprint(flowNamed)).not.toBe(flowFingerprint(plain));
  expect(flowFingerprint(flowPolicy)).not.toBe(flowFingerprint(plain));
  expect(flowFingerprint(flowPolicyRelocated)).toBe(flowFingerprint(flowPolicy));
  expect(flowFingerprint(flowPolicyRenamed)).not.toBe(flowFingerprint(flowPolicy));
  expect(flowFingerprint(flowPolicyReconfigured)).not.toBe(flowFingerprint(flowPolicy));
  expect(flowFingerprint(flowPolicyResited)).not.toBe(flowFingerprint(flowPolicy));
  expect(flowManifest(flowPolicy).memories).toEqual([
    {
      config: {
        entries: [{ key: { type: "str", value: "max_items" }, value: { type: "int", value: "4" } }],
        type: "mapping",
      },
      name: "reflective",
      site: { node: null, owner: ["root"], scope: [] },
      type: "reflective",
    },
  ]);
  expect(expansionDigest(flowPolicy)).toBe(expansionDigest(plain));
});

test("policy config keeps public behavior options", () => {
  const config = {
    max_items: 20n,
    recall_limit: 5n,
    token_budget: 4096n,
    stop_token: "semantic-stop-marker",
    continuationToken: "semantic-continuation-marker",
    query_options: { strategy: "semantic", path_filter: "docs/**" },
    backend: { mode: "memory" },
    database_query_options: { timeout: 3n },
  };

  const policy = new MemoryDecl("custom", "custom", config, ROOT);

  expect(policy.config).toEqual(config);
  const rendered = JSON.stringify(flowManifest(flowWith(policy)));
  expect(rendered).toContain("semantic-stop-marker");
  expect(rendered).toContain("semantic-continuation-marker");
});

test("manifest keeps every value of a forged policy config", () => {
  const policy: MemoryDecl = Object.assign(Object.create(MemoryDecl.prototype), {
    name: "reflective",
    type: "reflective",
    config: { nested: { api_token: "first-forged-value" } },
    site: ROOT,
    origin: null,
  });
  const flow = flowWith(policy);

  const firstManifest = flowManifest(flow);
  expect(JSON.stringify(firstManifest)).toContain("first-forged-value");
  Object.assign(policy, { config: { nested: { api_token: "second-forged-value" } } });
  expect(flowManifest(flow)).not.toEqual(firstManifest);
});

test("expansion digest excludes flow id", () => {
  const plainFlow = new CompiledFlow(new Map(), []);
  const namedFlow = new CompiledFlow(new Map(), [], { flowId: "named" });

  expect(expansionDigest(namedFlow)).toBe(expansionDigest(plainFlow));
});
