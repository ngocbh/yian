/**
 * Cascade resolution and replay over stable paths in the static call tree: mirrors
 * `tests/engine/test_llm_cascade.py`.
 *
 * Ported: every row but the one below. Changed: ints are `bigint`s; errors pin their class
 * (`RangeError` where Python raises `ValueError`); the LOOP body's CODE reference is TS's form,
 * loaded with the conformance directory as `baseDir` (`loadConformanceFlow`);
 * `test_replay_uses_logged_map_after_environment_changes` changes the environment around the
 * replay where Python also patches `connection_snapshot` to fail (an ES module export a Bun test
 * can't replace).
 *
 * TS-only: `bindHostCredentials` on a copied flow leaves the original's `llmConfig` unchanged; a
 * NaN run input is a `RangeError` naming the field and the input (Python's `ValueError` reads
 * `input 't': non-finite number nan is not a yian value`).
 *
 * Later (T4-8): `test_effective_map_resolves_environment_defaults_once` (it runs the flow).
 */

import { expect, test } from "bun:test";

import { applyEffectiveLlmMap, bindHostCredentials, resolveLlmCascade, resolveRunLlm } from "../../src/compile/llm_cascade.ts";
import { type CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { mergeLlmConfig } from "../../src/llm_clients/config.ts";
import { HOST_KEY_MARKER } from "../../src/llm_clients/connection.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";

import { loadConformanceFlow } from "../compose/declarations.ts";

const DEFAULTS = { provider: "anthropic", model: "default-model" };
const PROVIDER_MODELS = { openai: "openai-default", google: "google-default" };

/** Resolve `flow` under the host layer `host` with the test defaults and saved models. */
function cascade(flow: CompiledFlow, host: Record<string, unknown> = {}): Record<string, Record<string, unknown>> {
  return resolveLlmCascade(flow, host, { defaults: DEFAULTS, providerModels: PROVIDER_MODELS });
}

/** Return the compiled flow `text` loads to. */
async function compiled(text: string): Promise<CompiledFlow> {
  return (await loadFlow(text)).compiled;
}

/** Return the static child flow of the callsite `nodeId` in `flow`. */
function childOf(flow: CompiledFlow, nodeId: string): CompiledFlow {
  return (flow.nodes.get(nodeId) as unknown as { child: CompiledFlow }).child;
}

/** Return the AGENT at static path `path` (callsite ids, then the node id, joined with `/`). */
function agentAt(flow: CompiledFlow, path: string): AgentNode {
  const parts = path.split("/");
  let current = flow;
  for (const callsite of parts.slice(0, -1)) current = childOf(current, callsite);
  return current.nodes.get(parts.at(-1)!) as AgentNode;
}

/** Return the error `run` throws, failing the test when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

test("merge: specific wins per field", () => {
  const specific = { model: "claude-opus-4-8" };
  const base = { provider: "anthropic", model: "claude-sonnet-4-5", anthropic: { temperature: 0.2 } };
  // specific wins per field; base fills the gaps (provider, anthropic block)
  expect(mergeLlmConfig(specific, base)).toEqual({
    provider: "anthropic",
    model: "claude-opus-4-8",
    anthropic: { temperature: 0.2 },
  });
});

test("merge: empty layers", () => {
  expect(mergeLlmConfig({}, {})).toEqual({});
  expect(mergeLlmConfig({ model: "x" }, {})).toEqual({ model: "x" });
  expect(mergeLlmConfig({}, { model: "y" })).toEqual({ model: "y" });
});

test("merge treats null as an unset field in every layer", () => {
  const specific = { provider: null, model: "specific-model", anthropic: null };
  const base = { provider: null, model: null, anthropic: { temperature: 0.2 }, openai: null };
  expect(mergeLlmConfig(specific, base)).toEqual({ model: "specific-model", anthropic: { temperature: 0.2 } });
});

test("merge with null is associative", () => {
  const node = { provider: null, model: "node-model", anthropic: null };
  const flow = { provider: "anthropic", model: null, anthropic: { temperature: 0.4 } };
  const caller = { provider: "openai", model: "caller-model", anthropic: { temperature: null } };
  expect(mergeLlmConfig(node, mergeLlmConfig(flow, caller))).toEqual(
    mergeLlmConfig(mergeLlmConfig(node, flow), caller),
  );
});

test("merge does not mutate its inputs", () => {
  const a = { model: "x" };
  const b = { provider: "p" };
  mergeLlmConfig(a, b);
  expect(a).toEqual({ model: "x" });
  expect(b).toEqual({ provider: "p" });
});

test("a node fills its gaps from the flow, then the CLI", async () => {
  const flow = await compiled(`
id: f
name: f
llm_config: {provider: anthropic}
nodes:
  a: {kind: agent, prompt: hi, llm_config: {model: claude-opus-4-8}}
output: \${a.output}
`);
  cascade(flow, { anthropic: { temperature: 0.3 } });
  // node sets model; flow fills provider; CLI fills the anthropic block
  expect(agentAt(flow, "a").llmConfig).toEqual({
    model: "claude-opus-4-8",
    provider: "anthropic",
    anthropic: { temperature: 0.3 },
  });
});

test("authored null fields fall through the cascade", async () => {
  const flow = await compiled(`
id: f
name: f
llm_config: {provider: null, model: null, anthropic: {temperature: 0.7}}
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: {provider: null, model: null, anthropic: {temperature: null}}
output: \${a.output}
`);
  const a = agentAt(flow, "a");
  expect(flow.flowLlmConfig).toEqual({ provider: null, model: null, anthropic: { temperature: 0.7 } });
  expect(a.ownLlmConfig).toEqual({ provider: null, model: null, anthropic: { temperature: null } });

  const effective = cascade(flow);

  expect(effective["a"]).toEqual({ provider: "anthropic", model: "default-model", anthropic: { temperature: 0.7 } });
  expect(a.llmConfig).toEqual(effective["a"]!);
});

test("an inherit: false node ignores the authored inheritance layers but gets the defaults", async () => {
  const flow = await compiled(`
id: f
name: f
llm_config: {provider: anthropic, anthropic: {temperature: 0.9}}
nodes:
  a: {kind: agent, prompt: hi, llm_config: {model: m, inherit: false}}
output: \${a.output}
`);
  cascade(flow, { anthropic: { temperature: 0.1 } });
  expect(agentAt(flow, "a").llmConfig).toEqual({ model: "m", provider: "anthropic" });
});

test("resolution is idempotent from the node's own layer", async () => {
  // Re-resolving with a different CLI layer recomputes from ownLlmConfig, not the previously
  // baked effective record.
  const flow = await compiled(
    "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi, llm_config: {model: m}}\noutput: ${a.output}\n",
  );
  cascade(flow, { provider: "openai" });
  cascade(flow, { provider: "google" });
  expect(agentAt(flow, "a").llmConfig).toEqual({ model: "m", provider: "google" });
});

test("the cascade reaches nested def agents", async () => {
  const flow = await compiled(`
id: f
name: f
llm_config: {provider: anthropic}
defs:
  sub:
    nodes:
      inner: {kind: agent, prompt: hi}
    output: \${inner.output}
nodes:
  c: {kind: call, call: sub}
output: \${c.output}
`);
  cascade(flow, { model: "claude-opus-4-8" });
  // inner sets nothing -> fills from the top flow (provider) + CLI (model)
  expect(agentAt(flow, "c/inner").llmConfig).toEqual({ provider: "anthropic", model: "claude-opus-4-8" });
});

test("named and inline children receive the same LLM cascade", async () => {
  const named = await compiled(`id: named-cascade
name: named_cascade
llm_config: {provider: openai}
defs:
  child:
    nodes:
      inner:
        kind: agent
        prompt: hi
        llm_config: {openai: {temperature: 0.7}}
    output: \${inner.output}
nodes:
  invoke: {kind: call, call: child}
output: \${invoke.output}
`);
  const inline = await compiled(`id: inline-cascade
name: inline_cascade
llm_config: {provider: openai}
nodes:
  invoke:
    kind: call
    nodes:
      inner:
        kind: agent
        prompt: hi
        llm_config: {openai: {temperature: 0.7}}
    output: \${inner.output}
output: \${invoke.output}
`);

  const caller = { model: "caller-model" };
  const namedEffective = cascade(named, caller);
  const inlineEffective = cascade(inline, caller);

  const expected = { "invoke/inner": { provider: "openai", model: "caller-model", openai: { temperature: 0.7 } } };
  expect(inlineEffective).toEqual(expected);
  expect(namedEffective).toEqual(expected);
  expect(agentAt(inline, "invoke/inner").llmConfig).toEqual(agentAt(named, "invoke/inner").llmConfig!);
});

test("nested re-resolution recomputes from the node's own layer", async () => {
  // The riskier path: each resolution replaces node.child with a fresh copy, so re-running over a
  // CALL must still recompute the inner agent from its own config (never compound the previously
  // baked effective record).
  const flow = await compiled(`
id: f
name: f
defs:
  sub:
    nodes:
      inner: {kind: agent, prompt: hi}
    output: \${inner.output}
nodes:
  c: {kind: call, call: sub}
output: \${c.output}
`);
  cascade(flow, { provider: "openai", model: "x" });
  cascade(flow, { provider: "google" });
  // The second run dropped `model` (not in the new CLI layer); it did not keep it from run 1.
  expect(agentAt(flow, "c/inner").llmConfig).toEqual({ provider: "google", model: "google-default" });
});

test("the cascade reaches loop body agents", async () => {
  // A LOOP carries a static child body like CALL/MAP; its body agents inherit the parent flow and
  // CLI layers too.
  const flow = (
    await loadConformanceFlow(`
id: f
name: f
llm_config: {provider: anthropic}
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes:
      think: {kind: agent, prompt: hi}
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:make_report
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
output: \${chat_loop.output}
`)
  ).compiled;
  cascade(flow, { model: "claude-opus-4-8" });
  // The body agent sets nothing -> fills from the top flow (provider) + CLI (model).
  expect(agentAt(flow, "chat_loop/think").llmConfig).toEqual({ provider: "anthropic", model: "claude-opus-4-8" });
});

const PATH_FLOW = `
id: path-flow
name: path_flow
defs:
  child:
    nodes:
      inner: {kind: agent, prompt: inner}
    output: \${inner.output}
nodes:
  outer: {kind: agent, prompt: outer, llm_config: {model: outer-model}}
  invoke: {kind: call, call: child}
output: \${invoke.output}
`;

test("the effective map round trips static paths", async () => {
  const resolved = await compiled(PATH_FLOW);
  const effective = cascade(resolved, { provider: "openai" });

  expect(effective).toEqual({
    outer: { provider: "openai", model: "outer-model" },
    "invoke/inner": { provider: "openai", model: "openai-default" },
  });
  expect(effective["outer"]).not.toBe(agentAt(resolved, "outer").llmConfig!);
  expect(effective["outer"]).not.toBe(effective["invoke/inner"]!);
  expect(agentAt(resolved, "outer").llmConfig).not.toBe(agentAt(resolved, "invoke/inner").llmConfig);

  const replayed = await compiled(PATH_FLOW);
  applyEffectiveLlmMap(replayed, effective);
  expect(agentAt(replayed, "outer").llmConfig).toEqual(effective["outer"]!);
  expect(agentAt(replayed, "invoke/inner").llmConfig).toEqual(effective["invoke/inner"]!);
  expect(agentAt(replayed, "outer").llmConfig).not.toBe(effective["outer"]!);
});

test("the effective map rejects missing or extra paths", async () => {
  const effective = cascade(await compiled(PATH_FLOW));

  const missing = structuredClone(effective);
  delete missing["invoke/inner"];
  let flow = await compiled(PATH_FLOW);
  let error = thrown(() => applyEffectiveLlmMap(flow, missing));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/missing.*invoke\/inner/);

  const extra = { ...structuredClone(effective), ghost: { ...DEFAULTS } };
  flow = await compiled(PATH_FLOW);
  error = thrown(() => applyEffectiveLlmMap(flow, extra));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/extra.*ghost/);
});

const SHARED_LEAF_FLOW = `
id: shared-leaf
name: shared_leaf
defs:
  leaf:
    nodes:
      agent: {kind: agent, prompt: hi}
    output: \${agent.output}
  openai_parent:
    llm_config: {provider: openai}
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: \${leaf_call.output}
  anthropic_parent:
    llm_config: {provider: anthropic}
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: \${leaf_call.output}
nodes:
  left: {kind: call, call: openai_parent}
  right: {kind: call, call: anthropic_parent}
output: \${right.output}
`;

test("a shared child is de-aliased per parent provider", async () => {
  const flow = await compiled(SHARED_LEAF_FLOW);
  expect(childOf(childOf(flow, "left"), "leaf_call")).toBe(childOf(childOf(flow, "right"), "leaf_call"));

  const effective = cascade(flow);

  const left = childOf(childOf(flow, "left"), "leaf_call");
  const right = childOf(childOf(flow, "right"), "leaf_call");
  expect(left).not.toBe(right);
  expect(effective["left/leaf_call/agent"]!["provider"]).toBe("openai");
  expect(effective["right/leaf_call/agent"]!["provider"]).toBe("anthropic");
  expect((left.nodes.get("agent") as AgentNode).llmConfig!["provider"]).toBe("openai");
  expect((right.nodes.get("agent") as AgentNode).llmConfig!["provider"]).toBe("anthropic");
});

test("a replay uses the logged map after the environment changes", async () => {
  const recorded = resolveLlmCascade(
    await compiled(SHARED_LEAF_FLOW),
    {},
    { defaults: { provider: "openai", model: "recorded-model" }, providerModels: { anthropic: "recorded-anthropic" } },
  );
  const replayed = await compiled(SHARED_LEAF_FLOW);
  expect(childOf(childOf(replayed, "left"), "leaf_call")).toBe(childOf(childOf(replayed, "right"), "leaf_call"));

  const saved = { provider: process.env["YIAN_DEFAULT_PROVIDER"], model: process.env["YIAN_DEFAULT_MODEL"] };
  process.env["YIAN_DEFAULT_PROVIDER"] = "anthropic";
  process.env["YIAN_DEFAULT_MODEL"] = "changed-model";
  try {
    applyEffectiveLlmMap(replayed, recorded);
  } finally {
    for (const [name, value] of [
      ["YIAN_DEFAULT_PROVIDER", saved.provider],
      ["YIAN_DEFAULT_MODEL", saved.model],
    ] as const) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }

  const left = childOf(childOf(replayed, "left"), "leaf_call");
  const right = childOf(childOf(replayed, "right"), "leaf_call");
  expect(left).not.toBe(right);
  expect((left.nodes.get("agent") as AgentNode).llmConfig).toEqual(recorded["left/leaf_call/agent"]!);
  expect((right.nodes.get("agent") as AgentNode).llmConfig).toEqual(recorded["right/leaf_call/agent"]!);
  expect((left.nodes.get("agent") as AgentNode).llmConfig!["provider"]).toBe("openai");
  expect((right.nodes.get("agent") as AgentNode).llmConfig!["provider"]).toBe("anthropic");
});

const ONE_AGENT = `
id: f
name: f
nodes:
  a: {kind: agent, prompt: hi}
output: \${a.output}
`;

test.each([
  ["modle", { modle: "x" }],
  ["inherit", { provider: "openai", inherit: false }],
])("the host layer rejects unknown fields: %s", async (_name, layer) => {
  const flow = await compiled(ONE_AGENT);
  const error = thrown(() => cascade(flow, layer));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/llm_config has unknown field\(s\)/);
});

test("the host layer's Python block spelling names the dashed key", async () => {
  const flow = await compiled(ONE_AGENT);
  const error = thrown(() => cascade(flow, { qwen_cn: { top_k: 4n } }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("`qwen_cn` is not an llm_config key; write `qwen-cn`");
});

test("a host layer block merges per field with authored blocks", async () => {
  const flow = await compiled(`
id: f
name: f
llm_config:
  qwen-cn: {enable_thinking: true}
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: {qwen-cn: {top_k: 4}}
output: \${a.output}
`);
  const effective = cascade(flow, { provider: "qwen-cn", model: "qwen3-max", "qwen-cn": { top_k: 1n, seed: 7n } });
  expect(effective["a"]!["qwen-cn"]).toEqual({ enable_thinking: true, top_k: 4n, seed: 7n });
});

test("bindHostCredentials on a copied flow leaves the original's llmConfig unchanged", async () => {
  const flow = await compiled(PATH_FLOW);
  const effective = cascade(flow, { provider: "openai" });
  for (const config of Object.values(effective)) config[HOST_KEY_MARKER] = true;
  applyEffectiveLlmMap(flow, effective);
  const before = structuredClone(agentAt(flow, "invoke/inner").llmConfig);

  const copy = copyGraph(flow);
  bindHostCredentials(copy, "sk-secret", { "X-Tenant": "tenant-secret" });

  for (const path of ["outer", "invoke/inner"]) {
    expect(agentAt(copy, path).llmConfig!["api_key"]).toBe("sk-secret");
    expect(agentAt(copy, path).llmConfig!["headers"]).toEqual({ "X-Tenant": "tenant-secret" });
    expect(JSON.stringify(agentAt(flow, path).llmConfig)).not.toContain("secret");
  }
  expect(agentAt(flow, "invoke/inner").llmConfig).toEqual(before!);
});

test("a NaN run input names the field and the input", async () => {
  const flow = await compiled(`
id: f
name: f
input:
  t: !float 1.0
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: openai, model: m, openai: {temperature: '\${input.t}'}}
output: \${chat.output}
`);
  const error = thrown(() => resolveRunLlm(flow, {}, { defaults: {}, runInputs: { t: Number.NaN } }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("node 'chat' llm_config.openai.temperature: input 't': NaN is not a yian value");
});
