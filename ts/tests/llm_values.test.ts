/**
 * Mirrors `tests/test_llm_values.py` (all 32 rows). The `keepValues`, `resolveLayer`,
 * `checkResolvedLayer`, `agentLabel`/`fieldWhere`, `invocationConfig` and `callTimeConfig` rows
 * are TS-only: Python reaches those functions only through compile and runs. The
 * `callTimeConfig` rows pin the messages of `tests/engine/test_llm_config_values_call_time.py`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../src/defaults.ts";
import { OriginRef } from "../src/diagnostics.ts";
import { AuthError, OAuthCredentials } from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { HostConnection, SUPPORTED_LLM_PROVIDERS } from "../src/llm_clients/config.ts";
import { ConnectionSelection, connectionSnapshot, HOST_KEY_MARKER } from "../src/llm_clients/connection.ts";
import {
  agentLabel,
  CallTimeLLM,
  callTimeConfig,
  checkResolvedLayer,
  contractNullable,
  contributing,
  fieldWhere,
  invocationConfig,
  keepValues,
  LayerFieldError,
  layerRefs,
  readsOnlyInputs,
  refusedResolvedField,
  resolveLayer,
  resolveLayers,
  splitLayers,
  typeNullable,
  typesNullable,
  valueLeaves,
} from "../src/nodes/agent/llm_values.ts";
import { BindingError } from "../src/nodes/binding.ts";
import { Binding, ContractOrigins, type ValuePath, ValueContract, ValueTemplate } from "../src/nodes/contracts.ts";
import { Type, ValueKind, VariablePool } from "../src/typesys/index.ts";

const STR = Type.scalar(ValueKind.STRING);
const OPT_STR = new Type({ kind: ValueKind.STRING, nullable: true });

/** Return a typed binding of `source` declared as `typ`, authored at `origin`. */
function typed(source: string, typ: Type, origin: OriginRef | null = null): Binding {
  return new Binding(source, { origin, contract: new ValueContract(typ, new ContractOrigins()) });
}

/** Return a record type whose fields are all required. */
function record(fields: Record<string, Type>, nullable = false): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
    nullable,
  });
}

/** Return what of `lower` still contributes under `higher`, typed by tags only. */
function keep(lower: unknown, higher: unknown): unknown {
  return contributing(lower, higher, contractNullable);
}

/** Return the template tree a `splitLayers` binding carries. */
function carriedTree(carried: Binding | null): Record<string, unknown> {
  expect(carried).toBeInstanceOf(Binding);
  return (carried!.source as ValueTemplate).value as Record<string, unknown>;
}

// Names a field under each layer's subject, as a call-time agent `chat` does.
const WHERE = {
  nodeWhere: (field: ValuePath) => "node 'chat' llm_config" + field.map((part) => `.${part}`).join(""),
  flowWhere: (field: ValuePath) => "flow llm_config" + field.map((part) => `.${part}`).join(""),
};

/** Return a pool whose run input is `start` and whose nodes produced `outputs`. */
function pool(start: Record<string, unknown> = {}, outputs: Record<string, unknown> = {}): VariablePool {
  const values = new VariablePool();
  values.set(values.startId, start);
  for (const [nodeId, value] of Object.entries(outputs)) values.set(nodeId, value);
  return values;
}

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** Return the `BindingError` `run` throws, failing the test on any other outcome. */
function bindingError(run: () => unknown): BindingError {
  const error = thrown(run);
  expect(error).toBeInstanceOf(BindingError);
  return error as BindingError;
}

/** Return the `LayerFieldError` `run` throws, failing the test on any other outcome. */
function layerFieldError(run: () => unknown): LayerFieldError {
  const error = thrown(run);
  expect(error).toBeInstanceOf(LayerFieldError);
  return error as LayerFieldError;
}

// ---------- valueLeaves ----------

test("value leaves of each form", () => {
  const tagged = typed("${pick.output}", STR);
  const block = new ValueTemplate({
    model: "${input.m}",
    provider: "openai",
    anthropic: {
      max_tokens: tagged,
      thinking: { type: "enabled", budget_tokens: "${input.b}" },
    },
    openai: "${input.block}",
  });
  const whole = new Binding("${input.cfg}");

  expect(valueLeaves(block)).toEqual([
    [["model"], "${input.m}"],
    [["anthropic", "max_tokens"], tagged],
    [["anthropic", "thinking", "budget_tokens"], "${input.b}"],
    [["openai"], "${input.block}"],
  ]);
  expect(valueLeaves(whole)).toEqual([[[], whole]]);
  expect(valueLeaves({ model: "${not.a.value}" })).toEqual([]);
  expect(valueLeaves(null)).toEqual([]);
});

test("tagged literals and escaped text are not value leaves", () => {
  const layer = new ValueTemplate({ model: typed("gpt-4", STR), provider: "a$${b}", anthropic: "${input.a}" });

  expect(valueLeaves(layer)).toEqual([[["anthropic"], "${input.a}"]]);
});

// ---------- layerRefs / readsOnlyInputs ----------

test("layer refs and input-only detection", () => {
  const inputs = new ValueTemplate({ model: "${input.m}", openai: { reasoning: "x-${input.r}" } });
  const mixed = new ValueTemplate({ model: "${pick.output | input.m}" });
  const system = new Binding("${system.run_id}");

  expect(layerRefs(inputs)).toEqual(["input.m", "input.r"]);
  expect(layerRefs(mixed)).toEqual(["pick.output", "input.m"]);
  expect(layerRefs({ model: "m" })).toEqual([]);
  expect(readsOnlyInputs(inputs)).toBe(true);
  expect(readsOnlyInputs(new Binding("${input}"))).toBe(true);
  expect(readsOnlyInputs({ model: "m" })).toBe(true);
  expect(readsOnlyInputs(mixed)).toBe(false);
  expect(readsOnlyInputs(system)).toBe(false);
});

// ---------- contributing ----------

test("a literal shields a lower value", () => {
  const lower = new ValueTemplate({ model: "${pick.output}", provider: "${input.p}" });

  const kept = keep(lower, { model: "gpt-4" });

  expect(valueLeaves(kept)).toEqual([[["provider"], "${input.p}"]]);
});

test("a null literal does not shield", () => {
  const lower = new ValueTemplate({ model: "${pick.output}" });

  expect(keep(lower, { model: null })).toBe(lower);
});

test("a non-nullable value shields and a nullable one does not", () => {
  const lower = new ValueTemplate({ model: "${pick.output}" });
  const strict = new ValueTemplate({ model: typed("${ask.output}", STR) });
  const loose = new ValueTemplate({ model: typed("${ask.output}", OPT_STR) });
  const untyped = new ValueTemplate({ model: "${ask.output}" });
  const embedded = new ValueTemplate({ model: "gpt-${ask.output}" });

  expect(valueLeaves(keep(lower, strict))).toEqual([]);
  expect(keep(lower, loose)).toBe(lower);
  expect(keep(lower, untyped)).toBe(lower);
  expect(valueLeaves(keep(lower, embedded))).toEqual([]);
});

test("the nullable predicate decides untagged values", () => {
  const lower = new ValueTemplate({ model: "${pick.output}" });
  const higher = new ValueTemplate({ model: "${ask.output}" });
  const seen: [unknown, ValuePath][] = [];

  const neverNull = (leaf: unknown, inner: ValuePath): boolean => {
    seen.push([leaf, inner]);
    return false;
  };

  expect(valueLeaves(contributing(lower, higher, neverNull))).toEqual([]);
  expect(seen).toEqual([["${ask.output}", []]]);
});

test("shielding follows block fields", () => {
  const lower = new ValueTemplate(
    {
      anthropic: {
        max_tokens: "${input.n}",
        thinking: { type: "enabled", budget_tokens: "${input.b}" },
      },
    },
    [
      [["anthropic", "max_tokens"], new OriginRef("site/max")],
      [["anthropic", "thinking", "budget_tokens"], new OriginRef("site/budget")],
    ],
  );
  const higher = { anthropic: { thinking: { type: "disabled" } } };

  const kept = keep(lower, higher) as ValueTemplate;

  expect(valueLeaves(kept)).toEqual([[["anthropic", "max_tokens"], "${input.n}"]]);
  expect(kept.value).toEqual({ anthropic: { max_tokens: "${input.n}" } });
  expect(kept.origins).toEqual([[["anthropic", "max_tokens"], new OriginRef("site/max")]]);
});

test("a whole block survives a partial block", () => {
  const lower = new ValueTemplate({ anthropic: "${input.block}" });
  const higher = new ValueTemplate({ anthropic: { max_tokens: 10, temperature: "${input.t}" } });

  expect(keep(lower, higher)).toBe(lower);
});

test("a whole record survives a partial layer", () => {
  const lower = new Binding("${input.cfg}");

  expect(keep(lower, { model: "m", provider: "openai" })).toBe(lower);
});

test("a whole block shields the fields its type declares", () => {
  const lower = new ValueTemplate({ anthropic: { max_tokens: "${a.output}", top_k: "${b.output}" } });
  const block = typed("${input.block}", record({ max_tokens: Type.scalar(ValueKind.INTEGER) }));
  const higher = new ValueTemplate({ anthropic: block });

  const kept = keep(lower, higher);

  expect(valueLeaves(kept)).toEqual([[["anthropic", "top_k"], "${b.output}"]]);
});

test("a whole record shields by its declared fields", () => {
  const lower = new ValueTemplate({ model: "${pick.output}", provider: "${input.p}" });
  const strict = typed("${input.cfg}", record({ model: STR }));
  const looseField = typed("${input.cfg}", record({ model: OPT_STR }));
  const looseRecord = typed("${input.cfg}", record({ model: STR }, true));
  const optionalField = typed(
    "${input.cfg}",
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["model", STR]]), required: new Set() }),
  );

  expect(valueLeaves(keep(lower, strict))).toEqual([[["provider"], "${input.p}"]]);
  expect(keep(lower, looseField)).toBe(lower);
  expect(keep(lower, looseRecord)).toBe(lower);
  expect(keep(lower, optionalField)).toBe(lower);
  expect(keep(lower, new Binding("${input.cfg}"))).toBe(lower);
});

test("a whole record shields block fields through its type", () => {
  const lower = new ValueTemplate({ openai: { reasoning: "${pick.output}" } });
  const cfg = record({ openai: record({ reasoning: Type.scalar(ValueKind.OBJECT) }) });
  const higher = typed("${input.cfg}", cfg);

  expect(valueLeaves(keep(lower, higher))).toEqual([]);
});

test("typeNullable walks records and unions", () => {
  const one = record({ kind: new Type({ kind: ValueKind.STRING, tags: new Set(["a"]) }), model: STR });
  const two = record({ kind: new Type({ kind: ValueKind.STRING, tags: new Set(["b"]) }), model: OPT_STR });
  const union = new Type({ kind: ValueKind.OBJECT, variants: [one, two] });

  expect(typeNullable(one, ["model"])).toBe(false);
  expect(typeNullable(one, ["provider"])).toBe(true);
  expect(typeNullable(union, ["model"])).toBe(true);
  expect(typeNullable(union, ["kind"])).toBe(false);
  expect(typeNullable(Type.scalar(ValueKind.OBJECT), ["model"])).toBe(true);
  expect(typeNullable(null, [])).toBe(true);
  expect(typeNullable(Type.scalar(ValueKind.ANY), [])).toBe(false);
});

// ---------- splitLayers ----------

test("splitLayers carries a mixed flow layer whole", () => {
  const node = new ValueTemplate({ model: "${pick.output}" }, [[["model"], new OriginRef("node/model")]]);
  const flow = new ValueTemplate(
    { provider: "anthropic", temperature: 0.2, anthropic: { top_k: "${input.k}" } },
    [[["anthropic", "top_k"], new OriginRef("flow/top_k")]],
  );

  const carried = splitLayers(node, flow, true, contractNullable);

  expect(carried).toBeInstanceOf(Binding);
  const template = carried!.source;
  expect(template).toBeInstanceOf(ValueTemplate);
  expect((template as ValueTemplate).value).toEqual({ node: node.value, flow: flow.value });
  const origins = (template as ValueTemplate).origins;
  expect(origins).toHaveLength(2);
  expect(origins).toEqual(
    expect.arrayContaining([
      [["node", "model"], new OriginRef("node/model")],
      [["flow", "anthropic", "top_k"], new OriginRef("flow/top_k")],
    ]),
  );
});

test("splitLayers escapes a literal layer", () => {
  const node = new ValueTemplate({ model: "${pick.output}" });

  const carried = splitLayers(node, { provider: "a${b}$" }, true, contractNullable);

  expect(carriedTree(carried).flow).toEqual({ provider: "a$${b}$$" });
  expect(resolveLayers(carried!, pool({}, { pick: "m" }), WHERE)).toEqual([{ model: "m" }, { provider: "a${b}$" }]);
});

test("splitLayers drops the flow layer without inherit", () => {
  const node = new ValueTemplate({ model: "${pick.output}" });
  const flow = new ValueTemplate({ provider: "${pick2.output}" });

  const carried = splitLayers(node, flow, false, contractNullable);

  expect(carriedTree(carried)).toEqual({ node: node.value });
  expect(splitLayers({ model: "m" }, flow, false, contractNullable)).toBeNull();
});

test("splitLayers drops shielded flow values", () => {
  const flow = new ValueTemplate({ model: "${pick.output}", provider: "openai" });

  const carried = splitLayers({ model: "m" }, flow, true, contractNullable);

  expect(carried).toBeNull();
  const kept = splitLayers({}, flow, true, contractNullable);
  expect(carriedTree(kept)).toEqual({ node: {}, flow: flow.value });
});

test("splitLayers leaves out a flow layer the node wholly overrides", () => {
  const node = new ValueTemplate({ model: typed("${pick.output}", STR) });
  const flow = new ValueTemplate({ model: "${pick2.output}" });

  const carried = splitLayers(node, flow, true, contractNullable);

  expect(carriedTree(carried)).toEqual({ node: node.value });
  expect(carriedTree(splitLayers(node, {}, true, contractNullable))).toEqual({ node: node.value });
});

test("splitLayers carries a whole record", () => {
  const node = new Binding("${input.cfg}", { origin: new OriginRef("node/cfg") });

  const carried = splitLayers(node, { provider: "openai" }, true, contractNullable);

  expect(carriedTree(carried)).toEqual({ node, flow: { provider: "openai" } });
  expect(carriedTree(carried).node).toBe(node);
});

// ---------- resolveLayers ----------

test("resolveLayers against a pool", () => {
  const node = new ValueTemplate({ model: "${pick.output}", anthropic: { top_k: "${input.k}" } });
  const flow = new ValueTemplate({ provider: "anthropic", temperature: "${input.t}" });
  const carried = splitLayers(node, flow, true, contractNullable)!;

  const resolved = resolveLayers(carried, pool({ k: 5n, t: null }, { pick: "claude" }), WHERE);

  expect(resolved).toEqual([
    { model: "claude", anthropic: { top_k: 5n } },
    { provider: "anthropic", temperature: null },
  ]);
});

test("resolveLayers throws on a nested tag mismatch", () => {
  const node = new ValueTemplate({ model: typed("${pick.output}", STR) });
  const carried = splitLayers(node, {}, true, contractNullable)!;

  const error = bindingError(() => resolveLayers(carried, pool({}, { pick: 3n }), WHERE));

  expect(error.failureCode).toBe("binding.type");
  expect(error.message.startsWith("node 'chat' llm_config.model: ")).toBe(true);
});

test("resolveLayers names a flow field under the flow", () => {
  const flow = new ValueTemplate({ model: typed("${pick.output}", STR) });
  const carried = splitLayers({}, flow, true, contractNullable)!;

  const error = bindingError(() => resolveLayers(carried, pool({}, { pick: 3n }), WHERE));

  expect(error.failureCode).toBe("binding.type");
  expect(error.message.startsWith("flow llm_config.model: ")).toBe(true);
});

test("a null whole record resolves to an empty layer", () => {
  const carried = splitLayers(new Binding("${input.cfg}"), { model: "m" }, true, contractNullable)!;

  expect(resolveLayers(carried, pool({ cfg: null }), WHERE)).toEqual([{}, { model: "m" }]);
  expect(resolveLayers(carried, pool({ cfg: { model: "x" } }), WHERE)).toEqual([{ model: "x" }, { model: "m" }]);
});

test("a whole record that is not a mapping fails", () => {
  const origin = new OriginRef("node/cfg");
  const carried = splitLayers(new Binding("${input.cfg}", { origin }), {}, true, contractNullable)!;

  const error = bindingError(() => resolveLayers(carried, pool({ cfg: "gpt-4" }), WHERE));

  expect(error.failureCode).toBe("contract.input");
  expect(error.message).toBe("node 'chat' llm_config: llm_config must resolve to a mapping or null, got str");
  expect(error.origin).toEqual(origin);
});

// ---------- typesNullable ----------

test("typesNullable answers from the stored types", () => {
  const tagged = typed("${input.m}", STR);
  const layer = new ValueTemplate({ model: "${pick.output}", provider: tagged });
  const model = (layer.value as Record<string, unknown>).model;
  const proven = typesNullable(layer, [[["model"], Type.scalar(ValueKind.STRING)]]);
  expect(proven(model, [])).toBe(false);
  // A leaf without a stored type answers from its tag.
  expect(proven(tagged, [])).toBe(false);
  const unproven = typesNullable(layer, [[["model"], null]]);
  expect(unproven(model, [])).toBe(true);
});

// ---------- refusedResolvedField ----------

const HOST_ONLY = "credentials and where they are sent stay with the host or yian auth";
const SUBJECT = "node 'c/a' llm_config";

/** Return why `resolved`, written as `written`, sets a refused field under `SUBJECT`. */
function refused(written: unknown, resolved: Record<string, unknown>): string | null {
  return refusedResolvedField(written, resolved, SUBJECT);
}

test("a call-time whole record may not set a host-only field", () => {
  const carried = splitLayers(new Binding("${pick.output}"), {}, true, contractNullable)!;
  const record = { provider: "azure", model: "dep", base_url: "https://elsewhere.example" };
  const [nodeLayer] = resolveLayers(carried, pool({}, { pick: record }), WHERE);

  expect(refused(carriedTree(carried).node, nodeLayer)).toBe(`${SUBJECT}.base_url: ${HOST_ONLY}`);
});

test("a call-time flow whole record may not set a host-only field", () => {
  const carried = splitLayers({ model: "m" }, new Binding("${pick.output}"), true, contractNullable)!;
  const [, flowLayer] = resolveLayers(carried, pool({}, { pick: { api_key: "sk-x" } }), WHERE);

  const message = refusedResolvedField(carriedTree(carried).flow, flowLayer, "flow llm_config");

  expect(message).toBe(`flow llm_config.api_key: ${HOST_ONLY}`);
});

test("a host-only field value is refused even beside a literal azure provider", () => {
  const layer = new ValueTemplate({ provider: "azure", base_url: "${input.url}" });
  const resolved = { provider: "azure", base_url: "https://elsewhere.example" };

  expect(refused(layer, resolved)).toBe(`${SUBJECT}.base_url: ${HOST_ONLY}`);
});

test("a literal base_url passes only beside a literal azure provider", () => {
  const url = "https://res.openai.azure.com";
  const resolved = { provider: "azure", base_url: url };
  const literal = new ValueTemplate({ provider: "azure", base_url: url });
  const fromValue = new ValueTemplate({ provider: "${input.p}", base_url: url });
  const tagged = new ValueTemplate({ provider: new Binding("azure"), base_url: url });

  expect(refused(literal, resolved)).toBeNull();
  expect(refused(tagged, resolved)).toBeNull();
  expect(refused(fromValue, resolved)!.startsWith(`${SUBJECT}: llm_config.base_url `)).toBe(true);
});

test("a layer resolved from values may not set a recorded connection", () => {
  const resolved = { model: "m", _connection: { method: "api_key" } };

  expect(refused(new Binding("${input.cfg}"), resolved)).toBe(`${SUBJECT}._connection: internal to recorded runs`);
});

test("a layer without refused fields passes", () => {
  const layer = new ValueTemplate({ provider: "${input.p}", model: "m" });

  expect(refused(layer, { provider: "openai", model: "m" })).toBeNull();
  expect(refused(new Binding("${input.cfg}"), {})).toBeNull();
});

// ---------- keepValues ----------

test("a dropped field value removes its whole merge field and origins", () => {
  const layer = new ValueTemplate(
    { model: "${pick.output}", provider: "${input.p}", stop: ["x", "${pick.stop}"], temperature: 0.2 },
    [
      [["model"], new OriginRef("site/model")],
      [["provider"], new OriginRef("site/provider")],
      [["stop", 1], new OriginRef("site/stop")],
    ],
  );

  const kept = keepValues(layer, (leaf) => leaf === "${input.p}") as ValueTemplate;

  expect(kept.value).toEqual({ provider: "${input.p}", temperature: 0.2 });
  expect(kept.origins).toEqual([[["provider"], new OriginRef("site/provider")]]);
});

test("a dropped block-field value removes the block field only", () => {
  const layer = new ValueTemplate(
    { anthropic: { max_tokens: 10, thinking: { type: "enabled", budget_tokens: "${input.b}" } } },
    [[["anthropic", "thinking", "budget_tokens"], new OriginRef("site/budget")]],
  );

  const kept = keepValues(layer, () => false) as ValueTemplate;

  expect(kept.value).toEqual({ anthropic: { max_tokens: 10 } });
  expect(kept.origins).toEqual([]);
});

test("a dropped whole record gives an empty layer", () => {
  expect(keepValues(new Binding("${input.cfg}"), () => false)).toEqual({});
});

test("every value kept returns the layer itself", () => {
  const layer = new ValueTemplate({ model: "${pick.output}", provider: "openai" });

  expect(keepValues(layer, () => true)).toBe(layer);
});

// ---------- resolveLayer ----------

test("resolveLayer reads literal text back and checks typed values", () => {
  const layer = new ValueTemplate({ provider: "a$${b}", model: typed("${pick.output}", STR) });

  expect(resolveLayer({ provider: "a${b}$" }, pool())).toEqual({ provider: "a${b}$" });
  expect(resolveLayer(layer, pool({}, { pick: "m" }))).toEqual({ provider: "a${b}", model: "m" });
  expect(resolveLayer(null, pool())).toEqual({});
  expect(resolveLayer(new Binding("${input.cfg}"), pool({ cfg: null }))).toEqual({});
  const error = bindingError(() => resolveLayer(layer, pool({}, { pick: 3n })));
  expect(error.failureCode).toBe("binding.type");
});

test("resolveLayer names a field with where", () => {
  const layer = new ValueTemplate({ model: typed("${pick.output}", STR) });
  const where = (field: ValuePath): string => "flow llm_config" + field.map((part) => `.${part}`).join("");

  const error = bindingError(() => resolveLayer(layer, pool({}, { pick: 3n }), { where }));

  expect(error.message.startsWith("flow llm_config.model: ")).toBe(true);
  const record = bindingError(() => resolveLayer(new Binding("${input.cfg}"), pool({ cfg: 3n }), { where }));
  expect(record.message).toBe("flow llm_config: llm_config must resolve to a mapping or null, got int");
  const unnamed = bindingError(() => resolveLayer(new Binding("${input.cfg}"), pool({ cfg: 3n })));
  expect(unnamed.message).toBe("binding '__llm_config': llm_config must resolve to a mapping or null, got int");
});

// ---------- checkResolvedLayer ----------

const WHOLE = new Binding("${input.cfg}");

test("checkResolvedLayer refuses a host-only field at that field", () => {
  const error = layerFieldError(() =>
    checkResolvedLayer({ provider: "openai", api_key: "sk-x" }, SUBJECT, { written: WHOLE }),
  );

  expect(error.message).toBe(`${SUBJECT}.api_key: ${HOST_ONLY}`);
  expect(error.field).toEqual(["api_key"]);
  expect(error).toBeInstanceOf(RangeError);
});

test("checkResolvedLayer refuses an unknown key for the layer as a whole", () => {
  const error = layerFieldError(() => checkResolvedLayer({ modle: "m" }, SUBJECT, { written: WHOLE }));

  expect(error.message.startsWith(`${SUBJECT}: llm_config has unknown field(s) `)).toBe(true);
  expect(error.field).toEqual([]);
});

test("checkResolvedLayer names a bad block field by its key path", () => {
  const layer = { provider: "anthropic", model: "m", anthropic: { thinking: { type: "enabled", budget_tokens: 10n } } };

  const error = layerFieldError(() => checkResolvedLayer(layer, SUBJECT, { written: WHOLE }));

  expect(error.message.startsWith(`${SUBJECT}.anthropic.thinking.budget_tokens: `)).toBe(true);
  expect(error.message).not.toContain("`anthropic.thinking.budget_tokens`");
  expect(error.field).toEqual(["anthropic", "thinking", "budget_tokens"]);
});

test("checkResolvedLayer refuses an unknown provider", () => {
  const error = layerFieldError(() => checkResolvedLayer({ provider: "nope", model: "m" }, SUBJECT, { written: WHOLE }));

  expect(error.message).toBe(
    `${SUBJECT}.provider: not a known provider; use one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`,
  );
  expect(error.field).toEqual(["provider"]);
});

test("checkResolvedLayer puts the callsite after the field", () => {
  const error = layerFieldError(() =>
    checkResolvedLayer({ provider: "nope" }, "adaptive_questions llm_config", {
      written: WHOLE,
      callsite: " in 'outer'",
    }),
  );

  expect(error.message.startsWith("adaptive_questions llm_config.provider in 'outer': not a known provider")).toBe(true);
  expect(() => checkResolvedLayer({ provider: "openai", model: "m" }, SUBJECT, { written: WHOLE })).not.toThrow();
});

// ---------- agentLabel / fieldWhere ----------

test("agentLabel adds the callsite when nested", () => {
  expect(agentLabel(null, "outer")).toBeNull();
  expect(agentLabel("adaptive_questions llm_config", "")).toBe("adaptive_questions llm_config");
  expect(agentLabel("adaptive_questions llm_config", "outer/sub")).toBe("adaptive_questions llm_config in 'outer/sub'");
});

test("fieldWhere names a field under its subject", () => {
  expect(fieldWhere("node 'a' llm_config", ["openai", "temperature"])).toBe("node 'a' llm_config.openai.temperature");
  expect(fieldWhere("node 'a' llm_config", [])).toBe("node 'a' llm_config");
});

// ---------- invocationConfig ----------

test("invocationConfig joins the host's secrets only with the marker", () => {
  const credentials = new HostConnection({ apiKey: "sk-host", headers: { "X-Gate": "gw-secret" } });
  const marked = { provider: "anthropic", model: "m", [HOST_KEY_MARKER]: true, anthropic: { max_tokens: 10n } };
  const plain = { provider: "anthropic", model: "m" };

  const config = invocationConfig(marked, credentials);

  expect(config).toEqual({ ...marked, api_key: "sk-host", headers: { "X-Gate": "gw-secret" } });
  expect(marked).not.toHaveProperty("api_key");
  expect(config["anthropic"]).not.toBe(marked.anthropic);
  expect(invocationConfig(plain, credentials)).toEqual(plain);
  expect(invocationConfig(marked, null)).toEqual(marked);
  expect(invocationConfig(marked, new HostConnection())).toEqual(marked);
});

// ---------- callTimeConfig ----------

describe("callTimeConfig", () => {
  let home: string;
  let vars: Map<string, string>;
  const env: Env = (name) => vars.get(name);

  beforeEach(() => {
    home = mkdtempSync(join(tmpdir(), "yian-llm-values-"));
    vars = new Map([
      ["YIAN_DEFAULT_PROVIDER", "openai"],
      ["YIAN_DEFAULT_MODEL", "gpt-default"],
      ["OPENAI_API_KEY", "sk-env-secret"],
    ]);
  });

  afterEach(() => {
    rmSync(home, { recursive: true, force: true });
  });

  /** Return the test's credential store. */
  function store(): AuthStore {
    return new AuthStore(join(home, "auth.json"), { env });
  }

  /** Return what run start fixes for the agent `chat`, with the host key for `hostKeyProvider`. */
  function callTime(base: Record<string, unknown> = {}, hostKeyProvider: string | null = null): CallTimeLLM {
    const snapshot = connectionSnapshot({ env, store: store() });
    const selection = new ConnectionSelection({ snapshot, hostKeyProvider, hostBaseUrl: null, hostSettings: {} });
    return new CallTimeLLM({ path: "chat", base, selection });
  }

  /** Resolve `chat`'s config from its node and flow layers against `pick`'s output. */
  function config(
    node: unknown,
    flow: unknown,
    pick: unknown,
    options: { at?: CallTimeLLM; credentials?: HostConnection | null } = {},
  ): Record<string, unknown> {
    const template = splitLayers(node, flow, true, contractNullable)!;
    return callTimeConfig(template, pool({}, { pick }), options.at ?? callTime(), {
      label: null,
      credentials: options.credentials ?? null,
    });
  }

  test("a resolved config is canonical, with the run's connection", () => {
    const canonical = config(new ValueTemplate({ model: "${pick.output}" }), {}, "gpt-x", {
      at: callTime({ provider: "openai" }),
    });

    expect(canonical["provider"]).toBe("openai");
    expect(canonical["model"]).toBe("gpt-x");
    expect(canonical["_connection"]).toMatchObject({ method: "api_key", source: "environment" });
    expect(JSON.stringify(canonical, (_key, value) => (typeof value === "bigint" ? String(value) : value))).not.toContain(
      "sk-env-secret",
    );
  });

  test("a block the resolved connection refuses fails at its field", async () => {
    const credentials = new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: Date.now() / 1000 + 3600,
      accountId: "account",
    });
    await store().save("openai", credentials, "gpt-selected");
    const origin = new OriginRef("node/openai");
    const node = new ValueTemplate({ provider: "${pick.output}", openai: { max_output_tokens: 100n } }, [
      [["openai"], origin],
    ]);

    const error = bindingError(() => config(node, {}, "openai"));

    expect(error.failureCode).toBe("contract.input");
    expect(error.message.startsWith("node 'chat' llm_config.openai: ")).toBe(true);
    expect(error.message).toContain("`openai.max_output_tokens`");
    expect(error.message).toContain("ChatGPT subscription");
    expect(error.origin).toEqual(origin);
  });

  test("a block field from a value is checked", () => {
    const origin = new OriginRef("node/thinking");
    const node = new ValueTemplate(
      { provider: "anthropic", model: "claude-x", anthropic: { thinking: "${pick.output}" } },
      [[["anthropic", "thinking"], origin]],
    );

    const error = bindingError(() => config(node, {}, { type: "enabled", budget_tokens: 10n }));

    expect(error.failureCode).toBe("contract.input");
    expect(error.message.startsWith("node 'chat' llm_config.anthropic.thinking.budget_tokens: ")).toBe(true);
    expect(error.message).not.toContain("`anthropic.thinking.budget_tokens`");
    expect(error.message).toContain("1024");
    expect(error.origin).toEqual(origin);
  });

  test("a value that breaks its tag fails with binding.type", () => {
    const tag = new OriginRef("node/model-tag");
    const node = new ValueTemplate({ provider: "openai", model: typed("${pick.output}", STR, tag) });

    const error = bindingError(() => config(node, {}, null));

    expect(error.failureCode).toBe("binding.type");
    expect(error.message.startsWith("node 'chat' llm_config.model: ")).toBe(true);
    expect(error.origin).toEqual(tag);
  });

  test("a flow value that breaks its tag names the flow field", () => {
    const tag = new OriginRef("flow/model-tag");
    const flow = new ValueTemplate({ model: typed("${pick.output}", STR, tag) });

    const error = bindingError(() => config({ provider: "openai" }, flow, null));

    expect(error.failureCode).toBe("binding.type");
    expect(error.message.startsWith("flow llm_config.model: ")).toBe(true);
    expect(error.origin).toEqual(tag);
  });

  test("a non-finite value names the block field", () => {
    const origin = new OriginRef("node/temperature");
    const node = new ValueTemplate(
      {
        provider: "openai",
        model: "m",
        openai: { temperature: typed("${pick.output * 10}", Type.scalar(ValueKind.NUMBER)) },
      },
      [[["openai", "temperature"], origin]],
    );

    const error = bindingError(() => config(node, {}, 1e308));

    // TS's binder takes Infinity as a value (`rejectInvalidBinding`), so the block check refuses
    // it; Python's binder fails `binding.evaluate` with "non-finite number inf is not a yian value".
    expect(error.failureCode).toBe("contract.input");
    expect(error.message).toBe("node 'chat' llm_config.openai.temperature: Input should be a finite number");
    expect(error.origin).toEqual(origin);
  });

  test("a block value out of bounds names the field once", () => {
    const node = new ValueTemplate({ provider: "openai", model: "m", openai: { temperature: "${pick.output}" } });

    const error = bindingError(() => config(node, {}, 5.0));

    expect(error.failureCode).toBe("contract.input");
    expect(error.message).toBe("node 'chat' llm_config.openai.temperature: Input should be less than or equal to 2");
  });

  test("a provider resolving to an unknown one names the field", () => {
    const origin = new OriginRef("node/provider");
    const node = new ValueTemplate({ provider: "${pick.output}", model: "m" }, [[["provider"], origin]]);

    const error = bindingError(() => config(node, {}, "nope"));

    expect(error.failureCode).toBe("contract.input");
    expect(error.message.startsWith("node 'chat' llm_config.provider: not a known provider")).toBe(true);
    expect(error.origin).toEqual(origin);
  });

  test("a host-only field from a whole record value is refused", () => {
    const origin = new OriginRef("node/cfg");
    const record = { provider: "azure", model: "dep", base_url: "https://res.openai.azure.com" };

    const error = bindingError(() => config(new Binding("${pick.output}", { origin }), {}, record));

    expect(error.failureCode).toBe("contract.input");
    expect(error.message.startsWith("node 'chat' llm_config.base_url: ")).toBe(true);
    expect(error.message).toContain("stay with the host or yian auth");
    expect(error.message).not.toContain("res.openai.azure.com");
    expect(error.origin).toEqual(origin);
  });

  test("a config marked for the host key needs the key, and returns without it", () => {
    const node = new ValueTemplate({ provider: "${pick.output}", model: "claude-x" });
    const at = callTime({}, "anthropic");

    expect(thrown(() => config(node, {}, "anthropic", { at }))).toBeInstanceOf(AuthError);
    const canonical = config(node, {}, "anthropic", { at, credentials: new HostConnection({ apiKey: "sk-host" }) });
    expect(canonical).toEqual({ provider: "anthropic", model: "claude-x", [HOST_KEY_MARKER]: true });
  });
});
