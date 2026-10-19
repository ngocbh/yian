/** Mirrors `tests/test_llm_values.py`. */

import { expect, test } from "bun:test";

import { OriginRef } from "../src/diagnostics.ts";
import {
  contractNullable,
  contributing,
  keepValues,
  layerRefs,
  readsOnlyInputs,
  splitLayers,
  typeNullable,
  typesNullable,
  valueLeaves,
} from "../src/nodes/agent/llm_values.ts";
import { Binding, ContractOrigins, type ValuePath, ValueContract, ValueTemplate } from "../src/nodes/contracts.ts";
import { Type, ValueKind } from "../src/typesys/index.ts";

const STR = Type.scalar(ValueKind.STRING);
const OPT_STR = new Type({ kind: ValueKind.STRING, nullable: true });

/** Return a typed binding of `source` declared as `typ`. */
function typed(source: string, typ: Type): Binding {
  return new Binding(source, { contract: new ValueContract(typ, new ContractOrigins()) });
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
  return (carried!.source as ValueTemplate).value as Record<string, unknown>;
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
  const strict = new Binding("${input.cfg}", {
    contract: new ValueContract(record({ model: STR }), new ContractOrigins()),
  });
  const looseField = new Binding("${input.cfg}", {
    contract: new ValueContract(record({ model: OPT_STR }), new ContractOrigins()),
  });
  const looseRecord = new Binding("${input.cfg}", {
    contract: new ValueContract(record({ model: STR }, true), new ContractOrigins()),
  });
  const optionalField = new Binding("${input.cfg}", {
    contract: new ValueContract(
      new Type({ kind: ValueKind.OBJECT, fields: new Map([["model", STR]]), required: new Set() }),
      new ContractOrigins(),
    ),
  });

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
  // Added with resolveLayers: the carried layers resolve back to {model: "m"} and {provider: "a${b}$"}.
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

// ---------- typesNullable ----------

test("typesNullable answers from the stored types", () => {
  const tagged = new Binding("${input.m}", { contract: new ValueContract(Type.scalar(ValueKind.STRING)) });
  const layer = new ValueTemplate({ model: "${pick.output}", provider: tagged });
  const model = (layer.value as Record<string, unknown>).model;
  const proven = typesNullable(layer, [[["model"], Type.scalar(ValueKind.STRING)]]);
  expect(proven(model, [])).toBe(false);
  // A leaf without a stored type answers from its tag.
  expect(proven(tagged, [])).toBe(false);
  const unproven = typesNullable(layer, [[["model"], null]]);
  expect(unproven(model, [])).toBe(true);
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
