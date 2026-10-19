/**
 * The kind-blind binding-field walk: a port of `tests/compose/test_bindings.py`. No concrete kind
 * is ported yet, so each row uses a stand-in of the kind Python uses (`standIn` for the kinds
 * whose only binding field is `inputs`, test-local `StandInCall` and `StandInMap` for the kinds
 * with a scalar `over`) and passes an explicit `bindingFields` table built from their
 * `bindingFields`.
 *
 * TS-only: `mapBindingSources` hands the mapper a `ValueTemplate` whole, in a record field and in
 * a scalar field, and `keyless` drops the field and key.
 */

import { describe, expect, test } from "bun:test";

import { NodeDescriptor, type NodeDescriptorFields } from "../../src/compose/nodes/base.ts";
import { keyless, mapBindingSources, mapBindingStrings } from "../../src/compose/nodes/bindings.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ValueTemplate } from "../../src/nodes/contracts.ts";
import { StandInCase, standIn } from "./stand_ins.ts";

/** A `call` stand-in with the scalar `over` binding field beside `inputs`. */
class StandInCall extends NodeDescriptor {
  static override kind = "call";
  static override bindingFields: readonly string[] = ["inputs", "over"];
  readonly over: unknown;

  constructor(fields: NodeDescriptorFields & { over?: unknown }) {
    super(fields);
    this.over = fields.over ?? null;
  }
}

/** A `map` stand-in: Python's `MapDescriptor` subclasses `CallDescriptor`. */
class StandInMap extends StandInCall {
  static override kind = "map";
}

const StandInAgent = standIn("agent", { bindingFields: ["inputs"] });
const StandInTool = standIn("tool", { bindingFields: ["inputs"] });
const StandInHumanInput = standIn("human_input", { bindingFields: ["inputs"] });
const StandInLoop = standIn("loop", { bindingFields: ["inputs"] });
const StandInWait = standIn("wait");

const CLASSES: readonly DescriptorClass[] = [
  StandInAgent,
  StandInTool,
  StandInHumanInput,
  StandInLoop,
  StandInWait,
  StandInCase,
  StandInCall,
  StandInMap,
];
const bindingFields: ReadonlyMap<string, readonly string[]> = new Map(
  CLASSES.map((cls) => [cls.kind, cls.bindingFields]),
);

function up(_field: string, _key: string, value: string): string {
  return value.toUpperCase();
}

describe("mapBindingStrings", () => {
  test("agent inputs rewritten, non-strings left", () => {
    const desc = new StandInAgent({ id: "a", inputs: { x: "${src}", n: 5 } });
    const out = mapBindingStrings(desc, up, { bindingFields });
    expect(out.inputs).toEqual({ x: "${SRC}", n: 5 });
    // A fresh descriptor (not the same object): a rewrite happened.
    expect(out).not.toBe(desc);
  });

  test("tool inputs rewritten", () => {
    const desc = new StandInTool({ id: "t", inputs: { url: "${u}", n: 3 } });
    const out = mapBindingStrings(desc, up, { bindingFields });
    expect(out.inputs).toEqual({ url: "${U}", n: 3 });
  });

  test("call inputs and over rewritten", () => {
    const desc = new StandInCall({ id: "c", inputs: { seed: "${s}" }, over: "${xs}" });
    const out = mapBindingStrings(desc, up, { bindingFields });
    expect(out.inputs).toEqual({ seed: "${S}" });
    expect(out.over).toBe("${XS}");
    expect(out.kind).toBe("call");
  });

  test("map kind inputs and over rewritten", () => {
    const desc = new StandInMap({ id: "m", inputs: { seed: "${s}" }, over: "${xs}" });
    const out = mapBindingStrings(desc, up, { bindingFields });
    expect(out.inputs).toEqual({ seed: "${S}" });
    expect(out.over).toBe("${XS}");
    expect(out).toBeInstanceOf(StandInMap);
  });

  test("the mapper distinguishes an input key from the scalar field", () => {
    const desc = new StandInMap({ id: "m", inputs: { over: "${input_value}" }, over: "${items}" });
    const sites: [string, string, string][] = [];
    mapBindingStrings(
      desc,
      (field, key, value) => {
        sites.push([field, key, value]);
        return value;
      },
      { bindingFields },
    );
    expect(sites).toEqual([
      ["inputs", "over", "${input_value}"],
      ["over", "over", "${items}"],
    ]);
  });

  test("call over null left untouched", () => {
    const desc = new StandInCall({ id: "c", inputs: {}, over: null });
    const out = mapBindingStrings(desc, up, { bindingFields });
    expect(out.over).toBeNull();
  });

  test("no binding fields pass through unchanged", () => {
    // CASE and WAIT have no authored input-binding mapping -> identity (same object).
    for (const desc of [new StandInCase({ id: "k", on: "${sel}", cases: [] }), new StandInWait({ id: "w" })]) {
      expect(mapBindingStrings(desc, up, { bindingFields })).toBe(desc);
    }
  });

  test("human_input and loop inputs rewritten", () => {
    for (const desc of [
      new StandInHumanInput({ id: "h", inputs: { q: "${qs}" } }),
      new StandInLoop({ id: "l", inputs: { q: "${qs}" } }),
    ]) {
      const out = mapBindingStrings(desc, up, { bindingFields });
      expect(out.inputs).toEqual({ q: "${QS}" });
      expect(out).not.toBe(desc);
    }
  });

  test("template leaf strings rewritten with their binding key", () => {
    const origin = new OriginRef("/s:leaf");
    const template = new ValueTemplate({ a: "${src}", b: [1n, "${other}"] }, [[["a"], origin]]);
    const desc = new StandInAgent({ id: "a", inputs: { x: template } });
    const sites: [string, string, string][] = [];
    const out = mapBindingStrings(
      desc,
      (field, key, value) => {
        sites.push([field, key, value]);
        return value.toUpperCase();
      },
      { bindingFields },
    );
    expect(out.inputs["x"]).toEqual(new ValueTemplate({ a: "${SRC}", b: [1n, "${OTHER}"] }, [[["a"], origin]]));
    expect(sites).toEqual([
      ["inputs", "x", "${src}"],
      ["inputs", "x", "${other}"],
    ]);
  });
});

describe("mapBindingSources", () => {
  test("a ValueTemplate is passed whole, in a record field and a scalar field", () => {
    const inner = new ValueTemplate({ a: "${src}" });
    const items = new ValueTemplate(["${xs}"]);
    const desc = new StandInCall({ id: "c", inputs: { x: inner, n: 5 }, over: items });
    const sites: [string, string, unknown][] = [];
    const out = mapBindingSources(
      desc,
      (field, key, source) => {
        sites.push([field, key, source]);
        return "${replaced}";
      },
      { bindingFields },
    );
    expect(sites).toHaveLength(2);
    expect(sites[0]?.[2]).toBe(inner);
    expect(sites[1]?.[2]).toBe(items);
    expect(sites.map(([field, key]) => [field, key])).toEqual([
      ["inputs", "x"],
      ["over", "over"],
    ]);
    expect(out.inputs).toEqual({ x: "${replaced}", n: 5 });
    expect(out.over).toBe("${replaced}");
  });
});

describe("keyless", () => {
  test("adapts a value-only mapper, ignoring the field and key", () => {
    const desc = new StandInAgent({ id: "a", inputs: { x: "${src}" } });
    const out = mapBindingStrings(
      desc,
      keyless((value) => value.toUpperCase()),
      { bindingFields },
    );
    expect(out.inputs).toEqual({ x: "${SRC}" });
    expect(keyless((value) => `<${value}>`)("inputs", "x", "v")).toBe("<v>");
  });
});
