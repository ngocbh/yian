/** The kind-blind binding-field walk. */

import { describe, expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { mapBindingSources, mapBindingStrings } from "../../src/compose/nodes/bindings.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ValueTemplate } from "../../src/nodes/contracts.ts";

function up(_field: string, _key: string, value: string): string {
  return value.toUpperCase();
}

describe("mapBindingStrings", () => {
  test("agent inputs rewritten, non-strings left", () => {
    const desc = new AgentDescriptor({ id: "a", inputs: { x: "${src}", n: 5 } });
    const out = mapBindingStrings(desc, up);
    expect(out.inputs).toEqual({ x: "${SRC}", n: 5 });
    // A fresh descriptor (not the same object): a rewrite happened.
    expect(out).not.toBe(desc);
  });

  test("tool inputs rewritten", () => {
    const desc = new ToolDescriptor({ id: "t", toolRef: { group: null, name: "fetch" }, inputs: { url: "${u}", n: 3 } });
    const out = mapBindingStrings(desc, up);
    expect(out.inputs).toEqual({ url: "${U}", n: 3 });
  });

  test("call inputs and over rewritten", () => {
    const desc = new CallDescriptor({ id: "c", call: "sub", inputs: { seed: "${s}" }, over: "${xs}" });
    const out = mapBindingStrings(desc, up);
    expect(out.inputs).toEqual({ seed: "${S}" });
    expect(out.over).toBe("${XS}");
    expect(out.kind).toBe("call");
  });

  test("map kind inputs and over rewritten", () => {
    const desc = new MapDescriptor({ id: "m", call: "sub", inputs: { seed: "${s}" }, over: "${xs}" });
    const out = mapBindingStrings(desc, up);
    expect(out.inputs).toEqual({ seed: "${S}" });
    expect(out.over).toBe("${XS}");
    expect(out).toBeInstanceOf(MapDescriptor);
    expect(out.kind).toBe("map");
  });

  test("the mapper distinguishes an input key from the scalar field", () => {
    const desc = new MapDescriptor({ id: "m", call: "sub", inputs: { over: "${input_value}" }, over: "${items}" });
    const sites: [string, string, string][] = [];
    mapBindingStrings(
      desc,
      (field, key, value) => {
        sites.push([field, key, value]);
        return value;
      },
    );
    expect(sites).toEqual([
      ["inputs", "over", "${input_value}"],
      ["over", "over", "${items}"],
    ]);
  });

  test("call over null left untouched", () => {
    const desc = new CallDescriptor({ id: "c", call: "sub", inputs: {}, over: null });
    const out = mapBindingStrings(desc, up);
    expect(out.over).toBeNull();
  });

  test("no binding fields pass through unchanged", () => {
    // CASE has no authored input-binding mapping -> identity (same object).
    const desc = new CaseDescriptor({ id: "k", on: "${sel}", cases: [] });
    expect(mapBindingStrings(desc, up)).toBe(desc);
  });

  test("ask, sleep and loop inputs rewritten", () => {
    for (const [desc, inputs] of [
      [new ToolDescriptor({ id: "h", toolRef: { group: null, name: "ask" }, inputs: { questions: "${qs}" } }), { questions: "${QS}" }],
      [new ToolDescriptor({ id: "w", toolRef: { group: null, name: "sleep" }, inputs: { until: "${t}" } }), { until: "${T}" }],
      [new LoopDescriptor({ id: "l", call: "body", inputs: { q: "${qs}" } }), { q: "${QS}" }],
    ] as const) {
      const out = mapBindingStrings(desc, up);
      expect(out.inputs).toEqual(inputs);
      expect(out).not.toBe(desc);
    }
  });

  test("template leaf strings rewritten with their binding key", () => {
    const origin = new OriginRef("/s:leaf");
    const template = new ValueTemplate({ a: "${src}", b: [1n, "${other}"] }, [[["a"], origin]]);
    const desc = new AgentDescriptor({ id: "a", inputs: { x: template } });
    const sites: [string, string, string][] = [];
    const out = mapBindingStrings(
      desc,
      (field, key, value) => {
        sites.push([field, key, value]);
        return value.toUpperCase();
      },
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
    const desc = new CallDescriptor({ id: "c", call: "sub", inputs: { x: inner, n: 5 }, over: items });
    const sites: [string, string, unknown][] = [];
    const out = mapBindingSources(
      desc,
      (field, key, source) => {
        sites.push([field, key, source]);
        return "${replaced}";
      },
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
