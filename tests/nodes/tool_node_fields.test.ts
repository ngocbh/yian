import { describe, expect, test } from "bun:test";
import { Flow } from "../../src/compile/model.ts";
import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

/** A flow of the one TOOL node `node`. */
function singleNodeFlow(node: ToolNode): Flow {
  return new Flow(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map()]]),
    startId: node.id,
    endId: node.id,
  });
}

describe("ToolNode built-in fields", () => {
  test("builtin follows toolId", () => {
    expect(new ToolNode("t", { toolId: "take" }).builtin).toBe("take");
    expect(new ToolNode("s", { toolId: "sleep" }).builtin).toBe("sleep");
    expect(new ToolNode("a", { toolId: "ask" }).builtin).toBe("ask");
    // A plain call of the agent surface, until compose reads it as ask.
    expect(new ToolNode("u", { toolId: "ask_user" }).builtin).toBeNull();
    expect(new ToolNode("p", { toolId: "ping" }).builtin).toBeNull();
    const node = new ToolNode("p", { toolId: "ping" });
    node.toolId = "take";
    expect(node.builtin).toBe("take");
  });

  test("answerSchema defaults to []", () => {
    expect(new ToolNode("a", { toolId: "ask" }).answerSchema).toEqual([]);
    expect(new ToolNode("a", { toolId: "ask", answerSchema: null }).answerSchema).toEqual([]);
    const schema = [{ name: "answer", type: "str", required: true }];
    expect(new ToolNode("a", { toolId: "ask", answerSchema: schema }).answerSchema).toEqual(schema);
  });

  test("a TOOL manifest holds a non-empty answer schema, never builtin", () => {
    const plain = singleNodeFlow(new ToolNode("node", { toolId: "ask", retries: 0 }));
    const schema = [{ name: "answer", type: "str", required: true }];
    const withSchema = singleNodeFlow(new ToolNode("node", { toolId: "ask", retries: 0, answerSchema: schema }));
    const manifest = (flowManifest(withSchema).nodes as JsonObject)["node"] as JsonObject;
    expect(Object.keys(manifest)).not.toContain("builtin");
    expect(Object.keys(manifest)).toContain("answer_schema");
    expect(Object.keys((flowManifest(plain).nodes as JsonObject)["node"] as JsonObject)).not.toContain("answer_schema");
    expect(flowFingerprint(withSchema)).not.toBe(flowFingerprint(plain));
  });
});
