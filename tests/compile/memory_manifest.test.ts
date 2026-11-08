/**
 * Memory types in flow manifests: a memory type projects as its name and item record only, so a
 * recorded flow's fingerprint doesn't depend on the instance bound to it.
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { Edge, Flow } from "../../src/compile/model.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { readTypedefs, typeFor, withMemoryTypes } from "../../src/typesys/types.ts";
import { memoryType, type Type } from "../../src/typesys/values.ts";

/** Build a START -> CODE -> END fragment whose START takes the input `lessons` typed `typ`. */
function fragment(typ: Type): Flow {
  const start = new StartNode("entry", { params: [new Parameter("lessons", { contract: new ValueContract(typ), required: true })] });
  const body = new CodeNode("body", { code: 'return inputs["x"]' });
  body.params = [new Parameter("x")];
  const end = EndNode.value("terminal", { outputName: "result" });
  return new Flow(
    new Map<string, Node>([
      ["entry", start],
      ["body", body],
      ["terminal", end],
    ]),
    [
      new Edge("entry->body", "entry", "body", { ordering: true }),
      new Edge("body->terminal", "body", "terminal", { inputGroup: "result" }),
    ],
    {
      wiring: new Map([
        ["entry", new Map()],
        ["body", new Map([["x", new Binding("${entry.output}")]])],
        ["terminal", new Map([["result", new Binding("${body.output}")]])],
      ]),
      startId: "entry",
      endId: "terminal",
    },
  );
}

/** Return the manifest of the type `typ` as START's input contract holds it. */
function manifestOf(typ: Type): JsonObject {
  const start = (flowManifest(fragment(typ)).nodes as JsonObject).entry as JsonObject;
  const param = (start.params as JsonObject[])[0]!;
  return (param.contract as JsonObject).type as JsonObject;
}

const CASE = readTypedefs({ Case: { title: "str" } });

test("a memory type projects as its name, its item record and whether it is optional", () => {
  expect(manifestOf(typeFor("FileMemory", CASE))).toEqual({ item: null, kind: "memory", nullable: false, type: "FileMemory" });
  expect(manifestOf(typeFor("Optional(FileMemory)", CASE))).toEqual({
    item: null,
    kind: "memory",
    nullable: true,
    type: "FileMemory",
  });
  const cases = manifestOf(typeFor("VectorDBMemory(Case)", CASE));
  expect([cases.kind, cases.type, (cases.item as JsonObject).kind]).toEqual(["memory", "VectorDBMemory", "object"]);
});

test("a host memory type projects the same however its name was registered", () => {
  const registered = typeFor("TradeJournal", withMemoryTypes(new Map(), ["TradeJournal"]));
  expect(manifestOf(registered)).toEqual(manifestOf(memoryType("TradeJournal", null)));
  expect(flowFingerprint(fragment(registered))).toBe(flowFingerprint(fragment(memoryType("TradeJournal", null))));
});

test("different memory types fingerprint apart", () => {
  const print = (typ: Type) => flowFingerprint(fragment(typ));
  expect(print(memoryType("FileMemory", null))).not.toBe(print(memoryType("TradeJournal", null)));
  expect(print(typeFor("FileMemory", CASE))).not.toBe(print(typeFor("Optional(FileMemory)", CASE)));
  expect(print(typeFor("VectorDBMemory", CASE))).not.toBe(print(typeFor("VectorDBMemory(Case)", CASE)));
});
