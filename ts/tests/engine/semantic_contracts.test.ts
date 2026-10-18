/**
 * The provenance-bearing semantic IR vocabulary of `nodes/contracts.ts`, a port of
 * `tests/engine/test_semantic_contracts.py`: the carriers' fields and freezing, the one-layer
 * `asBinding`/`bindingSource` helpers, the `ContractOrigins` index (canonical order, rejected
 * entries, nearest-ancestor lookup), `Parameter` default states, `ValueContract`'s fresh origin
 * index, `PartialRecordContract`'s rejected shapes, and origin mapping on every carrier.
 *
 * Changed: `fields()` order is the own-key order of an instance, with camelCase names; a frozen
 * dataclass is a frozen object, and `FrozenInstanceError` is the `TypeError` an assignment to a
 * frozen object throws in a module; dataclass `==` is `toEqual`; `ValueError` is `RangeError`.
 *
 * TS-only: `SeedConnection` (a name, an index and name, a negative index, an empty name, a
 * three-part path), `ChildFlowBoundary` (equal or empty ids) and a frozen `HostToolCall`, which
 * no Python unit test covers.
 *
 * Left for later: `test_partial_record_contract_enforces_closed_shape_and_known_fields` and
 * `test_output_contract_field_error_rejects_fields_on_fieldless_types` land with the output
 * contract functions; `test_nodes_package_exports_the_canonical_contract_vocabulary` waits for
 * `nodes/index.ts` (T3-3).
 *
 * Not ported: the rows that differ from a good one only by a list where Python wants a tuple
 * (`ContractOrigins(entries=[])`, a list path, `PartialRecordContract(fields=[])`): TS has only
 * arrays.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  Assertion,
  asBinding,
  Binding,
  bindingSource,
  ChildFlowBoundary,
  ContractOrigins,
  HostToolCall,
  Parameter,
  PartialRecordContract,
  SeedConnection,
  ValueContract,
  type ValuePath,
} from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";

/** Build a short valid origin for carrier tests. */
function origin(key: string): OriginRef {
  return new OriginRef(key);
}

/** Assign `value` to `field` of a frozen carrier, as a caller ignoring `readonly` would. */
function assign(carrier: object, field: string, value: unknown): void {
  (carrier as Record<string, unknown>)[field] = value;
}

test("carriers have exact final fields", () => {
  const integer = new ValueContract(Type.scalar(ValueKind.INTEGER));
  const carriers = {
    binding: new Binding("x"),
    assertion: new Assertion("x", "x"),
    origins: new ContractOrigins(),
    contract: integer,
    partial: new PartialRecordContract([["x", null]]),
    parameter: new Parameter("x"),
  };
  for (const carrier of Object.values(carriers)) expect(Object.isFrozen(carrier)).toBe(true);

  expect(Object.keys(carriers.binding)).toEqual(["source", "origin", "contract"]);
  expect(Object.keys(carriers.assertion)).toEqual(["expression", "display", "origin"]);
  expect(Object.keys(carriers.origins)).toEqual(["default", "entries"]);
  expect(Object.keys(carriers.contract)).toEqual(["type", "origins"]);
  expect(Object.keys(carriers.partial)).toEqual(["fields"]);
  expect(Object.keys(carriers.parameter)).toEqual([
    "name",
    "contract",
    "required",
    "hasDefault",
    "default",
    "coerceStrings",
    "origin",
    "defaultOrigin",
  ]);
});

test("carriers are field-frozen but preserve arbitrary payload identity", () => {
  const source = { values: [1n] };
  const fallback = ["fallback"];
  const binding = new Binding(source);
  const parameter = new Parameter("x", { hasDefault: true, default: fallback });

  expect(binding.source).toBe(source);
  expect(parameter.default).toBe(fallback);
  expect(() => assign(binding, "origin", origin("changed"))).toThrow(TypeError);
  expect(() => assign(parameter, "required", true)).toThrow(TypeError);
});

test("binding helpers normalize once and preserve existing identity", () => {
  const raw = { x: 1n };
  const wrapped = asBinding(raw);
  const inner = new Binding("value", { origin: origin("inner") });
  const outer = new Binding(inner, { origin: origin("outer") });

  expect(wrapped).toEqual(new Binding(raw, { origin: null }));
  expect(wrapped.source).toBe(raw);
  expect(asBinding(inner)).toBe(inner);
  expect(bindingSource(raw)).toBe(raw);
  expect(bindingSource(inner)).toBe("value");
  expect(bindingSource(outer)).toBe(inner);
});

test("contract origins use one canonical order", () => {
  const paths: ValuePath[] = [["a"], ["a", "field"], ["a", 0], ["b"], [0]];
  const origins = new ContractOrigins({
    entries: paths.map((path, index) => [path, origin(String(index))] as const),
  });

  expect(origins.entries.map(([path]) => path)).toEqual(paths);
});

test.each<[string, unknown, ErrorConstructor, RegExp]>([
  ["an empty path", [[[], origin("root")]], RangeError, /empty/],
  [
    "a duplicate path",
    [
      [["same"], origin("one")],
      [["same"], origin("two")],
    ],
    RangeError,
    /duplicate/,
  ],
  [
    "an unsorted path",
    [
      [["z"], origin("z")],
      [["a"], origin("a")],
    ],
    RangeError,
    /sorted/,
  ],
  ["a boolean segment", [[[true], origin("bool")]], TypeError, /segment/],
])("contract origins reject %s", (_name, entries, error, message) => {
  expect(() => new ContractOrigins({ entries: entries as never })).toThrow(error);
  expect(() => new ContractOrigins({ entries: entries as never })).toThrow(message);
});

test("contract origins require well-typed structure", () => {
  expect(() => new ContractOrigins({ default: "root" as never })).toThrow(TypeError);
  expect(() => new ContractOrigins({ default: "root" as never })).toThrow(/OriginRef/);
  const entries = [[["field"], "not-an-origin"]] as never;
  expect(() => new ContractOrigins({ entries })).toThrow(TypeError);
  expect(() => new ContractOrigins({ entries })).toThrow(/OriginRef/);
});

test("contract origins select exact, then ancestor, then default", () => {
  const fallback = origin("fallback");
  const payload = origin("payload");
  const items = origin("items");
  const first = origin("first");
  const origins = new ContractOrigins({
    default: fallback,
    entries: [
      [["payload"], payload],
      [["payload", "items"], items],
      [["payload", "items", 0], first],
    ],
  });

  expect(origins.at(["payload", "items", 0])).toBe(first);
  expect(origins.at(["payload", "items", 0, "name"])).toBe(first);
  expect(origins.at(["payload", "items", 3])).toBe(items);
  expect(origins.at(["payload", "other"])).toBe(payload);
  expect(origins.at(["unknown"])).toBe(fallback);
  expect(origins.at([])).toBe(fallback);
  expect(() => origins.at([true] as never)).toThrow(TypeError);
  expect(() => origins.at([true] as never)).toThrow(/segment/);
});

test("parameter distinguishes an explicit null from no default", () => {
  const valueType = new Type({ kind: ValueKind.INTEGER, nullable: true });
  const contract = new ValueContract(valueType);
  const absent = new Parameter("value", { contract, coerceStrings: true });
  const explicitNull = new Parameter("value", {
    contract,
    hasDefault: true,
    default: null,
    coerceStrings: true,
  });

  expect(absent.hasDefault).toBe(false);
  expect(explicitNull.hasDefault).toBe(true);
  expect(absent.contract).toBe(contract);
  expect(explicitNull.contract).toBe(contract);
  expect(explicitNull.contract!.type).toBe(valueType);
  expect(absent.required).toBe(false);
  expect(explicitNull.required).toBe(false);
  expect(absent.coerceStrings).toBe(true);
  expect(explicitNull.coerceStrings).toBe(true);
  expect(absent).not.toEqual(explicitNull);
  expect("typeStr" in absent).toBe(false);
});

test.each<[string, ConstructorParameters<typeof Parameter>[1]]>([
  ["required with a default", { required: true, hasDefault: true }],
  ["a default without hasDefault", { default: 0n }],
  ["a default origin without hasDefault", { defaultOrigin: origin("default") }],
])("parameter rejects %s", (_name, options) => {
  expect(() => new Parameter("value", options)).toThrow(RangeError);
  expect(() => new Parameter("value", options)).toThrow(/default/);
});

test("value contract uses a fresh empty origin index", () => {
  const valueType = Type.scalar(ValueKind.STRING);
  const first = new ValueContract(valueType);
  const second = new ValueContract(valueType);

  expect(first.type).toBe(valueType);
  expect(first.origins).toEqual(new ContractOrigins());
  expect(first.origins).not.toBe(second.origins);
});

test("partial record contract rejects ambiguous representations", () => {
  const integer = new ValueContract(Type.scalar(ValueKind.INTEGER));
  expect(
    () =>
      new PartialRecordContract([
        ["x", integer],
        ["x", null],
      ]),
  ).toThrow(RangeError);
  expect(
    () =>
      new PartialRecordContract([
        ["x", integer],
        ["x", null],
      ]),
  ).toThrow(/duplicate/);
  expect(() => new PartialRecordContract([["x", integer]])).toThrow(RangeError);
  expect(() => new PartialRecordContract([["x", integer]])).toThrow(/fully typed/);
});

test("parameter maps every nested origin without copying semantics", () => {
  const valueType = Type.scalar(ValueKind.OBJECT);
  const defaultValue = { name: "Ada" };
  const parameter = new Parameter("profile", {
    contract: new ValueContract(
      valueType,
      new ContractOrigins({ default: origin("contract"), entries: [[["name"], origin("field")]] }),
    ),
    hasDefault: true,
    default: defaultValue,
    coerceStrings: true,
    origin: origin("parameter"),
    defaultOrigin: origin("default"),
  });

  const mapped = parameter.mapOrigins((ref) => origin(`mapped:${ref.key}`));

  expect(mapped.name).toBe(parameter.name);
  expect(mapped.default).toBe(defaultValue);
  expect(mapped.contract).not.toBeNull();
  expect(mapped.contract!.type).toBe(valueType);
  expect(mapped.contract!.origins.default).toEqual(origin("mapped:contract"));
  expect(mapped.contract!.origins.entries).toEqual([[["name"], origin("mapped:field")]]);
  expect(mapped.origin).toEqual(origin("mapped:parameter"));
  expect(mapped.defaultOrigin).toEqual(origin("mapped:default"));
});

test("each carrier maps present origins and skips null", () => {
  const calls: OriginRef[] = [];
  const mapper = (ref: OriginRef): OriginRef => {
    calls.push(ref);
    return origin(`new:${ref.key}`);
  };

  const present = origin("present");
  expect(new Binding("x", { origin: present }).mapOrigins(mapper).origin).toEqual(origin("new:present"));
  expect(new Assertion("x", "x", { origin: present }).mapOrigins(mapper).origin).toEqual(
    origin("new:present"),
  );
  expect(new ContractOrigins({ default: present }).mapOrigins(mapper).default).toEqual(
    origin("new:present"),
  );
  expect(
    new ValueContract(Type.scalar(ValueKind.STRING), new ContractOrigins({ default: present })).mapOrigins(
      mapper,
    ).origins.default,
  ).toEqual(origin("new:present"));
  const partial = new PartialRecordContract([
    ["known", new ValueContract(Type.scalar(ValueKind.STRING), new ContractOrigins({ default: present }))],
    ["opaque", null],
  ]);
  expect(partial.mapOrigins(mapper).fields[0]![1]!.origins.default).toEqual(origin("new:present"));

  const callCount = calls.length;
  new Binding("x").mapOrigins(mapper);
  new Assertion("x", "x").mapOrigins(mapper);
  new ContractOrigins().mapOrigins(mapper);
  new ValueContract(Type.scalar(ValueKind.STRING)).mapOrigins(mapper);
  new Parameter("x").mapOrigins(mapper);
  expect(calls.length).toBe(callCount);
});

test("origin mapper exceptions propagate", () => {
  class MappingError extends Error {
    override name = "MappingError";
  }
  const fail = (): OriginRef => {
    throw new MappingError("cannot rebase");
  };

  expect(() => new Binding("x", { origin: origin("x") }).mapOrigins(fail)).toThrow(MappingError);
  expect(() => new Binding("x", { origin: origin("x") }).mapOrigins(fail)).toThrow("cannot rebase");
});

const CARRIERS: [string, { mapOrigins(mapper: (ref: OriginRef) => OriginRef): unknown }][] = [
  ["a binding", new Binding("x", { origin: origin("binding") })],
  ["an assertion", new Assertion("x", "x", { origin: origin("assertion") })],
  ["a contract default", new ContractOrigins({ default: origin("contract-default") })],
  ["a contract entry", new ContractOrigins({ entries: [[["field"], origin("contract-entry")]] })],
  ["a parameter", new Parameter("x", { origin: origin("parameter") })],
  ["a parameter default", new Parameter("x", { hasDefault: true, defaultOrigin: origin("parameter-default") })],
];

for (const invalid of [null, "not-an-origin"]) {
  test.each(CARRIERS)(`origin mapper rejects ${String(invalid)} for %s`, (_name, carrier) => {
    const mapper = (() => invalid) as unknown as (ref: OriginRef) => OriginRef;
    expect(() => carrier.mapOrigins(mapper)).toThrow(TypeError);
    expect(() => carrier.mapOrigins(mapper)).toThrow(/mapper must return an OriginRef/);
  });
}

// --- TS-only rows --- //

test("a seed connection takes a name or an index and a name", () => {
  const byName = new SeedConnection("child/__start__", "x", ["x"]);
  const byItem = new SeedConnection("child/__start__", "x", [0, "x"]);

  expect(byName.seedPath).toEqual(["x"]);
  expect(byItem.seedPath).toEqual([0, "x"]);
  expect(Object.isFrozen(byItem)).toBe(true);
  expect(Object.isFrozen(byItem.seedPath)).toBe(true);
  expect(() => new SeedConnection("s", "x", [-1, "x"])).toThrow(RangeError);
  expect(() => new SeedConnection("s", "x", [-1, "x"])).toThrow("seed_path item index must be a non-negative integer");
  expect(() => new SeedConnection("", "x", ["x"])).toThrow("seed connection identifiers must be non-empty strings");
  expect(() => new SeedConnection("s", "", ["x"])).toThrow(RangeError);
  expect(() => new SeedConnection("s", "x", [""])).toThrow("seed_path must end in a non-empty parameter name");
  expect(() => new SeedConnection("s", "x", [0, 1, "x"])).toThrow("seed_path must be a name or an index/name tuple");
});

test("a child flow boundary needs two distinct non-empty ids", () => {
  const boundary = new ChildFlowBoundary("c/__start__", "c/__end__");

  expect(Object.keys(boundary)).toEqual(["startId", "endId"]);
  expect(Object.isFrozen(boundary)).toBe(true);
  expect(() => new ChildFlowBoundary("a", "a")).toThrow(RangeError);
  expect(() => new ChildFlowBoundary("a", "a")).toThrow("child START and END must be distinct");
  expect(() => new ChildFlowBoundary("", "a")).toThrow("child boundary identifiers must be non-empty strings");
});

test("a host tool call is frozen", () => {
  const integer = Type.scalar(ValueKind.INTEGER);
  const call = new HostToolCall("search", ["q"], new Map([["q", integer]]), null);

  expect(Object.keys(call)).toEqual(["toolId", "inputs", "inputTypes", "outputType"]);
  expect(call.inputs).toEqual(["q"]);
  expect(call.inputTypes.get("q")).toBe(integer);
  expect(Object.isFrozen(call)).toBe(true);
  expect(Object.isFrozen(call.inputs)).toBe(true);
  expect(() => assign(call, "toolId", "other")).toThrow(TypeError);
});
