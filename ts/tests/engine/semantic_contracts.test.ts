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
 * It also ports the output contract rows: a partial record's closed shape and known fields, and
 * field reads on fieldless types. A `FileRef` stands in for Python's `object()` in an opaque
 * field.
 *
 * TS-only: `SeedConnection` (a name, an index and name, a negative index, an empty name, a
 * three-part path), `ChildFlowBoundary` (equal or empty ids) and a frozen `HostToolCall`, which
 * no Python unit test covers; `outputContractType` (each arm and a non-contract),
 * `outputContractFileType` and `fileFieldsType`, `fileFieldError`'s text, `outputContractOrigin`
 * (a partial record field, a non-string head), and `validateOutputContract` keeping an int in a
 * `float` field, a `__proto__` field as data, its missing and unknown field messages, and a
 * `TypedValue`'s plain value under a null contract; and `sameOutputContract`, the TS form of the
 * contracts' dataclass `==` (types, origins by key in order, partial record fields, `null`).
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
  fileFieldError,
  fileFieldsType,
  HostToolCall,
  type OutputContract,
  outputContractFieldError,
  outputContractFileType,
  outputContractOrigin,
  outputContractType,
  Parameter,
  PartialRecordContract,
  sameOutputContract,
  SeedConnection,
  validateOutputContract,
  ValueContract,
  type ValuePath,
} from "../../src/nodes/contracts.ts";
import * as nodes from "../../src/nodes/index.ts";
import type { OutputContract as NodesOutputContract, ValuePath as NodesValuePath } from "../../src/nodes/index.ts";
import { FileRef, Type, TypeCheckError, TypedValue, ValueKind } from "../../src/typesys/index.ts";

/** Build a short valid origin for carrier tests. */
function origin(key: string): OriginRef {
  return new OriginRef(key);
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
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

test.each<[string, ConstructorParameters<typeof Parameter>[1], RegExp]>([
  ["required with a default", { required: true, hasDefault: true }, /^a required parameter cannot have a default$/],
  ["a default without hasDefault", { default: 0n }, /^parameter default requires has_default=true$/],
  [
    "a default origin without hasDefault",
    { defaultOrigin: origin("default") },
    /^parameter default_origin requires has_default=true$/,
  ],
])("parameter rejects %s", (_name, options, message) => {
  expect(() => new Parameter("value", options)).toThrow(RangeError);
  expect(() => new Parameter("value", options)).toThrow(message);
});

test("value contract uses a fresh empty origin index", () => {
  const valueType = Type.scalar(ValueKind.STRING);
  const first = new ValueContract(valueType);
  const second = new ValueContract(valueType);

  expect(first.type).toBe(valueType);
  expect(first.origins).toEqual(new ContractOrigins());
  expect(first.origins).not.toBe(second.origins);
});

test("partial record contract enforces closed shape and known fields", () => {
  const known = origin("known");
  const contract = new PartialRecordContract([
    ["count", new ValueContract(Type.scalar(ValueKind.INTEGER), new ContractOrigins({ default: known }))],
    ["note", null],
  ]);

  validateOutputContract(contract, { count: 1n, note: new FileRef({ uri: "file:///a" }) });
  expect(contract.names).toEqual(["count", "note"]);
  expect(outputContractFieldError(contract, ["count"], "p.output.count")).toBe(null);
  expect(outputContractFieldError(contract, ["note", "anything"], "p")).toBe(null);
  expect(outputContractFieldError(contract, ["absent"], "p.output.absent")).toBe(
    "reference ${p.output.absent} reads unknown field 'absent' on record",
  );
  expect(outputContractFieldError(contract, ["count", "x"], "p.output.count.x")).toBe(
    "reference ${p.output.count.x} reads field 'x' on type int, which has no fields",
  );
  expect(outputContractOrigin(contract, ["count"])).toBe(known);
  expect(outputContractOrigin(contract, ["note"])).toBe(null);

  const cases: [Record<string, unknown>, RegExp][] = [
    [{ count: "one", note: "x" }, /does not match declared type int/],
    [{ count: 1n }, /missing required/],
    [{ count: 1n, note: "x", extra: true }, /unknown fields/],
  ];
  for (const [value, message] of cases) {
    const error = thrown(() => validateOutputContract(contract, value));
    expect(error).toBeInstanceOf(TypeCheckError);
    expect(error.message).toMatch(message);
  }
});

test("output contract field error rejects fields on fieldless types", () => {
  const err = (typ: Type, ...fields: string[]): string | null =>
    outputContractFieldError(new ValueContract(typ), fields, "x");

  expect(err(Type.scalar(ValueKind.STRING), "a")).toBe(
    "reference ${x} reads field 'a' on type str, which has no fields",
  );
  expect(err(Type.scalar(ValueKind.LIST_STRING), "a")).toContain("on type list(str)");
  expect(err(new Type({ kind: ValueKind.INTEGER, nullable: true }), "a")).toContain("on type Optional(int)");
  expect(err(Type.scalar(ValueKind.NONE), "a")).toContain("on type None");
  expect(err(Type.scalar(ValueKind.OBJECT), "a", "b")).toBe(null);
  expect(err(Type.scalar(ValueKind.FILE), "uri")).toBe(null);
  expect(err(Type.scalar(ValueKind.STRING))).toBe(null); // no field read at all
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

test("nodes package exports the canonical contract vocabulary", () => {
  expect(nodes.Assertion).toBe(Assertion);
  expect(nodes.Binding).toBe(Binding);
  expect(nodes.ContractOrigins).toBe(ContractOrigins);
  expect(nodes.PartialRecordContract).toBe(PartialRecordContract);
  expect(nodes.Parameter).toBe(Parameter);
  expect(nodes.ValueContract).toBe(ValueContract);
  expect(nodes.asBinding).toBe(asBinding);
  expect(nodes.bindingSource).toBe(bindingSource);
  // The type-only names have no runtime value; these lines compile only if the index exports them.
  const contract: NodesOutputContract = new ValueContract(Type.scalar(ValueKind.STRING));
  const path: NodesValuePath = ["field", 0];
  expect(contract).toBeInstanceOf(ValueContract);
  expect(path).toEqual(["field", 0]);
});

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

const INT = Type.scalar(ValueKind.INTEGER);
const FLOAT = Type.scalar(ValueKind.NUMBER);
const FILE = Type.scalar(ValueKind.FILE);
const LIST_FILE = new Type({ kind: ValueKind.LIST_ANY, element: FILE });

test("outputContractType gives a value contract's type and null otherwise", () => {
  const value = new ValueContract(INT);
  expect(outputContractType(null)).toBe(null);
  expect(outputContractType(value)).toBe(INT);
  expect(outputContractType(new PartialRecordContract([["x", value], ["y", null]]))).toBe(null);
  const forged = "x" as unknown as OutputContract;
  expect(() => outputContractType(forged)).toThrow(TypeError);
  expect(() => outputContractType(forged)).toThrow(
    "output contract must be an exact ValueContract, PartialRecordContract, or null, got string",
  );
});

test("outputContractFileType and fileFieldsType give the type that restores files", () => {
  const record = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["f", FILE],
      ["n", INT],
    ]),
    required: new Set(["f", "n"]),
  });
  const fileOnly = new Type({ kind: ValueKind.OBJECT, fields: new Map([["f", FILE]]), required: new Set() });

  expect(outputContractFileType(new ValueContract(LIST_FILE))).toBe(LIST_FILE);
  expect(outputContractFileType(new ValueContract(record))).toBe(record);
  expect(outputContractFileType(new ValueContract(INT))).toBe(null);
  expect(outputContractFileType(null)).toBe(null);
  const partial = new PartialRecordContract([
    ["f", new ValueContract(FILE)],
    ["n", new ValueContract(INT)],
    ["o", null],
  ]);
  expect(outputContractFileType(partial)).toEqual(fileOnly);
  expect(
    fileFieldsType([
      ["f", FILE],
      ["n", INT],
    ]),
  ).toEqual(fileOnly);
  expect(fileFieldsType([["n", INT]])).toBe(null);
});

test("fileFieldError names the fields a file has", () => {
  const message = "reference ${a.f.size} reads field 'size' on a file, which has only uri, mime, and name";
  expect(fileFieldError("a.f.size", "size")).toBe(message);
  expect(outputContractFieldError(new ValueContract(FILE), ["size"], "a.f.size")).toBe(message);
});

test("outputContractOrigin reads a partial record field's origins", () => {
  const nested = origin("nested");
  const contract = new PartialRecordContract([
    ["xs", new ValueContract(LIST_FILE, new ContractOrigins({ entries: [[[0], nested]] }))],
    ["o", null],
  ]);
  expect(outputContractOrigin(contract, ["xs", 0, "uri"])).toBe(nested);
  expect(outputContractOrigin(contract, ["xs"])).toBe(null);
  expect(outputContractOrigin(contract, [0])).toBe(null);
  expect(outputContractOrigin(contract, [])).toBe(null);
  expect(outputContractOrigin(null, ["xs"])).toBe(null);
});

test("validateOutputContract keeps an int in a float field and rebuilds the record as data", () => {
  const contract = new PartialRecordContract([
    ["x", new ValueContract(FLOAT)],
    ["__proto__", null],
  ]);
  const value = JSON.parse('{"__proto__": "p"}') as Record<string, unknown>;
  value["x"] = 1n;

  const checked = validateOutputContract(contract, value) as Record<string, unknown>;
  expect(checked["x"]).toBe(1n);
  expect(Object.keys(checked)).toEqual(["__proto__", "x"]);
  expect(Object.getOwnPropertyDescriptor(checked, "__proto__")?.value).toBe("p");
  expect(Object.getPrototypeOf(checked)).toBe(Object.prototype);
});

test("validateOutputContract names missing and unknown fields", () => {
  const contract = new PartialRecordContract([
    ["b", null],
    ["a", null],
  ]);
  const missing = thrown(() => validateOutputContract(contract, {}));
  expect(missing).toBeInstanceOf(TypeCheckError);
  expect(missing.message).toBe('record missing required fields: ["b","a"]');
  expect((missing as TypeCheckError).path).toEqual(["b"]);
  const unknown = thrown(() => validateOutputContract(contract, { a: 1n, b: 2n, z: 3n, y: 4n }));
  expect(unknown.message).toBe('record has unknown fields: ["y","z"]');
  expect((unknown as TypeCheckError).path).toEqual(["y"]);
  expect(() => validateOutputContract(contract, [])).toThrow("[] is not an object for a partial record contract");
});

test("validateOutputContract gives a TypedValue's plain value under a null contract", () => {
  expect(validateOutputContract(null, new TypedValue(ValueKind.INTEGER, 3n))).toBe(3n);
  const plain = { a: 1n };
  expect(validateOutputContract(null, plain)).toBe(plain);
});

test("sameOutputContract compares value contracts by type and origins", () => {
  const record = (fieldType: Type) =>
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", fieldType]]), required: new Set(["a"]) });
  const at = (options: ConstructorParameters<typeof ContractOrigins>[0]) => new ContractOrigins(options);

  expect(sameOutputContract(new ValueContract(INT), new ValueContract(Type.scalar(ValueKind.INTEGER)))).toBe(true);
  expect(sameOutputContract(new ValueContract(INT), new ValueContract(FLOAT))).toBe(false);
  expect(sameOutputContract(new ValueContract(record(INT)), new ValueContract(record(INT)))).toBe(true);
  expect(sameOutputContract(new ValueContract(record(INT)), new ValueContract(record(FLOAT)))).toBe(false);

  // Origins compare by key, not by identity.
  const rooted = (key: string) => new ValueContract(INT, at({ default: origin(key) }));
  expect(sameOutputContract(rooted("r"), rooted("r"))).toBe(true);
  expect(sameOutputContract(rooted("r"), rooted("s"))).toBe(false);
  expect(sameOutputContract(rooted("r"), new ValueContract(INT))).toBe(false);

  const nested = (entries: readonly (readonly [ValuePath, OriginRef])[]) =>
    new ValueContract(record(INT), at({ entries }));
  const ax = () => nested([[["a"], origin("x")]]);
  const axBy = nested([
    [["a"], origin("x")],
    [["b"], origin("y")],
  ]);
  expect(sameOutputContract(ax(), ax())).toBe(true);
  expect(sameOutputContract(ax(), nested([[["b"], origin("x")]]))).toBe(false);
  expect(sameOutputContract(ax(), nested([[["a"], origin("y")]]))).toBe(false);
  expect(sameOutputContract(axBy, ax())).toBe(false);
  // Entries are canonical, so the same paths with their origins swapped differ.
  const ayBx = nested([
    [["a"], origin("y")],
    [["b"], origin("x")],
  ]);
  expect(sameOutputContract(axBy, ayBx)).toBe(false);
});

test("sameOutputContract compares partial records field by field, and null only equals null", () => {
  const int = new ValueContract(INT);
  const partial = (fields: readonly (readonly [string, ValueContract | null])[]) => new PartialRecordContract(fields);
  const aIntB = partial([
    ["a", int],
    ["b", null],
  ]);
  /** The record `{a: first, b: untyped}`. */
  const withA = (first: ValueContract | null) =>
    partial([
      ["a", first],
      ["b", null],
    ]);

  expect(sameOutputContract(aIntB, withA(new ValueContract(INT)))).toBe(true);
  expect(
    sameOutputContract(
      aIntB,
      partial([
        ["b", null],
        ["a", int],
      ]),
    ),
  ).toBe(false);
  expect(
    sameOutputContract(
      aIntB,
      partial([
        ["a", int],
        ["c", null],
      ]),
    ),
  ).toBe(false);
  expect(sameOutputContract(aIntB, withA(null))).toBe(false);
  expect(sameOutputContract(aIntB, withA(new ValueContract(FLOAT)))).toBe(false);
  expect(sameOutputContract(partial([["a", null]]), partial([["a", null]]))).toBe(true);
  expect(sameOutputContract(partial([["a", null]]), int)).toBe(false);

  expect(sameOutputContract(null, null)).toBe(true);
  expect(sameOutputContract(null, int)).toBe(false);
  expect(sameOutputContract(int, null)).toBe(false);
  expect(sameOutputContract(partial([["a", null]]), null)).toBe(false);
});
