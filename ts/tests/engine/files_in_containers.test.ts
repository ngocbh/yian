/**
 * Files inside typed containers, a port of the value checks (`buildValueAs`), file
 * normalization (`normalizeFiles`), the overlap tests, the two typed `bindParams` reads, the
 * `field_of` table, the two traced item reads and
 * `test_a_partial_loop_keeps_an_omitted_file_as_a_file` of
 * `tests/engine/test_files_in_containers.py`, and the inequality half of
 * `test_a_source_entry_with_a_record_reading_type_is_hashable` (through `sameSourceEntry`), with
 * `FileRef` constructor refusals (`RangeError`, as pydantic's `ValidationError` is a
 * `ValueError`). Where Python checks that a value is kept with `is`, these tests use `toBe`.
 *
 * Changed: in `test_a_partial_loop_keeps_an_omitted_file_as_a_file` the child is a stub object in
 * place of `_child_flow()`, and Python ints in carried values are `bigint`.
 *
 * TS-only: the `normalizeTypedFiles` tests, and the `normalizeFiles` rows keeping an `object`
 * value that holds NaN, turning a `-0` at an `object` position into `0`, and keeping a file
 * form whose text holds a lone surrogate.
 *
 * Ported, each flow loaded with its `defs:` and run through `runFlow` (with `learnable` where
 * Python passes it): `test_a_map_over_a_list_of_files_into_a_file_input_succeeds`,
 * `test_a_loop_carrying_a_file_succeeds`, `test_a_call_into_a_list_of_files_input_succeeds`,
 * `test_a_call_body_returning_a_record_with_a_file_succeeds`, for learnable false and true
 * `test_a_map_body_reads_its_file_input_as_a_file`,
 * `test_a_call_body_reads_its_list_of_files_input_as_files`,
 * `test_the_parent_reads_a_file_a_call_body_returns`,
 * `test_the_parent_reads_the_typed_file_of_a_partial_record_body_output`,
 * `test_the_parent_reads_a_file_a_loop_carries` and
 * `test_a_loop_reads_a_file_in_a_carried_object_as_a_mapping_on_every_iteration`, and for each
 * learnable and body output `test_a_partial_loop_compares_an_omitted_file_with_a_returned_one` (4)
 * and `test_a_partial_loop_compares_a_carried_file_the_body_does_not_take` (4). The CODE bodies
 * are TS: a body builds a file with the `FileRef` it imports from `src/typesys/values.ts` (as
 * Python's body calls `__import__`) and reads `x.constructor.name` where Python reads
 * `type(x).__name__`; ints arrive as `number` (decision 45) and the outputs' ints are bigints.
 *
 * Later (T4-8): needs `resumeFlow`: the `_LOOP_FLOW` row of
 * `test_resuming_a_finished_run_with_files_replays_its_output` and
 * `test_a_log_whose_loop_default_was_digested_as_a_mapping_still_resumes`.
 *
 * Later (T4-8): flow runs: the runs with no CALL, MAP or LOOP,
 * `test_a_tagged_union_of_a_file_and_an_object_member_loads`,
 * `test_an_object_binding_delivers_a_nested_file_as_its_mapping`,
 * `test_an_object_output_binding_over_a_record_with_a_file_succeeds`,
 * `test_a_file_binding_delivers_the_json_form_as_a_file`, `test_a_float_binding_still_delivers_an_int`,
 * `test_a_constructor_in_a_flow_delivers_a_nested_file_as_its_mapping`,
 * `test_a_binding_reads_the_fields_of_a_file`, `test_a_template_binding_reads_the_uri_of_a_file`,
 * `test_a_prompt_reads_the_name_of_a_file`, `test_a_case_routes_on_the_uri_of_a_file` (2),
 * `test_a_flow_assert_reads_the_uri_of_a_file`, `test_a_node_pre_assert_reads_the_uri_of_a_file`,
 * `test_a_node_post_assert_reads_the_uri_of_a_file_in_its_output`,
 * `test_another_field_of_a_file_is_missing_like_a_missing_key` (2),
 * `test_render_as_json_writes_a_file_as_its_mapping`,
 * `test_a_run_reads_the_fields_of_a_file_through_its_trace` (2),
 * `test_a_reader_sees_the_files_of_a_typed_output` (2),
 * `test_the_success_event_holds_the_files_of_a_typed_output` (2),
 * `test_an_untyped_output_holding_files_is_read_as_mappings` (2),
 * `test_a_flow_input_file_reaches_code_as_a_file` (2),
 * `test_backward_context_holds_the_files_of_a_typed_output_as_json`,
 * `test_the_run_result_holds_the_files_of_a_typed_output` (6) and
 * `test_the_run_result_holds_the_files_of_an_untyped_output_as_mappings` (2); and the resume rows
 * `test_resuming_a_run_with_a_typed_list_of_files_succeeds`, the `_READER_FLOW` row of
 * `test_resuming_a_finished_run_with_files_replays_its_output`,
 * `test_a_reader_after_resume_sees_the_files_of_a_replayed_output` (2),
 * `test_resuming_an_unfinished_run_returns_the_files_of_a_typed_output` (2) and
 * `test_resuming_a_finished_run_returns_the_files_of_a_typed_output` (6).
 *
 * Not ported: the hash half of `test_a_source_entry_with_a_record_reading_type_is_hashable`, since
 * a `SourceEntry` has no hash (maps key sources by `sourceKey`); the `field_of` rows on dict
 * subclasses, `defaultdict` and a `FileRef` subclass; a record is a plain object and a dotted read
 * gives its own keys. The tuple in `test_an_object_without_files_is_stored_as_given` is a list
 * here, and `test_normalize_files_reads_a_tuple_as_a_list_only_when_it_changes` is not ported (no
 * tuples in TS).
 */

import { describe, expect, test } from "bun:test";

import { evalExpr } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { ResolvedValue, SourceRef } from "../../src/expr/lineage.ts";
import { bindParams } from "../../src/nodes/binding.ts";
import { Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { runFlow } from "../../src/run.ts";
import { SourceEntry, sameSourceEntry } from "../../src/runtime/trace.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  buildValueAs,
  FileRef,
  fieldOf,
  normalizeFiles,
  normalizeTypedFiles,
  Type,
  TypeCheckError,
  TypedValue,
  typesOverlap,
  type Value,
  type ValueRecord,
  ValueKind,
} from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";

const REGISTRY = readTypedefs({
  Rec: { f: "file", n: "int" },
  Form: { uri: "str", mime: "Optional(str)", name: "Optional(str)" },
  IntUri: { uri: "int" },
  Other: { uri: "str", other: "str" },
  XFile: { x: "file" },
  XObject: { x: "object" },
  Box: { x: "object" },
  Num: { n: "float" },
  TFile: { kind: "Literal(file)", x: "file" },
  TObject: { kind: "Literal(object)", x: "object" },
});

/** Resolve a type expression against this module's typedefs. */
const t = (text: string): Type => typeFor(text, REGISTRY);

const FILE = Type.scalar(ValueKind.FILE);
const OBJECT = Type.scalar(ValueKind.OBJECT);
const ANY = Type.scalar(ValueKind.ANY);
const REF = new FileRef({ uri: "f" });

/** A fresh copy of a file's JSON form, `{uri: "f", mime: null, name: null}`. */
function form(): Value {
  return { uri: "f", mime: null, name: null };
}

/** Return the `TypeCheckError` `run` throws. */
function typeCheckError(run: () => unknown): TypeCheckError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("expected a TypeCheckError");
}

describe("a file slot", () => {
  test("a file slot reads the JSON form", () => {
    const typed = buildValueAs(FILE, form());
    expect(typed.kind).toBe(ValueKind.FILE);
    expect(typed.value).toEqual(REF);
    expect(typed.value).toBeInstanceOf(FileRef);
  });

  test("a file slot keeps a FileRef", () => {
    const typed = buildValueAs(FILE, REF);
    expect(typed.kind).toBe(ValueKind.FILE);
    expect(typed.value).toBe(REF);
  });

  test("the JSON form keeps its mime and name", () => {
    const value = { uri: "f", mime: "text/plain", name: "a.txt" };
    expect(buildValueAs(FILE, value).value).toEqual(new FileRef(value));
  });

  test.each<[string, Value, Value]>([
    ["list(file)", [form(), REF], [REF, REF]],
    ["Rec", { f: form(), n: 1n }, { f: REF, n: 1n }],
    ["Optional(file)", null, null],
    ["list(Optional(file))", [form(), null], [REF, null]],
    ["list(Rec)", [{ f: form(), n: 1n }], [{ f: REF, n: 1n }]],
  ])("a nested file slot reads the JSON form: %s", (declared, value, expected) => {
    expect(buildValueAs(t(declared), value).toObject()).toEqual(expected);
  });

  test.each<[string, Value]>([
    ["a missing mime", { uri: "f", name: null }],
    ["an extra size", { uri: "f", mime: null, name: null, size: 1n }],
    ["an int uri", { uri: 1n, mime: null, name: null }],
    ["an int mime", { uri: "f", mime: 2n, name: null }],
  ])("a file slot rejects a record that is not the JSON form: %s", (_id, value) => {
    expect(typeCheckError(() => buildValueAs(FILE, value)).message).toContain(
      "does not match declared type file",
    );
    const error = typeCheckError(() => buildValueAs(t("list(file)"), [value]));
    expect(error.message).toContain("does not match declared type file");
    expect(error.path).toEqual([0]);
  });
});

describe("an object slot reads a nested file as its mapping", () => {
  test("an object reads a nested file as its mapping", () => {
    expect(buildValueAs(OBJECT, { f: REF }).value).toEqual({ f: form() });
  });

  test("a list of objects reads a nested file as its mapping", () => {
    const flat = buildValueAs(ValueKind.LIST_OBJECT, [{ f: REF }, { g: [REF] }]);
    const refined = buildValueAs(t("list(object)"), [{ f: REF }]);
    expect(flat.value).toEqual([{ f: form() }, { g: [form()] }]);
    expect(refined.value).toEqual([{ f: form() }]);
  });

  test("an object field reads a nested file as its mapping", () => {
    expect(buildValueAs(t("Box"), { x: { f: REF } }).value).toEqual({ x: { f: form() } });
  });

  test("an object without files is stored as given", () => {
    const value = { items: [1n, 2n], nested: { a: [3n] } };
    const stored = buildValueAs(OBJECT, value).value as Record<string, Value>;
    expect(stored).toBe(value);
    expect(stored["items"]).toBe(value.items);
  });

  test("reading nested files leaves the given object unchanged", () => {
    const value = { f: REF, keep: { a: 1n } };
    const stored = buildValueAs(OBJECT, value).value as Record<string, Value>;
    expect(value).toEqual({ f: REF, keep: { a: 1n } });
    expect(value.f).toBe(REF);
    expect(stored["keep"]).toBe(value.keep);
  });

  test.each<[string, Type, Value]>([
    ["object holding a file", OBJECT, REF],
    ["list(object) holding a file", t("list(object)"), [REF]],
    ["Any holding a file", ANY, REF],
    ["Any holding a nested file", ANY, { f: REF }],
    ["list(Any) holding a file", t("list(Any)"), [REF]],
    ["list(Any) holding a nested file", t("list(Any)"), [{ f: REF }]],
  ])("a file where no file position reads it still fails: %s", (_id, declared, value) => {
    typeCheckError(() => buildValueAs(declared, value));
  });

  test("an object holding itself still fails", () => {
    const cyclic: Record<string, Value> = {};
    cyclic["self"] = cyclic;
    cyclic["f"] = REF;
    expect(typeCheckError(() => buildValueAs(OBJECT, cyclic)).message).toBe(
      "value contains itself",
    );
  });
});

describe("normalizeFiles", () => {
  test("reads the JSON form at a file position", () => {
    expect(normalizeFiles(t("Rec"), { f: form(), n: 1n })).toStrictEqual({ f: REF, n: 1n });
  });

  test("reads a nested file as a mapping at an object position", () => {
    expect(normalizeFiles(t("Box"), { x: { f: REF } })).toStrictEqual({ x: { f: form() } });
  });

  test("keeps an int at a float position", () => {
    const normalized = normalizeFiles(t("Num"), { n: 3n }) as ValueRecord;
    expect(typeof normalized["n"]).toBe("bigint");
  });

  test.each<[string, Value]>([
    ["int", 3n],
    ["Num", { n: 3n }],
    ["list(str)", ["a"]],
    ["Rec", { f: REF, n: 1n }],
    ["list(file)", [REF]],
    ["object", { a: [1n] }],
    ["Any", { uri: "f", mime: null, name: null }],
  ])("returns the same value when nothing changes: %s", (declared, value) => {
    expect(normalizeFiles(t(declared), value)).toBe(value);
  });

  test.each<[string, string[], Value]>([
    ["XFile first", ["XFile", "XObject"], { x: REF }],
    ["XObject first", ["XObject", "XFile"], { x: form() }],
  ])("picks an untagged union member like the write check: %s", (_id, members, expected) => {
    // Both members admit `{x: form}`, so the first one decides how `x` reads.
    const union = new Type({ kind: ValueKind.OBJECT, variants: members.map(t) });
    const value = { x: form() };
    expect(normalizeFiles(union, value)).toStrictEqual(expected);
    expect(buildValueAs(union, value).toObject()).toStrictEqual(expected);
  });

  test.each<[string, Value, Value]>([
    ["file", { kind: "file", x: form() }, { kind: "file", x: REF }],
    ["object", { kind: "object", x: { f: REF } }, { kind: "object", x: { f: form() } }],
  ])("picks a tagged union member by its tag: %s", (_id, value, expected) => {
    expect(normalizeFiles(t("Union(TFile, TObject)"), value)).toStrictEqual(expected);
  });

  test("reads an Optional union", () => {
    const declared = t("Optional(Union(TFile, TObject))");
    expect(normalizeFiles(declared, null)).toBeNull();
    expect(normalizeFiles(declared, { kind: "file", x: form() })).toStrictEqual({
      kind: "file",
      x: REF,
    });
  });

  test.each<[string, Value]>([
    ["Rec", "text"],
    ["list(file)", { f: form() }],
    ["Union(TFile, TObject)", { kind: "other", x: form() }],
    ["Union(TFile, TObject)", { x: form() }],
    ["Union(XFile, IntUri)", { y: form() }],
    ["Union(XFile, IntUri)", [form()]],
    // An object position reads with the write check's JSON walk, which refuses NaN, so the
    // whole value is kept; Python still maps the file.
    ["object", { f: REF, bad: NaN }],
  ])("keeps a value that fails the type: %s", (declared, value) => {
    expect(normalizeFiles(t(declared), value)).toBe(value);
  });

  test("turns a -0 at an object position into 0", () => {
    // Negative zero becomes 0 when a float becomes a value; Python keeps -0.0 and the value.
    const value = { x: { a: -0 } };
    const normalized = normalizeFiles(t("Box"), value) as ValueRecord;
    expect(normalized).not.toBe(value);
    expect(Object.is((normalized["x"] as ValueRecord)["a"], 0)).toBe(true);
  });

  test("keeps a file form whose text holds a lone surrogate", () => {
    // Text holding a lone surrogate is not a yian value, and normalizeFiles never throws for
    // a value that doesn't fit; Python builds the FileRef.
    const value = { f: { uri: "\ud800", mime: null, name: null }, n: 1n };
    expect(normalizeFiles(t("Rec"), value)).toBe(value);
  });

  test.each<[string, Value]>([
    ["file", form()],
    ["Rec", { f: form(), n: 1n }],
    ["list(Rec)", [{ f: REF, n: 1n }, { f: form(), n: 2n }]],
    ["Optional(list(Optional(file)))", [form(), null]],
    ["object", { f: REF, g: [REF, 1n] }],
    ["list(object)", [{ f: REF }, { a: 1n }]],
    ["Box", { x: { f: [REF] } }],
    ["Union(TFile, TObject)", { kind: "file", x: form() }],
    ["Union(XFile, IntUri)", { x: form() }],
  ])("agrees with the write check: %s", (declared, value) => {
    const typ = t(declared);
    expect(normalizeFiles(typ, value)).toStrictEqual(buildValueAs(typ, value).toObject());
  });
});

describe("normalizeTypedFiles", () => {
  test("a list of files decoded from the wire holds FileRefs", () => {
    const typ = t("list(file)");
    const written = buildValueAs(typ, [REF, form()]);
    const decoded = TypedValue.fromWire(written.toWire());
    expect(decoded.value).toStrictEqual([form(), form()]);
    const normalized = normalizeTypedFiles(typ, decoded);
    expect(normalized.kind).toBe(decoded.kind);
    expect(normalized.value).toStrictEqual([REF, REF]);
  });

  test("returns the same TypedValue when nothing changes", () => {
    const typed = buildValueAs(t("list(file)"), [REF]);
    expect(normalizeTypedFiles(t("list(file)"), typed)).toBe(typed);
  });

  test("a null type returns the TypedValue", () => {
    const typed = TypedValue.fromWire({ kind: "list[Any]", value: [form()] });
    expect(normalizeTypedFiles(null, typed)).toBe(typed);
  });
});

describe("untagged unions whose members can share a file", () => {
  test.each([
    ["file, object", FILE, OBJECT, true],
    ["file, Any", FILE, ANY, true],
    ["file, Form", FILE, t("Form"), true],
    ["XFile, XObject", t("XFile"), t("XObject"), true],
    ["file, int", FILE, t("int"), false],
    ["file, IntUri", FILE, t("IntUri"), false],
    ["file, Other", FILE, t("Other"), false],
  ] as const)("a file overlaps the types that admit its JSON form: %s", (_id, a, b, expected) => {
    expect(typesOverlap(a, b)).toBe(expected);
    expect(typesOverlap(b, a)).toBe(expected);
  });
});

// --- flows that carry files in typed containers --- //

// A CODE body has no `FileRef` in scope, so it imports this module to build one, as Python's
// body calls `__import__` on `yian.typesys.values`.
const VALUES_URL = new URL("../../src/typesys/values.ts", import.meta.url).href;
const F = `new (await import(${JSON.stringify(VALUES_URL)})).FileRef({ uri: "f" })`;

/** Load and run `text` with no inputs, asserting it succeeded; return its output. */
async function okRun(text: string, options: { learnable?: boolean } = {}): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text), {}, options);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

test("a map over a list of files into a file input succeeds", async () => {
  const output = await okRun(`\
id: p
name: p
defs:
  body:
    input:
      f: !file
    nodes:
      c:
        kind: code
        input:
          f: \${input.f}
        code: |
          return 'ok'
    output: \${c.output}
nodes:
  a:
    kind: code
    output: list(file)
    code: |
      return [${F}, ${F}]
  m:
    kind: map
    over: \${a.output}
    call: body
    input:
      f: \${item}
output: \${m.output}
`);

  expect(output).toEqual(["ok", "ok"]);
});

test("a loop carrying a file succeeds", async () => {
  const output = await okRun(`\
id: p
name: p
defs:
  body:
    input:
      f: !file
      n: !int
    nodes:
      c:
        kind: code
        input:
          f: \${input.f}
          n: \${input.n}
        output:
          f: file
          n: int
        code: |
          return { f: inputs['f'], n: inputs['n'] + 1 }
    output: \${c.output}
nodes:
  a:
    kind: code
    output: file
    code: |
      return ${F}
  l:
    kind: loop
    times: 3
    call: body
    input:
      f: \${a.output}
      n: 0
output: \${l.output}
`);

  expect((output as ValueRecord)["n"]).toBe(3n);
});

test("a call into a list of files input succeeds", async () => {
  const output = await okRun(`\
id: p
name: p
defs:
  body:
    input:
      fs: !list(file)
    nodes:
      c:
        kind: code
        input:
          fs: \${input.fs}
        code: |
          return inputs['fs'].length
    output: \${c.output}
nodes:
  a:
    kind: code
    output: list(file)
    code: |
      return [${F}]
  k:
    kind: call
    call: body
    input:
      fs: \${a.output}
output: \${k.output}
`);

  expect(output).toBe(1n);
});

test("a call body returning a record with a file succeeds", async () => {
  await okRun(`\
id: p
name: p
defs:
  body:
    nodes:
      c:
        kind: code
        output:
          f: file
        code: |
          return { f: ${F} }
    output: \${c.output}
nodes:
  k:
    kind: call
    call: body
  d:
    kind: code
    input:
      f: \${k.output.f}
    code: |
      return 'ok'
output: \${d.output}
`);
});

// --- typed bindings deliver the slot's file representation --- //

/** Bind one parameter to `!declared ${a.output}` over a pool holding `output`. */
function bind(declared: Type, output: Value): unknown {
  const pool = new VariablePool();
  pool.set("a", output);
  const record = bindParams(
    [new Parameter("x")],
    new Map([["x", new Binding("${a.output}", { contract: new ValueContract(declared) })]]),
    pool,
  );
  return record["x"];
}

test("an object binding reads a nested file as its mapping", () => {
  expect(bind(OBJECT, { f: REF })).toEqual({ f: form() });
});

test("a file binding reads the JSON form as a file", () => {
  expect(bind(t("Rec"), { f: form(), n: 1n })).toEqual({ f: REF, n: 1n });
});

/** A record holding an own `__proto__` key, as yian builds one. */
function protoKeyRecord(): Value {
  const record = {};
  Object.defineProperty(record, "__proto__", {
    value: 1n,
    writable: true,
    enumerable: true,
    configurable: true,
  });
  return record as Value;
}

describe("fieldOf", () => {
  const file = new FileRef({ uri: "f", mime: "m", name: "n" });
  test.each<[string, Value, string, [boolean, Value]]>([
    ["a key", { a: 1n }, "a", [true, 1n]],
    ["a key holding null", { a: null }, "a", [true, null]],
    ["a missing key", { a: 1n }, "b", [false, null]],
    ["a file's uri", file, "uri", [true, "f"]],
    ["a file's mime", file, "mime", [true, "m"]],
    ["a file's name", file, "name", [true, "n"]],
    ["an unset file field", new FileRef({ uri: "f" }), "name", [true, null]],
    ["another name on a file", new FileRef({ uri: "f" }), "other", [false, null]],
    ["__class__ on a file", new FileRef({ uri: "f" }), "__class__", [false, null]],
    ["model_dump on a file", new FileRef({ uri: "f" }), "model_dump", [false, null]],
    ["a name on text", "text", "upper", [false, null]],
    ["an index on a list", [1n], "0", [false, null]],
    ["a name on null", null, "a", [false, null]],
    // TS-only: a dotted read gives own keys only, never an inherited JS property.
    ["constructor on a record", { a: 1n }, "constructor", [false, null]],
    ["toString on a record", { a: 1n }, "toString", [false, null]],
    ["__proto__ on a record", { a: 1n }, "__proto__", [false, null]],
    ["constructor on a file", new FileRef({ uri: "f" }), "constructor", [false, null]],
    ["length on text", "text", "length", [false, null]],
    ["an own __proto__ key", protoKeyRecord(), "__proto__", [true, 1n]],
    ["a null-prototype record's key", Object.assign(Object.create(null), { a: 1n }), "a", [true, 1n]],
  ])("reads a record key or a file field: %s", (_id, value, name, expected) => {
    expect(fieldOf(value, name)).toEqual(expected);
  });
});

test("a traced item read of a file field routes to its source", () => {
  const item = new ResolvedValue(REF, true, new SourceRef("run", 1, ["xs", 0]));

  const traced = evalExpr(parseExpr("item.uri"), () => new ResolvedValue(null, false), { item, trace: true });

  expect(traced.value).toBe("f");
  expect(traced.route.source).toEqual(new SourceRef("run", 1, ["xs", 0, "uri"]));
});

test("a traced item read of another file field has no source route", () => {
  const item = new ResolvedValue(REF, true, new SourceRef("run", 1, ["xs", 0]));

  const traced = evalExpr(parseExpr("item.other"), () => new ResolvedValue(null, false), { item, trace: true });

  expect(traced.value).toBeNull();
  expect(traced.route.source).toBeNull();
});

describe("FileRef", () => {
  test("defaults mime and name to null and is frozen", () => {
    const file = new FileRef({ uri: "s3://b/k" });
    expect([file.uri, file.mime, file.name]).toEqual(["s3://b/k", null, null]);
    expect(Object.isFrozen(file)).toBe(true);
  });

  test.each<[string, unknown]>([
    ["a missing uri", {}],
    ["a non-text uri", { uri: 1n }],
    ["a null uri", { uri: null }],
    ["a non-text mime", { uri: "u", mime: 1n }],
    ["a non-text name", { uri: "u", name: false }],
    ["a uri with a lone surrogate", { uri: "\ud800" }],
    ["a name with a lone surrogate", { uri: "u", name: "a\udc00" }],
  ])("refuses %s", (_id, fields) => {
    expect(() => new FileRef(fields as never)).toThrow(RangeError);
  });

  test("names the field in its message", () => {
    expect(() => new FileRef({ uri: 1n } as never)).toThrow("file uri must be text, got 1");
    expect(() => new FileRef({ uri: "u", name: "a\udc00" })).toThrow(
      "file name holds a lone surrogate",
    );
  });
});

// --- committed values and trace reads decode a file through the target's type --- //

test.each([false, true])("a map body reads its file input as a file: learnable %p", async (learnable) => {
  const output = await okRun(
    `\
id: p
name: p
defs:
  body:
    input:
      f: !file
    nodes:
      c:
        kind: code
        input:
          x: \${input.f}
        code: |
          return inputs['x'].constructor.name
    output: \${c.output}
nodes:
  a:
    kind: code
    output: list(file)
    code: |
      return [${F}, ${F}]
  m:
    kind: map
    over: \${a.output}
    call: body
    input:
      f: \${item}
output: \${m.output}
`,
    { learnable },
  );

  expect(output).toEqual(["FileRef", "FileRef"]);
});

test.each([false, true])("a call body reads its list of files input as files: learnable %p", async (learnable) => {
  const output = await okRun(
    `\
id: p
name: p
defs:
  body:
    input:
      fs: !list(file)
    nodes:
      c:
        kind: code
        input:
          fs: \${input.fs}
        code: |
          return inputs['fs'].map((x) => x.constructor.name)
    output: \${c.output}
nodes:
  a:
    kind: code
    output: list(file)
    code: |
      return [${F}]
  k:
    kind: call
    call: body
    input:
      fs: \${a.output}
output: \${k.output}
`,
    { learnable },
  );

  expect(output).toEqual(["FileRef"]);
});

test.each([false, true])("the parent reads a file a call body returns: learnable %p", async (learnable) => {
  const output = await okRun(
    `\
id: p
name: p
defs:
  body:
    nodes:
      c:
        kind: code
        output:
          f: file
        code: |
          return { f: ${F} }
    output: \${c.output}
nodes:
  k:
    kind: call
    call: body
  d:
    kind: code
    input:
      x: \${k.output.f}
    code: |
      return inputs['x'].constructor.name
output: \${d.output}
`,
    { learnable },
  );

  expect(output).toBe("FileRef");
});

test.each([false, true])(
  "the parent reads the typed file of a partial record body output: learnable %p",
  async (learnable) => {
    const output = await okRun(
      `\
id: p
name: p
defs:
  body:
    nodes:
      c:
        kind: code
        code: |
          return { f: ${F}, g: ${F} }
    output:
      f: !file \${c.output.f}
      g: \${c.output.g}
nodes:
  k:
    kind: call
    call: body
  d:
    kind: code
    input:
      f: \${k.output.f}
      g: \${k.output.g}
    code: |
      return [inputs['f'].constructor.name, inputs['g']]
output: \${d.output}
`,
      { learnable },
    );

    expect(output).toEqual(["FileRef", form()]);
  },
);

const LOOP_FLOW = `\
id: p
name: p
defs:
  body:
    input:
      f: !file
      n: !int
    nodes:
      c:
        kind: code
        input:
          f: \${input.f}
          n: \${input.n}
        output:
          f: file
          n: int
        code: |
          return { f: inputs['f'], n: inputs['n'] + 1 }
    output: \${c.output}
nodes:
  a:
    kind: code
    output: file
    code: |
      return ${F}
  l:
    kind: loop
    times: 2
    call: body
    input:
      f: \${a.output}
      n: 0
  d:
    kind: code
    input:
      x: \${l.output.f}
      n: \${l.output.n}
    code: |
      return [inputs['x'].constructor.name, inputs['n']]
output: \${d.output}
`;

test.each([false, true])("the parent reads a file a loop carries: learnable %p", async (learnable) => {
  expect(await okRun(LOOP_FLOW, { learnable })).toEqual(["FileRef", 2n]);
});

/** A `times` loop `lp` over typed `f: file` and `n: int` whose body returns only `n`. */
function partialFileLoop(): LoopNode {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "times",
    times: 1,
    maxIters: 1,
    returnedFields: ["n"],
    childInputs: [
      new Parameter("f", { contract: new ValueContract(FILE), required: true }),
      new Parameter("n", { contract: new ValueContract(Type.scalar(ValueKind.INTEGER)), required: true }),
    ],
  });
  node.params = [new Parameter("f"), new Parameter("n")];
  return node;
}

test("a partial loop keeps an omitted file as a file", () => {
  const clone = partialFileLoop().respawn(1, { f: form(), n: 1n });

  const byName = new Map(clone.params!.map((param) => [param.name, param]));
  expect(byName.get("f")!.default).toEqual(REF);
  expect(Object.getPrototypeOf(byName.get("f")!.default)).toBe(FileRef.prototype);
  expect(byName.get("n")!.hasDefault).toBe(false);
});

const BODY_OUTPUTS: Record<string, string> = {
  whole: "${c.output}",
  typed: "\n      g: !file ${c.output.g}\n      n: !int ${c.output.n}",
};

/** Each `[learnable, body output name]` pair. */
const LEARNABLE_BY_BODY_OUTPUT = Object.keys(BODY_OUTPUTS)
  .sort()
  .flatMap((body) => [false, true].map((learnable) => [learnable, body] as [boolean, string]));

test.each(LEARNABLE_BY_BODY_OUTPUT)(
  "a partial loop compares an omitted file with a returned one: learnable %p, body output %s",
  async (learnable, bodyOutput) => {
    // The body returns `g` and `n` but not `f`, so `f` is carried from the seed.
    const output = await okRun(
      `\
id: p
name: p
defs:
  body:
    input:
      f: !file
      n: !int
    nodes:
      c:
        kind: code
        input:
          f: \${input.f}
          n: \${input.n}
        output:
          g: file
          n: int
        code: |
          return { g: inputs['f'], n: inputs['n'] + 1 }
    output: ${BODY_OUTPUTS[bodyOutput]}
nodes:
  a:
    kind: code
    output: file
    code: |
      return ${F}
  l:
    kind: loop
    until: f == g
    max: 3
    call: body
    input:
      f: \${a.output}
      g: !file \${a.output}
      n: 0
output: \${l.output.n}
`,
      { learnable },
    );

    expect(output).toBe(1n);
  },
);

test.each(LEARNABLE_BY_BODY_OUTPUT)(
  "a partial loop compares a carried file the body does not take: learnable %p, body output %s",
  async (learnable, bodyOutput) => {
    // The untyped `h` makes the loop's contract a partial record. The body takes only `n`,
    // so `f` is carried from the seed with no body input to read it through.
    const output = await okRun(
      `\
id: p
name: p
defs:
  body:
    input:
      n: !int
    nodes:
      c:
        kind: code
        input:
          n: \${input.n}
        output:
          g: file
          n: int
        code: |
          return { g: ${F}, n: inputs['n'] + 1 }
    output: ${BODY_OUTPUTS[bodyOutput]}
nodes:
  a:
    kind: code
    output: file
    code: |
      return ${F}
  u:
    kind: code
    code: |
      return ${F}
  l:
    kind: loop
    until: f == g
    max: 3
    call: body
    input:
      f: \${a.output}
      g: !file \${a.output}
      h: \${u.output}
      n: 0
output: \${l.output.n}
`,
      { learnable },
    );

    expect(output).toBe(1n);
  },
);

test.each([false, true])(
  "a loop reads a file in a carried object as a mapping on every iteration: learnable %p",
  async (learnable) => {
    // The seed binds `f` from `r`'s live output, where `f.x` is a file; from the second
    // iteration `f` comes back through the body's `object` output. Both must read alike.
    const output = await okRun(
      `\
id: p
name: p
defs:
  body:
    input:
      f: !object
      n: !int
    nodes:
      c:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs['n'] + 1
    output:
      f: \${input.f}
      n: \${c.output}
nodes:
  r:
    kind: code
    output:
      x: file
    code: |
      return { x: ${F} }
  l:
    kind: loop
    while: isinstance(f.x, object) and n < 2
    max: 3
    call: body
    input:
      f: \${r.output}
      n: 0
output: \${l.output.n}
`,
      { learnable },
    );

    expect(output).toBe(2n);
  },
);

test("a source entry with a record reading type differs from one without it", () => {
  const files = new Type({ kind: ValueKind.OBJECT, fields: new Map([["f", FILE]]), required: new Set() });
  const entry = new SourceEntry(null, 0, "null", files);

  expect(sameSourceEntry(entry, new SourceEntry(null, 0, "null", files))).toBe(true);
  expect(sameSourceEntry(entry, new SourceEntry(null, 0, "null"))).toBe(false);
});
