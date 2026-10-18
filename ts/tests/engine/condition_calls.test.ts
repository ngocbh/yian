/**
 * A condition may call only a builtin, with arguments it takes, or a typedef constructor: the
 * `parseCondition` tests of `tests/engine/test_condition_calls.py`, and the builtin registry's
 * argument binder (`bindArguments`) and `registerTemplateFn`.
 *
 * Left for later: the `load_flow`/`run_flow` half (from `_load_error` on) lands with compose and
 * `run.ts`. `test_a_builtin_without_a_readable_signature_is_not_arity_checked` has no
 * counterpart: every registered function declares its `params`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import {
  bindArguments,
  registerTemplateFn,
  TEMPLATE_FNS,
  type TemplateFn,
  type TemplateParam,
} from "../../src/expr/builtins.ts";
import { ExpressionError, parseCondition } from "../../src/expr/expressions.ts";
import { typeFor } from "../../src/typesys/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in `name` taking `params` for this test only. */
function standIn(name: string, params: TemplateParam[]): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call: () => null });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

/** Register stand-ins for the shipped builtins, with the shipped parameters. */
function shippedStandIns(): void {
  standIn("upper", [{ name: "s" }]);
  standIn("lower", [{ name: "s" }]);
  standIn("render_as_json", [{ name: "value" }, { name: "indent", default: 2 }]);
  standIn("join", [{ name: "value" }, { name: "sep", default: "\n" }]);
}

/** Expect `parseCondition(condition, types)` to fail with exactly `message`. */
function expectMessage(condition: string, message: string, types: ReadonlyMap<string, unknown> | null = null): void {
  let error: unknown;
  try {
    parseCondition(condition, types);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

test("unknown function does not parse as a condition", () => {
  shippedStandIns();
  for (const condition of [
    "nosuchfn(n) > 0",
    "${nosuchfn(n) > 0}",
    "true or nosuchfn(n)",
    "upper(nosuchfn(n)) == 'A'",
  ]) {
    expectMessage(condition, "unknown expression function 'nosuchfn'", new Map());
  }
});

test("a constructor parses when its type is known or left to compose", () => {
  const buy = typeFor("Buy", readTypedefs({ Buy: { qty: "int" } }));
  parseCondition("Buy(d).qty > 0", new Map([["Buy", buy]]));
  // Without a type table, compose resolves the name against the flow's typedefs.
  parseCondition("Buy(d).qty > 0");
  expectMessage("Buy(d).qty > 0", "unknown expression function 'Buy'", new Map());
});

test("a builtin type name called in a condition gets the conversion hint", () => {
  expectMessage(
    "int(x) > 0",
    "unknown expression function 'int'; types in expressions are typedef names; `int(...)` is not a conversion",
    new Map(),
  );
});

test("builtin with arguments it does not take does not parse", () => {
  shippedStandIns();
  const cases: [string, string][] = [
    ["upper(a, b) == 'A'", "expression function 'upper' takes (s): too many positional arguments"],
    ["upper() == 'A'", "expression function 'upper' takes (s): missing a required argument: 's'"],
    [
      "render_as_json(a, width=4) != ''",
      "expression function 'render_as_json' takes (value, indent=2): got an unexpected keyword argument 'width'",
    ],
  ];
  for (const [condition, message] of cases) expectMessage(condition, message);
});

test("a default shows in its text form", () => {
  shippedStandIns();
  expectMessage(
    "join(a, b, c) == ''",
    'expression function \'join\' takes (value, sep="\\n"): too many positional arguments',
  );
});

test("builtin calls that bind parse", () => {
  shippedStandIns();
  for (const condition of [
    "upper(s) == 'A'",
    "render_as_json(a, indent=4) != ''",
    "join(xs) == 'a'",
    "join(xs, sep=',') == 'a'",
    "isinstance(x, int)",
    "lower(upper(s)) == 'a'",
  ]) {
    expect(() => parseCondition(condition)).not.toThrow();
  }
});

test("a registered builtin is accepted", () => {
  standIn("shout", [{ name: "s" }, { name: "times", default: 1 }]);
  parseCondition("shout(s, times=2) == 'AA'");
  expectMessage(
    "shout(s, 2, 3) == 'AA'",
    "expression function 'shout' takes (s, times=1): too many positional arguments",
  );
});

test("a repeated keyword does not parse", () => {
  shippedStandIns();
  expectMessage("upper(s=a, s=b) == 'A'", "upper(...) repeats the keyword 's'");
});

describe("bindArguments", () => {
  const fn: TemplateFn = {
    params: [{ name: "value" }, { name: "indent", default: 2n }],
    call: () => null,
  };
  const keywords = (entries: Record<string, string>) => new Map(Object.entries(entries));

  test("Python's Signature.bind messages", () => {
    const cases: [string[], Record<string, string>, string][] = [
      [["a", "b", "c"], {}, "too many positional arguments"],
      [["a"], { value: "v" }, "multiple values for argument 'value'"],
      [["a", "b", "c"], { value: "v" }, "multiple values for argument 'value'"],
      [[], {}, "missing a required argument: 'value'"],
      [[], { indent: "i", zz: "z" }, "missing a required argument: 'value'"],
      [[], { x: "x", value: "v" }, "got an unexpected keyword argument 'x'"],
    ];
    for (const [positional, named, message] of cases) {
      expect(bindArguments(fn, positional, keywords(named))).toBe(message);
    }
  });

  test("the first unknown keyword in call order is named", () => {
    expect(bindArguments(fn, ["a"], keywords({ zz: "z", yy: "y" }))).toBe("got an unexpected keyword argument 'zz'");
  });

  test("arguments come back in parameter order with defaults filled in", () => {
    expect(bindArguments(fn, ["a"], new Map())).toEqual(["a", 2n]);
    expect(bindArguments(fn, [], keywords({ indent: "i", value: "v" }))).toEqual(["v", "i"]);
    expect(bindArguments(fn, ["a", "b"], new Map())).toEqual(["a", "b"]);
  });

  test("a name is shown as Python's repr", () => {
    expect(bindArguments(fn, [], keywords({ "it's": "x", value: "v" }))).toBe(
      'got an unexpected keyword argument "it\'s"',
    );
  });
});

describe("registerTemplateFn", () => {
  test("the last writer wins", () => {
    standIn("shout", [{ name: "s" }]);
    standIn("shout", [{ name: "s" }, { name: "times", default: 1 }]);
    expect(TEMPLATE_FNS.get("shout")!.params.map((p) => p.name)).toEqual(["s", "times"]);
  });

  test("a default is read as a host value", () => {
    standIn("shout", [{ name: "s" }, { name: "times", default: 1 }]);
    expect(TEMPLATE_FNS.get("shout")!.params[1]!.default).toBe(1n);
  });

  test("a default that is no yian value is rejected", () => {
    saved.push(["shout", TEMPLATE_FNS.get("shout")]);
    const fn: TemplateFn = { params: [{ name: "times", default: Number.NaN }], call: () => null };
    expect(() => registerTemplateFn("shout", fn)).toThrow(
      "expression function 'shout', default of 'times': NaN is not a yian value",
    );
  });
});
