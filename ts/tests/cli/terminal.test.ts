/**
 * Ports the terminal checks of `tests/cli/test_terminal_broker.py` and the prompt checks of
 * `tests/engine/test_cli_prompt.py` that drive a real widget: the scenarios of
 * `terminal_both_runtimes.ts`, run here on Bun and, with `YIAN_TEST_NODE`, built on Node. Also
 * ports the pure prompt helpers' checks of `tests/engine/test_cli_prompt.py`
 * (`test_input_label_*`, `test_parse_structured_input_*`),
 * `tests/cli/test_run_long_ints.py::test_structured_prompt_rejects_a_long_int`, which TS turns
 * around: an int has no digit limit, and
 * `test_terminal_broker.py::test_terminal_module_imports_only_the_standard_library`.
 */

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { afterEach, describe, expect, test } from "bun:test";

import { inputLabel, parseStructuredInput, Terminal } from "../../src/cli/terminal.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { AskUserHandler } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { SCENARIOS } from "./terminal_both_runtimes.ts";

const BUILT = join(import.meta.dir, "..", "..", "dist", "tests", "cli", "terminal_both_runtimes.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set value that
 * can't run, a Node older than the `engines` floor, or a missing build fails the whole file, so a
 * typo can't pass as a run with the Node case skipped. A copy of `provider_auth.test.ts`'s:
 * importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

describe("the terminal scenarios", () => {
  for (const [name, run] of Object.entries(SCENARIOS)) test(name, run, 30_000);

  test.skipIf(!NODE)(
    "all pass on Node, built",
    () => {
      const result = spawnSync(NODE!, [BUILT], { encoding: "utf8", timeout: 120_000 });
      const context = `signal: ${result.signal}\nstderr: ${result.stderr}\nstdout: ${result.stdout}`;
      const lines = result.stdout
        .trim()
        .split("\n")
        .map((line) => {
          try {
            return JSON.parse(line) as { name?: unknown; ok?: unknown; error?: string };
          } catch {
            throw new Error(`not a JSON line: ${line}\n${context}`);
          }
        });
      expect(lines.map((line) => line.name)).toEqual(Object.keys(SCENARIOS));
      for (const line of lines) expect(line, line.error).toEqual({ name: line.name, ok: true });
      expect(result.status, context).toBe(0);
    },
    150_000,
  );
});

test("askUser is an ask_user handler", () => {
  const handler: AskUserHandler = new Terminal().askUser;
  expect(typeof handler).toBe("function");
});

test("terminal.ts imports only node: modules and relative yian modules", () => {
  const source = readFileSync(new URL("../../src/cli/terminal.ts", import.meta.url), "utf8");
  const specifiers = Array.from(
    source.matchAll(/^\s*(?:import|export)\s[^;]*?\sfrom\s+"([^"]+)"|^\s*import\s+"([^"]+)"/gm),
    (match) => match[1] ?? match[2]!,
  );
  expect(specifiers).toContain("node:readline");
  for (const specifier of specifiers) expect(specifier).toMatch(/^(node:|\.\.?\/)/);
  expect(source).not.toMatch(/\bimport\(|\brequire\(/);
});

/** One source-free flow parameter of type `type`. */
function parameter(
  name: string,
  type: Type,
  options: { required?: boolean; hasDefault?: boolean; default?: unknown } = {},
): Parameter {
  return new Parameter(name, { contract: new ValueContract(type), coerceStrings: true, ...options });
}

describe("inputLabel", () => {
  test("a required input", () => {
    expect(inputLabel(parameter("topic", Type.scalar(ValueKind.STRING), { required: true }))).toBe("topic (str) *");
  });

  test("an optional date without a default shows an example", () => {
    const asOf = parameter("as_of", new Type({ kind: ValueKind.DATE, nullable: true }));
    expect(inputLabel(asOf)).toBe("as_of (Optional(date)) [optional] e.g. 2026-05-21");
  });

  test("an input with a default", () => {
    const window = parameter("window", Type.scalar(ValueKind.INTEGER), { hasDefault: true, default: 30n });
    expect(inputLabel(window)).toBe("window (int) [default: 30]");
  });

  test("a datetime example", () => {
    const at = parameter("at", Type.scalar(ValueKind.DATETIME), { required: true });
    expect(inputLabel(at)).toBe("at (datetime) * e.g. 2026-05-21T14:30");
  });

  test("a default suppresses the example", () => {
    const asOf = parameter("as_of", new Type({ kind: ValueKind.DATE, nullable: true }), {
      hasDefault: true,
      default: "2026-01-01",
    });
    expect(inputLabel(asOf)).toBe("as_of (Optional(date)) [default: 2026-01-01]");
  });

  test("a structural type shows its canonical label", () => {
    const profile = parameter(
      "profile",
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([
          ["z", Type.scalar(ValueKind.INTEGER)],
          ["a", new Type({ kind: ValueKind.STRING, nullable: true })],
        ]),
        required: new Set(["z"]),
      }),
      { required: true },
    );
    expect(inputLabel(profile)).toBe('profile ({"a":Optional(str),"z":int}) *');
  });

  test("an explicit null default shows in the text form", () => {
    const maybe = parameter("maybe", new Type({ kind: ValueKind.INTEGER, nullable: true }), {
      hasDefault: true,
      default: null,
    });
    expect(inputLabel(maybe)).toBe("maybe (Optional(int)) [default: null]");
  });

  test("a flat list", () => {
    expect(inputLabel(parameter("items", Type.scalar(ValueKind.LIST_STRING), { required: true }))).toBe(
      "items (list(str)) *",
    );
  });

  test("an untyped input", () => {
    expect(inputLabel(new Parameter("name", { required: true }))).toBe("name *");
  });

  test("an infinite float default shows in the text form", () => {
    const x = parameter("x", Type.scalar(ValueKind.NUMBER), { hasDefault: true, default: Infinity });
    expect(inputLabel(x)).toBe("x (float) [default: Infinity]");
  });
});

/** A list of records whose required `count` field is an int. */
const RECORD_LIST = new Type({
  kind: ValueKind.LIST_OBJECT,
  element: new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["count", Type.scalar(ValueKind.INTEGER)]]),
    required: new Set(["count"]),
  }),
});

/** The message of the verdict `parseStructuredInput` gives for `raw`. */
function verdict(raw: string, type: Type): string {
  try {
    parseStructuredInput(raw, type, "x");
  } catch (error) {
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).name).toBe("StructuredInputError");
    return (error as Error).message;
  }
  throw new Error(`no verdict for ${raw}`);
}

describe("parseStructuredInput", () => {
  afterEach(() => install(BUILTIN));

  test("returns native arrays and objects, ints as bigint", () => {
    expect(parseStructuredInput("[1, 2]", Type.scalar(ValueKind.LIST_INTEGER), "x")).toEqual([1n, 2n]);
    expect(parseStructuredInput('{"count": 2}', RECORD_LIST.element!, "x")).toEqual({ count: 2n });
  });

  test("reports concise value failures", () => {
    expect(verdict("[", Type.scalar(ValueKind.LIST_ANY))).toBe("Invalid JSON at line 1, column 2: Expecting value");
    expect(verdict("{}", Type.scalar(ValueKind.LIST_ANY))).toBe("Expected a JSON array");
    expect(verdict("[]", Type.scalar(ValueKind.OBJECT))).toBe("Expected a JSON object");
    expect(verdict('["bad"]', Type.scalar(ValueKind.LIST_INTEGER))).toBe(
      '$[0]: list element "bad" is not int (declared list(int))',
    );
    expect(verdict('[{"count": "bad"}]', RECORD_LIST)).toBe('$[0]["count"]: "bad" does not match declared type int');
  });

  test("refuses null at a nullable structural root", () => {
    expect(verdict("null", new Type({ kind: ValueKind.LIST_ANY, nullable: true }))).toBe("Expected a JSON array");
  });

  test("accepts null inside a nullable element", () => {
    const type = new Type({
      kind: ValueKind.LIST_INTEGER,
      element: new Type({ kind: ValueKind.INTEGER, nullable: true }),
    });
    expect(parseStructuredInput("[null, 3]", type, "x")).toEqual([null, 3n]);
  });

  test("keeps a flat list(Any) as written", () => {
    const raw = '[1, "two", null, {"ok": true}, [3]]';
    expect(parseStructuredInput(raw, Type.scalar(ValueKind.LIST_ANY), "x")).toEqual([
      1n,
      "two",
      null,
      { ok: true },
      [3n],
    ]);
  });

  test("reads the tagged infinity form and refuses bare non-finite tokens", () => {
    const raw = '[{"$float": "inf"}, {"$float": "-inf"}]';
    expect(parseStructuredInput(raw, Type.scalar(ValueKind.LIST_NUMBER), "x")).toEqual([Infinity, -Infinity]);
    expect(verdict("[Infinity]", Type.scalar(ValueKind.LIST_NUMBER))).toBe(
      "Invalid JSON at line 1, column 2: Infinity is not JSON",
    );
    expect(verdict("[NaN]", Type.scalar(ValueKind.LIST_NUMBER))).toBe(
      "Invalid JSON at line 1, column 2: NaN is not JSON",
    );
  });

  test("refuses a value nesting too deep", () => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    expect(verdict("[[[[]]]]", Type.scalar(ValueKind.LIST_ANY))).toBe(
      "input 'x' at [0][0][0]: value nests deeper than 3 levels, " +
        "the limit for a yian value (each record and list is a level)",
    );
  });

  test("accepts an int of any length", () => {
    const digits = "9".repeat(5000);
    expect(parseStructuredInput(`[1, ${digits}]`, Type.scalar(ValueKind.LIST_INTEGER), "xs")).toEqual([
      1n,
      BigInt(digits),
    ]);
  });

  test("refuses a type that is neither a list nor an object", () => {
    expect(() => parseStructuredInput("1", Type.scalar(ValueKind.INTEGER), "x")).toThrow(
      new TypeError("structured prompt type must be a list or object"),
    );
  });
});
