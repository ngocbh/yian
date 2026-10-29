/**
 * The terminal scenarios of `terminal_both_runtimes.ts`, run on Bun and, with `YIAN_TEST_NODE`,
 * built on Node; and the pure prompt helpers (input labels, structured input parsing).
 */

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { afterEach, describe, expect, test } from "bun:test";

import {
  draftLabel,
  draftText,
  inputLabel,
  joinDrafts,
  type Key,
  LineEditor,
  parseStructuredInput,
  Terminal,
  textDraft,
} from "../../src/cli/terminal.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { AskUserHandler } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { SCENARIOS } from "./terminal_both_runtimes.ts";
import { testNode } from "../helpers/test_node.ts";

const BUILT = join(import.meta.dirname, "..", "..", "dist", "tests", "cli", "terminal_both_runtimes.js");

const NODE = testNode(BUILT);

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

/** A bracketed paste of `text`, as the terminal hands it to an editor: one key. */
const paste = (text: string): Key => ({
  name: undefined,
  ctrl: false,
  meta: false,
  str: text,
  paste: true,
});

describe("a draft with pasted tokens", () => {
  test("keeps its tokens through set, expanded by draftText and labelled by draftLabel", () => {
    const editor = new LineEditor({ columns: () => 60, hint: "" });
    editor.key({ name: "s", ctrl: false, meta: false, str: "see ", paste: false });
    editor.key(paste("a\nb\nc\nd"));
    const draft = editor.draft();
    expect(draftText(draft)).toBe("see a\nb\nc\nd");
    expect(draftLabel(draft)).toBe("see [Pasted text #1 +4 lines]");
    expect(editor.text()).toBe("see a\nb\nc\nd");
    editor.set("");
    expect(editor.draft()).toEqual([]);
    editor.set(draft);
    expect(editor.draft()).toEqual(draft);
    // Backspace takes the token whole: it is still a token after the round trip.
    editor.key({ name: "backspace", ctrl: false, meta: false, str: "\x7f", paste: false });
    expect(editor.text()).toBe("see ");
  });

  test("set keeps a text as it is; textDraft makes a long one a token", () => {
    const editor = new LineEditor({ columns: () => 60, hint: "" });
    editor.set("1\n2\n3\n4");
    expect(editor.draft()).toEqual(["1\n2\n3\n4"]);
    expect(draftLabel(textDraft("1\n2\n3\n4"))).toBe("[Pasted text #1 +4 lines]");
    expect(textDraft("1\n2\n3")).toEqual(["1\n2\n3"]);
    expect(draftLabel(textDraft("w".repeat(801)))).toBe("[Pasted text #1 +1 line]");
  });

  test("a paste after a million characters typed still numbers its token", () => {
    const editor = new LineEditor({ columns: () => 60, hint: "" });
    const chunk = "a".repeat(800);
    for (let i = 0; i < 1300; i++) editor.key(paste(chunk));
    editor.key(paste("1\n2\n3\n4"));
    expect(draftLabel(editor.draft()).endsWith("a[Pasted text #1 +4 lines]")).toBe(true);
    editor.set(editor.draft());
    expect(editor.text().length).toBe(1300 * 800 + 7);
  });

  test("joinDrafts joins with spaces, skipping empty drafts, and numbers the tokens from 1", () => {
    const editor = new LineEditor({ columns: () => 60, hint: "" });
    editor.key(paste("1\n2\n3\n4"));
    const first = editor.draft();
    const joined = joinDrafts([first, [], first, ["tail"]]);
    expect(draftLabel(joined)).toBe("[Pasted text #1 +4 lines] [Pasted text #2 +4 lines] tail");
    expect(draftText(joined)).toBe("1\n2\n3\n4 1\n2\n3\n4 tail");
  });
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
