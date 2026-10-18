/**
 * The defaults table, `BUILTIN`, parsing and errors, a port of `tests/engine/test_defaults.py`
 * and of `tests/test_defaults_docs.py`. Not ported:
 * `test_int_digits_cannot_exceed_the_interpreter_limit` and
 * `test_int_digits_has_no_upper_bound_without_an_interpreter_limit` (TS has no int digit
 * limit, so no `MAX_INT_DIGITS` entry), `test_memory_json_nesting_follows_value_nesting` (TS has
 * no memory JSON nesting bound), and `test_set_defaults_fixture` and
 * `test_every_test_starts_from_builtin_values` (pytest fixtures; TS tests reset with
 * `forget` or `install(BUILTIN)` instead).
 */

import { readFileSync } from "node:fs";

import { describe, expect, test } from "bun:test";

import { BUILTIN, type Defaults, DefaultsError, ENTRIES, type Entry, read } from "../../src/defaults.ts";

const TS_ONLY = [
  "MODEL_RETRIES",
  "MODEL_FIRST_RESPONSE_TIMEOUT_SECONDS",
  "MODEL_IDLE_TIMEOUT_SECONDS",
  "REQUEST_TIMEOUT_SECONDS",
  "MAX_MCP_MESSAGE_BYTES",
];

/** Return the entry named `name`. */
function entryNamed(name: string): Entry {
  const found = ENTRIES.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`no entry ${name}`);
  return found;
}

/** Return the camelCase `Defaults` key of an upper snake case entry name. */
function camel(name: string): string {
  const [head = "", ...rest] = name.toLowerCase().split("_");
  return head + rest.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join("");
}

/** Return the values read from an environment holding only `values`. */
function readOnly(values: Readonly<Record<string, string>>): Defaults {
  const env = (name: string) => (Object.hasOwn(values, name) ? values[name] : undefined);
  return read(env);
}

/** Return whether `entry`'s variable accepts the text `value`. */
function accepts(entry: Entry, value: string): boolean {
  try {
    readOnly({ [entry.variable]: value });
  } catch (error) {
    if (error instanceof DefaultsError) return false;
    throw error;
  }
  return true;
}

/** Return the `DefaultsError` reading `variable=value` throws. */
function readError(variable: string, value: string): DefaultsError {
  try {
    readOnly({ [variable]: value });
  } catch (error) {
    expect(error).toBeInstanceOf(DefaultsError);
    return error as DefaultsError;
  }
  throw new Error(`${variable}=${value} was accepted`);
}

describe("the table", () => {
  test("entries match BUILTIN", () => {
    const names: readonly string[] = ENTRIES.map((entry) => entry.name);
    expect(new Set(names).size).toBe(names.length);
    expect(Object.keys(BUILTIN)).toEqual(names.map(camel));
    const builtin: Record<string, number> = BUILTIN;
    for (const entry of ENTRIES) {
      expect([entry.name, builtin[camel(entry.name)]]).toEqual([entry.name, entry.default]);
      expect(["limit", "setting", "input cap"]).toContain(entry.group);
      expect(entry.description).not.toBe("");
      expect(entry.description).not.toContain("\n");
      expect(Object.isFrozen(entry)).toBe(true);
      if (entry.default === entry.noLimit) continue;
      expect(entry.default).toBeGreaterThanOrEqual(entry.minimum);
      if (entry.maximum !== null) expect(entry.default).toBeLessThanOrEqual(entry.maximum);
    }
    expect(Object.isFrozen(BUILTIN)).toBe(true);
    expect(Object.isFrozen(ENTRIES)).toBe(true);
  });

  test("the TS entry set", () => {
    const names: readonly string[] = ENTRIES.map((entry) => entry.name);
    expect(names).not.toContain("MAX_INT_DIGITS");
    expect(Object.keys(BUILTIN)).not.toContain("memoryJsonNesting");
    for (const name of names) {
      expect(name).not.toContain("RESTART");
      expect(name).not.toContain("DIGITS");
    }
    const added = TS_ONLY.map((name) => {
      const { default: value, group, minimum, maximum, noLimit } = entryNamed(name);
      return [name, value, group, minimum, maximum, noLimit];
    });
    expect(added).toEqual([
      ["MODEL_RETRIES", 6, "setting", 0, null, null],
      ["MODEL_FIRST_RESPONSE_TIMEOUT_SECONDS", 240, "setting", 1, null, null],
      ["MODEL_IDLE_TIMEOUT_SECONDS", 240, "setting", 1, null, null],
      ["REQUEST_TIMEOUT_SECONDS", 30, "setting", 1, null, null],
      ["MAX_MCP_MESSAGE_BYTES", 16777216, "input cap", 1, null, null],
    ]);
    // The four settings follow MCP_CALL_TIMEOUT_SECONDS; the cap follows MAX_MODEL_LIST_BYTES.
    const at = (name: string) => names.indexOf(name);
    expect(names.slice(at("MCP_CALL_TIMEOUT_SECONDS") + 1, at("MCP_CALL_TIMEOUT_SECONDS") + 5)).toEqual(
      TS_ONLY.slice(0, 4),
    );
    expect(names[at("MAX_MODEL_LIST_BYTES") + 1]).toBe("MAX_MCP_MESSAGE_BYTES");
  });

  test("variable names are prefixed and do not collide", () => {
    const taken = ["YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL", "YIAN_AUTH_FILE"];
    const owners = ["MAX", "AGENT", "TOOL", "MCP", "MODEL", "REQUEST", "EPISODIC", "REFLECTIVE", "SESSION", "PROFILE"];
    for (const entry of ENTRIES) {
      expect(entry.name).toMatch(/^[A-Z]+(?:_[A-Z]+)+$/);
      expect(owners).toContain(entry.name.split("_")[0]!);
      expect(entry.variable).toBe("YIAN_" + entry.name);
      expect(taken).not.toContain(entry.variable);
      expect(entry.name.startsWith("DEFAULT_")).toBe(false);
    }
  });

  test("reflective budgets have a floor of 256", () => {
    for (const name of ["REFLECTIVE_MAX_LESSON_CHARS", "REFLECTIVE_MAX_EPISODE_CHARS", "REFLECTIVE_MAX_LIMIT_CHARS"]) {
      expect(entryNamed(name).minimum).toBe(256);
    }
  });
});

describe("read", () => {
  test("no variables is BUILTIN", () => {
    expect(read(() => undefined)).toEqual(BUILTIN);
  });

  test("an override takes effect", () => {
    expect(readOnly({ YIAN_MAX_VALUE_NESTING: "80" }).maxValueNesting).toBe(80);
  });

  test("an empty value counts as unset", () => {
    expect(readOnly({ YIAN_MAX_VALUE_NESTING: "" })).toEqual(BUILTIN);
  });

  const advice = "use an integer from 1 to 100, or unset it";
  test.each([
    ["abc", "not an integer"],
    ["1.5", "not an integer"],
    ["1e3", "not an integer"],
    [" 5", "not an integer"],
    ["1_000", "not an integer"],
    ["+5", "not an integer"],
    ["-0", "not an integer"],
    ["007", "not an integer"],
    ["٣", "not an integer"],
    ["5\n", "not an integer"],
    ["0", "out of range"],
    ["101", "out of range"],
    ["9".repeat(5000), "out of range"],
  ])("a bad value names the variable: %j", (value, problem) => {
    const error = readError("YIAN_MAX_VALUE_NESTING", value);
    const shown = value.length <= 40 ? value : value.slice(0, 37) + "...";
    expect(error.message).toBe(`YIAN_MAX_VALUE_NESTING=${JSON.stringify(shown)} is ${problem}; ${advice}`);
    expect(error.message.length).toBeLessThan(300);
    expect(error).toBeInstanceOf(RangeError);
    expect(error.name).toBe("DefaultsError");
  });

  test("a 5000-digit value shows 37 nines then ...", () => {
    const error = readError("YIAN_MAX_VALUE_NESTING", "9".repeat(5000));
    expect(error.message).toStartWith(`YIAN_MAX_VALUE_NESTING="${"9".repeat(37)}..." is out of range`);
  });

  test("a lone surrogate is escaped", () => {
    expect(readError("YIAN_MAX_VALUE_NESTING", "\uD800").message).toBe(
      `YIAN_MAX_VALUE_NESTING="\\ud800" is not an integer; ${advice}`,
    );
  });

  test("a long value is cut by code points", () => {
    const value = "😀".repeat(41);
    const error = readError("YIAN_MAX_VALUE_NESTING", value);
    expect(error.message).toBe(
      `YIAN_MAX_VALUE_NESTING=${JSON.stringify("😀".repeat(37) + "...")} is not an integer; ${advice}`,
    );
    const forty = "😀".repeat(40);
    expect(readError("YIAN_MAX_VALUE_NESTING", forty).message).toContain(JSON.stringify(forty));
  });

  test.each(ENTRIES.map((entry) => [entry.name, entry] as const))("each minimum and maximum: %s", (_, entry) => {
    expect(accepts(entry, String(entry.minimum))).toBe(true);
    if (entry.minimum - 1 !== entry.noLimit) expect(accepts(entry, String(entry.minimum - 1))).toBe(false);
    if (entry.maximum !== null) {
      expect(accepts(entry, String(entry.maximum))).toBe(true);
      expect(accepts(entry, String(entry.maximum + 1))).toBe(false);
    } else {
      expect(accepts(entry, "9007199254740991")).toBe(true);
      expect(accepts(entry, "9007199254740992")).toBe(false);
    }
  });

  test("an unbounded entry stops at Number.MAX_SAFE_INTEGER", () => {
    expect(readOnly({ YIAN_MAX_TOTAL_NODES: "9007199254740991" }).maxTotalNodes).toBe(Number.MAX_SAFE_INTEGER);
    for (const value of ["9007199254740992", "999999999999999999", "1234567890123456789"]) {
      expect(readError("YIAN_MAX_TOTAL_NODES", value).message).toBe(
        `YIAN_MAX_TOTAL_NODES="${value}" is out of range; use an integer from 1 to 9007199254740991, or unset it`,
      );
    }
  });

  test("the no-limit sentinel", () => {
    const entry = entryNamed("AGENT_MAX_TOOL_ITERATIONS");
    expect(accepts(entry, "-1")).toBe(true);
    expect(accepts(entry, "5")).toBe(true);
    expect(accepts(entry, "0")).toBe(false);
    expect(accepts(entry, "-2")).toBe(false);
    expect(readError(entry.variable, "0").message).toEndWith(", or -1 for no cap, or unset it");
  });
});

describe("docs/defaults.md", () => {
  const DOC = readFileSync(new URL("../../../docs/defaults.md", import.meta.url), "utf8");

  /** Return the stripped cells of one Markdown table row. */
  function cells(line: string): string[] {
    return line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  }

  /** Return the body rows of the one table in `text` whose first header cell is `first`. */
  function tableRows(first: string, text = DOC): string[][] {
    const header = [first, "Group", "Default", "Valid values", "Meaning"];
    const lines = text.split("\n");
    const starts = lines.flatMap((line, i) =>
      line.startsWith("|") && JSON.stringify(cells(line)) === JSON.stringify(header) ? [i] : [],
    );
    expect(starts.length).toBe(1);
    const rows: string[][] = [];
    for (const line of lines.slice(starts[0]! + 2)) {
      if (!line.startsWith("|")) break;
      rows.push(cells(line));
    }
    return rows;
  }

  /** Return `n` with a comma between each group of three digits, as Python's `{n:,}` does. */
  function grouped(n: number): string {
    return String(n).replace(/\B(?=(?:\d{3})+$)/g, ",");
  }

  /** Return the table row `entry` should have. */
  function expectedRow(entry: Entry): string[] {
    let valid =
      entry.maximum === null
        ? `${grouped(entry.minimum)} or more`
        : `${grouped(entry.minimum)} to ${grouped(entry.maximum)}`;
    if (entry.maximum !== null && entry.maximum === entry.default) valid += " (can only be lowered)";
    if (entry.noLimit !== null) valid += `, or ${entry.noLimit} for no cap`;
    return [`\`${entry.variable}\``, entry.group, grouped(entry.default), valid, entry.description];
  }

  test("the table parser finds rows", () => {
    const text = [
      "| Variable | Group | Default | Valid values | Meaning |",
      "| --- | --- | --- | --- | --- |",
      "| `YIAN_X` | limit | 1 | 1 or more | X |",
      "",
      "| not | the table |",
    ].join("\n");
    expect(tableRows("Variable", text)).toEqual([["`YIAN_X`", "limit", "1", "1 or more", "X"]]);
  });

  test("the cell format", () => {
    expect([grouped(-1), grouped(256), grouped(1024), grouped(16777216)]).toEqual(["-1", "256", "1,024", "16,777,216"]);
  });

  test("the Python table lists the shared entries in order", () => {
    const rows = tableRows("Variable");
    const skipped = rows.findIndex((row) => row[0] === "`YIAN_MAX_INT_DIGITS`");
    expect(rows[skipped - 1]?.[0]).toBe("`YIAN_MAX_SKIP_WORLDS`");
    rows.splice(skipped, 1);
    expect(rows).toEqual(ENTRIES.filter((entry) => !TS_ONLY.includes(entry.name)).map(expectedRow));
  });

  test("the TS yian table lists the TS-only entries in order", () => {
    expect(tableRows("TS yian variable")).toEqual(TS_ONLY.map((name) => expectedRow(entryNamed(name))));
  });
});
