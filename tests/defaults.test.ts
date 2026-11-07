/**
 * The defaults table: its entries, `BUILTIN`, parsing and errors, the process values, recorded
 * values and resume checks, the per-call values `withEnv` scopes, and the Reference page's table.
 */

import { readFileSync } from "node:fs";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import {
  BUILTIN,
  checkRecorded,
  current,
  type Defaults,
  DefaultsError,
  DefaultsMismatchError,
  ENTRIES,
  type Entry,
  type Env,
  forget,
  fromRecorded,
  install,
  nonDefaults,
  processEnv,
  read,
  reload,
  withEnv,
} from "../src/defaults.ts";

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

/** Return an environment holding only `vars`. */
function envOf(vars: Readonly<Record<string, string>>): Env {
  return (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined);
}

/** Return the values read from an environment holding only `values`. */
function readOnly(values: Readonly<Record<string, string>>): Defaults {
  return read(envOf(values));
}

/** Return the `DefaultsMismatchError` `run` throws. */
function mismatch(run: () => unknown): DefaultsMismatchError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(DefaultsMismatchError);
    return error as DefaultsMismatchError;
  }
  throw new Error("no DefaultsMismatchError");
}

const UNKNOWN = "this run started with YIAN_NOT_AN_ENTRY, which this version of yian does not know";

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

  test("variable names are prefixed and do not collide", () => {
    const taken = ["YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL", "YIAN_AUTH_FILE"];
    const owners = ["MAX", "AGENT", "TOOL", "MCP", "MODEL", "REQUEST", "SESSION"];
    for (const entry of ENTRIES) {
      expect(entry.name).toMatch(/^[A-Z]+(?:_[A-Z]+)+$/);
      expect(owners).toContain(entry.name.split("_")[0]!);
      expect(entry.variable).toBe("YIAN_" + entry.name);
      expect(taken).not.toContain(entry.variable);
      expect(entry.name.startsWith("DEFAULT_")).toBe(false);
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

  test("processEnv reads own properties only", () => {
    expect(processEnv("__proto__")).toBeUndefined();
    expect(processEnv("toString")).toBeUndefined();
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

describe("the Reference page's defaults table", () => {
  const DOC = readFileSync(new URL("../docs/src/content/docs/reference.md", import.meta.url), "utf8");

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

  /** Return `n` with a comma between each group of three digits. */
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

  test("the table lists every entry in order", () => {
    expect(tableRows("Variable")).toEqual(ENTRIES.map(expectedRow));
  });

  // Every read yian makes goes through an injected environment accessor; each later reader of
  // the environment adds its own reads, and the table documenting them, to this check.
  test("read asks for each entry's variable once, in table order, and only documented ones", () => {
    const asked: string[] = [];
    read((name) => {
      asked.push(name);
      return undefined;
    });
    expect(asked).toEqual(ENTRIES.map((entry) => entry.variable));
    const documented = tableRows("Variable").map((row) => row[0]!.replaceAll("`", ""));
    expect(asked.filter((name) => !documented.includes(name))).toEqual([]);
  });
});

describe("recorded values", () => {
  test("nonDefaults lists only changed values as bigints, sorted by name", () => {
    const values = readOnly({
      YIAN_MAX_VALUE_NESTING: "80",
      YIAN_AGENT_RETRIES: "5",
      YIAN_MAX_TOTAL_NODES: "10000",
    });
    const changed = nonDefaults(values);
    expect(changed).toEqual({ AGENT_RETRIES: 5n, MAX_VALUE_NESTING: 80n });
    expect(Object.keys(changed)).toEqual(["AGENT_RETRIES", "MAX_VALUE_NESTING"]);
    expect(Object.values(changed).map((value) => typeof value)).toEqual(["bigint", "bigint"]);
    expect(nonDefaults(BUILTIN)).toEqual({});
  });

  test("fromRecorded overlays built-ins", () => {
    expect(fromRecorded({ MAX_TOTAL_NODES: 5000n })).toEqual({ ...BUILTIN, maxTotalNodes: 5000 });
    expect(fromRecorded({ MAX_TOTAL_NODES: 5000 })).toEqual({ ...BUILTIN, maxTotalNodes: 5000 });
    expect(fromRecorded({})).toEqual(BUILTIN);
    expect(Object.isFrozen(fromRecorded({}))).toBe(true);
    expect(mismatch(() => fromRecorded({ NOT_AN_ENTRY: 1n })).message).toBe(UNKNOWN);
  });

  test("a recorded __proto__ is an unknown name", () => {
    const recorded = JSON.parse('{"__proto__": 1, "MAX_TOTAL_NODES": 5000}') as Record<string, number>;
    expect(mismatch(() => fromRecorded(recorded)).message).toBe(
      "this run started with YIAN___proto__, which this version of yian does not know",
    );
  });
});

describe("the process values", () => {
  beforeEach(() => install(BUILTIN));
  afterEach(forget);

  const UNSET_TOTAL =
    "YIAN_MAX_TOTAL_NODES is 20000 here, but this run started with the default 10000; " +
    "unset YIAN_MAX_TOTAL_NODES to resume it";

  test("current() reads once", () => {
    expect(current()).toEqual(BUILTIN);
    expect(reload(envOf({ YIAN_MAX_VALUE_NESTING: "80" })).maxValueNesting).toBe(80);
    expect(current().maxValueNesting).toBe(80);
    expect(current()).toBe(current());
    forget();
    expect(current()).toEqual(read(processEnv));
  });

  test("current() reads the process environment once, and a bad value on every call", () => {
    const saved = process.env["YIAN_MAX_VALUE_NESTING"];
    try {
      process.env["YIAN_MAX_VALUE_NESTING"] = "80";
      forget();
      expect(current().maxValueNesting).toBe(80);
      process.env["YIAN_MAX_VALUE_NESTING"] = "70";
      expect(current().maxValueNesting).toBe(80);
      process.env["YIAN_MAX_VALUE_NESTING"] = "abc";
      forget();
      for (let i = 0; i < 2; i++) expect(() => current()).toThrow(DefaultsError);
    } finally {
      if (saved === undefined) delete process.env["YIAN_MAX_VALUE_NESTING"];
      else process.env["YIAN_MAX_VALUE_NESTING"] = saved;
    }
  });

  test("a failed read is not kept", () => {
    const bad = envOf({ YIAN_MAX_VALUE_NESTING: "abc" });
    const kept = reload(envOf({ YIAN_MAX_VALUE_NESTING: "80" }));
    for (let i = 0; i < 2; i++) {
      expect(() => reload(bad)).toThrow('YIAN_MAX_VALUE_NESTING="abc" is not an integer');
    }
    expect(current()).not.toBe(kept);
    expect(current()).toEqual(read(processEnv));
    expect(reload(envOf({ YIAN_MAX_VALUE_NESTING: "70" })).maxValueNesting).toBe(70);
  });

  test("install skips the range check and keeps a frozen copy", () => {
    const values = { ...BUILTIN, maxValueNesting: 200_000 };
    install(values);
    Object.assign(values, { maxValueNesting: 1 });
    expect(current().maxValueNesting).toBe(200_000);
    expect(current().maxTotalNodes).toBe(10_000);
    expect(Object.isFrozen(current())).toBe(true);
  });

  test("checkRecorded", () => {
    checkRecorded({});
    checkRecorded({ MAX_TOTAL_NODES: 10_000n });

    reload(envOf({ YIAN_MAX_TOTAL_NODES: "20000" }));
    checkRecorded({ MAX_TOTAL_NODES: 20_000n });
    checkRecorded({ MAX_TOTAL_NODES: 20_000 });
    const error = mismatch(() => checkRecorded({ MAX_TOTAL_NODES: 5000n }));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.name).toBe("DefaultsMismatchError");
    expect(error.message).toBe(
      "YIAN_MAX_TOTAL_NODES is 20000 here, but this run started with 5000; set YIAN_MAX_TOTAL_NODES=5000 to resume it",
    );
    expect(mismatch(() => checkRecorded({})).message).toBe(UNSET_TOTAL);

    reload(envOf({ YIAN_MAX_TOTAL_NODES: "20000", YIAN_AGENT_RETRIES: "4" }));
    expect(mismatch(() => checkRecorded({})).message.split("\n")).toEqual([
      UNSET_TOTAL,
      "YIAN_AGENT_RETRIES is 4 here, but this run started with the default 2; unset YIAN_AGENT_RETRIES to resume it",
    ]);

    expect(mismatch(() => checkRecorded({ NOT_AN_ENTRY: 1n })).message).toContain("YIAN_NOT_AN_ENTRY");
  });

  test("checkRecorded reports unknown names and differences together", () => {
    reload(envOf({ YIAN_MAX_TOTAL_NODES: "20000" }));
    const error = mismatch(() => checkRecorded({ NOT_AN_ENTRY: 1n, AGENT_RETRIES: 0n, ANOTHER: 1n }));
    expect(error.message.split("\n")).toEqual([
      "this run started with YIAN_ANOTHER, which this version of yian does not know",
      UNKNOWN,
      UNSET_TOTAL,
      "YIAN_AGENT_RETRIES is 2 here, but this run started with 0; set YIAN_AGENT_RETRIES=0 to resume it",
    ]);
    expect(error.message).not.toContain("resumed here");
  });

  test("checkRecorded reports an out-of-range value as invalid", () => {
    const message = mismatch(() => checkRecorded({ MAX_VALUE_NESTING: 150n })).message;
    expect(message).toBe(
      "this run recorded an invalid value YIAN_MAX_VALUE_NESTING=150; " +
        "YIAN_MAX_VALUE_NESTING accepts an integer from 1 to 100",
    );
    expect(message).not.toContain("set YIAN_MAX_VALUE_NESTING=150");
  });

  test("checkRecorded prints a recorded value as recorded", () => {
    expect(mismatch(() => checkRecorded({ MAX_TOTAL_NODES: 10n ** 21n })).message).toBe(
      "this run recorded an invalid value YIAN_MAX_TOTAL_NODES=1000000000000000000000; " +
        "YIAN_MAX_TOTAL_NODES accepts an integer from 1 to 9007199254740991",
    );
    expect(mismatch(() => checkRecorded({ AGENT_RETRIES: 1.5 })).message).toBe(
      "this run recorded an invalid value YIAN_AGENT_RETRIES=1.5; " +
        "YIAN_AGENT_RETRIES accepts an integer from 0 to 9007199254740991",
    );
  });

  test("checkRecorded accepts an out-of-range value in effect", () => {
    install({ ...BUILTIN, maxValueNesting: 150 });
    checkRecorded({ MAX_VALUE_NESTING: 150n });
  });
});

describe("per-call values", () => {
  beforeEach(() => install(BUILTIN));
  afterEach(forget);

  const AT_80 = envOf({ YIAN_MAX_VALUE_NESTING: "80" });

  test("withEnv scopes the values env sets to fn", async () => {
    expect(withEnv(AT_80, () => current().maxValueNesting)).toBe(80);
    expect(current()).toEqual(BUILTIN);
    const later = await withEnv(AT_80, async () => {
      await Promise.resolve();
      return current().maxValueNesting;
    });
    expect(later).toBe(80);
    expect(current()).toEqual(BUILTIN);
  });

  test("processEnv keeps the values in effect", () => {
    install({ ...BUILTIN, maxValueNesting: 90 });
    expect(withEnv(processEnv, () => current()).maxValueNesting).toBe(90);
    expect(withEnv(AT_80, () => withEnv(processEnv, () => current().maxValueNesting))).toBe(80);
  });

  test("reload inside a scope sets the process values only", () => {
    withEnv(AT_80, () => {
      expect(reload(envOf({ YIAN_MAX_VALUE_NESTING: "70" })).maxValueNesting).toBe(70);
      expect(current().maxValueNesting).toBe(80);
    });
    expect(current().maxValueNesting).toBe(70);
  });

  test("a bad value throws before fn runs", () => {
    let called = false;
    const fn = () => {
      called = true;
    };
    expect(() => withEnv(envOf({ YIAN_MAX_VALUE_NESTING: "0" }), fn)).toThrow(DefaultsError);
    expect(called).toBe(false);

    const saved = process.env["YIAN_MAX_VALUE_NESTING"];
    try {
      process.env["YIAN_MAX_VALUE_NESTING"] = "0";
      forget();
      expect(() => withEnv(processEnv, fn)).toThrow(DefaultsError);
      expect(called).toBe(false);
    } finally {
      if (saved === undefined) delete process.env["YIAN_MAX_VALUE_NESTING"];
      else process.env["YIAN_MAX_VALUE_NESTING"] = saved;
    }
  });
});
