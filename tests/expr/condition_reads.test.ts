/**
 * `conditionReads` and `rewriteConditionReads`.
 *
 * `conditionReads` lists each reference occurrence of a parsed condition with the outermost
 * `:-` / `:?` / `|` sub-expression around it and whether that sub-expression co-skips as a
 * binding would. `rewriteConditionReads` renames reference leaves by occurrence, so the same path
 * read plainly and inside an escape can get different names. A comprehension's variables are
 * neither reads nor renamed; its iterables are.
 */

import { expect, test } from "bun:test";

import {
  type ConditionRead,
  conditionReads,
  parseCondition,
  type RefRenamed,
  rewriteConditionReads,
  rewriteConditionRefs,
} from "../../src/expr/index.ts";

/** Return each read of `when` as `[path, escape, coSkips]`. */
function reads(when: string): [string, number | null, boolean][] {
  return conditionReads(parseCondition(when)).map((r) => [r.path, r.escape, r.coSkips]);
}

test.each<[string, [string, number | null, boolean][]]>([
  ["a.output >= 1", [["a.output", null, false]]],
  ["(a.output :- 0) >= 1", [["a.output", 0, false]]],
  ["(a.output | 0) >= 1", [["a.output", 0, false]]],
  ['(a.output :? "m") >= 1', [["a.output", 0, false]]],
  [
    "(a.output | b.output) == 'x'",
    [
      ["a.output", 0, true],
      ["b.output", 0, true],
    ],
  ],
  [
    "(upper(a.output) :- b.output) == 'X'",
    [
      ["a.output", 0, true],
      ["b.output", 0, true],
    ],
  ],
  [
    "(a.output :? b.output) > 1",
    [
      ["a.output", 0, false],
      ["b.output", 0, false],
    ],
  ],
  [
    "((a.output :- b.output) | 0) > 1",
    [
      ["a.output", 0, false],
      ["b.output", 0, false],
    ],
  ],
  [
    "a.output > 5 or (a.output :- 0) < 1",
    [
      ["a.output", null, false],
      ["a.output", 0, false],
    ],
  ],
  [
    "(a.output :- 0) > 1 and (b.output | c.output) == 2",
    [
      ["a.output", 0, false],
      ["b.output", 1, true],
      ["c.output", 1, true],
    ],
  ],
  [
    "upper(a.output :- '') == 'X' and b.output",
    [
      ["a.output", 0, false],
      ["b.output", null, false],
    ],
  ],
  [
    "isinstance(d.output, Buy) and (d.output.qty :- 0) > 5",
    [
      ["d.output", null, false],
      ["d.output.qty", 0, false],
    ],
  ],
  [
    "x.output and a.output :- false",
    [
      ["x.output", 0, false],
      ["a.output", 0, false],
    ],
  ],
  ["${(a.output :- 0) >= 1}", [["a.output", 0, false]]],
  ["(1 :- 2) > a.output", [["a.output", null, false]]],
])("condition reads: %s", (when, expected) => {
  expect(reads(when)).toEqual(expected);
});

/** Return a rename replacing occurrence `i` whole with `names[i]`, keeping the others. */
function byIndex(names: Record<number, string>): (i: number, read: ConditionRead) => RefRenamed | null {
  return (i, read) => (i in names ? { text: names[i]!, used: read.path.split(".").length } : null);
}

test("rewrite renames each occurrence", () => {
  const when = "a.output > 5 or (a.output :- 0) < 1";
  expect(rewriteConditionReads(when, byIndex({ 0: "__r0", 1: "__r1" }))).toBe("__r0 > 5 or (__r1 :- 0) < 1");
});

test("rewrite keeps the whole-span braces", () => {
  expect(rewriteConditionReads("${(a.output :- 0) >= 1}", byIndex({ 0: "__r0" }))).toBe("${(__r0 :- 0) >= 1}");
});

test("rewrite leaves string literals alone", () => {
  expect(rewriteConditionReads("(a.output :- '}') == '}'", byIndex({ 0: "__r0" }))).toBe("(__r0 :- '}') == '}'");
});

test("rewrite to null keeps the leaf", () => {
  const when = "a.output > 5 or (a.output :- 0) < 1";
  expect(rewriteConditionReads(when, byIndex({ 1: "__r1" }))).toBe("a.output > 5 or (__r1 :- 0) < 1");
});

test("rewrite passes the read", () => {
  const seen: [number, string, number | null][] = [];
  rewriteConditionReads("(b.output | c.output) == 2", (i, read) => {
    seen.push([i, read.path, read.escape]);
    return null;
  });
  expect(seen).toEqual([
    [0, "b.output", 0],
    [1, "c.output", 0],
  ]);
});

test.each([
  "score.output > 0 and input.weight * 100 <= 5",
  "${score.output >= 0.5}",
  "isinstance(d.output, Buy) and d.output.qty > 5",
])("rewrite by path matches rewriteConditionRefs: %s", (when) => {
  const paths: Record<string, string> = {
    "score.output": "__s",
    "input.weight": "__w",
    "d.output": "__d",
    "d.output.qty": "__q",
  };
  const rename = (path: string) => (path in paths ? { text: paths[path]!, used: path.split(".").length } : null);
  expect(rewriteConditionReads(when, (_i, read) => rename(read.path))).toBe(rewriteConditionRefs(when, rename));
});

const BUDGET = 'len(join([m.text for t in input.history for m in t], "")) > input.budget';

test("comprehension variables are not condition reads", () => {
  expect(conditionReads(parseCondition(BUDGET)).map((r) => r.path)).toEqual(["input.history", "input.budget"]);
});

test("rewrite renames a comprehension's iterable, not its variables", () => {
  expect(rewriteConditionReads(BUDGET, (i, read) => ({ text: `__r${i}`, used: read.path.split(".").length }))).toBe('len(join([m.text for t in __r0 for m in t], "")) > __r1');
});

test("a condition read carries its text as written", () => {
  expect(conditionReads(parseCondition("x.output?.k > 1 or y")).map((read) => read.text)).toEqual(["x.output?.k", "y"]);
});

test("rewriteConditionReads replaces a prefix and keeps the steps after it", () => {
  expect(rewriteConditionReads("x.output?.k > 1", () => ({ text: "__r0", used: 2 }))).toBe("__r0?.k > 1");
});
