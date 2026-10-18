/**
 * `conditionReads` and `rewriteConditionReads`, a port of `tests/engine/test_condition_reads.py`.
 *
 * `conditionReads` lists each reference occurrence of a parsed condition with the outermost
 * `:-` / `:?` / `|` sub-expression around it and whether that sub-expression co-skips as a
 * binding would. `rewriteConditionReads` renames reference leaves by occurrence, so the same path
 * read plainly and inside an escape can get different names.
 */

import { expect, test } from "bun:test";

import {
  type ConditionRead,
  conditionReads,
  parseCondition,
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

/** Return a rename giving occurrence `i` the name `names[i]`, keeping the others. */
function byIndex(names: Record<number, string>): (i: number, read: ConditionRead) => string | null {
  return (i) => names[i] ?? null;
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
  const rename = (path: string) => paths[path] ?? null;
  expect(rewriteConditionReads(when, (_i, read) => rename(read.path))).toBe(rewriteConditionRefs(when, rename));
});
