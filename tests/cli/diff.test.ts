// A line diff in the unified format, for the chat's approval card.

import { describe, expect, test } from "bun:test";

import { unifiedDiff } from "../../src/cli/diff.ts";

const LABELS = { oldLabel: "a.yaml", newLabel: "a.yaml" };

describe("unifiedDiff", () => {
  test("identical texts have no hunks", () => {
    expect(unifiedDiff("a\nb\n", "a\nb\n", LABELS)).toBe("");
    expect(unifiedDiff("", "", LABELS)).toBe("");
  });

  test("an edit shows its removed and added lines with three lines of context", () => {
    const old = ["1", "2", "3", "4", "5", "6", "7", "8", "9"].join("\n") + "\n";
    const edited = ["1", "2", "3", "4", "five", "6", "7", "8", "9"].join("\n") + "\n";
    expect(unifiedDiff(old, edited, LABELS)).toBe(
      ["--- a.yaml", "+++ a.yaml", "@@ -2,7 +2,7 @@", " 2", " 3", " 4", "-5", "+five", " 6", " 7", " 8", ""].join("\n"),
    );
  });

  test("changes far apart are separate hunks; near ones share one", () => {
    const lines = Array.from({ length: 20 }, (_, i) => String(i + 1));
    const edited = lines.map((line) => (line === "2" || line === "18" ? `${line}!` : line));
    const diff = unifiedDiff(lines.join("\n") + "\n", edited.join("\n") + "\n", LABELS);
    expect(diff.split("\n").filter((line) => line.startsWith("@@"))).toEqual(["@@ -1,5 +1,5 @@", "@@ -15,6 +15,6 @@"]);
    const near = lines.map((line) => (line === "5" || line === "11" ? `${line}!` : line));
    const one = unifiedDiff(lines.join("\n") + "\n", near.join("\n") + "\n", LABELS);
    expect(one.split("\n").filter((line) => line.startsWith("@@"))).toEqual(["@@ -2,13 +2,13 @@"]);
  });

  test("a new file is all added lines", () => {
    expect(unifiedDiff("", "id: a\nname: a\n", { oldLabel: "/dev/null", newLabel: "a.yaml" })).toBe(
      ["--- /dev/null", "+++ a.yaml", "@@ -0,0 +1,2 @@", "+id: a", "+name: a", ""].join("\n"),
    );
  });

  test("a change only in line endings is visible", () => {
    const diff = unifiedDiff("a\r\nb\n", "a\nb\n", LABELS);
    expect(diff).toBe(["--- a.yaml", "+++ a.yaml", "@@ -1,2 +1,2 @@", "-a[CRLF]", "+a", " b", ""].join("\n"));
  });

  test("a last line without a newline is marked", () => {
    expect(unifiedDiff("a\n", "a", LABELS)).toBe(
      ["--- a.yaml", "+++ a.yaml", "@@ -1 +1 @@", "-a", "+a", "\\ No newline at end of file", ""].join("\n"),
    );
  });

  test("lines moved around keep the longest common run", () => {
    const diff = unifiedDiff("x\na\nb\nc\n", "a\nb\nc\nx\n", LABELS);
    expect(diff).toBe(["--- a.yaml", "+++ a.yaml", "@@ -1,4 +1,4 @@", "-x", " a", " b", " c", "+x", ""].join("\n"));
  });
});
