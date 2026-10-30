/**
 * The live rows (`live_rows.ts`): each frame one synchronized write that repaints only the rows
 * that changed, over the old ones, read back through the test screen.
 */

import { describe, expect, test } from "bun:test";

import { layout, LiveRows, liveRows, SYNC_OFF, SYNC_ON } from "../../src/cli/live_rows.ts";
import { FakeOutput, screen } from "./terminal_both_runtimes.ts";

const COLUMNS = 20;

/** Live rows on a fake output `COLUMNS` wide, and the writes each flush made. */
function setup() {
  const output = new FakeOutput({ columns: COLUMNS });
  const live = new LiveRows(output);
  const writes: string[] = [];
  type Cursor = { line: number; before: string } | null;
  const draw = (lines: string[], cursor: Cursor = null) => {
    const before = output.text.length;
    live.draw(lines, cursor);
    const text = output.text.slice(before);
    if (text !== "") writes.push(text);
    return text;
  };
  return { output, live, writes, draw, view: () => screen(output.text, COLUMNS) };
}

describe("a frame", () => {
  test("is one write wrapped in synchronized output, the cursor hidden while it draws", () => {
    const { output, draw } = setup();
    const text = draw(["one", "two"]);
    expect(output.writes).toBe(1);
    expect(text.startsWith(`${SYNC_ON}\x1b[?25l`)).toBe(true);
    expect(text.endsWith(SYNC_OFF)).toBe(true);
  });

  test("that keeps the row count clears nothing below and rewrites only the changed row", () => {
    const { draw, view } = setup();
    draw(["head", "· spin 1s", "box", "hint"]);
    const text = draw(["head", "✢ spin 1s", "box", "hint"]);
    expect(text).not.toContain("\x1b[J");
    expect(text).toContain("✢ spin 1s");
    expect(text).not.toContain("head");
    expect(text).not.toContain("box");
    expect(text).not.toContain("hint");
    expect(view().rows).toEqual(["head", "✢ spin 1s", "box", "hint"]);
  });

  test("writes nothing when nothing changed", () => {
    const { draw, writes } = setup();
    draw(["a", "b"]);
    draw(["a", "b"]);
    expect(writes).toHaveLength(1);
  });

  test("with fewer rows clears what is left below, once", () => {
    const { draw, view } = setup();
    draw(["a", "b", "c"]);
    const text = draw(["a"]);
    expect(text.match(/\x1b\[J/g)).toHaveLength(1);
    expect(view().rows).toEqual(["a"]);
  });

  test("with no rows leaves the cursor where the rows began", () => {
    const { draw, view } = setup();
    draw(["x"]);
    draw(["a", "b", "c"]);
    draw([]);
    expect(view().rows).toEqual([]);
    expect(view().cursor).toEqual({ row: 0, col: 0 });
  });

  test("clears what is left of a shorter row, but never after a row that fills the width", () => {
    const { draw, view } = setup();
    draw(["a long row here"]);
    expect(draw(["short"])).toContain("short\x1b[K");
    const full = "x".repeat(COLUMNS);
    expect(draw([full])).not.toContain(`${full}\x1b[K`);
    expect(view().rows).toEqual([full]);
  });

  test("places the cursor after `before` on its line and shows it", () => {
    const { draw, view } = setup();
    const text = draw(["top", "│ > hi", "bottom"], { line: 1, before: "│ > hi" });
    expect(view().cursor).toEqual({ row: 1, col: 6 });
    expect(text.endsWith(`\x1b[?25h${SYNC_OFF}`)).toBe(true);
  });
});

describe("staged rows", () => {
  test("are printed for good above the live rows in the same write", () => {
    const { live, output, draw, view } = setup();
    draw(["spin", "box"]);
    live.stage(["printed"]);
    draw(["spin", "box"]);
    expect(output.writes).toBe(2);
    expect(view().rows).toEqual(["printed", "spin", "box"]);
  });

  test("the screen shows already at the live rows' top are left as they are", () => {
    const { live, draw, view } = setup();
    draw(["line 1", "line 2", "partial", "spin"]);
    live.stage(["line 1"]);
    const text = draw(["line 2", "line 3", "partial!", "spin"]);
    expect(text).not.toContain("line 1");
    expect(text).not.toContain("line 2");
    expect(text).not.toContain("\x1b[J");
    expect(view().rows).toEqual(["line 1", "line 2", "line 3", "partial!", "spin"]);
  });

  test("kept in place move the cursor down with the rows under them", () => {
    const { live, draw, view } = setup();
    draw(["a", "b", "c"], { line: 1, before: "b" });
    live.stage(["a"]);
    // Nothing changes on screen: the cursor goes down a row, to "c" where it now is.
    draw(["b", "c"], { line: 1, before: "c" });
    expect(view().cursor).toEqual({ row: 2, col: 1 });
  });

  test("wait for the next flush", () => {
    const { live, output, view } = setup();
    live.stage(["later"]);
    expect(output.text).toBe("");
    live.flush();
    expect(view().rows).toEqual(["later"]);
  });
});

describe("after a resize", () => {
  test("a frame erases from the live rows' top and draws them all", () => {
    const { live, draw, view } = setup();
    draw(["a", "b"]);
    live.lose();
    const text = draw(["a", "b"]);
    expect(text).toContain("\x1b[J");
    expect(view().rows).toEqual(["a", "b"]);
  });

  test("a frame at a new width starts over as well", () => {
    const { output, draw } = setup();
    draw(["a", "b"]);
    output.columns = 30;
    expect(draw(["a", "b"])).toContain("\x1b[J");
  });
});

describe("after a clear", () => {
  test("the next frame erases the screen, not the scrollback, and draws every row at the top", () => {
    const { output, live, draw, view, writes } = setup();
    output.write("printed for good\r\n");
    draw(["reply", "spin", "box"], { line: 2, before: "b" });
    live.clear();
    const text = draw(["reply", "spin", "box"], { line: 2, before: "b" });
    expect(writes).toHaveLength(2);
    expect(text.startsWith(`${SYNC_ON}\x1b[?25l\x1b[2J\x1b[H`)).toBe(true);
    expect(text).not.toContain("\x1b[3J");
    for (const row of ["reply", "spin", "box"]) expect(text.split(row)).toHaveLength(2);
    expect(view().rows).toEqual(["reply", "spin", "box"]);
    expect(view().scrollback).toEqual(["printed for good", "reply", "spin", "box"]);
    expect(view().cursor).toEqual({ row: 2, col: 1 });
  });

  test("a flush with nothing set draws the rows on screen again, then frames go on as usual", () => {
    const { live, draw, view } = setup();
    draw(["a", "b"]);
    live.clear();
    live.stage(["card"]);
    live.flush();
    expect(view().rows).toEqual(["card", "a", "b"]);
    const text = draw(["a", "c"]);
    expect(text).not.toContain("\x1b[2J");
    expect(view().rows).toEqual(["card", "a", "c"]);
  });
});

test("a line wider than the terminal takes the rows it wraps onto", () => {
  expect(layout("x".repeat(25), COLUMNS, -1)).toMatchObject({ rows: 2, end: 5 });
  const { draw, view } = setup();
  draw(["x".repeat(25), "next"]);
  draw(["x".repeat(25), "then"]);
  expect(view().rows).toEqual(["x".repeat(COLUMNS), "xxxxx", "then"]);
});

test("a wide character wrapped at the last column leaves no old cell behind it", () => {
  const { draw, view } = setup();
  draw(["y".repeat(COLUMNS)]);
  draw([`${"x".repeat(COLUMNS - 1)}日本`]);
  expect(view().rows).toEqual(["x".repeat(COLUMNS - 1), "日本"]);
});

describe("the cursor", () => {
  test("stays hidden after a frame that doesn't place it while hidden, and shows once not", () => {
    const { live, draw, view } = setup();
    live.hideCursor(true);
    draw(["a"]);
    expect(view().modes.at(-1)).toBe("?25l");
    live.hideCursor(false);
    live.flush();
    expect(view().modes.at(-1)).toBe("?25h");
    expect(live.cursorHidden).toBe(false);
  });
});

test("one output has one set of live rows, shared by all that draw on it", () => {
  const output = new FakeOutput();
  expect(liveRows(output)).toBe(liveRows(output));
  expect(liveRows(new FakeOutput())).not.toBe(liveRows(output));
});
