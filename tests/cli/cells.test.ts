/** Measuring text by the cells a terminal draws it in (`cells.ts`). */

import { describe, expect, test } from "bun:test";

import { cells, charCells, wrapLine } from "../../src/cli/cells.ts";
import { WIDE_RANGES } from "../../src/cli/wide_cells.ts";

/**
 * Symbols and emoji Unicode gives East Asian Width W that a table of whole blocks counted one cell
 * wide, though terminals draw them two: what a held call's card must measure right.
 */
const WIDE_SYMBOLS =
  "⌚⌛〈〉⏩⏪⏫⏬⏰⏳◽◾☔☕♈♉♊♋♌♍♎♏♐♑♒♓♿⚓⚡⚪⚫⚽⚾⛄⛅⛎⛔⛪⛲⛳⛵⛺⛽✅✊✋✨❌❎❓❔❕❗➕➖➗➰➿⬛⬜⭐⭕🀄🃏🆎🆑🆒🆓🆔" +
  "🆕🆖🆗🆘🆙🆚🈀🈁🈂🈐🈑🈒🈓🈔🈕🈖🈗🈘🈙🈚🈛🈜🈝🈞🈟🈠🈡🈢🈣🈤🈥🈦🈧🈨🈩🈪🈫🈬🈭🈮🈯🈰🈱🈲🈳🈴🈵🈶🈷🈸🈹🈺🈻🉀🉁🉂" +
  "🉃🉄🉅🉆🉇🉈🉐🉑🉠🉡🉢🉣🉤🉥🟠🟡🟢🟣🟤🟥🟦🟧🟨🟩🟪🟫🟰";

describe("a character's cells", () => {
  test("are two for every East Asian Wide or Fullwidth character, the symbols and emoji among them", () => {
    const symbols = Array.from(WIDE_SYMBOLS);
    expect(symbols).toHaveLength(152);
    expect(symbols.filter((char) => charCells(char) !== 2)).toEqual([]);
    // Hangul Jamo Extended-A, small form variants, Kana Supplement, Tangut, and a CJK extension.
    for (const code of [0xa960, 0xfe10, 0xfe50, 0x1b000, 0x17000, 0x2a700, 0x30000]) {
      expect([code.toString(16), charCells(String.fromCodePoint(code))]).toEqual([code.toString(16), 2]);
    }
  });

  test("are one for narrow and ambiguous characters, and none for a combining mark or joiner", () => {
    expect(["a", "é", "①", "→", "│", "…", "🌡"].map(charCells)).toEqual([1, 1, 1, 1, 1, 1, 1]);
    expect(["中", "가", "Ａ", "😀"].map(charCells)).toEqual([2, 2, 2, 2]);
    expect(["́", "‍", "️"].map(charCells)).toEqual([0, 0, 0]);
  });

  test("come from a table of ranges in order, each apart from the next", () => {
    expect(WIDE_RANGES.length % 2).toBe(0);
    for (let i = 0; i < WIDE_RANGES.length; i += 2) {
      expect(WIDE_RANGES[i]!).toBeLessThanOrEqual(WIDE_RANGES[i + 1]!);
      if (i > 0) expect(WIDE_RANGES[i]!).toBeGreaterThan(WIDE_RANGES[i - 1]! + 1);
    }
  });

  test("count in a line, and wrap it", () => {
    expect(cells("✅ ok")).toBe(5);
    expect(wrapLine("✅✅✅", 4)).toEqual([
      [0, 2],
      [2, 3],
    ]);
  });
});
