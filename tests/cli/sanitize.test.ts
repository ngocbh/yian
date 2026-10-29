// Text made safe to print on a terminal.

import { describe, expect, test } from "bun:test";

import { sanitize } from "../../src/cli/sanitize.ts";

describe("sanitize", () => {
  test("drops CSI sequences, 7-bit and 8-bit", () => {
    expect(sanitize("a\x1b[31mred\x1b[0m b\x1b[2J\x1b[?25l")).toBe("ared b");
    expect(sanitize("a\x9b1Ab")).toBe("ab");
  });

  test("drops OSC sequences ended by BEL or ST", () => {
    expect(sanitize("a\x1b]52;c;aGk=\x07b")).toBe("ab");
    expect(sanitize("a\x1b]8;;https://x.example\x1b\\link\x1b]8;;\x1b\\")).toBe("alink");
    expect(sanitize("a\x9d0;title\x9cb")).toBe("ab");
  });

  test("drops DCS sequences", () => {
    expect(sanitize("a\x1bP1$r0m\x1b\\b")).toBe("ab");
    expect(sanitize("a\x90q#0\x9cb")).toBe("ab");
  });

  test("shows other C0 and C1 controls, keeping newlines and tabs", () => {
    expect(sanitize("a\x07b\x7fc\x85d\x1bce")).toBe("a^Gb^?c\uFFFDd^[ce");
    expect(sanitize("one\r\ntwo\tthree\rfour")).toBe("one\ntwo\tthree^Mfour");
    expect(sanitize("an unended \x1b]0;title")).toBe("an unended ^[]0;title");
  });

  test("shows BIDI controls and zero-width characters as escapes", () => {
    expect(sanitize("a\u202eb\u2066c\u2069d")).toBe("a\\u{202e}b\\u{2066}c\\u{2069}d");
    expect(sanitize("a\u200bb\u200fc\u2060d\ufeffe")).toBe("a\\u{200b}b\\u{200f}c\\u{2060}d\\u{feff}e");
  });

  test("shows other invisible format characters, astral ones included", () => {
    const text = "a\u061cb\u2028c\u2029d\u00ade\u180ef\u2061g\u2064h\u206ai\u206fj\ufff9k\ufffbl\u{e0041}m\u{e007f}n";
    expect(sanitize(text)).toBe(
      "a\\u{61c}b\\u{2028}c\\u{2029}d\\u{ad}e\\u{180e}f\\u{2061}g\\u{2064}h\\u{206a}i\\u{206f}j\\u{fff9}k\\u{fffb}l" +
        "\\u{e0041}m\\u{e007f}n",
    );
  });

  test("leaves other text as it is", () => {
    expect(sanitize("héllo 你好")).toBe("héllo 你好");
    expect(sanitize("\u{1f600} \u{20000}")).toBe("\u{1f600} \u{20000}");
  });

  test("the ask form shows sequences and tabs and keeps BIDI and zero-width characters", () => {
    const options = { sequences: "show", tabs: "show", invisible: "keep" } as const;
    expect(sanitize("H\x1b[1A\tx\u202e\u200d", options)).toBe("H^[[1A^Ix\u202e\u200d");
  });
});
