/**
 * Text made safe to print on a terminal. Text from a flow, a model or a file can carry escape
 * sequences that move the cursor, set the clipboard, forge a link or redraw what the user sees (a
 * fake approval card), and BIDI or zero-width characters that make it read differently than it is.
 *
 * Knows about: nothing else.
 */

/** How `sanitize` treats each kind of character it doesn't print as is. */
export interface SanitizeOptions {
  /**
   * A whole CSI, OSC or DCS (or SOS, PM, APC) sequence: `"drop"` it (the default), or `"show"` its
   * characters like any other control.
   */
  sequences?: "drop" | "show";
  /** A tab: `"keep"` it (the default) or `"show"` it as `^I`. */
  tabs?: "keep" | "show";
  /** A BIDI control or a zero-width character: `"show"` it as `\u{202e}` (the default) or `"keep"` it. */
  invisible?: "show" | "keep";
}

/** A CSI sequence, in its 7-bit or 8-bit form: parameters, intermediates and a final byte. */
const CSI = String.raw`(?:\x1b\[|\x9b)[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]`;

// The string bodies below stop at the first ESC or C1 character (ST and every introducer among
// them), so an unended one scans only to the next sequence and matching all of a text stays linear.

/** An OSC sequence, in its 7-bit or 8-bit form, ended by BEL or ST. */
const OSC = String.raw`(?:\x1b\]|\x9d)[^\x07\x1b\x80-\x9f]*(?:\x07|\x1b\\|\x9c)`;

/** A DCS, SOS, PM or APC string, in its 7-bit or 8-bit form, ended by ST. */
const STRING = String.raw`(?:\x1b[PX^_]|[\x90\x98\x9e\x9f])[^\x1b\x80-\x9f]*(?:\x1b\\|\x9c)`;

/** The source of a pattern matching one whole CSI, OSC or DCS (or SOS, PM, APC) sequence. */
export const SEQUENCE_SOURCE = `${CSI}|${OSC}|${STRING}`;

// An unended sequence isn't matched, so its characters are shown instead.
const SEQUENCE = new RegExp(SEQUENCE_SOURCE, "g");

/**
 * The BIDI marks, embeddings, overrides and isolates, the zero-width and other invisible format
 * characters, the Hangul and Khmer fillers that draw as blanks, the line and paragraph separators,
 * the interlinear annotation marks and the tags.
 */
const INVISIBLE = new RegExp(
  String.raw`[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f` +
    String.raw`\u3164\ufeff\uffa0\ufff9-\ufffb\u{e0000}-\u{e007f}]`,
  "gu",
);

/**
 * Return `text` safe to print: `\r\n` read as `\n`; escape sequences dropped; each other control
 * character but `\n` and `\t` (C0, DEL and C1, ESC included) shown as `^X` (`^[` for ESC, `^?` for
 * DEL) or, for C1, U+FFFD; BIDI controls, zero-width and other invisible characters shown as
 * `\u{202e}`. Other text, accents and wide characters included, is left as it is.
 */
export function sanitize(text: string, options: SanitizeOptions = {}): string {
  let safe = text.replaceAll("\r\n", "\n");
  if (options.sequences !== "show") safe = safe.replace(SEQUENCE, "");
  const control = options.tabs === "show" ? /[\x00-\x09\x0b-\x1f\x7f-\x9f]/g : /[\x00-\x08\x0b-\x1f\x7f-\x9f]/g;
  safe = safe.replace(control, (char) => {
    const code = char.charCodeAt(0);
    if (code === 0x7f) return "^?";
    return code < 0x20 ? `^${String.fromCharCode(code + 64)}` : "\uFFFD";
  });
  if (options.invisible !== "keep") {
    safe = safe.replace(INVISIBLE, (char) => `\\u{${char.codePointAt(0)!.toString(16)}}`);
  }
  return safe;
}

/** The `[start, end)` spans of the whole escape sequences in `text` that `sanitize` drops. */
export function sequenceSpans(text: string): [number, number][] {
  return [...text.matchAll(SEQUENCE)].map((match) => [match.index, match.index + match[0].length]);
}
