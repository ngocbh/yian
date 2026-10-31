/**
 * Source-neutral values for semantic failures and diagnostic presentation, plus the
 * text helpers yian's messages need.
 *
 * This dependency-free leaf separates semantic code from any authoring front end.
 * `Failure` carries stable meaning plus opaque `OriginRef` instances; a front end
 * implements `OriginResolver` to turn those references into concrete presentation
 * frames. `ResolverTable` composes those namespaces through exact immutable routes.
 * `compareCodeUnits` is the one string order for sorting (`compareCodePoints` serves formats fixed
 * to code-point order), `reprText` quotes text the way CPython 3.12's `repr` does, and
 * `getCloseMatches` finds "did you mean" suggestions the way CPython 3.12's `difflib` does, and
 * `shellQuote` writes a word for a command yian prints for you to paste.
 * `addNote`, `errorTypeName` and `noteCleanup` help report a thrown error.
 */

/**
 * Compare two strings by UTF-16 code units: the one string order for sorting in yian.
 * Sorting by code point would differ only for astral characters
 * (`rule: text`).
 */
export function compareCodeUnits(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Compare two strings by code point, for the few formats fixed to that order (a memory file's JSON
 * keys, prompt choices); everything else sorts with `compareCodeUnits`.
 */
export function compareCodePoints(a: string, b: string): number {
  // Equal code points take equal code units, so one index walks both strings.
  let i = 0;
  while (i < a.length && i < b.length) {
    const x = a.codePointAt(i)!;
    const y = b.codePointAt(i)!;
    if (x !== y) return x - y;
    i += x > 0xffff ? 2 : 1;
  }
  return a.length - b.length;
}

// Reproduces CPython 3.12's `repr` of a `str` (`unicode_repr`); the printable table is
// generated from CPython 3.12.14's `str.isprintable` (Unicode 15.0.0). Python Software
// Foundation License; see NOTICE.

// CPython 3.12.14's `str.isprintable()` over U+0000..U+10FFFF as alternating run
// lengths in base 36, starting with the non-printable run at U+0000. It is a pinned
// table rather than the engine's Unicode data (`\p{...}`), so Node and Bun agree with
// Python 3.12 whatever their Unicode version. Regenerate with CPython 3.12.14:
//
//   python3.12 -I -c "
//   d='0123456789abcdefghijklmnopqrstuvwxyz'
//   b36=lambda n: (b36(n//36) if n>=36 else '')+d[n%36]
//   runs=[];cur=False;n=0
//   for c in range(0x110000):
//       p=chr(c).isprintable()
//       if p==cur: n+=1
//       else: runs.append(n);cur=p;n=1
//   runs.append(n);print(' '.join(map(b36,runs)))"
const PRINTABLE_RUNS = `
  w 2n y c 1 ju 2 6 4 7 1 1 1 k 1 b1 1 12 2 1e 2 3 1 1j 8 r 4 6 h m 1 5c 1 1c 2 1n 2 2t e 1n 2
  1d 2 f 1 s 2 1 1 b 5 v 9 22 1 4h 1 8 2 2 2 m 1 7 1 1 3 4 2 9 2 2 2 4 8 1 4 2 1 5 2 p 2 3 1 6 4
  2 2 m 1 7 1 2 1 2 1 2 2 1 1 5 4 2 2 3 3 1 7 4 1 1 7 h a 3 1 9 1 3 1 m 1 7 1 2 1 5 2 a 1 3 1 3
  2 1 f 4 2 c 7 7 1 3 1 8 2 2 2 m 1 7 1 2 1 5 2 9 2 2 2 3 7 3 4 2 1 5 2 i a 2 1 6 3 3 1 4 3 2 1
  1 1 2 3 2 3 3 3 c 4 5 3 3 1 4 2 1 6 1 e l 5 d 1 3 1 n 1 g 2 9 1 3 1 4 7 2 1 3 2 1 2 4 2 a 7 m
  1 3 1 n 1 a 1 5 2 9 1 3 1 4 7 2 6 2 1 4 2 a 1 3 c d 1 3 1 1f 1 3 1 6 4 g 2 q 1 3 1 i 3 o 1 9 1
  1 2 7 3 1 4 6 1 1 1 8 6 a 2 3 c 1m 4 t 11 2 1 1 1 5 1 o 1 1 1 n 2 5 1 1 1 7 1 a 2 4 w 20 1 10
  4 13 1 10 1 f 1 d 11 5i 1 1 5 1 2 ah 1 4 2 7 1 1 1 4 2 15 1 4 2 x 1 4 2 7 1 1 1 4 2 f 1 1l 1 4
  2 1v 2 w 3 q 6 2e 2 6 2 hs 1 s 3 2h 7 m 9 o 9 k c d 1 3 1 2 c 2m 2 a 6 a 6 e 1 b 6 2h 7 17 5
  1y a v 1 c 4 c 4 1 3 16 2 5 b 18 4 q 6 b 3 1q 2 1t 1 t 2 b 6 a 6 e 2 v 1d 25 3 1b 1 38 8 1o 3
  f 3 1o 7 17 2 b 8 17 5 eu 2 6 2 12 2 6 2 8 1 1 1 1 1 1 1 v 2 1h 1 f 1 e 2 6 1 j 2 3 1 9 h o 8
  1b h 2 2 r 1 d 3 x f x f 3w 4 if p b l 1ec 2 w 1 9p 5 19 1 1 5 1 2 1k 7 2 e o 9 7 1 7 1 7 1 7
  1 7 1 7 1 7 1 7 1 3i y q 1 2h c 5y q c 5 1r 1 2e 2 2v 5 17 1 2m 1 2c c 1b 1 mlp 3 1j 9 9o k 54
  8 5n 5 2 1 1 1 5 o 1n 3 a 6 1k 8 1y 8 c 6 38 b u 3 26 1 b 4 x 1 1j 9 e 2 a 2 2v o s a 6 2 6 2
  6 9 7 1 7 1 1o 4 3i 2 a 6 8mc c n 4 1d 6is a6 2 2y 12 7 c 5 5 q 1 5 1 1 1 2 1 2 1 3h g cd 2 1i
  7 1 w 16 6 1f 1 j 1 4 4 5 1 3r 4 5a 3 6 2 6 2 6 2 3 3 7 1 7 d 2 2 c 1 q 1 j 1 2 1 f 2 e y 3f 5
  3 4 19 3 2g 1 d 3 1 1b 1a 3m t 3 1d f s 4 10 9 u 5 17 5 u 1 11 4 e 16 4e 2 a 6 10 4 10 4 14 8
  1g b c 1 f 1 7 1 2 1 b 1 f 1 7 1 2 1v 8n 9 m a 8 o 6 1 16 1 9 1x 6 2 1 1 18 1 2 3 1 2 n 1 20 8
  9 1c j 1 2 5 x 3 r 5 1 1s 1k 4 k 2 1e 1 2 5 8 1 3 1 t 2 3 4 a 7 9 7 1s w 13 4 c 9 1i 3 t 2 r 5
  q 7 4 c 7 28 21 1j 1f d 1f 7 1a 8 a 86 v 1 16 1 3 2 2 23 17 8 16 m q 12 s k n 9 26 4 10 9 1q 1
  5 d p 7 a 6 1h 1 i 8 13 9 2o 1 k b i 1 1b 1q 7 1 1 1 4 1 f 1 b 6 1n 5 a 6 4 1 8 2 2 2 m 1 7 1
  2 1 5 1 a 2 2 2 3 2 1 6 1 5 7 2 7 3 5 3v 2k 1 5 u 20 8 a 4m 1i 2 12 y 1x b a 6 d j 1m 6 a 1i r
  2 f 4 n 55 1o 2s 2b c 8 2 1 2 8 1 2 1 u 1 2 2 c 9 a 1y 8 2 1a 2 b r 20 8 2b d 21 7 a 6u 9 1 19
  1 e a t 3 w 2 m 1 e 21 7 1 2 1 18 3 1 1 2 1 9 8 a 6 6 1 2 1 11 1 2 1 6 7 a 8m p 7 h 1 15 3 s
  2e 1 f 1e d pn 2u 33 1 5 b 5g 218 2r d ts g m 33e g7 6nt ft 7 v 1 a 4 29 1 a 6 u 2 6 a 1y a a
  1 7 1 l 5 j j4 2j 2t 23 4 1l 7 h 1s 5 b 2 e 4qg 8 ye 16 9 6w7 4 1 7 1 2 1 83 f 1 t 3 2 1 e 4 8
  b0 1s4 2z 5 d 3 9 7 a 2 4 3mo 1a 2 n 9 38 1o 6u a 13 2 22 8 34 l 1y 3e k c k c 2f 9 p 3r 2d 1
  1z 1 2 2 1 2 2 2 4 1 c 1 1 1 7 1 1t 1 4 2 8 1 7 1 s 1 4 1 5 1 1 3 7 1 9g 2 84 2 ji f 5 1 f uo
  v 6 6 5x 7 1 h 2 7 1 2 1 5 5 1q x 1 34 19 3 e 2 a 4 2 8w v h 1m 5 1 cw 16 km 7 1 4 1 2 1 f 1
  5h 2 g 15 24 4 a 4 2 lt 1w 24 1p 5e 4 1 r 1 2 1 1 2 1 1 a 1 4 1 1 1 1 6 1 4 1 1 1 1 1 1 3 1 2
  1 1 2 1 1 1 1 1 1 1 1 1 1 2 1 1 2 4 1 7 1 4 1 4 1 1 1 a 1 h 5 3 1 5 1 h 1g 2 7i 18 4 2s c f 2
  f 1 f 1 11 a 4u 1k t d 18 4 9 7 2 e 6 4a rc 4 h 3 d 3 3b 4 2n 6 c 4 1 f c 4 1k 8 a 6 14 8 u 2
  2 26 9g c e 2 d 3 9 7 1a 1 7 8 e 4 9 7 9 7 43 1 1j 11 a sm wyo w 37e 6 66 2 4g2 e 5rl 2e7 f2
  15u 3t7 5 38g f9e8 6o 47bk
`;

/**
 * The start of each run of `PRINTABLE_RUNS`, decoded on first use; runs at odd indices
 * are printable.
 */
let runStarts: number[] | undefined;

/**
 * Return whether CPython 3.12 counts code point `cp` as printable, as `str.isprintable` does
 * for one character: no control, format, surrogate, private-use or unassigned code point, and
 * no separator but the ASCII space.
 */
export function isPrintable(cp: number): boolean {
  if (runStarts === undefined) {
    runStarts = [];
    let start = 0;
    for (const run of PRINTABLE_RUNS.trim().split(/\s+/)) {
      runStarts.push(start);
      start += parseInt(run, 36);
    }
  }
  // Binary search for the last run starting at or before `cp`.
  let lo = 0;
  let hi = runStarts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (runStarts[mid]! <= cp) lo = mid;
    else hi = mid - 1;
  }
  return lo % 2 === 1;
}

/** Return `cp` as `width` lowercase hex digits. */
function hex(cp: number, width: number): string {
  return cp.toString(16).padStart(width, "0");
}

/**
 * Return `text` quoted and escaped exactly as CPython 3.12's `repr` of a `str`: in
 * double quotes when it holds a single quote and no double quote, else in single
 * quotes, with characters CPython doesn't count as printable written as `\x`, `\u` or
 * `\U` escapes.
 */
export function reprText(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  // Iterating by code point yields a lone surrogate as one unit, as CPython sees it.
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += "\\" + ch;
    else if (ch === "\t") out += "\\t";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (cp < 0x20 || cp === 0x7f) out += "\\x" + hex(cp, 2);
    else if (cp < 0x7f || isPrintable(cp)) out += ch;
    else if (cp <= 0xff) out += "\\x" + hex(cp, 2);
    else if (cp <= 0xffff) out += "\\u" + hex(cp, 4);
    else out += "\\U" + hex(cp, 8);
  }
  return out + quote;
}

// Ported from CPython 3.12 `Lib/difflib.py` (`get_close_matches`, `SequenceMatcher`)
// under the Python Software Foundation License; see NOTICE. Modified: code points, ties
// by UTF-16 code units, no junk function.

/** A matching block `[i, j, size]`: `a[i:i+size]` equals `b[j:j+size]`. */
type Match = [i: number, j: number, size: number];

/** Return `2 * matches / length`, or 1 when both sequences are empty. */
function calculateRatio(matches: number, length: number): number {
  return length ? (2.0 * matches) / length : 1.0;
}

/**
 * Compares sequences of code points by their longest matching blocks: the part of
 * CPython's `difflib.SequenceMatcher` that `getCloseMatches` needs.
 */
class SequenceMatcher {
  /** The first sequence, set by `setSeq1`. */
  private a: string[] = [];
  /** The second sequence, set by `setSeq2`. */
  private b: string[] = [];
  /** Each element of `b` mapped to its increasing indices in `b`; popular elements are absent. */
  private b2j = new Map<string, number[]>();
  /** Each element of `b` mapped to its count, computed by `quickRatio`. */
  private fullbcount: Map<string, number> | null = null;
  /** The cached result of `getMatchingBlocks`. */
  private matchingBlocks: Match[] | null = null;

  /** Set the first sequence to compare; the second is unchanged. */
  setSeq1(a: string): void {
    // CPython returns early when `a` is the same object; recomputing gives the same result.
    this.a = Array.from(a);
    this.matchingBlocks = null;
  }

  /** Set the second sequence to compare, and index it; the first is unchanged. */
  setSeq2(b: string): void {
    this.b = Array.from(b);
    this.matchingBlocks = null;
    this.fullbcount = null;
    this.chainB();
  }

  /** Build `b2j` from `b`, without the elements the autojunk rule calls popular. */
  private chainB(): void {
    const b = this.b;
    const b2j = new Map<string, number[]>();
    this.b2j = b2j;
    for (let i = 0; i < b.length; i++) {
      const elt = b[i]!;
      let indices = b2j.get(elt);
      if (indices === undefined) {
        indices = [];
        b2j.set(elt, indices);
      }
      indices.push(i);
    }

    // There is no junk to purge: `get_close_matches` never passes `isjunk`.

    // Purge popular elements; autojunk is always on.
    const popular = new Set<string>();
    const n = b.length;
    if (n >= 200) {
      const ntest = Math.floor(n / 100) + 1;
      for (const [elt, idxs] of b2j) {
        if (idxs.length > ntest) popular.add(elt);
      }
      for (const elt of popular) b2j.delete(elt);
    }
  }

  /**
   * Find the longest matching block in `a[alo:ahi]` and `b[blo:bhi]`: of all maximal
   * blocks, the one that starts earliest in `a`, then earliest in `b`;
   * `[alo, blo, 0]` when none match.
   */
  findLongestMatch(alo: number, ahi: number, blo: number, bhi: number): Match {
    const { a, b, b2j } = this;
    let besti = alo;
    let bestj = blo;
    let bestsize = 0;
    // During an iteration of the loop, j2len[j] = length of the longest match ending
    // with a[i-1] and b[j].
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i++) {
      const newj2len = new Map<number, number>();
      for (const j of b2j.get(a[i]!) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        newj2len.set(j, k);
        if (k > bestsize) {
          besti = i - k + 1;
          bestj = j - k + 1;
          bestsize = k;
        }
      }
      j2len = newj2len;
    }

    // Extend the best by equal elements on each end: popular elements aren't in
    // `b2j`, so the loop above never includes them in a match.
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) {
      besti -= 1;
      bestj -= 1;
      bestsize += 1;
    }
    while (
      besti + bestsize < ahi &&
      bestj + bestsize < bhi &&
      a[besti + bestsize] === b[bestj + bestsize]
    ) {
      bestsize += 1;
    }

    // CPython next extends the match over junk elements on each side; with no junk
    // function there are none, so those two loops can't run and are left out.
    return [besti, bestj, bestsize];
  }

  /**
   * Return the matching blocks, increasing in `i` and `j`, adjacent blocks merged, and
   * ending with the dummy `[a.length, b.length, 0]`.
   */
  getMatchingBlocks(): Match[] {
    if (this.matchingBlocks !== null) return this.matchingBlocks;
    const la = this.a.length;
    const lb = this.b.length;

    // A queue of regions still to search rather than recursion, as in CPython.
    const queue: [number, number, number, number][] = [[0, la, 0, lb]];
    const matchingBlocks: Match[] = [];
    while (queue.length > 0) {
      const [alo, ahi, blo, bhi] = queue.pop()!;
      const x = this.findLongestMatch(alo, ahi, blo, bhi);
      const [i, j, k] = x;
      // a[alo:i] vs b[blo:j] unknown
      // a[i:i+k] same as b[j:j+k]
      // a[i+k:ahi] vs b[j+k:bhi] unknown
      if (k) {
        matchingBlocks.push(x);
        if (alo < i && blo < j) queue.push([alo, i, blo, j]);
        if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
      }
    }
    matchingBlocks.sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);

    // Collapse adjacent blocks.
    let i1 = 0;
    let j1 = 0;
    let k1 = 0;
    const nonAdjacent: Match[] = [];
    for (const [i2, j2, k2] of matchingBlocks) {
      if (i1 + k1 === i2 && j1 + k1 === j2) {
        k1 += k2;
      } else {
        if (k1) nonAdjacent.push([i1, j1, k1]);
        i1 = i2;
        j1 = j2;
        k1 = k2;
      }
    }
    if (k1) nonAdjacent.push([i1, j1, k1]);
    nonAdjacent.push([la, lb, 0]);
    this.matchingBlocks = nonAdjacent;
    return nonAdjacent;
  }

  /** Return the similarity `2 * M / T`, a number in [0, 1]. */
  ratio(): number {
    let matches = 0;
    for (const [, , size] of this.getMatchingBlocks()) matches += size;
    return calculateRatio(matches, this.a.length + this.b.length);
  }

  /** Return an upper bound on `ratio()`, from the multiset intersection of `a` and `b`. */
  quickRatio(): number {
    if (this.fullbcount === null) {
      this.fullbcount = new Map();
      for (const elt of this.b) {
        this.fullbcount.set(elt, (this.fullbcount.get(elt) ?? 0) + 1);
      }
    }
    const fullbcount = this.fullbcount;
    // avail[x] is the number of times x appears in `b` less the number of times seen
    // in `a` so far.
    const avail = new Map<string, number>();
    let matches = 0;
    for (const elt of this.a) {
      const numb = avail.get(elt) ?? fullbcount.get(elt) ?? 0;
      avail.set(elt, numb - 1);
      if (numb > 0) matches += 1;
    }
    return calculateRatio(matches, this.a.length + this.b.length);
  }

  /** Return an upper bound on `ratio()` from the lengths alone. */
  realQuickRatio(): number {
    const la = this.a.length;
    const lb = this.b.length;
    return calculateRatio(Math.min(la, lb), la + lb);
  }
}

// Characters a POSIX shell word may hold unquoted.
const SHELL_SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;

/**
 * Return `text` as one POSIX shell word: as is when it holds only safe characters, else in single
 * quotes, each `'` written `'\''`. Used for the commands yian prints for you to paste.
 */
export function shellQuote(text: string): string {
  return SHELL_SAFE.test(text) ? text : `'${text.replaceAll("'", "'\\''")}'`;
}

/**
 * Return the best "good enough" matches for `word` among `possibilities`, best first:
 * at most `n`, each scoring at least `cutoff`. An exact port of CPython's
 * `difflib.get_close_matches`; equal
 * scores break by `compareCodeUnits`, descending. Throws `RangeError` when `n` is not
 * positive or `cutoff` is outside [0, 1].
 */
export function getCloseMatches(
  word: string,
  possibilities: Iterable<string>,
  n = 3,
  cutoff = 0.6,
): string[] {
  if (!(n > 0)) throw new RangeError(`n must be > 0: ${n}`);
  if (!(0.0 <= cutoff && cutoff <= 1.0)) {
    throw new RangeError(`cutoff must be in [0.0, 1.0]: ${cutoff}`);
  }
  const result: [number, string][] = [];
  const s = new SequenceMatcher();
  s.setSeq2(word);
  for (const x of possibilities) {
    s.setSeq1(x);
    if (s.realQuickRatio() >= cutoff && s.quickRatio() >= cutoff && s.ratio() >= cutoff) {
      result.push([s.ratio(), x]);
    }
  }
  // What `heapq.nlargest(n, result)` returns; equal pairs are indistinguishable.
  result.sort(([scoreX, x], [scoreY, y]) => scoreY - scoreX || compareCodeUnits(y, x));
  return result.slice(0, n).map(([, x]) => x);
}

/**
 * Append the note `text()` to `error`'s own `notes` array, creating it when absent.
 * A failure, such as a frozen error, a non-array `notes`, a primitive or a throwing
 * `text`, loses the note.
 */
export function addNote(error: unknown, text: () => string): void {
  try {
    if (typeof error !== "object" || error === null) return;
    const target = error as { notes?: unknown };
    const note = text();
    if (!Object.hasOwn(target, "notes")) target.notes = [];
    if (Array.isArray(target.notes)) target.notes.push(note);
  } catch {}
}

/**
 * The type name of a thrown value: the constructor's name, else a string
 * `name`, else `Error`.
 */
export function errorTypeName(value: unknown): string {
  const constructorName = (value as { constructor?: { name?: unknown } } | null | undefined)?.constructor?.name;
  if (typeof constructorName === "string" && constructorName) return constructorName;
  const name = (value as { name?: unknown } | null | undefined)?.name;
  return typeof name === "string" ? name : "Error";
}

/**
 * Attach one cleanup failure to `target` as the note `<label> <name>: <message>`, then the notes
 * the failure carries.
 */
export function noteCleanup(target: unknown, cleanup: unknown, label: string): void {
  addNote(target, () => {
    const [name, message] =
      cleanup instanceof Error ? [cleanup.name, cleanup.message] : [errorTypeName(cleanup), String(cleanup)];
    return `${label} ${name}: ${message}`;
  });
  const notes = (cleanup as { notes?: unknown } | null | undefined)?.notes;
  if (Array.isArray(notes)) for (const note of notes) addNote(target, () => String(note));
}

const CODES = [
  "binding.evaluate",
  "binding.type",
  "contract.input",
  "assertion.failed",
  "node.execute",
  "contract.node_result",
  "contract.output",
  "contract.route",
  "contract.growth",
  "graph.terminal_unreachable",
  "loop.max_exceeded",
  "agent.max_tool_iterations",
  "provider.auth",
  "agent.output_invalid",
  "provider.rate_limit",
  "provider.quota",
  "provider.unavailable",
  "provider.context_length",
  "provider.refused",
  "condition.evaluate",
  "tool.timeout",
  "agent.interrupted",
] as const;

/**
 * The closed list of semantic categories for expected flow failures. Each code is
 * `<area>.<condition>`, listed on the Reference page
 * (`docs/src/content/docs/reference.md`) and stable once added; codes are never derived from
 * error class names.
 */
export type FailureCode = (typeof CODES)[number];

/** Every code `Failure` accepts, at run time. */
export const FAILURE_CODES: ReadonlySet<string> = new Set(CODES);

/**
 * A concrete 1-based source span, `[start, end)` with an exclusive end column.
 *
 * Line 1, column 1 is the first character; columns count UTF-16 code units. `endCol`
 * is one past the last covered column, so a zero-width caret has start equal to end.
 * Produced and resolved by an authoring front end such as the Compose loader.
 *
 * Because the end is exclusive, `endLine` is not necessarily a line this span covers;
 * render with `coveredLines` rather than reading `endLine` directly.
 */
export class SourceRange {
  readonly startLine: number;
  readonly startCol: number;
  readonly endLine: number;
  readonly endCol: number;

  /** Create a span from its 1-based start and its exclusive end. */
  constructor(startLine: number, startCol: number, endLine: number, endCol: number) {
    this.startLine = startLine;
    this.startCol = startCol;
    this.endLine = endLine;
    this.endCol = endCol;
  }

  /**
   * Return the 1-based lines this span actually covers, in order: what a renderer
   * highlights. The result is never empty and starts at `startLine`; a zero-width caret
   * still marks its own line.
   *
   * The end is exclusive, so `endLine` is covered only if the span reaches real text on
   * it. A YAML block construct (a block mapping, or a `|-` scalar) ends at the start of
   * the next token, on a later line whose text before `endCol` is only indentation; that
   * line holds none of this span and is dropped. Naively taking `endLine` is what makes
   * an error on one node's `code:` highlight the next node's header.
   *
   * `sourceText` is the full flow text this range was resolved against, needed to tell
   * an end just past real text from an end in indentation. Lines break at `\n`, `\r\n`
   * and `\r` only (`rule: columns`), and blankness is JS `trim()` (`rule: text`).
   */
  coveredLines(sourceText: string): number[] {
    const lines = sourceText.split(/\r\n|\r|\n/);
    // No empty line follows a final line break.
    if (lines[lines.length - 1] === "") lines.pop();
    const lineCount = lines.length;
    if (!(1 <= this.startLine && this.startLine <= lineCount)) return [this.startLine];

    let last: number;
    if (this.endLine > lineCount) {
      // A stale range may extend beyond the retained source. The available
      // tail is still real context, but nonexistent line numbers are not.
      last = lineCount;
    } else {
      last = this.endLine;
      const prefix =
        0 < last && last <= lineCount ? lines[last - 1]!.slice(0, this.endCol - 1) : "";
      if (!prefix.trim()) last -= 1;
    }
    const end = Math.max(last, this.startLine);
    const covered: number[] = [];
    for (let line = this.startLine; line <= end; line++) {
      covered.push(line);
    }
    return covered;
  }
}

/** An opaque source identity interpreted only by its producing front end. */
export class OriginRef {
  /** A non-empty key unique within one returned authored flow. */
  readonly key: string;

  /** Create an origin, rejecting keys that cannot serve as stable opaque identities. */
  constructor(key: string) {
    if (typeof key !== "string" || key === "") {
      throw new RangeError("origin key must be a non-empty string");
    }
    this.key = key;
  }
}

/** One immutable source-neutral account of an expected flow failure. */
export class Failure {
  /** Stable semantic category from `FAILURE_CODES`. */
  readonly code: FailureCode;
  /** Human-readable failure detail; it may be empty, and holds no lone surrogate. */
  readonly message: string;
  /**
   * Runtime execution identity, or `null` when the failure is not attributable to one
   * executing node.
   */
  readonly nodeId: string | null;
  /**
   * Authored identities ordered outermost to innermost; the final item is the most
   * precise available operation.
   */
  readonly trace: readonly OriginRef[];
  /**
   * The thrown error's `name`, retained only for debugging, never for semantic
   * classification; it holds no lone surrogate.
   */
  readonly causeType: string | null;
  /** The thrown error's `stack`, retained as opaque debugging text; it holds no lone surrogate. */
  readonly traceback: string | null;

  /**
   * Create a failure, rejecting values that cannot enter the canonical failure
   * pipeline. `trace` defaults to empty and is copied; `causeType` and `traceback`
   * default to `null`. A lone surrogate in `message`, `causeType` or `traceback` becomes
   * U+FFFD, so a thrown error's text can always be written to the run log.
   */
  constructor(options: {
    code: FailureCode;
    message: string;
    nodeId: string | null;
    trace?: readonly OriginRef[];
    causeType?: string | null;
    traceback?: string | null;
  }) {
    const { code, message, nodeId, trace = [], causeType = null, traceback = null } = options;
    if (typeof code !== "string") throw new TypeError("failure code must be a string");
    if (!FAILURE_CODES.has(code)) throw new RangeError(`unknown failure code ${reprText(code)}`);
    if (typeof message !== "string") throw new TypeError("failure message must be a string");
    if (nodeId !== null && typeof nodeId !== "string") {
      throw new TypeError("failure node_id must be a string or null");
    }
    if (!Array.isArray(trace)) throw new TypeError("failure trace must be an array");
    for (const origin of trace) {
      if (!(origin instanceof OriginRef)) {
        throw new TypeError("failure trace item must be an OriginRef");
      }
    }
    if (causeType !== null && typeof causeType !== "string") {
      throw new TypeError("failure cause_type must be a string or null");
    }
    if (traceback !== null && typeof traceback !== "string") {
      throw new TypeError("failure traceback must be a string or null");
    }
    this.code = code;
    this.message = message.toWellFormed();
    this.nodeId = nodeId;
    this.trace = [...trace];
    this.causeType = causeType?.toWellFormed() ?? null;
    this.traceback = traceback?.toWellFormed() ?? null;
  }

  /** Return the JSON form: snake_case keys in field order, every key present. */
  toJSON(): {
    code: FailureCode;
    message: string;
    node_id: string | null;
    trace: { key: string }[];
    cause_type: string | null;
    traceback: string | null;
  } {
    return {
      code: this.code,
      message: this.message,
      node_id: this.nodeId,
      trace: this.trace.map((origin) => ({ key: origin.key })),
      cause_type: this.causeType,
      traceback: this.traceback,
    };
  }
}

/** Whether two failures have the same code, message, node id, trace origin keys in order, cause type and traceback. */
export function sameFailure(a: Failure, b: Failure): boolean {
  return (
    a.code === b.code &&
    a.message === b.message &&
    a.nodeId === b.nodeId &&
    a.trace.length === b.trace.length &&
    a.trace.every((origin, i) => origin.key === b.trace[i]!.key) &&
    a.causeType === b.causeType &&
    a.traceback === b.traceback
  );
}

/** A resolved source frame ready for a format-neutral diagnostic renderer. */
export class DiagnosticFrame {
  /** Human-facing source name, such as a path or notebook cell. */
  readonly label: string;
  /** Complete source text that `range` indexes. */
  readonly text: string;
  /** Concrete primary location within `text`. */
  readonly range: SourceRange;
  /** Optional context attached to this frame. */
  readonly note: string | null;

  /** Create a frame; `note` defaults to `null`. */
  constructor(options: { label: string; text: string; range: SourceRange; note?: string | null }) {
    const { label, text, range, note = null } = options;
    this.label = label;
    this.text = text;
    this.range = range;
    this.note = note;
  }
}

/** One immutable presentation-ready diagnostic. */
export class Diagnostic {
  /** Stable semantic or load-time category. */
  readonly code: string;
  /** Human-readable failure detail. */
  readonly message: string;
  /** Resolved source context ordered outermost to innermost. */
  readonly frames: readonly DiagnosticFrame[];
  /** Presentation context not tied to one frame. */
  readonly notes: readonly string[];
  /** Optional runtime execution identity. */
  readonly nodeId: string | null;
  /** Optional thrown error `name` for debugging. */
  readonly causeType: string | null;
  /** Optional opaque stack text. */
  readonly traceback: string | null;

  /**
   * Create a diagnostic. `frames` and `notes` default to empty and are copied;
   * `nodeId`, `causeType` and `traceback` default to `null`.
   */
  constructor(options: {
    code: string;
    message: string;
    frames?: readonly DiagnosticFrame[];
    notes?: readonly string[];
    nodeId?: string | null;
    causeType?: string | null;
    traceback?: string | null;
  }) {
    const {
      code,
      message,
      frames = [],
      notes = [],
      nodeId = null,
      causeType = null,
      traceback = null,
    } = options;
    this.code = code;
    this.message = message;
    this.frames = [...frames];
    this.notes = [...notes];
    this.nodeId = nodeId;
    this.causeType = causeType;
    this.traceback = traceback;
  }
}

/** The seam from opaque origins to source frames, independent of authoring format. */
export interface OriginResolver {
  /** Resolve `origin`, or return `null` when no frame is available. */
  resolve(origin: OriginRef): DiagnosticFrame | null;
}

/**
 * An immutable, collision-safe composition of opaque origin namespaces.
 *
 * Each public `OriginRef.key` routes to exactly one resolver and one origin in that
 * resolver's local namespace. Keys stay opaque: the table does no prefix parsing and
 * never falls through to another resolver. Delegate exceptions propagate; delegate
 * results must honor the `OriginResolver` return contract.
 *
 * Construct an empty table directly, or use `fromEntries` for exact routes. Raw target
 * maps are deliberately not part of the public constructor.
 */
export class ResolverTable {
  // Not `readonly`: `fromEntries` assigns it once on the new table; nothing mutates it.
  #targets: ReadonlyMap<string, { resolver: OriginResolver; origin: OriginRef }>;

  /** Create an empty table. */
  constructor() {
    this.#targets = new Map();
  }

  /**
   * Build a table from exact routes: each entry is a public origin, the delegate
   * resolver, and the exact origin that delegate understands. The routes are copied.
   * Throws `TypeError` for a malformed entry, origin or delegate, and `RangeError` when
   * the same public key occurs more than once.
   */
  static fromEntries(
    entries: Iterable<readonly [OriginRef, OriginResolver, OriginRef]>,
  ): ResolverTable {
    const targets = new Map<string, { resolver: OriginResolver; origin: OriginRef }>();
    for (const entry of entries) {
      if (!Array.isArray(entry) || entry.length !== 3) {
        throw new TypeError("resolver table entry must be a three-item tuple");
      }
      const [newOrigin, delegate, oldOrigin] = entry;
      if (!(newOrigin instanceof OriginRef)) {
        throw new TypeError("resolver table new origin must be an OriginRef");
      }
      if (typeof delegate?.resolve !== "function") {
        throw new TypeError("resolver table delegate must implement resolve");
      }
      if (!(oldOrigin instanceof OriginRef)) {
        throw new TypeError("resolver table old origin must be an OriginRef");
      }
      if (targets.has(newOrigin.key)) {
        throw new RangeError(`duplicate resolver table origin ${reprText(newOrigin.key)}`);
      }
      targets.set(newOrigin.key, { resolver: delegate, origin: oldOrigin });
    }
    const table = new ResolverTable();
    table.#targets = targets;
    return table;
  }

  /** The number of public-origin routes. */
  get size(): number {
    return this.#targets.size;
  }

  /** Return whether `origin`'s opaque key has a route; a non-`OriginRef` has none. */
  has(origin: OriginRef): boolean {
    return origin instanceof OriginRef && this.#targets.has(origin.key);
  }

  /**
   * Resolve a public origin through its exact delegate route: the delegate's result,
   * or `null` when no route exists or the delegate cannot resolve its local origin.
   * Throws `TypeError` when `origin` is not an `OriginRef` or the delegate returns
   * neither a `DiagnosticFrame` nor `null`.
   */
  resolve(origin: OriginRef): DiagnosticFrame | null {
    if (!(origin instanceof OriginRef)) {
      throw new TypeError("resolver table resolve requires an OriginRef");
    }
    const target = this.#targets.get(origin.key);
    if (target === undefined) return null;
    const frame = target.resolver.resolve(target.origin);
    if (frame !== null && !(frame instanceof DiagnosticFrame)) {
      throw new TypeError("resolver table delegate must return a DiagnosticFrame or null");
    }
    return frame;
  }
}
