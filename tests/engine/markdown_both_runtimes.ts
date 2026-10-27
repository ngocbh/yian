/**
 * Markdown store operations written once for both runtimes: `markdown_store.test.ts` runs them on
 * Bun and, built, imports them on Node next to the built store. They take the store as a parameter
 * so this module imports no runtime code.
 */

import type { MarkdownStore } from "../../src/memory/storage/markdown.ts";

/** The CRLF document the fixture generator writes twice. */
const CRLF_DOC = "# Ada\r\nPrefers email.\r\nCafé ☕ 日本語 😀\r\n";

/**
 * Run the fixture generator's operations on `store`. Python's ints are `bigint`s and its floats
 * `number`s: a `number` `3` would be written `3.0`.
 */
export async function replay(store: MarkdownStore): Promise<void> {
  const [, empty] = await store.read();
  const v1 = await store.write("# Ada\n", {
    baseVersion: empty,
    provenance: {
      run_id: "r1",
      commit_id: "c1",
      seen_version: empty,
      feedback: ["Prefers email.", { score: 3n, ok: true, none: null, ratio: 0.25, whole: 100 }],
    },
  });
  const v2 = await store.write(CRLF_DOC, {
    baseVersion: v1,
    provenance: {
      run_id: "r2",
      commit_id: "c2",
      seen_version: v1,
      feedback: ["naïve — “quotes”", 'tab\there\nnewline "q" back\\slash', "😀", "\u0001\u007f\u2028"],
      Zeta: [],
      _under: {},
      alpha: { b: 2n ** 70n, a: -5n, B: [1.5, -0.5] },
    },
  });
  const [, current] = await store.read();
  const v3 = await store.write("\ufeff# BOM doc\rold mac\nno trailing newline", { baseVersion: current });
  const v4 = await store.write("", { baseVersion: v3, provenance: { run_id: "r4", commit_id: "c4", feedback: [] } });
  await store.write(CRLF_DOC, { baseVersion: v4, provenance: { run_id: "r5", commit_id: "c5" } });
  await store.read(v2);
}

/** What `readAndExtend` returns: plain JSON, so Node can print it. */
export interface ExtendResult {
  current: readonly [string, string];
  kept: (readonly [string, string])[];
  written: string;
}

/** Read the current document and each of `versions` from `store`, then write a new document on the current version. */
export async function readAndExtend(store: MarkdownStore, versions: readonly string[]): Promise<ExtendResult> {
  const current = await store.read();
  const kept = [];
  for (const version of versions) kept.push(await store.read(version));
  const written = await store.write("# Ada\nFrom TS.\n", {
    baseVersion: current[1],
    provenance: { run_id: "ts", score: 0.5 },
  });
  return { current, kept, written };
}
