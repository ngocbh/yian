/** FileMemory's journal: its lines and their fsyncs, change ids, and writers across sessions. */

import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

import type { Author, MemorySession } from "../../src/memory/base.ts";
import { FileMemory, type FileMemoryOptions, memoryVersion, readMembers } from "../../src/memory/file.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { canonicalJson, type JsonObject } from "../../src/typesys/json.ts";

let dir: string;
let root: string;
const signal = new AbortController().signal;
const author: Author = { kind: "agent", name: "answer", run: "r", node: "map/0/answer" };
const START = Date.UTC(2026, 10, 7);

beforeEach(() => {
  dir = fs.mkdtempSync(join(tmpdir(), "yian-fj-"));
  root = join(dir, "m");
});

afterEach(() => {
  mock.restore();
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Open a FileMemory on `root`, creating it, with a clock that starts at `START`. */
async function openAt(options: Partial<FileMemoryOptions> = {}): Promise<MemorySession> {
  return new FileMemory({ root, id: "t/x", scheduler: virtualScheduler(START), ...options }).open({
    create: true,
    itemType: null,
    signal,
  });
}

/** Return the journal's lines, parsed. */
function journalLines(): JsonObject[] {
  const text = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8");
  expect(text.endsWith("\n")).toBe(true);
  return text
    .slice(0, -1)
    .split("\n")
    .map((line) => JSON.parse(line) as JsonObject);
}

/** Return the memory's version as its members read now. */
function versionNow(): string {
  return memoryVersion(readMembers(root, { maxDepth: 4, maxFileChars: 100_000 }).files);
}

test("append applies; the journal holds the generation, one entry and one commit line", async () => {
  const s = await openAt();
  const change = { op: "append", path: "notes/a.md", text: "x" };
  const applied = await s.apply({ id: "r/answer/1/0", change, author, signal });
  const version = versionNow();
  expect(applied).toEqual({ ok: true, version, result: { path: "notes/a.md", hash: sha256("x"), version } });
  const lines = journalLines();
  expect(lines).toHaveLength(3);
  expect(Object.keys(lines[0]!)).toEqual(["generation"]);
  expect(lines[0]!.generation).toMatch(/^[0-9a-f]{16}$/);
  expect(lines[1]).toEqual({
    id: "r/answer/1/0",
    digest: sha256(canonicalJson(change)),
    op: "append",
    path: "notes/a.md",
    author: { kind: "agent", name: "answer", run: "r", node: "map/0/answer" },
    at: "2026-11-07T00:00:00.000Z",
    before: "absent",
    after: sha256("x"),
    inverse: { absent: true },
  });
  expect(lines[2]).toEqual({ id: "r/answer/1/0", ok: true, version, result: applied.ok ? applied.result : {} });
  expect(fs.statSync(join(root, ".yian/journal.jsonl")).mode & 0o777).toBe(0o600);

  await s.apply({ id: "r/answer/1/1", change: { op: "append", path: "notes/a.md", text: "y" }, author, signal });
  // An append's inverse cuts the file back to its old bytes: it needs none of the old text.
  expect(journalLines()[3]).toMatchObject({ before: sha256("x"), after: sha256("x\ny"), inverse: { truncate: 1 } });
  expect(journalLines()).toHaveLength(5);
  await s.close();
});

test.if(process.platform === "linux")(
  "each journal line is fsynced, and the content goes through a synced temp file in .yian/tmp before the commit",
  async () => {
    const s = await openAt();
    const events: string[] = [];
    /** The path of `fd` relative to the root, a temp file's name hidden. */
    const where = (fd: number) =>
      relative(root, fs.readlinkSync(`/proc/self/fd/${fd}`)).replace(/^\.yian\/tmp\/.+$/, ".yian/tmp/*") || ".";
    const name = (path: fs.PathLike) => relative(root, String(path)).replace(/^\.yian\/tmp\/.+$/, ".yian/tmp/*");
    const { writeSync, fsyncSync, renameSync } = fs;
    spyOn(fs, "writeSync").mockImplementation(((fd: number, ...rest: unknown[]) => {
      events.push(`write ${where(fd)}`);
      return (writeSync as (...args: unknown[]) => number)(fd, ...rest);
    }) as typeof fs.writeSync);
    spyOn(fs, "fsyncSync").mockImplementation((fd: number) => {
      events.push(`fsync ${where(fd)}`);
      fsyncSync(fd);
    });
    spyOn(fs, "renameSync").mockImplementation((from: fs.PathLike, to: fs.PathLike) => {
      events.push(`rename ${name(from)} ${name(to)}`);
      renameSync(from, to);
    });

    await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "x" }, author, signal });
    expect(events).toEqual([
      // The first write makes the journal, its generation line synced with its folder entry.
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
      "fsync .yian",
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
      "write .yian/tmp/*",
      "fsync .yian/tmp/*",
      "rename .yian/tmp/* a.md",
      "fsync .",
      "fsync .yian/tmp",
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
    ]);
    events.length = 0;
    await s.apply({ id: "b", change: { op: "append", path: "a.md", text: "y" }, author, signal });
    expect(events).toEqual([
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
      "write .yian/tmp/*",
      "fsync .yian/tmp/*",
      "rename .yian/tmp/* a.md",
      "fsync .",
      "fsync .yian/tmp",
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
    ]);
    expect(fs.readdirSync(join(root, ".yian/tmp"))).toEqual([]);
    await s.close();
  },
);

test("a refused change writes nothing, not even the journal", async () => {
  const s = await openAt({ maxFileChars: 3 });
  const applied = await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "long" }, author, signal });
  expect(applied.ok).toBe(false);
  expect(fs.existsSync(join(root, ".yian/journal.jsonl"))).toBe(false);
  await s.close();
});

test("a journal line cut off by a crash is dropped before the next line is written", async () => {
  const s = await openAt();
  await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "x" }, author, signal });
  fs.appendFileSync(join(root, ".yian/journal.jsonl"), '{"id":"b","dig');
  await s.apply({ id: "c", change: { op: "append", path: "a.md", text: "y" }, author, signal });
  expect(journalLines().map((line) => line.id ?? "generation")).toEqual(["generation", "a", "a", "c", "c"]);
  await s.close();
});

test("a line that isn't a journal line makes the next write fail, naming the line", async () => {
  const s = await openAt();
  await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "x" }, author, signal });
  fs.appendFileSync(join(root, ".yian/journal.jsonl"), '{"hello":1}\n');
  await expect(
    s.apply({ id: "b", change: { op: "append", path: "a.md", text: "y" }, author, signal }),
  ).rejects.toThrow(
    "`.yian/journal.jsonl` line 4 isn't a journal line; restore the journal or remove it " +
      "(undo history and change ids are lost)",
  );
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("x");
  await s.close();
});
