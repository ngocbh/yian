/** FileMemory: its identity, members, version, recall, read tools and writes. */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BUILTIN_MEMORY_TOOLS, checkContract } from "../../src/memory/base.ts";
import { FileMemory, memoryVersion, readMembers } from "../../src/memory/file.ts";

let dir: string;
let root: string;
const signal = new AbortController().signal;
const o = { create: true, itemType: null, signal };

beforeEach(() => {
  dir = fs.mkdtempSync(join(tmpdir(), "yian-fm-"));
  root = join(dir, "m");
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Write `text` to `rel` under `base`, making its folders. */
function put(base: string, rel: string, text: string | Uint8Array): void {
  fs.mkdirSync(join(base, rel, ".."), { recursive: true });
  fs.writeFileSync(join(base, rel), text);
}

test("a new memory takes its id, or at-<16 hex> of the real path, and stores it", async () => {
  const m = new FileMemory({ root: join(dir, "a") });
  const s = await m.open(o);
  expect(s.created).toBe(true);
  expect(s.info.identity).toMatch(/^at-[0-9a-f]{16}$/);
  expect(s.info.identity).toBe(`at-${sha256(fs.realpathSync(join(dir, "a"))).slice(0, 16)}`);
  expect(fs.readFileSync(join(dir, "a/.yian/identity"), "utf8")).toBe(s.info.identity);
  expect(fs.statSync(join(dir, "a")).mode & 0o777).toBe(0o700);
  expect(fs.statSync(join(dir, "a/.yian")).mode & 0o777).toBe(0o700);
  expect(fs.statSync(join(dir, "a/.yian/identity")).mode & 0o777).toBe(0o600);
  await s.close();

  const named = await new FileMemory({ root: join(dir, "b"), id: "people/ada" }).open(o);
  expect(named.info.identity).toBe("people/ada");
  await named.close();
});

test("a symlinked root is resolved once, at construction, and its identity hashes the real path", async () => {
  fs.mkdirSync(join(dir, "real"));
  fs.symlinkSync(join(dir, "real"), join(dir, "link"));
  const s = await new FileMemory({ root: join(dir, "link") }).open(o);
  expect(s.info.identity).toBe(`at-${sha256(fs.realpathSync(join(dir, "real"))).slice(0, 16)}`);
  expect(fs.existsSync(join(dir, "real/.yian/identity"))).toBe(true);
  await s.close();
});

test("a stored identity wins; a different constructor id is refused", async () => {
  await (await new FileMemory({ root, id: "people/ada" }).open(o)).close();
  await expect(new FileMemory({ root, id: "people/bob" }).open(o)).rejects.toThrow(
    "this memory is `people/ada`, not `people/bob`",
  );
  const again = await new FileMemory({ root }).open(o);
  expect(again.created).toBe(false);
  expect(again.info.identity).toBe("people/ada");
  await again.close();
  const resumed = await new FileMemory({ root, id: "people/ada" }).open({ ...o, create: false });
  expect(resumed.created).toBe(false);
  await resumed.close();
});

test("open({create: false}) refuses a missing memory and creates nothing", async () => {
  await expect(new FileMemory({ root }).open({ ...o, create: false })).rejects.toThrow(
    `\`${root}\` isn't a memory yet: there is no folder`,
  );
  expect(fs.existsSync(root)).toBe(false);
});

test("open({create: true}) adopts an existing folder with files and no .yian/, keeping them", async () => {
  put(root, "profile.md", "Ada.");
  fs.chmodSync(root, 0o755);
  const s = await new FileMemory({ root, id: "people/ada", index: "profile.md" }).open(o);
  expect(s.created).toBe(true);
  expect(fs.readFileSync(join(root, "profile.md"), "utf8")).toBe("Ada.");
  expect(fs.readFileSync(join(root, ".yian/identity"), "utf8")).toBe("people/ada");
  // An adopted folder keeps its own mode; only what the memory makes is private.
  expect(fs.statSync(root).mode & 0o777).toBe(0o755);
  await s.close();
});

test("open({create: false}) refuses an existing folder with no .yian/: it isn't a memory yet", async () => {
  put(root, "profile.md", "Ada.");
  await expect(new FileMemory({ root }).open({ ...o, create: false })).rejects.toThrow(
    `\`${root}\` isn't a memory yet`,
  );
  expect(fs.existsSync(join(root, ".yian"))).toBe(false);
});

test("a root that is a file, or a .yian that isn't a folder, is refused", async () => {
  put(dir, "file", "x");
  await expect(new FileMemory({ root: join(dir, "file") }).open(o)).rejects.toThrow("isn't a folder");
  put(root, ".yian", "x");
  await expect(new FileMemory({ root }).open(o)).rejects.toThrow("`.yian` isn't a folder");
});

test("an item type is refused: a FileMemory holds files, not typed items", async () => {
  await expect(new FileMemory({ root }).open({ ...o, itemType: { type: "object" } })).rejects.toThrow(
    "a FileMemory holds files, not typed items",
  );
});

test("the session's info holds the type, identity, settings with no location, and the contract's tools", async () => {
  const s = await new FileMemory({ root, id: "t/x", maxFiles: 7, index: "INDEX.md" }).open(o);
  expect(s.info.type).toBe("FileMemory");
  expect(s.info.tools.map((tool) => tool.name)).toEqual([...BUILTIN_MEMORY_TOOLS.FileMemory]);
  expect(s.info.tools).toEqual(FileMemory.contract.tools);
  expect(s.info.settings).toEqual({
    index: "INDEX.md",
    max_file_chars: 100_000n,
    max_total_chars: 1_000_000n,
    max_files: 7n,
    max_depth: 4n,
    max_recall_chars: 8_000n,
    max_read_chars: 20_000n,
    max_results: 100n,
    max_undo: 100n,
    max_change_ids: 10_000n,
    max_history_chars: 1_000_000n,
  });
  expect(JSON.stringify(s.info.settings, (_, v) => (typeof v === "bigint" ? `${v}` : v))).not.toContain(dir);
  expect(() => checkContract(new FileMemory({ root }))).not.toThrow();
  for (const tool of FileMemory.contract.tools) {
    expect(tool.description).not.toMatch(/\d/);
  }
  await s.close();
});

test("members: hidden, badly named, binary and oversized files are skipped; a symlink is refused", () => {
  put(root, "MEMORY.md", "index");
  put(root, "notes/a.md", "a");
  put(root, ".hidden.md", "h");
  put(root, ".git/config", "g");
  put(root, "-dash.md", "d");
  put(root, "sp ace.md", "s");
  put(root, "bin.dat", new Uint8Array([0xff, 0xfe, 0x00]));
  put(root, "big.md", "x".repeat(11));
  put(dir, "outside.md", "o");
  fs.symlinkSync(join(dir, "outside.md"), join(root, "link.md"));
  fs.symlinkSync(join(root, "notes"), join(root, "linked"));
  put(root, "a/b/c/d/deep.md", "ok");
  put(root, "a/b/c/d/e/deeper.md", "too deep");
  put(root, ".yian/identity", "t/x");

  const members = readMembers(root, { maxDepth: 4, maxFileChars: 10 });
  expect(members.files.map((file) => file.path)).toEqual(["MEMORY.md", "a/b/c/d/deep.md", "notes/a.md"]);
  // `.yian/` is the memory's own and never counts; everything else that isn't a member does.
  expect(members.skipped).toBe(9);
  expect(members.files[0]).toEqual({ path: "MEMORY.md", text: "index", hash: sha256("index"), chars: 5 });
});

test("a member's size counts characters, not bytes", () => {
  put(root, "a.md", "é😀");
  expect(readMembers(root, { maxDepth: 4, maxFileChars: 2 }).files.map((file) => file.chars)).toEqual([2]);
});

test("the version is sha256 over sorted (path, file hash) pairs of members only", () => {
  put(root, "b.md", "B");
  put(root, "a/z.md", "Z");
  put(root, "a.md", "A");
  put(root, ".git/config", "ignored");
  put(root, "bin.dat", new Uint8Array([0xff]));
  const members = readMembers(root, { maxDepth: 4, maxFileChars: 100 });
  const pairs = [
    ["a.md", sha256("A")],
    ["a/z.md", sha256("Z")],
    ["b.md", sha256("B")],
  ];
  expect(memoryVersion(members.files)).toBe(sha256(JSON.stringify(pairs)));
  fs.writeFileSync(join(root, ".git/config"), "changed");
  expect(memoryVersion(readMembers(root, { maxDepth: 4, maxFileChars: 100 }).files)).toBe(
    sha256(JSON.stringify(pairs)),
  );
  expect(memoryVersion([])).toBe(sha256("[]"));
});

test.each([
  ["../x.md", "a `..` part"],
  ["a/../../x.md", "a `..` part"],
  ["./x.md", "a `.` part"],
  ["a//x.md", "an empty part"],
  [".yian/identity", "a hidden name"],
  [".git/config", "a hidden name"],
  ["-x.md", "a name other than letters, digits, `.`, `_` and `-`, starting with a letter or digit"],
  ["/etc/passwd", "an absolute path"],
  ["a/b/c/d/e/f.md", "more than 4 folders deep"],
  ["", "an empty path"],
])("an index option %s is refused at construction, naming why", (index, why) => {
  expect(() => new FileMemory({ root, index })).toThrow(/index/);
  expect(() => new FileMemory({ root, index })).toThrow(why);
});

test("the depth cap counts the folders above a file", () => {
  expect(() => new FileMemory({ root, index: "a/b/c/d/x.md" })).not.toThrow();
  expect(() => new FileMemory({ root, index: "a/b/x.md", maxDepth: 1 })).toThrow("more than 1 folder deep");
});

test("bad settings are refused at construction", () => {
  expect(() => new FileMemory({ root: "" })).toThrow("root");
  expect(() => new FileMemory({ root, id: "" })).toThrow("id");
  expect(() => new FileMemory({ root, maxFiles: 0 })).toThrow("maxFiles");
  expect(() => new FileMemory({ root, maxDepth: 1.5 })).toThrow("maxDepth");
  expect(() => new FileMemory({ root, lockTimeoutMs: -1 })).toThrow("lockTimeoutMs");
  expect(() => new FileMemory({ root, maxUndo: 0 })).not.toThrow();
});
