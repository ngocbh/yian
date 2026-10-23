/**
 * TS-only: a loaded flow keeps the text of every flow file the loader read, and `run_started`
 * records it, with the skills, the tool specs, the yian version and the parent run; resume
 * compiles the recorded flow, never the files on disk.
 */

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow, recordedFiles } from "../../src/compose/loader.ts";
import { handChild } from "./../compose/declarations.ts";

// Not resolved: on macOS the temp folder sits under the /var -> /private/var symlink.
const tmp = mkdtempSync(join(tmpdir(), "yian-run-started-sources-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let folders = 0;
/** A fresh folder under the test's temp folder. */
const freshDir = (name: string) => {
  const dir = join(tmp, `${name}-${++folders}`);
  mkdirSync(dir, { recursive: true });
  return dir;
};

/** Write each `files` text at its path under `dir`. */
function writeTree(dir: string, files: Readonly<Record<string, string>>): void {
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), text);
  }
}

/** A flow `id` taking `q` that calls `alias` (the `uses:` ref `ref`) with it. */
const caller = (id: string, alias: string, ref: string, extra = "") =>
  `id: ${id}\nname: ${id}\n${extra}input:\n  q: !str\nuses:\n  ${alias}: ${ref}\n` +
  `nodes:\n  c:\n    kind: call\n    call: ${alias}\n    input:\n      q: \${input.q}\noutput: \${c.output}\n`;

/** A flow `id` taking `q` that runs the tool `lookup` on it. */
const leaf = (id: string) =>
  `id: ${id}\nname: ${id}\ninput:\n  q: !str\n` +
  `nodes:\n  fetch:\n    kind: tool\n    tool_id: lookup\n    input:\n      q: \${input.q}\noutput: \${fetch.output}\n`;

const ROOT = caller("root", "child", "lib/child");
const CHILD = leaf("child");
const NESTED_CHILD = caller("child", "grand", "grand");
const GRAND = leaf("grand");
const PATHS_ROOT = caller("root", "util", "util", "system:\n  paths: [../shared]\n");
const UTIL = leaf("util");

/** The four load shapes: the files to write under a folder, and the folder the root is in. */
const SHAPES: Record<string, { files: Record<string, string>; root: string; keys: Record<string, string> }> = {
  "one level": {
    files: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
    root: ".",
    keys: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
  },
  "two levels": {
    files: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
    root: ".",
    keys: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
  },
  "system.paths": {
    files: { "proj/flow.yaml": PATHS_ROOT, "shared/util.yaml": UTIL },
    root: "proj",
    keys: { "flow.yaml": PATHS_ROOT, "../shared/util.yaml": UTIL },
  },
};

/** Load `<dir>/flow.yaml` as the CLI does: its folder searched first, named by its file name. */
async function loadRoot(dir: string): Promise<ComposedFlow> {
  const path = join(dir, "flow.yaml");
  return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dir], sourceLabel: basename(path) });
}

/** A flow's source as plain data. */
function sourceOf(flow: ComposedFlow) {
  const source = flow.source;
  if (source === null) return null;
  const { path, dir, searchPaths, files } = source;
  return { path, dir, searchPaths, files: Object.fromEntries(files) };
}

for (const [name, shape] of Object.entries(SHAPES)) {
  test(`a loaded flow keeps the text of each file it read: ${name}`, async () => {
    const dir = freshDir("load");
    writeTree(dir, shape.files);
    const root = join(dir, shape.root);

    expect(sourceOf(await loadRoot(root))).toEqual({
      path: "flow.yaml",
      dir: realpathSync(root),
      searchPaths: ["."],
      files: shape.keys,
    });
  });
}

for (const [name, shape] of Object.entries(SHAPES)) {
  test(`the recorded files load the same flow once the folder is gone: ${name}`, async () => {
    const dir = freshDir("reload");
    writeTree(dir, shape.files);
    const loaded = await loadRoot(join(dir, shape.root));
    const { path, dir: real, searchPaths, files } = loaded.source!;
    rmSync(dir, { recursive: true });

    const again = await loadFlow(files.get(path)!, {
      sourceLabel: path,
      baseDir: real,
      searchPaths: searchPaths.map((item) => join(real, item)),
      files: recordedFiles(real, files),
    });

    expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
    expect(sourceOf(again)).toEqual(sourceOf(loaded));
  });
}

test("a folder reached through a symlink keys its files from the real folder", async () => {
  const dir = freshDir("symlink");
  writeTree(join(dir, "real"), SHAPES["one level"]!.files);
  symlinkSync(join(dir, "real"), join(dir, "link"));

  const link = join(dir, "link");
  const loaded = await loadFlow(ROOT, { searchPaths: [link], baseDir: link, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    path: "flow.yaml",
    dir: realpathSync(join(dir, "real")),
    searchPaths: ["."],
    files: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
  });
});

test("a host resolver's loaded children join the root's files, keyed from the root's folder", async () => {
  const dir = freshDir("host");
  writeTree(dir, SHAPES["two levels"]!.files);
  const childResolver = async (ref: string) => {
    const path = join(dir, `${ref}.yaml`);
    return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dirname(path)], sourceLabel: basename(path) });
  };

  const loaded = await loadFlow(ROOT, { childResolver, baseDir: dir, sourceLabel: "flow.yaml" });

  // Resume finds each child where the host found it: in the root's folder.
  expect(sourceOf(loaded)).toEqual({
    path: "flow.yaml",
    dir: realpathSync(dir),
    searchPaths: ["."],
    files: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
  });
});

test("a flow built in code, or loaded over one, has no source", async () => {
  const hand = handChild({ output: null });
  expect(hand.source).toBeNull();
  expect(new ComposedFlow(hand.compiled).source).toBeNull();

  const dir = freshDir("hand");
  const text = "id: root\nname: root\nuses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\n";
  const over = await loadFlow(text, {
    childResolver: () => handChild({ output: null }),
    baseDir: dir,
  });
  expect(over.source).toBeNull();
});

test("a host child resume couldn't find where the host found it leaves the root without a source", async () => {
  const dir = freshDir("elsewhere");
  // The host reads the child from another folder, so the root's folder doesn't hold it.
  writeTree(dir, { "other/child.yaml": CHILD });
  const childResolver = async () => {
    const path = join(dir, "other", "child.yaml");
    return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dirname(path)], sourceLabel: basename(path) });
  };

  expect((await loadFlow(ROOT, { childResolver, baseDir: dir, sourceLabel: "flow.yaml" })).source).toBeNull();
});
