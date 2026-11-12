// Loading a flow confined to a root folder, from disk or from text not on disk.

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { loadConfined, loadConfinedText } from "../../src/cli/confined_load.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

const FLOW = "id: f\nname: f\ninput:\n  n: !int\nnodes: {}\noutput: ${input.n}\n";

/** A flow calling the flow `ref` through a `uses:` alias. */
function parentFlow(ref: string): string {
  return (
    "id: main\nname: main\ninput:\n  n: !int\nnodes:\n  work:\n    kind: call\n    call: child\n" +
    `    input:\n      n: \${input.n}\noutput: \${work.output}\nuses:\n  child: ${ref}\n`
  );
}

/** A flow calling the flow `ref` twice, through the `uses:` aliases `a` and `b`. */
function twiceFlow(id: string, ref: string, refB: string = ref): string {
  return (
    `id: ${id}\nname: ${id}\nnodes:\n  x:\n    kind: call\n    call: a\n  y:\n    kind: call\n    call: b\n` +
    `output: \${x.output}\nuses:\n  a: ${ref}\n  b: ${refB}\n`
  );
}

/**
 * Write the diamond chain `l0.yaml` .. `l<depth>.yaml` into `dir`: each level calls the next one
 * through two aliases, so the last level is reached along 2^depth import paths.
 */
function writeDiamond(dir: string, depth: number): void {
  for (let i = 0; i < depth; i++) writeFileSync(join(dir, `l${i}.yaml`), twiceFlow(`l${i}`, `l${i + 1}`));
  writeFileSync(join(dir, `l${depth}.yaml`), `id: l${depth}\nname: l${depth}\nnodes:\n  v: !str leaf\noutput: \${v.output}\n`);
}

/**
 * An env that otherwise acts as `NO_ENV`, and how many flows have loaded through it so far: every
 * `loadFlow` call reads each `YIAN_*` variable once, before it loads, so the reads of one variable
 * count the loads (the file reader is a named import, which a spy can't see).
 */
function countingEnv(): { env: Env; loads: () => number } {
  let reads = 0;
  const env: Env = (name) => {
    if (name === "YIAN_MAX_CALL_NESTING") reads++;
    return NO_ENV(name);
  };
  return { env, loads: () => reads };
}

let tmp: string;
let root: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-confined-load-")));
  root = join(tmp, "company");
  mkdirSync(join(root, "lib"), { recursive: true });
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** The `LoadError` `promise` rejects with. */
async function loadError(promise: Promise<unknown>): Promise<LoadError> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof LoadError) return error;
    throw error;
  }
  throw new Error("expected a LoadError");
}

describe("loadConfined", () => {
  test("loads a flow and its uses: child inside the root", async () => {
    writeFileSync(join(root, "main.yaml"), parentFlow("lib/child"));
    writeFileSync(join(root, "lib", "child.yaml"), FLOW);
    const flow = await loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV);
    expect(flow.name).toBe("main");
  });

  test("a uses: child imported along several paths loads once", async () => {
    writeDiamond(root, 4);
    const { env, loads } = countingEnv();
    await loadConfined(root, join(root, "l0.yaml"), new Set(), env);
    expect(loads()).toBe(5);
  });

  test("a cycle through a child imported twice is still a cycle", async () => {
    writeFileSync(join(root, "l0.yaml"), twiceFlow("l0", "l1"));
    const l1 = "id: l1\nname: l1\nnodes:\n  x:\n    kind: call\n    call: c\noutput: ${x.output}\nuses:\n  c: l0\n";
    writeFileSync(join(root, "l1.yaml"), l1);
    const error = await loadError(loadConfined(root, join(root, "l0.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.load");
    expect(error.diagnostic.message).toContain("flow-reference cycle at l0.yaml");
  });

  test("a child imported under two aliases is held to each alias's pinned version", async () => {
    writeFileSync(join(root, "main.yaml"), twiceFlow("main", "lib/child", "lib/child@2"));
    const child = 'id: child\nname: child\nversion: "1"\nnodes:\n  v: !str leaf\noutput: ${v.output}\n';
    writeFileSync(join(root, "lib", "child.yaml"), child);
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.load");
    expect(error.diagnostic.message).toContain("pinned lib/child@2");
  });

  test("an absolute uses: ref outside the root is outside the root, even with a decoy at the joined path", async () => {
    writeFileSync(join(tmp, "evil.yaml"), FLOW);
    // A flow could ship this copy to pass the check while `yian run` loads the file outside.
    mkdirSync(join(root, tmp), { recursive: true });
    writeFileSync(join(root, tmp, "evil.yaml"), FLOW);
    writeFileSync(join(root, "main.yaml"), parentFlow(join(tmp, "evil")));
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.outside_root");
    expect(error.diagnostic.message).toBe(`${join("..", "evil.yaml")} is outside the root`);
  });

  test("an absolute uses: ref to a missing file outside the root is outside the root, not missing", async () => {
    writeFileSync(join(root, "main.yaml"), parentFlow("/etc/passwd"));
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.outside_root");
  });

  test("an absolute uses: ref inside the root loads that file, not a copy at the joined path", async () => {
    writeFileSync(join(root, "lib", "child.yaml"), "id: [broken\n");
    mkdirSync(join(root, root, "lib"), { recursive: true });
    writeFileSync(join(root, root, "lib", "child.yaml"), FLOW);
    writeFileSync(join(root, "main.yaml"), parentFlow(join(root, "lib", "child")));
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.load");
    expect(error.diagnostic.message).toContain("invalid YAML");
  });

  test.each([
    ["a bare ref", () => "lib/child"],
    ["a ./ ref", () => "./lib/child"],
    ["a ../ ref", () => "../shared/child"],
    ["an absolute ref", () => join(root, "shared", "child")],
  ])("%s loads the child the default loader loads", async (_name, ref) => {
    const app = join(root, "app");
    mkdirSync(join(app, "lib"), { recursive: true });
    mkdirSync(join(root, "shared"), { recursive: true });
    writeFileSync(join(app, "lib", "child.yaml"), FLOW.replace("name: f", "name: app-lib"));
    writeFileSync(join(root, "shared", "child.yaml"), FLOW.replace("name: f", "name: shared"));
    // Where a join would put an absolute ref.
    mkdirSync(join(app, root, "shared"), { recursive: true });
    writeFileSync(join(app, root, "shared", "child.yaml"), FLOW.replace("name: f", "name: decoy"));
    const text = parentFlow(ref());
    writeFileSync(join(app, "main.yaml"), text);
    const confined = await loadConfined(root, join(app, "main.yaml"), new Set(), NO_ENV);
    const options = { searchPaths: [app], baseDir: app, codeRoot: root, sourceLabel: join("app", "main.yaml"), env: NO_ENV };
    const searched = await loadFlow(text, options);
    expect(confined.source).toEqual(searched.source);
    expect(Object.values(confined.source!.files).map((file) => file.text)).not.toContain(
      FLOW.replace("name: f", "name: decoy"),
    );
  });

  // Guards against a regression: the default loader also shares one flow per file between importers.
  test("loads a diamond to the flow the default loader loads", async () => {
    writeDiamond(root, 3);
    const confined = await loadConfined(root, join(root, "l0.yaml"), new Set(), NO_ENV);
    const options = { searchPaths: [root], baseDir: root, codeRoot: root, sourceLabel: "l0.yaml", env: NO_ENV };
    const searched = await loadFlow(twiceFlow("l0", "l1"), options);
    expect(confined.source).toEqual(searched.source);
    expect(confined.name).toBe(searched.name);
    expect(confined.input).toEqual(searched.input);
  });
});

describe("loadConfined child errors", () => {
  test("a child that isn't UTF-8 names its path from the root and the reason", async () => {
    writeFileSync(join(root, "main.yaml"), parentFlow("lib/child"));
    writeFileSync(join(root, "lib", "child.yaml"), new Uint8Array([0x69, 0x64, 0x3a, 0x20, 0xff, 0xfe, 0x0a]));
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.message).toBe(
      "callable 'child' (uses: 'lib/child'): could not resolve: can't read lib/child.yaml: not UTF-8 text",
    );
  });

  test("a ref written with .yaml says to drop the extension", async () => {
    writeFileSync(join(root, "main.yaml"), parentFlow("./lib/child.yaml"));
    writeFileSync(join(root, "lib", "child.yaml"), FLOW);
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.message).toBe(
      "callable 'child' (uses: './lib/child.yaml'): could not resolve: flow './lib/child.yaml' not found: " +
        "lib/child.yaml.yaml; write a uses: ref without its extension",
    );
  });

  test("a ref whose file is .yml says only .yaml files are imported", async () => {
    writeFileSync(join(root, "main.yaml"), parentFlow("lib/child"));
    writeFileSync(join(root, "lib", "child.yml"), FLOW);
    const error = await loadError(loadConfined(root, join(root, "main.yaml"), new Set(), NO_ENV));
    expect(error.diagnostic.message).toBe(
      "callable 'child' (uses: 'lib/child'): could not resolve: flow 'lib/child' not found: lib/child.yaml; " +
        "uses: imports only .yaml files",
    );
  });
});

describe("loadConfinedText", () => {
  test("loads text not on disk, its children from the virtual path's folder", async () => {
    writeFileSync(join(root, "lib", "child.yaml"), FLOW);
    const flow = await loadConfinedText(root, join(root, "new.yaml"), parentFlow("lib/child"), NO_ENV);
    expect(flow.name).toBe("main");
    expect(flow.input.map((decl) => decl.name)).toEqual(["n"]);
  });

  test("a uses: child imported along several paths loads once", async () => {
    writeDiamond(root, 4);
    const { env, loads } = countingEnv();
    await loadConfinedText(root, join(root, "new.yaml"), twiceFlow("new", "l1"), env);
    expect(loads()).toBe(5);
  });

  test("refuses a uses: child outside the root", async () => {
    writeFileSync(join(tmp, "child.yaml"), FLOW);
    const error = await loadError(loadConfinedText(root, join(root, "new.yaml"), parentFlow("../child"), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.outside_root");
    expect(error.diagnostic.message).toBe(`${join("..", "child.yaml")} is outside the root`);
  });

  test("a uses: child that doesn't exist is not found", async () => {
    const error = await loadError(loadConfinedText(root, join(root, "new.yaml"), parentFlow("lib/missing"), NO_ENV));
    expect(error.diagnostic.code).toBe("compose.load");
    expect(error.diagnostic.message).toEndWith(`flow 'lib/missing' not found: ${join("lib", "missing.yaml")}`);
  });

  test("refuses a virtual path outside the root", async () => {
    const error = await loadError(loadConfinedText(root, join(tmp, "new.yaml"), FLOW, NO_ENV));
    expect(error.diagnostic.code).toBe("compose.outside_root");
  });

  test("labels a parse error with the virtual path relative to the root", async () => {
    const error = await loadError(loadConfinedText(root, join(root, "lib", "bad.yaml"), "id: [\n", NO_ENV));
    expect(error.diagnostic.code).toBe("compose.load");
    expect(error.diagnostic.frames[0]?.label).toBe(join("lib", "bad.yaml"));
  });
});
