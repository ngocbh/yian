// Loading a flow confined to a root folder, from disk or from text not on disk.

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { loadConfined, loadConfinedText } from "../../src/cli/confined_load.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

const FLOW = "id: f\nname: f\ninput:\n  n: !int\nnodes: {}\noutput: ${input.n}\n";

/** A flow calling the flow `ref` through a `uses:` alias. */
function parentFlow(ref: string): string {
  return (
    "id: main\nname: main\ninput:\n  n: !int\nnodes:\n  work:\n    kind: call\n    call: child\n" +
    `    input:\n      n: \${input.n}\noutput: \${work.output}\nuses:\n  child: ${ref}\n`
  );
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
});

describe("loadConfinedText", () => {
  test("loads text not on disk, its children from the virtual path's folder", async () => {
    writeFileSync(join(root, "lib", "child.yaml"), FLOW);
    const flow = await loadConfinedText(root, join(root, "new.yaml"), parentFlow("lib/child"), NO_ENV);
    expect(flow.name).toBe("main");
    expect(flow.input.map((decl) => decl.name)).toEqual(["n"]);
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
