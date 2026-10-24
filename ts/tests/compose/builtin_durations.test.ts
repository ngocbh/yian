/** A built-in's duration (`timeout:`, `for:`) is expanded to seconds at load; `sleep`'s inputs. */
import { expect, test } from "bun:test";

import { bindingSource, outputContractType } from "../../src/nodes/contracts.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";
import { assembleText, loadError } from "./declarations.ts";

/** Return a flow with inbox `m`, input `t` of type `tType`, and a take with `timeout` line `timeout`. */
const takeFlow = (timeout: string, tType = "str") => `id: f
name: f
input:
  m: !Inbox(str)
  t: !${tType}
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
      timeout: ${timeout}
output: \${input.t}
`;

/** Return the wiring source of input `name` of node `nodeId` in flow `text`. */
function wired(text: string, nodeId: string, name: string): unknown {
  return bindingSource(assembleText(text).flow.compiled.wiring.get(nodeId)!.get(name));
}

const refused = (text: string) => loadError(() => assembleText(text)).message;

test.each(["24h", "86400s"])("timeout: %s compiles to seconds", (timeout) => {
  expect(wired(takeFlow(timeout), "heard", "timeout")).toBe(86400n);
});

test("a plain number is refused with a hint in seconds", () => {
  const message = refused(takeFlow("86400"));
  expect(message).toContain("node 'heard' (tool_id: take) input 'timeout': a duration is text such as 24h");
  expect(message).toContain("write 86400s");
});

test("a malformed duration is refused", () => {
  expect(refused(takeFlow("1.5h"))).toContain("a duration is a positive integer and a unit");
});

test("a reference typed str loads unchanged", () => {
  expect(wired(takeFlow("${input.t}"), "heard", "timeout")).toBe("${input.t}");
});

test("a reference not typed str is refused", () => {
  expect(refused(takeFlow("${input.t}", "int"))).toContain("a duration reference must be typed str");
});
