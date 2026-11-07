/**
 * `long_term_memory:` is gone: on a flow, a def or an agent it is a load error at its key that
 * says to declare the memory as an input, and no allowed-key list names it.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import {
  assembled,
  type CaseRow,
  fileOf,
  loadErrorAsync,
  parsedNodes,
  runCase,
  runTextCase,
} from "../helpers/declarations.ts";

test.each([
  "load-core/long_term_memory__at_the_flow_is_refused",
  "load-core/seed_e09_unknown_top_level_key",
])("parseFile: %s", (id) => runTextCase({ id, call: fileOf }));

test.each<CaseRow>([
  { id: "load-core/long_term_memory__in_a_def_is_refused", call: assembled },
  { id: "load-map-call/engine_defs__def_unknown_field_is_rejected", call: assembled },
  { id: "load-core/long_term_memory__on_an_agent_is_refused", call: parsedNodes },
  { id: "load-core/compose_node_when_hint__when_on_an_agent_node", call: parsedNodes },
  { id: "load-agent/seed_e40_agent_controls_no_range", call: parsedNodes },
])("$id", runCase);

test("the older memory_policy: spelling is refused with the same fix", async () => {
  const error = await loadErrorAsync(loadFlow("id: f\nname: f\nmemory_policy: reflective\nnodes: {}\n"));
  expect(error.message).toBe(
    "`memory_policy:` was removed: declare the memory as an input (`lessons: !FileMemory`) and render " +
      "`${lessons}` in a prompt",
  );
});
