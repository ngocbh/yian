/**
 * A port of `tests/engine/test_cli_prompt.py`: `yian run`'s banner, input labels, typed input
 * prompts and run-error rendering.
 *
 * Ported: `test_parse_kv_keeps_structured_looking_values_as_strings`,
 * `test_structured_prompt_cancellation_bypasses_validation` (on `Terminal`; its scenario "Ctrl-C in
 * a list editor returns null" doesn't check that no verdict is drawn),
 * every `test_run_error_*` row (`renderRunError` over the seeds' conformance cases, loaded under
 * the Python seed file names as their labels) and
 * `test_any_input_from_the_command_line_reaches_the_flow_as_text`. Python's `_Loaded` is an
 * object with a `diagnose` method, and `replace(loaded, origin_resolver=None)` a `ComposedFlow`
 * over the same compiled flow without a resolver. The AGENT seeds e07 and e08 run with
 * `YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL` in their env, standing in for the suite-wide default
 * model Python's `tests/conftest.py` sets (`hostEnv` drops the shell's).
 *
 * Changed: the captured traceback under `engineTrace` is the box titled `traceback` (Python's
 * `python traceback`), holding the thrown error's stack.
 * `test_parse_structured_input_does_not_hide_validator_faults`: the fault comes from a malformed
 * type (a `list(int)` whose element bounds are reversed, so `buildValueAs` throws a plain
 * `RangeError`) rather than a patched validator; it must escape as that `RangeError`, not as a
 * `StructuredInputError` (itself a `RangeError`).
 *
 * Already ported:
 * - In `../cli/run.test.ts`: `test_flow_banner_name_and_description`,
 *   `test_flow_banner_includes_version` and `test_flow_banner_none_when_no_metadata`
 *   ("flowBanner"); `test_gather_non_tty_leaves_optional_and_defaulted_inputs_absent`,
 *   `test_gather_non_tty_reports_all_missing_required_inputs_in_order` and
 *   `test_gather_non_tty_filters_required_inputs_by_exact_key_membership` ("run's input gate",
 *   through `main`).
 * - In `../cli/terminal.test.ts` ("inputLabel"): `test_input_label_required`,
 *   `test_input_label_optional_no_default`, `test_input_label_optional_with_default`,
 *   `test_input_label_datetime_example`, `test_input_label_default_suppresses_example`,
 *   `test_input_label_derives_a_canonical_structural_type`,
 *   `test_input_label_displays_an_explicit_null_default` (Changed: `[default: null]`, decision 7)
 *   and `test_input_label_supports_a_flat_list_contract`. ("parseStructuredInput"; ints are
 *   bigints, decision 4a; messages quote in the text form, decision 7):
 *   `test_parse_structured_input_returns_native_arrays_and_objects`,
 *   `test_parse_structured_input_reports_concise_value_failures`,
 *   `test_parse_structured_input_rejects_null_at_nullable_structural_root`,
 *   `test_parse_structured_input_accepts_null_inside_nullable_element`,
 *   `test_parse_structured_input_keeps_flat_list_any_behavior` and
 *   `test_parse_structured_input_keeps_json_loads_nonfinite_number_behavior` (Changed: the tagged
 *   infinity form reads, bare `NaN`/`Infinity` are refused, decision 34).
 * - In `../cli/terminal_both_runtimes.ts`, each scenario naming its rows above itself, a real
 *   widget driven through a fake TTY (bool titles `true`/`false`, decision 7):
 *   `test_structured_prompt_retries_nested_failure_and_returns_native_value`,
 *   `test_object_prompt_returns_native_value_and_json_instruction`,
 *   `test_optional_structured_whitespace_is_validated_then_omitted`,
 *   `test_required_structured_whitespace_is_rejected_then_retried`,
 *   `test_nullable_structured_enter_rejects_textual_null_then_returns_native`,
 *   `test_nullable_open_scalar_enter_preserves_literal_null`,
 *   `test_nullable_open_prompt_cancellation_at_action`,
 *   `test_nullable_open_prompt_cancellation_in_editor`,
 *   `test_nullable_open_blank_can_return_to_omit`,
 *   `test_nullable_open_blank_can_return_to_set_null`,
 *   `test_nullable_open_blank_can_reenter_real_value`,
 *   `test_nullable_open_set_null_does_not_open_editor`,
 *   `test_required_nullable_scalar_editor_preserves_blank`,
 *   `test_required_nullable_structured_editor_retries_blank`,
 *   `test_nonnullable_defaulted_open_blank_stays_direct_omission`,
 *   `test_prompt_labels_carry_marks`, `test_prompt_skips_already_supplied`,
 *   `test_gather_treats_present_falsey_values_as_supplied_without_questionary`,
 *   `test_prompt_accepting_omission_preserves_start_owned_defaults` (its run half, START applying
 *   the defaults, in `../cli/run.test.ts`: "accepting each omission leaves START's defaults"),
 *   `test_prompt_collects_false_enum_and_text_overrides`,
 *   `test_finite_prompt_set_null_gathers_present_none`,
 *   `test_null_only_prompt_offers_controls_but_no_value_editor`,
 *   `test_nullable_finite_prompt_omission_stays_absent`,
 *   `test_nullable_literal_distinguishes_null_tag_from_set_null`,
 *   `test_nonnullable_finite_default_does_not_offer_set_null`,
 *   `test_required_null_capable_prompt_preserves_safe_default`,
 *   `test_prompt_distinguishes_literal_values_from_omission_labels`,
 *   `test_prompt_distinguishes_cancellation_from_omission`,
 *   `test_prompt_preserves_required_false_enum_and_blank_text`,
 *   `test_prompt_leaves_explicit_null_defaults_omitted_for_every_widget_kind`,
 *   `test_prompt_returns_optional_whitespace_to_palette_then_omits`,
 *   `test_any_prompt_uses_the_scalar_text_editor_without_set_null` and
 *   `test_optional_any_prompt_offers_set_null`.
 *
 * Not ported:
 * - `test_structured_prompt_does_not_turn_plain_value_error_into_a_verdict`: it monkeypatches a
 *   module function the prompt imports, and an ES module binding has no such seam; the malformed
 *   type that stands in for the parse row throws earlier here, in `inputLabel`'s
 *   `canonicalTypeLabel`, before the editor's validator runs.
 * - `test_nullable_open_repeated_palettes_use_fresh_choices` and
 *   `test_prompt_controls_are_fresh_for_each_widget`: questionary `Choice` object identity; TS
 *   draws its choices and shares no widget objects (the palette flows themselves are scenarios).
 * - `test_ensure_cwd_importable_prepends_cwd`: Python's `sys.path`; a TS CODE reference is a file
 *   path (decision 11).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { parseKv, renderRunError } from "../../src/cli/run.ts";
import { Terminal, parseStructuredInput } from "../../src/cli/terminal.ts";
import type { Writer } from "../../src/cli/utils.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Diagnostic, Failure, OriginRef } from "../../src/diagnostics.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { RunResult, runFlow } from "../../src/run.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { FakeInput, FakeOutput } from "../cli/terminal_both_runtimes.ts";
import { caseText, linkConformanceFixtures, loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-cli-prompt-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

// Seed e25's `uses:` child, on the search path beside the conformance fixtures its CODE names.
const libBoom = caseText("run-map-call/seed_e25_external_raise", { file: "lib_boom.yaml" }).text;
writeFileSync(join(dir, "lib_boom.yaml"), libBoom);
linkConformanceFixtures(dir);

/** The env the AGENT seeds run under: the default model Python's conftest sets for every test. */
const AGENT_ENV = hostEnv(dir, { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" });

/** Load conformance case `id`'s flow labelled `label`, with this file's directory on the search path. */
function loadSeed(id: string, label: string, runEnv = env): Promise<ComposedFlow> {
  return loadConformanceFlow(caseText(id).text, { searchPaths: [dir], sourceLabel: label, env: runEnv });
}

/** What `renderRunError` writes for `result` against `flow`. */
async function rendered(result: RunResult, flow: ComposedFlow, engineTrace = false): Promise<string> {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  await renderRunError(result, flow, engineTrace, out, env);
  return chunks.join("");
}

test("parse structured input does not hide validator faults", () => {
  // `new Type` doesn't validate its bounds, so the fault surfaces in `buildValueAs`.
  const malformed = new Type({
    kind: ValueKind.LIST_INTEGER,
    element: new Type({ kind: ValueKind.INTEGER, bounds: [5n, 1n] }),
  });

  let captured: unknown;
  try {
    parseStructuredInput("[3]", malformed, "x");
  } catch (error) {
    captured = error;
  }

  // A plain RangeError, not the StructuredInputError (a RangeError subclass) a verdict would be.
  expect(captured).toBeInstanceOf(RangeError);
  expect((captured as RangeError).constructor).toBe(RangeError);
  expect((captured as RangeError).name).toBe("RangeError");
  expect((captured as RangeError).message).toBe("type bounds must be ordered from minimum to maximum");
});

test("parse kv keeps structured looking values as strings", () => {
  const parsed = parseKv(["items=[1, 2]", 'profile={"count": 2}']);

  expect(parsed).toEqual({ items: "[1, 2]", profile: '{"count": 2}' });
  expect(Object.values(parsed).every((value) => typeof value === "string")).toBe(true);
});

test("structured prompt cancellation bypasses validation", async () => {
  const input = new FakeInput();
  const output = new FakeOutput();
  const terminal = new Terminal({ input, output });
  const items = new Parameter("items", {
    contract: new ValueContract(Type.scalar(ValueKind.LIST_STRING)),
    required: true,
    coerceStrings: true,
  });
  const gathered = terminal.promptInputs([items], {});

  // An unparseable draft, then Ctrl-C: the draft is never validated.
  input.write("[\x03");

  expect(await gathered).toBeNull();
  expect(output.text).not.toContain("Invalid JSON");
});

// --- renderRunError: a runtime failure boxes the .yaml at the precise line --- //

test("run error uses loaded canonical diagnosis", async () => {
  const failure = new Failure({
    code: "node.execute",
    message: "semantic failure",
    nodeId: "work",
    trace: [new OriginRef("site")],
  });
  const result = new RunResult({ input: {}, status: "failed", runId: "cli-test", log: null, failure });
  const seen: Failure[] = [];
  const loaded = {
    diagnose(candidate: Failure) {
      seen.push(candidate);
      return new Diagnostic({
        code: candidate.code,
        message: candidate.message,
        notes: ["canonical presentation"],
        nodeId: candidate.nodeId,
      });
    },
  };

  const out = await rendered(result, loaded as unknown as ComposedFlow);

  expect(seen).toEqual([failure]);
  expect(seen[0]).toBe(failure);
  expect(out).toContain("semantic failure");
  expect(out).toContain("canonical presentation");
});

test("run error boxes failing node", async () => {
  // e07 omits `as_of`, so the canonical failure trace points at the failing `:?` binding.
  const loaded = await loadSeed("run-core/seed_e07_required_missing", "e07-required-missing.yaml", AGENT_ENV);
  const result = await runFlow(loaded, { topic: "X" }, { env: AGENT_ENV });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e07-required-missing.yaml:23"); // the as_of binding, not the node header
  expect(out).toContain("report"); // still inside the source window
  expect(out).toContain("as_of is required for the report");
});

test("run error uses reference code origin", async () => {
  // A referenced CODE failure carries its authored `code:` scalar in the canonical origin trace.
  const loaded = await loadSeed("run-leaves/seed_e20_code_raises", "e20-code-raises.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e20-code-raises.yaml:12");
  expect(out).toContain("╭");
});

test("run error boxes boundary assert line", async () => {
  // e18's boundary assert fails before any node runs, at the exact `asserts:` expression.
  const loaded = await loadSeed("run-core/seed_e18_false_boundary_assert", "e18-false-boundary-assert.yaml");
  const result = await runFlow(loaded, { window: -5n }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e18-false-boundary-assert.yaml:15"); // the `input.window > 0` line
  expect(out).toContain("assert failed");
});

test("run error boxes code wrong type output", async () => {
  // e21's CODE returns a value its declared `output: int` refuses, at the `output:` declaration.
  const loaded = await loadSeed("run-leaves/seed_e21_code_wrong_type", "e21-code-wrong-type.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e21-code-wrong-type.yaml:12"); // the `output: int` field line
  expect(out).toContain("╭");
});

test("run error boxes flow input line", async () => {
  // e08's `window` can't coerce to int, so its origin selects the declaration.
  const loaded = await loadSeed("run-core/seed_e08_input_type_mismatch", "e08-input-type-mismatch.yaml", AGENT_ENV);
  const result = await runFlow(loaded, { topic: "X", window: "soon" }, { env: AGENT_ENV });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e08-input-type-mismatch.yaml:20"); // the `window: int` declaration line
  expect(out).toContain("╭");
});

test("run error plain when unlocatable", async () => {
  // A failure with no authored origin stays a semantic message with no frame.
  const result = new RunResult({
    input: {},
    status: "failed",
    runId: "cli-test",
    log: null,
    failure: new Failure({ code: "graph.terminal_unreachable", message: "boom", nodeId: null }),
    events: [],
  });
  const loaded = await loadSeed("run-core/seed_e18_false_boundary_assert", "e18-false-boundary-assert.yaml");
  const out = await rendered(result, loaded);
  expect(out.trim()).toBe("boom");
  expect(out).not.toContain("╭"); // no boxed frame
});

test("run error renders namespaced failure origin trace", async () => {
  // The failure trace carries the owning CALL site despite the namespaced node id (`run/boom`).
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("e24-nested-code-raise.yaml:27"); // the owning `run:` call node line
  expect(out).toContain("╭"); // boxed, not a plain `run failed:` line
});

test("run error boxes loop max exceeded at loop node", async () => {
  // The LOOP origin stays put even when the runtime driver id carries a respawn suffix (`polish~5`).
  const loaded = await loadFlow(caseText("load-loop/seed_28_refine_loop").text, {
    sourceLabel: "28-refine-loop.yaml",
    env,
  });
  const loopOrigin = loaded.compiled.nodes.get("polish")!.origin;
  expect(loopOrigin).not.toBeNull();
  const result = new RunResult({
    input: {},
    status: "failed",
    runId: "cli-test",
    log: null,
    failure: new Failure({
      code: "node.execute",
      message: "loop 'polish' exceeded max (5)",
      nodeId: "polish~5",
      trace: [loopOrigin!],
      causeType: "LoopMaxExceeded",
    }),
    events: [],
  });
  const out = await rendered(result, loaded);
  expect(out).toContain("╭"); // boxed, not a plain `run failed:` line
  expect(out).toContain("28-refine-loop.yaml:45"); // the authored `polish:` loop node
  expect(out).toContain("exceeded max (5)");
});

test("run error traceback only under engine trace", async () => {
  // The captured stack of a CODE throw shows only under `engineTrace`.
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.failure).not.toBeNull();
  expect(result.failure!.traceback).toBeTruthy(); // captured at the node-failure boundary

  expect(await rendered(result, loaded, false)).not.toContain("─ traceback ");

  const out = await rendered(result, loaded, true);
  expect(out).toContain("─ traceback ");
  expect(out).toContain("intentional CODE failure"); // the originating error
});

// --- renderRunError: ordered canonical frames across child flows --- //

test("run error multi frame stack into def", async () => {
  // e24's trace resolves the top-level CALL site, then the failing CODE origin inside `defs:inner`.
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("e24-nested-code-raise.yaml:27"); // frame 0: the owning `run:` call node
  expect(out).toContain("e24-nested-code-raise.yaml:20"); // frame 1: the authored `code:` value
  expect(out).toContain("intentional CODE failure");
});

test("run error multi frame stack into external", async () => {
  // e25's origins cross the file boundary: the CALL site here, the failing CODE in lib_boom.yaml.
  const loaded = await loadSeed("run-map-call/seed_e25_external_raise", "e25-external-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("e25-external-raise.yaml:19"); // frame 0: the `go:` call node
  expect(out).toContain("lib_boom.yaml:18"); // frame 1: the external file's `code:` line
});

test("run error multi frame three levels", async () => {
  // e26's origin trace resolves all three levels, outermost to innermost.
  const loaded = await loadSeed("run-map-call/seed_e26_three_level_raise", "e26-three-level-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("e26-three-level-raise.yaml:37"); // frame 0: the top `outer:` call node
  expect(out).toContain("e26-three-level-raise.yaml:29"); // frame 1: `via:` call in `middle`
  expect(out).toContain("e26-three-level-raise.yaml:20"); // frame 2: `boom` code in `deep`
});

test("run error multi frame on nested human input failure", async () => {
  // Seed 25 asks from inside a called `defs:` child (`gate/approve`); an answer
  // outside `Approval` fails the typed write, framed at the `gate:` CALL, then the child's `output:`.
  const loaded = await loadSeed("run-map-call/seed_25_nested_suspension", "25-nested-suspension.yaml");
  const result = await runFlow(loaded, { action: "ship it" }, { env, tools: { ask: () => "maybe" } });
  expect(result.status).toBe("failed");
  const out = await rendered(result, loaded);
  expect(out).toContain("source locations:");
  expect(out).toContain("25-nested-suspension.yaml:41"); // frame 0: the `gate:` call node
  expect(out).toContain("25-nested-suspension.yaml:31"); // frame 1: the precise `output:` line
  expect(out).toContain("is not a member of variant");
});

test("run error unavailable presentation stays unlocated", async () => {
  // A missing resolver can't be bypassed by inferring locations from the graph.
  const loaded = await loadSeed("run-map-call/seed_e24_nested_code_raise", "e24-nested-code-raise.yaml");
  const result = await runFlow(loaded, { topic: "X" }, { env });
  expect(result.status).toBe("failed");
  const unresolved = new ComposedFlow(loaded.compiled, {
    version: loaded.version,
    name: loaded.name,
    description: loaded.description,
    importDepth: loaded.importDepth,
  });
  const out = await rendered(result, unresolved);
  expect(out).not.toContain("╭");
  expect(out).toContain("Source presentation was unavailable.");
  expect(out).toContain("intentional CODE failure");
});

const ANY_INPUT_FLOW = `id: echo
name: echo
input:
  x: !Any
nodes:
  keep:
    kind: code
    input:
      x: \${input.x}
    code: |
      return inputs["x"]
    output: Any
output: \${keep.output}
`;

test("any input from the command line reaches the flow as text", async () => {
  const result = await runFlow(await loadFlow(ANY_INPUT_FLOW, { env }), parseKv(["x=3"]), { env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("3");
});
