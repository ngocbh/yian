/**
 * Input binding: resolve a node's declared params into a typed input record (the read boundary).
 *
 * The node/flow split: a node declares its signature as source-neutral `Parameter` values (name
 * plus contract/default policy, no source); the flow owns the sources in
 * `CompiledFlow.wiring[nodeId][param]`. `bindParams` joins each parameter to its `wiring[name]`
 * source (a template value: a whole `${...}` resolved against the pool with its type kept, an
 * embedded `${...}` written as text, or a literal), building the `{name: value}` record the
 * engine's `evalNode` hands the node instead of the pool. Type-checking consumes an optional
 * resolved `ValueContract`; untyped parameters accept any resolved value. A typed `Binding`
 * additionally checks its resolved value against the binding's own declared contract and
 * converts only its file and object representation (`normalizeFiles`): a file's JSON form at a
 * `file` position becomes a `FileRef`, a nested `FileRef` at an `object` position becomes its
 * record.
 *
 * A `ValueTemplate` source is an authored container: each leaf string is evaluated as a template
 * at its position and each nested typed `Binding` is checked against its own contract. A plain
 * record or list source is always a literal.
 *
 * A MAP body binds a per-element `item` scope: `${item}` / `${item.path}` resolve from the
 * current element passed as `item` (a body-local scope, not a pool head). Without `item` this is
 * the ordinary non-MAP bind path.
 *
 * Binding never converts a string. Supplied text is converted only at the run boundary:
 * `coerceInputs` applies each top-level parameter's `coerceStrings` policy before START binds,
 * so a value crossing a CALL/MAP/LOOP edge into a child START is validated as is.
 *
 * Imports `expr` for source evaluation and `typesys` for contract validation; both sit
 * below/beside `nodes` in the layer ladder.
 */

import { compareCodeUnits, type OriginRef, reprText } from "../diagnostics.ts";
import type { ResolvedValue } from "../expr/lineage.ts";
import { buildValueAs, isRecord, TypeCheckError, type Value, ValueKind, type ValueRecord } from "../typesys/index.ts";
import { fromHost, jsTypeName, rejectDeepNesting, setField, shownText } from "../typesys/values.ts";
import { Binding, isExactly, Parameter, type ValuePath, validateOptionalOrigin } from "./contracts.ts";

const BINDING_FAILURE_CODES = ["binding.evaluate", "binding.type", "contract.input"] as const;

/** The semantic stage a `BindingError` failed at. */
export type BindingFailureCode = (typeof BINDING_FAILURE_CODES)[number];

/** Resolves a reference path to its value and route. */
export type SourceResolver = (path: string) => ResolvedValue;

/**
 * Resolves a node's reserved source by name and applies the node's normalization to the value
 * before retaining it.
 */
export type ReservedBinder = (name: string, source: unknown, normalize: (value: unknown) => unknown) => unknown;

/** Names the place inside a parameter's value an error is at, from its path there. */
export type ErrorLabel = string | ((path: ValuePath) => string);

/**
 * An expected binding operation failed with one semantic stage and origin.
 *
 * `failureCode` distinguishes expression/default/required evaluation (`binding.evaluate`), a
 * value violating its typed binding's declared type (`binding.type`), and a supplied value
 * violating its input contract (`contract.input`). `origin` is the most precise available
 * authored operation and feeds the canonical failure trace.
 */
export class BindingError extends RangeError {
  override name = "BindingError";
  /** The semantic stage that failed. */
  readonly failureCode: BindingFailureCode;
  /** The most precise authored operation known; callers fill it in when it is `null`. */
  origin: OriginRef | null;

  /**
   * Retain one validated binder-owned semantic classification; `origin` defaults to `null`.
   *
   * @throws RangeError when `failureCode` isn't a binding failure code.
   * @throws TypeError when `origin` is neither `null` nor an `OriginRef`.
   */
  constructor(
    message: string,
    options: { failureCode: BindingFailureCode; origin?: OriginRef | null; cause?: unknown },
  ) {
    const { failureCode, origin = null, cause } = options;
    super(message, cause === undefined ? undefined : { cause });
    if (!(BINDING_FAILURE_CODES as readonly string[]).includes(failureCode)) {
      throw new RangeError(`invalid binding failure code ${reprText(String(failureCode))}`);
    }
    validateOptionalOrigin(origin, "binding failure origin");
    this.failureCode = failureCode;
    this.origin = origin;
  }
}

/** Return one canonical binding carrier's origin, or `null` for raw input. Used by node kinds. */
export function sourceOrigin(source: unknown): OriginRef | null {
  return source instanceof Binding ? source.origin : null;
}

/**
 * Throw a `binding.evaluate` `BindingError` if `value` is not a yian value. Used by node kinds.
 *
 * A value is rejected when it nests deeper than `YIAN_MAX_VALUE_NESTING` levels, or when host
 * intake refuses it at any depth: NaN, `undefined`, text holding a lone surrogate, or anything
 * but a plain record, a plain array, a `FileRef` or a scalar. Infinity is a value.
 */
export function rejectInvalidBinding(value: unknown, where: string, origin: OriginRef | null): void {
  try {
    // Depth first, so intake never meets an unbounded value. Intake only checks here: its copy
    // would read an integral float `2` as the int `2n`.
    rejectDeepNesting(value, where);
    fromHost(value, where);
  } catch (error) {
    if (error instanceof TypeCheckError) {
      throw new BindingError(error.message, { failureCode: "binding.evaluate", origin, cause: error });
    }
    throw error;
  }
}

/**
 * Require the canonical exact-array parameter carrier at every binding seam. Used by node kinds.
 *
 * @throws TypeError when `params` isn't a plain array or holds anything but exact `Parameter`s.
 */
export function validateParameters(params: readonly Parameter[]): void {
  if (!Array.isArray(params) || Object.getPrototypeOf(params) !== Array.prototype) {
    throw new TypeError(`node params must be an exact list, got ${jsTypeName(params)}`);
  }
  for (const parameter of params) {
    if (!isExactly(parameter, Parameter)) {
      throw new TypeError(`node params must contain Parameter values, got ${jsTypeName(parameter)}`);
    }
  }
}

const TRUE_SPELLINGS: ReadonlySet<string> = new Set(["1", "true", "yes", "y", "on"]);
const FALSE_SPELLINGS: ReadonlySet<string> = new Set(["0", "false", "no", "n", "off"]);

// ASCII digits with single `_` separators, as Python's `int()` and `float()` read them.
const DIGITS = "[0-9](?:_?[0-9])*";
const INT_TEXT = new RegExp(`^[+-]?${DIGITS}$`);
const FLOAT_TEXT = new RegExp(`^[+-]?(?:${DIGITS}(?:\\.(?:${DIGITS})?)?|\\.${DIGITS})(?:[eE][+-]?${DIGITS})?$`);

/** Return an int spelling (`INT_TEXT`) as its `bigint`. */
function intFromText(text: string): bigint {
  try {
    return BigInt(text.replaceAll("_", ""));
  } catch (error) {
    // A runtime refuses an enormous int with its own RangeError.
    if (error instanceof RangeError) throw new TypeCheckError("integer is too large for this runtime");
    throw error;
  }
}

/**
 * Coerce a supplied string under one parameter's explicit scalar policy.
 *
 * Only text is converted, and only for a parameter with `coerceStrings` and a contract. After
 * `trim()`, an `int` input reads an int spelling (`[+-]?` and ASCII digits with single `_`
 * separators) at any size. A `float` input reads `inf` and `-inf` as infinity, an int spelling
 * as that int, and a float spelling (`2.0`, `1e3`, `.5`) as a float. A `bool` input reads
 * `1`/`true`/`yes`/`y`/`on` and `0`/`false`/`no`/`n`/`off`, ignoring case. Text that does not
 * convert is returned as given, untrimmed, for the contract to reject.
 *
 * @returns the converted int, float or boolean, or `value` unchanged.
 * @throws TypeError when `parameter` is not an exact `Parameter`.
 * @throws TypeCheckError `number is too large for a float` when a float spelling overflows a
 *   double, and `integer is too large for this runtime` when the runtime refuses an int.
 */
export function coerceParam(parameter: Parameter, value: unknown): unknown {
  if (!isExactly(parameter, Parameter)) {
    throw new TypeError(`parameter must be an exact Parameter, got ${jsTypeName(parameter)}`);
  }
  if (typeof value !== "string" || !parameter.coerceStrings || parameter.contract === null) {
    return value;
  }

  const raw = value.trim();
  const kind = parameter.contract.type.kind;
  if (kind === ValueKind.INTEGER) {
    return INT_TEXT.test(raw) ? intFromText(raw) : value;
  }
  if (kind === ValueKind.NUMBER) {
    if (raw === "inf") return Infinity;
    if (raw === "-inf") return -Infinity;
    if (INT_TEXT.test(raw)) return intFromText(raw);
    if (!FLOAT_TEXT.test(raw)) return value;
    const number = Number(raw.replaceAll("_", ""));
    if (!Number.isFinite(number)) throw new TypeCheckError("number is too large for a float");
    // yian has no negative zero, so `-0.0` and `-1e-400` read as 0.
    return number === 0 ? 0 : number;
  }
  if (kind === ValueKind.BOOLEAN) {
    const spelling = raw.toLowerCase();
    if (TRUE_SPELLINGS.has(spelling)) return true;
    if (FALSE_SPELLINGS.has(spelling)) return false;
    // Any other text stays a string so the bool contract rejects it.
    return value;
  }
  return value;
}

/**
 * Reject top-level inputs that are not a record or name an undeclared input.
 *
 * @throws TypeError when `raw` is not a record.
 * @throws RangeError when a key names no declared input; the message lists the unknown names,
 *   sorted and cut to the shown-value cap, and the declared names.
 */
export function checkInputNames(params: readonly Parameter[], raw: unknown): void {
  if (!isRecord(raw)) {
    throw new TypeError(`inputs must be a mapping of input names to values, got ${jsTypeName(raw)}`);
  }
  const declared = params.map((parameter) => parameter.name);
  const unknown = Object.keys(raw)
    .filter((name) => !declared.includes(name))
    .sort(compareCodeUnits);
  if (unknown.length > 0) {
    const noun = unknown.length === 1 ? "input" : "inputs";
    const shown = shownText(unknown.map(reprText).join(", "));
    throw new RangeError(`unknown ${noun} ${shown}; declared inputs: ${declared.join(", ") || "(none)"}`);
  }
}

/**
 * Coerce supplied top-level values, each keyed by a declared input name, into a fresh record.
 *
 * @throws TypeError when `params` isn't canonical or `raw` is not a record.
 * @throws RangeError when a key names no declared input.
 * @throws TypeCheckError when an input's text can't be read, as `input '<name>': <reason>`.
 */
export function coerceInputs(
  params: readonly Parameter[],
  raw: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  validateParameters(params);
  checkInputNames(params, raw);
  const byName = new Map(params.map((parameter) => [parameter.name, parameter]));
  // The `as Value` casts below only satisfy `setField`: the record goes back to the caller
  // unchecked, as Python's dict does.
  const coerced: ValueRecord = {};
  for (const name of Object.keys(raw)) {
    try {
      setField(coerced, name, coerceParam(byName.get(name)!, raw[name]) as Value);
    } catch (error) {
      if (error instanceof TypeCheckError) throw new TypeCheckError(`input ${reprText(name)}: ${error.message}`);
      throw error;
    }
  }
  return coerced;
}

/**
 * Fill validated defaults for omitted parameters into a fresh record, without coercing payloads.
 *
 * A present own key, `null` included, always shadows a default. Defaults are validated against
 * any semantic contract but are kept as given, not as the typed value system's normalized form.
 *
 * @throws BindingError with `failureCode` `binding.evaluate` when a default nests deeper than
 *   `YIAN_MAX_VALUE_NESTING` levels, is not a yian value (NaN, for one), or violates its
 *   parameter's contract.
 */
export function applyDefaults(
  params: readonly Parameter[],
  inputs: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  validateParameters(params);
  // The `as Value` casts below only satisfy `setField`: the record goes back to the caller
  // unchecked, as Python's dict does.
  const output: ValueRecord = {};
  for (const name of Object.keys(inputs)) setField(output, name, inputs[name] as Value);
  for (const parameter of params) {
    if (Object.hasOwn(output, parameter.name) || !parameter.hasDefault) continue;
    const origin = parameter.defaultOrigin ?? parameter.origin;
    rejectInvalidBinding(parameter.default, `default for ${reprText(parameter.name)}`, origin);
    if (parameter.contract !== null) {
      try {
        buildValueAs(parameter.contract.type, parameter.default);
      } catch (error) {
        if (error instanceof TypeCheckError) {
          throw new BindingError(error.message, { failureCode: "binding.evaluate", origin, cause: error });
        }
        throw error;
      }
    }
    setField(output, parameter.name, parameter.default as Value);
  }
  return output;
}
