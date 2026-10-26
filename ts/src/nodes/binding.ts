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
import {
  ConstructorTypeError,
  type DeclaredTypes,
  ExpressionError,
  evalBinding,
  evalTemplate,
  type RootResolver,
  resolveReference,
  scanTemplate,
} from "../expr/index.ts";
import { ResolvedValue, ValueRoute } from "../expr/lineage.ts";
import { Span } from "../expr/template.ts";
import {
  buildValueAs,
  isRecord,
  normalizeFiles,
  type Type,
  TypeCheckError,
  type TypedValue,
  type Value,
  ValueKind,
  type ValueRecord,
  type VariablePool,
} from "../typesys/index.ts";
import {
  fromHost,
  isList,
  jsTypeName,
  rejectDeepNesting,
  renderValuePath,
  setField,
  shownText,
  valueRepr,
} from "../typesys/values.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  Parameter,
  type ValueContract,
  type ValuePath,
  ValueTemplate,
  validateOptionalOrigin,
} from "./contracts.ts";

/** The failure codes of a binding stage, the ones an evaluation fails with before its inputs are bound. */
export const BINDING_FAILURE_CODES = ["binding.evaluate", "binding.type", "contract.input"] as const;

/** The semantic stage a `BindingError` failed at. */
export type BindingFailureCode = (typeof BINDING_FAILURE_CODES)[number];

/** Reads a reference's root, with its value and route (`RootResolver`). */
export type SourceResolver = RootResolver;

/**
 * Resolves a node's reserved source by name and applies the node's normalization to the value
 * before retaining it.
 */
export type ReservedBinder = (name: string, source: unknown, normalize: (value: unknown) => unknown) => unknown;

/** Names the place inside a parameter's value an error is at, from its path there. */
export type ErrorLabel = string | ((path: ValuePath) => string);

/**
 * Detached delivered inputs and the routes evaluated while binding them. Frozen.
 *
 * An incomplete bind has no usable `boundInputs`. Its route and evaluated keys retain only the
 * safe context already collected before the failure. `evaluatedKeys` distinguishes a template
 * evaluated here from a materialized string passed through literally, even when evaluation
 * produced a literal route.
 */
export class BindingCapture {
  readonly #boundInputs: ValueRecord | null;
  readonly #evaluatedKeys: ReadonlySet<string>;
  /** The record route of every recorded input, by name. */
  readonly route: ValueRoute;
  /** Each item record's capture under the index its node gave it, in recording order. */
  readonly items: readonly (readonly [number, BindingCapture])[];

  /** Retain a capture; `boundInputs` is `null` for an incomplete bind. */
  constructor(
    boundInputs: ValueRecord | null,
    route: ValueRoute,
    evaluatedKeys: ReadonlySet<string>,
    items: readonly (readonly [number, BindingCapture])[] = [],
  ) {
    this.#boundInputs = boundInputs;
    this.route = route;
    this.#evaluatedKeys = new Set(evaluatedKeys);
    this.items = Object.freeze(items.map(([index, capture]) => Object.freeze([index, capture] as const)));
    Object.freeze(this);
  }

  /** Return a fresh copy of the complete pre-execution input record, or `null` when incomplete. */
  get boundInputs(): ValueRecord | null {
    return this.#boundInputs === null ? null : (copyValue(this.#boundInputs) as ValueRecord);
  }

  /** Return a fresh set of the inputs whose source evaluated a `${...}` span. */
  get evaluatedKeys(): ReadonlySet<string> {
    return new Set(this.#evaluatedKeys);
  }
}

/** Collect one parameter record at its existing read and copying boundary. */
export class BindingCollector {
  #values: ValueRecord = {};
  #routes = new Map<string, ValueRoute>();
  #evaluated = new Set<string>();
  #items = new Map<number, BindingCollector>();
  #complete = false;
  #frozen = false;

  #requireOpen(): void {
    if (this.#frozen) throw new Error("binding capture is already frozen");
  }

  /**
   * Retain a route from the same scan and evaluation that produced a value.
   *
   * Used by the binder and the node evaluator (a reserved value's route); not for hosts.
   *
   * @throws Error when the collector is frozen.
   */
  recordSource(name: string, route: ValueRoute, evaluated: boolean): void {
    this.#requireOpen();
    this.#routes.set(name, route);
    if (evaluated) this.#evaluated.add(name);
    else this.#evaluated.delete(name);
  }

  /**
   * Snapshot the complete delivered record before node execution.
   *
   * @param record The exact record returned by binding, including any materialized values
   *   supplied by node-owned hooks; it is copied. A name with no recorded route gets a literal one.
   * @throws Error when the collector is frozen.
   */
  finish(record: ValueRecord): void {
    this.#requireOpen();
    const values = copyValue(record) as ValueRecord;
    this.#values = values;
    this.#routes = new Map(
      Object.keys(values).map((name) => [name, this.#routes.get(name) ?? ValueRoute.literal(values[name])]),
    );
    for (const name of this.#evaluated) {
      if (!Object.hasOwn(record, name)) this.#evaluated.delete(name);
    }
    this.#complete = true;
  }

  /**
   * Return an already evaluated input route for local-scope projection.
   *
   * @throws RangeError when no route is recorded under `name`.
   */
  routeFor(name: string): ValueRoute {
    const route = this.#routes.get(name);
    if (route === undefined) throw new RangeError(`no route recorded for binding ${reprText(name)}`);
    return route;
  }

  /**
   * Allocate an item record under the explicit index supplied by its node.
   *
   * @throws Error when the collector is frozen.
   * @throws RangeError when `index` is not a nonnegative integer or was already recorded.
   */
  forItem(index: number): BindingCollector {
    this.#requireOpen();
    if (!Number.isSafeInteger(index) || index < 0) {
      throw new RangeError("binding item index must be a nonnegative integer");
    }
    if (this.#items.has(index)) throw new RangeError(`binding item index ${index} was already recorded`);
    const collector = new BindingCollector();
    this.#items.set(index, collector);
    return collector;
  }

  /**
   * Keep partial context while discarding the usable input record, here and in every item.
   *
   * @throws Error when the collector is frozen.
   */
  invalidate(): void {
    this.#requireOpen();
    this.#complete = false;
    for (const item of this.#items.values()) item.invalidate();
  }

  /**
   * Close this collector and its items and return its detached binding capture.
   *
   * @throws Error when the collector is already frozen.
   */
  freeze(): BindingCapture {
    this.#requireOpen();
    this.#frozen = true;
    return new BindingCapture(
      this.#complete ? this.#values : null,
      ValueRoute.record(this.#routes),
      this.#evaluated,
      [...this.#items].map(([index, item]) => [index, item.freeze()] as const),
    );
  }
}

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
 * @param subject What owns the params, as the error messages name it.
 * @throws TypeError when `params` isn't a plain array or holds anything but exact `Parameter`s.
 */
export function validateParameters(params: readonly Parameter[], subject = "node"): void {
  if (!isList(params)) {
    throw new TypeError(`${subject} params must be an exact list, got ${jsTypeName(params)}`);
  }
  for (const parameter of params) {
    if (!isExactly(parameter, Parameter)) {
      throw new TypeError(`${subject} params must contain Parameter values, got ${jsTypeName(parameter)}`);
    }
  }
}

const TRUE_SPELLINGS: ReadonlySet<string> = new Set(["1", "true", "yes", "y", "on"]);
const FALSE_SPELLINGS: ReadonlySet<string> = new Set(["0", "false", "no", "n", "off"]);

// ASCII digits with single `_` separators, as Python's `int()` and `float()` read them, except
// that Python also reads other Unicode decimal digits and these spellings don't.
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
    const shown = shownText(unknown.map((name) => JSON.stringify(name)).join(", "));
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
 * @param subject What owns the params, as a params error names it.
 * @throws TypeError as `validateParameters`.
 * @throws BindingError with `failureCode` `binding.evaluate` when a default nests deeper than
 *   `YIAN_MAX_VALUE_NESTING` levels, is not a yian value (NaN, for one), or violates its
 *   parameter's contract.
 */
export function applyDefaults(
  params: readonly Parameter[],
  inputs: Readonly<Record<string, unknown>>,
  subject = "node",
): Record<string, unknown> {
  validateParameters(params, subject);
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

/** Spell a parameter name and a path inside its value, as in `p.b[0]`. */
function pathLabel(name: string, path: ValuePath): string {
  return name + path.map((part) => (typeof part === "string" ? `.${part}` : `[${part}]`)).join("");
}

/**
 * Name a binding error's location: a parameter (or its label) and a path inside it.
 *
 * Without a label this is `binding 'p.b[0]'`. With a text label it is the label, followed by
 * ` at <path>` when `path` is not empty, the path rendered as host intake renders it (`a[1].b`).
 * A function label names `path` itself.
 */
function bindingWhere(name: string, path: ValuePath, label: ErrorLabel | null): string {
  if (label === null) return `binding ${reprText(pathLabel(name, path))}`;
  if (typeof label === "function") return label(path);
  return path.length > 0 ? `${label} at ${renderValuePath(path)}` : label;
}

/** A source part's value, its route, and whether it evaluated a `${...}` span. */
type SourceRead = [value: unknown, route: ValueRoute | null, evaluated: boolean];

/** Evaluate template strings for one parameter, with or without route capture. */
class SourceReader {
  readonly pool: VariablePool;
  readonly item: Value | ResolvedValue;
  readonly trace: boolean;
  readonly sourceResolver: SourceResolver | null;
  readonly types: ReadonlyMap<string, Type> | null;
  readonly declared: DeclaredTypes | null;
  readonly itemType: Type | null;

  constructor(
    pool: VariablePool,
    item: Value | ResolvedValue,
    options: {
      trace: boolean;
      sourceResolver: SourceResolver | null;
      types: ReadonlyMap<string, Type> | null;
      declared: DeclaredTypes | null;
      itemType: Type | null;
    },
  ) {
    this.pool = pool;
    this.item = item;
    this.trace = options.trace;
    this.sourceResolver = options.sourceResolver;
    this.types = options.types;
    this.declared = options.declared;
    this.itemType = options.itemType;
  }

  /**
   * Return a template string's value, its route, and whether it had spans.
   *
   * The route and the spans flag are computed only when tracing; otherwise they are `null` and
   * `false`.
   *
   * @throws ExpressionError (a `RequiredError` among them) when the template does not parse or
   *   evaluate.
   */
  string(text: string): SourceRead {
    if (!this.trace) {
      // The item may be a traced caller's `ResolvedValue`; an untraced read needs just its value.
      const item = this.item instanceof ResolvedValue ? this.item.value : this.item;
      const { types, declared, itemType } = this;
      const resolve = (names: readonly string[]) => resolveReference(names, this.pool);
      const value = evalBinding(text, resolve, { item, types, declared, itemType });
      return [value, null, false];
    }
    const resolve = this.sourceResolver ?? ((names: readonly string[]) => resolveReference(names, this.pool));
    const segments = scanTemplate(text);
    const { item, types, declared, itemType } = this;
    const result = evalTemplate(segments, resolve, { trace: true, item, types, declared, itemType });
    return [result.value, result.route, segments.some((segment) => segment instanceof Span)];
  }

  /**
   * Evaluate one node of a template tree at `path`.
   *
   * Each leaf string is a template; a nested typed `Binding` evaluates its source in place, then
   * checks the result against its contract, converting only its file and object representation.
   * Other leaves are literals. `label`, when set, names the parameter in an error message in
   * place of `binding '<name>'`.
   *
   * @throws BindingError when a leaf fails to evaluate (located at its registered origin, else
   *   the nearest enclosing binding's origin, else `origin`), a nested typed binding's value is
   *   not a yian value (too deep, or refused by host intake), or it violates its contract.
   */
  tree(
    node: unknown,
    path: ValuePath,
    name: string,
    template: ValueTemplate,
    origin: OriginRef | null,
    label: ErrorLabel | null,
  ): SourceRead {
    if (node instanceof Binding) {
      let [value, route, evaluated] = this.tree(node.source, path, name, template, node.origin ?? origin, label);
      if (node.contract !== null) {
        const where = bindingWhere(name, path, label);
        try {
          // A value that is not a yian value fails as such, not as a type mismatch.
          rejectInvalidBinding(value, where, node.origin);
          value = checkBindingType(where, node, node.contract, value);
        } catch (error) {
          if (error instanceof BindingError && error.origin === null) {
            error.origin = template.originAt(path) ?? origin;
          }
          throw error;
        }
      }
      return [value, route, evaluated];
    }
    if (typeof node === "string") {
      try {
        return this.string(node);
      } catch (error) {
        if (error instanceof ExpressionError) {
          throw new BindingError(`${bindingWhere(name, path, label)}: ${error.message}`, {
            failureCode: failureCodeOf(error),
            origin: template.originAt(path) ?? origin,
            cause: error,
          });
        }
        throw error;
      }
    }
    if (isRecord(node)) {
      const values: ValueRecord = {};
      const fields: [string, ValueRoute][] = [];
      let anyEvaluated = false;
      for (const key of Object.keys(node)) {
        const [value, route, evaluated] = this.tree(node[key], [...path, key], name, template, origin, label);
        // The tree's values are checked at the binding boundary; the cast only satisfies `setField`.
        setField(values, key, value as Value);
        if (route !== null) fields.push([key, route]);
        anyEvaluated ||= evaluated;
      }
      return [values, this.trace ? ValueRoute.record(fields) : null, anyEvaluated];
    }
    if (Array.isArray(node)) {
      const items: unknown[] = [];
      const routes: ValueRoute[] = [];
      let anyEvaluated = false;
      for (const [index, child] of node.entries()) {
        const [value, route, evaluated] = this.tree(child, [...path, index], name, template, origin, label);
        items.push(value);
        if (route !== null) routes.push(route);
        anyEvaluated ||= evaluated;
      }
      const listRoute = this.trace
        ? new ValueRoute("list", {
            children: routes,
            expression: `[${routes.map((route) => route.expression).join(", ")}]`,
          })
        : null;
      return [items, listRoute, anyEvaluated];
    }
    return [node, this.trace ? ValueRoute.literal(node) : null, false];
  }
}

/**
 * Resolve one raw or carrier-wrapped source via the template language. Used by node kinds.
 *
 * Exactly one `Binding` layer is removed at this read boundary; a `Binding` directly inside it
 * stays an ordinary literal payload. A string that is exactly one `${...}` gives the typed value
 * of that expression; a `${...}` embedded in surrounding text is written into it as text; a
 * string with no `${...}` is a plain literal (after `$$` becomes `$`). `${item}` /
 * `${item.path}` resolve from the MAP-body-local `item` scope. A `ValueTemplate` evaluates each
 * leaf string the same way at its position and checks each nested typed `Binding`; its route is
 * a record or list of the parts' routes. Any other source, a plain record or list included, is a
 * literal.
 *
 * `collector` (default `null`) retains the selected route under `name`; `sourceResolver`
 * (default `null`) pairs values with accepted producer identities in one read, and is used only
 * with a `collector`. Without it, capture records no producer identity. `types` (default `null`,
 * no types) is the table an `isinstance(x, T)` or a `T(...)` constructor looks `T` up in.
 * `declared` (default `null`) gives a reference root's declared type and `itemType` (default
 * `null`) `item`'s, so a field they declare reads null when it's absent.
 * `label`, when set, names the parameter in a template error message in place of
 * `binding '<name>'`; a function label names the path inside the template the error is at.
 * `name` defaults to `""` and `item` to `null`.
 *
 * @throws BindingError when a source does not parse or evaluate (`binding.evaluate`), a
 *   `${T(...)}` check fails (`binding.type`), or a nested typed binding rejects its value:
 *   `expr`'s errors never cross the `nodes <- expr` boundary.
 * @throws Error when `collector` is frozen, or a traced read produced no route.
 */
export function resolveSource(
  source: unknown,
  pool: VariablePool,
  options: {
    item?: Value | ResolvedValue;
    collector?: BindingCollector | null;
    name?: string;
    label?: ErrorLabel | null;
    sourceResolver?: SourceResolver | null;
    types?: ReadonlyMap<string, Type> | null;
    declared?: DeclaredTypes | null;
    itemType?: Type | null;
  } = {},
): unknown {
  const { item = null, collector = null, name = "", label = null, sourceResolver = null, types = null } = options;
  const { declared = null, itemType = null } = options;
  const origin = sourceOrigin(source);
  const sourceValue = bindingSource(source);
  if (typeof sourceValue !== "string" && !(sourceValue instanceof ValueTemplate)) {
    collector?.recordSource(name, ValueRoute.literal(sourceValue), false);
    return sourceValue;
  }
  const reader = new SourceReader(pool, item, { trace: collector !== null, sourceResolver, types, declared, itemType });
  let read: SourceRead;
  if (sourceValue instanceof ValueTemplate) {
    read = reader.tree(sourceValue.value, [], name, sourceValue, origin, label);
  } else {
    try {
      read = reader.string(sourceValue);
    } catch (error) {
      if (error instanceof ExpressionError) {
        throw new BindingError(error.message, { failureCode: failureCodeOf(error), origin, cause: error });
      }
      throw error;
    }
  }
  const [value, route, evaluated] = read;
  if (collector !== null) {
    if (route === null) throw new Error(`traced binding ${reprText(name)} produced no source route`);
    collector.recordSource(name, route, evaluated);
  }
  return value;
}

/** Classify a source evaluation error: a failed `${T(...)}` check is a type error. */
function failureCodeOf(error: ExpressionError): BindingFailureCode {
  return error instanceof ConstructorTypeError ? "binding.type" : "binding.evaluate";
}

/**
 * Validate a resolved value against its typed binding's declared contract.
 *
 * The check runs before the parameter contract. It delivers the payload with only its file and
 * object representation read as the declared type reads it (`normalizeFiles`): a file's JSON
 * form at a `file` position becomes a `FileRef`, and a nested `FileRef` at an `object` position
 * becomes its record. Nothing else converts (an int declared `float` stays an int), and a string
 * never satisfies a non-string declared type.
 *
 * @param where Names the binding in the failure message, e.g. `binding 'p'`.
 * @param binding The typed binding, whose origin is the fallback location.
 * @param contract The binding's declared contract.
 * @param value The value the binding's source resolved to, already checked to be a yian value.
 * @returns the value to deliver; `value` itself when nothing converts.
 * @throws BindingError with `failureCode` `binding.type` when the value violates the declared
 *   type, located at the nearest contract origin for the rejected path, else the binding's origin.
 */
function checkBindingType(where: string, binding: Binding, contract: ValueContract, value: unknown): Value {
  try {
    buildValueAs(contract.type, value);
  } catch (error) {
    if (error instanceof TypeCheckError) {
      throw new BindingError(`${where}: ${error.message}`, {
        failureCode: "binding.type",
        origin: contract.origins.at(error.path) ?? binding.origin,
        cause: error,
      });
    }
    throw error;
  }
  return normalizeFiles(contract.type, value as Value);
}

/**
 * Return a fresh copy of a delivered value: new arrays and records (own data properties), with
 * `FileRef`s and scalars kept. A container shared inside `value` stays shared in the copy. Used
 * by node kinds.
 */
export function copyValue(value: Value, copies: Map<object, Value> = new Map()): Value {
  if (Array.isArray(value) || isRecord(value)) {
    // The memo also bounds the work: a value reusing one child at each of its allowed levels
    // would otherwise take time exponential in its depth to copy.
    const shared = copies.get(value);
    if (shared !== undefined) return shared;
    if (Array.isArray(value)) {
      const items: Value[] = [];
      copies.set(value, items);
      for (const item of value) items.push(copyValue(item, copies));
      return items;
    }
    const record: ValueRecord = {};
    copies.set(value, record);
    for (const key of Object.keys(value)) setField(record, key, copyValue(value[key] as Value, copies));
    return record;
  }
  // yian has no negative zero: a float that becomes a binding's value is written as 0.
  return Object.is(value, -0) ? 0 : value;
}

/**
 * Resolve each declared `Parameter` into a typed record: the engine's read boundary.
 *
 * Joins each param to its source in the flow-owned `wiring` (`name -> source`): coalesce,
 * default, required, type check, deep copy and the `item` scope. A present typed `Binding` first
 * checks its resolved value against the binding's declared contract, converting only its file
 * and object representation; the parameter's own contract then runs as usual. An omitted input
 * (no wiring entry) fills its declared default or fails when required; a present source that
 * resolves to `null` shadows the default (`f(x=None)` semantics).
 *
 * @param params The node-side declared signature to bind.
 * @param wiring The flow-owned `name -> source` map for this node.
 * @param pool The pool each `${...}` source resolves against.
 * @param options.item The current element for a MAP body's `${item}` scope; `null` (the
 *   default) is the non-MAP path.
 * @param options.collector Retains delivered values and routes independently of node execution;
 *   `null` (the default) captures nothing.
 * @param options.sourceResolver One-read value/presence/source lookup used only when capturing
 *   bindings; `null` (the default) reads the pool and records no producer identity.
 * @param options.types The type table an `isinstance(x, T)` or a `T(...)` in a source looks `T`
 *   up in, keyed by type name (the node's `types`); `null` (the default) means no types.
 * @param options.declared The declared type of a reference root, given its names (the flow's pool
 *   roots); `null` (the default) leaves every field read undeclared.
 * @param options.itemType The declared type of `item`; `null` (the default) leaves it undeclared.
 * @param options.labels Param name -> author-facing phrase an error message uses in place of
 *   `input '<param>'` / `binding '<param>'` (the node's binding labels); a param without one
 *   keeps the default text. Defaults to `null`.
 * @param options.finishCapture The evaluator's seam (default `true`): `false` retains routes while
 *   deferring the input snapshot until reserved values and wrapper defaults are complete.
 * @returns the bound `{name: value}` record handed to the node instead of the pool, built with
 *   own data properties.
 * @throws TypeError when `params` isn't canonical.
 * @throws BindingError on an expression, default or required failure, a resolved value or filled
 *   default that nests deeper than `YIAN_MAX_VALUE_NESTING` levels or is refused by host intake
 *   (NaN, or a non-plain object such as a nested `Binding` carrier in a literal source, which
 *   Python passed through as a literal; both fail with `binding.evaluate`), a typed binding's
 *   rejection, or a supplied input's contract rejection; `failureCode` names the stage.
 */
export function bindParams(
  params: readonly Parameter[],
  wiring: ReadonlyMap<string, unknown>,
  pool: VariablePool,
  options: {
    item?: Value | ResolvedValue;
    collector?: BindingCollector | null;
    sourceResolver?: SourceResolver | null;
    types?: ReadonlyMap<string, Type> | null;
    labels?: ReadonlyMap<string, string> | null;
    finishCapture?: boolean;
    declared?: DeclaredTypes | null;
    itemType?: Type | null;
  } = {},
): ValueRecord {
  const { item = null, collector = null, sourceResolver = null, types = null, labels = null, finishCapture = true } =
    options;
  const { declared = null, itemType = null } = options;
  validateParameters(params);
  collector?.invalidate();
  const record: ValueRecord = {};
  for (const p of params) {
    try {
      // An absent entry (the caller omitted the input) vs a present one, bound-null included.
      const present = wiring.has(p.name);
      let usedDefault = false;
      const source = present ? wiring.get(p.name) : null;
      const label = labels?.get(p.name) ?? null;
      const inputWhere = label ?? `input ${reprText(p.name)}`;
      let value = resolveSource(source, pool, {
        item,
        collector: present ? collector : null,
        name: p.name,
        label,
        sourceResolver,
        types,
        declared,
        itemType,
      });
      rejectInvalidBinding(value, inputWhere, sourceOrigin(source));
      if (present && source instanceof Binding && source.contract !== null) {
        value = checkBindingType(bindingWhere(p.name, [], label), source, source.contract, value);
      }
      if (value === null) {
        // Only an omitted input fills its declared default or fails when required: a present
        // source that resolves to null shadows the default (`f(x=None)` semantics, as
        // `applyDefaults`). Loader-built body parameters carry no default; START parameters do,
        // so the spliced child START owns omitted-input defaulting.
        if (!present && p.hasDefault) {
          value = p.default;
          usedDefault = true;
          rejectInvalidBinding(value, `default for ${reprText(p.name)}`, p.defaultOrigin ?? p.origin);
          collector?.recordSource(p.name, ValueRoute.literal(value), false);
        } else if (!present && p.required) {
          throw new BindingError(`required input ${reprText(p.name)} is unbound (from=${valueRepr(source)})`, {
            failureCode: "binding.evaluate",
            origin: p.origin,
          });
        } else if (!present) {
          // Omitted, no default, not required: leave the key absent so the wrapper's default
          // can fill it (or the render throws on an unbound name).
          continue;
        }
        // Otherwise present but null: an explicit bound-null that still passes through the
        // optional contract (nullable or untyped accepts it; a non-nullable contract refuses).
      }
      if (p.contract !== null) {
        // No string conversion here: a CALL/MAP/LOOP edge is not a flow boundary. The run
        // converts the top-level flow's supplied inputs (`coerceInputs`) before START binds them.
        let checked: TypedValue;
        try {
          checked = buildValueAs(p.contract.type, value);
        } catch (error) {
          if (!(error instanceof TypeCheckError)) throw error;
          throw new BindingError(`${inputWhere}: ${error.message}`, {
            failureCode: usedDefault ? "binding.evaluate" : "contract.input",
            origin: usedDefault ? (p.defaultOrigin ?? p.origin) : (p.contract.origins.at(error.path) ?? p.origin),
            cause: error,
          });
        }
        if (!usedDefault) value = checked.toObject();
      }
      // `value` passed `rejectInvalidBinding`, so it is a yian value.
      setField(record, p.name, copyValue(value as Value));
    } catch (error) {
      if (error instanceof BindingError && error.origin === null) error.origin = p.origin;
      throw error;
    }
  }
  if (collector !== null && finishCapture) collector.finish(record);
  return record;
}
