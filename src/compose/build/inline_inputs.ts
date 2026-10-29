/**
 * Strict type inference for generated inline-composite input declarations.
 *
 * Inline composite bodies become private child flows, so every outer binding must produce a
 * concrete child-parameter type. This module infers only types that the Compose source proves
 * structurally; it never falls back to the runtime value builder's lenient `Any` behavior.
 * Top-level strings and the leaf strings of a value template are binding templates; a span-free
 * literal container's strings are ordinary strings.
 *
 * Knows about: `expr` template syntax, node-owned semantic binding/output-contract carriers,
 * `typesys` structural types, and Compose source ranges/errors.
 * Never imports: the loader, `compile`, the runtime, or node kinds.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { TEMPLATE_FNS } from "../../expr/builtins.ts";
import { ExpressionError, ISINSTANCE, type RefStep, splitReference, stepsText } from "../../expr/expressions.ts";
import { type ExprNode, Token, Tree } from "../../expr/grammar.ts";
import { scanTemplate, Span } from "../../expr/template.ts";
import {
  Binding,
  bindingSource,
  fileFieldError,
  isExactly,
  type OutputContract,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { messagesKey, NEW_MESSAGES } from "../../typesys/pool.ts";
import { typeName } from "../../typesys/types.ts";
import {
  canonicalTypeLabel,
  fileFieldType,
  isListKind,
  isRecord,
  listKindForElement,
  Type,
  typeEquals,
  unionField,
  ValueKind,
  writeAsJson,
  writeAsText,
} from "../../typesys/values.ts";
import { LoadDiagnostics, type LoadError } from "../helpers/errors.ts";
import {
  directReferenceParts,
  hasOptionalStep,
  outputContractPathError,
  projectOutputContract,
  type TypeFn,
} from "../../compile/output_contracts.ts";
import { SYSTEM_AMBIENTS } from "../../compile/validation.ts";

/**
 * Return the advice for an inline binding whose type can't be proven.
 *
 * @param bindingLabel The binding's node/input path, such as `"each.input.text"`; its last
 *   segment names the slot.
 * @returns Advice to tag the value with its type or use a typed named body.
 */
export function explicitTypeHint(bindingLabel: string): string {
  const name = bindingLabel.split(".").at(-1)!;
  return `declare its type with a tag (\`${name}: !<type> ...\`) or use an explicitly typed named \`defs:\` body`;
}

/**
 * Resolves a producer key to its output type or canonical output contract: a node id for its
 * output, or `messagesKey(id)` for an AGENT's new messages. It receives the requesting binding
 * label and range so lazy-build failures keep their source site.
 */
export type ProducerType = (
  producerKey: string,
  bindingLabel: string,
  bindingRange: SourceRange | null,
) => Type | OutputContract;

/** Lazily supplies a MAP element type for a direct `${item...}` read. */
export type ItemType = (bindingLabel: string, bindingRange: SourceRange | null) => Type;

const SCALAR_TOKEN_KINDS: ReadonlyMap<string, ValueKind> = new Map([
  ["NULL", ValueKind.NONE],
  ["BOOL", ValueKind.BOOLEAN],
  ["STRING", ValueKind.STRING],
]);

/** The shared context of one binding's inference walk. */
interface Walk {
  readonly parentInputTypes: ReadonlyMap<string, Type>;
  readonly producerType: ProducerType;
  readonly itemType: ItemType | null;
  readonly typeFn: TypeFn | null;
  readonly bindingLabel: string;
  readonly bindingRange: SourceRange | null;
  readonly diagnostics: LoadDiagnostics;
}

/**
 * Infer one generated child parameter's strict structural type.
 *
 * `bindingLabel` names the outer binding for diagnostics, such as `"each.input.text"`. A string
 * passed directly as `value` and each leaf string of a `ValueTemplate` are scanned as binding
 * templates; a typed binding nested in a template gives its declared type. A plain record/list
 * value is a literal, so its strings infer as `str` without scanning. A lone direct reference
 * obtains its root type from the parent signature or an injected producer/MAP-item provider, then
 * strictly walks record fields. A whole-value `T(...)` constructor has type `T` (`typeFn` resolves
 * it), with field reads on the result walked the same way.
 *
 * @example
 * `inferInlineBindingType(3n, { parentInputTypes: new Map(), producerType, bindingLabel:
 * "call.input.limit" })` is `Type.scalar(ValueKind.INTEGER)`.
 *
 * @param value The authored outer binding value.
 * @param options.parentInputTypes Parent flow input declarations available through
 *   `${input.<name>}`.
 * @param options.producerType Resolves a node id, or `messagesKey(id)` for an AGENT's new
 *   messages, to its output type or canonical output contract.
 * @param options.itemType Lazily supplies a MAP element type for a direct `${item...}` read;
 *   `null` (the default) rejects item references outside an outer MAP input.
 * @param options.typeFn Resolves a constructor name to its type; `null` (the default) leaves
 *   constructors unproven.
 * @param options.bindingLabel The author-facing node/input path included in inference errors.
 * @param options.bindingRange The binding value's source range, or `null` (the default) for
 *   spanless input.
 * @param options.diagnostics Factory for located load errors (default: source-free).
 * @returns The exact scalar or recursively structural type proven by `value`.
 * @throws LoadError when the value has no statically proven type under the inline-body rules.
 */
export function inferInlineBindingType(
  value: unknown,
  options: {
    parentInputTypes: ReadonlyMap<string, Type>;
    producerType: ProducerType;
    itemType?: ItemType | null;
    typeFn?: TypeFn | null;
    bindingLabel: string;
    bindingRange?: SourceRange | null;
    diagnostics?: LoadDiagnostics | null;
  },
): Type {
  const walk: Walk = {
    parentInputTypes: options.parentInputTypes,
    producerType: options.producerType,
    itemType: options.itemType ?? null,
    typeFn: options.typeFn ?? null,
    bindingLabel: options.bindingLabel,
    bindingRange: options.bindingRange ?? null,
    diagnostics: options.diagnostics ?? new LoadDiagnostics(),
  };
  const source = bindingSource(value);
  let inferred: Type;
  if (typeof source === "string") {
    inferred = inferBindingString(source, walk);
  } else if (source instanceof ValueTemplate) {
    inferred = inferTemplate(source.value, walk.bindingLabel, walk);
  } else {
    inferred = inferLiteral(source, walk.bindingLabel, new Set(), walk);
  }
  return requireProvenType(inferred, walk);
}

/**
 * Require every structural refinement in `typ` to be statically proven.
 *
 * A refined record is closed: each declared field must either be required or nullable, and every
 * child type is checked recursively. A union requires each member to be proven. An unrefined
 * object remains valid as a whole value. Every list storage kind, including `list(Optional(Any))`
 * with no element, requires a recursively proven structural element.
 *
 * @param typ The type to prove without copying or mutating it.
 * @param options.bindingLabel The author-facing node/input path included in inference errors.
 * @param options.bindingRange The binding value's source range, or `null` (the default) for
 *   spanless input.
 * @param options.diagnostics Factory for located load errors (default: source-free).
 * @returns The exact same object passed as `typ` after successful proof.
 * @throws LoadError when a record or list refinement is incomplete or internally invalid.
 */
export function requireProvenType(
  typ: Type,
  options: { bindingLabel: string; bindingRange?: SourceRange | null; diagnostics?: LoadDiagnostics | null },
): Type {
  const bindingLabel = options.bindingLabel;
  const bindingRange = options.bindingRange ?? null;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const prove = (child: Type) => requireProvenType(child, { bindingLabel, bindingRange, diagnostics });
  const fail = (reason: string) => inferenceError(bindingLabel, bindingRange, reason, diagnostics);

  if (typ.fields !== undefined) {
    const fields = typ.fields;
    const required = typ.required ?? new Set<string>();
    const missing = [...required].filter((name) => !fields.has(name));
    if (missing.length > 0) {
      throw fail(`record requires unknown fields ${writeAsText(missing.sort(compareCodeUnits))}`);
    }
    for (const [name, fieldType] of fields) {
      if (!required.has(name) && !fieldType.nullable) {
        throw fail(`record field ${reprText(name)} is neither required nor nullable`);
      }
      prove(fieldType);
    }
  }

  for (const member of typ.variants ?? []) prove(member);

  if (isListKind(typ.kind)) {
    if (typ.element === undefined) {
      throw fail(`list type ${reprText(canonicalTypeLabel(typ))} has no proven element type`);
    }
    prove(typ.element);
  }

  return typ;
}

/**
 * Return a recursively copied, deterministically ordered inline type.
 *
 * Record fields use code-unit order. Their exact valid required membership is retained, including
 * required nullable fields; a refined record with no `required` normalizes to an empty set. Other
 * type metadata is copied without changing its structural meaning.
 *
 * @param typ The already-proven type to canonicalize.
 * @returns A deep canonical copy; `typ` is never mutated.
 */
export function canonicalizeInlineType(typ: Type): Type {
  let fields: Map<string, Type> | undefined;
  let required: Set<string> | undefined;
  if (typ.fields === undefined) {
    required = typ.required === undefined ? undefined : new Set([...typ.required].sort(compareCodeUnits));
  } else {
    const source = typ.fields;
    fields = new Map(
      [...source.keys()].sort(compareCodeUnits).map((name) => [name, canonicalizeInlineType(source.get(name)!)]),
    );
    const sourceRequired = typ.required ?? new Set<string>();
    required = new Set([...fields.keys()].filter((name) => sourceRequired.has(name)));
  }

  const tags = typ.tags === undefined ? undefined : new Set([...typ.tags].sort(compareCodeUnits));
  const element = typ.element === undefined ? undefined : canonicalizeInlineType(typ.element);
  // Union members are already in canonical order; only their contents are copied.
  const variants = typ.variants?.map(canonicalizeInlineType);
  return new Type({
    kind: typ.kind,
    fields,
    required,
    tags,
    element,
    nullable: typ.nullable,
    bounds: typ.bounds,
    variants,
  });
}

/** Infer a top-level string using binding-template semantics. */
function inferBindingString(source: string, walk: Walk): Type {
  let segments;
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw inferenceError(
      walk.bindingLabel,
      walk.bindingRange,
      `invalid binding template: ${error.message}`,
      walk.diagnostics,
      error,
    );
  }

  const only = segments[0];
  if (segments.length !== 1 || !(only instanceof Span)) return Type.scalar(ValueKind.STRING);

  const expression = only.tree;
  // A `?.` step reads null when its receiver is null or lacks the key, as `inferExprType` types it.
  const directParts = directReferenceParts(expression);
  if (directParts !== null) {
    const typ = inferDirectReference(directParts, walk);
    return hasOptionalStep(expression as Tree) ? typ.replace({ nullable: true }) : typ;
  }

  const constructed = constructorType(expression, walk.typeFn);
  if (constructed !== null) {
    const [typ, name, steps] = constructed;
    const field = walkRecordType(
      typ,
      steps.map((step) => step.name),
      `${name}(...)` + stepsText(steps),
      walk,
    );
    return steps.some((step) => step.optional) ? field.replace({ nullable: true }) : field;
  }

  if (expression instanceof Token) {
    if (expression.type === "NUMBER") {
      return Type.scalar(expression.value.includes(".") ? ValueKind.NUMBER : ValueKind.INTEGER);
    }
    const kind = SCALAR_TOKEN_KINDS.get(expression.type);
    if (kind !== undefined) return Type.scalar(kind);
  }

  throw inferenceError(
    walk.bindingLabel,
    walk.bindingRange,
    "a whole-value expression has no supported static scalar type",
    walk.diagnostics,
  );
}

/** Return `[T, name, steps]` for a whole-value `T(...)` or `T(...).f`, else `null`. */
function constructorType(expression: ExprNode, typeFn: TypeFn | null): [Type, string, RefStep[]] | null {
  if (typeFn === null || !(expression instanceof Tree) || expression.data !== "refcall") return null;
  const children = expression.children;
  if (!children.some((child) => child instanceof Tree && child.data === "call_suffix")) return null;
  const name = String(children[0]);
  if (name === ISINSTANCE || TEMPLATE_FNS.has(name)) return null;
  const typ = typeFn(name);
  if (typ === null) return null;
  return [typ, name, splitReference(expression).steps];
}

/** Resolve one direct reference root, then apply the shared strict field walk. */
function inferDirectReference(parts: readonly string[], walk: Walk): Type {
  const path = parts.join(".");
  const head = parts[0]!;
  const fail = (reason: string) => inferenceError(walk.bindingLabel, walk.bindingRange, reason, walk.diagnostics);
  let root: Type | OutputContract;
  let fields: readonly string[];

  if (head === "input") {
    if (parts.length < 2) throw fail(`reference \${${path}} must be input.<name>[.field]`);
    const inputName = parts[1]!;
    const declared = walk.parentInputTypes.get(inputName);
    if (declared === undefined) {
      throw fail(`reference \${${path}} points to unknown parent input ${reprText(inputName)}`);
    }
    root = declared;
    fields = parts.slice(2);
  } else if (head === "item") {
    if (walk.itemType === null) throw fail(`reference \${${path}} is only typed inside an outer MAP input`);
    root = walk.itemType(walk.bindingLabel, walk.bindingRange);
    fields = parts.slice(1);
  } else if (head === "inputs" || head === "outputs") {
    throw fail(`reference \${${path}} uses retired plural head ${reprText(head)}`);
  } else if (head === "system") {
    const kind = parts.length >= 2 ? SYSTEM_AMBIENTS.get(parts[1]!) : undefined;
    if (kind === undefined) throw fail(`reference \${${path}} has no statically proven system type`);
    root = Type.scalar(kind);
    fields = parts.slice(2);
  } else {
    if (parts.length < 2 || (parts[1] !== "output" && parts[1] !== NEW_MESSAGES)) {
      throw fail(`reference \${${path}} must select ${head}.output[.field] or an AGENT's ${head}.${NEW_MESSAGES}`);
    }
    const key = parts[1] === NEW_MESSAGES ? messagesKey(head) : head;
    root = walk.producerType(key, walk.bindingLabel, walk.bindingRange);
    fields = parts.slice(2);

    if (isExactly(root, ValueContract) || isExactly(root, PartialRecordContract)) {
      const pathError = outputContractPathError(root, fields, path);
      if (pathError !== null) throw fail(pathError);
      const projected = projectOutputContract(root, fields);
      if (!isExactly(projected, ValueContract)) {
        throw fail(`reference \${${path}} has no statically proven exact type`);
      }
      root = projected.type;
      fields = [];
    }
  }

  return walkRecordType(requireProvenType(root, walk), fields, path, walk);
}

/**
 * Walk proven record fields and return a fresh result with path nullability.
 *
 * A union field is read through the shared output-contract walk, so its rules and messages match
 * every other reference check. A file's `uri`, `mime` and `name` are typed by `fileFieldType`.
 */
function walkRecordType(root: Type, fields: readonly string[], path: string, walk: Walk): Type {
  const fail = (reason: string) => inferenceError(walk.bindingLabel, walk.bindingRange, reason, walk.diagnostics);
  let current = root;
  let pathNullable = root.nullable;
  for (const [index, field] of fields.entries()) {
    if (current.variants !== undefined) {
      const unionError = outputContractPathError(new ValueContract(current), fields.slice(index), path);
      if (unionError !== null) throw fail(unionError);
      const read = unionField(current, field);
      const readType = read.kind === "read" ? read.type : null;
      if (readType === null) throw fail(`reference \${${path}} has no statically proven exact type`);
      current = readType;
    } else if (current.fields === undefined && current.kind === ValueKind.FILE) {
      const readType = fileFieldType(field);
      if (readType === null) throw fail(fileFieldError(path, field));
      current = readType;
    } else if (current.fields === undefined) {
      throw fail(`reference \${${path}} traverses ${reprText(field)} without proven record fields`);
    } else {
      const fieldType = current.fields.get(field);
      if (fieldType === undefined) throw fail(`reference \${${path}} reads unknown record field ${reprText(field)}`);
      current = fieldType;
    }
    pathNullable = pathNullable || current.nullable;
  }

  return current.replace({ nullable: pathNullable });
}

/** Infer one node of a value template: strings are templates, typed bindings declared. */
function inferTemplate(node: unknown, path: string, walk: Walk): Type {
  if (node instanceof Binding) {
    if (node.contract !== null) return node.contract.type;
    node = node.source;
  }
  if (typeof node === "string") return inferBindingString(node, walk);

  if (isRecord(node)) {
    const fields = new Map(
      Object.keys(node)
        .sort(compareCodeUnits)
        .map((key) => [key, inferTemplate(node[key], `${path}[${writeAsJson(key)}]`, walk)]),
    );
    return new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
  }
  if (Array.isArray(node)) {
    if (node.length === 0) {
      throw inferenceError(
        walk.bindingLabel,
        walk.bindingRange,
        `empty literal list at ${path} has no provable element type`,
        walk.diagnostics,
      );
    }
    const element = commonElementType(
      node.map((item, index) => inferTemplate(item, `${path}[${index}]`, walk)),
      path,
      walk,
      "list",
    );
    return new Type({ kind: listKindForElement(element), element });
  }
  return inferLiteral(node, path, new Set(), walk);
}

/** Infer a recursive literal; nested strings never enter the template parser. */
function inferLiteral(value: unknown, path: string, activeContainers: Set<object>, walk: Walk): Type {
  if (value === null) return Type.scalar(ValueKind.NONE);
  if (typeof value === "boolean") return Type.scalar(ValueKind.BOOLEAN);
  if (typeof value === "bigint") return Type.scalar(ValueKind.INTEGER);
  if (typeof value === "number") return Type.scalar(ValueKind.NUMBER);
  if (typeof value === "string") return Type.scalar(ValueKind.STRING);

  if (isRecord(value)) return inferMapping(value, path, activeContainers, walk);
  if (Array.isArray(value)) return inferList(value, path, activeContainers, walk);

  throw inferenceError(
    walk.bindingLabel,
    walk.bindingRange,
    `literal at ${path} has unsupported type ${reprText(typeName(value))}`,
    walk.diagnostics,
  );
}

/**
 * Infer a sorted closed record from one literal mapping. A record key is always text, so no
 * key needs checking.
 */
function inferMapping(
  value: Readonly<Record<string, unknown>>,
  path: string,
  activeContainers: Set<object>,
  walk: Walk,
): Type {
  enterContainer(value, path, activeContainers, walk);
  try {
    const fields = new Map(
      Object.keys(value)
        .sort(compareCodeUnits)
        .map((key) => [key, inferLiteral(value[key], `${path}[${writeAsJson(key)}]`, activeContainers, walk)]),
    );
    return new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
  } finally {
    activeContainers.delete(value);
  }
}

/** Infer one homogeneous structural list, widening only int/float scalars. */
function inferList(value: readonly unknown[], path: string, activeContainers: Set<object>, walk: Walk): Type {
  enterContainer(value, path, activeContainers, walk);
  try {
    if (value.length === 0) {
      throw inferenceError(
        walk.bindingLabel,
        walk.bindingRange,
        `empty literal list at ${path} has no provable element type`,
        walk.diagnostics,
      );
    }
    const element = commonElementType(
      value.map((item, index) => inferLiteral(item, `${path}[${index}]`, activeContainers, walk)),
      path,
      walk,
    );
    return new Type({ kind: listKindForElement(element), element });
  } finally {
    activeContainers.delete(value);
  }
}

/** Return the common type up to `Optional`, or the scalar numeric widening, for a list. */
function commonElementType(elementTypes: readonly Type[], path: string, walk: Walk, noun = "literal list"): Type {
  // A generated parameter must have one concrete type, so unlike a template's output contract
  // (left unproven) a list with no common type fails the load.
  const bare = elementTypes.map((element) => element.replace({ nullable: false }));
  if (bare.every((element) => typeEquals(element, bare[0]!))) {
    return bare[0]!.replace({ nullable: elementTypes.some((element) => element.nullable) });
  }

  const integer = Type.scalar(ValueKind.INTEGER);
  const number = Type.scalar(ValueKind.NUMBER);
  if (elementTypes.every((element) => typeEquals(element, integer) || typeEquals(element, number))) return number;

  throw inferenceError(
    walk.bindingLabel,
    walk.bindingRange,
    `${noun} at ${path} has heterogeneous element types`,
    walk.diagnostics,
  );
}

/** Add one container to the active recursion path and reject true cycles. */
function enterContainer(value: object, path: string, activeContainers: Set<object>, walk: Walk): void {
  if (activeContainers.has(value)) {
    throw inferenceError(
      walk.bindingLabel,
      walk.bindingRange,
      `cyclic literal container encountered at ${path}`,
      walk.diagnostics,
    );
  }
  activeContainers.add(value);
}

/** Create the one located error shape used by strict inline inference. */
function inferenceError(
  bindingLabel: string,
  bindingRange: SourceRange | null,
  reason: string,
  diagnostics: LoadDiagnostics,
  cause?: unknown,
): LoadError {
  return diagnostics.error(
    `cannot infer type for inline binding ${reprText(bindingLabel)}: ${reason}; ${explicitTypeHint(bindingLabel)}`,
    { range: bindingRange, cause },
  );
}
