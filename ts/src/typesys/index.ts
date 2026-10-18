/**
 * The type system's public surface, as Python's `yian.typesys`: typed values, the type
 * language and the variable pool.
 *
 * Python's `AnyValue`, `DateValue` and `DateTimeValue` carriers are all `TypedValue` here.
 * `StorageKind`, `Value`, `ValueRecord` and `PoolWire` name shapes Python leaves untyped;
 * `TypedValue` and `VariablePool` use them in their public signatures. `isRecord` is the
 * plain-object test `buildValueAs` uses, so callers read a record exactly as the type system
 * does (Python checks `isinstance(value, dict)` inline).
 */

export type { PoolWire } from "./pool.ts";
export { VariablePool } from "./pool.ts";
export type { RecordDef, TypeRegistry, VariantDef } from "./types.ts";
export { BUILTIN_TYPE_NAMES, parseType, resolveType, typeFor } from "./types.ts";
export type {
  FieldAbsent,
  FieldPartial,
  FieldRead,
  StorageKind,
  Value,
  ValueRecord,
} from "./values.ts";
export {
  buildValue,
  buildValueAs,
  canonicalTypeLabel,
  canonicalUnionOrder,
  FileRef,
  isRecord,
  listKindForElement,
  normalizeFiles,
  restrictUnion,
  Type,
  TypeCheckError,
  TypedValue,
  typesOverlap,
  unionField,
  unionMemberForTag,
  unionTagFields,
  ValueKind,
} from "./values.ts";
