## Table of Contents

- [1. Type Safety & Primitives](#1-type-safety-primitives)
  - [❌ Don't](#-dont)
  - [✅ Do](#-do)
- [2. Functions & Callbacks](#2-functions-callbacks)
  - [Return Types](#return-types)
  - [Overloads](#overloads)
  - [Parameters](#parameters)
- [3. Interfaces & Data Structures](#3-interfaces-data-structures)
- [4. Naming Conventions](#4-naming-conventions)
- [5. Advanced Types & Modifiers](#5-advanced-types-modifiers)


# Workspace Rules: TypeScript Best Practices

This document aggregates critical TypeScript best practices for LLM code generation, optimized for correctness and maintainability.

## 1. Type Safety & Primitives

### ❌ Don't
- **Boxed Types**: Never use `Number`, `String`, `Boolean`, `Symbol`, or `Object`.
- **`any`**: Avoid `any`. It disables type checking.
- **`var`**: Never use `var`.
- **Empty Interfaces**: Do not define interfaces with no members.
- **Unused Generics**: Do not define generic type parameters that are unused.

### ✅ Do
- **Primitives**: Use `number`, `string`, `boolean`, `symbol`, `object`.
- **`unknown`**: Use `unknown` instead of `any` for values to be passed through without interaction.
- **`let` / `const`**: Use block-scoped variables.
- **Enums**: Use `enum` for named constants; export globally.
- **`readonly`**: Mark immutable properties as `readonly`.

```typescript
// Good
interface Position {
  readonly lat: number;
  readonly long: number;
}
```

## 2. Functions & Callbacks

### Return Types
- **Callbacks**: Use `void` return type for callbacks whose value is ignored.
  - *Why*: Prevents accidental usage of unchecked return values.
  ```typescript
  // Good
  function fn(cb: () => void) { cb(); }
  ```

### Overloads
- **Ordering**: Place **specific** overloads before **general** ones.
- **Optional Params**: Use optional parameters (`?`) instead of overloads for trailing arguments.
- **Union Types**: Use Union Types (`string | number`) instead of overloads for single argument differences.
- **Callback Arity**: Use a single overload with maximum arity; do not overload on callback argument count.

```typescript
// Good: Union over Overload
function setOffset(val: number | string): void;

// Good: Optional over Overload
function diff(a: string, b?: string): number;
```

### Parameters
- **Callback Params**: Do **not** make callback parameters optional unless the callback *must* be called with fewer arguments. Callers can always ignore arguments.
  ```typescript
  // Good
  interface Fetcher {
    getObject(done: (data: unknown, time: number) => void): void;
  }
  ```

## 3. Interfaces & Data Structures

- **Describe Data**: Use interfaces to describe object shapes.
- **Extension**: Extend interfaces (`interface B extends A`) to reduce duplication.
- **Destructuring**: Use ES6 destructuring for properties.
- **Factories**: Use Abstract Factory pattern for complex object creation.

## 4. Naming Conventions

| Entity | Convention | Example |
| :--- | :--- | :--- |
| Variable / Function | `camelCase` | `getUserData` |
| Global Constant | `UPPER_CASE` | `MAX_RETRIES` |
| Class / Interface | `PascalCase` | `UserProfile` |
| Type / Enum | `PascalCase` | `ResponseStatus` |
| File Name | `camelCase` | `userProfile.ts` |

## 5. Advanced Types & Modifiers

- **Access Modifiers**: Explicitly use `private`, `protected`, `public`.
- **Utility Types**: Leverage `Partial<T>`, `Required<T>`, `Pick<T, K>`, `Omit<T, K>`.

```typescript
interface Dog { name: string; age: number; }
// Make all properties optional
const partialDog: Partial<Dog> = {};
```
