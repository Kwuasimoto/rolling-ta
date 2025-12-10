---
trigger: always_on
---
## Table of Contents

- [1. Core TypeScript Principles](#1-core-typescript-principles)
  - [Configuration & Safety](#configuration-safety)
  - [Error Handling: The Result Pattern](#error-handling-the-result-pattern)
- [2. SOLID Architecture in React/TS](#2-solid-architecture-in-reactts)
- [3. Design Patterns Implementation](#3-design-patterns-implementation)
  - [Creational](#creational)
  - [Structural](#structural)
  - [Behavioral](#behavioral)
- [4. Refactoring & Anti-Patterns](#4-refactoring-anti-patterns)
- [5. React & UI Guidelines](#5-react-ui-guidelines)
- [6. Tauri & Rust Interop](#6-tauri-rust-interop)


# Workspace Rules: Code Style

This document defines the coding standards for the Keen project (Deno + React + Rust). It integrates our workspace rules for **SOLID principles**, **Design Patterns**, and **Code Smell Avoidance**.

## 1. Core TypeScript Principles

### Configuration & Safety
- **Strict Mode**: Enabled (`"strict": true`). No implicit `any`.
- **Explicit Returns**: Public methods/exported functions must have return types.
- **Type Definitions**: Use `interface` for object shapes (extensible), `type` for unions/primitives.

### Error Handling: The Result Pattern
We prefer `Result<T>` over throwing exceptions for predictable control flow.

```typescript
// ✅ Good: Type-safe failure handling
async function getUser(id: string): Promise<Result<User>> {
  try {
    const user = await db.findUser(id);
    return user ? { data: user } : { error: new BaseError("User not found") };
  } catch (err) {
    return { error: new BaseError("DB Error", { cause: err }) };
  }
}

// Usage
const { data, error } = await getUser(uid);
if (error) return handleError(error);
console.log(data.name); // Safe access
```

- **BaseError**: Internal errors with stack traces.
- **SafeError**: User-facing errors (sanitized).
- **Assertions**: Use `assert()` to check invariants; log violations, don't crash.

## 2. SOLID Architecture in React/TS

- **Single Responsibility (SRP)**:
  - *Rule*: One component = One reason to change.
  - *Apply*: Split "God Components" into `LogicHook` + `UIComponent`.
- **Open/Closed (OCP)**:
  - *Rule*: Extend via composition, don't modify internals.
  - *Apply*: Use `children` prop and slots for layout components instead of boolean flags.
- **Liskov Substitution (LSP)**:
  - *Rule*: Sub-components must honor parent contracts.
  - *Apply*: `Button` and `IconButton` should accept the same base `HTMLButtonProps`.
- **Interface Segregation (ISP)**:
  - *Rule*: No massive props interfaces.
  - *Apply*: Component should only ask for the data it needs, not the whole `User` object.
- **Dependency Inversion (DIP)**:
  - *Rule*: Depend on abstractions (Context/Hooks), not concretions.
  - *Apply*: Inject services via React Context or custom hooks (`useAuth()`) rather than importing `AuthService` directly.

## 3. Design Patterns Implementation

### Creational
- **Factory**: Use helper functions to create complex initial states or test data.
- **Builder**: Use for constructing complex configuration objects (e.g., `TauriWindowConfig`).
- **Singleton**: Use `Zustand` stores or `Context` providers for global state (Theme, Auth).

### Structural
- **Facade**: Create custom hooks (`useCamera()`) to hide complex API interactions (Tauri commands + local logic).
- **Adapter**: Transform API responses into UI-ready interfaces at the service boundary.
- **Composite**: Build UI layouts using recursive component patterns (e.g., File Tree).

### Behavioral
- **Observer**: `useEffect` listening to event emitters or store subscriptions.
- **Command**: Encapsulate user actions (Undo/Redo) as objects with `execute()` and `undo()` methods.
- **Strategy**: Pass behavior functions or components as props (`renderItem`, `onSort`).

## 4. Refactoring & Anti-Patterns

Refer to `gravity_workspace_rules.md` for full details.

| Code Smell | React/TS Solution |
| :--- | :--- |
| **Long Method** | Extract logic into a custom hook (`useFormLogic`). |
| **Large Class** | Split into smaller functional components or utility modules. |
| **Prop Drilling** | Use Composition (`children`) or Context API. |
| **Primitive Obsession** | Use Value Objects (e.g., `UserId` type alias) instead of raw strings. |
| **Effect Spaghetti** | multiple `useEffect`s with single concerns > one giant `useEffect`. |

## 5. React & UI Guidelines

- **Functional Components**: Always use functional components with hooks.
- **Naming**: PascalCase for components (`UserCard`), camelCase for hooks (`useUser`).
- **File Structure**: Co-locate styles, tests, and types with the component.
- **Performance**: Use `useMemo` for expensive calculations, `useCallback` for stable function references passed to children.

## 6. Tauri & Rust Interop

- **Type Safety**: Define shared types in a `types/` directory. Ensure Rust structs match TS interfaces.
- **Commands**: Wrap `invoke` calls in typed service functions.
  ```typescript
  // src/services/app.ts
  export const appService = {
    greet: (name: string) => invoke<string>('greet', { name })
  };
  ```
- **Events**: Use typed event listeners for Rust-to-Frontend communication.
