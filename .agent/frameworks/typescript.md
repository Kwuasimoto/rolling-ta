---
id: typescript
title: "TypeScript Best Practices"
description: "TypeScript guidelines for type safety, naming conventions, strict mode, and Tauri interoperability."
category: frameworks
tags: [typescript, type-safety, best-practices, tauri]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [rust, ../architecture/type-system]
---
## Table of Contents

- [1. Type Safety & Primitives](#1-type-safety-primitives)
  - [❌ Don't](#-dont)
  - [✅ Do](#-do)
- [2. Functions & Callbacks](#2-functions-callbacks)
- [3. Interfaces & Data Structures](#3-interfaces-data-structures)
- [4. Naming Conventions](#4-naming-conventions)
- [5. Advanced Types & Modifiers](#5-advanced-types-modifiers)
- [6. Strict Mode Configuration](#6-strict-mode-configuration)
- [7. Tauri-Specific Patterns](#7-tauri-specific-patterns)
- [See Also](#see-also)


# TypeScript Best Practices

## 1. Type Safety & Primitives

### ❌ Don't
- **Boxed Types**: `Number`, `String`.
- **`any`**: Use `unknown`.

### ✅ Do
- **Primitives**: `number`, `string`.
- **`unknown`**: Forces checks.
- **`readonly`**: Immutable props.

## 2. Functions & Callbacks

- **Void Callbacks**: Prevent accidental return usage.
- **Overloads**: Specific before general.

## 3. Interfaces & Data Structures

- **Interfaces**: Describe shapes.
- **Extension**: Reduce duplication.

## 4. Naming Conventions

- **camelCase**: Vars, funcs, files.
- **PascalCase**: Classes, Interfaces, components.
- **UPPER_CASE**: Constants.

## 5. Advanced Types & Modifiers

- **Access Modifiers**: `private`, `protected`.
- **Utility Types**: `Partial`, `Pick`, `Omit`.
- **Type Guards**: Runtime checks.

## 6. Strict Mode Configuration

Ensure `"strict": true` in `tsconfig.json`.

## 7. Tauri-Specific Patterns

- **Type-safe Commands**: Match Rust structs.
- **Event Types**: Strongly typed listeners.

## See Also

- [Rust](rust.md)
- [Result Type](../architecture/result-type.md)
- [Coding Style](../guidelines/coding-style.md)
