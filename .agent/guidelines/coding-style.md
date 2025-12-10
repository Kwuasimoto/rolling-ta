---
id: coding-style
title: "Coding Style & Naming Conventions"
description: "Standards for formatting, naming, component structure, and patterns."
category: guidelines
tags: [coding-style, naming-conventions, formatting, eslint, prettier]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [typography, testing, ../architecture/solid-principles, ../frameworks/typescript]
---
## Table of Contents

- [Formatting](#formatting)
- [Naming Conventions](#naming-conventions)
  - [Case Styles](#case-styles)
- [Icon Imports (lucide-react)](#icon-imports-lucide-react)
- [File Naming Conventions](#file-naming-conventions)
- [Component Structure](#component-structure)
- [Error Handling: The Result Pattern](#error-handling-the-result-pattern)
- [SOLID Architecture in React/TS](#solid-architecture-in-reactts)
- [Tauri & Rust Interop](#tauri-rust-interop)
- [See Also](#see-also)


# Coding Style & Naming Conventions

## Formatting
- **Prettier**: 2 spaces, semi: true, printWidth: 90.
- **ESLint**: `next/core-web-vitals`. Fix warnings.

## Naming Conventions

### Case Styles
- **PascalCase**: Components, Types (`UserProfile`, `User`).
- **camelCase**: Functions, Variables (`useAuth`, `isLoading`).
- **kebab-case**: Routes, Assets (`user-settings/`, `hero.png`).
- **UPPER_SNAKE_CASE**: Env Vars, Constants (`API_TIMEOUT`).

## Icon Imports (lucide-react)
**CRITICAL**: Postfix imports with "Icon" (e.g., `UserIcon`) to avoid conflicts.

## File Naming Conventions
- **Components**: `ComponentName.tsx`
- **Utilities**: `utils.ts`, `helpers.ts`
- **Server Actions**: `lib/actions/featureName.ts`
- **Repositories**: `lib/repository/impl/repoName.ts`

## Component Structure
- **Function Declarations**: Prefer `function MyComponent() {}`.
- **Server Components**: Async, default export.
- **Client Components**: `'use client'` directive.

## Error Handling: The Result Pattern
Always return `Result<T>` ({ data, error }) instead of throwing exceptions.

## SOLID Architecture in React/TS
- **SRP**: Split complex components.
- **OCP**: Extend via props/composition.
- **LSP**: Consistent prop interfaces.
- **ISP**: Small props interfaces.
- **DIP**: Custom hooks/Context over direct imports.

## Tauri & Rust Interop
- **Type Safety**: Shared types in `types/`.
- **Commands**: Typed wrappers around `invoke`.
- **Events**: Typed listeners.

## See Also
- [Typography](typography.md)
- [Testing](testing.md)
- [SOLID Principles](../architecture/solid-principles.md)
- [Result Type](../architecture/result-type.md)
