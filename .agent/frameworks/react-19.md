---
id: react-19
title: "React 19 Best Practices"
description: "Guidelines for React 19 features including the Compiler, new hooks (use, useActionState), and Server vs Client components."
category: frameworks
tags: [react, hooks, server-components, compiler, optimization]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [nextjs-16, ../patterns/state-management]
---
## Table of Contents

- [React Compiler](#react-compiler)
- [New Hooks & APIs](#new-hooks-apis)
- [Server vs Client Components](#server-vs-client-components)
- [Rendering async data](#rendering-async-data)
- [See Also](#see-also)


# React 19 Best Practices

## React Compiler

- **Automatic Optimization**: No manual `useMemo`/`useCallback`.
- **Configuration**: Enabled via `babel-plugin-react-compiler`.

## New Hooks & APIs

- **`use()`**: Unwrap promises/context.
- **`useActionState()`**: Form submission state.
- **`useFormStatus()`**: Pending state.
- **`useOptimistic()`**: UI updates.

## Server vs Client Components

- **Default**: Server Components.
- **Client Components**: Interactivity only (`"use client"`).
- **Data Fetching**: Server-side only (via props).

## Rendering async data

- **Suspense Boundaries**: For async loading states.

## See Also

- [Next.js 16](nextjs-16.md)
- [State Management](../patterns/state-management.md)
- [Server Actions](../patterns/server-actions.md)
