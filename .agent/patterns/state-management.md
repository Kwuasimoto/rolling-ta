---
id: state-management
title: "State Management (Zustand)"
description: "Guidelines on when and how to use Zustand for client-side state management."
category: patterns
tags: [state-management, zustand, client-state, react]
type: pattern
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../frameworks/react-19, ../architecture/repository-pattern]
---
## Table of Contents

- [When to Use Zustand](#when-to-use-zustand)
- [Zustand Store Pattern](#zustand-store-pattern)
- [Populating Zustand from Server Components](#populating-zustand-from-server-components)
- [Zustand Store Usage Rules](#zustand-store-usage-rules)
- [See Also](#see-also)


# Client-Side State Management with Zustand

## When to Use Zustand
**ONLY for complex client-side state shared across components.**
Do NOT use for server data, local state, or forms.

## Zustand Store Pattern
Only use the `create` function.

## Populating Zustand from Server Components
**Pattern**: Fetch in Server Component -> Pass to Client Component -> useEffect populates store.

## Zustand Store Usage Rules
1. **Stores in `lib/state/`**.
2. **Only use `create`**.
3. **Initialize from props**.
4. **Use selectors**.
5. **Keep stores flat**.

## See Also
- [Repository Pattern](../architecture/repository-pattern.md)
- [React 19](../frameworks/react-19.md)
