---
id: nextjs-16
title: "Next.js 16 Best Practices"
description: "Best practices for Next.js 16 App Router, Server Components, Server Actions, and performance/caching strategies."
category: frameworks
tags: [nextjs, app-router, server-components, server-actions, performance]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [react-19, ../architecture/repository-pattern, ../patterns/server-actions]
---
## Table of Contents

- [Server Components & Actions](#server-components-actions)
- [Caching & Performance](#caching-performance)
- [File Organization](#file-organization)
- [Metadata & SEO](#metadata-seo)
- [See Also](#see-also)


# Next.js 16 App Router Best Practices

## Server Components & Actions

- **Server Components**: Fetch data directly via repositories.
- **Server Actions**: Mutations in `src/lib/actions/*.ts`.
- **Repository Pattern**: Use for all read operations.

## Caching & Performance

- **Opt-in Caching**: Dynamic by default.
- **Partial Pre-Rendering (PPR)**: Use for instant navigation.
- **Avoid Client Fetching**: Use Server Components.

## File Organization

- **Route Groups**: `(auth)`, `(dashboard)`.
- **Middleware**: For auth checks in root.
- **Components**: `layout.tsx`, `loading.tsx`, `error.tsx`.

## Metadata & SEO

- Export `metadata` or `generateMetadata()`.
- Use dynamic metadata for pages.

## See Also

- [React 19](react-19.md)
- [Repository Pattern](../architecture/repository-pattern.md)
- [Server Actions](../patterns/server-actions.md)
- [Project Structure](../architecture/project-structure.md)
