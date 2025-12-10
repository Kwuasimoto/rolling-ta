---
id: type-system
title: "Type System Strategy"
description: "Architecture for strict type safety, separating server-side (snake_case) and client-side (camelCase) types."
category: architecture
tags: [typescript, type-safety, zod, supabase]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [repository-pattern, project-structure]
---
## Table of Contents

- [Two-Tier Type System](#two-tier-type-system)
  - [1. Server-Side Types (snake_case)](#1-server-side-types-snake_case)
  - [2. Client-Side Types (camelCase)](#2-client-side-types-camelcase)
  - [Why This Separation?](#why-this-separation)
- [Case Conversion Strategy](#case-conversion-strategy)
  - [Option 1: Manual Mapping (Recommended for Performance)](#option-1-manual-mapping-recommended-for-performance)
  - [Option 2: ts-case-convert (Recommended for Convenience)](#option-2-ts-case-convert-recommended-for-convenience)
- [See Also](#see-also)


# Type System Strategy

## Two-Tier Type System

This architecture uses **two distinct type systems** to optimize for both TypeScript performance and developer experience.

### 1. Server-Side Types (snake_case)

**Source**: Generated from Supabase schema via CLI
**Usage**: Repositories, database queries, Server Components

```typescript
// types/database.types.ts (auto-generated)
export type User = {
  user_id: string
  full_name: string
  created_at: string
}
```

### 2. Client-Side Types (camelCase)

**Source**: Zod schemas in `lib/validation/schemas/*.ts`
**Usage**: React components, forms, client-side logic

```typescript
// lib/validation/types.ts
export type User = z.infer<typeof userSchema>
```

### Why This Separation?
- Supabase types respect database conventions (snake_case)
- React/TypeScript ecosystem expects camelCase
- Zod schemas become single source of truth for client-side validation

## Case Conversion Strategy

### Option 1: Manual Mapping (Recommended for Performance)

```typescript
export function mapDbToApp(dbUser: Database.User): User {
  const { user_id, full_name, ...rest } = dbUser
  return {
    userId: user_id,
    fullName: full_name,
    ...rest,
  }
}
```

### Option 2: ts-case-convert (Recommended for Convenience)

```typescript
import { snakeToCamel } from 'ts-case-convert'
const camelUser = snakeToCamel(dbUser)
```

## See Also

- [Repository Pattern](repository-pattern.md)
- [Project Structure](project-structure.md)
