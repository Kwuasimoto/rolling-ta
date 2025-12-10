---
id: server-actions
title: "Server Actions Pattern"
description: "Guidelines for implementing Server Actions for mutations, validation, and error handling."
category: patterns
tags: [server-actions, mutations, validation, security, nextjs]
type: pattern
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [forms-validation, ../architecture/repository-pattern, ../architecture/result-type]
---
## Table of Contents

- [What are Server Actions?](#what-are-server-actions)
- [Key Principles](#key-principles)
- [Server Action Pattern](#server-action-pattern)
- [Validation Rules](#validation-rules)
  - [Client-Side Validation](#client-side-validation)
  - [Server-Side Validation](#server-side-validation)
- [Security Best Practices](#security-best-practices)
- [usage in Components](#usage-in-components)
- [See Also](#see-also)


# Server Actions

## What are Server Actions?
Async functions for **mutations only** (POST, PUT, DELETE). Replace `/api` routes.

## Key Principles
1. **Mutations ONLY**.
2. **Reads use Repositories**.
3. **Always return `Result<T>`**.
4. **Validate with Zod**.
5. **Use `"use server"`**.

## Server Action Pattern

```typescript
"use server"
// ... imports
export async function createPostAction(input: CreatePostInput): Promise<Result<Post>> {
  // 1. Validate input
  // 2. Perform mutation
  // 3. Map and return
}
```

## Validation Rules

### Client-Side Validation
Validate forms before calling action.

### Server-Side Validation
**Always validate again server-side**.

## Security Best Practices
1. **Never expose detailed errors**.
2. **Log detailed errors internally**.
3. **Validate inputs**.
4. **Use RLS policies**.

## usage in Components
- **Server Components**: Read via Repository.
- **Client Components**: Mutate via Server Action.

## See Also
- [Repository Pattern](../architecture/repository-pattern.md)
- [Result Type](../architecture/result-type.md)
- [Forms & Validation](forms-validation.md)
