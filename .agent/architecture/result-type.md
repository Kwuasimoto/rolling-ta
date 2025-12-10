---
id: result-type
title: "Result Type Pattern"
description: "Specification of the Result<T> pattern for type-safe error handling and narrowing across the application."
category: architecture
tags: [error-handling, typescript, patterns, result-type]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [repository-pattern, ../patterns/forms-validation]
---
## Table of Contents

- [The Result Type](#the-result-type)
- [Why Result Type?](#why-result-type)
- [Usage in Components](#usage-in-components)
- [See Also](#see-also)


# Result Type Pattern

**ALL repository methods and Server Actions MUST return `Result<T>`:**

## The Result Type

```typescript
// lib/types/result.ts
export type Result<T> =
  | { data?: T; error: string }
  | { data: T; error?: string }
```

## Why Result Type?

This pattern enables **type narrowing** and eliminates verbose undefined checks:

```typescript
// ✅ With Result<T> - clean type narrowing
const userResult = await userRepository.readById(id)
if ("error" in userResult) {
  return <ErrorState message={userResult.error} />
}
// TypeScript knows userResult.data is User (not undefined)
return <UserProfile user={userResult.data} />
```

## Usage in Components

```typescript
export default async function ServerComponent() {
  const itemResult = await itemRepository.getByUserId(userId)

  // Pattern 1: Early return with "error" in check
  if ("error" in itemResult) {
    console.error(itemResult.error)
    return <ErrorState />
  }

  // itemResult.data is guaranteed to be T here
  return <SuccessComponent data={itemResult.data} />
}
```

## See Also

- [Repository Pattern](repository-pattern.md)
- [Server Actions](../patterns/server-actions.md)
