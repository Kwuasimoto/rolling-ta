---
id: repository-pattern
title: "Repository Pattern"
description: "Implementation details of the Repository Pattern for data access, including data flow, error handling, and Dev/Prod factory switching."
category: architecture
tags: [repository-pattern, data-access, architecture, solid]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [project-structure, solid-principles]
---
## Table of Contents

- [Data Flow (6-Step Process)](#data-flow-6-step-process)
- [Abstract Repository Base Class](#abstract-repository-base-class)
- [Concrete Repository Implementation Example](#concrete-repository-implementation-example)
- [Factory Pattern for Dev/Prod Environments](#factory-pattern-for-devprod-environments)
- [Repository Usage Rules](#repository-usage-rules)
- [See Also](#see-also)


# Repository Pattern Architecture

## Data Flow (6-Step Process)

Every repository read operation follows this exact sequence:

1. **Perform async read transaction** with Supabase client
2. **Check for read error** from Supabase transaction
3. **Convert snake_case → camelCase** using mapper function
4. **Validate camelCase data** with Zod schema
5. **Check for validation error**
6. **Return `Result<T>`** type

## Abstract Repository Base Class

```typescript
export default abstract class Repository<R> {
  // ... (Abstract methods: readAll, readById)
  // ... (Base read method with error handling and validation)
}
```

## Concrete Repository Implementation Example

```typescript
class UserRepository extends Repository<User> {
  // ... (Implementation of readAll and readById using Supabase)
  // ... (Mapping database types to application types)
}
export const userRepository = createUserRepository()
```

## Factory Pattern for Dev/Prod Environments

```typescript
export function createClient() {
  if (process.env.NODE_ENV === 'development') {
    return createMockClient(process.env.SUPABASE_LOCAL_URL) // Docker
  }
  return createSupabaseClient() // Actual Supabase
}
```

## Repository Usage Rules

1. **Create new instance for each component** (avoid passing as props)
2. **Only use for reads** - mutations go in Server Actions
3. **Always return `Result<T>`** for consistent error handling
4. **Validate with Zod** before returning data

## See Also

- [SOLID Principles](solid-principles.md)
- [Type System](type-system.md)
- [Result Type](result-type.md)
