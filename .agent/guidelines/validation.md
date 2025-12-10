# Validation Guidelines

## Zod v4 Migration Rules

**CRITICAL**: This project uses **Zod v4**. You MUST strictly adhere to the following rules. Do NOT use deprecated Zod 3 patterns.

### 1. Top-Level Validations
Zod v4 moves many string validations to the top-level namespace. The old method chaining is **DEPRECATED**.

| Deprecated (Zod 3) | Required (Zod v4) |
| :--- | :--- |
| `z.string().email()` | `z.email()` |
| `z.string().url()` | `z.url()` |
| `z.string().uuid()` | `z.uuid()` |
| `z.string().emoji()` | `z.emoji()` |
| `z.string().cuid()` | `z.cuid()` |
| `z.string().ulid()` | `z.ulid()` |
| `z.string().ip()` | `z.ipv4()` or `z.ipv6()` |

**Exception**: `z.string().min()`, `.max()`, `.length()` are still chainable methods on `z.string()`.

### 2. Error Customization
The `invalid_type_error`, `required_error`, and `errorMap` parameters have been **DROPPED** or deprecated. You must use the unified `error` parameter.

**❌ Bad (Zod 3)**
```typescript
z.string({ 
  required_error: "Required", 
  invalid_type_error: "Must be string" 
});

z.string().min(5, { message: "Too short" }); // `message` key is deprecated
```

**✅ Good (Zod v4)**
```typescript
z.string({ 
  error: "Must be a string" // Simple message
});

z.string({
  error: (issue) => issue.code === "invalid_type" ? "Not a string" : "Required"
});

z.string().min(5, { error: "Too short" });
```

### 3. Object Schemas
Modifier methods have been replaced by top-level constructors.

| Deprecated (Zod 3) | Required (Zod v4) |
| :--- | :--- |
| `z.object({...}).strict()` | `z.strictObject({...})` |
| `z.object({...}).passthrough()` | `z.looseObject({...})` |
| `z.object({...}).strip()` | `z.object({...})` (Default behavior) |
| `z.object({...}).nonstrict()` | **REMOVED** |
| `z.object({...}).deepPartial()` | **REMOVED** (No direct alternative) |

**Note on Optionality**: `z.unknown()` and `z.any()` are NOT optional by default in inferred types anymore.
```typescript
z.object({ a: z.any() }) // Inferred as { a: any }, NOT { a?: any }
```

### 4. Custom Refinements
- **No `ctx.path`**: You cannot access `ctx.path` in `superRefine`.
- **No 2nd Argument**: `.refine(validator, messageMapper)` overload is removed.

**❌ Bad (Zod 3)**
```typescript
z.string().refine(val => val.length > 5, val => ({ message: `${val} is too short` }));
```

**✅ Good (Zod v4)**
```typescript
z.string().refine(val => val.length > 5, { 
  error: (val) => `${val} is too short` // or just a string message
});
```
