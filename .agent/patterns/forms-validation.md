---
id: forms-validation
title: "Forms & Validation"
description: "Patterns for client-side form handling with React Hook Form and Zod, and error handling strategies."
category: patterns
tags: [forms, validation, zod, react-hook-form, error-handling]
type: pattern
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [server-actions, ../architecture/type-system]
---
## Table of Contents

- [react-hook-form + Zod Resolver Pattern](#react-hook-form-zod-resolver-pattern)
- [Error Display Requirements](#error-display-requirements)
- [Zod Error Handling Best Practices](#zod-error-handling-best-practices)
  - [Error Formatting Methods](#error-formatting-methods)
  - [Server Action Error Handling Pattern](#server-action-error-handling-pattern)
  - [Repository Error Handling Pattern](#repository-error-handling-pattern)
  - [Client-Side Error Handling (Forms)](#client-side-error-handling-forms)
- [Security Rules](#security-rules)
- [See Also](#see-also)


# Form Handling & Client-Side Validation

## react-hook-form + Zod Resolver Pattern

**All forms MUST validate client-side before calling Server Actions.**

## Error Display Requirements

- **Show all validation errors clearly**.
- **Use shadcn Field/Form components**.
- **Validate before server action**.

## Zod Error Handling Best Practices

**CRITICAL: Never expose detailed validation errors to clients in Server Actions.**

### Error Formatting Methods
- `z.treeifyError()`: Nested.
- `z.prettifyError()`: Logging only.

### Server Action Error Handling Pattern
Log details internally, return generic messages.

### Repository Error Handling Pattern
Log details, throw generic `ForgedRepositoryError`.

### Client-Side Error Handling (Forms)
Can show detailed errors.

## Security Rules

1. **Server Actions**: Log debug, return generic.
2. **Repositories**: Log debug, throw generic.
3. **Never expose**: Internal logic/stack traces.

## See Also

- [Type System](../architecture/type-system.md)
- [Server Actions](server-actions.md)
- [Logging](logging.md)
