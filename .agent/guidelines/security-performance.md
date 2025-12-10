---
id: security-performance
title: "Security & Performance Checklist"
description: "Checklists for ensuring application security (validation, auth, protection) and performance (images, caching, optimization)."
category: guidelines
tags: [security, performance, checklist, validation, best-practices]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../frameworks/supabase, ../patterns/input-validation, ../frameworks/nextjs-16]
---
## Table of Contents

- [Security](#security)
  - [Input Validation](#input-validation)
  - [Data Protection](#data-protection)
  - [Authentication & Authorization](#authentication-authorization)
- [Performance](#performance)
  - [Images](#images)
  - [React Optimization](#react-optimization)
  - [Data Handling](#data-handling)
  - [Caching](#caching)
- [Validation Examples](#validation-examples)
- [See Also](#see-also)


# Security & Performance Checklist

## Security

### Input Validation
- [ ] **Server Actions**: Return `Result<T>`.
- [ ] **Repositories**: Validate with Zod.
- [ ] **Forms**: Validate client-side.

### Data Protection
- [ ] **Server Logic**: Keep sensitive logic on server.
- [ ] **Env Vars**: Use `NEXT_PUBLIC\_` sparingly.
- [ ] **Errors**: Never expose details.

### Authentication & Authorization
- [ ] **Supabase RLS**: Enforce at DB level.
- [ ] **Auth**: Use asymmetric tokens.
- [ ] **Cookies**: HTTP-only.

## Performance

### Images
- [ ] **Format**: AVIF/WebP.
- [ ] **Component**: Use `<Image>`.

### React Optimization
- [ ] **Hooks**: Avoid excessive useMemo/useCallback.
- [ ] **Architecture**: Prefer Server Components.

### Data Handling
- [ ] **Zustand**: Only for complex client state.
- [ ] **Repositories**: Separate reads/mutations.

### Caching
- [ ] **Next.js**: Use `unstable_cache`.
- [ ] **PPR**: Use Partial Pre-Rendering.

## Validation Examples
(See original file for code examples on Server Action Validation, Repo Validation, Env Vars, and RLS).

## See Also
- [Server Actions](../patterns/server-actions.md)
- [Repository Pattern](../architecture/repository-pattern.md)
- [Forms & Validation](../patterns/forms-validation.md)
