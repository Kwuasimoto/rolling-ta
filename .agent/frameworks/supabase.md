---
id: supabase
title: "Supabase Best Practices"
description: "Integration guide for Supabase Auth, Database, RLS, and Realtime with Next.js 16."
category: frameworks
tags: [supabase, auth, database, rls, realtime]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../architecture/repository-pattern, ../patterns/server-actions]
---
## Table of Contents

- [Authentication Setup (Asymmetric Tokens)](#authentication-setup-asymmetric-tokens)
  - [Environment Variables](#environment-variables)
  - [Client Utilities](#client-utilities)
- [Asymmetric Token Authentication](#asymmetric-token-authentication)
- [Cookie-Based Auth](#cookie-based-auth)
- [Database Access](#database-access)
- [Real-time & Storage](#real-time-storage)
  - [Real-time Subscriptions](#real-time-subscriptions)
  - [Storage](#storage)
- [See Also](#see-also)


# Supabase Integration Best Practices

## Authentication Setup (Asymmetric Tokens)

**Use asymmetric tokens for fast, efficient authentication.**

### Environment Variables
```env
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
SUPABASE_LOCAL_URL=...
```

### Client Utilities
- `lib/supabase/client.ts`: Browser client.
- `lib/supabase/server.ts`: Server client (cookies).
- `lib/supabase/factory.ts`: Dev/Prod switch.

## Asymmetric Token Authentication

**Middleware pattern for efficient validation**:
Use `getClaims()` for fast validation without DB round-trips.

## Cookie-Based Auth

- **HTTP-only cookies**: Protection against XSS.
- **Middleware**: Validates via `getClaims()`.

## Database Access

1. **Repositories**: Reads (Server Components).
2. **Server Actions**: Mutations.
3. **Types**: Generated locally.
4. **RLS**: Mandatory for all tables.

## Real-time & Storage

### Real-time Subscriptions
Client Components only (useEffect).

### Storage
Server-side upload and signed URLs.

## See Also

- [Repository Pattern](../architecture/repository-pattern.md)
- [Server Actions](../patterns/server-actions.md)
- [Type System](../architecture/type-system.md)
