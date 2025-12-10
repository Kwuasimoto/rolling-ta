---
id: project-structure
title: "Project Structure"
description: "Detailed overview of the project's directory structure, module organization, and file purpose."
category: architecture
tags: [project-structure, directory-layout, organization, architecture]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [repository-pattern, ../rules/code-style]
---
## Table of Contents

- [Directory Layout](#directory-layout)
- [See Also](#see-also)


# Project Structure & Module Organization

## Directory Layout

```text
src/
├── app/                       # Next.js 16 App Router (routes, layouts, pages)
│   ├── (auth)/               # Route groups for auth pages
│   └── layout.tsx            # Root layout with providers
├── components/
│   ├── ui/                   # shadcn components (button, dialog, etc.)
│   ├── composite/            # Server Components (data fetching) + Client Components (interactivity)
│   └── providers/            # Context providers (theme, auth, etc.)
└── lib/
    ├── actions/              # Server Actions (POST, PUT, DELETE mutations ONLY)
    ├── fixtures/            # Page information constants and seed data
    ├── supabase/
    │   ├── client.ts        # Client Component Supabase client (browser)
    │   ├── server.ts        # Server Component Supabase client
    │   └── factory.ts       # Factory for dev/prod environment switching
    ├── repository/
    │   ├── base/            # Abstract Repository<T> base class
    │   ├── errors.ts        # ForgedRepositoryError types
    │   └── impl/            # Concrete repository implementations
    ├── validation/
    │   ├── schemas/         # Zod schemas (camelCase)
    │   └── types.ts         # Exported z.infer<typeof schema> types
    ├── mappers/             # snake_case → camelCase conversion utilities
    ├── loggers/             # Custom logger implementations
    ├── hooks/               # Custom React hooks
    ├── state/               # Zustand stores
    ├── types/
    │   ├── database.types.ts  # Supabase-generated types
    │   └── result.ts        # Result<T> type definition
    ├── styles/              # Global styles and Tailwind config
    ├── logging.ts           # Logging system
    └── utils.ts             # Shared utilities

public/                      # Static assets
e2e/                         # Playwright tests
middleware.ts                # Auth token refresh and validation
.env.local                   # Environment variables (gitignored)
```

**IMPORTANT**: `/api` directory should **NEVER** be used. Always fetch data via repositories in Server Components.

## See Also

- [Repository Pattern](repository-pattern.md)
- [Result Type](result-type.md)
- [Server Actions](../patterns/server-actions.md)
