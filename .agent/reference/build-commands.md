---
id: build-commands
title: "Build & Development Commands"
description: "Reference for common pnpm commands used for development, building, testing, and maintenance."
category: reference
tags: [commands, pnpm, build, dev, test]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../guidelines/quick-start, ../guidelines/testing]
---
## Table of Contents

- [Package Manager](#package-manager)
- [Development](#development)
- [Build & Production](#build-production)
- [Code Quality](#code-quality)
- [UI Components](#ui-components)
- [Testing](#testing)
- [See Also](#see-also)


# Build, Test, and Development Commands

## Package Manager
**Use `pnpm`**.

## Development
- `pnpm dev`: Start Turbopack dev server.

## Build & Production
- `pnpm build`: Production build.
- `pnpm start`: Serve production build.

## Code Quality
- `pnpm lint`: Run ESLint.
- `pnpm format`: Run Prettier.
- `pnpm exec tsc --noEmit`: Type check.

## UI Components
- `pnpm ui:add [component]`: Add shadcn component.

## Testing
- `pnpm exec playwright test`: Run E2E tests.
- `pnpm test e2e/test.spec.ts`: Run specific test.

## See Also
- [Testing](../guidelines/testing.md)
- [Quick Start](../guidelines/quick-start.md)
