---
id: testing
title: "Testing Guidelines"
description: "Best practices for E2E testing with Playwright, including DOM selection, organization, and fixtures."
category: guidelines
tags: [testing, playwright, e2e, best-practices]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../frameworks/supabase, ../architecture/repository-pattern, quick-start]
---
## Table of Contents

- [Playwright (Primary E2E Framework)](#playwright-primary-e2e-framework)
- [DOM Element Selection Priority](#dom-element-selection-priority)
  - [Component Test ID Pattern](#component-test-id-pattern)
- [Test Organization & DRY Principles](#test-organization-dry-principles)
- [Test Fixtures & Constants](#test-fixtures-constants)
- [Database State Management](#database-state-management)
- [Testing Supabase](#testing-supabase)
- [See Also](#see-also)


# Testing Guidelines

## Playwright (Primary E2E Framework)

**Prioritize Playwright for all E2E testing.**

## DOM Element Selection Priority
**CRITICAL**: Use `.getByTestId()` as the primary selector.

1. **`getByTestId()`** (Primary)
2. **`getByRole()`** (Secondary)
3. **`getByLabel()`** (Tertiary)

### Component Test ID Pattern
Add `data-testid` attributes to all interactive elements.

## Test Organization & DRY Principles
- Create helpers in `e2e/helpers/`.
- Use `@e2e/*` aliases.

## Test Fixtures & Constants
Centralize data in `e2e/fixtures.ts`.

## Database State Management
Reset DB between tests using seed scripts.

## Testing Supabase
- Use factory pattern.
- Mock repositories for component tests.
- Test RLS policies.

## See Also
- [Supabase](../frameworks/supabase.md)
- [Repository Pattern](../architecture/repository-pattern.md)
- [Coding Style](coding-style.md)
