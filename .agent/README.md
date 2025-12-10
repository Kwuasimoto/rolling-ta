---
id: agent-documentation-index
title: "Agent Documentation Index"
description: "Central index for the project's agentic documentation, patterns, and guidelines."
category: index
tags: [documentation, architecture, agents, patterns]
type: reference
version: "2025-Q4"
created: 2025-12-06
updated: 2025-12-06
related: [brain]
---
## Table of Contents

- [Agent Documentation Index](#agent-documentation-index)
  - [Overview](#overview)
  - [Structure](#structure)
  - [Documentation Index](#documentation-index)
    - [Architecture (Foundational Patterns)](#architecture-foundational-patterns)
    - [Patterns (Implementation Guides)](#patterns-implementation-guides)
    - [Frameworks](#frameworks)
    - [Guidelines](#guidelines)
    - [Reference](#reference)
    - [Agents](#agents)
    - [Commands](#commands)
  - [Related topics](#related-topics)


# Agent Documentation Index

This directory contains modular, focused documentation for project architecture, patterns, and best practices.
It serves as the knowledge base for the agentic workflow.

## Overview

The `.agent/` directory is structured to provide specific context to AI agents and developers. It is organized by concern—architecture, patterns, frameworks, guidelines—to ensure that context retrieval is precise and relevant.

**Key Goals:**
- **Modularity**: Small, focused files instead of monolithic docs.
- **Discoverability**: Clear hierarchy for easy navigation.
- **Standardization**: Consistent patterns for code and architecture.

## Structure

```
.agent/
├── agents/           # Agent configuration files
├── architecture/     # Foundational design patterns
├── patterns/         # Implementation guides
├── frameworks/       # Framework-specific best practices
├── guidelines/       # Coding standards and workflows
├── reference/        # Quick references and lookups
├── rules/            # Core coding principles (SOLID, patterns, smells)
└── scaffold/         # Project scaffolding templates
```

## Documentation Index

### Architecture (Foundational Patterns)

- **[SOLID Principles](architecture/solid-principles.md)** - SRP, OCP, LSP, ISP, DIP
- **[Design Patterns](architecture/design-patterns.md)** - Creational, Structural, Behavioral
- **[Type System](architecture/type-system.md)** - Two-tier types (snake_case ↔ camelCase)
- **[Result Type](architecture/result-type.md)** - Type-safe error handling
- **[Repository Pattern](architecture/repository-pattern.md)** - Data flow abstraction
- **[Project Structure](architecture/project-structure.md)** - Directory organization

### Patterns (Implementation Guides)

- **[Forms & Validation](patterns/forms-validation.md)** - react-hook-form + Zod
- **[State Management](patterns/state-management.md)** - Zustand patterns
- **[Logging](patterns/logging.md)** - makeLog() broadcast pattern
- **[Server Actions](patterns/server-actions.md)** - Mutations, validation, security

### Frameworks

- **[Rust](frameworks/rust.md)** - Safety, concurrency, backend patterns
- **[TypeScript](frameworks/typescript.md)** - Type safety, conventions
- **[React 19](frameworks/react-19.md)** - Compiler, new hooks, Server Components
- **[Next.js 16](frameworks/nextjs-16.md)** - Server Components, App Router, API Routes

### Guidelines

- **[Coding Style](guidelines/coding-style.md)** - Formatting, naming, Result pattern
- **[Code Smells](guidelines/code-smells.md)** - Detection and refactoring
- **[Typography](guidelines/typography.md)** - CSS text utilities
- **[Testing](guidelines/testing.md)** - Playwright E2E, test organization
- **[Commit & PR](guidelines/commit-pr.md)** - Messages, PR format
- **[Security & Performance](guidelines/security-performance.md)** - Validation, RLS
- **[Quick Start](guidelines/quick-start.md)** - New project setup

### Reference

- **[Build Commands](reference/build-commands.md)** - pnpm scripts
- **[Tech Stack](reference/tech-stack.md)** - Versions, dependencies
- **[Context Fetching](reference/context-fetching.md)** - MCP → Docs → Code priority

### Agents

- **[design-review](agents/design-review-agent.md)** - UI/UX review
- **[ui-implementation](agents/ui-implementation-agent.md)** - Next.js 16 UI engineer
- **[task-planning](agents/task-planning-agent.md)** - High-level planner
- **[supabase-specialist](agents/supabase-specialist.md)** - RLS, auth, migrations, edge functions
- **[form-wizard](agents/form-wizard.md)** - Forms, validation, server vs client patterns
- **[sanity-content](agents/sanity-content.md)** - GROQ, schemas, Adapter pattern for CMS
- **[api-security](agents/api-security.md)** - Server action hardening, CSRF, rate limiting
- **[performance-optimizer](agents/performance-optimizer.md)** - Caching, PPR, Core Web Vitals, images
- **[playwright-test-planner](agents/playwright-test-planner.md)** - E2E planning
- **[playwright-test-generator](agents/playwright-test-generator.md)** - Test generation
- **[playwright-test-healer](agents/playwright-test-healer.md)** - Test debugging

### Commands

- **/design-review** - Comprehensive UI design review

## Related topics

- [BRAIN.md](../BRAIN.md) - The orchestration brain
- [Markdown Overhaul](../markdown-overhaul.md) - Documentation standards

