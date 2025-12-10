---
id: quick-start
title: "Quick Start for New Projects"
description: "Step-by-step specific guide for setting up a Next.js 16 project with this architecture."
category: guidelines
tags: [setup, configuration, project-scaffolding, nextjs]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../architecture/project-structure, ../frameworks/supabase]
---
## Table of Contents

- [1. Clone/Scaffold](#1-clonescaffold)
- [2. Install Dependencies](#2-install-dependencies)
- [3. Install Dev Dependencies](#3-install-dev-dependencies)
- [4. Configure Prettier](#4-configure-prettier)
- [5. Configure ESLint](#5-configure-eslint)
- [6. Configure lint-staged](#6-configure-lint-staged)
- [7. Configure MCP Servers](#7-configure-mcp-servers)
- [8. Configure Next.js](#8-configure-nextjs)
- [9. Install shadcn](#9-install-shadcn)
- [10. Set Up Environment Variables](#10-set-up-environment-variables)
- [11. Generate Supabase Types](#11-generate-supabase-types)
- [12. Create Project Structure](#12-create-project-structure)
- [13. Configure Middleware](#13-configure-middleware)
- [14. Enable React Compiler](#14-enable-react-compiler)
- [15. Copy Documentation](#15-copy-documentation)
- [Version Information](#version-information)
- [See Also](#see-also)


# Quick Start for New Projects

## 1. Clone/Scaffold
`npx create-next-app@latest --typescript --tailwind --app`

## 2. Install Dependencies
`pnpm add @supabase/ssr @supabase/supabase-js zod react-hook-form @hookform/resolvers zustand`

## 3. Install Dev Dependencies
`pnpm add -D @playwright/test prettier prettier-plugin-tailwindcss eslint @eslint/eslintrc lint-staged`

## 4. Configure Prettier
Use `prettier.config.js` with tailwind plugin.

## 5. Configure ESLint
Use `eslint.config.mjs` extending next/core-web-vitals.

## 6. Configure lint-staged
Setup `.lintstagedrc.json` for pre-commit checks.

## 7. Configure MCP Servers
Setup `.mcp.json` for Supabase, Shadcn, etc.

## 8. Configure Next.js
Update `next.config.ts` for Supabase images and React Compiler.

## 9. Install shadcn
`pnpm dlx shadcn@latest init`
**Use `pnpm ui:add` for components.**

## 10. Set Up Environment Variables
Configure `.env.local`.

## 11. Generate Supabase Types
Run `supabase gen types typescript`.

## 12. Create Project Structure
Implement standard folders: `lib/{actions, repository, validation, ...}`.

## 13. Configure Middleware
Setup `middleware.ts` for auth.

## 14. Enable React Compiler
In `next.config.ts`.

## 15. Copy Documentation
Pull `.agent/` folder.

## Version Information
- Next.js 16+
- React 19.2+
- Supabase SSR

## See Also
- [Project Structure](../architecture/project-structure.md)
- [Supabase](../frameworks/supabase.md)
- [Server Actions](../patterns/server-actions.md)
