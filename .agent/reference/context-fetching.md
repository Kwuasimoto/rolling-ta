---
id: context-fetching
title: "Context Fetching Order"
description: "PROTOCOL: The mandatory order of operations for finding context, patterns, and documentation."
category: reference
tags: [context, documentation, protocol, mcp]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [tech-stack, ../guidelines/quick-start]
---
## Table of Contents

- [1. MCP Servers (Preferred)](#1-mcp-servers-preferred)
- [2. Official Documentation via WebFetch](#2-official-documentation-via-webfetch)
- [3. Repository Documentation & Code](#3-repository-documentation-code)
- [Golden Rule](#golden-rule)
- [See Also](#see-also)


# Context Fetching Order

## 1. MCP Servers (Preferred)
**First source for patterns.**
- `playwright`, `next-devtools`, `supabase`, `shadcn`.

## 2. Official Documentation via WebFetch
If MCP fails, fetch official docs.

## 3. Repository Documentation & Code
Last resort.
- `.agent/` documentation.
- Existing code.

## Golden Rule
**Never make up patterns.**

## See Also
- [Tech Stack](tech-stack.md)
- [Quick Start](../guidelines/quick-start.md)
