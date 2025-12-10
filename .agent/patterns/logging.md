---
id: logging
title: "Logging Pattern"
description: "Architecture and best practices for the flexible logging system using the broadcast pattern."
category: patterns
tags: [logging, debugging, architecture, broadcast-pattern]
type: pattern
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../architecture/solid-principles, ../architecture/project-structure]
---
## Table of Contents

- [Logger Architecture](#logger-architecture)
- [Why This Pattern?](#why-this-pattern)
- [Logging Best Practices](#logging-best-practices)
- [Logging in Repositories](#logging-in-repositories)
- [Logging in Server Actions](#logging-in-server-actions)
- [Custom Logger Implementations](#custom-logger-implementations)
- [See Also](#see-also)


# Logging

## Logger Architecture

Uses a flexible broadcast pattern. **ALL logging MUST use `makeLog`**.

## Why This Pattern?

1. **Single Responsibility**: Separate implementation from usage.
2. **Open/Closed**: Easy to add new loggers.

## Logging Best Practices

1. **Use `makeLog`**.
2. **Include location context**.
3. **Use appropriate levels**: `info`, `debug`, `error`.
4. **Provide context objects**.

## Logging in Repositories

Log at key points in the 6-step data flow.

## Logging in Server Actions

Log initiation, success, and detailed errors (internally).

## Custom Logger Implementations

Implement `ForgedLogger` interface for new destinations.

## See Also

- [SOLID Principles](../architecture/solid-principles.md)
- [Repository Pattern](../architecture/repository-pattern.md)
- [Server Actions](server-actions.md)
