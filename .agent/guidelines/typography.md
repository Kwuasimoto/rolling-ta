---
id: typography
title: "Typography System"
description: "Semantic CSS utility classes for consistent typography throughout the application."
category: guidelines
tags: [typography, css, tailwind, design-system]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [coding-style, ../architecture/project-structure]
---
## Table of Contents

- [Available Typography Classes](#available-typography-classes)
  - [Headings (font-geo)](#headings-font-geo)
  - [Body Text (Geist Sans)](#body-text-geist-sans)
  - [Secondary Information](#secondary-information)
  - [Semantic Colors](#semantic-colors)
- [Font Family Rules](#font-family-rules)
- [Why CSS Classes?](#why-css-classes)
- [See Also](#see-also)


# Typography System

**CRITICAL: ALL typography MUST use CSS utility classes from `src/app/globals.css`.**

## Available Typography Classes

### Headings (font-geo)
- `.typography-h1`: Page titles (4xl).
- `.typography-h2`: Section headings (3xl).
- `.typography-h3`: Subsection types (2xl).
- `.typography-h4`: Panel headings (xl).

### Body Text (Geist Sans)
- `.typography-body`: Standard.
- `.typography-p`: With margins.
- `.typography-small`: Compact labels.
- `.typography-large`: Emphasized.

### Secondary Information
- `.typography-muted`: Secondary text.
- `.typography-help`: Helper text.
- `.typography-label`: Form labels.
- `.typography-caption`: Captions.
- `.typography-code`: Code snippets.

### Semantic Colors
- `.typography-error`
- `.typography-success`

## Font Family Rules
- **font-geo**: Headings only.
- **font-wd40**: Buttons (auto).
- **Default**: Everything else.

## Why CSS Classes?
Simple, composable, single source of truth.

## See Also
- [Coding Style](coding-style.md)
- [Project Structure](../architecture/project-structure.md)
