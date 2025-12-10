---
id: commit-pr
title: "Commit & Pull Request Guidelines"
description: "Standards for git commit messages and pull request descriptions."
category: guidelines
tags: [git, commit-messages, pull-requests, workflow]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [coding-style, testing, security-performance]
---
## Table of Contents

- [Commit Messages](#commit-messages)
  - [Format](#format)
  - [Best Practices](#best-practices)
- [Pull Requests](#pull-requests)
  - [Summary](#summary)
  - [Screenshots](#screenshots)
  - [PR Checklist](#pr-checklist)
  - [Template](#template)
- [See Also](#see-also)


# Commit & Pull Request Guidelines

## Commit Messages

### Format
Use **present-tense, imperative mood**:
`Add user repository`

### Best Practices
- Under 72 chars.
- Reference issues (`Refs #123`).
- Squash WIP commits.

## Pull Requests

### Summary
Describe **What** changed and **Why**.

### Screenshots
Include Mobile, Tablet, and Desktop views for UI changes.

### PR Checklist
- [ ] `pnpm lint` passes
- [ ] `tsc --noEmit` passes
- [ ] Playwright tests pass
- [ ] Server Actions use `Result<T>`
- [ ] Zod validation used everywhere
- [ ] Interactive elements have `data-testid`

### Template
(Standard template provided in original file - kept concise here for reference).

## See Also
- [Coding Style](coding-style.md)
- [Testing](testing.md)
- [Security & Performance](security-performance.md)
