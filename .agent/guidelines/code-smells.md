---
id: code-smells
title: "Code Smell Avoidance"
description: "Rules for avoiding common code smells like Bloaters, Object-Orientation Abusers, and Couplers."
category: guidelines
tags: [code-smells, refactoring, best-practices, clean-code]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [coding-style, ../architecture/solid-principles, ../architecture/design-patterns]
---
## Table of Contents

- [1. Bloaters](#1-bloaters)
  - [Long Method](#long-method)
  - [Large Class](#large-class)
  - [Long Parameter List](#long-parameter-list)
  - [Data Clumps](#data-clumps)
- [2. Object-Orientation Abusers](#2-object-orientation-abusers)
  - [Switch Statements](#switch-statements)
  - [Temporary Field](#temporary-field)
- [3. Change Preventers](#3-change-preventers)
  - [Divergent Change](#divergent-change)
  - [Shotgun Surgery](#shotgun-surgery)
- [4. Dispensables](#4-dispensables)
  - [Comments](#comments)
  - [Duplicate Code](#duplicate-code)
  - [Dead Code](#dead-code)
- [5. Couplers](#5-couplers)
  - [Feature Envy](#feature-envy)
  - [Message Chains](#message-chains)
- [Detection Checklist](#detection-checklist)
- [See Also](#see-also)


# Code Smell Avoidance

## 1. Bloaters
_Code that has grown too large._

### Long Method
- **Detection**: > 20-30 lines.
- **Action**: Extract validation/logic helpers.

### Large Class
- **Detection**: > 10 fields / 20 methods.
- **Action**: Split into components/modules.

### Long Parameter List
- **Detection**: > 3-4 input arguments.
- **Action**: Use Parameter Objects (interfaces/structs).

### Data Clumps
- **Detection**: Fields always appearing together (e.g., `x, y`).
- **Action**: Extract into Value Objects.

## 2. Object-Orientation Abusers
_Incorrect OO application._

### Switch Statements
- **Avoid**: Logic switching on types.
- **Action**: Use Polymorphism/Traits.

### Temporary Field
- **Avoid**: Fields used only in one method.
- **Action**: Pass as arguments.

## 3. Change Preventers
_Structures making change risky._

### Divergent Change
- **Avoid**: One class, many reasons to change.
- **Action**: Split by responsibility (SRP).

### Shotgun Surgery
- **Avoid**: A single feature needing edits in many files.
- **Action**: Centralize logic.

## 4. Dispensables
_Pointless code._

### Comments
- **Avoid**: Explaining *what* code does.
- **Action**: Explain *why* instead. Use self-documenting names.

### Duplicate Code
- **Avoid**: Copy-paste.
- **Action**: Extract shared logic (hooks/utils).

### Dead Code
- **Action**: Delete it.

## 5. Couplers
_Excessive coupling._

### Feature Envy
- **Avoid**: Method using another object's data excessively.
- **Action**: Move logic to the data owner.

### Message Chains
- **Avoid**: `a.getB().getC().do()`.
- **Action**: Flatten or use Facades.

## Detection Checklist
- [ ] Methods > 30 lines?
- [ ] Classes > 20 methods?
- [ ] Duplicate code?
- [ ] Unused code?

## See Also
- [SOLID Principles](../architecture/solid-principles.md)
- [Design Patterns](../architecture/design-patterns.md)
- [Coding Style](coding-style.md)
