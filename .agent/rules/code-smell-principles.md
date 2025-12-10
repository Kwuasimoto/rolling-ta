---
trigger: always_on
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


# Workspace Rules: Code Smell Avoidance

This document outlines the implementation plan and rules for avoiding common code smells in the Gravity codebase. These rules are designed to guide LLM code generation and developer practices to ensure a maintainable, scalable, and robust architecture.

## 1. Bloaters
*Code, methods, and classes that have grown too large and hard to work with.*

### Long Method
- **Detection**: Methods exceeding 20-30 lines of code or handling multiple levels of abstraction.
- **Avoid**: Writing monolithic functions that perform parsing, validation, and business logic all in one.
- **Action**: Extract sub-tasks into private helper methods with descriptive names.
- **LLM Instruction**: "Refactor this long method by extracting the inner loop logic into a separate function named `processItem`."

### Large Class
- **Detection**: Classes with too many fields (>10) or methods (>20), or classes that violate the Single Responsibility Principle (SRP).
- **Avoid**: "God Objects" that manage UI state, API calls, and data parsing simultaneously.
- **Action**: Identify distinct responsibilities and extract them into separate classes or components.
- **LLM Instruction**: "Split `UserManager` into `UserRepository` (data access) and `UserAuthService` (authentication logic)."

### Long Parameter List
- **Detection**: Methods taking more than 3-4 arguments.
- **Avoid**: Passing numerous flags or configuration values individually.
- **Action**: Introduce a Parameter Object (interface/type) or a Configuration Object.
- **LLM Instruction**: "Refactor `createUser(name, email, age, address, phone)` to accept a `CreateUserDto` object."

### Data Clumps
- **Detection**: Identical groups of fields appearing in multiple classes or method signatures (e.g., `x, y, z` or `start, end`).
- **Avoid**: Passing these fields separately.
- **Action**: Extract them into a value object or class (e.g., `Point`, `DateRange`).

## 2. Object-Orientation Abusers
*Incorrect or incomplete application of object-oriented programming principles.*

### Switch Statements
- **Detection**: Complex `switch` or `if/else` chains that check for type codes or properties to determine behavior.
- **Avoid**: Hardcoding logic that changes based on object type.
- **Action**: Use Polymorphism. Create a common interface and move the logic into concrete classes.
- **LLM Instruction**: "Replace this switch statement on `shapeType` with a `calculateArea()` method in the `Shape` interface."

### Temporary Field
- **Detection**: Fields that are only set and used in certain circumstances or specific methods.
- **Avoid**: Cluttering the class state with temporary variables.
- **Action**: Pass these values as method arguments or extract the related functionality into a new class where these fields are always relevant.

## 3. Change Preventers
*Structures that make changes difficult or risky.*

### Divergent Change
- **Detection**: You find yourself changing the same class for many different reasons (e.g., adding a new database type AND changing the UI format).
- **Avoid**: Mixing different domains or layers in one class.
- **Action**: Split the class so each has one reason to change (SRP).

### Shotgun Surgery
- **Detection**: Making a single conceptual change requires small edits to many different classes.
- **Avoid**: Spreading related logic (e.g., logging, permissions) across the entire codebase.
- **Action**: Centralize the logic in a single service, utility, or aspect.

## 4. Dispensables
*Pointless and unneeded code that reduces clarity.*

### Comments
- **Detection**: Comments explaining *what* the code does (e.g., `// Increment i by 1`).
- **Avoid**: Using comments to excuse bad naming or complex logic.
- **Action**: Rename variables and functions to be self-explanatory. Use comments only to explain *why* a specific decision was made.

### Duplicate Code
- **Detection**: Identical or very similar code blocks in multiple places.
- **Avoid**: Copy-pasting logic.
- **Action**: Extract the common code into a utility function, base class, or shared component.

### Dead Code
- **Detection**: Unused variables, parameters, fields, methods, or commented-out blocks.
- **Avoid**: Keeping "zombie" code "just in case".
- **Action**: Delete it. Rely on version control to retrieve it if needed later.

## 5. Couplers
*Excessive coupling between classes.*

### Feature Envy
- **Detection**: A method accesses the data of another object more than its own data.
- **Avoid**: Placing logic in the wrong class.
- **Action**: Move the method (or the part of it that accesses the data) to the class that owns the data.

### Message Chains
- **Detection**: Long chains of method calls like `a.getB().getC().doSomething()`.
- **Avoid**: Tight coupling to the structure of related objects.
- **Action**: Hide the delegate. Create a method on `A` that delegates to `C`, or pass `C` directly.