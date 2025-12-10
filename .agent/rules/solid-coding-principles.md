---
trigger: always_on
---
## Table of Contents

- [1. Single Responsibility Principle (SRP)](#1-single-responsibility-principle-srp)
  - [Guidelines:](#guidelines)
  - [Checklist:](#checklist)
- [2. Open/Closed Principle (OCP)](#2-openclosed-principle-ocp)
  - [Guidelines:](#guidelines)
  - [Checklist:](#checklist)
- [3. Liskov Substitution Principle (LSP)](#3-liskov-substitution-principle-lsp)
  - [Guidelines:](#guidelines)
  - [Checklist:](#checklist)
- [4. Interface Segregation Principle (ISP)](#4-interface-segregation-principle-isp)
  - [Guidelines:](#guidelines)
  - [Checklist:](#checklist)
- [5. Dependency Inversion Principle (DIP)](#5-dependency-inversion-principle-dip)
  - [Guidelines:](#guidelines)
  - [Checklist:](#checklist)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [Application in this Project](#application-in-this-project)


# Workspace Rules: SOLID Coding Principles

This document outlines the SOLID principles that must be followed to ensure a robust, maintainable, and scalable code architecture. All code modifications and new feature implementations should adhere to these guidelines.

## 1. Single Responsibility Principle (SRP)
> "A class should have one, and only one, reason to change."

### Guidelines:
- **Separation of Concerns**: Ensure that UI components, business logic, and data access layers are distinct.
- **Small Functions/Classes**: Keep functions and classes focused on a single task. If a class handles validation *and* database saving, split it.
- **Refactoring Trigger**: If you find yourself scrolling too much to find a method, or if a class has extensive imports from different domains, it likely violates SRP.

### Checklist:
- [ ] Does this component/function do only one thing?
- [ ] Can I describe the responsibility of this class in one sentence without using "and"?

## 2. Open/Closed Principle (OCP)
> "Software entities should be open for extension, but closed for modification."

### Guidelines:
- **Use Abstractions**: Rely on interfaces or abstract classes to define behavior.
- **Strategy Pattern**: Encapsulate varying algorithms or behaviors in separate classes implementing a common interface.
- **Avoid Switch Statements**: Replace complex `switch` or `if-else` chains based on types with polymorphism where possible.

### Checklist:
- [ ] Can I add new functionality (e.g., a new payment method) without changing existing code?
- [ ] Am I using interfaces to define contracts between modules?

## 3. Liskov Substitution Principle (LSP)
> "Subtypes must be substitutable for their base types without altering the correctness of the program."

### Guidelines:
- **Behavior Consistency**: Derived classes must honor the contracts (preconditions and postconditions) of the base class.
- **No "Not Implemented"**: Avoid creating subclasses that throw "Not Implemented" exceptions for methods defined in the parent interface.
- **Type Safety**: Ensure that subclasses don't return stricter types or accept looser types than the parent, unless safe to do so.

### Checklist:
- [ ] Can I replace an instance of the parent class with this subclass without breaking the app?
- [ ] Does the subclass fully implement the interface?

## 4. Interface Segregation Principle (ISP)
> "Clients should not be forced to depend upon interfaces that they do not use."

### Guidelines:
- **Role Interfaces**: Create specific interfaces for specific client needs (e.g., `IReadable`, `IWritable`) rather than a massive `IDatabase` interface.
- **Avoid Fat Interfaces**: If an interface has methods that are not relevant to all implementers, split it.

### Checklist:
- [ ] Does the implementing class use all methods of the interface?
- [ ] Are there empty method implementations? (Sign of violation)

## 5. Dependency Inversion Principle (DIP)
> "High-level modules should not depend on low-level modules. Both should depend on abstractions."

### Guidelines:
- **Dependency Injection**: Pass dependencies into classes (via constructor or props) rather than instantiating them inside.
- **Depend on Interfaces**: Type hints and imports should refer to interfaces or abstract types, not concrete implementations (e.g., depend on `ILogger`, not `FileLogger`).
- **Inversion of Control**: Let the framework or a main entry point wire up dependencies.

### Checklist:
- [ ] Am I using `new` to create service instances inside a logic class? (Avoid this)
- [ ] Do high-level policies depend on low-level details? (They shouldn't)

---

## Anti-Patterns to Avoid
- **God Objects**: Massive classes that know too much or do too much.
- **Tight Coupling**: Classes that directly reference concrete classes of other modules.
- **Spaghetti Code**: Unstructured flow of control that is hard to follow.
- **Rigidity**: Code that is hard to change because every change affects too many other parts of the system.

## Application in this Project
- **React Components**: Separate presentation (JSX) from logic (Custom Hooks).
- **Services**: Define service interfaces (e.g., `IAuthService`) and implement them separately.
- **Utilities**: Keep utility functions pure and stateless.