---
trigger: always_on
---
## Table of Contents

- [1. Creational Patterns](#1-creational-patterns)
  - [Factory Method](#factory-method)
  - [Builder](#builder)
  - [Singleton](#singleton)
- [2. Structural Patterns](#2-structural-patterns)
  - [Adapter](#adapter)
  - [Composite](#composite)
  - [Facade](#facade)
- [3. Behavioral Patterns](#3-behavioral-patterns)
  - [Observer](#observer)
  - [Strategy](#strategy)
  - [Command](#command)
  - [State](#state)


# Workspace Rules: Design Patterns

This document outlines the implementation plan and rules for applying common design patterns in the Gravity codebase. These patterns are selected to solve specific recurring design problems and ensure a flexible, reusable, and maintainable architecture.

## 1. Creational Patterns
*Mechanisms for object creation that increase flexibility and reuse.*

### Factory Method
- **Intent**: Define an interface for creating an object, but let subclasses decide which class to instantiate.
- **Use Case**: When a class can't anticipate the class of objects it must create, or wants to delegate creation to subclasses.
- **LLM Instruction**: "Implement a `Dialog` class with a `createButton()` factory method. Subclasses `WindowsDialog` and `WebDialog` should return `WindowsButton` and `WebButton` respectively."

### Builder
- **Intent**: Construct a complex object step by step. Allows producing different types and representations of an object using the same construction code.
- **Use Case**: When constructing an object involves many optional parameters or steps (e.g., complex SQL queries, HTTP requests).
- **LLM Instruction**: "Use the Builder pattern to construct `Request` objects. Allow chaining methods like `.setMethod()`, `.addHeader()`, and `.setBody()`."

### Singleton
- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Use Case**: Managing shared resources like database connections, logging, or configuration settings.
- **Warning**: Avoid overuse as it can introduce global state and make testing difficult.
- **LLM Instruction**: "Implement `DatabaseConnection` as a Singleton to ensure only one connection pool is created for the application."

## 2. Structural Patterns
* assembling objects and classes into larger structures while keeping them flexible.*

### Adapter
- **Intent**: Allow objects with incompatible interfaces to collaborate.
- **Use Case**: Integrating legacy code or third-party libraries where the interface doesn't match your application's needs.
- **LLM Instruction**: "Create an `AnalyticsAdapter` that wraps the third-party `GoogleAnalytics` library to match our internal `IAnalyticsService` interface."

### Composite
- **Intent**: Compose objects into tree structures to represent part-whole hierarchies. Treat individual objects and compositions uniformly.
- **Use Case**: UI component trees, file systems, or any recursive structure.
- **LLM Instruction**: "Implement a `Graphic` interface. `Dot` and `Circle` are leaf nodes, while `CompoundGraphic` is a container that can hold any `Graphic`."

### Facade
- **Intent**: Provide a simplified interface to a library, a framework, or any other complex set of classes.
- **Use Case**: Hiding the complexity of a subsystem (e.g., a video conversion library) behind a simple API.
- **LLM Instruction**: "Create a `VideoConverterFacade` that handles the complex interactions between `CodecFactory`, `BitrateReader`, and `AudioMixer`."

## 3. Behavioral Patterns
*Algorithms and the assignment of responsibilities between objects.*

### Observer
- **Intent**: Define a subscription mechanism to notify multiple objects about any events that happen to the object they're observing.
- **Use Case**: Event handling systems, UI updates when data changes.
- **LLM Instruction**: "Implement an `EventManager` that allows `Listeners` to subscribe to specific event types and be notified when `notify()` is called."

### Strategy
- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable.
- **Use Case**: Selecting a sorting algorithm at runtime, or choosing a payment method.
- **LLM Instruction**: "Create a `PaymentContext` that accepts a `PaymentStrategy`. Implement `CreditCardStrategy` and `PayPalStrategy` as concrete implementations."

### Command
- **Intent**: Turn a request into a stand-alone object that contains all information about the request.
- **Use Case**: Queueing operations, undo/redo functionality, or parameterizing objects with operations.
- **LLM Instruction**: "Encapsulate editor actions (Copy, Paste, Cut) as Command objects so they can be stored in a history stack for Undo functionality."

### State
- **Intent**: Let an object alter its behavior when its internal state changes. It appears as if the object changed its class.
- **Use Case**: Managing complex state transitions (e.g., a document in Draft, Review, and Published states).
- **LLM Instruction**: "Refactor the `Document` class to use the State pattern. Create `DraftState`, `ReviewState`, and `PublishedState` classes that handle `publish()` differently."