You are very close. The main opportunity is to formalize how custom errors carry structured context while keeping your Result<T, E> flow ergonomic, type-safe, and consistent across Tauri boundaries.

Below is how I would optimize this.

1. Problems in the current shape

Record<string, any> loses type safety
Each error really has its own context type.

Constructor signature is too generic
You want each error to declare what args it needs.

ZodError example shadows the real ZodError
Naming collision and confusion.

Error factory doesn’t constrain args to the error type
So mistakes won’t be caught at compile time.

Ctor is required but treated as optional
API inconsistency.

2. Introduce a typed “contextual error” base

Create a base class that carries typed context:

export abstract class ContextualError<A = unknown> extends Error {
  readonly args: A;

  protected constructor(message: string, args: A) {
    super(message);
    this.args = args;
    this.name = new.target.name;
  }
}


Now every custom error has a well-defined args shape.

3. Strongly type the constructor
type ErrorCtor<E extends Error, A = unknown> =
  new (message: string, args: A) => E;


This ties:

the error type E

to its argument type A.

4. Refine the factory
function errorFactory<E extends Error, A>(
  message: string,
  Ctor: ErrorCtor<E, A>,
  args: A
): E {
  return new Ctor(message, args);
}

5. Improve err helper

Two overloads:

default Error with no args

custom error with typed args

export const ok = <T>(data: T): Result<T, never> => ({
  data,
  error: null,
});

export function err(message: string): Result<never, Error>;
export function err<E extends Error, A>(
  message: string,
  Ctor: ErrorCtor<E, A>,
  args: A
): Result<never, E>;
export function err<E extends Error, A>(
  message: string,
  Ctor?: ErrorCtor<E, A>,
  args?: A
): Result<never, Error | E> {
  if (!Ctor) {
    return { data: null, error: new Error(message) };
  }
  return { data: null, error: errorFactory(message, Ctor, args as A) };
}


This keeps usage clean while preserving type inference.

6. Implement a proper Zod validation error

Avoid naming collision with Zod’s own type:

import { ZodError as ZodSchemaError } from "zod";

export class ValidationError extends ContextualError<ZodSchemaError> {
  readonly validationMessage: string;

  constructor(message: string, error: ZodSchemaError) {
    super(message, error);

    const tree = error.flatten();
    const lines = [
      ...tree.formErrors,
      ...Object.entries(tree.fieldErrors).flatMap(
        ([field, errs]) => errs?.map(e => `${field}: ${e}`) ?? []
      ),
    ];

    this.validationMessage =
      lines.length > 0
        ? lines.join("\n")
        : "No validation errors reported.";
  }

  zodMessage() {
    return this.validationMessage;
  }
}


Key points:

Typed context: ZodSchemaError.

Keeps raw Zod error accessible as this.args.

Precomputes a friendly message.