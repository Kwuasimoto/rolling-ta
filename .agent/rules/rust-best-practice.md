## Table of Contents

- [1. Type System & Ownership](#1-type-system-ownership)
  - [❌ Don't](#-dont)
  - [✅ Do](#-do)
- [2. Unsafe Code](#2-unsafe-code)
- [3. Input Validation & Sanitization](#3-input-validation-sanitization)
- [4. Dependencies & Supply Chain](#4-dependencies-supply-chain)
- [5. Security Features & Configuration](#5-security-features-configuration)
- [6. Concurrency](#6-concurrency)
- [7. Cryptography](#7-cryptography)
  - [❌ Don't](#-dont)
  - [✅ Do](#-do)
- [8. Testing & Quality Assurance](#8-testing-quality-assurance)


# Workspace Rules: Rust Best Practices

This document aggregates critical Rust security and coding best practices for LLM code generation.

## 1. Type System & Ownership

### ❌ Don't
- **Primitive Obsession**: Avoid using raw primitives (`u32`, `String`) for domain concepts.
- **`Any` / `dyn Trait`**: Avoid unless strictly necessary.
- **`unwrap()`**: Do not use `.unwrap()` or `.expect()` in production code; handle errors gracefully.

### ✅ Do
- **NewType Pattern**: Use tuple structs to enforce type safety.
  ```rust
  struct UserId(u32);
  struct OrderId(u32);
  ```
- **Safe Abstractions**: Prefer `Option<T>` and `Result<T, E>` over nulls or exceptions.
- **Ownership**: Leverage borrow checker to prevent memory errors.

## 2. Unsafe Code

- **Minimize**: Only use `unsafe` when interacting with FFI or hardware.
- **Isolate**: Wrap `unsafe` blocks in safe abstractions.
- **Review**: Document safety comments (`// SAFETY: ...`) explaining why the block is safe.

## 3. Input Validation & Sanitization

- **Untrusted Input**: Treat all external data (user, network, file) as untrusted.
- **Sanitization**: Validate lengths, ranges, and patterns. Escape special characters to prevent Injection attacks.
- **Parsing**: Use robust parsing libraries (e.g., `serde`) instead of manual string manipulation.

## 4. Dependencies & Supply Chain

- **Audit**: Run `cargo audit` regularly to detect vulnerabilities.
- **Update**: Keep dependencies fresh.
- **Vet**: Prefer widely used, maintained crates. Avoid abandoned libraries.

## 5. Security Features & Configuration

- **Overflow Checks**: Enable integer overflow checks in `release` profile.
  ```toml
  [profile.release]
  overflow-checks = true
  ```
- **Mitigations**: Do not disable stack canaries or ASLR.
- **Sandboxing**: Use OS-level sandboxing where applicable.

## 6. Concurrency

- **Safe Primitives**: Use `Arc<Mutex<T>>` or `RwLock<T>` for shared state.
- **Channels**: Prefer message passing (`mpsc`) over shared memory.
- **Send/Sync**: Rely on compiler checks for thread safety; do not manually implement `Send`/`Sync` unless necessary and verified.

## 7. Cryptography

### ❌ Don't
- **Roll Your Own**: Never implement crypto algorithms manually.
- **Weak Algos**: Avoid MD5, SHA1, DES.

### ✅ Do
- **Proven Crates**: Use `ring`, `rustls`, `aes-gcm`, `chacha20poly1305`.
- **High-Level APIs**: Prefer libraries that abstract nonce management and key generation.

## 8. Testing & Quality Assurance

- **Clippy**: Enforce lints with `cargo clippy -- -D warnings`.
- **Fuzzing**: Use `cargo-fuzz` for input parsing logic.
- **Security Tests**: Write tests specifically for edge cases and malicious inputs.
