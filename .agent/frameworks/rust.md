---
id: rust
title: "Rust Best Practices"
description: "Critical Rust security, safety, and coding best practices for the backend."
category: frameworks
tags: [rust, security, safety, best-practices, tauri]
type: reference
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [typescript, ../architecture/solid-principles]
---
## Table of Contents

- [1. Type System & Ownership](#1-type-system-ownership)
  - [❌ Don't](#-dont)
  - [✅ Do](#-do)
- [2. Unsafe Code](#2-unsafe-code)
- [3. Input Validation & Sanitization](#3-input-validation-sanitization)
- [4. Dependencies & Supply Chain](#4-dependencies-supply-chain)
- [5. Security Features & Configuration](#5-security-features-configuration)
  - [Overflow Checks](#overflow-checks)
- [6. Concurrency](#6-concurrency)
- [7. Cryptography](#7-cryptography)
- [8. Testing & Quality Assurance](#8-testing-quality-assurance)
- [Tauri-Specific Considerations](#tauri-specific-considerations)
- [See Also](#see-also)


# Rust Best Practices

## 1. Type System & Ownership

### ❌ Don't
- **Primitive Obsession**: `u32`, `String`.
- **`unwrap()`**: Avoid in production.

### ✅ Do
- **NewType Pattern**: Tuple structs.
- **Safe Abstractions**: `Option<T>`, `Result<T, E>`.
- **Ownership**: Borrow checker.

## 2. Unsafe Code

- **Minimize**: Only for FFI/hardware.
- **Isolate**: Wrap in safe abstractions.
- **Review**: Document safety comments.

## 3. Input Validation & Sanitization

- **Untrusted Input**: Treat all external data as untrusted.
- **Sanitization**: Validate lengths, ranges.
- **Parsing**: Use `serde`.

## 4. Dependencies & Supply Chain

- **Audit**: `cargo audit`.
- **Update**: `cargo update`.

## 5. Security Features & Configuration

### Overflow Checks
Enable in release profile:
```toml
[profile.release]
overflow-checks = true
```

## 6. Concurrency

- **Safe Primitives**: `Arc<Mutex<T>>`, `RwLock<T>`.
- **Channels**: `mpsc`.

## 7. Cryptography

- **Proven Crates**: `rustls`, `aes_gcm`, `sha2`.
- **No Homebrew**: Never implement own crypto.

## 8. Testing & Quality Assurance

- **Clippy**: Enforce lints.
- **Fuzzing**: `cargo-fuzz`.
- **Security Tests**: Edge cases.

## Tauri-Specific Considerations

- **Command Security**: Validate inputs, return `Result`.
- **State Management**: Use `tauri::State`.

## See Also

- [SOLID Principles](../architecture/solid-principles.md)
- [Result Type](../architecture/result-type.md)
- [TypeScript](typescript.md)
