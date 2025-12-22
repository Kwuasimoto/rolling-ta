//! Rust reference tests for rolling-ta indicators.
//!
//! These tests verify Rust implementations match Python rolling-ta outputs.
//! Each indicator has its own module with a consolidated `*_modes_equivalent` test
//! that validates batch, streaming, and hybrid modes in one test.
//!
//! ## Test Structure
//!
//! - `common.rs` - Shared test utilities (xlsx loading, OHLCV builders, comparisons)
//! - `sma.rs` - Simple Moving Average test//! - `ema.rs` - Exponential Moving Average tests (pending migration)
//! - ... etc
//!
//! ## Column Reference (from tests/fixtures/data_sheets.py)
//!
//! All xlsx files have NO headers, columns are 0-indexed:
//! - btc-sma.xlsx: timestamp(0), close(1), sma(2)
//! - btc-ema.xlsx: timestamp(0), close(1), ema(2)
//! - ... etc

#[path = "rust/common.rs"]
mod common;

#[path = "rust/sma.rs"]
mod sma;

#[path = "rust/candles.rs"]
mod candles;

// TODO: Migrate to new Indicator trait (calc(&[Ohlcv]), next(), no update())
// #[path = "rust/ema.rs"]
// mod ema;
//
// #[path = "rust/wma.rs"]
// mod wma;
//
// #[path = "rust/hma.rs"]
// mod hma;
//
// #[path = "rust/rsi.rs"]
// mod rsi;
//
// #[path = "rust/bb.rs"]
// mod bb;
//
// #[path = "rust/atr.rs"]
// mod atr;
//
// #[path = "rust/obv.rs"]
// mod obv;
//
// #[path = "rust/adx.rs"]
// mod adx;
//
// #[path = "rust/vwap.rs"]
// mod vwap;
//
// #[path = "rust/roc.rs"]
// mod roc;
//
// #[path = "rust/lr.rs"]
// mod lr;
