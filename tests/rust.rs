//! Rust reference tests for rolling-ta indicators.
//!
//! These tests verify Rust implementations match Python rolling-ta outputs.
//! Each indicator has its own module with a consolidated `*_modes_equivalent` test
//! that validates batch, streaming, and hybrid modes in one test.
//!
//! ## Test Structure
//!
//! - `common.rs` - Shared test utilities (xlsx loading, OHLCV builders, comparisons)
//! - `sma.rs` - Simple Moving Average tests
//! - `ema.rs` - Exponential Moving Average tests (pending)
//! - ... etc
//!
//! ## Column Reference (from tests/fixtures/data_sheets.py)
//!
//! All xlsx files have NO headers, columns are 0-indexed:
//! - btc-sma.xlsx: timestamp(0), close(1), sma(2)
//! - btc-ema.xlsx: timestamp(0), close(1), ema(2)
//! - btc-wma.xlsx: timestamp(0), close(1), weights(2), weighted_sum(3), wma(4)
//! - btc-hma.xlsx: timestamp(0), close(1), ..., hma(11)
//! - btc-rsi.xlsx: timestamp(0), close(1), ..., rsi(6)
//! - btc-bb.xlsx: timestamp(0), close(1), sma(2), upper(3), lower(4)
//! - btc-atr.xlsx: timestamp(0), high(1), low(2), close(3), tr(4), atr(5)
//! - btc-obv.xlsx: timestamp(0), close(1), volume(2), up(3), down(4), obv(5)
//! - btc-adx.xlsx: timestamp(0), high(1), low(2), close(3), ..., +dmi(12), -dmi(13), dx(14), adx(15)
//! - btc-linear_regression.xlsx: timestamp(0), high(1), low(2), close(3), typical(4),
//!                               row(5), intercept(6), slope(7), lr2(8), forecast(9)
//! - btc-vwap.xlsx: timestamp(0), timestamp_mod(1), high(2), low(3), close(4),
//!                  typical(5), volume(6), raw_accum(7), vol_accum(8), vwap(9)
//! - btc-roc.xlsx: timestamp(0), open(1), high(2), low(3), close(4), volume(5), roc_14(6)

#[path = "rust/common.rs"]
mod common;

#[path = "rust/sma.rs"]
mod sma;

#[path = "rust/ema.rs"]
mod ema;

#[path = "rust/wma.rs"]
mod wma;

#[path = "rust/hma.rs"]
mod hma;

#[path = "rust/rsi.rs"]
mod rsi;

#[path = "rust/bb.rs"]
mod bb;

#[path = "rust/atr.rs"]
mod atr;

#[path = "rust/obv.rs"]
mod obv;

#[path = "rust/adx.rs"]
mod adx;

#[path = "rust/vwap.rs"]
mod vwap;

#[path = "rust/roc.rs"]
mod roc;

#[path = "rust/lr.rs"]
mod lr;
