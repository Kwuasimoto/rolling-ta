//! Rust reference tests for rolling-ta indicators.
//!
//! These tests verify Rust implementations match Python rolling-ta outputs.
//! Each indicator has its own module with tests for:
//! - `*_batch_vs_reference` - Validates calc() against Python xlsx data
//! - `*_streaming_next_vs_batch` - Validates next() matches calc() output
//! - `*_next_with_fixed_window` - Validates fixed-window next() pattern
//!
//! ## Test Status
//!
//! All batch and streaming tests pass for enabled indicators.
//! Fixed window tests pass for: SMA, EMA, WMA, HMA, Ichimoku, BB, ROC, Donchian

#[path = "rust/common.rs"]
mod common;

// ============================================================
// Trend Indicators
// ============================================================

// SMA: All 3 tests pass
#[path = "rust/sma.rs"]
mod sma;

// EMA: All 3 tests pass
#[path = "rust/ema.rs"]
mod ema;

// WMA: All 3 tests pass
#[path = "rust/wma.rs"]
mod wma;

// HMA: All 3 tests pass
#[path = "rust/hma.rs"]
mod hma;

// Ichimoku: All 3 tests pass
#[path = "rust/ichimoku.rs"]
mod ichimoku;

// ADX: All 2 tests pass (batch + streaming)
#[path = "rust/adx.rs"]
mod adx;

// DMI: All 2 tests pass (batch + streaming)
#[path = "rust/dmi.rs"]
mod dmi;

// LR: All 7 tests pass (batch + streaming for LR, LR2, LRF + optimized path)
#[path = "rust/lr.rs"]
mod lr;

// ============================================================
// Momentum Indicators
// ============================================================

// RSI: All 2 tests pass (batch + streaming)
#[path = "rust/rsi.rs"]
mod rsi;

// StochRSI: All 2 tests pass (batch + streaming, no fixed_window - uses backward smoothing)
#[path = "rust/stoch_rsi.rs"]
mod stoch_rsi;

// ROC: All 3 tests pass (batch + streaming + fixed_window)
#[path = "rust/roc.rs"]
mod roc;

// BOP: All 2 tests pass (batch + streaming)
#[path = "rust/bop.rs"]
mod bop;

// ============================================================
// Volatility Indicators
// ============================================================

// ATR: All 3 tests pass (TR batch + ATR batch + streaming)
#[path = "rust/atr.rs"]
mod atr;

// BB: All 3 tests pass (batch + streaming + fixed_window)
#[path = "rust/bb.rs"]
mod bb;

// Donchian: All 3 tests pass (batch + streaming + fixed_window)
#[path = "rust/donchian.rs"]
mod donchian;

// ============================================================
// Volume Indicators
// ============================================================

// OBV: All 2 tests pass (batch + streaming)
#[path = "rust/obv.rs"]
mod obv;

// VWAP: All 2 tests pass (batch + streaming)
#[path = "rust/vwap.rs"]
mod vwap;

// MFI: All 2 tests pass (batch + streaming, no fixed_window - requires accumulated state)
#[path = "rust/mfi.rs"]
mod mfi;

// CMF: 1 test passes (streaming only - no Python reference data in xlsx)
#[path = "rust/cmf.rs"]
mod cmf;

// ============================================================
// Candle/Window Tests
// ============================================================

#[path = "rust/candles.rs"]
mod candles;

