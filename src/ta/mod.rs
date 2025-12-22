//! Technical Analysis module.
//!
//! This module provides the core `Indicator` trait and all indicator implementations.

pub mod config;
pub mod error;
pub mod manager;
pub mod math;
pub mod state;
pub mod temporal;
pub mod types;
pub mod utils;

// Indicator categories
pub mod momentum;
pub mod trend;
pub mod volatility;
pub mod volume;

use error::TAResult;
use state::IndicatorState;
use types::Ohlcv;

/// Core indicator trait.
///
/// All indicators implement this trait, providing a unified interface for
/// batch calculation and streaming updates.
///
/// # Contract
///
/// - `calc()` processes historical batch data, transitions state to Ready
/// - `next()` processes snapshot slices from a shared window (parallel-safe)
/// - `reset()` returns indicator to Uninitialized state
///
/// # Data Format
///
/// All methods use `&[Ohlcv]` (array-of-structs) for consistency:
/// - Tests mirror runtime exactly
/// - No conversion between formats
/// - Use `Ohlcv::closes()`, `Ohlcv::highs()` etc. to extract fields
///
/// # Example
///
/// ```ignore
/// use rolling_ta::prelude::*;
/// use rolling_ta::trend::SMA;
///
/// let mut sma = SMA::new(SMAConfig::new(14));
///
/// // Batch mode (historical backfill)
/// let candles: Vec<Ohlcv> = load_from_db();
/// sma.calc(&candles)?;
/// assert!(sma.state().is_ready());
///
/// // Streaming mode (shared window snapshot)
/// let snapshot = shared_window.read().unwrap().snapshot_last(14);
/// if let Some(value) = sma.next(&snapshot) {
///     println!("SMA: {}", value);
/// }
/// ```
pub trait Indicator: Send + Sync {
    /// Output type produced by this indicator.
    ///
    /// Single-value indicators use `f64`.
    /// Multi-value indicators use custom output structs.
    type Output: Clone + Send;

    /// Configuration type for this indicator.
    type Config: Clone + Default;

    /// Current calculation state.
    fn state(&self) -> IndicatorState;

    /// Batch calculation over historical candle data.
    ///
    /// Processes all data and transitions to `Ready` state.
    /// Returns `Err` if data is insufficient for warmup.
    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self>;

    /// Streaming calculation from shared-window snapshot.
    ///
    /// Receives a slice of candles (typically last N based on warmup_period).
    /// Returns `None` during warmup, `Some(output)` when ready.
    ///
    /// This method is designed for use with `SharedWindow` and Rayon:
    /// - Takes immutable snapshot (parallel-safe)
    /// - Computes directly from slice (no internal window needed)
    /// - Tracks state and history internally
    ///
    /// # Example
    /// ```ignore
    /// let snapshot = window.read().unwrap().snapshot_last(indicator.warmup_period());
    /// if let Some(value) = indicator.next(&snapshot) {
    ///     // Use value
    /// }
    /// ```
    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output>;

    /// Get the most recent output value.
    ///
    /// Returns `None` if indicator hasn't produced output yet.
    fn latest(&self) -> Option<Self::Output>;

    /// Reset indicator to uninitialized state.
    ///
    /// Clears all internal state and history.
    fn reset(&mut self);

    /// Minimum data points required before producing output.
    fn warmup_period(&self) -> usize;
}

/// Extension trait for indicators that store historical values.
pub trait HistoricalIndicator: Indicator {
    /// Get all calculated output values.
    fn history(&self) -> &[Self::Output];

    /// Get output at specific index.
    ///
    /// Supports negative indexing: -1 = last, -2 = second to last, etc.
    fn get(&self, index: isize) -> Option<Self::Output>;

    /// Number of stored values.
    fn len(&self) -> usize;

    /// Check if history is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
