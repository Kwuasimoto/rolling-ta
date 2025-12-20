//! Technical Analysis module.
//!
//! This module provides the core `Indicator` trait and all indicator implementations.

pub mod config;
pub mod error;
pub mod math;
pub mod state;
pub mod types;
pub mod utils;

// Indicator categories
pub mod momentum;
pub mod trend;
pub mod volatility;
pub mod volume;

use error::TAResult;
use state::IndicatorState;
use types::{Ohlcv, OhlcvSeries};

/// Core indicator trait.
///
/// All indicators implement this trait, providing a unified interface for
/// both batch calculation and streaming updates.
///
/// # Contract
///
/// - `calc()` processes historical batch data, transitions state to Ready
/// - `update()` processes single ticks after warmup is complete
/// - `reset()` returns indicator to Uninitialized state
///
/// # Example
///
/// ```ignore
/// use rolling_ta::prelude::*;
/// use rolling_ta::trend::SMA;
///
/// let mut sma = SMA::new(SMAConfig::new(14));
///
/// // Batch mode
/// let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, /* ... */]);
/// sma.calc(&data)?;
/// assert!(sma.state().is_ready());
///
/// // Streaming mode
/// let tick = Ohlcv::from_close(4.0);
/// if let Some(value) = sma.update(&tick)? {
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

    /// Batch calculation over historical data.
    ///
    /// Processes all data and transitions to `Ready` state.
    /// Returns `Err` if data is insufficient for warmup.
    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self>;

    /// Single-tick streaming update.
    ///
    /// Returns `None` during warmup, `Some(output)` when ready.
    /// Can be called without prior `calc()` - will warm up from ticks.
    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>>;

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
