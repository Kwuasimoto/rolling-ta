//! Temporal aggregation for tick-to-candle conversion.
//!
//! This module provides the `CandleBuilder` component for aggregating
//! incoming ticks into OHLCV candles based on timeframe boundaries.
//!
//! # Architecture (SOLID / SRP)
//!
//! - **CandleBuilder**: Aggregates ticks → candles (single responsibility)
//! - **RollingWindow**: Stores candles (separate responsibility)
//! - **Indicators**: Compute from `&[Ohlcv]` slices (separate responsibility)
//!
//! # Real-Time Updates
//!
//! CandleBuilder supports real-time trading scenarios:
//! - `push()` returns `Some(candle)` when a candle completes
//! - `current()` provides the incomplete candle for live display
//! - Indicators can optionally include the incomplete candle in snapshots
//!
//! # Example
//!
//! ```rust
//! use rolling_ta::ta::temporal::{CandleBuilder, Tick, Timeframe};
//! use rolling_ta::ta::math::RollingWindow;
//! use rolling_ta::prelude::Ohlcv;
//!
//! // Create 1-minute candle builder
//! let mut builder = CandleBuilder::new(Timeframe::M1);
//! let mut window = RollingWindow::new(100);
//!
//! // Process ticks
//! let tick1 = Tick::new(1000, 100.5, 10.0);
//! if let Some(completed) = builder.push(&tick1) {
//!     window.push(completed);
//! }
//!
//! // Access incomplete candle for real-time display
//! if let Some(current) = builder.current() {
//!     println!("Current: O={} H={} L={} C={}",
//!         current.open.0, current.high.0, current.low.0, current.close.0);
//! }
//! ```

use crate::ta::types::{Ohlcv, Price, Volume};

/// Timeframe for candle aggregation.
///
/// Defines the duration of each candle. When a tick arrives that belongs
/// to a new time period, the current candle is finalized and a new one begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    // Minutes
    M1,
    M5,
    M15,
    M30,
    // Hours
    H1,
    H2,
    H4,
    H8,
    H12,
    // Days
    D1,
    D3,
    // Weekly
    W1,
    // Monthly (approximation: 30 days)
    MN1,
}

impl Timeframe {
    /// Duration of this timeframe in seconds.
    #[inline]
    pub fn as_seconds(&self) -> i64 {
        match self {
            Self::M1 => 60,
            Self::M5 => 300,
            Self::M15 => 900,
            Self::M30 => 1800,
            Self::H1 => 3600,
            Self::H2 => 7200,
            Self::H4 => 14400,
            Self::H8 => 28800,
            Self::H12 => 43200,
            Self::D1 => 86400,
            Self::D3 => 259200,
            Self::W1 => 604800,
            Self::MN1 => 2592000, // 30 days approximation
        }
    }

    /// Calculate the period start timestamp for a given timestamp.
    ///
    /// Returns the beginning of the period that contains `timestamp`.
    ///
    /// # Example
    /// ```
    /// use rolling_ta::ta::temporal::Timeframe;
    ///
    /// let tf = Timeframe::M1; // 1-minute
    /// assert_eq!(tf.period_start(65), 60);  // 65 seconds → period starting at 60
    /// assert_eq!(tf.period_start(120), 120); // Exactly at boundary
    /// ```
    #[inline]
    pub fn period_start(&self, timestamp: i64) -> i64 {
        let secs = self.as_seconds();
        timestamp - (timestamp % secs)
    }

    /// Check if two timestamps are in different periods.
    ///
    /// Returns `true` if `curr_ts` starts a new period relative to `prev_ts`.
    #[inline]
    pub fn is_new_period(&self, prev_ts: i64, curr_ts: i64) -> bool {
        self.period_start(prev_ts) != self.period_start(curr_ts)
    }

    /// Create from interval in seconds.
    ///
    /// Returns the matching Timeframe variant, or `None` if not a standard interval.
    pub fn from_seconds(seconds: i64) -> Option<Self> {
        match seconds {
            60 => Some(Self::M1),
            300 => Some(Self::M5),
            900 => Some(Self::M15),
            1800 => Some(Self::M30),
            3600 => Some(Self::H1),
            7200 => Some(Self::H2),
            14400 => Some(Self::H4),
            28800 => Some(Self::H8),
            43200 => Some(Self::H12),
            86400 => Some(Self::D1),
            259200 => Some(Self::D3),
            604800 => Some(Self::W1),
            2592000 => Some(Self::MN1),
            _ => None,
        }
    }
}

impl Default for Timeframe {
    fn default() -> Self {
        Self::M1
    }
}

impl std::fmt::Display for Timeframe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::M1 => write!(f, "1m"),
            Self::M5 => write!(f, "5m"),
            Self::M15 => write!(f, "15m"),
            Self::M30 => write!(f, "30m"),
            Self::H1 => write!(f, "1h"),
            Self::H2 => write!(f, "2h"),
            Self::H4 => write!(f, "4h"),
            Self::H8 => write!(f, "8h"),
            Self::H12 => write!(f, "12h"),
            Self::D1 => write!(f, "1d"),
            Self::D3 => write!(f, "3d"),
            Self::W1 => write!(f, "1w"),
            Self::MN1 => write!(f, "1M"),
        }
    }
}

/// A single tick (trade/price update).
///
/// Represents the smallest unit of market data before aggregation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tick {
    /// Unix timestamp in seconds.
    pub timestamp: i64,
    /// Trade price.
    pub price: f64,
    /// Trade volume (0.0 if not applicable).
    pub volume: f64,
}

impl Tick {
    /// Create a new tick.
    #[inline]
    pub fn new(timestamp: i64, price: f64, volume: f64) -> Self {
        Self {
            timestamp,
            price,
            volume,
        }
    }

    /// Create a tick with no volume.
    #[inline]
    pub fn price_only(timestamp: i64, price: f64) -> Self {
        Self {
            timestamp,
            price,
            volume: 0.0,
        }
    }
}

/// Aggregates ticks into OHLCV candles based on timeframe boundaries.
///
/// # Single Responsibility
///
/// CandleBuilder's only job is tick → candle aggregation. It does NOT:
/// - Store historical candles (use `RollingWindow`)
/// - Compute indicators (use `Indicator` implementations)
/// - Manage thread-safety (wrap in `Arc<Mutex<>>` if needed)
///
/// # Real-Time Flow
///
/// ```text
/// Tick → CandleBuilder.push() → Option<Ohlcv> (complete) → RollingWindow
///                             ↳ current() (incomplete) → real-time display
/// ```
///
/// # Example
///
/// ```rust
/// use rolling_ta::ta::temporal::{CandleBuilder, Tick, Timeframe};
///
/// let mut builder = CandleBuilder::new(Timeframe::M1);
///
/// // First tick starts a candle
/// assert!(builder.push(&Tick::new(60, 100.0, 10.0)).is_none());
/// assert!(builder.current().is_some());
///
/// // Same period updates the candle
/// assert!(builder.push(&Tick::new(90, 102.0, 5.0)).is_none());
///
/// // New period completes the candle and starts a new one
/// let completed = builder.push(&Tick::new(120, 101.0, 8.0));
/// assert!(completed.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct CandleBuilder {
    timeframe: Timeframe,
    current: Option<Ohlcv>,
}

impl CandleBuilder {
    /// Create a new CandleBuilder for the given timeframe.
    pub fn new(timeframe: Timeframe) -> Self {
        Self {
            timeframe,
            current: None,
        }
    }

    /// Process a tick.
    ///
    /// Returns `Some(candle)` if the tick completes a candle (crosses period boundary).
    /// Returns `None` if the tick updates the current incomplete candle.
    ///
    /// The returned candle has its timestamp set to the period start, not the tick timestamp.
    pub fn push(&mut self, tick: &Tick) -> Option<Ohlcv> {
        let period_start = self.timeframe.period_start(tick.timestamp);

        match &mut self.current {
            Some(candle) => {
                // Check if this tick belongs to a new period
                if candle.timestamp.0 != period_start {
                    // Complete the current candle
                    let completed = self.current.take().unwrap();

                    // Start new candle with this tick
                    self.current = Some(Ohlcv::new(
                        period_start,
                        tick.price,
                        tick.price,
                        tick.price,
                        tick.price,
                        tick.volume,
                    ));

                    Some(completed)
                } else {
                    // Update current candle (same period)
                    candle.high = Price(candle.high.0.max(tick.price));
                    candle.low = Price(candle.low.0.min(tick.price));
                    candle.close = Price(tick.price);
                    candle.volume = Volume(candle.volume.0 + tick.volume);
                    None
                }
            }
            None => {
                // First tick - start a new candle
                self.current = Some(Ohlcv::new(
                    period_start,
                    tick.price,
                    tick.price,
                    tick.price,
                    tick.price,
                    tick.volume,
                ));
                None
            }
        }
    }

    /// Get the current incomplete candle (for real-time display).
    ///
    /// Returns `None` if no ticks have been processed yet.
    #[inline]
    pub fn current(&self) -> Option<&Ohlcv> {
        self.current.as_ref()
    }

    /// Get a clone of the current candle.
    ///
    /// Useful for including in snapshots for real-time indicator updates.
    #[inline]
    pub fn current_candle(&self) -> Option<Ohlcv> {
        self.current
    }

    /// Force-complete the current candle.
    ///
    /// Use this for market close or other scenarios where you need to
    /// finalize the current candle regardless of time boundaries.
    ///
    /// Returns the completed candle, or `None` if no candle was in progress.
    #[inline]
    pub fn flush(&mut self) -> Option<Ohlcv> {
        self.current.take()
    }

    /// Get the configured timeframe.
    #[inline]
    pub fn timeframe(&self) -> Timeframe {
        self.timeframe
    }

    /// Reset to initial state, discarding any incomplete candle.
    #[inline]
    pub fn reset(&mut self) {
        self.current = None;
    }

    /// Check if a candle is currently being built.
    #[inline]
    pub fn has_current(&self) -> bool {
        self.current.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Timeframe Tests
    // ============================================================

    #[test]
    fn timeframe_as_seconds() {
        assert_eq!(Timeframe::M1.as_seconds(), 60);
        assert_eq!(Timeframe::M5.as_seconds(), 300);
        assert_eq!(Timeframe::M15.as_seconds(), 900);
        assert_eq!(Timeframe::M30.as_seconds(), 1800);
        assert_eq!(Timeframe::H1.as_seconds(), 3600);
        assert_eq!(Timeframe::D1.as_seconds(), 86400);
        assert_eq!(Timeframe::W1.as_seconds(), 604800);
    }

    #[test]
    fn timeframe_period_start() {
        let tf = Timeframe::M1; // 60 seconds

        // Period boundaries
        assert_eq!(tf.period_start(0), 0);
        assert_eq!(tf.period_start(59), 0);
        assert_eq!(tf.period_start(60), 60);
        assert_eq!(tf.period_start(119), 60);
        assert_eq!(tf.period_start(120), 120);

        // 5-minute timeframe
        let tf5 = Timeframe::M5; // 300 seconds
        assert_eq!(tf5.period_start(0), 0);
        assert_eq!(tf5.period_start(299), 0);
        assert_eq!(tf5.period_start(300), 300);
        assert_eq!(tf5.period_start(599), 300);
    }

    #[test]
    fn timeframe_is_new_period() {
        let tf = Timeframe::M1;

        // Same period
        assert!(!tf.is_new_period(0, 59));
        assert!(!tf.is_new_period(60, 119));

        // Different periods
        assert!(tf.is_new_period(59, 60));
        assert!(tf.is_new_period(0, 60));
        assert!(tf.is_new_period(60, 120));
    }

    #[test]
    fn timeframe_from_seconds() {
        assert_eq!(Timeframe::from_seconds(60), Some(Timeframe::M1));
        assert_eq!(Timeframe::from_seconds(300), Some(Timeframe::M5));
        assert_eq!(Timeframe::from_seconds(3600), Some(Timeframe::H1));
        assert_eq!(Timeframe::from_seconds(123), None); // Non-standard
    }

    #[test]
    fn timeframe_display() {
        assert_eq!(format!("{}", Timeframe::M1), "1m");
        assert_eq!(format!("{}", Timeframe::M5), "5m");
        assert_eq!(format!("{}", Timeframe::H1), "1h");
        assert_eq!(format!("{}", Timeframe::D1), "1d");
        assert_eq!(format!("{}", Timeframe::W1), "1w");
        assert_eq!(format!("{}", Timeframe::MN1), "1M");
    }

    #[test]
    fn timeframe_default() {
        assert_eq!(Timeframe::default(), Timeframe::M1);
    }

    // ============================================================
    // Tick Tests
    // ============================================================

    #[test]
    fn tick_new() {
        let tick = Tick::new(1000, 100.5, 50.0);
        assert_eq!(tick.timestamp, 1000);
        assert_eq!(tick.price, 100.5);
        assert_eq!(tick.volume, 50.0);
    }

    #[test]
    fn tick_price_only() {
        let tick = Tick::price_only(1000, 100.5);
        assert_eq!(tick.timestamp, 1000);
        assert_eq!(tick.price, 100.5);
        assert_eq!(tick.volume, 0.0);
    }

    // ============================================================
    // CandleBuilder Tests
    // ============================================================

    #[test]
    fn candle_builder_first_tick() {
        let mut builder = CandleBuilder::new(Timeframe::M1);
        assert!(!builder.has_current());

        // First tick should not return a completed candle
        let result = builder.push(&Tick::new(65, 100.0, 10.0));
        assert!(result.is_none());
        assert!(builder.has_current());

        // Check the current candle
        let current = builder.current().unwrap();
        assert_eq!(current.timestamp.0, 60); // Period start, not tick timestamp
        assert_eq!(current.open.0, 100.0);
        assert_eq!(current.high.0, 100.0);
        assert_eq!(current.low.0, 100.0);
        assert_eq!(current.close.0, 100.0);
        assert_eq!(current.volume.0, 10.0);
    }

    #[test]
    fn candle_builder_same_period_updates() {
        let mut builder = CandleBuilder::new(Timeframe::M1);

        // First tick
        builder.push(&Tick::new(60, 100.0, 10.0));

        // Higher price - should update high
        let result = builder.push(&Tick::new(70, 105.0, 5.0));
        assert!(result.is_none());

        let current = builder.current().unwrap();
        assert_eq!(current.open.0, 100.0); // Unchanged
        assert_eq!(current.high.0, 105.0); // Updated
        assert_eq!(current.low.0, 100.0); // Unchanged
        assert_eq!(current.close.0, 105.0); // Updated
        assert_eq!(current.volume.0, 15.0); // Accumulated

        // Lower price - should update low
        builder.push(&Tick::new(80, 98.0, 8.0));
        let current = builder.current().unwrap();
        assert_eq!(current.high.0, 105.0);
        assert_eq!(current.low.0, 98.0);
        assert_eq!(current.close.0, 98.0);
        assert_eq!(current.volume.0, 23.0);
    }

    #[test]
    fn candle_builder_new_period_completes() {
        let mut builder = CandleBuilder::new(Timeframe::M1);

        // Build first candle
        builder.push(&Tick::new(60, 100.0, 10.0));
        builder.push(&Tick::new(80, 105.0, 5.0));
        builder.push(&Tick::new(100, 102.0, 8.0));

        // New period should complete the first candle
        let completed = builder.push(&Tick::new(120, 103.0, 12.0));
        assert!(completed.is_some());

        let completed = completed.unwrap();
        assert_eq!(completed.timestamp.0, 60);
        assert_eq!(completed.open.0, 100.0);
        assert_eq!(completed.high.0, 105.0);
        assert_eq!(completed.low.0, 100.0);
        assert_eq!(completed.close.0, 102.0);
        assert_eq!(completed.volume.0, 23.0);

        // Current should be the new candle
        let current = builder.current().unwrap();
        assert_eq!(current.timestamp.0, 120);
        assert_eq!(current.open.0, 103.0);
        assert_eq!(current.volume.0, 12.0);
    }

    #[test]
    fn candle_builder_flush() {
        let mut builder = CandleBuilder::new(Timeframe::M1);

        // No candle yet
        assert!(builder.flush().is_none());

        // Build a candle
        builder.push(&Tick::new(60, 100.0, 10.0));
        builder.push(&Tick::new(80, 105.0, 5.0));

        // Flush should return the incomplete candle
        let flushed = builder.flush();
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().high.0, 105.0);

        // Builder should be empty now
        assert!(!builder.has_current());
        assert!(builder.current().is_none());
    }

    #[test]
    fn candle_builder_reset() {
        let mut builder = CandleBuilder::new(Timeframe::M1);
        builder.push(&Tick::new(60, 100.0, 10.0));
        assert!(builder.has_current());

        builder.reset();
        assert!(!builder.has_current());
        assert!(builder.current().is_none());
    }

    #[test]
    fn candle_builder_multiple_periods() {
        let mut builder = CandleBuilder::new(Timeframe::M1);
        let mut completed_candles = Vec::new();

        // Simulate 3 complete candles + 1 incomplete
        let ticks = [
            // Period 60-119
            Tick::new(60, 100.0, 10.0),
            Tick::new(90, 105.0, 5.0),
            // Period 120-179
            Tick::new(120, 102.0, 8.0),
            Tick::new(150, 108.0, 12.0),
            // Period 180-239
            Tick::new(180, 104.0, 6.0),
            Tick::new(210, 101.0, 4.0),
            // Period 240+ (incomplete)
            Tick::new(240, 106.0, 9.0),
        ];

        for tick in &ticks {
            if let Some(completed) = builder.push(tick) {
                completed_candles.push(completed);
            }
        }

        assert_eq!(completed_candles.len(), 3);
        assert_eq!(completed_candles[0].timestamp.0, 60);
        assert_eq!(completed_candles[1].timestamp.0, 120);
        assert_eq!(completed_candles[2].timestamp.0, 180);

        // Current should be period 240
        let current = builder.current().unwrap();
        assert_eq!(current.timestamp.0, 240);
    }

    #[test]
    fn candle_builder_5_minute_timeframe() {
        let mut builder = CandleBuilder::new(Timeframe::M5); // 300 seconds

        // Ticks within first 5-minute period
        builder.push(&Tick::new(0, 100.0, 10.0));
        builder.push(&Tick::new(60, 102.0, 5.0));
        builder.push(&Tick::new(180, 98.0, 8.0));
        builder.push(&Tick::new(299, 101.0, 3.0));

        let current = builder.current().unwrap();
        assert_eq!(current.timestamp.0, 0);
        assert_eq!(current.open.0, 100.0);
        assert_eq!(current.high.0, 102.0);
        assert_eq!(current.low.0, 98.0);
        assert_eq!(current.close.0, 101.0);
        assert_eq!(current.volume.0, 26.0);

        // Cross to next period
        let completed = builder.push(&Tick::new(300, 103.0, 7.0));
        assert!(completed.is_some());
        assert_eq!(completed.unwrap().timestamp.0, 0);

        let current = builder.current().unwrap();
        assert_eq!(current.timestamp.0, 300);
    }

    #[test]
    fn candle_builder_current_candle_clone() {
        let mut builder = CandleBuilder::new(Timeframe::M1);
        builder.push(&Tick::new(60, 100.0, 10.0));

        // current_candle returns a clone
        let clone = builder.current_candle();
        assert!(clone.is_some());
        assert_eq!(clone.unwrap().open.0, 100.0);

        // Original still accessible
        assert!(builder.current().is_some());
    }

    #[test]
    fn candle_builder_timeframe_getter() {
        let builder = CandleBuilder::new(Timeframe::H1);
        assert_eq!(builder.timeframe(), Timeframe::H1);
    }

    // ============================================================
    // Integration: CandleBuilder + RollingWindow
    // ============================================================

    #[test]
    fn candle_builder_with_rolling_window() {
        use crate::ta::math::RollingWindow;

        let mut builder = CandleBuilder::new(Timeframe::M1);
        let mut window = RollingWindow::new(10);

        // Simulate tick stream
        let ticks = [
            Tick::new(60, 100.0, 10.0),
            Tick::new(90, 105.0, 5.0),
            Tick::new(120, 102.0, 8.0), // New period
            Tick::new(150, 108.0, 12.0),
            Tick::new(180, 104.0, 6.0), // New period
        ];

        for tick in &ticks {
            if let Some(completed) = builder.push(tick) {
                window.push(completed);
            }
        }

        // Should have 2 completed candles in window
        assert_eq!(window.len(), 2);

        // Can take snapshot for indicators
        let snapshot = window.snapshot();
        assert_eq!(snapshot[0].timestamp.0, 60);
        assert_eq!(snapshot[1].timestamp.0, 120);

        // For real-time, can include current incomplete candle
        if let Some(current) = builder.current_candle() {
            let mut live_snapshot = snapshot.clone();
            live_snapshot.push(current);
            assert_eq!(live_snapshot.len(), 3);
            assert_eq!(live_snapshot[2].timestamp.0, 180);
        }
    }
}