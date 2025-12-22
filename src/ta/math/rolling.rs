//! Rolling window utilities.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use crate::prelude::Ohlcv;

/// Type alias for a shared, thread-safe rolling window.
///
/// This enables multiple indicators to share a single window without
/// duplicating data. The window owner pushes candles with a write lock,
/// and indicators read snapshots with a read lock.
///
/// # Example
/// ```
/// use rolling_ta::ta::math::{SharedWindow, RollingWindow};
/// use rolling_ta::prelude::Ohlcv;
/// use std::sync::{Arc, RwLock};
///
/// // Create shared window
/// let window: SharedWindow = Arc::new(RwLock::new(
///     RollingWindow::with_timeframe(1440, 60)
/// ));
///
/// // Push candle (write lock)
/// {
///     let mut w = window.write().unwrap();
///     w.push(Ohlcv::from_close(100.0));
/// }
///
/// // Read snapshot (read lock)
/// let snapshot = {
///     let r = window.read().unwrap();
///     r.snapshot()
/// };
/// ```
pub type SharedWindow = Arc<RwLock<RollingWindow>>;

/// Temporal state for candle period detection.
///
/// Encapsulates timeframe interval and tracks the last processed timestamp.
/// Used by rolling windows to determine whether to push a new value or
/// update the latest value (for same-period candle updates).
#[derive(Debug, Clone, Copy, Default)]
pub struct Temporal {
    /// Interval in seconds (60=1m, 300=5m, 900=15m, 0=disabled)
    timeframe: i64,
    /// Timestamp of the last processed candle (seconds)
    last_timestamp: Option<i64>,
}

impl Temporal {
    /// Create with timeframe (0 = non-temporal mode).
    #[inline]
    pub fn new(timeframe: i64) -> Self {
        Self {
            timeframe,
            last_timestamp: None,
        }
    }

    /// Create disabled (always pushes, never updates).
    #[inline]
    pub fn disabled() -> Self {
        Self {
            timeframe: 0,
            last_timestamp: None,
        }
    }

    /// Check if incoming timestamp is in same candle period as last.
    ///
    /// Returns `false` if:
    /// - No last_timestamp recorded (first value)
    /// - Timeframe is 0 (non-temporal mode)
    /// - Timestamps are in different periods
    #[inline]
    pub fn is_same_period(&self, incoming_ts: i64) -> bool {
        if self.timeframe == 0 {
            return false;
        }
        match self.last_timestamp {
            None => false,
            Some(last_ts) => (incoming_ts / self.timeframe) == (last_ts / self.timeframe),
        }
    }

    /// Record a timestamp (call after pushing new candle).
    #[inline]
    pub fn record(&mut self, timestamp: i64) {
        self.last_timestamp = Some(timestamp);
    }

    /// Reset temporal state.
    #[inline]
    pub fn reset(&mut self) {
        self.last_timestamp = None;
    }

    /// Get timeframe.
    #[inline]
    pub fn timeframe(&self) -> i64 {
        self.timeframe
    }

    /// Get last timestamp.
    #[inline]
    pub fn last_timestamp(&self) -> Option<i64> {
        self.last_timestamp
    }

    /// Check if temporal mode is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.timeframe > 0
    }
}

/// Fixed-capacity rolling window.
///
/// Efficiently maintains a sliding window of values with O(1) push/pop.
/// Optionally supports temporal candle management via the `Temporal` field.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    buffer: VecDeque<Ohlcv>,
    capacity: usize,
    sum: f64, // Of close values for n capacity
    temporal: Temporal,
}

impl RollingWindow {
    /// Create a new rolling window with given capacity.
    /// Temporal mode is disabled by default.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
            temporal: Temporal::disabled(),
        }
    }

    /// Create a new rolling window with temporal management.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of values in the window
    /// * `timeframe` - Interval in seconds (60=1m, 300=5m, 900=15m)
    pub fn with_timeframe(capacity: usize, timeframe: i64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
            temporal: Temporal::new(timeframe),
        }
    }

    /// Push a candle, removing oldest if at capacity.
    /// Returns the removed candle if any.
    ///
    /// Note: This ignores temporal state. For temporal-aware operations,
    /// use `push_with_timestamp()` instead.
    #[inline]
    pub fn push(&mut self, candle: Ohlcv) -> Option<Ohlcv> {
        self.sum += candle.close.0;

        if self.buffer.len() >= self.capacity {
            let removed = self.buffer.pop_front().unwrap();
            self.sum -= removed.close.0;
            self.buffer.push_back(candle);
            Some(removed)
        } else {
            self.buffer.push_back(candle);
            None
        }
    }

    /// Push a single value (convenience wrapper for Ohlcv::from_close).
    ///
    /// This is useful for indicators that only work with close prices
    /// or single values (e.g., typical price for linear regression).
    /// Returns the close value of the removed candle if any.
    ///
    /// Note: This ignores temporal state. For temporal-aware operations,
    /// use `push_with_timestamp()` instead.
    #[inline]
    pub fn push_value(&mut self, value: f64) -> Option<f64> {
        self.push(Ohlcv::from_close(value)).map(|c| c.close.0)
    }

    /// Update the most recent candle (back of window).
    ///
    /// Adjusts cached sum accordingly. Used for updating a candle
    /// that is still forming (same time period).
    ///
    /// Returns the old candle, or `None` if window is empty.
    #[inline]
    pub fn update_back(&mut self, new_candle: Ohlcv) -> Option<Ohlcv> {
        if let Some(back) = self.buffer.back_mut() {
            let old = *back;
            self.sum = self.sum - old.close.0 + new_candle.close.0;
            *back = new_candle;
            Some(old)
        } else {
            None
        }
    }

    /// Process a candle with timestamp (push or update based on period).
    ///
    /// If temporal mode is enabled and the timestamp is in the same
    /// period as the last recorded timestamp, updates the latest candle.
    /// Otherwise, pushes a new candle.
    ///
    /// Returns `true` if pushed (new candle), `false` if updated (same candle).
    pub fn push_with_timestamp(&mut self, candle: Ohlcv) -> bool {
        if self.temporal.is_same_period(candle.timestamp.0) {
            self.update_back(candle);
            false
        } else {
            self.push(candle);
            self.temporal.record(candle.timestamp.0);
            true
        }
    }

    /// Process a value with timestamp (push or update based on period).
    ///
    /// Convenience wrapper that creates an Ohlcv from close price.
    /// See `push_with_timestamp` for full documentation.
    ///
    /// Returns `true` if pushed (new value), `false` if updated (same period).
    pub fn push_value_with_timestamp(&mut self, timestamp: i64, value: f64) -> bool {
        let candle = Ohlcv::new(timestamp, value, value, value, value, 0.0);
        self.push_with_timestamp(candle)
    }

    /// Get current sum of all values.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get current mean of all values.
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.buffer.is_empty() {
            f64::NAN
        } else {
            self.sum / self.buffer.len() as f64
        }
    }

    /// Number of values currently in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if window is full (at capacity).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the oldest value.
    #[inline]
    pub fn front(&self) -> Option<Ohlcv> {
        self.buffer.front().copied()
    }

    /// Get the newest value.
    #[inline]
    pub fn back(&self) -> Option<Ohlcv> {
        self.buffer.back().copied()
    }

    /// Get value at index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<Ohlcv> {
        self.buffer.get(index).copied()
    }

    /// Clear the window and reset temporal state.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.temporal.reset();
    }

    /// Iterate over values from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = Ohlcv> + '_ {
        self.buffer.iter().copied()
    }

    /// Get as slice (may not be contiguous).
    pub fn as_slices(&self) -> (&[Ohlcv], &[Ohlcv]) {
        self.buffer.as_slices()
    }

    pub fn to_vec(&self) -> Vec<Ohlcv> {
        self.buffer.iter().cloned().collect()
    }

    /// Get close values as a Vec<f64>.
    ///
    /// This is useful for indicators that only need close prices
    /// (e.g., linear regression on typical prices).
    #[inline]
    pub fn to_values(&self) -> Vec<f64> {
        self.buffer.iter().map(|c| c.close.0).collect()
    }

    /// Take a snapshot of all candles in the window.
    ///
    /// Returns a new `Vec<Ohlcv>` containing copies of all candles from
    /// oldest to newest. This is the primary method for indicators to
    /// obtain data from a shared window.
    ///
    /// # Example
    /// ```
    /// use rolling_ta::ta::math::RollingWindow;
    /// use rolling_ta::prelude::Ohlcv;
    ///
    /// let mut window = RollingWindow::new(3);
    /// window.push(Ohlcv::from_close(1.0));
    /// window.push(Ohlcv::from_close(2.0));
    ///
    /// let snapshot = window.snapshot();
    /// assert_eq!(snapshot.len(), 2);
    /// ```
    #[inline]
    pub fn snapshot(&self) -> Vec<Ohlcv> {
        self.buffer.iter().copied().collect()
    }

    /// Take a snapshot of the last N candles in the window.
    ///
    /// Returns a new `Vec<Ohlcv>` containing the most recent N candles.
    /// If the window has fewer than N candles, returns all available candles.
    ///
    /// This is useful for indicators that only need a subset of the window
    /// (e.g., a 14-period SMA only needs 14 candles, even if the shared
    /// window holds 1440 candles for other indicators like VWAP).
    ///
    /// # Example
    /// ```
    /// use rolling_ta::ta::math::RollingWindow;
    /// use rolling_ta::prelude::Ohlcv;
    ///
    /// let mut window = RollingWindow::new(10);
    /// for i in 1..=10 {
    ///     window.push(Ohlcv::from_close(i as f64));
    /// }
    ///
    /// let last_3 = window.snapshot_last(3);
    /// assert_eq!(last_3.len(), 3);
    /// assert_eq!(last_3[0].close.0, 8.0);
    /// assert_eq!(last_3[2].close.0, 10.0);
    /// ```
    #[inline]
    pub fn snapshot_last(&self, n: usize) -> Vec<Ohlcv> {
        let len = self.buffer.len();
        if n >= len {
            self.buffer.iter().copied().collect()
        } else {
            self.buffer.iter().skip(len - n).copied().collect()
        }
    }

    /// Access temporal state (immutable).
    #[inline]
    pub fn temporal(&self) -> &Temporal {
        &self.temporal
    }

    /// Access temporal state (mutable).
    #[inline]
    pub fn temporal_mut(&mut self) -> &mut Temporal {
        &mut self.temporal
    }
}

/// High-low tracking window for Donchian-style calculations.
///
/// Optionally supports temporal candle management via the `Temporal` field.
#[derive(Debug, Clone)]
pub struct HighLowWindow {
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    capacity: usize,
    temporal: Temporal,
}

impl HighLowWindow {
    /// Create a new high-low window with given capacity.
    /// Temporal mode is disabled by default.
    pub fn new(capacity: usize) -> Self {
        Self {
            highs: VecDeque::with_capacity(capacity),
            lows: VecDeque::with_capacity(capacity),
            capacity,
            temporal: Temporal::disabled(),
        }
    }

    /// Create a new high-low window with temporal management.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of high/low pairs in the window
    /// * `timeframe` - Interval in seconds (60=1m, 300=5m, 900=15m)
    pub fn with_timeframe(capacity: usize, timeframe: i64) -> Self {
        Self {
            highs: VecDeque::with_capacity(capacity),
            lows: VecDeque::with_capacity(capacity),
            capacity,
            temporal: Temporal::new(timeframe),
        }
    }

    /// Push high and low values.
    ///
    /// Note: This ignores temporal state. For temporal-aware operations,
    /// use `push_with_timestamp()` instead.
    pub fn push(&mut self, high: f64, low: f64) {
        if self.highs.len() >= self.capacity {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        self.highs.push_back(high);
        self.lows.push_back(low);
    }

    /// Update the latest high/low values (for same-period updates).
    ///
    /// Extends high upward and low downward (candle still forming).
    pub fn update_back(&mut self, high: f64, low: f64) {
        if let Some(h) = self.highs.back_mut() {
            *h = h.max(high);
        }
        if let Some(l) = self.lows.back_mut() {
            *l = l.min(low);
        }
    }

    /// Process high/low with timestamp (push or update based on period).
    ///
    /// If temporal mode is enabled and the timestamp is in the same
    /// period as the last recorded timestamp, updates the latest values.
    /// Otherwise, pushes new values.
    ///
    /// Returns `true` if pushed (new candle), `false` if updated (same candle).
    pub fn push_with_timestamp(&mut self, timestamp: i64, high: f64, low: f64) -> bool {
        if self.temporal.is_same_period(timestamp) {
            self.update_back(high, low);
            false
        } else {
            self.push(high, low);
            self.temporal.record(timestamp);
            true
        }
    }

    /// Get highest high in the window.
    #[inline]
    pub fn highest(&self) -> f64 {
        self.highs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get lowest low in the window.
    #[inline]
    pub fn lowest(&self) -> f64 {
        self.lows
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get midpoint (highest + lowest) / 2.
    #[inline]
    pub fn midpoint(&self) -> f64 {
        (self.highest() + self.lowest()) / 2.0
    }

    /// Check if window is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.highs.len() >= self.capacity
    }

    /// Number of values in window.
    #[inline]
    pub fn len(&self) -> usize {
        self.highs.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.highs.is_empty()
    }

    /// Clear the window and reset temporal state.
    pub fn clear(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.temporal.reset();
    }

    /// Access temporal state (immutable).
    #[inline]
    pub fn temporal(&self) -> &Temporal {
        &self.temporal
    }

    /// Access temporal state (mutable).
    #[inline]
    pub fn temporal_mut(&mut self) -> &mut Temporal {
        &mut self.temporal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_window_basic() {
        let mut window = RollingWindow::new(3);

        assert!(window.push_value(1.0).is_none());
        assert!(window.push_value(2.0).is_none());
        assert!(window.push_value(3.0).is_none());
        assert!(window.is_full());
        assert_eq!(window.sum(), 6.0);
        assert_eq!(window.mean(), 2.0);

        // Push beyond capacity
        let removed = window.push_value(4.0);
        assert_eq!(removed, Some(1.0));
        assert_eq!(window.sum(), 9.0); // 2 + 3 + 4
        assert_eq!(window.mean(), 3.0);
    }

    #[test]
    fn rolling_window_empty() {
        let window = RollingWindow::new(3);
        assert!(window.mean().is_nan());
        assert!(window.is_empty());
    }

    #[test]
    fn high_low_window() {
        let mut window = HighLowWindow::new(3);

        window.push(10.0, 5.0);
        window.push(12.0, 6.0);
        window.push(8.0, 4.0);

        assert_eq!(window.highest(), 12.0);
        assert_eq!(window.lowest(), 4.0);
        assert_eq!(window.midpoint(), 8.0);

        // Push beyond capacity
        window.push(9.0, 7.0);
        assert_eq!(window.highest(), 12.0); // 12, 8, 9 -> 12
        assert_eq!(window.lowest(), 4.0);   // 6, 4, 7 -> 4
    }

    // Temporal tests

    #[test]
    fn temporal_disabled() {
        let temporal = Temporal::disabled();
        assert_eq!(temporal.timeframe(), 0);
        assert!(!temporal.is_enabled());
        // Disabled temporal always returns false for is_same_period
        assert!(!temporal.is_same_period(1000));
    }

    #[test]
    fn temporal_same_period() {
        let mut temporal = Temporal::new(300); // 5-minute timeframe
        assert!(temporal.is_enabled());

        // No last timestamp yet - should return false
        assert!(!temporal.is_same_period(1000));

        // Record a timestamp
        temporal.record(1000);
        assert_eq!(temporal.last_timestamp(), Some(1000));

        // Same period: 1000/300 = 3, 1060/300 = 3
        assert!(temporal.is_same_period(1060));
        assert!(temporal.is_same_period(1199)); // Still period 3 (1199/300 = 3)

        // Different period: 1200/300 = 4, 1500/300 = 5
        assert!(!temporal.is_same_period(1200));
        assert!(!temporal.is_same_period(1500));
    }

    #[test]
    fn temporal_period_boundaries() {
        let mut temporal = Temporal::new(60); // 1-minute timeframe

        // Period 0: 0-59
        temporal.record(0);
        assert!(temporal.is_same_period(59)); // Still period 0
        assert!(!temporal.is_same_period(60)); // Period 1

        // Period 1: 60-119
        temporal.record(60);
        assert!(temporal.is_same_period(119)); // Still period 1
        assert!(!temporal.is_same_period(120)); // Period 2
    }

    #[test]
    fn temporal_reset() {
        let mut temporal = Temporal::new(60);
        temporal.record(1000);
        assert!(temporal.last_timestamp().is_some());

        temporal.reset();
        assert!(temporal.last_timestamp().is_none());
        // After reset, is_same_period returns false (no last timestamp)
        assert!(!temporal.is_same_period(1000));
    }

    #[test]
    fn rolling_window_update_back() {
        let mut window = RollingWindow::new(3);
        window.push_value(1.0);
        window.push_value(2.0);
        window.push_value(3.0);
        assert_eq!(window.sum(), 6.0);
        assert_eq!(window.back().map(|c| c.close.0), Some(3.0));

        // Update back from 3.0 to 5.0
        let old = window.update_back(Ohlcv::from_close(5.0));
        assert_eq!(old.map(|c| c.close.0), Some(3.0));
        assert_eq!(window.back().map(|c| c.close.0), Some(5.0));
        assert_eq!(window.sum(), 8.0); // 1 + 2 + 5 = 8
        assert_eq!(window.mean(), 8.0 / 3.0);
    }

    #[test]
    fn rolling_window_update_back_empty() {
        let mut window = RollingWindow::new(3);
        let old = window.update_back(Ohlcv::from_close(5.0));
        assert!(old.is_none());
    }

    #[test]
    fn rolling_window_push_with_timestamp() {
        // 5-minute timeframe (300 seconds)
        let mut window = RollingWindow::with_timeframe(14, 300);
        assert_eq!(window.temporal().timeframe(), 300);

        // First candle (pushed)
        assert!(window.push_value_with_timestamp(1000, 100.0)); // true - first candle
        assert_eq!(window.len(), 1);
        assert_eq!(window.back().map(|c| c.close.0), Some(100.0));

        // Same candle updates (same 5-min period: 1000/300 = 3)
        assert!(!window.push_value_with_timestamp(1060, 101.0)); // false - updated
        assert_eq!(window.len(), 1); // Still 1 entry
        assert_eq!(window.back().map(|c| c.close.0), Some(101.0));

        assert!(!window.push_value_with_timestamp(1120, 102.0)); // false - updated
        assert_eq!(window.len(), 1);
        assert_eq!(window.back().map(|c| c.close.0), Some(102.0));

        // New candle (different period: 1500/300 = 5)
        assert!(window.push_value_with_timestamp(1500, 105.0)); // true - new candle
        assert_eq!(window.len(), 2);
        assert_eq!(window.back().map(|c| c.close.0), Some(105.0));

        // Verify temporal state
        assert_eq!(window.temporal().last_timestamp(), Some(1500));
    }

    #[test]
    fn rolling_window_non_temporal_mode() {
        // Non-temporal mode (timeframe = 0 via new())
        let mut window = RollingWindow::new(3);
        assert!(!window.temporal().is_enabled());

        // Every push_value_with_timestamp should push (never update)
        assert!(window.push_value_with_timestamp(1000, 1.0));
        assert!(window.push_value_with_timestamp(1000, 2.0)); // Same timestamp, still pushes
        assert!(window.push_value_with_timestamp(1000, 3.0));
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn high_low_window_update_back() {
        let mut window = HighLowWindow::new(3);
        window.push(10.0, 5.0);
        assert_eq!(window.highest(), 10.0);
        assert_eq!(window.lowest(), 5.0);

        // Update with higher high and lower low
        window.update_back(12.0, 3.0);
        assert_eq!(window.highest(), 12.0);
        assert_eq!(window.lowest(), 3.0);

        // Update with values that don't extend the range
        window.update_back(11.0, 4.0);
        assert_eq!(window.highest(), 12.0); // Still 12 (max)
        assert_eq!(window.lowest(), 3.0);   // Still 3 (min)
    }

    #[test]
    fn high_low_window_push_with_timestamp() {
        // 1-minute timeframe (60 seconds)
        let mut window = HighLowWindow::with_timeframe(10, 60);

        // First candle at period 16 (960-1019)
        assert!(window.push_with_timestamp(960, 105.0, 100.0));
        assert_eq!(window.len(), 1);
        assert_eq!(window.highest(), 105.0);
        assert_eq!(window.lowest(), 100.0);

        // Same period - extends the candle (960/60 = 16, 990/60 = 16)
        assert!(!window.push_with_timestamp(990, 107.0, 99.0));
        assert_eq!(window.len(), 1);
        assert_eq!(window.highest(), 107.0); // Extended up
        assert_eq!(window.lowest(), 99.0);   // Extended down

        // Same period - partial extension (1010/60 = 16)
        assert!(!window.push_with_timestamp(1010, 106.0, 100.0));
        assert_eq!(window.len(), 1);
        assert_eq!(window.highest(), 107.0); // Still 107
        assert_eq!(window.lowest(), 99.0);   // Still 99

        // New period (1020/60 = 17)
        assert!(window.push_with_timestamp(1020, 103.0, 101.0));
        assert_eq!(window.len(), 2);
        assert_eq!(window.highest(), 107.0); // Max across both candles
        assert_eq!(window.lowest(), 99.0);   // Min across both candles
    }

    #[test]
    fn rolling_window_clear_resets_temporal() {
        let mut window = RollingWindow::with_timeframe(3, 60);
        window.push_value_with_timestamp(1000, 100.0);
        assert!(window.temporal().last_timestamp().is_some());

        window.clear();
        assert!(window.is_empty());
        assert!(window.temporal().last_timestamp().is_none());
    }

    #[test]
    fn high_low_window_clear_resets_temporal() {
        let mut window = HighLowWindow::with_timeframe(3, 60);
        window.push_with_timestamp(1000, 105.0, 100.0);
        assert!(window.temporal().last_timestamp().is_some());

        window.clear();
        assert!(window.is_empty());
        assert!(window.temporal().last_timestamp().is_none());
    }

    // Snapshot tests

    #[test]
    fn rolling_window_snapshot() {
        let mut window = RollingWindow::new(5);
        for i in 1..=3 {
            window.push(Ohlcv::from_close(i as f64));
        }

        let snapshot = window.snapshot();
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot[0].close.0, 1.0);
        assert_eq!(snapshot[1].close.0, 2.0);
        assert_eq!(snapshot[2].close.0, 3.0);
    }

    #[test]
    fn rolling_window_snapshot_empty() {
        let window = RollingWindow::new(5);
        let snapshot = window.snapshot();
        assert!(snapshot.is_empty());
    }

    #[test]
    fn rolling_window_snapshot_last() {
        let mut window = RollingWindow::new(10);
        for i in 1..=10 {
            window.push(Ohlcv::from_close(i as f64));
        }

        // Get last 3
        let last_3 = window.snapshot_last(3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[0].close.0, 8.0);
        assert_eq!(last_3[1].close.0, 9.0);
        assert_eq!(last_3[2].close.0, 10.0);

        // Get last 5
        let last_5 = window.snapshot_last(5);
        assert_eq!(last_5.len(), 5);
        assert_eq!(last_5[0].close.0, 6.0);
        assert_eq!(last_5[4].close.0, 10.0);
    }

    #[test]
    fn rolling_window_snapshot_last_more_than_available() {
        let mut window = RollingWindow::new(10);
        for i in 1..=3 {
            window.push(Ohlcv::from_close(i as f64));
        }

        // Request more than available - should return all
        let snapshot = window.snapshot_last(100);
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot[0].close.0, 1.0);
        assert_eq!(snapshot[2].close.0, 3.0);
    }

    #[test]
    fn rolling_window_snapshot_last_zero() {
        let mut window = RollingWindow::new(5);
        for i in 1..=3 {
            window.push(Ohlcv::from_close(i as f64));
        }

        let snapshot = window.snapshot_last(0);
        assert!(snapshot.is_empty());
    }
}
