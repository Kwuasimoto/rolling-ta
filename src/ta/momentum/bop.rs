//! Balance of Power (BOP) indicator.
//!
//! BOP measures the strength of buyers versus sellers by comparing
//! the close-open range to the high-low range.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use std::collections::VecDeque;

use crate::ta::{
    config::BOPConfig,
    error::TAResult,
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Balance of Power indicator.
///
/// Measures buyer/seller strength by comparing close-open range to high-low range.
///
/// # Formula
///
/// Raw BOP = (Close - Open) / (High - Low)
/// BOP = SMA(Raw BOP, smoothing)
///
/// # Interpretation
///
/// - BOP > 0: Buyers in control
/// - BOP < 0: Sellers in control
/// - BOP near 0: Equilibrium
/// - Values range from -1 to +1
///
/// # Example
///
/// ```
/// use rolling_ta::ta::momentum::{BOP, BOPConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut bop = BOP::new(BOPConfig::new(14));
/// let candles = vec![
///     Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     // ... more candles
/// ];
/// bop.calc(&candles).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BOP {
    config: BOPConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,

    // Rolling window of raw BOP values for smoothing
    raw_bop_window: VecDeque<f64>,
    raw_bop_sum: f64,

    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl BOP {
    /// Create a new BOP indicator.
    pub fn new(config: BOPConfig) -> Self {
        let smoothing = config.smoothing.max(1);
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            raw_bop_window: VecDeque::with_capacity(smoothing),
            raw_bop_sum: 0.0,
            last_len: 0,
        }
    }

    /// Get the current BOP value.
    #[inline]
    pub fn bop_value(&self) -> f64 {
        self.latest.unwrap_or(f64::NAN)
    }

    /// Calculate raw BOP for a candle.
    #[inline]
    fn calculate_raw_bop(candle: &Ohlcv) -> f64 {
        let range = candle.high.0 - candle.low.0;
        if range > 0.0 {
            (candle.close.0 - candle.open.0) / range
        } else {
            0.0 // No range means no directional pressure
        }
    }

    /// Get effective smoothing period.
    #[inline]
    fn smoothing(&self) -> usize {
        self.config.smoothing.max(1)
    }

    /// Add raw BOP to window and return smoothed value.
    fn add_and_smooth(&mut self, raw_bop: f64) -> f64 {
        let smoothing = self.smoothing();

        // Remove oldest if at capacity
        if self.raw_bop_window.len() >= smoothing {
            if let Some(old) = self.raw_bop_window.pop_front() {
                self.raw_bop_sum -= old;
            }
        }

        // Add new value
        self.raw_bop_window.push_back(raw_bop);
        self.raw_bop_sum += raw_bop;

        // Return SMA
        self.raw_bop_sum / self.raw_bop_window.len() as f64
    }
}

impl Default for BOP {
    fn default() -> Self {
        Self::new(BOPConfig::default())
    }
}

impl Indicator for BOP {
    type Output = f64;
    type Config = BOPConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();
        let smoothing = self.smoothing();

        // Reset state
        self.history = Vec::with_capacity(n);
        self.raw_bop_window.clear();
        self.raw_bop_sum = 0.0;
        self.latest = None;

        if n == 0 {
            self.last_len = 0;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Fill NaN for warmup period
        for _ in 0..(smoothing - 1) {
            self.history.push(f64::NAN);
        }

        // Process all candles
        for i in 0..n {
            let raw_bop = Self::calculate_raw_bop(&data[i]);
            let smoothed = self.add_and_smooth(raw_bop);

            // After warmup, record the smoothed value
            if i >= smoothing - 1 {
                self.history.push(smoothed);
            }
        }

        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let smoothing = self.smoothing();

        if len == 0 {
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First time - process all candles
                for i in 0..len {
                    let raw_bop = Self::calculate_raw_bop(&candles[i]);
                    let smoothed = self.add_and_smooth(raw_bop);

                    if i >= smoothing - 1 {
                        self.history.push(smoothed);
                    }
                }

                self.latest = self.history.last().copied();
                self.last_len = len;
                self.state = IndicatorState::Ready;
                return self.latest;
            }

            // New candle(s) - process incrementally
            for i in self.last_len..len {
                let raw_bop = Self::calculate_raw_bop(&candles[i]);
                let smoothed = self.add_and_smooth(raw_bop);

                if i >= smoothing - 1 {
                    self.history.push(smoothed);
                }
            }

            self.latest = self.history.last().copied();
            self.last_len = len;
        }
        // BOP doesn't need same-candle update complexity since it's a simple SMA
        // The values would just update based on the current candle

        self.state = IndicatorState::Ready;
        self.latest
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.raw_bop_window.clear();
        self.raw_bop_sum = 0.0;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.smoothing()
    }
}

impl HistoricalIndicator for BOP {
    fn history(&self) -> &[Self::Output] {
        &self.history
    }

    fn get(&self, index: isize) -> Option<Self::Output> {
        let len = self.history.len() as isize;
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };
        self.history.get(actual_index).copied()
    }

    fn len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_candles(data: &[(f64, f64, f64, f64)]) -> Vec<Ohlcv> {
        // (open, high, low, close)
        data.iter()
            .enumerate()
            .map(|(i, &(o, h, l, c))| Ohlcv::new(i as i64, o, h, l, c, 1000.0))
            .collect()
    }

    #[test]
    fn bop_raw_calculation() {
        // Close at high: BOP = (105 - 100) / (105 - 100) = 1.0
        let candle_bullish = Ohlcv::new(0, 100.0, 105.0, 100.0, 105.0, 1000.0);
        assert!((BOP::calculate_raw_bop(&candle_bullish) - 1.0).abs() < 1e-10);

        // Close at low: BOP = (100 - 105) / (105 - 100) = -1.0
        let candle_bearish = Ohlcv::new(0, 105.0, 105.0, 100.0, 100.0, 1000.0);
        assert!((BOP::calculate_raw_bop(&candle_bearish) - (-1.0)).abs() < 1e-10);

        // Close = Open: BOP = 0
        let candle_doji = Ohlcv::new(0, 102.5, 105.0, 100.0, 102.5, 1000.0);
        assert!((BOP::calculate_raw_bop(&candle_doji) - 0.0).abs() < 1e-10);

        // No range (high == low): BOP = 0
        let candle_no_range = Ohlcv::new(0, 100.0, 100.0, 100.0, 100.0, 1000.0);
        assert!((BOP::calculate_raw_bop(&candle_no_range) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn bop_batch_calculation() {
        let mut bop = BOP::new(BOPConfig::new(3));

        let candles = create_candles(&[
            (100.0, 105.0, 100.0, 104.0), // BOP ≈ 0.8
            (104.0, 108.0, 102.0, 107.0), // BOP ≈ 0.5
            (107.0, 110.0, 105.0, 109.0), // BOP ≈ 0.4
            (109.0, 112.0, 107.0, 108.0), // BOP ≈ -0.2
        ]);

        bop.calc(&candles).unwrap();

        assert!(bop.state().is_ready());
        // History includes NaN warmup + computed values: 2 NaN + 2 values = 4
        assert_eq!(bop.len(), 4);
        // But only 2 computed values (non-NaN)
        let computed: Vec<_> = bop.history().iter().filter(|v| !v.is_nan()).collect();
        assert_eq!(computed.len(), 2);
    }

    #[test]
    fn bop_strong_buyers() {
        let mut bop = BOP::new(BOPConfig::new(3));

        // All closes at highs = strong buying
        let candles = create_candles(&[
            (100.0, 110.0, 100.0, 110.0), // BOP = 1.0
            (110.0, 120.0, 110.0, 120.0), // BOP = 1.0
            (120.0, 130.0, 120.0, 130.0), // BOP = 1.0
        ]);

        bop.calc(&candles).unwrap();

        let value = bop.latest().unwrap();
        assert!((value - 1.0).abs() < 1e-10, "BOP should be 1.0: {}", value);
    }

    #[test]
    fn bop_strong_sellers() {
        let mut bop = BOP::new(BOPConfig::new(3));

        // All closes at lows = strong selling
        let candles = create_candles(&[
            (110.0, 110.0, 100.0, 100.0), // BOP = -1.0
            (100.0, 100.0, 90.0, 90.0),   // BOP = -1.0
            (90.0, 90.0, 80.0, 80.0),     // BOP = -1.0
        ]);

        bop.calc(&candles).unwrap();

        let value = bop.latest().unwrap();
        assert!((value - (-1.0)).abs() < 1e-10, "BOP should be -1.0: {}", value);
    }

    #[test]
    fn bop_neutral() {
        let mut bop = BOP::new(BOPConfig::new(3));

        // All closes = opens = neutral
        let candles = create_candles(&[
            (105.0, 110.0, 100.0, 105.0), // BOP = 0.0
            (105.0, 115.0, 95.0, 105.0),  // BOP = 0.0
            (105.0, 120.0, 90.0, 105.0),  // BOP = 0.0
        ]);

        bop.calc(&candles).unwrap();

        let value = bop.latest().unwrap();
        assert!((value - 0.0).abs() < 1e-10, "BOP should be 0.0: {}", value);
    }

    #[test]
    fn bop_streaming_matches_batch() {
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
            (108.0, 112.0, 106.0, 107.0),
            (107.0, 109.0, 104.0, 105.0),
            (105.0, 108.0, 103.0, 107.0),
        ]);

        // Batch calculation
        let mut batch = BOP::new(BOPConfig::new(3));
        batch.calc(&candles).unwrap();

        // Streaming calculation
        let mut stream = BOP::new(BOPConfig::new(3));
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        // Compare computed values only (skip NaN warmup)
        let batch_computed: Vec<f64> = batch.history().iter().filter(|v| !v.is_nan()).copied().collect();
        let stream_computed: Vec<f64> = stream.history().iter().filter(|v| !v.is_nan()).copied().collect();

        assert_eq!(batch_computed.len(), stream_computed.len(), "Computed value counts should match");

        for i in 0..batch_computed.len() {
            assert!(
                (batch_computed[i] - stream_computed[i]).abs() < 1e-10,
                "Mismatch at index {}: batch={}, stream={}",
                i, batch_computed[i], stream_computed[i]
            );
        }
    }

    #[test]
    fn bop_values_in_range() {
        let mut bop = BOP::new(BOPConfig::new(3));

        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 100.0, 101.0),
            (101.0, 106.0, 99.0, 105.0),
            (105.0, 110.0, 103.0, 104.0),
            (104.0, 108.0, 102.0, 107.0),
        ]);

        bop.calc(&candles).unwrap();

        // Check computed values only (skip NaN warmup)
        for value in bop.history().iter().filter(|v| !v.is_nan()) {
            assert!(
                *value >= -1.0 && *value <= 1.0,
                "BOP should be between -1 and 1, got {}",
                value
            );
        }
    }

    #[test]
    fn bop_no_smoothing() {
        // Smoothing = 1 means no smoothing
        let mut bop = BOP::new(BOPConfig::new(1));

        let candles = create_candles(&[
            (100.0, 110.0, 100.0, 110.0), // BOP = 1.0
            (110.0, 110.0, 100.0, 100.0), // BOP = -1.0
            (100.0, 105.0, 100.0, 102.5), // BOP = 0.5
        ]);

        bop.calc(&candles).unwrap();

        assert_eq!(bop.len(), 3); // No warmup needed with smoothing=1
        assert!((bop.get(0).unwrap() - 1.0).abs() < 1e-10);
        assert!((bop.get(1).unwrap() - (-1.0)).abs() < 1e-10);
        assert!((bop.get(2).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn bop_reset() {
        let mut bop = BOP::new(BOPConfig::new(3));
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
        ]);

        bop.calc(&candles).unwrap();
        assert!(bop.state().is_ready());
        assert!(!bop.is_empty());

        bop.reset();
        assert!(bop.state().is_uninitialized());
        assert!(bop.is_empty());
        assert!(bop.latest().is_none());
    }

    #[test]
    fn bop_warmup_period() {
        let bop = BOP::new(BOPConfig::new(14));
        assert_eq!(bop.warmup_period(), 14);

        let bop_no_smooth = BOP::new(BOPConfig::new(0));
        assert_eq!(bop_no_smooth.warmup_period(), 1); // Min 1
    }

    #[test]
    fn bop_empty_data() {
        let mut bop = BOP::default();
        let candles: Vec<Ohlcv> = vec![];

        bop.calc(&candles).unwrap();
        assert!(bop.state().is_ready());
        assert!(bop.is_empty());
        assert!(bop.latest().is_none());
    }

    #[test]
    fn bop_insufficient_data() {
        let mut bop = BOP::new(BOPConfig::new(14));
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
        ]);

        bop.calc(&candles).unwrap();
        assert!(bop.state().is_ready());
        // History has NaN warmup values but no computed values
        let computed: Vec<_> = bop.history().iter().filter(|v| !v.is_nan()).collect();
        assert!(computed.is_empty(), "Should have no computed values");
    }
}
