//! Volume Weighted Average Price (VWAP) indicator.
//!
//! VWAP is the ratio of cumulative (price × volume) to cumulative volume,
//! typically reset at time boundaries (e.g., daily).
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::VWAPConfig,
    error::TAResult,
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Volume Weighted Average Price indicator.
///
/// VWAP represents the average price weighted by volume, commonly used
/// as a trading benchmark. It resets at configurable time intervals
/// (daily by default).
///
/// # Formula
///
/// ```text
/// typical_price = (high + low + close) / 3
/// VWAP = Σ(typical_price × volume) / Σ(volume)
/// ```
///
/// # Interpretation
///
/// - Price above VWAP suggests bullish sentiment
/// - Price below VWAP suggests bearish sentiment
/// - VWAP is commonly used as a trading benchmark
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volume::{VWAP, VWAPConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut vwap = VWAP::new(VWAPConfig::new(0)); // No reset
/// let candles = vec![
///     Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     Ohlcv::new(2, 106.0, 107.0, 102.0, 104.0, 900.0),
/// ];
/// vwap.calc(&candles).unwrap();
///
/// assert!(vwap.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct VWAP {
    config: VWAPConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Committed cumulative sum of (typical_price × volume) - before current candle
    committed_raw_accum: f64,
    /// Committed cumulative sum of volume - before current candle
    committed_vol_accum: f64,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
    /// Last committed timestamp for reset detection
    last_committed_timestamp: i64,
}

impl VWAP {
    /// Create a new VWAP indicator with the given configuration.
    pub fn new(config: VWAPConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            committed_raw_accum: 0.0,
            committed_vol_accum: 0.0,
            last_len: 0,
            last_committed_timestamp: 0,
        }
    }

    /// Get the current raw accumulator value (sum of price × volume).
    #[inline]
    pub fn raw_accum(&self) -> f64 {
        self.committed_raw_accum
    }

    /// Get the current volume accumulator value.
    #[inline]
    pub fn vol_accum(&self) -> f64 {
        self.committed_vol_accum
    }

    /// Check if timestamp triggers a reset based on the reset interval.
    #[inline]
    fn should_reset(&self, timestamp: i64) -> bool {
        self.config.reset_interval > 0 && timestamp % self.config.reset_interval == 0
    }

    /// Calculate typical price: (high + low + close) / 3
    #[inline]
    fn typical_price(high: f64, low: f64, close: f64) -> f64 {
        (high + low + close) / 3.0
    }

    /// Calculate VWAP from accumulators
    #[inline]
    fn calculate_vwap(raw_accum: f64, vol_accum: f64) -> f64 {
        if vol_accum > 0.0 {
            raw_accum / vol_accum
        } else {
            f64::NAN
        }
    }
}

impl Default for VWAP {
    fn default() -> Self {
        Self::new(VWAPConfig::default())
    }
}

impl Indicator for VWAP {
    type Output = f64;
    type Config = VWAPConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();

        // Reset state
        self.history = Vec::with_capacity(n);
        self.committed_raw_accum = 0.0;
        self.committed_vol_accum = 0.0;

        if n == 0 {
            self.latest = None;
            self.last_len = 0;
            self.last_committed_timestamp = 0;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        let mut raw_accum = 0.0;
        let mut vol_accum = 0.0;

        for candle in data {
            let timestamp = candle.timestamp.0;

            // Reset accumulators at time boundary
            if self.should_reset(timestamp) {
                raw_accum = 0.0;
                vol_accum = 0.0;
            }

            let typical = Self::typical_price(candle.high.0, candle.low.0, candle.close.0);
            raw_accum += typical * candle.volume.0;
            vol_accum += candle.volume.0;

            let vwap = Self::calculate_vwap(raw_accum, vol_accum);
            self.history.push(vwap);
        }

        self.committed_raw_accum = raw_accum;
        self.committed_vol_accum = vol_accum;
        self.latest = self.history.last().copied();
        self.last_len = n;
        self.last_committed_timestamp = data[n - 1].timestamp.0;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();

        if len == 0 {
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First candle(s) - process all from scratch
                let mut raw_accum = 0.0;
                let mut vol_accum = 0.0;

                for (i, candle) in candles.iter().enumerate() {
                    let timestamp = candle.timestamp.0;

                    // Reset at time boundary
                    if self.should_reset(timestamp) {
                        raw_accum = 0.0;
                        vol_accum = 0.0;
                    }

                    let typical = Self::typical_price(candle.high.0, candle.low.0, candle.close.0);
                    raw_accum += typical * candle.volume.0;
                    vol_accum += candle.volume.0;

                    let vwap = Self::calculate_vwap(raw_accum, vol_accum);
                    self.history.push(vwap);

                    // Commit all but the last candle
                    if i < len - 1 {
                        self.committed_raw_accum = raw_accum;
                        self.committed_vol_accum = vol_accum;
                        self.last_committed_timestamp = timestamp;
                    }
                }

                self.last_len = len;
                self.latest = self.history.last().copied();
                self.state = IndicatorState::Ready;
                return self.latest;
            }

            // New candle(s) added - first commit the previous tentative candle
            if !self.history.is_empty() && self.last_len >= 1 {
                let prev_candle = &candles[self.last_len - 1];
                let prev_timestamp = prev_candle.timestamp.0;

                // Check if previous candle triggered a reset
                if self.should_reset(prev_timestamp) {
                    self.committed_raw_accum = 0.0;
                    self.committed_vol_accum = 0.0;
                }

                let typical = Self::typical_price(
                    prev_candle.high.0,
                    prev_candle.low.0,
                    prev_candle.close.0,
                );
                self.committed_raw_accum += typical * prev_candle.volume.0;
                self.committed_vol_accum += prev_candle.volume.0;
                self.last_committed_timestamp = prev_timestamp;
            }

            // Process all new candles since last_len
            for i in self.last_len..len {
                let candle = &candles[i];
                let timestamp = candle.timestamp.0;

                // Start with committed state
                let mut raw_accum = self.committed_raw_accum;
                let mut vol_accum = self.committed_vol_accum;

                // Check if this candle triggers a reset
                if self.should_reset(timestamp) {
                    raw_accum = 0.0;
                    vol_accum = 0.0;
                }

                let typical = Self::typical_price(candle.high.0, candle.low.0, candle.close.0);
                raw_accum += typical * candle.volume.0;
                vol_accum += candle.volume.0;

                let vwap = Self::calculate_vwap(raw_accum, vol_accum);
                self.history.push(vwap);

                // Commit if not the last candle
                if i < len - 1 {
                    self.committed_raw_accum = raw_accum;
                    self.committed_vol_accum = vol_accum;
                    self.last_committed_timestamp = timestamp;
                }
            }

            self.last_len = len;
            self.latest = self.history.last().copied();
        } else {
            // Same candle - compute tentatively without committing
            let current_candle = &candles[len - 1];
            let timestamp = current_candle.timestamp.0;

            // Start with committed state
            let mut raw_accum = self.committed_raw_accum;
            let mut vol_accum = self.committed_vol_accum;

            // Check if this candle triggers a reset
            if self.should_reset(timestamp) {
                raw_accum = 0.0;
                vol_accum = 0.0;
            }

            let typical = Self::typical_price(
                current_candle.high.0,
                current_candle.low.0,
                current_candle.close.0,
            );
            raw_accum += typical * current_candle.volume.0;
            vol_accum += current_candle.volume.0;

            let vwap = Self::calculate_vwap(raw_accum, vol_accum);

            // Update last history entry
            if !self.history.is_empty() {
                *self.history.last_mut().unwrap() = vwap;
            }

            self.latest = Some(vwap);
        }

        self.state = IndicatorState::Ready;
        self.latest
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.committed_raw_accum = 0.0;
        self.committed_vol_accum = 0.0;
        self.last_len = 0;
        self.last_committed_timestamp = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        1 // VWAP is available immediately
    }
}

impl HistoricalIndicator for VWAP {
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

    #[test]
    fn vwap_batch_calculation() {
        let mut vwap = VWAP::new(VWAPConfig::new(0)); // No reset
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0), // typical = 101.0
            Ohlcv::new(1, 100.0, 108.0, 99.0, 102.0, 1100.0), // typical = 103.0
            Ohlcv::new(2, 102.0, 107.0, 100.0, 104.0, 900.0), // typical = 103.67
        ];

        vwap.calc(&candles).unwrap();

        assert!(vwap.state().is_ready());
        assert_eq!(vwap.len(), 3);

        // First candle: typical = (105 + 98 + 100) / 3 = 101.0
        // VWAP = 101.0 * 1000 / 1000 = 101.0
        let v0 = vwap.get(0).unwrap();
        assert!((v0 - 101.0).abs() < 0.01);
    }

    #[test]
    fn vwap_streaming_next() {
        let mut vwap = VWAP::new(VWAPConfig::new(0)); // No reset
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            Ohlcv::new(1, 100.0, 108.0, 99.0, 102.0, 1100.0),
            Ohlcv::new(2, 102.0, 107.0, 100.0, 104.0, 900.0),
        ];

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = vwap.next(snapshot);
            assert!(result.is_some(), "Should have result at len {}", i);
        }

        assert!(vwap.state().is_ready());
        assert_eq!(vwap.len(), 3);

        // First candle: typical = 101.0, VWAP = 101.0
        let v0 = vwap.get(0).unwrap();
        assert!((v0 - 101.0).abs() < 0.01);
    }

    #[test]
    fn vwap_batch_vs_streaming_equivalence() {
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            Ohlcv::new(1, 100.0, 108.0, 99.0, 102.0, 1100.0),
            Ohlcv::new(2, 102.0, 107.0, 100.0, 104.0, 900.0),
            Ohlcv::new(3, 104.0, 110.0, 102.0, 108.0, 1200.0),
            Ohlcv::new(4, 108.0, 112.0, 106.0, 110.0, 800.0),
        ];

        // Batch
        let mut batch = VWAP::new(VWAPConfig::new(0));
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = VWAP::new(VWAPConfig::new(0));
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let batch_val = batch.get(i as isize).unwrap();
            let stream_val = stream.get(i as isize).unwrap();
            assert!(
                (batch_val - stream_val).abs() < 1e-10,
                "Mismatch at index {}: batch={}, stream={}",
                i,
                batch_val,
                stream_val
            );
        }
    }

    #[test]
    fn vwap_reset_at_boundary() {
        // Test that VWAP resets at time boundaries
        let mut vwap = VWAP::new(VWAPConfig::new(100)); // Reset every 100 units

        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 95.0, 100.0, 1000.0),   // typical=100, reset at 0
            Ohlcv::new(50, 100.0, 110.0, 90.0, 100.0, 1000.0),  // typical=100, no reset
            Ohlcv::new(100, 100.0, 115.0, 85.0, 100.0, 1000.0), // typical=100, reset at 100
        ];

        vwap.calc(&candles).unwrap();

        // At timestamp 100, accumulators should have reset
        // So VWAP at index 2 should be just that candle's typical price
        let v2 = vwap.get(2).unwrap();
        let expected_typical = (115.0 + 85.0 + 100.0) / 3.0; // 100.0
        assert!((v2 - expected_typical).abs() < 0.01);
    }

    #[test]
    fn vwap_no_reset_when_interval_zero() {
        // With reset_interval = 0, should never reset
        let mut vwap = VWAP::new(VWAPConfig::new(0));

        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 95.0, 100.0, 1000.0),
            Ohlcv::new(100, 100.0, 110.0, 90.0, 100.0, 1000.0),
            Ohlcv::new(200, 100.0, 115.0, 85.0, 100.0, 1000.0),
        ];

        vwap.calc(&candles).unwrap();

        // All three candles should contribute to VWAP
        // All have typical = 100.0, volume = 1000
        // VWAP = (100*1000 + 100*1000 + 100*1000) / 3000 = 100.0
        let v2 = vwap.get(2).unwrap();
        assert!((v2 - 100.0).abs() < 0.01);
    }

    #[test]
    fn vwap_same_candle_update() {
        let mut vwap = VWAP::new(VWAPConfig::new(0));

        // First candle: typical = (105 + 98 + 100) / 3 = 101.0
        let candles1 = vec![Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0)];
        vwap.next(&candles1);
        assert!((vwap.latest().unwrap() - 101.0).abs() < 0.01);

        // Second candle: typical = (108 + 99 + 102) / 3 = 103.0
        // VWAP = (101*1000 + 103*1100) / 2100 = 102.047619
        let mut candles2 = candles1.clone();
        candles2.push(Ohlcv::new(1, 100.0, 108.0, 99.0, 102.0, 1100.0));
        vwap.next(&candles2);
        assert!((vwap.latest().unwrap() - 102.047619).abs() < 0.01);

        // Same candle - price changes: typical = (110 + 95 + 105) / 3 = 103.33
        // VWAP = (101*1000 + 103.33*1100) / 2100 = 102.17
        candles2[1] = Ohlcv::new(1, 100.0, 110.0, 95.0, 105.0, 1100.0);
        vwap.next(&candles2);
        let expected = (101.0 * 1000.0 + 103.333333 * 1100.0) / 2100.0;
        assert!(
            (vwap.latest().unwrap() - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            vwap.latest().unwrap()
        );
    }

    #[test]
    fn vwap_reset() {
        let mut vwap = VWAP::default();
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0),
        ];

        vwap.calc(&candles).unwrap();
        assert!(vwap.state().is_ready());
        assert_eq!(vwap.len(), 2);

        vwap.reset();
        assert!(vwap.state().is_uninitialized());
        assert_eq!(vwap.len(), 0);
        assert_eq!(vwap.raw_accum(), 0.0);
        assert_eq!(vwap.vol_accum(), 0.0);
    }

    #[test]
    fn vwap_empty_data() {
        let mut vwap = VWAP::default();
        let candles: Vec<Ohlcv> = vec![];

        vwap.calc(&candles).unwrap();
        assert!(vwap.state().is_ready());
        assert_eq!(vwap.len(), 0);
        assert!(vwap.latest().is_none());
    }

    #[test]
    fn vwap_streaming_with_reset() {
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 95.0, 100.0, 1000.0),   // reset at 0
            Ohlcv::new(50, 100.0, 110.0, 90.0, 100.0, 1000.0),  // no reset
            Ohlcv::new(100, 100.0, 115.0, 85.0, 100.0, 1000.0), // reset at 100
        ];

        // Batch
        let mut batch = VWAP::new(VWAPConfig::new(100));
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = VWAP::new(VWAPConfig::new(100));
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let batch_val = batch.get(i as isize).unwrap();
            let stream_val = stream.get(i as isize).unwrap();
            assert!(
                (batch_val - stream_val).abs() < 1e-10,
                "Mismatch at index {}: batch={}, stream={}",
                i,
                batch_val,
                stream_val
            );
        }
    }
}
