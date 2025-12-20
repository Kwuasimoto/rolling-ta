//! Volume Weighted Average Price (VWAP) indicator.
//!
//! VWAP is the ratio of cumulative (price × volume) to cumulative volume,
//! typically reset at time boundaries (e.g., daily).

use crate::ta::{
    config::VWAPConfig,
    error::TAResult,
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
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
/// # Example
///
/// ```
/// use rolling_ta::ta::volume::{VWAP, VWAPConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut vwap = VWAP::new(VWAPConfig::new(86400)); // Daily reset
/// let data = OhlcvSeries::from_tuples(&[
///     (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     (1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     (2, 106.0, 107.0, 102.0, 104.0, 900.0),
/// ]);
/// vwap.calc(&data).unwrap();
///
/// assert!(vwap.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct VWAP {
    config: VWAPConfig,
    state: IndicatorState,
    /// Cumulative sum of (typical_price × volume).
    raw_accum: f64,
    /// Cumulative sum of volume.
    vol_accum: f64,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl VWAP {
    /// Create a new VWAP indicator with the given configuration.
    pub fn new(config: VWAPConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            raw_accum: 0.0,
            vol_accum: 0.0,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the current raw accumulator value (sum of price × volume).
    #[inline]
    pub fn raw_accum(&self) -> f64 {
        self.raw_accum
    }

    /// Get the current volume accumulator value.
    #[inline]
    pub fn vol_accum(&self) -> f64 {
        self.vol_accum
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

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let n = data.len();
        if n == 0 {
            self.history.clear();
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Reset state
        self.history = Vec::with_capacity(n);
        self.raw_accum = 0.0;
        self.vol_accum = 0.0;

        let timestamps = &data.timestamps;
        let highs = &data.highs;
        let lows = &data.lows;
        let closes = &data.closes;
        let volumes = &data.volumes;

        for i in 0..n {
            let timestamp = timestamps[i];

            // Reset accumulators at time boundary
            if self.should_reset(timestamp) {
                self.raw_accum = 0.0;
                self.vol_accum = 0.0;
            }

            let typical = Self::typical_price(highs[i], lows[i], closes[i]);
            self.raw_accum += typical * volumes[i];
            self.vol_accum += volumes[i];

            let vwap = if self.vol_accum > 0.0 {
                self.raw_accum / self.vol_accum
            } else {
                f64::NAN
            };

            self.history.push(vwap);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let timestamp = tick.timestamp.0;

        // Reset accumulators at time boundary
        if self.should_reset(timestamp) {
            self.raw_accum = 0.0;
            self.vol_accum = 0.0;
        }

        let typical = Self::typical_price(tick.high.0, tick.low.0, tick.close.0);
        self.raw_accum += typical * tick.volume.0;
        self.vol_accum += tick.volume.0;

        let vwap = if self.vol_accum > 0.0 {
            self.raw_accum / self.vol_accum
        } else {
            f64::NAN
        };

        self.history.push(vwap);
        self.latest = Some(vwap);

        if self.state.is_uninitialized() {
            self.state = IndicatorState::Ready;
        }

        Ok(Some(vwap))
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.raw_accum = 0.0;
        self.vol_accum = 0.0;
        self.history.clear();
        self.latest = None;
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
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 98.0, 100.0, 1000.0), // typical = 101.0
            (1, 100.0, 108.0, 99.0, 102.0, 1100.0), // typical = 103.0
            (2, 102.0, 107.0, 100.0, 104.0, 900.0), // typical = 103.67
        ]);

        vwap.calc(&data).unwrap();

        assert!(vwap.state().is_ready());
        assert_eq!(vwap.len(), 3);

        // First candle: typical = (105 + 98 + 100) / 3 = 101.0
        // VWAP = 101.0 * 1000 / 1000 = 101.0
        let v0 = vwap.get(0).unwrap();
        assert!((v0 - 101.0).abs() < 0.01);
    }

    #[test]
    fn vwap_streaming_update() {
        let mut vwap = VWAP::new(VWAPConfig::new(0)); // No reset

        // First tick: typical = (105 + 98 + 100) / 3 = 101.0
        let result = vwap
            .update(&Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0))
            .unwrap();
        assert!((result.unwrap() - 101.0).abs() < 0.01);

        // Second tick: typical = (108 + 99 + 102) / 3 = 103.0
        // cumulative: (101*1000 + 103*1100) / 2100 = (101000 + 113300) / 2100 = 102.05
        let result = vwap
            .update(&Ohlcv::new(1, 100.0, 108.0, 99.0, 102.0, 1100.0))
            .unwrap();
        assert!((result.unwrap() - 102.047619).abs() < 0.01);
    }

    #[test]
    fn vwap_reset_at_boundary() {
        // Test that VWAP resets at time boundaries
        let mut vwap = VWAP::new(VWAPConfig::new(100)); // Reset every 100 units

        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 95.0, 100.0, 1000.0),   // typical=100, reset at 0
            (50, 100.0, 110.0, 90.0, 100.0, 1000.0),  // typical=100, no reset
            (100, 100.0, 115.0, 85.0, 100.0, 1000.0), // typical=100, reset at 100
        ]);

        vwap.calc(&data).unwrap();

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

        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 95.0, 100.0, 1000.0),
            (100, 100.0, 110.0, 90.0, 100.0, 1000.0),
            (200, 100.0, 115.0, 85.0, 100.0, 1000.0),
        ]);

        vwap.calc(&data).unwrap();

        // All three candles should contribute to VWAP
        // All have typical = 100.0, volume = 1000
        // VWAP = (100*1000 + 100*1000 + 100*1000) / 3000 = 100.0
        let v2 = vwap.get(2).unwrap();
        assert!((v2 - 100.0).abs() < 0.01);
    }

    #[test]
    fn vwap_reset() {
        let mut vwap = VWAP::default();
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            (1, 100.0, 108.0, 99.0, 105.0, 1100.0),
        ]);

        vwap.calc(&data).unwrap();
        assert!(vwap.state().is_ready());
        assert_eq!(vwap.len(), 2);

        vwap.reset();
        assert!(vwap.state().is_uninitialized());
        assert_eq!(vwap.len(), 0);
        assert_eq!(vwap.raw_accum(), 0.0);
        assert_eq!(vwap.vol_accum(), 0.0);
    }
}
