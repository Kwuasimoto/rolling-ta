//! Rate of Change (ROC) indicator.
//!
//! ROC measures the percentage change in price between the current price
//! and the price n periods ago.

use std::collections::VecDeque;

use crate::ta::{
    config::ROCConfig,
    error::TAResult,
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    HistoricalIndicator, Indicator,
};

/// Rate of Change (ROC) momentum indicator.
///
/// Measures the percentage change in price over a specified period.
///
/// # Formula
///
/// ```text
/// ROC = ((Close - Close[n periods ago]) / Close[n periods ago]) × 100
/// ```
///
/// # Interpretation
///
/// - Positive ROC: Price is higher than n periods ago (bullish)
/// - Negative ROC: Price is lower than n periods ago (bearish)
/// - Zero line crossovers can signal trend changes
///
/// # Example
///
/// ```
/// use rolling_ta::ta::momentum::{ROC, ROCConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut roc = ROC::new(ROCConfig::new(12));
/// let data = OhlcvSeries::from_closes(&[
///     100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0,
///     110.0, 109.0, 111.0, 113.0, 115.0, 114.0, 116.0,
/// ]);
/// roc.calc(&data).unwrap();
///
/// assert!(roc.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct ROC {
    config: ROCConfig,
    state: IndicatorState,
    /// Rolling window of closes for streaming updates.
    window: VecDeque<f64>,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl ROC {
    /// Create a new ROC indicator with the given configuration.
    pub fn new(config: ROCConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            window: VecDeque::with_capacity(config.period + 1),
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate ROC value.
    #[inline]
    fn calculate_roc(current: f64, previous: f64) -> f64 {
        if previous == 0.0 {
            f64::NAN
        } else {
            ((current - previous) / previous) * 100.0
        }
    }
}

impl Default for ROC {
    fn default() -> Self {
        Self::new(ROCConfig::default())
    }
}

impl Indicator for ROC {
    type Output = f64;
    type Config = ROCConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        // Reset state
        self.history = vec![f64::NAN; n];
        self.window.clear();
        let closes = &data.closes;

        // Calculate ROC for each point where we have enough history
        for i in 0..n {
            if i >= period {
                let roc = Self::calculate_roc(closes[i], closes[i - period]);
                self.history[i] = roc;
            }
        }

        // Populate window for streaming (last `period` closes)
        let start = if n > period { n - period } else { 0 };
        for i in start..n {
            self.window.push_back(closes[i]);
        }

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let close = tick.close.0;
        let period = self.config.period;

        // Add current close to window
        self.window.push_back(close);

        // Check if we have enough data
        if self.window.len() <= period {
            // Still warming up
            self.history.push(f64::NAN);
            if self.state.is_uninitialized() {
                self.state = IndicatorState::Warming {
                    count: self.window.len(),
                };
            } else if let IndicatorState::Warming { count } = self.state {
                self.state = IndicatorState::Warming { count: count + 1 };
            }
            return Ok(None);
        }

        // Remove oldest value to maintain window size
        let old_close = self.window.pop_front().unwrap();

        // Calculate ROC
        let roc = Self::calculate_roc(close, old_close);
        self.history.push(roc);
        self.latest = Some(roc);
        self.state = IndicatorState::Ready;

        Ok(Some(roc))
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.window.clear();
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period + 1 // Need period + 1 points for first ROC
    }
}

impl HistoricalIndicator for ROC {
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
    fn roc_batch_calculation() {
        let mut roc = ROC::new(ROCConfig::new(3));
        let data = OhlcvSeries::from_closes(&[100.0, 102.0, 104.0, 106.0, 108.0, 110.0]);

        roc.calc(&data).unwrap();

        assert!(roc.state().is_ready());
        assert_eq!(roc.len(), 6);

        // First 3 values should be NaN (period = 3)
        assert!(roc.get(0).unwrap().is_nan());
        assert!(roc.get(1).unwrap().is_nan());
        assert!(roc.get(2).unwrap().is_nan());

        // ROC at index 3: (106 - 100) / 100 * 100 = 6%
        let roc3 = roc.get(3).unwrap();
        assert!((roc3 - 6.0).abs() < 0.0001);

        // ROC at index 4: (108 - 102) / 102 * 100 = 5.88%
        let roc4 = roc.get(4).unwrap();
        assert!((roc4 - 5.88235).abs() < 0.001);

        // ROC at index 5: (110 - 104) / 104 * 100 = 5.77%
        let roc5 = roc.get(5).unwrap();
        assert!((roc5 - 5.76923).abs() < 0.001);
    }

    #[test]
    fn roc_streaming_update() {
        let mut roc = ROC::new(ROCConfig::new(3));

        // Warmup - first 3 ticks
        assert!(roc.update(&Ohlcv::from_close(100.0)).unwrap().is_none());
        assert!(roc.update(&Ohlcv::from_close(102.0)).unwrap().is_none());
        assert!(roc.update(&Ohlcv::from_close(104.0)).unwrap().is_none());

        // Fourth tick - should get first ROC
        let result = roc.update(&Ohlcv::from_close(106.0)).unwrap();
        assert!(result.is_some());
        let roc_val = result.unwrap();
        // (106 - 100) / 100 * 100 = 6%
        assert!((roc_val - 6.0).abs() < 0.0001);
    }

    #[test]
    fn roc_streaming_matches_batch() {
        let closes = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 108.0, 106.0];
        let data = OhlcvSeries::from_closes(&closes);

        // Batch
        let mut batch_roc = ROC::new(ROCConfig::new(3));
        batch_roc.calc(&data).unwrap();

        // Streaming
        let mut stream_roc = ROC::new(ROCConfig::new(3));
        for close in &closes {
            stream_roc.update(&Ohlcv::from_close(*close)).unwrap();
        }

        assert_eq!(batch_roc.len(), stream_roc.len());
        for i in 0..batch_roc.len() {
            let batch_val = batch_roc.get(i as isize).unwrap();
            let stream_val = stream_roc.get(i as isize).unwrap();

            if batch_val.is_nan() && stream_val.is_nan() {
                continue;
            }
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
    fn roc_uptrend() {
        let mut roc = ROC::new(ROCConfig::new(5));
        // Strong uptrend: 100 -> 120 (20% increase)
        let closes: Vec<f64> = (0..10).map(|i| 100.0 + (i as f64 * 2.0)).collect();
        let data = OhlcvSeries::from_closes(&closes);

        roc.calc(&data).unwrap();

        let latest = roc.latest().unwrap();
        // Should be positive in uptrend
        assert!(latest > 0.0, "ROC in uptrend should be positive, got {}", latest);
    }

    #[test]
    fn roc_downtrend() {
        let mut roc = ROC::new(ROCConfig::new(5));
        // Strong downtrend: 120 -> 100 (16.7% decrease)
        let closes: Vec<f64> = (0..10).map(|i| 120.0 - (i as f64 * 2.0)).collect();
        let data = OhlcvSeries::from_closes(&closes);

        roc.calc(&data).unwrap();

        let latest = roc.latest().unwrap();
        // Should be negative in downtrend
        assert!(latest < 0.0, "ROC in downtrend should be negative, got {}", latest);
    }

    #[test]
    fn roc_flat_market() {
        let mut roc = ROC::new(ROCConfig::new(5));
        // Flat market
        let closes = vec![100.0; 10];
        let data = OhlcvSeries::from_closes(&closes);

        roc.calc(&data).unwrap();

        let latest = roc.latest().unwrap();
        // Should be zero in flat market
        assert!(
            latest.abs() < 0.0001,
            "ROC in flat market should be ~0, got {}",
            latest
        );
    }

    #[test]
    fn roc_reset() {
        let mut roc = ROC::new(ROCConfig::new(3));
        let data = OhlcvSeries::from_closes(&[100.0, 102.0, 104.0, 106.0]);

        roc.calc(&data).unwrap();
        assert!(roc.state().is_ready());
        assert_eq!(roc.len(), 4);

        roc.reset();
        assert!(roc.state().is_uninitialized());
        assert_eq!(roc.len(), 0);
        assert!(roc.latest().is_none());
    }
}
