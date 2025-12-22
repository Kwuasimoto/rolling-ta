//! Rate of Change (ROC) indicator.
//!
//! ROC measures the percentage change in price between the current price
//! and the price n periods ago.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::ROCConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
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
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut roc = ROC::new(ROCConfig::new(12));
/// let candles: Vec<Ohlcv> = (0..15).map(|i| Ohlcv::from_close(100.0 + i as f64)).collect();
/// roc.calc(&candles).unwrap();
///
/// assert!(roc.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct ROC {
    config: ROCConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl ROC {
    /// Create a new ROC indicator with the given configuration.
    pub fn new(config: ROCConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            last_len: 0,
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

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period + 1 {
            return Err(TAError::InsufficientData {
                required: period + 1,
                actual: n,
            });
        }

        if period == 0 {
            return Err(TAError::InvalidPeriod(0));
        }

        // Reset state
        self.history = vec![f64::NAN; n];

        // Calculate ROC for each point where we have enough history
        for i in period..n {
            let current = data[i].close.0;
            let previous = data[i - period].close.0;
            self.history[i] = Self::calculate_roc(current, previous);
        }

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < period + 1 {
            // Not enough data yet
            if len > self.last_len || self.last_len == 0 {
                self.history.push(f64::NAN);
                self.last_len = len;
                self.state = IndicatorState::Warming { count: len };
            }
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        // Calculate ROC from current and period-ago closes
        let current = candles[len - 1].close.0;
        let previous = candles[len - 1 - period].close.0;
        let roc = Self::calculate_roc(current, previous);

        if is_new_candle {
            self.history.push(roc);
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Same candle - update last value
            *self.history.last_mut().unwrap() = roc;
        }

        self.latest = Some(roc);
        self.state = IndicatorState::Ready;

        Some(roc)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
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

    fn candles_from_closes(closes: &[f64]) -> Vec<Ohlcv> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Ohlcv::new(i as i64, close, close, close, close, 1000.0))
            .collect()
    }

    #[test]
    fn roc_batch_calculation() {
        let mut roc = ROC::new(ROCConfig::new(3));
        let candles = candles_from_closes(&[100.0, 102.0, 104.0, 106.0, 108.0, 110.0]);

        roc.calc(&candles).unwrap();

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
    fn roc_streaming_next() {
        let mut roc = ROC::new(ROCConfig::new(3));
        let candles = candles_from_closes(&[100.0, 102.0, 104.0, 106.0, 108.0, 110.0]);

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = roc.next(snapshot);

            // Should get result when we have period + 1 = 4 candles
            if i >= 4 {
                assert!(result.is_some(), "Should have result at len {}", i);
            }
        }

        assert!(roc.state().is_ready());

        // Verify final values match expected
        // ROC at index 3: (106 - 100) / 100 * 100 = 6%
        let roc3 = roc.get(3).unwrap();
        assert!((roc3 - 6.0).abs() < 0.0001);
    }

    #[test]
    fn roc_batch_vs_streaming_equivalence() {
        let closes = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 108.0, 106.0];
        let candles = candles_from_closes(&closes);
        let config = ROCConfig::new(3);

        // Batch
        let mut batch = ROC::new(config);
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = ROC::new(config);
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let batch_val = batch.get(i as isize).unwrap();
            let stream_val = stream.get(i as isize).unwrap();

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
        let candles = candles_from_closes(&closes);

        roc.calc(&candles).unwrap();

        let latest = roc.latest().unwrap();
        // Should be positive in uptrend
        assert!(latest > 0.0, "ROC in uptrend should be positive, got {}", latest);
    }

    #[test]
    fn roc_downtrend() {
        let mut roc = ROC::new(ROCConfig::new(5));
        // Strong downtrend: 120 -> 100 (16.7% decrease)
        let closes: Vec<f64> = (0..10).map(|i| 120.0 - (i as f64 * 2.0)).collect();
        let candles = candles_from_closes(&closes);

        roc.calc(&candles).unwrap();

        let latest = roc.latest().unwrap();
        // Should be negative in downtrend
        assert!(latest < 0.0, "ROC in downtrend should be negative, got {}", latest);
    }

    #[test]
    fn roc_flat_market() {
        let mut roc = ROC::new(ROCConfig::new(5));
        // Flat market
        let closes = vec![100.0; 10];
        let candles = candles_from_closes(&closes);

        roc.calc(&candles).unwrap();

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
        let candles = candles_from_closes(&[100.0, 102.0, 104.0, 106.0]);

        roc.calc(&candles).unwrap();
        assert!(roc.state().is_ready());
        assert_eq!(roc.len(), 4);

        roc.reset();
        assert!(roc.state().is_uninitialized());
        assert_eq!(roc.len(), 0);
        assert!(roc.latest().is_none());
    }

    #[test]
    fn roc_insufficient_data() {
        let mut roc = ROC::new(ROCConfig::new(12));
        let candles = candles_from_closes(&[100.0; 10]);

        let result = roc.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }
}
