//! Bollinger Bands indicator.
//!
//! Bollinger Bands consist of a middle band (SMA) and two outer bands
//! based on standard deviation, used to identify volatility.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::BBConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Bollinger Bands output values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBOutput {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

impl Default for BBOutput {
    fn default() -> Self {
        Self {
            upper: f64::NAN,
            middle: f64::NAN,
            lower: f64::NAN,
        }
    }
}

/// Bollinger Bands indicator.
///
/// Uses SMA for the middle band and standard deviation for the outer bands.
///
/// # Formula
///
/// - Middle Band = SMA(close, period)
/// - Upper Band = Middle Band + (std_dev * multiplier)
/// - Lower Band = Middle Band - (std_dev * multiplier)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volatility::{BB, BBConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut bb = BB::new(BBConfig::new(20, 2.0));
/// let candles: Vec<Ohlcv> = (0..25).map(|i| {
///     Ohlcv::from_close(100.0 + (i as f64 * 0.5).sin() * 5.0)
/// }).collect();
/// bb.calc(&candles).unwrap();
///
/// assert!(bb.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct BB {
    config: BBConfig,
    state: IndicatorState,
    history: Vec<BBOutput>,
    latest: Option<BBOutput>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl BB {
    /// Create a new Bollinger Bands indicator.
    pub fn new(config: BBConfig) -> Self {
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

    /// Get the standard deviation multiplier.
    #[inline]
    pub fn std_dev_mult(&self) -> f64 {
        self.config.std_dev_mult
    }

    /// Calculate SMA and population standard deviation in a single pass.
    /// Returns (mean, std_dev)
    #[inline]
    fn mean_and_std_dev(closes: &[f64]) -> (f64, f64) {
        let n = closes.len() as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }

        let sum: f64 = closes.iter().sum();
        let mean = sum / n;

        let variance: f64 = closes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, variance.sqrt())
    }

    /// Compute BB output from slice of closes.
    #[inline]
    fn compute_from_closes(closes: &[f64], mult: f64) -> BBOutput {
        let (mean, std_dev) = Self::mean_and_std_dev(closes);
        let weighted_std = std_dev * mult;

        BBOutput {
            upper: mean + weighted_std,
            middle: mean,
            lower: mean - weighted_std,
        }
    }
}

impl Default for BB {
    fn default() -> Self {
        Self::new(BBConfig::default())
    }
}

impl Indicator for BB {
    type Output = BBOutput;
    type Config = BBConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        if period == 0 {
            return Err(TAError::InvalidPeriod(0));
        }

        // Reset state
        self.history = Vec::with_capacity(n);
        let mult = self.config.std_dev_mult;

        // Fill with NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(BBOutput::default());
        }

        // Calculate BB from period-1 onwards
        for i in (period - 1)..n {
            // Extract closes for the window
            let window_closes: Vec<f64> = data[(i + 1 - period)..=i]
                .iter()
                .map(|c| c.close.0)
                .collect();

            let output = Self::compute_from_closes(&window_closes, mult);
            self.history.push(output);
        }

        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < period {
            // Not enough data yet
            if len > self.last_len || self.last_len == 0 {
                self.history.push(BBOutput::default());
                self.last_len = len;
                self.state = IndicatorState::Warming { count: len };
            }
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        // Extract last `period` closes
        let window_closes: Vec<f64> = candles[(len - period)..]
            .iter()
            .map(|c| c.close.0)
            .collect();

        let output = Self::compute_from_closes(&window_closes, self.config.std_dev_mult);

        if is_new_candle {
            self.history.push(output);
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Same candle - update last value
            *self.history.last_mut().unwrap() = output;
        }

        self.latest = Some(output);
        self.state = IndicatorState::Ready;

        Some(output)
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
        self.config.period
    }
}

impl HistoricalIndicator for BB {
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
            .map(|(i, &close)| Ohlcv::new(i as i64, close, close + 2.0, close - 1.0, close, 1000.0))
            .collect()
    }

    #[test]
    fn bb_batch_calculation() {
        let mut bb = BB::new(BBConfig::new(5, 2.0));
        let candles = candles_from_closes(&[
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
        ]);

        bb.calc(&candles).unwrap();

        assert!(bb.state().is_ready());
        assert_eq!(bb.len(), 10);

        // First 4 should be NaN
        for i in 0..4 {
            let output = bb.get(i as isize).unwrap();
            assert!(output.upper.is_nan());
            assert!(output.middle.is_nan());
            assert!(output.lower.is_nan());
        }

        // From index 4, should have valid values
        let output = bb.get(4).unwrap();
        assert!(!output.upper.is_nan());
        assert!(!output.middle.is_nan());
        assert!(!output.lower.is_nan());

        // Upper should be > middle > lower
        assert!(output.upper > output.middle);
        assert!(output.middle > output.lower);
    }

    #[test]
    fn bb_streaming_next() {
        let mut bb = BB::new(BBConfig::new(3, 2.0));
        let candles = candles_from_closes(&[100.0, 102.0, 104.0, 103.0, 105.0]);

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = bb.next(snapshot);

            // Should get result when we have period = 3 candles
            if i >= 3 {
                assert!(result.is_some(), "Should have result at len {}", i);
                let output = result.unwrap();
                assert!(output.upper > output.middle);
                assert!(output.middle > output.lower);
            }
        }

        assert!(bb.state().is_ready());
    }

    #[test]
    fn bb_batch_vs_streaming_equivalence() {
        let closes: Vec<f64> = (0..20)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let candles = candles_from_closes(&closes);
        let config = BBConfig::new(5, 2.0);

        // Batch
        let mut batch = BB::new(config);
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = BB::new(config);
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        // Compare valid values
        let batch_valid: Vec<_> = batch
            .history()
            .iter()
            .filter(|o| !o.middle.is_nan())
            .collect();
        let stream_valid: Vec<_> = stream
            .history()
            .iter()
            .filter(|o| !o.middle.is_nan())
            .collect();

        assert_eq!(
            batch_valid.len(),
            stream_valid.len(),
            "batch={}, stream={}",
            batch_valid.len(),
            stream_valid.len()
        );

        for (i, (b, s)) in batch_valid.iter().zip(stream_valid.iter()).enumerate() {
            assert!(
                (b.upper - s.upper).abs() < 1e-10,
                "Upper mismatch at {}: batch={}, stream={}",
                i,
                b.upper,
                s.upper
            );
            assert!(
                (b.middle - s.middle).abs() < 1e-10,
                "Middle mismatch at {}: batch={}, stream={}",
                i,
                b.middle,
                s.middle
            );
            assert!(
                (b.lower - s.lower).abs() < 1e-10,
                "Lower mismatch at {}: batch={}, stream={}",
                i,
                b.lower,
                s.lower
            );
        }
    }

    #[test]
    fn bb_bands_widen_with_volatility() {
        let mut bb_stable = BB::new(BBConfig::new(5, 2.0));
        let mut bb_volatile = BB::new(BBConfig::new(5, 2.0));

        // Stable prices
        let stable = candles_from_closes(&[100.0, 100.0, 100.0, 100.0, 100.0]);

        // Volatile prices (same mean)
        let volatile = candles_from_closes(&[80.0, 120.0, 80.0, 120.0, 100.0]);

        bb_stable.calc(&stable).unwrap();
        bb_volatile.calc(&volatile).unwrap();

        let stable_output = bb_stable.latest().unwrap();
        let volatile_output = bb_volatile.latest().unwrap();

        // Stable should have tight bands (std_dev ≈ 0)
        let stable_width = stable_output.upper - stable_output.lower;
        let volatile_width = volatile_output.upper - volatile_output.lower;

        assert!(
            volatile_width > stable_width,
            "Volatile bands ({}) should be wider than stable bands ({})",
            volatile_width,
            stable_width
        );
    }

    #[test]
    fn bb_reset() {
        let mut bb = BB::new(BBConfig::new(5, 2.0));
        let candles = candles_from_closes(&[100.0, 102.0, 101.0, 103.0, 105.0]);

        bb.calc(&candles).unwrap();
        assert!(bb.state().is_ready());
        assert_eq!(bb.len(), 5);

        bb.reset();
        assert!(bb.state().is_uninitialized());
        assert_eq!(bb.len(), 0);
    }

    #[test]
    fn bb_mean_and_std_dev() {
        // Test: std dev of [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5, variance = 4, std_dev = 2
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std_dev) = BB::mean_and_std_dev(&data);
        assert!((mean - 5.0).abs() < 1e-10);
        assert!((std_dev - 2.0).abs() < 1e-10);
    }

    #[test]
    fn bb_insufficient_data() {
        let mut bb = BB::new(BBConfig::new(20, 2.0));
        let candles = candles_from_closes(&[100.0; 10]);

        let result = bb.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }
}
