//! Moving Average Convergence Divergence (MACD) indicator.
//!
//! MACD is a trend-following momentum indicator that shows the relationship
//! between two exponential moving averages of an asset's price.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::{EMAConfig, MACDConfig},
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

use super::EMA;

/// MACD output values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MACDOutput {
    /// MACD line (fast EMA - slow EMA)
    pub macd: f64,
    /// Signal line (EMA of MACD)
    pub signal: f64,
    /// Histogram (MACD - Signal)
    pub histogram: f64,
}

impl Default for MACDOutput {
    fn default() -> Self {
        Self {
            macd: f64::NAN,
            signal: f64::NAN,
            histogram: f64::NAN,
        }
    }
}

/// MACD indicator.
///
/// Composes two EMAs (fast and slow) and a signal line.
/// This indicator computes directly from `&[Ohlcv]` slices, making it suitable
/// for SharedWindow architecture and parallel computation with Rayon.
///
/// # Formula
///
/// - MACD Line = EMA(fast_period) - EMA(slow_period)
/// - Signal Line = EMA(MACD Line, signal_period)
/// - Histogram = MACD Line - Signal Line
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{MACD, MACDConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut macd = MACD::new(MACDConfig::new(12, 26, 9));
/// let candles: Vec<Ohlcv> = (0..50).map(|i| Ohlcv::from_close(100.0 + i as f64)).collect();
/// macd.calc(&candles).unwrap();
///
/// assert!(macd.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct MACD {
    config: MACDConfig,
    state: IndicatorState,
    fast_ema: EMA,
    slow_ema: EMA,
    /// Signal line multiplier: 2 / (signal_period + 1)
    signal_mult: f64,
    /// Current signal EMA value
    signal_value: Option<f64>,
    /// Count of MACD values seen for signal warmup
    signal_warmup: usize,
    /// Sum of MACD values during signal warmup (for initial SMA)
    signal_sum: f64,
    history: Vec<MACDOutput>,
    latest: Option<MACDOutput>,
    /// Track last snapshot length for next() to detect new candles.
    last_len: usize,
}

impl MACD {
    /// Create a new MACD indicator.
    pub fn new(config: MACDConfig) -> Self {
        let signal_mult = 2.0 / (config.signal_period as f64 + 1.0);

        Self {
            config,
            state: IndicatorState::Uninitialized,
            fast_ema: EMA::new(EMAConfig::new(config.fast_period)),
            slow_ema: EMA::new(EMAConfig::new(config.slow_period)),
            signal_mult,
            signal_value: None,
            signal_warmup: 0,
            signal_sum: 0.0,
            history: Vec::new(),
            latest: None,
            last_len: 0,
        }
    }

    /// Get the fast period.
    #[inline]
    pub fn fast_period(&self) -> usize {
        self.config.fast_period
    }

    /// Get the slow period.
    #[inline]
    pub fn slow_period(&self) -> usize {
        self.config.slow_period
    }

    /// Get the signal period.
    #[inline]
    pub fn signal_period(&self) -> usize {
        self.config.signal_period
    }

    /// Compute MACD output from fast and slow EMA values.
    /// Updates signal line state internally.
    fn compute_output(&mut self, fast: f64, slow: f64) -> MACDOutput {
        let macd = fast - slow;

        // Handle signal warmup
        if self.signal_warmup < self.config.signal_period {
            self.signal_warmup += 1;
            self.signal_sum += macd;

            if self.signal_warmup < self.config.signal_period {
                // Still warming up signal
                return MACDOutput {
                    macd,
                    signal: f64::NAN,
                    histogram: f64::NAN,
                };
            }

            // First valid signal (SMA of first signal_period MACD values)
            let signal = self.signal_sum / self.config.signal_period as f64;
            self.signal_value = Some(signal);
            return MACDOutput {
                macd,
                signal,
                histogram: macd - signal,
            };
        }

        // Apply EMA smoothing to signal
        let prev_signal = self.signal_value.unwrap_or(macd);
        let signal = macd * self.signal_mult + prev_signal * (1.0 - self.signal_mult);
        self.signal_value = Some(signal);

        MACDOutput {
            macd,
            signal,
            histogram: macd - signal,
        }
    }
}

impl Default for MACD {
    fn default() -> Self {
        Self::new(MACDConfig::default())
    }
}

impl Indicator for MACD {
    type Output = MACDOutput;
    type Config = MACDConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();
        let min_required = self.warmup_period();

        if n < min_required {
            return Err(TAError::InsufficientData {
                required: min_required,
                actual: n,
            });
        }

        // Calculate both EMAs
        self.fast_ema.calc(data)?;
        self.slow_ema.calc(data)?;

        // Collect EMA values to avoid borrow issues
        let fast_values: Vec<f64> = self.fast_ema.history().to_vec();
        let slow_values: Vec<f64> = self.slow_ema.history().to_vec();

        // Reset MACD state
        self.history = Vec::with_capacity(n);
        self.signal_value = None;
        self.signal_warmup = 0;
        self.signal_sum = 0.0;

        // MACD line becomes valid when slow EMA is valid (slow_period - 1)
        let macd_start = self.config.slow_period - 1;

        // Fill NaN until we have valid MACD
        for _ in 0..macd_start {
            self.history.push(MACDOutput::default());
        }

        // Calculate MACD values and signal
        for i in macd_start..n {
            let fast = fast_values[i];
            let slow = slow_values[i];
            let output = self.compute_output(fast, slow);
            self.history.push(output);
        }

        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();

        // MACD becomes valid when slow EMA is ready (slow_period candles)
        // Signal needs additional signal_period - 1 MACD values
        let min_for_macd = self.config.slow_period;

        if len < min_for_macd {
            // Not enough data - still warming up
            self.state = self.state.increment(min_for_macd);
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        // Get EMA values from the internal EMAs
        let fast = self.fast_ema.next(candles)?;
        let slow = self.slow_ema.next(candles)?;

        let output = if is_new_candle {
            // New candle - advance signal state
            self.compute_output(fast, slow)
        } else {
            // Same candle - just compute without advancing signal state
            let macd = fast - slow;
            if let Some(signal) = self.signal_value {
                MACDOutput {
                    macd,
                    signal,
                    histogram: macd - signal,
                }
            } else {
                MACDOutput {
                    macd,
                    signal: f64::NAN,
                    histogram: f64::NAN,
                }
            }
        };

        // Update state
        self.latest = Some(output);
        self.state = if self.signal_value.is_some() {
            IndicatorState::Ready
        } else {
            IndicatorState::Warming {
                count: self.signal_warmup,
            }
        };

        // Update history
        if is_new_candle {
            self.history.push(output);
            self.last_len = len;
        } else if !self.history.is_empty() {
            *self.history.last_mut().unwrap() = output;
        }

        Some(output)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_value = None;
        self.signal_warmup = 0;
        self.signal_sum = 0.0;
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        // Slow EMA warmup + signal warmup
        self.config.slow_period + self.config.signal_period - 1
    }
}

impl HistoricalIndicator for MACD {
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
            .map(|(i, &close)| Ohlcv::new(i as i64, close, close, close, close, 0.0))
            .collect()
    }

    fn generate_trend_data(n: usize) -> Vec<Ohlcv> {
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        candles_from_closes(&closes)
    }

    #[test]
    fn macd_batch_calculation() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        let data = generate_trend_data(20);

        macd.calc(&data).unwrap();

        assert!(macd.state().is_ready());
        assert_eq!(macd.len(), 20);

        // First 4 (slow_period - 1) should have NaN MACD
        for i in 0..4 {
            let output = macd.get(i as isize).unwrap();
            assert!(output.macd.is_nan());
        }

        // From index 4, MACD should be valid
        let output = macd.get(4).unwrap();
        assert!(!output.macd.is_nan());

        // Signal becomes valid at slow_period + signal_period - 2 = 5 + 3 - 2 = 6
        let output = macd.get(6).unwrap();
        assert!(!output.signal.is_nan());
        assert!(!output.histogram.is_nan());
    }

    #[test]
    fn macd_uptrend_positive() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        // Strong uptrend - fast EMA should be above slow EMA
        let data = generate_trend_data(20);

        macd.calc(&data).unwrap();

        let latest = macd.latest().unwrap();
        // In uptrend, MACD should be positive (fast > slow)
        assert!(
            latest.macd > 0.0,
            "MACD should be positive in uptrend, got {}",
            latest.macd
        );
    }

    #[test]
    fn macd_downtrend_negative() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        // Downtrend
        let closes: Vec<f64> = (0..20).map(|i| 200.0 - (i as f64 * 0.5)).collect();
        let data = candles_from_closes(&closes);

        macd.calc(&data).unwrap();

        let latest = macd.latest().unwrap();
        // In downtrend, MACD should be negative (fast < slow)
        assert!(
            latest.macd < 0.0,
            "MACD should be negative in downtrend, got {}",
            latest.macd
        );
    }

    #[test]
    fn macd_streaming_next() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        let candles = generate_trend_data(20);

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = macd.next(snapshot);

            // Should get result once we have enough data
            if i >= macd.warmup_period() {
                assert!(result.is_some(), "Should have result at len {}", i);
            }
        }

        assert!(macd.state().is_ready());
        let latest = macd.latest().unwrap();
        assert!(!latest.macd.is_nan());
        assert!(!latest.signal.is_nan());
    }

    #[test]
    fn macd_histogram_is_difference() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        let data = generate_trend_data(20);

        macd.calc(&data).unwrap();

        // Check histogram = macd - signal
        for output in macd.history().iter() {
            if !output.signal.is_nan() {
                let expected_hist = output.macd - output.signal;
                assert!(
                    (output.histogram - expected_hist).abs() < 1e-10,
                    "Histogram should equal MACD - Signal"
                );
            }
        }
    }

    #[test]
    fn macd_batch_vs_streaming_equivalence() {
        let candles = generate_trend_data(30);
        let config = MACDConfig::new(3, 5, 3);

        // Batch
        let mut batch = MACD::new(config);
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = MACD::new(config);
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        // Compare computed values (skip NaN)
        let batch_valid: Vec<_> = batch
            .history()
            .iter()
            .filter(|o| !o.macd.is_nan())
            .collect();
        let stream_valid: Vec<_> = stream
            .history()
            .iter()
            .filter(|o| !o.macd.is_nan())
            .collect();

        assert_eq!(batch_valid.len(), stream_valid.len());

        for (b, s) in batch_valid.iter().zip(stream_valid.iter()) {
            assert!(
                (b.macd - s.macd).abs() < 1e-10,
                "MACD mismatch: batch={}, stream={}",
                b.macd,
                s.macd
            );
        }
    }

    #[test]
    fn macd_reset() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        let data = generate_trend_data(20);

        macd.calc(&data).unwrap();
        assert!(macd.state().is_ready());

        macd.reset();
        assert!(macd.state().is_uninitialized());
        assert_eq!(macd.len(), 0);
        assert!(macd.latest().is_none());
    }

    #[test]
    fn macd_insufficient_data() {
        let mut macd = MACD::new(MACDConfig::new(12, 26, 9));
        let candles = generate_trend_data(30);

        // warmup = 26 + 9 - 1 = 34
        let result = macd.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn macd_parallel_safe() {
        use std::sync::Arc;

        let candles = Arc::new(generate_trend_data(50));

        let mut macd1 = MACD::new(MACDConfig::new(3, 5, 3));
        let mut macd2 = MACD::new(MACDConfig::new(3, 5, 3));

        // Both can read from the same snapshot
        let snapshot: &[Ohlcv] = &candles;
        let r1 = macd1.next(snapshot);
        let r2 = macd2.next(snapshot);

        assert!(r1.is_some());
        assert!(r2.is_some());
        assert!((r1.unwrap().macd - r2.unwrap().macd).abs() < 1e-10);
    }
}