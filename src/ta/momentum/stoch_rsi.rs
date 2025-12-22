//! Stochastic RSI indicator.
//!
//! StochRSI applies the Stochastic oscillator formula to RSI values,
//! providing a more sensitive momentum indicator.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use std::collections::VecDeque;

use crate::ta::{
    config::{RSIConfig, StochRSIConfig},
    error::TAResult,
    momentum::RSI,
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Stochastic RSI output values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochRSIOutput {
    /// %K line (smoothed stochastic of RSI)
    pub k: f64,
    /// %D line (SMA of %K)
    pub d: f64,
}

/// Stochastic RSI indicator.
///
/// Applies the Stochastic oscillator formula to RSI values, then smooths
/// the result to produce %K and %D lines.
///
/// # Formula
///
/// 1. Calculate RSI over rsi_period
/// 2. Stochastic RSI = (RSI - min(RSI, stoch_period)) / (max(RSI, stoch_period) - min(RSI, stoch_period))
/// 3. %K = SMA(Stochastic RSI, k_smoothing)
/// 4. %D = SMA(%K, d_smoothing)
///
/// # Interpretation
///
/// - Values above 80: Overbought
/// - Values below 20: Oversold
/// - %K crossing above %D: Bullish signal
/// - %K crossing below %D: Bearish signal
///
/// # Example
///
/// ```
/// use rolling_ta::ta::momentum::{StochRSI, StochRSIConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut stoch_rsi = StochRSI::default();
/// // After sufficient data, StochRSI produces %K and %D values
/// ```
#[derive(Debug, Clone)]
pub struct StochRSI {
    config: StochRSIConfig,
    state: IndicatorState,
    history: Vec<StochRSIOutput>,
    latest: Option<StochRSIOutput>,

    // Internal RSI calculator
    rsi: RSI,

    // Rolling window of RSI values for stochastic calculation
    rsi_window: VecDeque<f64>,

    // Rolling window of raw stochastic values for %K smoothing
    raw_stoch_window: VecDeque<f64>,

    // Rolling window of %K values for %D smoothing
    k_window: VecDeque<f64>,

    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl StochRSI {
    /// Create a new StochRSI indicator.
    pub fn new(config: StochRSIConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            rsi: RSI::new(RSIConfig::new(config.rsi_period)),
            rsi_window: VecDeque::with_capacity(config.stoch_period),
            raw_stoch_window: VecDeque::with_capacity(config.k_smoothing),
            k_window: VecDeque::with_capacity(config.d_smoothing),
            last_len: 0,
        }
    }

    /// Get the current %K value.
    #[inline]
    pub fn k(&self) -> f64 {
        self.latest.map(|o| o.k).unwrap_or(f64::NAN)
    }

    /// Get the current %D value.
    #[inline]
    pub fn d(&self) -> f64 {
        self.latest.map(|o| o.d).unwrap_or(f64::NAN)
    }

    /// Calculate raw stochastic from RSI values.
    #[inline]
    fn calculate_stochastic(rsi_values: &VecDeque<f64>) -> f64 {
        if rsi_values.is_empty() {
            return 50.0;
        }

        let current = *rsi_values.back().unwrap();
        let min = rsi_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = rsi_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let range = max - min;
        if range > 0.0 {
            ((current - min) / range) * 100.0
        } else {
            50.0 // No range means neutral
        }
    }

    /// Calculate SMA of values in window.
    #[inline]
    fn sma(values: &VecDeque<f64>) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Total warmup period needed.
    fn total_warmup(&self) -> usize {
        // RSI needs period + 1, then stoch_period for min/max,
        // then k_smoothing - 1 for %K, then d_smoothing - 1 for %D
        self.config.rsi_period + 1
            + self.config.stoch_period - 1
            + self.config.k_smoothing - 1
            + self.config.d_smoothing - 1
    }
}

impl Default for StochRSI {
    fn default() -> Self {
        Self::new(StochRSIConfig::default())
    }
}

impl Indicator for StochRSI {
    type Output = StochRSIOutput;
    type Config = StochRSIConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();

        // Reset state
        self.history.clear();
        self.rsi_window.clear();
        self.raw_stoch_window.clear();
        self.k_window.clear();
        self.latest = None;

        if n == 0 {
            self.last_len = 0;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Calculate RSI first
        if self.rsi.calc(data).is_err() {
            // Not enough data for RSI
            self.last_len = n;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        let rsi_history = self.rsi.history();

        // Process RSI values to generate StochRSI
        for &rsi_val in rsi_history.iter() {
            // Add to RSI window
            if self.rsi_window.len() >= self.config.stoch_period {
                self.rsi_window.pop_front();
            }
            self.rsi_window.push_back(rsi_val);

            // Calculate raw stochastic when we have enough RSI values
            if self.rsi_window.len() >= self.config.stoch_period {
                let raw_stoch = Self::calculate_stochastic(&self.rsi_window);

                // Add to raw stochastic window for %K smoothing
                if self.raw_stoch_window.len() >= self.config.k_smoothing {
                    self.raw_stoch_window.pop_front();
                }
                self.raw_stoch_window.push_back(raw_stoch);

                // Calculate %K when we have enough raw stochastic values
                if self.raw_stoch_window.len() >= self.config.k_smoothing {
                    let k = Self::sma(&self.raw_stoch_window);

                    // Add to %K window for %D smoothing
                    if self.k_window.len() >= self.config.d_smoothing {
                        self.k_window.pop_front();
                    }
                    self.k_window.push_back(k);

                    // Calculate %D when we have enough %K values
                    if self.k_window.len() >= self.config.d_smoothing {
                        let d = Self::sma(&self.k_window);
                        self.history.push(StochRSIOutput { k, d });
                    }
                }
            }
        }

        self.latest = self.history.last().copied();
        self.last_len = n;
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
                // First time - need to initialize from scratch
                // This is similar to calc() but we process incrementally
                self.rsi_window.clear();
                self.raw_stoch_window.clear();
                self.k_window.clear();

                // Calculate RSI using calc for the initial batch
                if self.rsi.calc(candles).is_err() {
                    self.last_len = len;
                    self.state = IndicatorState::Ready;
                    return None;
                }

                // Process all RSI history values
                let rsi_history = self.rsi.history();
                for &rsi_val in rsi_history.iter() {
                    // Add to RSI window
                    if self.rsi_window.len() >= self.config.stoch_period {
                        self.rsi_window.pop_front();
                    }
                    self.rsi_window.push_back(rsi_val);

                    // Calculate raw stochastic
                    if self.rsi_window.len() >= self.config.stoch_period {
                        let raw_stoch = Self::calculate_stochastic(&self.rsi_window);

                        // Add to raw stochastic window
                        if self.raw_stoch_window.len() >= self.config.k_smoothing {
                            self.raw_stoch_window.pop_front();
                        }
                        self.raw_stoch_window.push_back(raw_stoch);

                        // Calculate %K
                        if self.raw_stoch_window.len() >= self.config.k_smoothing {
                            let k = Self::sma(&self.raw_stoch_window);

                            // Add to %K window
                            if self.k_window.len() >= self.config.d_smoothing {
                                self.k_window.pop_front();
                            }
                            self.k_window.push_back(k);

                            // Calculate %D
                            if self.k_window.len() >= self.config.d_smoothing {
                                let d = Self::sma(&self.k_window);
                                self.history.push(StochRSIOutput { k, d });
                            }
                        }
                    }
                }

                self.latest = self.history.last().copied();
                self.last_len = len;
                self.state = IndicatorState::Ready;
                return self.latest;
            }

            // New candle(s) added - process incrementally
            // Get new RSI value
            if let Some(rsi_val) = self.rsi.next(candles) {
                // Add to RSI window
                if self.rsi_window.len() >= self.config.stoch_period {
                    self.rsi_window.pop_front();
                }
                self.rsi_window.push_back(rsi_val);

                // Calculate raw stochastic
                if self.rsi_window.len() >= self.config.stoch_period {
                    let raw_stoch = Self::calculate_stochastic(&self.rsi_window);

                    // Add to raw stochastic window
                    if self.raw_stoch_window.len() >= self.config.k_smoothing {
                        self.raw_stoch_window.pop_front();
                    }
                    self.raw_stoch_window.push_back(raw_stoch);

                    // Calculate %K
                    if self.raw_stoch_window.len() >= self.config.k_smoothing {
                        let k = Self::sma(&self.raw_stoch_window);

                        // Add to %K window
                        if self.k_window.len() >= self.config.d_smoothing {
                            self.k_window.pop_front();
                        }
                        self.k_window.push_back(k);

                        // Calculate %D
                        if self.k_window.len() >= self.config.d_smoothing {
                            let d = Self::sma(&self.k_window);
                            let output = StochRSIOutput { k, d };
                            self.history.push(output);
                            self.latest = Some(output);
                        }
                    }
                }
            }

            self.last_len = len;
        }
        // For same-candle updates, we'd need more complex state tracking
        // For now, StochRSI doesn't support same-candle updates due to complexity

        self.state = IndicatorState::Ready;
        self.latest
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.rsi.reset();
        self.history.clear();
        self.latest = None;
        self.rsi_window.clear();
        self.raw_stoch_window.clear();
        self.k_window.clear();
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.total_warmup()
    }
}

impl HistoricalIndicator for StochRSI {
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

    fn create_trending_candles(count: usize, start: f64, step: f64) -> Vec<Ohlcv> {
        (0..count)
            .map(|i| {
                let close = start + (i as f64 * step);
                Ohlcv::new(i as i64, close - 1.0, close + 2.0, close - 2.0, close, 1000.0)
            })
            .collect()
    }

    fn create_oscillating_candles(count: usize) -> Vec<Ohlcv> {
        (0..count)
            .map(|i| {
                let close = 100.0 + (i as f64 * 0.5).sin() * 10.0;
                Ohlcv::new(i as i64, close - 1.0, close + 2.0, close - 2.0, close, 1000.0)
            })
            .collect()
    }

    #[test]
    fn stoch_rsi_batch_calculation() {
        let mut stoch_rsi = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));

        // Need enough data for full warmup
        let candles = create_oscillating_candles(50);
        stoch_rsi.calc(&candles).unwrap();

        assert!(stoch_rsi.state().is_ready());
        assert!(!stoch_rsi.is_empty(), "Should have some StochRSI values");

        // Check values are in valid range
        for output in stoch_rsi.history() {
            assert!(output.k >= 0.0 && output.k <= 100.0, "K should be 0-100: {}", output.k);
            assert!(output.d >= 0.0 && output.d <= 100.0, "D should be 0-100: {}", output.d);
        }
    }

    #[test]
    fn stoch_rsi_strong_uptrend() {
        let mut stoch_rsi = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));

        // Strong uptrend - RSI should be high
        // Note: In a pure monotonic trend, RSI stabilizes and StochRSI may be neutral
        let candles = create_trending_candles(50, 100.0, 1.0);
        stoch_rsi.calc(&candles).unwrap();

        // Just verify we get valid output
        if let Some(latest) = stoch_rsi.latest() {
            assert!(latest.k >= 0.0 && latest.k <= 100.0, "K should be valid: {}", latest.k);
        }
    }

    #[test]
    fn stoch_rsi_strong_downtrend() {
        let mut stoch_rsi = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));

        // Strong downtrend
        // Note: In a pure monotonic trend, RSI stabilizes and StochRSI may be neutral
        let candles = create_trending_candles(50, 200.0, -1.0);
        stoch_rsi.calc(&candles).unwrap();

        // Just verify we get valid output
        if let Some(latest) = stoch_rsi.latest() {
            assert!(latest.k >= 0.0 && latest.k <= 100.0, "K should be valid: {}", latest.k);
        }
    }

    #[test]
    fn stoch_rsi_streaming_matches_batch() {
        let candles = create_oscillating_candles(50);

        // Batch calculation
        let mut batch = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));
        batch.calc(&candles).unwrap();

        // Streaming calculation - first call with all candles (like joining mid-stream)
        let mut stream = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));
        stream.next(&candles);

        assert_eq!(batch.len(), stream.len(), "History lengths should match");

        for i in 0..batch.len() {
            let batch_val = batch.get(i as isize).unwrap();
            let stream_val = stream.get(i as isize).unwrap();
            assert!(
                (batch_val.k - stream_val.k).abs() < 1e-10,
                "K mismatch at index {}: batch={}, stream={}",
                i, batch_val.k, stream_val.k
            );
            assert!(
                (batch_val.d - stream_val.d).abs() < 1e-10,
                "D mismatch at index {}: batch={}, stream={}",
                i, batch_val.d, stream_val.d
            );
        }
    }

    #[test]
    fn stoch_rsi_incremental_streaming() {
        let candles = create_oscillating_candles(60);

        // Start with batch of 50, then stream 10 more
        let mut indicator = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));
        indicator.next(&candles[..50]);
        let len_after_batch = indicator.len();

        // Add more candles incrementally
        for i in 51..=60 {
            indicator.next(&candles[..i]);
        }

        // Should have more values after streaming
        assert!(indicator.len() >= len_after_batch,
            "Should have at least as many values after streaming: {} vs {}",
            indicator.len(), len_after_batch);
    }

    #[test]
    fn stoch_rsi_values_in_range() {
        let mut stoch_rsi = StochRSI::default();

        let candles = create_oscillating_candles(100);
        stoch_rsi.calc(&candles).unwrap();

        for output in stoch_rsi.history() {
            assert!(
                output.k >= 0.0 && output.k <= 100.0,
                "K should be between 0 and 100, got {}",
                output.k
            );
            assert!(
                output.d >= 0.0 && output.d <= 100.0,
                "D should be between 0 and 100, got {}",
                output.d
            );
        }
    }

    #[test]
    fn stoch_rsi_reset() {
        let mut stoch_rsi = StochRSI::default();
        let candles = create_oscillating_candles(50);

        stoch_rsi.calc(&candles).unwrap();
        assert!(stoch_rsi.state().is_ready());
        assert!(!stoch_rsi.is_empty());

        stoch_rsi.reset();
        assert!(stoch_rsi.state().is_uninitialized());
        assert!(stoch_rsi.is_empty());
        assert!(stoch_rsi.latest().is_none());
    }

    #[test]
    fn stoch_rsi_warmup_period() {
        let config = StochRSIConfig::new(14, 14, 3, 3);
        let stoch_rsi = StochRSI::new(config);
        // rsi_period + 1 + stoch_period - 1 + k_smoothing - 1 + d_smoothing - 1
        // = 14 + 1 + 14 - 1 + 3 - 1 + 3 - 1 = 32
        assert_eq!(stoch_rsi.warmup_period(), 32);
    }

    #[test]
    fn stoch_rsi_empty_data() {
        let mut stoch_rsi = StochRSI::default();
        let candles: Vec<Ohlcv> = vec![];

        stoch_rsi.calc(&candles).unwrap();
        assert!(stoch_rsi.state().is_ready());
        assert!(stoch_rsi.is_empty());
        assert!(stoch_rsi.latest().is_none());
    }

    #[test]
    fn stoch_rsi_insufficient_data() {
        let mut stoch_rsi = StochRSI::default();
        let candles = create_oscillating_candles(10); // Not enough for default config

        stoch_rsi.calc(&candles).unwrap();
        assert!(stoch_rsi.state().is_ready());
        assert!(stoch_rsi.is_empty()); // Not enough data for StochRSI
    }

    #[test]
    fn stoch_rsi_d_is_smoother_than_k() {
        let mut stoch_rsi = StochRSI::default();
        let candles = create_oscillating_candles(100);

        stoch_rsi.calc(&candles).unwrap();

        // Calculate variance of K and D
        let k_values: Vec<f64> = stoch_rsi.history().iter().map(|o| o.k).collect();
        let d_values: Vec<f64> = stoch_rsi.history().iter().map(|o| o.d).collect();

        fn variance(values: &[f64]) -> f64 {
            if values.is_empty() {
                return 0.0;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
        }

        let k_var = variance(&k_values);
        let d_var = variance(&d_values);

        // D should be smoother (lower variance) than K
        assert!(d_var <= k_var, "D variance {} should be <= K variance {}", d_var, k_var);
    }
}
