//! Stochastic RSI indicator.
//!
//! StochRSI applies the Stochastic oscillator formula to RSI values,
//! providing a more sensitive momentum indicator.
//!
//! This indicator uses backward smoothing to match Python rolling-ta behavior:
//! 1. Calculate RSI
//! 2. Calculate raw stochastic over stoch_period
//! 3. Apply k_smoothing backward through the array
//! 4. Apply d_smoothing backward through the array

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
    /// %K line (smoothed stochastic of RSI), range 0-1
    pub k: f64,
    /// %D line (SMA of %K), range 0-1
    pub d: f64,
}

/// Stochastic RSI indicator.
///
/// Applies the Stochastic oscillator formula to RSI values, then smooths
/// the result using backward smoothing to produce %K and %D lines.
///
/// # Formula (Python-compatible backward smoothing)
///
/// 1. Calculate RSI over rsi_period
/// 2. Raw Stoch = (RSI - min(RSI, stoch_period)) / (max(RSI, stoch_period) - min(RSI, stoch_period))
/// 3. %K = Backward SMA of Raw Stoch over k_smoothing
/// 4. %D = Backward SMA of %K over d_smoothing
///
/// # Interpretation
///
/// - Values above 0.8: Overbought
/// - Values below 0.2: Oversold
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
/// // After sufficient data, StochRSI produces %K and %D values (0-1 range)
/// ```
#[derive(Debug, Clone)]
pub struct StochRSI {
    config: StochRSIConfig,
    state: IndicatorState,
    history: Vec<StochRSIOutput>,
    latest: Option<StochRSIOutput>,

    // Internal RSI calculator
    rsi: RSI,

    // For streaming mode
    rsi_window: VecDeque<f64>,
    raw_stoch_window: VecDeque<f64>,
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

    /// Total warmup period needed.
    fn total_warmup(&self) -> usize {
        // RSI needs rsi_period, then stoch_period for min/max,
        // then k_smoothing for %K, then d_smoothing for %D
        self.config.rsi_period + self.config.stoch_period + self.config.k_smoothing + self.config.d_smoothing - 2
    }

    /// Calculate raw stochastic (0-1 range) for RSI values.
    /// Uses a window of stoch_period RSI values, returns (current - min) / (max - min)
    fn calc_raw_stochastic(rsi_slice: &[f64]) -> f64 {
        if rsi_slice.is_empty() {
            return 0.0;
        }
        let current = rsi_slice[rsi_slice.len() - 1];
        let min = rsi_slice.iter().copied().fold(f64::INFINITY, f64::min);
        let max = rsi_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range > 0.0 {
            (current - min) / range
        } else {
            0.0
        }
    }

    /// Apply backward smoothing to an array.
    /// For index i, calculates SMA of values [i-smoothing+1..=i] and stores at [i].
    /// Modifies indices from first_idx to n-1 (matching Python's behavior).
    fn apply_backward_smoothing(values: &mut [f64], first_idx: usize, smoothing: usize) {
        let n = values.len();
        if first_idx >= n {
            return;
        }
        // Work backwards from end to first_idx (matching Python's loop direction)
        for i in (first_idx..n).rev() {
            if i + 1 >= smoothing {
                let start = i + 1 - smoothing;
                let sum: f64 = values[start..=i].iter().sum();
                values[i] = sum / smoothing as f64;
            }
        }
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
        let rsi_period = self.config.rsi_period;
        let stoch_period = self.config.stoch_period;
        let k_smoothing = self.config.k_smoothing;
        let d_smoothing = self.config.d_smoothing;

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

        // Step 1: Calculate RSI
        self.rsi.calc(data)?;
        let rsi_history = self.rsi.history();

        if rsi_history.is_empty() {
            self.last_len = n;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Create arrays for the full data length (matching Python behavior)
        let mut stoch_k = vec![0.0_f64; n];
        let mut stoch_d = vec![0.0_f64; n];

        // Step 2: Calculate raw stochastic values
        // RSI history is indexed same as data: rsi_history[i] = RSI at data index i
        // First valid RSI is at index rsi_period (indices 0..rsi_period are NaN)
        // Raw stochastic needs stoch_period RSI values
        // First raw stoch at data index: rsi_period + stoch_period - 1
        let raw_stoch_start = rsi_period + stoch_period - 1;

        for i in raw_stoch_start..n {
            // RSI window for stochastic: [i - stoch_period + 1 ..= i]
            let rsi_start_idx = i + 1 - stoch_period;
            let rsi_end_idx = i;

            // Only compute if we have valid RSI values (not in warmup region)
            if rsi_start_idx >= rsi_period && rsi_end_idx < rsi_history.len() {
                let rsi_slice = &rsi_history[rsi_start_idx..=rsi_end_idx];
                stoch_k[i] = Self::calc_raw_stochastic(rsi_slice);
            }
        }

        // Step 3: Apply backward smoothing for %K
        // Python: for i in range(size, rsi_period + k_period + k_smoothing, -1)
        // Loop goes from size down to threshold+1 (exclusive), storing at i-1
        // For threshold=27: i goes [200, 199, ..., 28], stores at [199, 198, ..., 27]
        // So first_idx = rsi_period + stoch_period + k_smoothing = 27
        let k_smooth_first = rsi_period + stoch_period + k_smoothing;
        Self::apply_backward_smoothing(&mut stoch_k, k_smooth_first, k_smoothing);

        // Step 4: Copy to stoch_d and apply backward smoothing for %D
        // Python: for i in range(size, rsi_period + k_period + d_smoothing, -1)
        stoch_d.copy_from_slice(&stoch_k);
        let d_smooth_first = rsi_period + stoch_period + d_smoothing;
        Self::apply_backward_smoothing(&mut stoch_d, d_smooth_first, d_smoothing);

        // Build history with StochRSIOutput
        // Output starts at warmup index where both K and D are valid
        for i in 0..n {
            self.history.push(StochRSIOutput {
                k: stoch_k[i],
                d: stoch_d[i],
            });
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

        // For streaming, use calc() approach since backward smoothing
        // requires all values to be present
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            // Recalculate everything using calc() approach
            let _ = self.calc(candles);
        }

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
        let mut stoch_rsi = StochRSI::new(StochRSIConfig::new(14, 10, 3, 3));

        // Need enough data for full warmup
        let candles = create_oscillating_candles(50);
        stoch_rsi.calc(&candles).unwrap();

        assert!(stoch_rsi.state().is_ready());
        assert!(!stoch_rsi.is_empty(), "Should have StochRSI values");

        // Check that output values are in valid 0-1 range (after warmup)
        let warmup = stoch_rsi.warmup_period();
        for (i, output) in stoch_rsi.history().iter().enumerate() {
            if i >= warmup {
                assert!(output.k >= 0.0 && output.k <= 1.0, "K should be 0-1 at {}: {}", i, output.k);
                assert!(output.d >= 0.0 && output.d <= 1.0, "D should be 0-1 at {}: {}", i, output.d);
            }
        }
    }

    #[test]
    fn stoch_rsi_values_in_range() {
        let mut stoch_rsi = StochRSI::new(StochRSIConfig::new(14, 10, 3, 3));
        let candles = create_oscillating_candles(100);
        stoch_rsi.calc(&candles).unwrap();

        let warmup = stoch_rsi.warmup_period();
        for (i, output) in stoch_rsi.history().iter().enumerate() {
            if i >= warmup {
                assert!(
                    output.k >= 0.0 && output.k <= 1.0,
                    "K should be between 0 and 1, got {} at index {}",
                    output.k, i
                );
                assert!(
                    output.d >= 0.0 && output.d <= 1.0,
                    "D should be between 0 and 1, got {} at index {}",
                    output.d, i
                );
            }
        }
    }

    #[test]
    fn stoch_rsi_streaming_matches_batch() {
        let candles = create_oscillating_candles(50);

        // Batch calculation
        let mut batch = StochRSI::new(StochRSIConfig::new(14, 10, 3, 3));
        batch.calc(&candles).unwrap();

        // Streaming calculation
        let mut stream = StochRSI::new(StochRSIConfig::new(14, 10, 3, 3));
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
        let config = StochRSIConfig::new(14, 10, 3, 3);
        let stoch_rsi = StochRSI::new(config);
        // rsi_period + stoch_period + k_smoothing + d_smoothing - 2
        // = 14 + 10 + 3 + 3 - 2 = 28
        assert_eq!(stoch_rsi.warmup_period(), 28);
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
}
