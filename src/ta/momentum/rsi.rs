//! Relative Strength Index (RSI) indicator.
//!
//! The RSI is a momentum oscillator that measures the speed and change of price
//! movements. It oscillates between 0 and 100.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::RSIConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Relative Strength Index indicator.
///
/// Uses Wilder's smoothing method (exponential smoothing with alpha = 1/period).
///
/// # Formula
///
/// RS = Average Gain / Average Loss
/// RSI = 100 - (100 / (1 + RS)) = 100 * AvgGain / (AvgGain + AvgLoss)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::momentum::{RSI, RSIConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut rsi = RSI::new(RSIConfig::new(14));
/// let candles: Vec<Ohlcv> = (0..20).map(|i| Ohlcv::from_close(100.0 + i as f64)).collect();
/// rsi.calc(&candles).unwrap();
///
/// assert!(rsi.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct RSI {
    config: RSIConfig,
    state: IndicatorState,
    /// Committed average gain (Wilder smoothed)
    avg_gain: f64,
    /// Committed average loss (Wilder smoothed)
    avg_loss: f64,
    /// Previous close for delta calculation
    prev_close: f64,
    /// Sum of gains during warmup (before first RSI)
    warmup_gain_sum: f64,
    /// Sum of losses during warmup
    warmup_loss_sum: f64,
    /// Number of deltas seen during warmup
    warmup_count: usize,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl RSI {
    /// Create a new RSI indicator with the given configuration.
    pub fn new(config: RSIConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_close: 0.0,
            warmup_gain_sum: 0.0,
            warmup_loss_sum: 0.0,
            warmup_count: 0,
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

    /// Calculate RSI from avg_gain and avg_loss.
    #[inline]
    fn calculate_rsi(avg_gain: f64, avg_loss: f64) -> f64 {
        if avg_gain + avg_loss == 0.0 {
            50.0 // Neutral when no movement
        } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        }
    }
}

impl Default for RSI {
    fn default() -> Self {
        Self::new(RSIConfig::default())
    }
}

impl Indicator for RSI {
    type Output = f64;
    type Config = RSIConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        // Need period + 1 points (period deltas)
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

        // Calculate gains and losses
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let delta = data[i].close.0 - data[i - 1].close.0;
            if delta > 0.0 {
                gains[i] = delta;
            } else if delta < 0.0 {
                losses[i] = -delta;
            }
        }

        // Initial averages (SMA over first period)
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        self.history[period] = Self::calculate_rsi(avg_gain, avg_loss);

        // Wilder's smoothing for subsequent values
        let p_1 = (period - 1) as f64;
        for i in (period + 1)..n {
            avg_gain = (avg_gain * p_1 + gains[i]) / period as f64;
            avg_loss = (avg_loss * p_1 + losses[i]) / period as f64;
            self.history[i] = Self::calculate_rsi(avg_gain, avg_loss);
        }

        self.avg_gain = avg_gain;
        self.avg_loss = avg_loss;
        self.prev_close = data.last().unwrap().close.0;
        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len == 0 {
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        let current_close = candles.last().unwrap().close.0;

        // Handle warmup phase
        // warmup_count tracks number of deltas accumulated (not candles)
        // We need `period` deltas, which requires period + 1 candles
        if self.warmup_count < period {
            if is_new_candle {
                // Check if this is the very first candle we've seen
                if self.last_len == 0 {
                    // First candle - just store close, need second for first delta
                    self.prev_close = current_close;
                    self.last_len = len;
                    self.history.push(f64::NAN);
                    self.state = IndicatorState::Warming { count: 1 };
                    return None;
                }

                // Compute delta from previous close
                let delta = current_close - self.prev_close;
                let gain = delta.max(0.0);
                let loss = (-delta).max(0.0);

                self.warmup_gain_sum += gain;
                self.warmup_loss_sum += loss;
                self.warmup_count += 1;
                self.prev_close = current_close;
                self.last_len = len;

                if self.warmup_count >= period {
                    // First valid RSI - use SMA of accumulated gains/losses
                    self.avg_gain = self.warmup_gain_sum / period as f64;
                    self.avg_loss = self.warmup_loss_sum / period as f64;
                    let rsi = Self::calculate_rsi(self.avg_gain, self.avg_loss);
                    self.history.push(rsi);
                    self.latest = Some(rsi);
                    self.state = IndicatorState::Ready;
                    return Some(rsi);
                }

                self.history.push(f64::NAN);
                self.state = IndicatorState::Warming {
                    count: self.warmup_count + 1, // +1 for display (candles seen)
                };
                return None;
            } else {
                // Same snapshot during warmup - no state change
                return None;
            }
        }

        // Ready state - apply Wilder smoothing
        let delta = current_close - self.prev_close;
        let gain = delta.max(0.0);
        let loss = (-delta).max(0.0);

        let p_1 = (period - 1) as f64;
        let p = period as f64;

        if is_new_candle {
            // Commit new state
            self.avg_gain = (self.avg_gain * p_1 + gain) / p;
            self.avg_loss = (self.avg_loss * p_1 + loss) / p;
            self.prev_close = current_close;
            self.last_len = len;

            let rsi = Self::calculate_rsi(self.avg_gain, self.avg_loss);
            self.history.push(rsi);
            self.latest = Some(rsi);
            Some(rsi)
        } else {
            // Same candle - compute tentatively without committing
            let tentative_gain = (self.avg_gain * p_1 + gain) / p;
            let tentative_loss = (self.avg_loss * p_1 + loss) / p;
            let rsi = Self::calculate_rsi(tentative_gain, tentative_loss);

            // Update history for current candle
            if !self.history.is_empty() {
                *self.history.last_mut().unwrap() = rsi;
            }
            self.latest = Some(rsi);
            Some(rsi)
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.prev_close = 0.0;
        self.warmup_gain_sum = 0.0;
        self.warmup_loss_sum = 0.0;
        self.warmup_count = 0;
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period + 1 // Need period deltas
    }
}

impl HistoricalIndicator for RSI {
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

    #[test]
    fn rsi_batch_calculation() {
        let mut rsi = RSI::new(RSIConfig::new(14));
        // Simple up/down pattern
        let closes: Vec<f64> = (0..30).map(|i| 100.0 + (i % 3) as f64).collect();
        let candles = candles_from_closes(&closes);

        rsi.calc(&candles).unwrap();

        assert!(rsi.state().is_ready());
        assert_eq!(rsi.len(), 30);

        // First 14 values should be NaN
        for i in 0..14 {
            assert!(rsi.get(i as isize).unwrap().is_nan());
        }

        // RSI should be valid from index 14 onwards
        let value = rsi.get(14).unwrap();
        assert!(!value.is_nan());
        assert!(value >= 0.0 && value <= 100.0);
    }

    #[test]
    fn rsi_streaming_next() {
        let mut rsi = RSI::new(RSIConfig::new(3));
        let closes = [100.0, 101.0, 102.0, 101.0, 102.0, 103.0];
        let candles = candles_from_closes(&closes);

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = rsi.next(snapshot);

            // Should get result when we have period + 1 = 4 candles
            if i >= 4 {
                assert!(result.is_some(), "Should have result at len {}", i);
                let rsi_val = result.unwrap();
                assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
            }
        }

        assert!(rsi.state().is_ready());
    }

    #[test]
    fn rsi_always_in_range() {
        let mut rsi = RSI::new(RSIConfig::new(5));
        let closes: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0).collect();
        let candles = candles_from_closes(&closes);

        rsi.calc(&candles).unwrap();

        for val in rsi.history().iter().filter(|v| !v.is_nan()) {
            assert!(*val >= 0.0, "RSI {} below 0", val);
            assert!(*val <= 100.0, "RSI {} above 100", val);
        }
    }

    #[test]
    fn rsi_strong_uptrend() {
        let mut rsi = RSI::new(RSIConfig::new(5));
        // Strong uptrend
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let candles = candles_from_closes(&closes);

        rsi.calc(&candles).unwrap();

        let latest = rsi.latest().unwrap();
        // Should be very high (above 80) in strong uptrend
        assert!(latest > 80.0, "RSI in uptrend should be > 80, got {}", latest);
    }

    #[test]
    fn rsi_strong_downtrend() {
        let mut rsi = RSI::new(RSIConfig::new(5));
        // Strong downtrend
        let closes: Vec<f64> = (0..20).map(|i| 120.0 - i as f64).collect();
        let candles = candles_from_closes(&closes);

        rsi.calc(&candles).unwrap();

        let latest = rsi.latest().unwrap();
        // Should be very low (below 20) in strong downtrend
        assert!(latest < 20.0, "RSI in downtrend should be < 20, got {}", latest);
    }

    #[test]
    fn rsi_flat_market() {
        let mut rsi = RSI::new(RSIConfig::new(5));
        // Flat market
        let closes = vec![100.0; 20];
        let candles = candles_from_closes(&closes);

        rsi.calc(&candles).unwrap();

        let latest = rsi.latest().unwrap();
        // Should be around 50 in flat market
        assert!(
            (latest - 50.0).abs() < 1.0,
            "RSI in flat market should be ~50, got {}",
            latest
        );
    }

    #[test]
    fn rsi_batch_vs_streaming_equivalence() {
        let closes: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0).collect();
        let candles = candles_from_closes(&closes);
        let config = RSIConfig::new(5);

        // Batch
        let mut batch = RSI::new(config);
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = RSI::new(config);
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        // Compare valid values
        let batch_valid: Vec<_> = batch.history().iter().filter(|v| !v.is_nan()).collect();
        let stream_valid: Vec<_> = stream.history().iter().filter(|v| !v.is_nan()).collect();

        assert_eq!(
            batch_valid.len(),
            stream_valid.len(),
            "batch={}, stream={}",
            batch_valid.len(),
            stream_valid.len()
        );

        for (i, (b, s)) in batch_valid.iter().zip(stream_valid.iter()).enumerate() {
            assert!(
                (*b - *s).abs() < 1e-10,
                "RSI mismatch at {}: batch={}, stream={}",
                i,
                b,
                s
            );
        }
    }

    #[test]
    fn rsi_reset() {
        let mut rsi = RSI::new(RSIConfig::new(5));
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let candles = candles_from_closes(&closes);

        rsi.calc(&candles).unwrap();
        assert!(rsi.state().is_ready());

        rsi.reset();
        assert!(rsi.state().is_uninitialized());
        assert_eq!(rsi.len(), 0);
        assert!(rsi.latest().is_none());
    }

    #[test]
    fn rsi_insufficient_data() {
        let mut rsi = RSI::new(RSIConfig::new(14));
        let candles = candles_from_closes(&[100.0; 10]);

        let result = rsi.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }
}
