//! Average True Range (ATR) indicator.
//!
//! The ATR is a technical analysis indicator that measures market volatility.
//! It is an exponentially smoothed moving average of the True Range.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::ATRConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

use super::tr::TR;

/// Average True Range indicator.
///
/// Composes the TR indicator and applies Wilder's smoothing.
///
/// # Formula
///
/// First ATR = SMA of TR over period
/// Subsequent ATR = (Previous ATR × (period - 1) + Current TR) / period
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volatility::{ATR, ATRConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut atr = ATR::new(ATRConfig::new(14));
/// let candles: Vec<Ohlcv> = (0..20).map(|i| {
///     Ohlcv::new(i, 100.0 + i as f64, 105.0 + i as f64, 98.0 + i as f64, 103.0 + i as f64, 1000.0)
/// }).collect();
/// atr.calc(&candles).unwrap();
///
/// assert!(atr.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct ATR {
    config: ATRConfig,
    state: IndicatorState,
    /// Committed ATR value (Wilder smoothed)
    atr_value: f64,
    /// Sum of TR values during warmup
    warmup_tr_sum: f64,
    /// Number of TR values seen during warmup
    warmup_count: usize,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last seen timestamp for next() to detect new candles
    last_ts: i64,
    /// Previous close for TR calculation
    prev_close: f64,
}

impl ATR {
    /// Create a new ATR indicator with the given configuration.
    pub fn new(config: ATRConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            atr_value: 0.0,
            warmup_tr_sum: 0.0,
            warmup_count: 0,
            history: Vec::new(),
            latest: None,
            last_ts: i64::MIN,
            prev_close: 0.0,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }
}

impl Default for ATR {
    fn default() -> Self {
        Self::new(ATRConfig::default())
    }
}

impl Indicator for ATR {
    type Output = f64;
    type Config = ATRConfig;

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

        // Calculate TR values inline
        let mut tr_values = Vec::with_capacity(n);

        // First candle: TR = High - Low
        tr_values.push(data[0].high.0 - data[0].low.0);

        // Subsequent candles
        for i in 1..n {
            let tr = TR::calculate_tr(data[i].high.0, data[i].low.0, data[i - 1].close.0);
            tr_values.push(tr);
        }

        // Reset ATR state
        self.history = vec![f64::NAN; n];

        // First ATR is SMA of first `period` TR values
        let initial_sum: f64 = tr_values[..period].iter().sum();
        self.atr_value = initial_sum / period as f64;
        self.history[period - 1] = self.atr_value;

        // Wilder's smoothing for subsequent values
        let p_1 = (period - 1) as f64;
        for i in period..n {
            self.atr_value = (self.atr_value * p_1 + tr_values[i]) / period as f64;
            self.history[i] = self.atr_value;
        }

        self.prev_close = data.last().unwrap().close.0;
        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_ts = data.last().unwrap().timestamp.0;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len == 0 {
            return None;
        }

        let current = candles.last().unwrap();
        let current_ts = current.timestamp.0;
        let high = current.high.0;
        let low = current.low.0;
        let close = current.close.0;

        // Determine if this is a new candle or same snapshot (using timestamp)
        let is_new_candle = current_ts != self.last_ts;

        // Calculate TR for current candle
        let tr = if self.last_ts == i64::MIN {
            // Very first candle - TR is just high - low
            high - low
        } else if len >= 2 {
            // Use previous candle's close from the snapshot
            let prev_close = candles[len - 2].close.0;
            TR::calculate_tr(high, low, prev_close)
        } else {
            // Single candle in snapshot but we've seen data before
            TR::calculate_tr(high, low, self.prev_close)
        };

        // Handle warmup phase
        if self.warmup_count < period {
            if is_new_candle {
                self.warmup_tr_sum += tr;
                self.warmup_count += 1;
                self.prev_close = close;
                self.last_ts = current_ts;

                if self.warmup_count >= period {
                    // First valid ATR - use SMA of accumulated TR values
                    self.atr_value = self.warmup_tr_sum / period as f64;
                    self.history.push(self.atr_value);
                    self.latest = Some(self.atr_value);
                    self.state = IndicatorState::Ready;
                    return Some(self.atr_value);
                }

                self.history.push(f64::NAN);
                self.state = IndicatorState::Warming {
                    count: self.warmup_count,
                };
                return None;
            } else {
                // Same snapshot during warmup - no state change
                return None;
            }
        }

        // Ready state - apply Wilder smoothing
        let p_1 = (period - 1) as f64;
        let p = period as f64;

        if is_new_candle {
            // Commit new state
            self.atr_value = (self.atr_value * p_1 + tr) / p;
            self.prev_close = close;
            self.last_ts = current_ts;

            self.history.push(self.atr_value);
            self.latest = Some(self.atr_value);
            Some(self.atr_value)
        } else {
            // Same candle - compute tentatively without committing
            let tentative_atr = (self.atr_value * p_1 + tr) / p;

            // Update history for current candle
            if !self.history.is_empty() {
                *self.history.last_mut().unwrap() = tentative_atr;
            }
            self.latest = Some(tentative_atr);
            Some(tentative_atr)
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.atr_value = 0.0;
        self.warmup_tr_sum = 0.0;
        self.warmup_count = 0;
        self.history.clear();
        self.latest = None;
        self.last_ts = i64::MIN;
        self.prev_close = 0.0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for ATR {
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

    fn create_candle(ts: i64, open: f64, high: f64, low: f64, close: f64) -> Ohlcv {
        Ohlcv::new(ts, open, high, low, close, 1000.0)
    }

    fn generate_ohlcv_data(n: usize) -> Vec<Ohlcv> {
        let mut data = Vec::with_capacity(n);
        let mut close = 100.0;

        for i in 0..n {
            let volatility = 2.0 + (i as f64 * 0.1).sin().abs() * 3.0;
            let high = close + volatility;
            let low = close - volatility * 0.8;
            let new_close = close + (i as f64 * 0.3).sin() * 2.0;

            data.push(create_candle(i as i64, close, high, low, new_close));
            close = new_close;
        }

        data
    }

    #[test]
    fn atr_batch_calculation() {
        let mut atr = ATR::new(ATRConfig::new(5));
        let data = generate_ohlcv_data(20);

        atr.calc(&data).unwrap();

        assert!(atr.state().is_ready());
        assert_eq!(atr.len(), 20);

        // First 4 values should be NaN (need 5 for first ATR)
        for i in 0..4 {
            assert!(atr.get(i as isize).unwrap().is_nan());
        }

        // ATR should be valid from index 4 onwards
        let value = atr.get(4).unwrap();
        assert!(!value.is_nan());
        assert!(value > 0.0, "ATR should be positive");
    }

    #[test]
    fn atr_streaming_next() {
        let mut atr = ATR::new(ATRConfig::new(3));
        let candles = vec![
            create_candle(0, 100.0, 105.0, 98.0, 103.0),
            create_candle(1, 103.0, 108.0, 101.0, 106.0),
            create_candle(2, 106.0, 110.0, 104.0, 109.0),
            create_candle(3, 109.0, 112.0, 107.0, 111.0),
        ];

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = atr.next(snapshot);

            // Should get result when we have period = 3 candles
            if i >= 3 {
                assert!(result.is_some(), "Should have result at len {}", i);
                let atr_val = result.unwrap();
                assert!(atr_val > 0.0);
            }
        }

        assert!(atr.state().is_ready());
    }

    #[test]
    fn atr_always_positive() {
        let mut atr = ATR::new(ATRConfig::new(5));
        let data = generate_ohlcv_data(50);

        atr.calc(&data).unwrap();

        for val in atr.history().iter().filter(|v| !v.is_nan()) {
            assert!(*val > 0.0, "ATR should always be > 0, got {}", val);
        }
    }

    #[test]
    fn atr_batch_vs_streaming_equivalence() {
        let candles = generate_ohlcv_data(30);
        let config = ATRConfig::new(5);

        // Batch
        let mut batch = ATR::new(config);
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = ATR::new(config);
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
                "ATR mismatch at {}: batch={}, stream={}",
                i,
                b,
                s
            );
        }
    }

    #[test]
    fn atr_smoothing() {
        let mut atr = ATR::new(ATRConfig::new(5));
        let candles = generate_ohlcv_data(30);

        atr.calc(&candles).unwrap();

        // Calculate TR values to compare
        let mut tr_values = Vec::with_capacity(candles.len());
        tr_values.push(candles[0].high.0 - candles[0].low.0);
        for i in 1..candles.len() {
            let tr =
                TR::calculate_tr(candles[i].high.0, candles[i].low.0, candles[i - 1].close.0);
            tr_values.push(tr);
        }

        let atr_values: Vec<f64> = atr.history().iter().copied().filter(|v| !v.is_nan()).collect();

        // Calculate variance of both
        let tr_mean: f64 = tr_values.iter().sum::<f64>() / tr_values.len() as f64;
        let atr_mean: f64 = atr_values.iter().sum::<f64>() / atr_values.len() as f64;

        let tr_var: f64 =
            tr_values.iter().map(|v| (v - tr_mean).powi(2)).sum::<f64>() / tr_values.len() as f64;
        let atr_var: f64 = atr_values
            .iter()
            .map(|v| (v - atr_mean).powi(2))
            .sum::<f64>()
            / atr_values.len() as f64;

        assert!(
            atr_var <= tr_var,
            "ATR variance ({}) should be <= TR variance ({})",
            atr_var,
            tr_var
        );
    }

    #[test]
    fn atr_reset() {
        let mut atr = ATR::new(ATRConfig::new(5));
        let data = generate_ohlcv_data(20);

        atr.calc(&data).unwrap();
        assert!(atr.state().is_ready());

        atr.reset();
        assert!(atr.state().is_uninitialized());
        assert_eq!(atr.len(), 0);
        assert!(atr.latest().is_none());
    }

    #[test]
    fn atr_insufficient_data() {
        let mut atr = ATR::new(ATRConfig::new(14));
        let candles = generate_ohlcv_data(10);

        let result = atr.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }
}
