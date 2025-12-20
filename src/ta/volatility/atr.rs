//! Average True Range (ATR) indicator.
//!
//! The ATR is a technical analysis indicator that measures market volatility.
//! It is an exponentially smoothed moving average of the True Range.

use crate::ta::{
    config::ATRConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
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
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut atr = ATR::new(ATRConfig::new(14));
/// let data = OhlcvSeries::from_tuples(&[
///     (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     (1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     // ... more data ...
/// ]);
/// // Need 14+ candles for ATR to be ready
/// ```
#[derive(Debug, Clone)]
pub struct ATR {
    config: ATRConfig,
    state: IndicatorState,
    tr: TR,
    atr_value: f64,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl ATR {
    /// Create a new ATR indicator with the given configuration.
    pub fn new(config: ATRConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            tr: TR::default(),
            atr_value: 0.0,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Get a reference to the internal TR indicator.
    #[inline]
    pub fn tr(&self) -> &TR {
        &self.tr
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

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
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

        // Calculate TR first
        self.tr.calc(data)?;
        let tr_values = self.tr.history();

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

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let period = self.config.period;

        // Update TR first
        self.tr.update(tick)?;
        let tr_current = self.tr.latest().unwrap_or(0.0);

        // Handle initial warmup
        if self.state.is_uninitialized() {
            self.atr_value = tr_current; // Start accumulating
            self.history.push(f64::NAN);
            self.state = IndicatorState::Warming { count: 1 };
            return Ok(None);
        }

        match self.state {
            IndicatorState::Warming { count } => {
                // Accumulate TR values for initial SMA
                self.atr_value += tr_current;

                if count >= period - 1 {
                    // Calculate initial ATR as SMA
                    self.atr_value /= period as f64;
                    self.history.push(self.atr_value);
                    self.latest = Some(self.atr_value);
                    self.state = IndicatorState::Ready;
                    Ok(Some(self.atr_value))
                } else {
                    self.history.push(f64::NAN);
                    self.state = IndicatorState::Warming { count: count + 1 };
                    Ok(None)
                }
            }
            IndicatorState::Ready => {
                // Wilder's smoothing
                let p_1 = (period - 1) as f64;
                self.atr_value = (self.atr_value * p_1 + tr_current) / period as f64;
                self.history.push(self.atr_value);
                self.latest = Some(self.atr_value);
                Ok(Some(self.atr_value))
            }
            IndicatorState::Uninitialized => unreachable!(),
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.tr.reset();
        self.atr_value = 0.0;
        self.history.clear();
        self.latest = None;
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

    fn generate_ohlcv_data(n: usize) -> OhlcvSeries {
        let mut data = Vec::with_capacity(n);
        let mut close = 100.0;

        for i in 0..n {
            let volatility = 2.0 + (i as f64 * 0.1).sin().abs() * 3.0;
            let high = close + volatility;
            let low = close - volatility * 0.8;
            let new_close = close + (i as f64 * 0.3).sin() * 2.0;

            data.push((i as i64, close, high, low, new_close, 1000.0 + i as f64 * 10.0));
            close = new_close;
        }

        OhlcvSeries::from_tuples(&data)
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
    fn atr_streaming_update() {
        let mut atr = ATR::new(ATRConfig::new(3));

        // Warmup
        assert!(atr.update(&Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0)).unwrap().is_none());
        assert!(atr.update(&Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0)).unwrap().is_none());

        // Third value - should get first ATR
        let result = atr.update(&Ohlcv::new(2, 106.0, 110.0, 104.0, 109.0, 1200.0)).unwrap();
        assert!(result.is_some());
        let atr_val = result.unwrap();
        assert!(atr_val > 0.0);
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
    fn atr_smoothing() {
        let mut atr = ATR::new(ATRConfig::new(5));
        let data = generate_ohlcv_data(30);

        atr.calc(&data).unwrap();

        // ATR should be smoother than TR (less volatile)
        let tr_values: Vec<f64> = atr.tr().history().iter().copied().collect();
        let atr_values: Vec<f64> = atr.history().iter().copied().filter(|v| !v.is_nan()).collect();

        // Calculate variance of both
        let tr_mean: f64 = tr_values.iter().sum::<f64>() / tr_values.len() as f64;
        let atr_mean: f64 = atr_values.iter().sum::<f64>() / atr_values.len() as f64;

        let tr_var: f64 = tr_values.iter().map(|v| (v - tr_mean).powi(2)).sum::<f64>() / tr_values.len() as f64;
        let atr_var: f64 = atr_values.iter().map(|v| (v - atr_mean).powi(2)).sum::<f64>() / atr_values.len() as f64;

        assert!(atr_var <= tr_var, "ATR variance ({}) should be <= TR variance ({})", atr_var, tr_var);
    }
}
