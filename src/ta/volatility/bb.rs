//! Bollinger Bands indicator.
//!
//! Bollinger Bands consist of a middle band (SMA) and two outer bands
//! based on standard deviation, used to identify volatility.

use std::collections::VecDeque;

use crate::ta::{
    config::BBConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    trend::SMA,
    types::{Ohlcv, OhlcvSeries},
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
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut bb = BB::new(BBConfig::new(20, 2.0));
/// let data = OhlcvSeries::from_closes(&[
///     100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0,
///     106.0, 105.5, 107.0, 106.0, 108.0, 107.5, 109.0, 108.0, 110.0, 109.0,
/// ]);
/// bb.calc(&data).unwrap();
///
/// assert!(bb.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct BB {
    config: BBConfig,
    state: IndicatorState,
    sma: SMA,
    window: VecDeque<f64>,
    history: Vec<BBOutput>,
    latest: Option<BBOutput>,
}

impl BB {
    /// Create a new Bollinger Bands indicator.
    pub fn new(config: BBConfig) -> Self {
        use crate::ta::config::SMAConfig;

        Self {
            config,
            state: IndicatorState::Uninitialized,
            sma: SMA::new(SMAConfig::new(config.period)),
            window: VecDeque::with_capacity(config.period),
            history: Vec::new(),
            latest: None,
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

    /// Calculate population standard deviation of a slice.
    #[inline]
    fn pop_std_dev(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mean: f64 = data.iter().sum::<f64>() / n;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Calculate population standard deviation from VecDeque.
    #[inline]
    fn pop_std_dev_deque(data: &VecDeque<f64>) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mean: f64 = data.iter().sum::<f64>() / n;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
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

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        // Calculate SMA first
        self.sma.calc(data)?;
        let sma_values = self.sma.history();

        // Reset BB state
        self.history = Vec::with_capacity(n);
        let closes = &data.closes;
        let mult = self.config.std_dev_mult;

        // Fill with NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(BBOutput::default());
        }

        // Calculate BB from period-1 onwards
        for i in (period - 1)..n {
            let ma = sma_values[i];
            let window_slice = &closes[(i + 1 - period)..=i];
            let std_dev = Self::pop_std_dev(window_slice);
            let weighted_std = std_dev * mult;

            self.history.push(BBOutput {
                upper: ma + weighted_std,
                middle: ma,
                lower: ma - weighted_std,
            });
        }

        // Set up window for streaming
        self.window.clear();
        for &v in &closes[(n - period)..] {
            self.window.push_back(v);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let period = self.config.period;
        let close = tick.close.0;

        // Update SMA
        let sma_result = self.sma.update(tick)?;

        // Handle warming
        if self.window.len() < period {
            self.window.push_back(close);

            if self.window.len() < period {
                self.history.push(BBOutput::default());
                self.state = IndicatorState::Warming {
                    count: self.window.len(),
                };
                return Ok(None);
            }
        } else {
            self.window.pop_front();
            self.window.push_back(close);
        }

        // Now we have enough data
        let ma = sma_result.unwrap_or_else(|| self.sma.latest().unwrap_or(0.0));
        let std_dev = Self::pop_std_dev_deque(&self.window);
        let weighted_std = std_dev * self.config.std_dev_mult;

        let output = BBOutput {
            upper: ma + weighted_std,
            middle: ma,
            lower: ma - weighted_std,
        };

        self.history.push(output);
        self.latest = Some(output);
        self.state = IndicatorState::Ready;

        Ok(Some(output))
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.sma.reset();
        self.window.clear();
        self.history.clear();
        self.latest = None;
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

    #[test]
    fn bb_batch_calculation() {
        let mut bb = BB::new(BBConfig::new(5, 2.0));
        let data = OhlcvSeries::from_closes(&[
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
        ]);

        bb.calc(&data).unwrap();

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
    fn bb_streaming_update() {
        let mut bb = BB::new(BBConfig::new(3, 2.0));

        // Warmup
        assert!(bb
            .update(&Ohlcv::new(0, 100.0, 102.0, 99.0, 100.0, 1000.0))
            .unwrap()
            .is_none());
        assert!(bb
            .update(&Ohlcv::new(1, 100.0, 103.0, 99.0, 102.0, 1000.0))
            .unwrap()
            .is_none());

        // Third value - should get first BB
        let result = bb
            .update(&Ohlcv::new(2, 102.0, 105.0, 101.0, 104.0, 1000.0))
            .unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        assert!(output.upper > output.middle);
        assert!(output.middle > output.lower);
    }

    #[test]
    fn bb_bands_widen_with_volatility() {
        let mut bb_stable = BB::new(BBConfig::new(5, 2.0));
        let mut bb_volatile = BB::new(BBConfig::new(5, 2.0));

        // Stable prices
        let stable = OhlcvSeries::from_closes(&[100.0, 100.0, 100.0, 100.0, 100.0]);

        // Volatile prices (same mean)
        let volatile = OhlcvSeries::from_closes(&[80.0, 120.0, 80.0, 120.0, 100.0]);

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
        let data = OhlcvSeries::from_closes(&[100.0, 102.0, 101.0, 103.0, 105.0]);

        bb.calc(&data).unwrap();
        assert!(bb.state().is_ready());
        assert_eq!(bb.len(), 5);

        bb.reset();
        assert!(bb.state().is_uninitialized());
        assert_eq!(bb.len(), 0);
    }

    #[test]
    fn bb_pop_std_dev() {
        // Test: std dev of [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5, variance = 4, std_dev = 2
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std_dev = BB::pop_std_dev(&data);
        assert!((std_dev - 2.0).abs() < 1e-10);
    }
}
