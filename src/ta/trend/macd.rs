//! Moving Average Convergence Divergence (MACD) indicator.
//!
//! MACD is a trend-following momentum indicator that shows the relationship
//! between two exponential moving averages of an asset's price.

use crate::ta::{
    config::{EMAConfig, MACDConfig},
    error::{TAError, TAResult},
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
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
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut macd = MACD::new(MACDConfig::new(12, 26, 9));
/// // Need 26+ candles for MACD to warm up (slow period)
/// // Need 26 + 9 - 1 = 34 candles for signal to warm up
/// ```
#[derive(Debug, Clone)]
pub struct MACD {
    config: MACDConfig,
    state: IndicatorState,
    fast_ema: EMA,
    slow_ema: EMA,
    // Signal line state (EMA of MACD values)
    signal_mult: f64,
    signal_value: f64,
    signal_warmup: usize,
    history: Vec<MACDOutput>,
    latest: Option<MACDOutput>,
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
            signal_value: 0.0,
            signal_warmup: 0,
            history: Vec::new(),
            latest: None,
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

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let n = data.len();
        let min_required = self.config.slow_period;

        if n < min_required {
            return Err(TAError::InsufficientData {
                required: min_required,
                actual: n,
            });
        }

        // Calculate both EMAs
        self.fast_ema.calc(data)?;
        self.slow_ema.calc(data)?;

        let fast_history = self.fast_ema.history();
        let slow_history = self.slow_ema.history();

        // Reset MACD state
        self.history = Vec::with_capacity(n);

        // Collect MACD values for signal calculation
        let mut macd_values: Vec<f64> = Vec::with_capacity(n);

        // MACD line becomes valid when slow EMA is valid (slow_period - 1)
        let macd_start = self.config.slow_period - 1;

        // Fill NaN until we have valid MACD
        for _ in 0..macd_start {
            self.history.push(MACDOutput::default());
            macd_values.push(f64::NAN);
        }

        // Calculate MACD values
        for i in macd_start..n {
            let macd = fast_history[i] - slow_history[i];
            macd_values.push(macd);
        }

        // Calculate signal line (EMA of MACD values)
        // Signal starts after macd_start + signal_period - 1
        let signal_start = macd_start + self.config.signal_period - 1;

        // Initialize signal EMA using first signal_period MACD values
        let signal_init_start = macd_start;
        let signal_init_end = signal_start + 1;

        let mut signal = if signal_init_end <= n {
            // SMA of first signal_period MACD values
            let sum: f64 = macd_values[signal_init_start..signal_init_end].iter().sum();
            sum / self.config.signal_period as f64
        } else {
            0.0
        };

        // Fill MACD-only values (no signal yet)
        for i in macd_start..signal_start.min(n) {
            self.history.push(MACDOutput {
                macd: macd_values[i],
                signal: f64::NAN,
                histogram: f64::NAN,
            });
        }

        // Calculate full MACD output with signal
        if signal_start < n {
            // First signal value
            let macd_val = macd_values[signal_start];
            self.history.push(MACDOutput {
                macd: macd_val,
                signal,
                histogram: macd_val - signal,
            });

            // Subsequent values with EMA smoothing
            for i in (signal_start + 1)..n {
                let macd_val = macd_values[i];
                signal = macd_val * self.signal_mult + signal * (1.0 - self.signal_mult);
                self.history.push(MACDOutput {
                    macd: macd_val,
                    signal,
                    histogram: macd_val - signal,
                });
            }
        }

        self.signal_value = signal;
        self.signal_warmup = self.config.signal_period;
        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        // Update both EMAs
        let fast_result = self.fast_ema.update(tick)?;
        let slow_result = self.slow_ema.update(tick)?;

        // Need both EMAs to be valid for MACD
        match (fast_result, slow_result) {
            (Some(fast), Some(slow)) => {
                let macd = fast - slow;

                // Handle signal warmup
                if self.signal_warmup < self.config.signal_period {
                    self.signal_warmup += 1;

                    if self.signal_warmup == 1 {
                        // First MACD value
                        self.signal_value = macd;
                    } else {
                        // Accumulate for initial SMA
                        self.signal_value += macd;
                    }

                    if self.signal_warmup < self.config.signal_period {
                        let output = MACDOutput {
                            macd,
                            signal: f64::NAN,
                            histogram: f64::NAN,
                        };
                        self.history.push(output);
                        self.state = IndicatorState::Warming {
                            count: self.slow_ema.warmup_period() + self.signal_warmup,
                        };
                        return Ok(Some(output));
                    }

                    // First valid signal (SMA)
                    self.signal_value /= self.config.signal_period as f64;
                }

                // Apply EMA smoothing to signal
                self.signal_value =
                    macd * self.signal_mult + self.signal_value * (1.0 - self.signal_mult);

                let output = MACDOutput {
                    macd,
                    signal: self.signal_value,
                    histogram: macd - self.signal_value,
                };

                self.history.push(output);
                self.latest = Some(output);
                self.state = IndicatorState::Ready;

                Ok(Some(output))
            }
            _ => {
                // Still warming up EMAs
                self.history.push(MACDOutput::default());
                let count = self.fast_ema.history().len().max(self.slow_ema.history().len());
                self.state = IndicatorState::Warming { count };
                Ok(None)
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_value = 0.0;
        self.signal_warmup = 0;
        self.history.clear();
        self.latest = None;
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

    fn generate_trend_data(n: usize) -> OhlcvSeries {
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        OhlcvSeries::from_closes(&closes)
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
        let data = OhlcvSeries::from_closes(&closes);

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
    fn macd_streaming_update() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));

        // Feed data one by one
        for i in 0..20 {
            let tick = Ohlcv::from_close(100.0 + i as f64);
            let _ = macd.update(&tick);
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
    fn macd_reset() {
        let mut macd = MACD::new(MACDConfig::new(3, 5, 3));
        let data = generate_trend_data(20);

        macd.calc(&data).unwrap();
        assert!(macd.state().is_ready());

        macd.reset();
        assert!(macd.state().is_uninitialized());
        assert_eq!(macd.len(), 0);
    }
}
