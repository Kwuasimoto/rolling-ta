//! Directional Movement Indicator (DMI).
//!
//! The DMI measures the strength of a trend by calculating positive and negative
//! directional movement. It provides +DI and -DI values that indicate trend direction.

use crate::ta::{
    config::DMIConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    volatility::TR,
    HistoricalIndicator, Indicator,
};

/// Output from DMI indicator containing +DI and -DI.
#[derive(Debug, Clone, Copy, Default)]
pub struct DMIOutput {
    pub plus_di: f64,
    pub minus_di: f64,
}

impl DMIOutput {
    pub fn new(plus_di: f64, minus_di: f64) -> Self {
        Self { plus_di, minus_di }
    }
}

/// Directional Movement Indicator.
///
/// Calculates +DI and -DI using smoothed directional movement and true range.
///
/// # Formula
///
/// +DM = max(high - prev_high, 0) if > max(prev_low - low, 0), else 0
/// -DM = max(prev_low - low, 0) if > max(high - prev_high, 0), else 0
///
/// Smoothing uses Wilder's method: prev - (prev/period) + current
///
/// +DI = (smoothed_+DM / smoothed_TR) * 100
/// -DI = (smoothed_-DM / smoothed_TR) * 100
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{DMI, DMIConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut dmi = DMI::new(DMIConfig::new(14));
/// let data = OhlcvSeries::from_tuples(&[
///     (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     (1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     // ... more data needed for valid DMI
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct DMI {
    config: DMIConfig,
    state: IndicatorState,
    tr: TR,
    // Smoothed values for streaming
    smoothed_plus_dm: f64,
    smoothed_minus_dm: f64,
    smoothed_tr: f64,
    // Previous candle values for DM calculation
    prev_high: f64,
    prev_low: f64,
    // History
    history: Vec<DMIOutput>,
    latest: Option<DMIOutput>,
}

impl DMI {
    /// Create a new DMI indicator with the given configuration.
    pub fn new(config: DMIConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            tr: TR::default(),
            smoothed_plus_dm: 0.0,
            smoothed_minus_dm: 0.0,
            smoothed_tr: 0.0,
            prev_high: 0.0,
            prev_low: 0.0,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate directional movement values.
    #[inline]
    fn calculate_dm(high: f64, low: f64, prev_high: f64, prev_low: f64) -> (f64, f64) {
        let move_up = high - prev_high;
        let move_down = prev_low - low;

        let plus_dm = if move_up > 0.0 && move_up > move_down {
            move_up
        } else {
            0.0
        };

        let minus_dm = if move_down > 0.0 && move_down > move_up {
            move_down
        } else {
            0.0
        };

        (plus_dm, minus_dm)
    }

    /// Apply Wilder's smoothing: prev - (prev/period) + current
    #[inline]
    fn wilder_smooth(prev: f64, current: f64, period: usize) -> f64 {
        prev - (prev / period as f64) + current
    }

    /// Get +DI latest value.
    #[inline]
    pub fn plus_di_latest(&self) -> Option<f64> {
        self.latest.map(|o| o.plus_di)
    }

    /// Get -DI latest value.
    #[inline]
    pub fn minus_di_latest(&self) -> Option<f64> {
        self.latest.map(|o| o.minus_di)
    }

    /// Get the smoothed +DM value (for ADX composition).
    #[inline]
    pub fn smoothed_plus_dm(&self) -> f64 {
        self.smoothed_plus_dm
    }

    /// Get the smoothed -DM value (for ADX composition).
    #[inline]
    pub fn smoothed_minus_dm(&self) -> f64 {
        self.smoothed_minus_dm
    }

    /// Get the smoothed TR value (for ADX composition).
    #[inline]
    pub fn smoothed_tr(&self) -> f64 {
        self.smoothed_tr
    }

    /// Get reference to the internal TR indicator.
    #[inline]
    pub fn tr(&self) -> &TR {
        &self.tr
    }
}

impl Default for DMI {
    fn default() -> Self {
        Self::new(DMIConfig::default())
    }
}

impl Indicator for DMI {
    type Output = DMIOutput;
    type Config = DMIConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        // Need period + 1 candles for initial smoothing (first candle has no DM)
        if n < period + 1 {
            return Err(TAError::InsufficientData {
                required: period + 1,
                actual: n,
            });
        }

        if period == 0 {
            return Err(TAError::InvalidPeriod(0));
        }

        // Calculate TR first
        self.tr.calc(data)?;
        let tr_values = self.tr.history();

        let highs = &data.highs;
        let lows = &data.lows;

        // Reset state
        self.history = vec![DMIOutput::default(); n];

        // Calculate DM for all candles (index 0 has no previous, so DM = 0)
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];

        for i in 1..n {
            let (pdm, ndm) = Self::calculate_dm(highs[i], lows[i], highs[i - 1], lows[i - 1]);
            plus_dm[i] = pdm;
            minus_dm[i] = ndm;
        }

        // Initial smoothing: sum of first period values (indices 1 to period)
        let mut smoothed_plus_dm: f64 = plus_dm[1..=period].iter().sum();
        let mut smoothed_minus_dm: f64 = minus_dm[1..=period].iter().sum();
        let mut smoothed_tr: f64 = tr_values[1..=period].iter().sum();

        // First valid DMI at index = period
        let plus_di = if smoothed_tr > 0.0 {
            (smoothed_plus_dm / smoothed_tr) * 100.0
        } else {
            0.0
        };
        let minus_di = if smoothed_tr > 0.0 {
            (smoothed_minus_dm / smoothed_tr) * 100.0
        } else {
            0.0
        };
        self.history[period] = DMIOutput::new(plus_di, minus_di);

        // Calculate subsequent values using Wilder's smoothing
        for i in (period + 1)..n {
            smoothed_plus_dm = Self::wilder_smooth(smoothed_plus_dm, plus_dm[i], period);
            smoothed_minus_dm = Self::wilder_smooth(smoothed_minus_dm, minus_dm[i], period);
            smoothed_tr = Self::wilder_smooth(smoothed_tr, tr_values[i], period);

            let plus_di = if smoothed_tr > 0.0 {
                (smoothed_plus_dm / smoothed_tr) * 100.0
            } else {
                0.0
            };
            let minus_di = if smoothed_tr > 0.0 {
                (smoothed_minus_dm / smoothed_tr) * 100.0
            } else {
                0.0
            };
            self.history[i] = DMIOutput::new(plus_di, minus_di);
        }

        // Store state for streaming updates
        self.smoothed_plus_dm = smoothed_plus_dm;
        self.smoothed_minus_dm = smoothed_minus_dm;
        self.smoothed_tr = smoothed_tr;
        self.prev_high = *highs.last().unwrap();
        self.prev_low = *lows.last().unwrap();
        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let period = self.config.period;
        let high = tick.high.0;
        let low = tick.low.0;

        // Update TR
        self.tr.update(tick)?;
        let tr_current = self.tr.latest().unwrap_or(0.0);

        if self.state.is_uninitialized() {
            // First tick - initialize but don't accumulate (TR[0] is not valid for smoothing)
            self.history.push(DMIOutput::default());
            self.prev_high = high;
            self.prev_low = low;
            self.state = IndicatorState::Warming { count: 1 };
            return Ok(None);
        }

        // Calculate DM
        let (plus_dm, minus_dm) = Self::calculate_dm(high, low, self.prev_high, self.prev_low);

        match self.state {
            IndicatorState::Warming { count } => {
                // Accumulate for initial smoothing (starting from tick 1)
                self.smoothed_plus_dm += plus_dm;
                self.smoothed_minus_dm += minus_dm;
                self.smoothed_tr += tr_current;

                if count >= period {
                    // Ready to output first valid DMI
                    let plus_di = if self.smoothed_tr > 0.0 {
                        (self.smoothed_plus_dm / self.smoothed_tr) * 100.0
                    } else {
                        0.0
                    };
                    let minus_di = if self.smoothed_tr > 0.0 {
                        (self.smoothed_minus_dm / self.smoothed_tr) * 100.0
                    } else {
                        0.0
                    };

                    let output = DMIOutput::new(plus_di, minus_di);
                    self.history.push(output);
                    self.latest = Some(output);
                    self.prev_high = high;
                    self.prev_low = low;
                    self.state = IndicatorState::Ready;
                    return Ok(Some(output));
                } else {
                    self.history.push(DMIOutput::default());
                    self.prev_high = high;
                    self.prev_low = low;
                    self.state = IndicatorState::Warming { count: count + 1 };
                    return Ok(None);
                }
            }
            IndicatorState::Ready => {
                // Apply Wilder's smoothing
                self.smoothed_plus_dm = Self::wilder_smooth(self.smoothed_plus_dm, plus_dm, period);
                self.smoothed_minus_dm =
                    Self::wilder_smooth(self.smoothed_minus_dm, minus_dm, period);
                self.smoothed_tr = Self::wilder_smooth(self.smoothed_tr, tr_current, period);

                let plus_di = if self.smoothed_tr > 0.0 {
                    (self.smoothed_plus_dm / self.smoothed_tr) * 100.0
                } else {
                    0.0
                };
                let minus_di = if self.smoothed_tr > 0.0 {
                    (self.smoothed_minus_dm / self.smoothed_tr) * 100.0
                } else {
                    0.0
                };

                let output = DMIOutput::new(plus_di, minus_di);
                self.history.push(output);
                self.latest = Some(output);
                self.prev_high = high;
                self.prev_low = low;

                Ok(Some(output))
            }
            IndicatorState::Uninitialized => unreachable!(),
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.tr.reset();
        self.smoothed_plus_dm = 0.0;
        self.smoothed_minus_dm = 0.0;
        self.smoothed_tr = 0.0;
        self.prev_high = 0.0;
        self.prev_low = 0.0;
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period + 1
    }
}

impl HistoricalIndicator for DMI {
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
    fn dmi_batch_calculation() {
        let mut dmi = DMI::new(DMIConfig::new(14));
        let data = generate_ohlcv_data(50);

        dmi.calc(&data).unwrap();

        assert!(dmi.state().is_ready());
        assert_eq!(dmi.len(), 50);

        // First 14 values should be default (period = 14, first valid at index 14)
        for i in 0..14 {
            let val = dmi.get(i as isize).unwrap();
            assert_eq!(val.plus_di, 0.0);
            assert_eq!(val.minus_di, 0.0);
        }

        // DMI should be valid from index 14 onwards
        let val = dmi.get(14).unwrap();
        assert!(val.plus_di >= 0.0 && val.plus_di <= 100.0);
        assert!(val.minus_di >= 0.0 && val.minus_di <= 100.0);
    }

    #[test]
    fn dmi_values_in_range() {
        let mut dmi = DMI::new(DMIConfig::new(14));
        let data = generate_ohlcv_data(100);

        dmi.calc(&data).unwrap();

        for output in dmi.history().iter().skip(14) {
            assert!(
                output.plus_di >= 0.0 && output.plus_di <= 100.0,
                "+DI should be in [0, 100], got {}",
                output.plus_di
            );
            assert!(
                output.minus_di >= 0.0 && output.minus_di <= 100.0,
                "-DI should be in [0, 100], got {}",
                output.minus_di
            );
        }
    }

    #[test]
    fn dmi_uptrend() {
        // Create strong uptrend data
        let mut data_vec = Vec::new();
        for i in 0..50 {
            let base = 100.0 + i as f64 * 2.0; // Strong upward trend
            data_vec.push((i as i64, base, base + 3.0, base - 1.0, base + 1.5, 1000.0));
        }
        let data = OhlcvSeries::from_tuples(&data_vec);

        let mut dmi = DMI::new(DMIConfig::new(14));
        dmi.calc(&data).unwrap();

        // In a strong uptrend, +DI should generally be > -DI
        let last = dmi.latest().unwrap();
        assert!(
            last.plus_di > last.minus_di,
            "In uptrend, +DI ({}) should > -DI ({})",
            last.plus_di,
            last.minus_di
        );
    }

    #[test]
    fn dmi_downtrend() {
        // Create strong downtrend data
        let mut data_vec = Vec::new();
        for i in 0..50 {
            let base = 200.0 - i as f64 * 2.0; // Strong downward trend
            data_vec.push((i as i64, base, base + 1.0, base - 3.0, base - 1.5, 1000.0));
        }
        let data = OhlcvSeries::from_tuples(&data_vec);

        let mut dmi = DMI::new(DMIConfig::new(14));
        dmi.calc(&data).unwrap();

        // In a strong downtrend, -DI should generally be > +DI
        let last = dmi.latest().unwrap();
        assert!(
            last.minus_di > last.plus_di,
            "In downtrend, -DI ({}) should > +DI ({})",
            last.minus_di,
            last.plus_di
        );
    }

    #[test]
    fn dmi_streaming_matches_batch() {
        let data = generate_ohlcv_data(30);

        // Batch calculation
        let mut batch_dmi = DMI::new(DMIConfig::new(5));
        batch_dmi.calc(&data).unwrap();

        // Streaming calculation
        let mut stream_dmi = DMI::new(DMIConfig::new(5));
        for i in 0..data.len() {
            let tick = data.get(i).unwrap();
            stream_dmi.update(&tick).unwrap();
        }

        // Compare results (allow small floating point differences)
        let epsilon = 1e-6;
        for i in 0..batch_dmi.len() {
            let batch_val = batch_dmi.get(i as isize).unwrap();
            let stream_val = stream_dmi.get(i as isize).unwrap();

            assert!(
                (batch_val.plus_di - stream_val.plus_di).abs() < epsilon,
                "Index {}: batch +DI ({}) != stream +DI ({})",
                i,
                batch_val.plus_di,
                stream_val.plus_di
            );
            assert!(
                (batch_val.minus_di - stream_val.minus_di).abs() < epsilon,
                "Index {}: batch -DI ({}) != stream -DI ({})",
                i,
                batch_val.minus_di,
                stream_val.minus_di
            );
        }
    }
}
