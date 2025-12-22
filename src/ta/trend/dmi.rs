//! Directional Movement Indicator (DMI).
//!
//! The DMI measures the strength of a trend by calculating positive and negative
//! directional movement. It provides +DI and -DI values that indicate trend direction.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::DMIConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
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
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut dmi = DMI::new(DMIConfig::new(14));
/// let candles: Vec<Ohlcv> = (0..20)
///     .map(|i| Ohlcv::new(i, 100.0 + i as f64, 105.0 + i as f64, 98.0 + i as f64, 103.0 + i as f64, 1000.0))
///     .collect();
/// dmi.calc(&candles).unwrap();
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
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
    /// Committed state for same-candle updates
    committed_smoothed_plus_dm: f64,
    committed_smoothed_minus_dm: f64,
    committed_smoothed_tr: f64,
    committed_prev_high: f64,
    committed_prev_low: f64,
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
            last_len: 0,
            committed_smoothed_plus_dm: 0.0,
            committed_smoothed_minus_dm: 0.0,
            committed_smoothed_tr: 0.0,
            committed_prev_high: 0.0,
            committed_prev_low: 0.0,
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

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
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

        // Reset state
        self.history = vec![DMIOutput::default(); n];

        // Calculate DM for all candles (index 0 has no previous, so DM = 0)
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];

        for i in 1..n {
            let (pdm, ndm) =
                Self::calculate_dm(data[i].high.0, data[i].low.0, data[i - 1].high.0, data[i - 1].low.0);
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
        self.prev_high = data.last().unwrap().high.0;
        self.prev_low = data.last().unwrap().low.0;
        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len == 0 || period == 0 {
            return None;
        }

        // Need period + 1 candles minimum
        if len < period + 1 {
            return None;
        }

        // Determine if this is a new candle
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First time initialization - compute from scratch
                // Calculate TR for all candles
                self.tr.calc(candles).ok()?;
                let tr_values = self.tr.history();

                // Calculate DM for all candles
                let mut plus_dm = vec![0.0; len];
                let mut minus_dm = vec![0.0; len];

                for i in 1..len {
                    let (pdm, ndm) = Self::calculate_dm(
                        candles[i].high.0,
                        candles[i].low.0,
                        candles[i - 1].high.0,
                        candles[i - 1].low.0,
                    );
                    plus_dm[i] = pdm;
                    minus_dm[i] = ndm;
                }

                // Initial smoothing: sum of first period values
                self.smoothed_plus_dm = plus_dm[1..=period].iter().sum();
                self.smoothed_minus_dm = minus_dm[1..=period].iter().sum();
                self.smoothed_tr = tr_values[1..=period].iter().sum();

                // Fill history
                self.history = vec![DMIOutput::default(); len];

                // First valid DMI at index = period
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
                self.history[period] = DMIOutput::new(plus_di, minus_di);

                // Calculate all subsequent values
                for i in (period + 1)..len {
                    // Save committed state BEFORE processing the last candle
                    if i == len - 1 {
                        self.committed_smoothed_plus_dm = self.smoothed_plus_dm;
                        self.committed_smoothed_minus_dm = self.smoothed_minus_dm;
                        self.committed_smoothed_tr = self.smoothed_tr;
                        self.committed_prev_high = candles[i - 1].high.0;
                        self.committed_prev_low = candles[i - 1].low.0;
                    }

                    self.smoothed_plus_dm =
                        Self::wilder_smooth(self.smoothed_plus_dm, plus_dm[i], period);
                    self.smoothed_minus_dm =
                        Self::wilder_smooth(self.smoothed_minus_dm, minus_dm[i], period);
                    self.smoothed_tr = Self::wilder_smooth(self.smoothed_tr, tr_values[i], period);

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
                    self.history[i] = DMIOutput::new(plus_di, minus_di);
                }

                // If we only have exactly period+1 candles, committed state is the initial smoothing
                if len == period + 1 {
                    self.committed_smoothed_plus_dm = self.smoothed_plus_dm;
                    self.committed_smoothed_minus_dm = self.smoothed_minus_dm;
                    self.committed_smoothed_tr = self.smoothed_tr;
                    self.committed_prev_high = candles[len - 2].high.0;
                    self.committed_prev_low = candles[len - 2].low.0;
                }

                self.prev_high = candles.last().unwrap().high.0;
                self.prev_low = candles.last().unwrap().low.0;
                self.latest = self.history.last().copied();
                self.last_len = len;
                self.state = IndicatorState::Ready;

                return self.latest;
            }

            // New candle added - save current state as committed, then compute new value
            self.committed_smoothed_plus_dm = self.smoothed_plus_dm;
            self.committed_smoothed_minus_dm = self.smoothed_minus_dm;
            self.committed_smoothed_tr = self.smoothed_tr;
            self.committed_prev_high = self.prev_high;
            self.committed_prev_low = self.prev_low;

            // Get TR for the new candle
            self.tr.next(candles)?;
            let tr_current = self.tr.latest().unwrap_or(0.0);

            // Calculate DM for new candle
            let current = candles.last().unwrap();
            let (plus_dm, minus_dm) = Self::calculate_dm(
                current.high.0,
                current.low.0,
                self.prev_high,
                self.prev_low,
            );

            // Apply Wilder smoothing
            self.smoothed_plus_dm = Self::wilder_smooth(self.smoothed_plus_dm, plus_dm, period);
            self.smoothed_minus_dm = Self::wilder_smooth(self.smoothed_minus_dm, minus_dm, period);
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
            self.prev_high = current.high.0;
            self.prev_low = current.low.0;
            self.last_len = len;

            Some(output)
        } else {
            // Same candle - restore committed state and recompute with updated values
            let temp_smoothed_plus_dm = self.committed_smoothed_plus_dm;
            let temp_smoothed_minus_dm = self.committed_smoothed_minus_dm;
            let temp_smoothed_tr = self.committed_smoothed_tr;

            // Get TR for this candle
            self.tr.next(candles)?;
            let tr_current = self.tr.latest().unwrap_or(0.0);

            // Calculate DM with committed prev values
            let current = candles.last().unwrap();
            let (plus_dm, minus_dm) = Self::calculate_dm(
                current.high.0,
                current.low.0,
                self.committed_prev_high,
                self.committed_prev_low,
            );

            // Apply Wilder smoothing
            self.smoothed_plus_dm = Self::wilder_smooth(temp_smoothed_plus_dm, plus_dm, period);
            self.smoothed_minus_dm = Self::wilder_smooth(temp_smoothed_minus_dm, minus_dm, period);
            self.smoothed_tr = Self::wilder_smooth(temp_smoothed_tr, tr_current, period);

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

            // Update last history entry
            if let Some(last) = self.history.last_mut() {
                *last = output;
            }
            self.latest = Some(output);
            self.prev_high = current.high.0;
            self.prev_low = current.low.0;

            Some(output)
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
        self.last_len = 0;
        self.committed_smoothed_plus_dm = 0.0;
        self.committed_smoothed_minus_dm = 0.0;
        self.committed_smoothed_tr = 0.0;
        self.committed_prev_high = 0.0;
        self.committed_prev_low = 0.0;
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

            data.push(create_candle(
                i as i64,
                close,
                high,
                low,
                new_close,
            ));
            close = new_close;
        }

        data
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
        let data: Vec<Ohlcv> = (0..50)
            .map(|i| {
                let base = 100.0 + i as f64 * 2.0; // Strong upward trend
                create_candle(i as i64, base, base + 3.0, base - 1.0, base + 1.5)
            })
            .collect();

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
        let data: Vec<Ohlcv> = (0..50)
            .map(|i| {
                let base = 200.0 - i as f64 * 2.0; // Strong downward trend
                create_candle(i as i64, base, base + 1.0, base - 3.0, base - 1.5)
            })
            .collect();

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
        for i in 1..=data.len() {
            stream_dmi.next(&data[..i]);
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

    #[test]
    fn dmi_same_candle_update() {
        let data = generate_ohlcv_data(20);

        let mut dmi = DMI::new(DMIConfig::new(5));

        // Process all candles
        for i in 1..=data.len() {
            dmi.next(&data[..i]);
        }

        // Record state after all candles
        let original_output = dmi.latest().unwrap();

        // Now simulate same-candle update with modified last candle
        let mut modified_data = data.clone();
        let last_idx = modified_data.len() - 1;
        let last_candle = &modified_data[last_idx];
        modified_data[last_idx] = create_candle(
            last_candle.timestamp.0,
            last_candle.open.0,
            last_candle.high.0 + 2.0, // Higher high
            last_candle.low.0,
            last_candle.close.0 + 1.0, // Higher close
        );

        // Same-candle update (same length)
        dmi.next(&modified_data);
        let modified_output = dmi.latest().unwrap();

        // Values should be different due to the modified candle
        assert!(
            (original_output.plus_di - modified_output.plus_di).abs() > 1e-10
                || (original_output.minus_di - modified_output.minus_di).abs() > 1e-10,
            "DMI should change with modified last candle"
        );

        // Now add a NEW candle (length increases)
        let mut extended_data = modified_data.clone();
        let last = extended_data.last().unwrap();
        extended_data.push(create_candle(
            last.timestamp.0 + 1,
            last.close.0,
            last.close.0 + 3.0,
            last.close.0 - 1.0,
            last.close.0 + 2.0,
        ));

        dmi.next(&extended_data);

        // History should now have one more entry
        assert_eq!(dmi.len(), 21);
    }

    #[test]
    fn dmi_reset() {
        let mut dmi = DMI::new(DMIConfig::new(14));
        let data = generate_ohlcv_data(50);

        dmi.calc(&data).unwrap();
        assert!(dmi.state().is_ready());

        dmi.reset();
        assert!(dmi.state().is_uninitialized());
        assert_eq!(dmi.len(), 0);
        assert!(dmi.latest().is_none());
        assert_eq!(dmi.last_len, 0);
    }
}
