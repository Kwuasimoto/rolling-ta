//! Average Directional Index (ADX).
//!
//! The ADX measures the strength of a trend regardless of direction.
//! It is derived from the DMI indicator's +DI and -DI values.

use crate::ta::{
    config::ADXConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    HistoricalIndicator, Indicator,
};

use super::dmi::DMI;
use super::DMIConfig;

/// Output from ADX indicator containing +DI, -DI, DX, and ADX.
#[derive(Debug, Clone, Copy, Default)]
pub struct ADXOutput {
    pub plus_di: f64,
    pub minus_di: f64,
    pub dx: f64,
    pub adx: f64,
}

impl ADXOutput {
    pub fn new(plus_di: f64, minus_di: f64, dx: f64, adx: f64) -> Self {
        Self {
            plus_di,
            minus_di,
            dx,
            adx,
        }
    }
}

/// Average Directional Index indicator.
///
/// Calculates ADX (trend strength) by smoothing the DX values derived from DMI.
///
/// # Formula
///
/// DX = |+DI - -DI| / (+DI + -DI) * 100
///
/// First ADX = SMA of first `adx_period` DX values (after DMI warmup)
/// Subsequent ADX = (prev_ADX * (adx_period - 1) + current_DX) / adx_period
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{ADX, ADXConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut adx = ADX::new(ADXConfig::new(14, 14));
/// let data = OhlcvSeries::from_tuples(&[
///     (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     // ... many more candles needed for ADX
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct ADX {
    config: ADXConfig,
    state: IndicatorState,
    dmi: DMI,
    adx_value: f64,
    // For streaming: accumulate DX values for initial SMA
    dx_sum: f64,
    dx_count: usize,
    // History
    history: Vec<ADXOutput>,
    latest: Option<ADXOutput>,
}

impl ADX {
    /// Create a new ADX indicator with the given configuration.
    pub fn new(config: ADXConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            dmi: DMI::new(DMIConfig::new(config.dmi_period)),
            adx_value: 0.0,
            dx_sum: 0.0,
            dx_count: 0,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the DMI period.
    #[inline]
    pub fn dmi_period(&self) -> usize {
        self.config.dmi_period
    }

    /// Get the ADX period.
    #[inline]
    pub fn adx_period(&self) -> usize {
        self.config.adx_period
    }

    /// Calculate DX from +DI and -DI.
    #[inline]
    fn calculate_dx(plus_di: f64, minus_di: f64) -> f64 {
        let sum = plus_di + minus_di;
        if sum == 0.0 {
            0.0
        } else {
            ((plus_di - minus_di).abs() / sum) * 100.0
        }
    }

    /// Get a reference to the internal DMI indicator.
    #[inline]
    pub fn dmi(&self) -> &DMI {
        &self.dmi
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

    /// Get DX latest value.
    #[inline]
    pub fn dx_latest(&self) -> Option<f64> {
        self.latest.map(|o| o.dx)
    }

    /// Get ADX latest value.
    #[inline]
    pub fn adx_latest(&self) -> Option<f64> {
        self.latest.map(|o| o.adx)
    }
}

impl Default for ADX {
    fn default() -> Self {
        Self::new(ADXConfig::default())
    }
}

impl Indicator for ADX {
    type Output = ADXOutput;
    type Config = ADXConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let dmi_period = self.config.dmi_period;
        let adx_period = self.config.adx_period;
        let n = data.len();

        // First valid ADX at index = dmi_period + adx_period - 1
        let warmup = dmi_period + adx_period;
        if n < warmup {
            return Err(TAError::InsufficientData {
                required: warmup,
                actual: n,
            });
        }

        if dmi_period == 0 || adx_period == 0 {
            return Err(TAError::InvalidPeriod(0));
        }

        // Calculate DMI first
        self.dmi.calc(data)?;
        let dmi_history = self.dmi.history();

        // Reset state
        self.history = vec![ADXOutput::default(); n];

        // Calculate DX for all indices and populate +DI, -DI, DX even before ADX is ready
        let mut dx_values = vec![0.0_f64; n];
        for i in dmi_period..n {
            let dmi_val = &dmi_history[i];
            dx_values[i] = Self::calculate_dx(dmi_val.plus_di, dmi_val.minus_di);
            // Store +DI, -DI, DX (with ADX still NaN/0)
            self.history[i] = ADXOutput::new(
                dmi_val.plus_di,
                dmi_val.minus_di,
                dx_values[i],
                f64::NAN, // ADX not yet ready
            );
        }

        // Initial ADX: SMA of first adx_period valid DX values
        // Valid DX starts at index dmi_period
        let adx_start_idx = dmi_period + adx_period - 1;
        let initial_sum: f64 = dx_values[dmi_period..=adx_start_idx].iter().sum();
        self.adx_value = initial_sum / adx_period as f64;

        // Update first valid ADX
        self.history[adx_start_idx].adx = self.adx_value;

        // Wilder's smoothing for subsequent values
        let weight = (adx_period - 1) as f64;
        for i in (adx_start_idx + 1)..n {
            self.adx_value = (self.adx_value * weight + dx_values[i]) / adx_period as f64;
            self.history[i].adx = self.adx_value;
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let dmi_period = self.config.dmi_period;
        let adx_period = self.config.adx_period;

        // Update DMI first
        let dmi_result = self.dmi.update(tick)?;

        if self.state.is_uninitialized() {
            // First tick - DMI will start warming up
            self.history.push(ADXOutput::default());
            self.state = IndicatorState::Warming { count: 1 };
            return Ok(None);
        }

        // Get current DMI values
        let dmi_val = self.dmi.latest().unwrap_or_default();

        match self.state {
            IndicatorState::Warming { count } => {
                // DMI has its own warmup, so wait for it
                if dmi_result.is_none() || count < dmi_period {
                    // Still waiting for DMI
                    self.history.push(ADXOutput::default());
                    self.state = IndicatorState::Warming { count: count + 1 };
                    return Ok(None);
                }

                // DMI is ready, now accumulate DX for ADX warmup
                let dx = Self::calculate_dx(dmi_val.plus_di, dmi_val.minus_di);
                self.dx_sum += dx;
                self.dx_count += 1;

                if self.dx_count >= adx_period {
                    // Ready to output first ADX
                    self.adx_value = self.dx_sum / adx_period as f64;
                    let output = ADXOutput::new(dmi_val.plus_di, dmi_val.minus_di, dx, self.adx_value);
                    self.history.push(output);
                    self.latest = Some(output);
                    self.state = IndicatorState::Ready;
                    return Ok(Some(output));
                } else {
                    // Store +DI, -DI, DX even when ADX not ready yet (use NaN for ADX)
                    let output = ADXOutput::new(dmi_val.plus_di, dmi_val.minus_di, dx, f64::NAN);
                    self.history.push(output);
                    self.state = IndicatorState::Warming { count: count + 1 };
                    return Ok(None);
                }
            }
            IndicatorState::Ready => {
                // Calculate DX
                let dx = Self::calculate_dx(dmi_val.plus_di, dmi_val.minus_di);

                // Wilder's smoothing for ADX
                let weight = (adx_period - 1) as f64;
                self.adx_value = (self.adx_value * weight + dx) / adx_period as f64;

                let output = ADXOutput::new(dmi_val.plus_di, dmi_val.minus_di, dx, self.adx_value);
                self.history.push(output);
                self.latest = Some(output);

                Ok(Some(output))
            }
            IndicatorState::Uninitialized => unreachable!(),
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.dmi.reset();
        self.adx_value = 0.0;
        self.dx_sum = 0.0;
        self.dx_count = 0;
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.dmi_period + self.config.adx_period
    }
}

impl HistoricalIndicator for ADX {
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
    fn adx_batch_calculation() {
        let mut adx = ADX::new(ADXConfig::new(14, 14));
        let data = generate_ohlcv_data(50);

        adx.calc(&data).unwrap();

        assert!(adx.state().is_ready());
        assert_eq!(adx.len(), 50);

        // First valid +DI/-DI/DX at index 14 (dmi_period)
        // ADX should be NaN before index 27
        for i in 0..14 {
            let val = adx.get(i as isize).unwrap();
            assert_eq!(val.plus_di, 0.0, "Index {} +DI should be 0", i);
            assert_eq!(val.minus_di, 0.0, "Index {} -DI should be 0", i);
        }

        // +DI/-DI/DX valid from index 14
        let val14 = adx.get(14).unwrap();
        assert!(val14.plus_di >= 0.0, "+DI at index 14 should be >= 0");
        assert!(val14.minus_di >= 0.0, "-DI at index 14 should be >= 0");
        assert!(val14.adx.is_nan(), "ADX at index 14 should be NaN");

        // ADX should be valid from index 27 onwards (14 + 14 - 1)
        let val27 = adx.get(27).unwrap();
        assert!(!val27.adx.is_nan(), "ADX at index 27 should not be NaN");
        assert!(val27.adx > 0.0, "ADX at index 27 should be > 0, got {}", val27.adx);
        assert!(
            val27.adx <= 100.0,
            "ADX should be <= 100, got {}",
            val27.adx
        );
    }

    #[test]
    fn adx_values_in_range() {
        let mut adx = ADX::new(ADXConfig::new(14, 14));
        let data = generate_ohlcv_data(100);

        adx.calc(&data).unwrap();

        // ADX valid from index 27 onwards
        for output in adx.history().iter().skip(27) {
            assert!(
                !output.adx.is_nan() && output.adx >= 0.0 && output.adx <= 100.0,
                "ADX should be in [0, 100], got {}",
                output.adx
            );
        }

        // +DI/-DI/DX valid from index 14 onwards
        for output in adx.history().iter().skip(14) {
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
            assert!(
                output.dx >= 0.0 && output.dx <= 100.0,
                "DX should be in [0, 100], got {}",
                output.dx
            );
        }
    }

    #[test]
    fn adx_strong_trend() {
        // Create strong trending data
        let mut data_vec = Vec::new();
        for i in 0..60 {
            let base = 100.0 + i as f64 * 2.0;
            data_vec.push((i as i64, base, base + 3.0, base - 1.0, base + 1.5, 1000.0));
        }
        let data = OhlcvSeries::from_tuples(&data_vec);

        let mut adx = ADX::new(ADXConfig::new(14, 14));
        adx.calc(&data).unwrap();

        // In a strong trend, ADX should be relatively high
        let last = adx.latest().unwrap();
        assert!(
            last.adx > 20.0,
            "ADX in strong trend should be > 20, got {}",
            last.adx
        );
    }

    #[test]
    fn adx_dx_calculation() {
        // Test DX formula directly
        // DX = |+DI - -DI| / (+DI + -DI) * 100

        // Case 1: +DI = 30, -DI = 10
        let dx1 = ADX::calculate_dx(30.0, 10.0);
        assert!((dx1 - 50.0).abs() < 0.001, "Expected DX=50, got {}", dx1);

        // Case 2: +DI = 10, -DI = 30
        let dx2 = ADX::calculate_dx(10.0, 30.0);
        assert!((dx2 - 50.0).abs() < 0.001, "Expected DX=50, got {}", dx2);

        // Case 3: +DI = 25, -DI = 25
        let dx3 = ADX::calculate_dx(25.0, 25.0);
        assert!((dx3 - 0.0).abs() < 0.001, "Expected DX=0, got {}", dx3);

        // Case 4: Both zero
        let dx4 = ADX::calculate_dx(0.0, 0.0);
        assert!((dx4 - 0.0).abs() < 0.001, "Expected DX=0, got {}", dx4);
    }

    #[test]
    fn adx_streaming_matches_batch() {
        let data = generate_ohlcv_data(50);

        // Batch calculation
        let mut batch_adx = ADX::new(ADXConfig::new(5, 5));
        batch_adx.calc(&data).unwrap();

        // Streaming calculation
        let mut stream_adx = ADX::new(ADXConfig::new(5, 5));
        for i in 0..data.len() {
            let tick = data.get(i).unwrap();
            stream_adx.update(&tick).unwrap();
        }

        // Compare results (allow small floating point differences)
        let epsilon = 1e-6;
        for i in 0..batch_adx.len() {
            let batch_val = batch_adx.get(i as isize).unwrap();
            let stream_val = stream_adx.get(i as isize).unwrap();

            // Compare ADX (handle NaN)
            if batch_val.adx.is_nan() && stream_val.adx.is_nan() {
                // Both NaN is OK
            } else if batch_val.adx.is_nan() || stream_val.adx.is_nan() {
                panic!(
                    "Index {}: batch ADX ({}) and stream ADX ({}) NaN mismatch",
                    i, batch_val.adx, stream_val.adx
                );
            } else {
                assert!(
                    (batch_val.adx - stream_val.adx).abs() < epsilon,
                    "Index {}: batch ADX ({}) != stream ADX ({})",
                    i,
                    batch_val.adx,
                    stream_val.adx
                );
            }

            // Compare +DI, -DI, DX (should always match)
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
            assert!(
                (batch_val.dx - stream_val.dx).abs() < epsilon,
                "Index {}: batch DX ({}) != stream DX ({})",
                i,
                batch_val.dx,
                stream_val.dx
            );
        }
    }
}
