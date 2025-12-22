//! Average Directional Index (ADX).
//!
//! The ADX measures the strength of a trend regardless of direction.
//! It is derived from the DMI indicator's +DI and -DI values.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::ADXConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
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
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut adx = ADX::new(ADXConfig::new(14, 14));
/// let candles: Vec<Ohlcv> = (0..50)
///     .map(|i| Ohlcv::new(i, 100.0 + i as f64, 105.0 + i as f64, 98.0 + i as f64, 103.0 + i as f64, 1000.0))
///     .collect();
/// adx.calc(&candles).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ADX {
    config: ADXConfig,
    state: IndicatorState,
    dmi: DMI,
    adx_value: f64,
    // History
    history: Vec<ADXOutput>,
    latest: Option<ADXOutput>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
    /// Committed state for same-candle updates
    committed_adx_value: f64,
}

impl ADX {
    /// Create a new ADX indicator with the given configuration.
    pub fn new(config: ADXConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            dmi: DMI::new(DMIConfig::new(config.dmi_period)),
            adx_value: 0.0,
            history: Vec::new(),
            latest: None,
            last_len: 0,
            committed_adx_value: 0.0,
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

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
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
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let dmi_period = self.config.dmi_period;
        let adx_period = self.config.adx_period;

        if len == 0 || dmi_period == 0 || adx_period == 0 {
            return None;
        }

        // Need dmi_period + adx_period candles minimum for valid ADX
        let warmup = dmi_period + adx_period;
        if len < warmup {
            return None;
        }

        // Determine if this is a new candle
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First time initialization - compute from scratch via calc()
                if self.dmi.calc(candles).is_err() {
                    return None;
                }
                let dmi_history = self.dmi.history();

                // Calculate DX for all indices
                let mut dx_values = vec![0.0_f64; len];
                for i in dmi_period..len {
                    let dmi_val = &dmi_history[i];
                    dx_values[i] = Self::calculate_dx(dmi_val.plus_di, dmi_val.minus_di);
                }

                // Fill history
                self.history = vec![ADXOutput::default(); len];
                for i in dmi_period..len {
                    let dmi_val = &dmi_history[i];
                    self.history[i] = ADXOutput::new(
                        dmi_val.plus_di,
                        dmi_val.minus_di,
                        dx_values[i],
                        f64::NAN,
                    );
                }

                // Initial ADX: SMA of first adx_period valid DX values
                let adx_start_idx = dmi_period + adx_period - 1;
                let initial_sum: f64 = dx_values[dmi_period..=adx_start_idx].iter().sum();
                self.adx_value = initial_sum / adx_period as f64;
                self.history[adx_start_idx].adx = self.adx_value;

                // Wilder's smoothing for subsequent values
                let weight = (adx_period - 1) as f64;
                for i in (adx_start_idx + 1)..len {
                    // Save committed state BEFORE processing the last candle
                    if i == len - 1 {
                        self.committed_adx_value = self.adx_value;
                    }
                    self.adx_value = (self.adx_value * weight + dx_values[i]) / adx_period as f64;
                    self.history[i].adx = self.adx_value;
                }

                // If we have exactly warmup candles, committed is the initial SMA
                if len == warmup {
                    self.committed_adx_value = self.adx_value;
                }

                self.latest = self.history.last().copied();
                self.last_len = len;
                self.state = IndicatorState::Ready;

                return self.latest;
            }

            // New candle added - save current state as committed
            self.committed_adx_value = self.adx_value;

            // Get DMI for the new candle
            let dmi_result = self.dmi.next(candles)?;

            // Calculate DX
            let dx = Self::calculate_dx(dmi_result.plus_di, dmi_result.minus_di);

            // Wilder's smoothing for ADX
            let weight = (adx_period - 1) as f64;
            self.adx_value = (self.adx_value * weight + dx) / adx_period as f64;

            let output = ADXOutput::new(dmi_result.plus_di, dmi_result.minus_di, dx, self.adx_value);
            self.history.push(output);
            self.latest = Some(output);
            self.last_len = len;

            Some(output)
        } else {
            // Same candle - restore committed state and recompute
            let temp_adx_value = self.committed_adx_value;

            // Get DMI for this candle (same-candle update)
            let dmi_result = self.dmi.next(candles)?;

            // Calculate DX
            let dx = Self::calculate_dx(dmi_result.plus_di, dmi_result.minus_di);

            // Wilder's smoothing for ADX
            let weight = (adx_period - 1) as f64;
            self.adx_value = (temp_adx_value * weight + dx) / adx_period as f64;

            let output = ADXOutput::new(dmi_result.plus_di, dmi_result.minus_di, dx, self.adx_value);

            // Update last history entry
            if let Some(last) = self.history.last_mut() {
                *last = output;
            }
            self.latest = Some(output);

            Some(output)
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.dmi.reset();
        self.adx_value = 0.0;
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
        self.committed_adx_value = 0.0;
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
        let data: Vec<Ohlcv> = (0..60)
            .map(|i| {
                let base = 100.0 + i as f64 * 2.0;
                create_candle(i as i64, base, base + 3.0, base - 1.0, base + 1.5)
            })
            .collect();

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
        for i in 1..=data.len() {
            stream_adx.next(&data[..i]);
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

    #[test]
    fn adx_same_candle_update() {
        let data = generate_ohlcv_data(30);

        let mut adx = ADX::new(ADXConfig::new(5, 5));

        // Process all candles
        for i in 1..=data.len() {
            adx.next(&data[..i]);
        }

        // Record state after all candles
        let original_output = adx.latest().unwrap();

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
        adx.next(&modified_data);
        let modified_output = adx.latest().unwrap();

        // Values should be different due to the modified candle
        assert!(
            (original_output.adx - modified_output.adx).abs() > 1e-10
                || (original_output.plus_di - modified_output.plus_di).abs() > 1e-10
                || (original_output.minus_di - modified_output.minus_di).abs() > 1e-10,
            "ADX should change with modified last candle"
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

        adx.next(&extended_data);

        // History should now have one more entry
        assert_eq!(adx.len(), 31);
    }

    #[test]
    fn adx_reset() {
        let mut adx = ADX::new(ADXConfig::new(14, 14));
        let data = generate_ohlcv_data(50);

        adx.calc(&data).unwrap();
        assert!(adx.state().is_ready());

        adx.reset();
        assert!(adx.state().is_uninitialized());
        assert_eq!(adx.len(), 0);
        assert!(adx.latest().is_none());
        assert_eq!(adx.last_len, 0);
    }
}
