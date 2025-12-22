//! Donchian Channels indicator.
//!
//! Donchian Channels consist of an upper band (highest high), lower band (lowest low),
//! and middle band (midpoint), used to identify breakouts and trend direction.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::DonchianConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Donchian Channels output values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DonchianOutput {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

impl Default for DonchianOutput {
    fn default() -> Self {
        Self {
            upper: f64::NAN,
            middle: f64::NAN,
            lower: f64::NAN,
        }
    }
}

/// Donchian Channels indicator.
///
/// Identifies the highest high and lowest low over a specified period.
///
/// # Formula
///
/// - Upper Band = highest high over N periods
/// - Lower Band = lowest low over N periods
/// - Middle Band = (Upper + Lower) / 2
///
/// # Interpretation
///
/// - Price touching upper band suggests bullish momentum
/// - Price touching lower band suggests bearish momentum
/// - Channel width indicates volatility
/// - Breakouts above/below bands can signal trend continuation
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volatility::{Donchian, DonchianConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut donchian = Donchian::new(DonchianConfig::new(20));
/// let candles: Vec<Ohlcv> = (0..25).map(|i| {
///     let base = 100.0 + (i as f64 * 0.5).sin() * 5.0;
///     Ohlcv::new(i as i64, base, base + 2.0, base - 1.0, base + 0.5, 1000.0)
/// }).collect();
/// donchian.calc(&candles).unwrap();
///
/// assert!(donchian.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct Donchian {
    config: DonchianConfig,
    state: IndicatorState,
    history: Vec<DonchianOutput>,
    latest: Option<DonchianOutput>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl Donchian {
    /// Create a new Donchian Channels indicator.
    pub fn new(config: DonchianConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
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

    /// Compute highest high from slice of candles.
    #[inline]
    fn highest_high(candles: &[Ohlcv]) -> f64 {
        candles
            .iter()
            .map(|c| c.high.0)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Compute lowest low from slice of candles.
    #[inline]
    fn lowest_low(candles: &[Ohlcv]) -> f64 {
        candles
            .iter()
            .map(|c| c.low.0)
            .fold(f64::INFINITY, f64::min)
    }

    /// Compute Donchian output from slice of candles.
    #[inline]
    fn compute_from_candles(candles: &[Ohlcv]) -> DonchianOutput {
        let upper = Self::highest_high(candles);
        let lower = Self::lowest_low(candles);
        let middle = (upper + lower) / 2.0;

        DonchianOutput {
            upper,
            middle,
            lower,
        }
    }
}

impl Default for Donchian {
    fn default() -> Self {
        Self::new(DonchianConfig::default())
    }
}

impl Indicator for Donchian {
    type Output = DonchianOutput;
    type Config = DonchianConfig;

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

        // Reset state
        self.history = Vec::with_capacity(n);

        // Fill with NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(DonchianOutput::default());
        }

        // Calculate Donchian from period-1 onwards
        for i in (period - 1)..n {
            let window = &data[(i + 1 - period)..=i];
            let output = Self::compute_from_candles(window);
            self.history.push(output);
        }

        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < period {
            // Not enough data yet
            if len > self.last_len || self.last_len == 0 {
                self.history.push(DonchianOutput::default());
                self.last_len = len;
                self.state = IndicatorState::Warming { count: len };
            }
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        // Extract last `period` candles
        let window = &candles[(len - period)..];
        let output = Self::compute_from_candles(window);

        if is_new_candle {
            self.history.push(output);
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Same candle - update last value
            *self.history.last_mut().unwrap() = output;
        }

        self.latest = Some(output);
        self.state = IndicatorState::Ready;

        Some(output)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for Donchian {
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

    fn candles_with_range(data: &[(f64, f64, f64)]) -> Vec<Ohlcv> {
        // (high, low, close)
        data.iter()
            .enumerate()
            .map(|(i, &(high, low, close))| {
                let open = (high + low) / 2.0;
                Ohlcv::new(i as i64, open, high, low, close, 1000.0)
            })
            .collect()
    }

    #[test]
    fn donchian_batch_calculation() {
        let mut donchian = Donchian::new(DonchianConfig::new(5));
        // Create candles with known highs and lows
        let candles = candles_with_range(&[
            (105.0, 95.0, 100.0),
            (108.0, 98.0, 103.0),
            (110.0, 100.0, 105.0),
            (107.0, 97.0, 102.0),
            (112.0, 102.0, 108.0), // period 5: high=112, low=95
            (109.0, 99.0, 104.0),  // period 5: high=112, low=97
            (115.0, 105.0, 110.0), // period 5: high=115, low=97
        ]);

        donchian.calc(&candles).unwrap();

        assert!(donchian.state().is_ready());
        assert_eq!(donchian.len(), 7);

        // First 4 should be NaN
        for i in 0..4 {
            let output = donchian.get(i as isize).unwrap();
            assert!(output.upper.is_nan());
            assert!(output.middle.is_nan());
            assert!(output.lower.is_nan());
        }

        // At index 4: highest high = 112, lowest low = 95
        let output4 = donchian.get(4).unwrap();
        assert!((output4.upper - 112.0).abs() < 0.0001);
        assert!((output4.lower - 95.0).abs() < 0.0001);
        assert!((output4.middle - 103.5).abs() < 0.0001); // (112 + 95) / 2

        // At index 6: highest high = 115, lowest low = 97
        let output6 = donchian.get(6).unwrap();
        assert!((output6.upper - 115.0).abs() < 0.0001);
        assert!((output6.lower - 97.0).abs() < 0.0001);
    }

    #[test]
    fn donchian_streaming_next() {
        let mut donchian = Donchian::new(DonchianConfig::new(3));
        let candles = candles_with_range(&[
            (105.0, 95.0, 100.0),
            (108.0, 92.0, 103.0),
            (110.0, 98.0, 105.0),
            (107.0, 96.0, 102.0),
            (112.0, 100.0, 108.0),
        ]);

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = donchian.next(snapshot);

            // Should get result when we have period = 3 candles
            if i >= 3 {
                assert!(result.is_some(), "Should have result at len {}", i);
                let output = result.unwrap();
                assert!(output.upper > output.lower);
                assert!(output.middle > output.lower);
                assert!(output.middle < output.upper);
            }
        }

        assert!(donchian.state().is_ready());
    }

    #[test]
    fn donchian_batch_vs_streaming_equivalence() {
        let candles = candles_with_range(&[
            (105.0, 95.0, 100.0),
            (108.0, 92.0, 103.0),
            (110.0, 98.0, 105.0),
            (107.0, 96.0, 102.0),
            (112.0, 100.0, 108.0),
            (109.0, 97.0, 104.0),
            (115.0, 103.0, 110.0),
            (111.0, 99.0, 106.0),
        ]);
        let config = DonchianConfig::new(4);

        // Batch
        let mut batch = Donchian::new(config);
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = Donchian::new(config);
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        // Compare valid values
        let batch_valid: Vec<_> = batch
            .history()
            .iter()
            .filter(|o| !o.middle.is_nan())
            .collect();
        let stream_valid: Vec<_> = stream
            .history()
            .iter()
            .filter(|o| !o.middle.is_nan())
            .collect();

        assert_eq!(
            batch_valid.len(),
            stream_valid.len(),
            "batch={}, stream={}",
            batch_valid.len(),
            stream_valid.len()
        );

        for (i, (b, s)) in batch_valid.iter().zip(stream_valid.iter()).enumerate() {
            assert!(
                (b.upper - s.upper).abs() < 1e-10,
                "Upper mismatch at {}: batch={}, stream={}",
                i,
                b.upper,
                s.upper
            );
            assert!(
                (b.middle - s.middle).abs() < 1e-10,
                "Middle mismatch at {}: batch={}, stream={}",
                i,
                b.middle,
                s.middle
            );
            assert!(
                (b.lower - s.lower).abs() < 1e-10,
                "Lower mismatch at {}: batch={}, stream={}",
                i,
                b.lower,
                s.lower
            );
        }
    }

    #[test]
    fn donchian_channels_widen_with_volatility() {
        let mut donchian_stable = Donchian::new(DonchianConfig::new(5));
        let mut donchian_volatile = Donchian::new(DonchianConfig::new(5));

        // Stable prices - narrow range
        let stable = candles_with_range(&[
            (101.0, 99.0, 100.0),
            (101.0, 99.0, 100.0),
            (101.0, 99.0, 100.0),
            (101.0, 99.0, 100.0),
            (101.0, 99.0, 100.0),
        ]);

        // Volatile prices - wide range
        let volatile = candles_with_range(&[
            (120.0, 80.0, 100.0),
            (115.0, 85.0, 100.0),
            (125.0, 75.0, 100.0),
            (118.0, 82.0, 100.0),
            (122.0, 78.0, 100.0),
        ]);

        donchian_stable.calc(&stable).unwrap();
        donchian_volatile.calc(&volatile).unwrap();

        let stable_output = donchian_stable.latest().unwrap();
        let volatile_output = donchian_volatile.latest().unwrap();

        let stable_width = stable_output.upper - stable_output.lower;
        let volatile_width = volatile_output.upper - volatile_output.lower;

        assert!(
            volatile_width > stable_width,
            "Volatile channels ({}) should be wider than stable channels ({})",
            volatile_width,
            stable_width
        );
    }

    #[test]
    fn donchian_breakout_detection() {
        let mut donchian = Donchian::new(DonchianConfig::new(5));
        let candles = candles_with_range(&[
            (105.0, 95.0, 100.0),
            (106.0, 96.0, 101.0),
            (104.0, 94.0, 99.0),
            (107.0, 97.0, 102.0),
            (105.0, 95.0, 100.0),
            (115.0, 105.0, 112.0), // Breakout candle - close > previous upper
        ]);

        donchian.calc(&candles).unwrap();

        // At index 4, upper was 107
        let before_breakout = donchian.get(4).unwrap();
        assert!((before_breakout.upper - 107.0).abs() < 0.0001);

        // At index 5, the close (112) > previous upper (107)
        // New upper should be 115
        let after_breakout = donchian.get(5).unwrap();
        assert!((after_breakout.upper - 115.0).abs() < 0.0001);
    }

    #[test]
    fn donchian_reset() {
        let mut donchian = Donchian::new(DonchianConfig::new(5));
        let candles = candles_with_range(&[
            (105.0, 95.0, 100.0),
            (108.0, 98.0, 103.0),
            (110.0, 100.0, 105.0),
            (107.0, 97.0, 102.0),
            (112.0, 102.0, 108.0),
        ]);

        donchian.calc(&candles).unwrap();
        assert!(donchian.state().is_ready());
        assert_eq!(donchian.len(), 5);

        donchian.reset();
        assert!(donchian.state().is_uninitialized());
        assert_eq!(donchian.len(), 0);
    }

    #[test]
    fn donchian_insufficient_data() {
        let mut donchian = Donchian::new(DonchianConfig::new(20));
        let candles = candles_with_range(&[(105.0, 95.0, 100.0); 10]);

        let result = donchian.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn donchian_upper_always_greater_than_lower() {
        let mut donchian = Donchian::new(DonchianConfig::new(5));
        let candles = candles_with_range(&[
            (105.0, 95.0, 100.0),
            (108.0, 92.0, 103.0),
            (110.0, 88.0, 105.0),
            (107.0, 91.0, 102.0),
            (112.0, 90.0, 108.0),
            (109.0, 93.0, 104.0),
            (115.0, 87.0, 110.0),
        ]);

        donchian.calc(&candles).unwrap();

        for output in donchian.history().iter().filter(|o| !o.upper.is_nan()) {
            assert!(
                output.upper >= output.lower,
                "Upper ({}) should be >= lower ({})",
                output.upper,
                output.lower
            );
        }
    }
}
