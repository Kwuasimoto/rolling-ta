//! Hull Moving Average (HMA) indicator.
//!
//! HMA is a fast and smooth moving average that reduces lag significantly
//! while maintaining curve smoothness.

use crate::ta::{
    config::HMAConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Hull Moving Average indicator.
///
/// Composes WMA to create a fast, smooth moving average.
/// This indicator computes directly from `&[Ohlcv]` slices, making it suitable for
/// SharedWindow architecture and parallel computation with Rayon.
///
/// # Formula
///
/// 1. WMA_half = WMA(close, period/2)
/// 2. WMA_full = WMA(close, period)
/// 3. Raw HMA = 2 * WMA_half - WMA_full
/// 4. HMA = WMA(Raw HMA, sqrt(period))
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{HMA, HMAConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut hma = HMA::new(HMAConfig::new(16));
/// let candles: Vec<Ohlcv> = (0..30).map(|i| Ohlcv::from_close(100.0 + i as f64)).collect();
/// hma.calc(&candles).unwrap();
///
/// assert!(hma.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct HMA {
    config: HMAConfig,
    state: IndicatorState,
    half_period: usize,
    sqrt_period: usize,
    weight_sum_full: usize,
    weight_sum_half: usize,
    weight_sum_sqrt: usize,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles.
    last_len: usize,
}

impl HMA {
    /// Create a new HMA indicator.
    pub fn new(config: HMAConfig) -> Self {
        let half_period = config.half_period().max(1);
        let sqrt_period = config.sqrt_period().max(1);

        Self {
            config,
            state: IndicatorState::Uninitialized,
            half_period,
            sqrt_period,
            weight_sum_full: config.period * (config.period + 1) / 2,
            weight_sum_half: half_period * (half_period + 1) / 2,
            weight_sum_sqrt: sqrt_period * (sqrt_period + 1) / 2,
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

    /// Calculate WMA from a slice of close prices.
    #[inline]
    fn wma_from_closes(closes: &[f64], weight_sum: usize) -> f64 {
        let mut weighted_sum = 0.0;
        for (i, &close) in closes.iter().enumerate() {
            weighted_sum += close * (i + 1) as f64;
        }
        weighted_sum / weight_sum as f64
    }

    /// Calculate WMA from a slice of candles (uses close prices).
    #[inline]
    fn wma_from_candles(candles: &[Ohlcv], weight_sum: usize) -> f64 {
        let mut weighted_sum = 0.0;
        for (i, candle) in candles.iter().enumerate() {
            weighted_sum += candle.close.0 * (i + 1) as f64;
        }
        weighted_sum / weight_sum as f64
    }

    /// Calculate HMA value at a specific position in the data.
    /// Requires at least `period` candles ending at `end_idx`.
    fn compute_hma_at(&self, candles: &[Ohlcv], end_idx: usize) -> f64 {
        let period = self.config.period;

        // Calculate WMA_full and WMA_half at this position
        let window_full = &candles[end_idx + 1 - period..=end_idx];
        let wma_full = Self::wma_from_candles(window_full, self.weight_sum_full);

        let window_half = &candles[end_idx + 1 - self.half_period..=end_idx];
        let wma_half = Self::wma_from_candles(window_half, self.weight_sum_half);

        2.0 * wma_half - wma_full
    }

    /// Calculate final HMA from candles slice.
    /// Returns None if insufficient data.
    fn compute_hma(&self, candles: &[Ohlcv]) -> Option<f64> {
        let min_required = self.warmup_period();

        if candles.len() < min_required {
            return None;
        }

        let len = candles.len();

        // Calculate interim values for the last sqrt_period positions
        let mut interim_values = Vec::with_capacity(self.sqrt_period);
        for i in 0..self.sqrt_period {
            let end_idx = len - self.sqrt_period + i;
            let interim = self.compute_hma_at(candles, end_idx);
            interim_values.push(interim);
        }

        // Calculate final HMA as WMA of interim values
        Some(Self::wma_from_closes(&interim_values, self.weight_sum_sqrt))
    }
}

impl Default for HMA {
    fn default() -> Self {
        Self::new(HMAConfig::default())
    }
}

impl Indicator for HMA {
    type Output = f64;
    type Config = HMAConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();
        let period = self.config.period;
        let min_required = self.warmup_period();

        if n < min_required {
            return Err(TAError::InsufficientData {
                required: min_required,
                actual: n,
            });
        }

        // Reset state
        self.history = Vec::with_capacity(n);

        // Fill NaN for warmup period
        let warmup = min_required - 1;
        for _ in 0..warmup {
            self.history.push(f64::NAN);
        }

        // Pre-calculate all WMA values for efficiency
        let mut wma_full_history = vec![f64::NAN; n];
        let mut wma_half_history = vec![f64::NAN; n];

        // WMA_full starts being valid at period - 1
        for i in (period - 1)..n {
            let window = &data[i + 1 - period..=i];
            wma_full_history[i] = Self::wma_from_candles(window, self.weight_sum_full);
        }

        // WMA_half starts being valid at half_period - 1
        for i in (self.half_period - 1)..n {
            let window = &data[i + 1 - self.half_period..=i];
            wma_half_history[i] = Self::wma_from_candles(window, self.weight_sum_half);
        }

        // Calculate interim values: 2 * wma_half - wma_full
        // Valid from period - 1 (when wma_full becomes valid)
        let mut interim = vec![f64::NAN; n];
        for i in (period - 1)..n {
            interim[i] = 2.0 * wma_half_history[i] - wma_full_history[i];
        }

        // Calculate final HMA as WMA of interim with sqrt_period
        // First valid HMA at index: period - 1 + sqrt_period - 1 = warmup
        for i in warmup..n {
            let window_start = i + 1 - self.sqrt_period;
            let window = &interim[window_start..=i];
            let hma = Self::wma_from_closes(window, self.weight_sum_sqrt);
            self.history.push(hma);
        }

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let min_required = self.warmup_period();

        if len < min_required {
            // Not enough data - still warming up
            self.state = self.state.increment(min_required);
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        // Calculate HMA from the slice
        let hma = self.compute_hma(candles)?;

        // Update state
        self.latest = Some(hma);
        self.state = IndicatorState::Ready;

        // Only push to history if this is a new candle
        if is_new_candle {
            self.history.push(hma);
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Update last history entry in place
            *self.history.last_mut().unwrap() = hma;
        }

        Some(hma)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
    }

    fn warmup_period(&self) -> usize {
        self.config.period + self.sqrt_period - 1
    }
}

impl HistoricalIndicator for HMA {
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

    /// Helper to build candles from close prices.
    fn candles_from_closes(closes: &[f64]) -> Vec<Ohlcv> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Ohlcv::new(i as i64, close, close, close, close, 0.0))
            .collect()
    }

    fn generate_trend_data(n: usize) -> Vec<Ohlcv> {
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        candles_from_closes(&closes)
    }

    #[test]
    fn hma_batch_calculation() {
        let mut hma = HMA::new(HMAConfig::new(9));
        let candles = generate_trend_data(30);

        hma.calc(&candles).unwrap();

        assert!(hma.state().is_ready());
        assert_eq!(hma.len(), 30);

        // Warmup period = 9 + 3 - 1 = 11, so first 10 should be NaN
        for i in 0..10 {
            assert!(
                hma.get(i as isize).unwrap().is_nan(),
                "Index {} should be NaN",
                i
            );
        }

        // From index 10, should have valid values
        let value = hma.get(10).unwrap();
        assert!(!value.is_nan(), "Index 10 should be valid");
    }

    #[test]
    fn hma_streaming_next() {
        let mut hma = HMA::new(HMAConfig::new(9));

        // Feed growing snapshots
        let candles = generate_trend_data(30);

        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = hma.next(snapshot);

            // Should get result once we have enough data
            if i >= hma.warmup_period() {
                assert!(result.is_some(), "Should have result at len {}", i);
            }
        }

        assert!(hma.state().is_ready());
        let latest = hma.latest();
        assert!(latest.is_some());
        assert!(!latest.unwrap().is_nan());
    }

    #[test]
    fn hma_faster_than_sma() {
        // HMA should respond faster to price changes
        use crate::ta::trend::SMA;
        use crate::ta::config::SMAConfig;

        let mut hma = HMA::new(HMAConfig::new(9));
        let mut sma = SMA::new(SMAConfig::new(9));

        // Create data with sudden price jump
        let mut closes: Vec<f64> = vec![100.0; 20];
        closes.extend(vec![110.0; 10]);
        let candles = candles_from_closes(&closes);

        hma.calc(&candles).unwrap();
        sma.calc(&candles).unwrap();

        // After the jump, HMA should be closer to 110 than SMA
        let hma_latest = hma.latest().unwrap();
        let sma_latest = sma.latest().unwrap();

        assert!(
            hma_latest > sma_latest,
            "HMA ({}) should respond faster than SMA ({}) to price jump",
            hma_latest,
            sma_latest
        );
    }

    #[test]
    fn hma_batch_vs_streaming_equivalence() {
        let candles = generate_trend_data(30);
        let period = 9;

        // Batch
        let mut batch = HMA::new(HMAConfig::new(period));
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = HMA::new(HMAConfig::new(period));
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        // Compare computed values (skip NaN)
        let batch_computed: Vec<f64> = batch.history().iter().filter(|v| !v.is_nan()).copied().collect();
        let stream_computed = stream.history();

        assert_eq!(batch_computed.len(), stream_computed.len());

        for (b, s) in batch_computed.iter().zip(stream_computed.iter()) {
            assert!(
                (b - s).abs() < 1e-10,
                "Mismatch: batch={}, stream={}",
                b,
                s
            );
        }
    }

    #[test]
    fn hma_reset() {
        let mut hma = HMA::new(HMAConfig::new(9));
        let candles = generate_trend_data(30);

        hma.calc(&candles).unwrap();
        assert!(hma.state().is_ready());

        hma.reset();
        assert!(hma.state().is_uninitialized());
        assert_eq!(hma.len(), 0);
        assert!(hma.latest().is_none());
    }

    #[test]
    fn hma_insufficient_data() {
        let mut hma = HMA::new(HMAConfig::new(16));
        let candles = generate_trend_data(10);

        // period=16, sqrt_period=4, warmup = 16 + 4 - 1 = 19
        let result = hma.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn hma_parallel_safe() {
        use std::sync::Arc;

        let candles = Arc::new(generate_trend_data(30));

        let mut hma1 = HMA::new(HMAConfig::new(9));
        let mut hma2 = HMA::new(HMAConfig::new(9));

        // Both can read from the same snapshot
        let snapshot: &[Ohlcv] = &candles;
        let r1 = hma1.next(snapshot);
        let r2 = hma2.next(snapshot);

        assert_eq!(r1, r2);
        assert!(r1.is_some());
    }
}
