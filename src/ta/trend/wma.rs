//! Weighted Moving Average (WMA) indicator.

use crate::ta::{
    config::WMAConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Weighted Moving Average indicator.
///
/// Assigns linearly increasing weights to more recent prices.
/// This indicator computes directly from `&[Ohlcv]` slices, making it suitable for
/// SharedWindow architecture and parallel computation with Rayon.
///
/// # Formula
///
/// WMA = (P1×1 + P2×2 + ... + Pn×n) / (1 + 2 + ... + n)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{WMA, WMAConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut wma = WMA::new(WMAConfig::new(3));
/// let candles: Vec<Ohlcv> = vec![
///     Ohlcv::from_close(1.0),
///     Ohlcv::from_close(2.0),
///     Ohlcv::from_close(3.0),
///     Ohlcv::from_close(4.0),
///     Ohlcv::from_close(5.0),
/// ];
/// wma.calc(&candles).unwrap();
///
/// assert!(wma.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct WMA {
    config: WMAConfig,
    state: IndicatorState,
    weight_sum: usize,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles.
    last_len: usize,
}

impl WMA {
    /// Create a new WMA indicator with the given configuration.
    pub fn new(config: WMAConfig) -> Self {
        let weight_sum = config.weight_sum();

        Self {
            config,
            state: IndicatorState::Uninitialized,
            weight_sum,
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

    /// Get the weight sum.
    #[inline]
    pub fn weight_sum(&self) -> usize {
        self.weight_sum
    }

    /// Calculate WMA from the last N candles in a slice.
    #[inline]
    fn compute_wma(candles: &[Ohlcv], period: usize, weight_sum: usize) -> f64 {
        let start = candles.len().saturating_sub(period);
        let mut weighted_sum = 0.0;
        for (i, candle) in candles[start..].iter().enumerate() {
            weighted_sum += candle.close.0 * (i + 1) as f64;
        }
        weighted_sum / weight_sum as f64
    }
}

impl Default for WMA {
    fn default() -> Self {
        Self::new(WMAConfig::default())
    }
}

impl Indicator for WMA {
    type Output = f64;
    type Config = WMAConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;

        if data.len() < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: data.len(),
            });
        }

        if period == 0 {
            return Err(TAError::InvalidPeriod(0));
        }

        // Reset state
        self.history = Vec::with_capacity(data.len());

        // Fill NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(f64::NAN);
        }

        // Calculate WMA for each position
        for i in (period - 1)..data.len() {
            let window = &data[i + 1 - period..=i];
            let mut weighted_sum = 0.0;
            for (j, candle) in window.iter().enumerate() {
                weighted_sum += candle.close.0 * (j + 1) as f64;
            }
            let wma = weighted_sum / self.weight_sum as f64;
            self.history.push(wma);
        }

        self.latest = self.history.last().copied();
        self.last_len = data.len();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let period = self.config.period;
        let len = candles.len();

        if len < period {
            // Not enough data - still warming up
            self.state = self.state.increment(period);
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        // Calculate WMA from last N candles
        let wma = Self::compute_wma(candles, period, self.weight_sum);

        // Update state
        self.latest = Some(wma);
        self.state = IndicatorState::Ready;

        // Only push to history if this is a new candle
        if is_new_candle {
            self.history.push(wma);
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Update last history entry in place
            *self.history.last_mut().unwrap() = wma;
        }

        Some(wma)
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
        self.config.period
    }
}

impl HistoricalIndicator for WMA {
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

    #[test]
    fn wma_batch_calculation() {
        let mut wma = WMA::new(WMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        wma.calc(&candles).unwrap();

        assert!(wma.state().is_ready());
        assert_eq!(wma.len(), 5);

        // First two are NaN
        assert!(wma.get(0).unwrap().is_nan());
        assert!(wma.get(1).unwrap().is_nan());

        // Weight sum for period 3: 1+2+3 = 6
        // Index 2: (1×1 + 2×2 + 3×3) / 6 = (1+4+9)/6 = 14/6 ≈ 2.333
        assert!((wma.get(2).unwrap() - 2.333).abs() < 0.01);

        // Index 3: (2×1 + 3×2 + 4×3) / 6 = (2+6+12)/6 = 20/6 ≈ 3.333
        assert!((wma.get(3).unwrap() - 3.333).abs() < 0.01);

        // Index 4: (3×1 + 4×2 + 5×3) / 6 = (3+8+15)/6 = 26/6 ≈ 4.333
        assert!((wma.get(4).unwrap() - 4.333).abs() < 0.01);
    }

    #[test]
    fn wma_streaming_next() {
        let mut wma = WMA::new(WMAConfig::new(3));

        // Warmup - not enough candles
        let snap1 = candles_from_closes(&[1.0]);
        assert!(wma.next(&snap1).is_none());

        let snap2 = candles_from_closes(&[1.0, 2.0]);
        assert!(wma.next(&snap2).is_none());

        // Now should be ready - first WMA
        let snap3 = candles_from_closes(&[1.0, 2.0, 3.0]);
        let result = wma.next(&snap3);
        assert!(result.is_some());
        // WMA = (1×1 + 2×2 + 3×3) / 6 = 14/6 ≈ 2.333
        assert!((result.unwrap() - 2.333).abs() < 0.01);

        // Continue streaming
        let snap4 = candles_from_closes(&[1.0, 2.0, 3.0, 4.0]);
        let result = wma.next(&snap4);
        // WMA = (2×1 + 3×2 + 4×3) / 6 = 20/6 ≈ 3.333
        assert!((result.unwrap() - 3.333).abs() < 0.01);
    }

    #[test]
    fn wma_weight_sum_calculation() {
        let wma = WMA::new(WMAConfig::new(5));
        // 1+2+3+4+5 = 15
        assert_eq!(wma.weight_sum(), 15);
    }

    #[test]
    fn wma_reset() {
        let mut wma = WMA::new(WMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        wma.calc(&candles).unwrap();

        assert!(wma.state().is_ready());

        wma.reset();

        assert!(wma.state().is_uninitialized());
        assert!(wma.history.is_empty());
        assert!(wma.latest().is_none());
    }

    #[test]
    fn wma_negative_indexing() {
        let mut wma = WMA::new(WMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        wma.calc(&candles).unwrap();

        // -1 should be last value (4.333)
        assert!((wma.get(-1).unwrap() - 4.333).abs() < 0.01);
        // -2 should be second to last (3.333)
        assert!((wma.get(-2).unwrap() - 3.333).abs() < 0.01);
    }

    #[test]
    fn wma_insufficient_data() {
        let mut wma = WMA::new(WMAConfig::new(10));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0]);

        let result = wma.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn wma_next_uses_last_n_candles() {
        let mut wma = WMA::new(WMAConfig::new(3));

        // Large snapshot, WMA should use last 3 candles
        let candles = candles_from_closes(&[100.0, 200.0, 1.0, 2.0, 3.0]);

        // WMA of last 3: (1×1 + 2×2 + 3×3) / 6 = 14/6 ≈ 2.333
        let result = wma.next(&candles);
        assert!(result.is_some());
        assert!((result.unwrap() - 2.333).abs() < 0.01);
    }

    #[test]
    fn wma_parallel_safe() {
        use std::sync::Arc;

        let candles = Arc::new(candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]));

        let mut wma1 = WMA::new(WMAConfig::new(3));
        let mut wma2 = WMA::new(WMAConfig::new(3));

        // Both can read from the same snapshot
        let snapshot: &[Ohlcv] = &candles;
        let r1 = wma1.next(snapshot);
        let r2 = wma2.next(snapshot);

        assert_eq!(r1, r2);
        // WMA of last 3: (3×1 + 4×2 + 5×3) / 6 = 26/6 ≈ 4.333
        assert!((r1.unwrap() - 4.333).abs() < 0.01);
    }

    #[test]
    fn wma_batch_vs_streaming_equivalence() {
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let period = 3;

        // Batch
        let mut batch = WMA::new(WMAConfig::new(period));
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = WMA::new(WMAConfig::new(period));
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
}