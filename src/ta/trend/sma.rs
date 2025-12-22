//! Simple Moving Average (SMA) indicator.

use crate::ta::{
    config::SMAConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Simple Moving Average indicator.
///
/// Calculates the arithmetic mean of prices over a rolling window.
/// This indicator is stateless regarding window ownership - it computes
/// directly from `&[Ohlcv]` slices, making it suitable for SharedWindow
/// architecture and parallel computation with Rayon.
///
/// # Formula
///
/// SMA = (P1 + P2 + ... + Pn) / n
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{SMA, SMAConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut sma = SMA::new(SMAConfig::new(3));
/// let candles: Vec<Ohlcv> = vec![
///     Ohlcv::from_close(1.0),
///     Ohlcv::from_close(2.0),
///     Ohlcv::from_close(3.0),
///     Ohlcv::from_close(4.0),
///     Ohlcv::from_close(5.0),
/// ];
/// sma.calc(&candles).unwrap();
///
/// assert!(sma.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct SMA {
    config: SMAConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl SMA {
    /// Create a new SMA indicator with the given configuration.
    pub fn new(config: SMAConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Compute SMA from the last N candles of a slice.
    ///
    /// This is the core computation used by both `calc()` and `next()`.
    /// Parallel-safe: only reads from immutable slice.
    #[inline]
    fn compute_from_slice(candles: &[Ohlcv], period: usize) -> f64 {
        let len = candles.len();
        if len < period {
            return f64::NAN;
        }

        let sum: f64 = candles
            .iter()
            .rev()
            .take(period)
            .map(|c| c.close.0)
            .sum();

        sum / period as f64
    }
}

impl Default for SMA {
    fn default() -> Self {
        Self::new(SMAConfig::default())
    }
}

impl Indicator for SMA {
    type Output = f64;
    type Config = SMAConfig;

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

        self.history = Vec::with_capacity(data.len());

        // Fill NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(f64::NAN);
        }

        // Compute initial sum for first SMA
        let mut sum: f64 = data.iter().take(period).map(|c| c.close.0).sum();
        let first_sma = sum / period as f64;
        self.history.push(first_sma);

        // Rolling calculation: subtract oldest, add newest
        for i in period..data.len() {
            sum = sum - data[i - period].close.0 + data[i].close.0;
            let sma = sum / period as f64;
            self.history.push(sma);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let period = self.config.period;

        if candles.len() < period {
            // Not enough data - still warming up
            self.state = self.state.increment(period);
            return None;
        }

        // Compute SMA from last `period` candles
        let sma = Self::compute_from_slice(candles, period);

        // Update state
        self.latest = Some(sma);
        self.history.push(sma);
        self.state = IndicatorState::Ready;

        Some(sma)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for SMA {
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

    /// Helper to build candles from close prices (for testing).
    fn candles_from_closes(closes: &[f64]) -> Vec<Ohlcv> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Ohlcv::new(i as i64, close, close, close, close, 0.0))
            .collect()
    }

    #[test]
    fn sma_batch_calculation() {
        let mut sma = SMA::new(SMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        sma.calc(&candles).unwrap();

        assert!(sma.state().is_ready());
        assert_eq!(sma.len(), 5);

        // First two are NaN
        assert!(sma.get(0).unwrap().is_nan());
        assert!(sma.get(1).unwrap().is_nan());

        // (1+2+3)/3 = 2
        assert!((sma.get(2).unwrap() - 2.0).abs() < 0.0001);
        // (2+3+4)/3 = 3
        assert!((sma.get(3).unwrap() - 3.0).abs() < 0.0001);
        // (3+4+5)/3 = 4
        assert!((sma.get(4).unwrap() - 4.0).abs() < 0.0001);
    }

    #[test]
    fn sma_streaming_next() {
        let mut sma = SMA::new(SMAConfig::new(3));

        // Warmup - not enough candles
        let snap1 = candles_from_closes(&[1.0]);
        assert!(sma.next(&snap1).is_none());

        let snap2 = candles_from_closes(&[1.0, 2.0]);
        assert!(sma.next(&snap2).is_none());

        // Now should be ready
        let snap3 = candles_from_closes(&[1.0, 2.0, 3.0]);
        let result = sma.next(&snap3);
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001); // (1+2+3)/3 = 2

        // Continue streaming
        let snap4 = candles_from_closes(&[1.0, 2.0, 3.0, 4.0]);
        let result = sma.next(&snap4);
        assert!((result.unwrap() - 3.0).abs() < 0.0001); // (2+3+4)/3 = 3
    }

    #[test]
    fn sma_negative_indexing() {
        let mut sma = SMA::new(SMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        sma.calc(&candles).unwrap();

        // -1 should be last value (4.0)
        assert!((sma.get(-1).unwrap() - 4.0).abs() < 0.0001);
        // -2 should be second to last (3.0)
        assert!((sma.get(-2).unwrap() - 3.0).abs() < 0.0001);
    }

    #[test]
    fn sma_insufficient_data() {
        let mut sma = SMA::new(SMAConfig::new(10));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0]);

        let result = sma.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn sma_reset() {
        let mut sma = SMA::new(SMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        sma.calc(&candles).unwrap();

        assert!(sma.state().is_ready());

        sma.reset();

        assert!(sma.state().is_uninitialized());
        assert!(sma.history.is_empty());
        assert!(sma.latest().is_none());
    }

    #[test]
    fn sma_compute_from_slice() {
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Last 3: [3, 4, 5] -> mean = 4
        let sma = SMA::compute_from_slice(&candles, 3);
        assert!((sma - 4.0).abs() < 0.0001);

        // Last 5: [1, 2, 3, 4, 5] -> mean = 3
        let sma = SMA::compute_from_slice(&candles, 5);
        assert!((sma - 3.0).abs() < 0.0001);

        // Insufficient data
        let sma = SMA::compute_from_slice(&candles, 10);
        assert!(sma.is_nan());
    }

    #[test]
    fn sma_next_only_uses_last_n_candles() {
        let mut sma = SMA::new(SMAConfig::new(3));

        // Large snapshot, but SMA only needs last 3
        let candles = candles_from_closes(&[100.0, 200.0, 1.0, 2.0, 3.0]);
        let result = sma.next(&candles);

        // Should compute from last 3: [1, 2, 3] -> mean = 2
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001);
    }

    #[test]
    fn sma_parallel_safe() {
        // Verify next() works with shared data (simulating Rayon usage)
        use std::sync::Arc;

        let candles = Arc::new(candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]));

        let mut sma1 = SMA::new(SMAConfig::new(3));
        let mut sma2 = SMA::new(SMAConfig::new(3));

        // Both can read from the same snapshot
        let snapshot: &[Ohlcv] = &candles;
        let r1 = sma1.next(snapshot);
        let r2 = sma2.next(snapshot);

        assert_eq!(r1, r2);
        assert!((r1.unwrap() - 4.0).abs() < 0.0001);
    }
}
