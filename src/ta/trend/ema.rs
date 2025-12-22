//! Exponential Moving Average (EMA) indicator.

use crate::ta::{
    config::EMAConfig,
    error::{TAError, TAResult},
    math::{ema_multiplier, ema_step},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Exponential Moving Average indicator.
///
/// Gives more weight to recent prices, making it more responsive to new information.
/// This indicator computes directly from `&[Ohlcv]` slices, making it suitable for
/// SharedWindow architecture and parallel computation with Rayon.
///
/// # Formula
///
/// EMA = (Price - Previous EMA) * Multiplier + Previous EMA
/// Multiplier = 2 / (Period + 1)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{EMA, EMAConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut ema = EMA::new(EMAConfig::new(3));
/// let candles: Vec<Ohlcv> = vec![
///     Ohlcv::from_close(1.0),
///     Ohlcv::from_close(2.0),
///     Ohlcv::from_close(3.0),
///     Ohlcv::from_close(4.0),
///     Ohlcv::from_close(5.0),
/// ];
/// ema.calc(&candles).unwrap();
///
/// assert!(ema.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct EMA {
    config: EMAConfig,
    state: IndicatorState,
    multiplier: f64,
    /// Previous EMA value for incremental calculation.
    prev_ema: Option<f64>,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles.
    last_len: usize,
}

impl EMA {
    /// Create a new EMA indicator with the given configuration.
    pub fn new(config: EMAConfig) -> Self {
        let multiplier = ema_multiplier(config.period);

        Self {
            config,
            state: IndicatorState::Uninitialized,
            multiplier,
            prev_ema: None,
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

    /// Get the EMA multiplier.
    #[inline]
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Compute initial SMA from the last N candles (for EMA seeding).
    #[inline]
    fn compute_initial_sma(candles: &[Ohlcv], period: usize) -> f64 {
        let sum: f64 = candles
            .iter()
            .rev()
            .take(period)
            .map(|c| c.close.0)
            .sum();
        sum / period as f64
    }
}

impl Default for EMA {
    fn default() -> Self {
        Self::new(EMAConfig::default())
    }
}

impl Indicator for EMA {
    type Output = f64;
    type Config = EMAConfig;

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
        self.prev_ema = None;

        // Fill NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(f64::NAN);
        }

        // Calculate initial SMA for seeding EMA
        let mut ema: f64 = data.iter().take(period).map(|c| c.close.0).sum();
        ema /= period as f64;
        self.history.push(ema);

        // EMA calculation for remaining data
        for i in period..data.len() {
            ema = ema_step(data[i].close.0, ema, self.multiplier);
            self.history.push(ema);
        }

        self.prev_ema = Some(ema);
        self.latest = Some(ema);
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
        let is_new_candle = len > self.last_len || self.prev_ema.is_none();

        let ema = if let Some(prev) = self.prev_ema {
            if is_new_candle {
                // New candle - apply EMA step
                ema_step(candles[len - 1].close.0, prev, self.multiplier)
            } else {
                // Same snapshot size - recalculate from previous EMA base
                // This handles the case where the same snapshot is passed again
                // (shouldn't normally happen in production, but handle gracefully)
                ema_step(candles[len - 1].close.0, prev, self.multiplier)
            }
        } else if len == period {
            // First calculation with exactly enough data - seed with SMA
            Self::compute_initial_sma(candles, period)
        } else {
            // First calculation with more data than period - need to catch up
            // Compute initial SMA from first `period` candles
            let mut ema: f64 = candles.iter().take(period).map(|c| c.close.0).sum();
            ema /= period as f64;

            // Apply EMA steps for remaining candles
            for i in period..len {
                ema = ema_step(candles[i].close.0, ema, self.multiplier);
            }
            ema
        };

        // Update state
        self.prev_ema = Some(ema);
        self.latest = Some(ema);
        self.state = IndicatorState::Ready;

        // Only push to history if this is a new candle
        if is_new_candle {
            self.history.push(ema);
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Update last history entry in place
            *self.history.last_mut().unwrap() = ema;
        }

        Some(ema)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.prev_ema = None;
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for EMA {
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
    fn ema_batch_calculation() {
        let mut ema = EMA::new(EMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        ema.calc(&candles).unwrap();

        assert!(ema.state().is_ready());
        assert_eq!(ema.len(), 5);

        // First two are NaN
        assert!(ema.get(0).unwrap().is_nan());
        assert!(ema.get(1).unwrap().is_nan());

        // Index 2: SMA of first 3 = (1+2+3)/3 = 2
        assert!((ema.get(2).unwrap() - 2.0).abs() < 0.0001);

        // Multiplier = 2/(3+1) = 0.5
        // Index 3: (4 - 2) * 0.5 + 2 = 3
        assert!((ema.get(3).unwrap() - 3.0).abs() < 0.0001);

        // Index 4: (5 - 3) * 0.5 + 3 = 4
        assert!((ema.get(4).unwrap() - 4.0).abs() < 0.0001);
    }

    #[test]
    fn ema_streaming_next() {
        let mut ema = EMA::new(EMAConfig::new(3));

        // Warmup - not enough candles
        let snap1 = candles_from_closes(&[1.0]);
        assert!(ema.next(&snap1).is_none());

        let snap2 = candles_from_closes(&[1.0, 2.0]);
        assert!(ema.next(&snap2).is_none());

        // Now should be ready - first EMA is SMA
        let snap3 = candles_from_closes(&[1.0, 2.0, 3.0]);
        let result = ema.next(&snap3);
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001); // SMA(1,2,3) = 2

        // Continue streaming
        let snap4 = candles_from_closes(&[1.0, 2.0, 3.0, 4.0]);
        let result = ema.next(&snap4);
        assert!((result.unwrap() - 3.0).abs() < 0.0001); // (4-2)*0.5+2 = 3
    }

    #[test]
    fn ema_multiplier_calculation() {
        let ema = EMA::new(EMAConfig::new(14));
        // 2 / (14 + 1) = 2/15 ≈ 0.1333
        assert!((ema.multiplier() - 0.1333).abs() < 0.001);
    }

    #[test]
    fn ema_reset() {
        let mut ema = EMA::new(EMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        ema.calc(&candles).unwrap();

        assert!(ema.state().is_ready());

        ema.reset();

        assert!(ema.state().is_uninitialized());
        assert!(ema.history.is_empty());
        assert!(ema.latest().is_none());
    }

    #[test]
    fn ema_negative_indexing() {
        let mut ema = EMA::new(EMAConfig::new(3));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        ema.calc(&candles).unwrap();

        // -1 should be last value (4.0)
        assert!((ema.get(-1).unwrap() - 4.0).abs() < 0.0001);
        // -2 should be second to last (3.0)
        assert!((ema.get(-2).unwrap() - 3.0).abs() < 0.0001);
    }

    #[test]
    fn ema_insufficient_data() {
        let mut ema = EMA::new(EMAConfig::new(10));
        let candles = candles_from_closes(&[1.0, 2.0, 3.0]);

        let result = ema.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn ema_next_catches_up_from_beginning() {
        let mut ema = EMA::new(EMAConfig::new(3));

        // Large snapshot - EMA catches up from the beginning
        let candles = candles_from_closes(&[100.0, 200.0, 1.0, 2.0, 3.0]);

        // Multiplier = 2/(3+1) = 0.5
        // SMA(100,200,1) = 100.333...
        // EMA at idx 3: (2 - 100.333) * 0.5 + 100.333 = 51.166...
        // EMA at idx 4: (3 - 51.166) * 0.5 + 51.166 = 27.083...
        let result = ema.next(&candles);
        assert!(result.is_some());
        assert!((result.unwrap() - 27.0833).abs() < 0.01);
    }

    #[test]
    fn ema_parallel_safe() {
        use std::sync::Arc;

        let candles = Arc::new(candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]));

        let mut ema1 = EMA::new(EMAConfig::new(3));
        let mut ema2 = EMA::new(EMAConfig::new(3));

        // Both can read from the same snapshot
        let snapshot: &[Ohlcv] = &candles;
        let r1 = ema1.next(snapshot);
        let r2 = ema2.next(snapshot);

        assert_eq!(r1, r2);
        // EMA catches up from beginning:
        // SMA(1,2,3) = 2.0
        // EMA(4) = (4-2)*0.5+2 = 3.0
        // EMA(5) = (5-3)*0.5+3 = 4.0
        assert!((r1.unwrap() - 4.0).abs() < 0.0001);
    }

    #[test]
    fn ema_batch_vs_streaming_equivalence() {
        let candles = candles_from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let period = 3;

        // Batch
        let mut batch = EMA::new(EMAConfig::new(period));
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = EMA::new(EMAConfig::new(period));
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
