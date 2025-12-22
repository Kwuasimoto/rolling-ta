//! True Range (TR) indicator.
//!
//! The True Range measures volatility by taking the greatest of:
//! - Current High - Current Low
//! - |Current High - Previous Close|
//! - |Current Low - Previous Close|
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::TRConfig,
    error::TAResult,
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// True Range indicator.
///
/// Measures the true range of price movement, accounting for gaps.
///
/// # Formula
///
/// TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volatility::{TR, TRConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut tr = TR::default();
/// let candles = vec![
///     Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0),
/// ];
/// tr.calc(&candles).unwrap();
///
/// assert!(tr.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct TR {
    config: TRConfig,
    state: IndicatorState,
    prev_close: f64,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl TR {
    /// Create a new TR indicator.
    pub fn new(config: TRConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            prev_close: 0.0,
            history: Vec::new(),
            latest: None,
            last_len: 0,
        }
    }

    /// Calculate true range for a single candle.
    #[inline]
    pub fn calculate_tr(high: f64, low: f64, prev_close: f64) -> f64 {
        let hl = high - low;
        let hc = (high - prev_close).abs();
        let lc = (low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Get the latest true range value.
    #[inline]
    pub fn tr_latest(&self) -> Option<f64> {
        self.latest
    }

    /// Get the previous close (for ATR composition).
    #[inline]
    pub fn prev_close(&self) -> f64 {
        self.prev_close
    }
}

impl Default for TR {
    fn default() -> Self {
        Self::new(TRConfig)
    }
}

impl Indicator for TR {
    type Output = f64;
    type Config = TRConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();
        if n == 0 {
            self.history.clear();
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Reset state
        self.history = Vec::with_capacity(n);

        // First candle: TR = High - Low (no previous close)
        self.history.push(data[0].high.0 - data[0].low.0);

        // Subsequent candles
        for i in 1..n {
            let tr = Self::calculate_tr(data[i].high.0, data[i].low.0, data[i - 1].close.0);
            self.history.push(tr);
        }

        self.prev_close = data.last().unwrap().close.0;
        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();

        if len == 0 {
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        let current = candles.last().unwrap();
        let high = current.high.0;
        let low = current.low.0;
        let close = current.close.0;

        let tr = if self.last_len == 0 {
            // Very first candle - TR is just high - low
            high - low
        } else if len >= 2 {
            // Use previous candle's close from the snapshot
            let prev_close = candles[len - 2].close.0;
            Self::calculate_tr(high, low, prev_close)
        } else {
            // Single candle in snapshot but we've seen data before
            // Use stored prev_close
            Self::calculate_tr(high, low, self.prev_close)
        };

        if is_new_candle {
            self.history.push(tr);
            self.prev_close = close;
            self.last_len = len;
        } else if !self.history.is_empty() {
            // Same candle - update last value
            *self.history.last_mut().unwrap() = tr;
        }

        self.latest = Some(tr);
        self.state = IndicatorState::Ready;

        Some(tr)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.prev_close = 0.0;
        self.history.clear();
        self.latest = None;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        1 // TR is available immediately
    }
}

impl HistoricalIndicator for TR {
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

    #[test]
    fn tr_basic_calculation() {
        let mut tr = TR::default();
        let candles = vec![
            create_candle(0, 100.0, 105.0, 98.0, 103.0),
            create_candle(1, 103.0, 108.0, 101.0, 106.0),
            create_candle(2, 106.0, 110.0, 104.0, 109.0),
        ];

        tr.calc(&candles).unwrap();

        assert!(tr.state().is_ready());
        assert_eq!(tr.len(), 3);

        // First candle: TR = 105 - 98 = 7
        assert_eq!(tr.get(0).unwrap(), 7.0);

        // Second candle: max(108-101, |108-103|, |101-103|) = max(7, 5, 2) = 7
        assert_eq!(tr.get(1).unwrap(), 7.0);

        // Third candle: max(110-104, |110-106|, |104-106|) = max(6, 4, 2) = 6
        assert_eq!(tr.get(2).unwrap(), 6.0);
    }

    #[test]
    fn tr_gap_up() {
        let mut tr = TR::default();
        let candles = vec![
            create_candle(0, 100.0, 105.0, 98.0, 100.0),
            create_candle(1, 110.0, 115.0, 108.0, 112.0), // Gap up from 100 to 110
        ];

        tr.calc(&candles).unwrap();

        // Second candle TR should capture the gap
        // max(115-108, |115-100|, |108-100|) = max(7, 15, 8) = 15
        assert_eq!(tr.get(1).unwrap(), 15.0);
    }

    #[test]
    fn tr_streaming_next() {
        let mut tr = TR::default();
        let candles = vec![
            create_candle(0, 100.0, 105.0, 98.0, 103.0),
            create_candle(1, 103.0, 108.0, 101.0, 106.0),
            create_candle(2, 106.0, 110.0, 104.0, 109.0),
        ];

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = tr.next(snapshot);
            assert!(result.is_some(), "TR should always produce value");
        }

        assert!(tr.state().is_ready());
        assert_eq!(tr.len(), 3);

        // Values should match batch
        assert_eq!(tr.get(0).unwrap(), 7.0);
        assert_eq!(tr.get(1).unwrap(), 7.0);
        assert_eq!(tr.get(2).unwrap(), 6.0);
    }

    #[test]
    fn tr_batch_vs_streaming_equivalence() {
        let candles = vec![
            create_candle(0, 100.0, 105.0, 98.0, 103.0),
            create_candle(1, 103.0, 108.0, 101.0, 106.0),
            create_candle(2, 106.0, 110.0, 104.0, 109.0),
            create_candle(3, 109.0, 112.0, 107.0, 111.0),
            create_candle(4, 111.0, 115.0, 109.0, 113.0),
        ];

        // Batch
        let mut batch = TR::default();
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = TR::default();
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            assert!(
                (batch.get(i as isize).unwrap() - stream.get(i as isize).unwrap()).abs() < 1e-10,
                "TR mismatch at {}: batch={}, stream={}",
                i,
                batch.get(i as isize).unwrap(),
                stream.get(i as isize).unwrap()
            );
        }
    }

    #[test]
    fn tr_always_positive() {
        let mut tr = TR::default();
        let candles: Vec<Ohlcv> = (0..50)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.5).sin() * 10.0;
                create_candle(i, base, base + 2.0, base - 1.0, base + 0.5)
            })
            .collect();

        tr.calc(&candles).unwrap();

        for val in tr.history() {
            assert!(*val >= 0.0, "TR should always be >= 0, got {}", val);
        }
    }

    #[test]
    fn tr_reset() {
        let mut tr = TR::default();
        let candles = vec![
            create_candle(0, 100.0, 105.0, 98.0, 103.0),
            create_candle(1, 103.0, 108.0, 101.0, 106.0),
        ];

        tr.calc(&candles).unwrap();
        assert!(tr.state().is_ready());

        tr.reset();
        assert!(tr.state().is_uninitialized());
        assert_eq!(tr.len(), 0);
        assert!(tr.latest().is_none());
    }
}
