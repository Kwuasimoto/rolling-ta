//! True Range (TR) indicator.
//!
//! The True Range measures volatility by taking the greatest of:
//! - Current High - Current Low
//! - |Current High - Previous Close|
//! - |Current Low - Previous Close|

use crate::ta::{
    config::TRConfig,
    error::TAResult,
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
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
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut tr = TR::default();
/// let data = OhlcvSeries::from_tuples(&[
///     (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     (1, 103.0, 108.0, 101.0, 106.0, 1100.0),
/// ]);
/// tr.calc(&data).unwrap();
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

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let n = data.len();
        if n == 0 {
            self.history.clear();
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Reset state
        self.history = Vec::with_capacity(n);

        let highs = &data.highs;
        let lows = &data.lows;
        let closes = &data.closes;

        // First candle: TR = High - Low (no previous close)
        self.history.push(highs[0] - lows[0]);

        // Subsequent candles
        for i in 1..n {
            let tr = Self::calculate_tr(highs[i], lows[i], closes[i - 1]);
            self.history.push(tr);
        }

        self.prev_close = *closes.last().unwrap();
        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let high = tick.high.0;
        let low = tick.low.0;
        let close = tick.close.0;

        if self.state.is_uninitialized() {
            // First tick - TR is just high - low
            let tr = high - low;
            self.history.push(tr);
            self.latest = Some(tr);
            self.prev_close = close;
            self.state = IndicatorState::Ready;
            return Ok(Some(tr));
        }

        // Calculate TR using previous close
        let tr = Self::calculate_tr(high, low, self.prev_close);
        self.history.push(tr);
        self.latest = Some(tr);
        self.prev_close = close;

        Ok(Some(tr))
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.prev_close = 0.0;
        self.history.clear();
        self.latest = None;
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

    #[test]
    fn tr_basic_calculation() {
        let mut tr = TR::default();
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
            (1, 103.0, 108.0, 101.0, 106.0, 1100.0),
            (2, 106.0, 110.0, 104.0, 109.0, 1200.0),
        ]);

        tr.calc(&data).unwrap();

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
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            (1, 110.0, 115.0, 108.0, 112.0, 1100.0), // Gap up from 100 to 110
        ]);

        tr.calc(&data).unwrap();

        // Second candle TR should capture the gap
        // max(115-108, |115-100|, |108-100|) = max(7, 15, 8) = 15
        assert_eq!(tr.get(1).unwrap(), 15.0);
    }

    #[test]
    fn tr_streaming_update() {
        let mut tr = TR::default();

        // First tick
        let result = tr.update(&Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0)).unwrap();
        assert_eq!(result, Some(7.0)); // 105 - 98

        // Second tick
        let result = tr.update(&Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0)).unwrap();
        assert_eq!(result, Some(7.0)); // max(7, 5, 2)
    }

    #[test]
    fn tr_always_positive() {
        let mut tr = TR::default();
        let closes: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0).collect();
        let data = OhlcvSeries::from_closes(&closes);

        tr.calc(&data).unwrap();

        for val in tr.history() {
            assert!(*val >= 0.0, "TR should always be >= 0, got {}", val);
        }
    }
}
