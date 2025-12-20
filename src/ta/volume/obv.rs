//! On-Balance Volume (OBV) indicator.
//!
//! OBV is a momentum indicator that uses volume flow to predict price changes.

use crate::ta::{
    config::OBVConfig,
    error::TAResult,
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    HistoricalIndicator, Indicator,
};

/// On-Balance Volume indicator.
///
/// Tracks cumulative volume based on price direction.
///
/// # Formula
///
/// - If close > prev_close: OBV = prev_OBV + volume
/// - If close < prev_close: OBV = prev_OBV - volume
/// - If close = prev_close: OBV = prev_OBV
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volume::{OBV, OBVConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut obv = OBV::default();
/// let data = OhlcvSeries::from_tuples(&[
///     (0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     (1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     (2, 106.0, 107.0, 102.0, 104.0, 900.0),
/// ]);
/// obv.calc(&data).unwrap();
///
/// assert!(obv.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct OBV {
    config: OBVConfig,
    state: IndicatorState,
    obv_value: f64,
    prev_close: f64,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl OBV {
    /// Create a new OBV indicator.
    pub fn new(config: OBVConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            obv_value: 0.0,
            prev_close: 0.0,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the current OBV value.
    #[inline]
    pub fn obv_value(&self) -> f64 {
        self.obv_value
    }
}

impl Default for OBV {
    fn default() -> Self {
        Self::new(OBVConfig)
    }
}

impl Indicator for OBV {
    type Output = f64;
    type Config = OBVConfig;

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
        let closes = &data.closes;
        let volumes = &data.volumes;

        // First candle: OBV starts at 0
        self.obv_value = 0.0;
        self.history.push(0.0);

        // Subsequent candles
        for i in 1..n {
            let close = closes[i];
            let prev_close = closes[i - 1];
            let volume = volumes[i];

            if close > prev_close {
                self.obv_value += volume;
            } else if close < prev_close {
                self.obv_value -= volume;
            }
            // If close == prev_close, OBV stays the same

            self.history.push(self.obv_value);
        }

        self.prev_close = *closes.last().unwrap();
        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let close = tick.close.0;
        let volume = tick.volume.0;

        if self.state.is_uninitialized() {
            // First tick - OBV starts at 0
            self.obv_value = 0.0;
            self.history.push(0.0);
            self.latest = Some(0.0);
            self.prev_close = close;
            self.state = IndicatorState::Ready;
            return Ok(Some(0.0));
        }

        // Update OBV based on price direction
        if close > self.prev_close {
            self.obv_value += volume;
        } else if close < self.prev_close {
            self.obv_value -= volume;
        }

        self.history.push(self.obv_value);
        self.latest = Some(self.obv_value);
        self.prev_close = close;

        Ok(Some(self.obv_value))
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.obv_value = 0.0;
        self.prev_close = 0.0;
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        1 // OBV is available immediately
    }
}

impl HistoricalIndicator for OBV {
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
    fn obv_batch_calculation() {
        let mut obv = OBV::default();
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 98.0, 100.0, 1000.0), // OBV = 0
            (1, 100.0, 108.0, 99.0, 105.0, 1100.0), // close up: OBV = 1100
            (2, 105.0, 107.0, 102.0, 103.0, 900.0), // close down: OBV = 1100 - 900 = 200
            (3, 103.0, 106.0, 101.0, 103.0, 800.0), // close same: OBV = 200
            (4, 103.0, 108.0, 100.0, 107.0, 1200.0), // close up: OBV = 200 + 1200 = 1400
        ]);

        obv.calc(&data).unwrap();

        assert!(obv.state().is_ready());
        assert_eq!(obv.len(), 5);

        assert_eq!(obv.get(0).unwrap(), 0.0);
        assert_eq!(obv.get(1).unwrap(), 1100.0);
        assert_eq!(obv.get(2).unwrap(), 200.0);
        assert_eq!(obv.get(3).unwrap(), 200.0); // Same close
        assert_eq!(obv.get(4).unwrap(), 1400.0);
    }

    #[test]
    fn obv_streaming_update() {
        let mut obv = OBV::default();

        // First tick
        let result = obv.update(&Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0)).unwrap();
        assert_eq!(result, Some(0.0));

        // Second tick - close up
        let result = obv.update(&Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0)).unwrap();
        assert_eq!(result, Some(1100.0));

        // Third tick - close down
        let result = obv.update(&Ohlcv::new(2, 105.0, 107.0, 102.0, 103.0, 900.0)).unwrap();
        assert_eq!(result, Some(200.0));
    }

    #[test]
    fn obv_uptrend() {
        let mut obv = OBV::default();
        // Consistent uptrend
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 102.0, 99.0, 101.0, 1000.0),
            (1, 101.0, 103.0, 100.0, 102.0, 1000.0),
            (2, 102.0, 104.0, 101.0, 103.0, 1000.0),
            (3, 103.0, 105.0, 102.0, 104.0, 1000.0),
        ]);

        obv.calc(&data).unwrap();

        // OBV should be increasing
        let latest = obv.latest().unwrap();
        assert_eq!(latest, 3000.0); // 0 + 1000 + 1000 + 1000
    }

    #[test]
    fn obv_downtrend() {
        let mut obv = OBV::default();
        // Consistent downtrend
        let data = OhlcvSeries::from_tuples(&[
            (0, 104.0, 105.0, 103.0, 104.0, 1000.0),
            (1, 104.0, 104.0, 102.0, 103.0, 1000.0),
            (2, 103.0, 103.0, 101.0, 102.0, 1000.0),
            (3, 102.0, 102.0, 100.0, 101.0, 1000.0),
        ]);

        obv.calc(&data).unwrap();

        // OBV should be decreasing
        let latest = obv.latest().unwrap();
        assert_eq!(latest, -3000.0); // 0 - 1000 - 1000 - 1000
    }

    #[test]
    fn obv_reset() {
        let mut obv = OBV::default();
        let data = OhlcvSeries::from_tuples(&[
            (0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            (1, 100.0, 108.0, 99.0, 105.0, 1100.0),
        ]);

        obv.calc(&data).unwrap();
        assert!(obv.state().is_ready());
        assert_eq!(obv.len(), 2);

        obv.reset();
        assert!(obv.state().is_uninitialized());
        assert_eq!(obv.len(), 0);
        assert_eq!(obv.obv_value(), 0.0);
    }
}
