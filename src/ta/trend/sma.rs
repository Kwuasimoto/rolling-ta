//! Simple Moving Average (SMA) indicator.

use crate::ta::{
    config::SMAConfig,
    error::{TAError, TAResult},
    math::RollingWindow,
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    Indicator, HistoricalIndicator,
};

/// Simple Moving Average indicator.
///
/// Calculates the arithmetic mean of prices over a rolling window.
///
/// # Formula
///
/// SMA = (P1 + P2 + ... + Pn) / n
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{SMA, SMAConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut sma = SMA::new(SMAConfig::new(3));
/// let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// sma.calc(&data).unwrap();
///
/// assert!(sma.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct SMA {
    config: SMAConfig,
    state: IndicatorState,
    window: RollingWindow,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl SMA {
    /// Create a new SMA indicator with the given configuration.
    pub fn new(config: SMAConfig) -> Self {
        Self {
            window: RollingWindow::new(config.period),
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

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
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
        self.window = RollingWindow::new(period);
        self.history = Vec::with_capacity(data.len());

        let closes = &data.closes;

        // Build initial window
        for i in 0..period {
            self.window.push(closes[i]);
            self.history.push(f64::NAN);
        }

        // First valid SMA
        let first_sma = self.window.mean();
        self.history[period - 1] = first_sma;

        // Rolling calculation
        for i in period..closes.len() {
            self.window.push(closes[i]);
            let sma = self.window.mean();
            self.history.push(sma);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let close = tick.close.0;

        self.window.push(close);
        self.state = self.state.increment(self.config.period);

        if self.state.is_ready() {
            let sma = self.window.mean();
            self.history.push(sma);
            self.latest = Some(sma);
            Ok(Some(sma))
        } else {
            self.history.push(f64::NAN);
            Ok(None)
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.window = RollingWindow::new(self.config.period);
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

    #[test]
    fn sma_batch_calculation() {
        let mut sma = SMA::new(SMAConfig::new(3));
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        sma.calc(&data).unwrap();

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
    fn sma_streaming_update() {
        let mut sma = SMA::new(SMAConfig::new(3));

        // Warmup
        assert!(sma.update(&Ohlcv::from_close(1.0)).unwrap().is_none());
        assert!(sma.update(&Ohlcv::from_close(2.0)).unwrap().is_none());

        // Now should be ready
        let result = sma.update(&Ohlcv::from_close(3.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001);

        // Continue streaming
        let result = sma.update(&Ohlcv::from_close(4.0)).unwrap();
        assert!((result.unwrap() - 3.0).abs() < 0.0001);
    }

    #[test]
    fn sma_negative_indexing() {
        let mut sma = SMA::new(SMAConfig::new(3));
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        sma.calc(&data).unwrap();

        // -1 should be last value (4.0)
        assert!((sma.get(-1).unwrap() - 4.0).abs() < 0.0001);
        // -2 should be second to last (3.0)
        assert!((sma.get(-2).unwrap() - 3.0).abs() < 0.0001);
    }

    #[test]
    fn sma_insufficient_data() {
        let mut sma = SMA::new(SMAConfig::new(10));
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0]);

        let result = sma.calc(&data);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn sma_reset() {
        let mut sma = SMA::new(SMAConfig::new(3));
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        sma.calc(&data).unwrap();

        assert!(sma.state().is_ready());

        sma.reset();

        assert!(sma.state().is_uninitialized());
        assert!(sma.history.is_empty());
        assert!(sma.latest().is_none());
    }
}
