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
    history: Vec<f64>,
    latest: Option<f64>,
    window: RollingWindow,
}

impl SMA {
    /// Create a new SMA indicator with the given configuration.
    pub fn new(config: SMAConfig) -> Self {
        let window = if config.timeframe > 0 {
            RollingWindow::with_timeframe(config.period, config.timeframe)
        } else {
            RollingWindow::new(config.period)
        };

        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            window,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Get the timeframe (0 = temporal mode disabled).
    #[inline]
    pub fn timeframe(&self) -> i64 {
        self.config.timeframe
    }

    /// Check if temporal mode is enabled.
    #[inline]
    pub fn is_temporal(&self) -> bool {
        self.config.timeframe > 0
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

        // Reset window (preserves capacity and temporal config)
        self.window.clear();
        self.history = Vec::with_capacity(data.len());

        // Build initial window
        for i in 0..period {
            let candle = data.get(i).unwrap();
            self.window.push(candle);
            self.history.push(f64::NAN);
        }

        // First valid SMA
        let first_sma = self.window.mean();
        self.history[period - 1] = first_sma;

        // Rolling calculation
        for i in period..data.len() {
            let candle = data.get(i).unwrap();
            self.window.push(candle);
            let sma = self.window.mean();
            self.history.push(sma);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        // Use temporal-aware push (handles both new candle and same-period update)
        let is_new_candle = self.window.push_with_timestamp(*tick);

        if is_new_candle {
            // New candle: increment state and append to history
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
        } else {
            // Same candle period: update latest value in place
            if self.state.is_ready() {
                let sma = self.window.mean();
                // Update the last history entry instead of appending
                if let Some(last) = self.history.last_mut() {
                    *last = sma;
                }
                self.latest = Some(sma);
                Ok(Some(sma))
            } else {
                // Still warming up, just update the last NAN entry
                Ok(None)
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.window.clear();
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

    // Temporal tests

    #[test]
    fn sma_temporal_config() {
        // Non-temporal (default)
        let sma = SMA::new(SMAConfig::new(14));
        assert!(!sma.is_temporal());
        assert_eq!(sma.timeframe(), 0);

        // Temporal mode
        let sma = SMA::new(SMAConfig::with_timeframe(14, 60));
        assert!(sma.is_temporal());
        assert_eq!(sma.timeframe(), 60);
    }

    #[test]
    fn sma_temporal_same_period_updates_in_place() {
        // 1-minute timeframe (60 seconds)
        let mut sma = SMA::new(SMAConfig::with_timeframe(3, 60));

        // Warmup with different candle periods
        // Period 16: timestamp 960
        sma.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        assert_eq!(sma.len(), 1);

        // Period 17: timestamp 1020
        sma.update(&Ohlcv::new(1020, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        assert_eq!(sma.len(), 2);

        // Period 18: timestamp 1080 - this should be ready
        let result = sma.update(&Ohlcv::new(1080, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001); // (1+2+3)/3 = 2
        assert_eq!(sma.len(), 3);

        // Same period (still 18): timestamp 1100 - should update in place
        let result = sma.update(&Ohlcv::new(1100, 6.0, 6.0, 6.0, 6.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 3.0).abs() < 0.0001); // (1+2+6)/3 = 3
        assert_eq!(sma.len(), 3); // Still 3, not 4!

        // Same period again: timestamp 1110 - should update in place again
        let result = sma.update(&Ohlcv::new(1110, 9.0, 9.0, 9.0, 9.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 4.0).abs() < 0.0001); // (1+2+9)/3 = 4
        assert_eq!(sma.len(), 3); // Still 3!
    }

    #[test]
    fn sma_temporal_new_period_pushes() {
        // 5-minute timeframe (300 seconds)
        let mut sma = SMA::new(SMAConfig::with_timeframe(3, 300));

        // Period 3: timestamps 900-1199
        sma.update(&Ohlcv::new(1000, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        sma.update(&Ohlcv::new(1100, 1.5, 1.5, 1.5, 1.5, 0.0)).unwrap(); // Same period, updates
        assert_eq!(sma.len(), 1);

        // Period 4: timestamps 1200-1499
        sma.update(&Ohlcv::new(1200, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        assert_eq!(sma.len(), 2);

        // Period 5: timestamps 1500-1799 - should be ready
        let result = sma.update(&Ohlcv::new(1500, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert!(result.is_some());
        // Window has: 1.5 (updated), 2.0, 3.0 -> mean = 6.5/3 ≈ 2.1667
        assert!((result.unwrap() - 2.1667).abs() < 0.01);
        assert_eq!(sma.len(), 3);

        // Period 6: timestamp 1800 - new period
        let result = sma.update(&Ohlcv::new(1800, 4.0, 4.0, 4.0, 4.0, 0.0)).unwrap();
        assert!(result.is_some());
        // Window has: 2.0, 3.0, 4.0 -> mean = 3.0
        assert!((result.unwrap() - 3.0).abs() < 0.0001);
        assert_eq!(sma.len(), 4);
    }

    #[test]
    fn sma_non_temporal_always_pushes() {
        // Non-temporal mode (timeframe = 0)
        let mut sma = SMA::new(SMAConfig::new(3));

        // All at same timestamp - should still push each one
        sma.update(&Ohlcv::new(1000, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        sma.update(&Ohlcv::new(1000, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        sma.update(&Ohlcv::new(1000, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();

        assert_eq!(sma.len(), 3);
        assert!(sma.state().is_ready());
    }

    #[test]
    fn sma_temporal_reset_preserves_config() {
        let mut sma = SMA::new(SMAConfig::with_timeframe(3, 60));

        // Add some data
        sma.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        sma.update(&Ohlcv::new(1020, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        sma.update(&Ohlcv::new(1080, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();

        assert!(sma.is_temporal());
        assert_eq!(sma.timeframe(), 60);

        sma.reset();

        // Temporal config should be preserved
        assert!(sma.is_temporal());
        assert_eq!(sma.timeframe(), 60);
        assert!(sma.state().is_uninitialized());
    }
}
