//! Weighted Moving Average (WMA) indicator.

use std::collections::VecDeque;

use crate::ta::{
    config::WMAConfig,
    error::{TAError, TAResult},
    math::Temporal,
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    Indicator, HistoricalIndicator,
};

/// Weighted Moving Average indicator.
///
/// Assigns linearly increasing weights to more recent prices.
///
/// # Formula
///
/// WMA = (P1×1 + P2×2 + ... + Pn×n) / (1 + 2 + ... + n)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{WMA, WMAConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut wma = WMA::new(WMAConfig::new(3));
/// let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// wma.calc(&data).unwrap();
///
/// assert!(wma.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct WMA {
    config: WMAConfig,
    state: IndicatorState,
    weight_sum: usize,
    window: VecDeque<f64>,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Temporal state for candle period detection.
    temporal: Temporal,
}

impl WMA {
    /// Create a new WMA indicator with the given configuration.
    pub fn new(config: WMAConfig) -> Self {
        let weight_sum = config.weight_sum();
        let temporal = if config.timeframe > 0 {
            Temporal::new(config.timeframe)
        } else {
            Temporal::disabled()
        };

        Self {
            config,
            state: IndicatorState::Uninitialized,
            weight_sum,
            window: VecDeque::with_capacity(config.period),
            history: Vec::new(),
            latest: None,
            temporal,
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

    /// Calculate WMA from current window.
    fn calculate_wma(&self) -> f64 {
        let mut weighted_sum = 0.0;
        for (i, &price) in self.window.iter().enumerate() {
            weighted_sum += price * (i + 1) as f64;
        }
        weighted_sum / self.weight_sum as f64
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
        self.window = VecDeque::with_capacity(period);
        self.history = Vec::with_capacity(data.len());

        let closes = &data.closes;

        // Fill initial window
        for i in 0..period {
            self.window.push_back(closes[i]);
            self.history.push(f64::NAN);
        }

        // First WMA
        let first_wma = self.calculate_wma();
        self.history[period - 1] = first_wma;

        // Rolling calculation
        for i in period..closes.len() {
            self.window.pop_front();
            self.window.push_back(closes[i]);
            let wma = self.calculate_wma();
            self.history.push(wma);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let close = tick.close.0;
        let timestamp = tick.timestamp.0;
        let period = self.config.period;

        // Check if this is a same-candle update (temporal mode)
        let is_same_period = self.temporal.is_same_period(timestamp);

        if is_same_period {
            // Same candle period - update the back of the window in place
            if let Some(back) = self.window.back_mut() {
                *back = close;
            }

            if self.state.is_ready() {
                let wma = self.calculate_wma();
                // Update last history entry in place
                if let Some(last) = self.history.last_mut() {
                    *last = wma;
                }
                self.latest = Some(wma);
                Ok(Some(wma))
            } else {
                Ok(None)
            }
        } else {
            // New candle period - normal update path
            self.temporal.record(timestamp);

            if self.window.len() >= period {
                self.window.pop_front();
            }
            self.window.push_back(close);

            self.state = self.state.increment(period);

            if self.state.is_ready() {
                let wma = self.calculate_wma();
                self.history.push(wma);
                self.latest = Some(wma);
                Ok(Some(wma))
            } else {
                self.history.push(f64::NAN);
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
        self.temporal.reset();
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

    #[test]
    fn wma_batch_calculation() {
        let mut wma = WMA::new(WMAConfig::new(3));
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        wma.calc(&data).unwrap();

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
    fn wma_streaming_update() {
        let mut wma = WMA::new(WMAConfig::new(3));

        // Warmup
        assert!(wma.update(&Ohlcv::from_close(1.0)).unwrap().is_none());
        assert!(wma.update(&Ohlcv::from_close(2.0)).unwrap().is_none());

        // Third value - should get first WMA
        let result = wma.update(&Ohlcv::from_close(3.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.333).abs() < 0.01);

        // Fourth value
        let result = wma.update(&Ohlcv::from_close(4.0)).unwrap();
        assert!((result.unwrap() - 3.333).abs() < 0.01);
    }

    #[test]
    fn wma_weight_sum() {
        let wma = WMA::new(WMAConfig::new(5));
        // 1+2+3+4+5 = 15
        assert_eq!(wma.weight_sum, 15);
    }

    // Temporal tests

    #[test]
    fn wma_temporal_config() {
        // Non-temporal (default)
        let wma = WMA::new(WMAConfig::new(14));
        assert!(!wma.is_temporal());
        assert_eq!(wma.timeframe(), 0);

        // Temporal mode
        let wma = WMA::new(WMAConfig::with_timeframe(14, 60));
        assert!(wma.is_temporal());
        assert_eq!(wma.timeframe(), 60);
    }

    #[test]
    fn wma_temporal_same_period_updates_in_place() {
        // 1-minute timeframe (60 seconds), period 3
        let mut wma = WMA::new(WMAConfig::with_timeframe(3, 60));

        // Warmup with different candle periods
        // Period 16: timestamp 960
        wma.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        assert_eq!(wma.len(), 1);

        // Period 17: timestamp 1020
        wma.update(&Ohlcv::new(1020, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        assert_eq!(wma.len(), 2);

        // Period 18: timestamp 1080 - this should be ready
        // WMA = (1×1 + 2×2 + 3×3) / 6 = 14/6 ≈ 2.333
        let result = wma.update(&Ohlcv::new(1080, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.333).abs() < 0.01);
        assert_eq!(wma.len(), 3);

        // Same period (still 18): timestamp 1100 - should update in place
        // Window is now [1, 2, 6], WMA = (1×1 + 2×2 + 6×3) / 6 = 23/6 ≈ 3.833
        let result = wma.update(&Ohlcv::new(1100, 6.0, 6.0, 6.0, 6.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 3.833).abs() < 0.01);
        assert_eq!(wma.len(), 3); // Still 3, not 4!

        // Same period again: timestamp 1110 - should update in place again
        // Window is now [1, 2, 9], WMA = (1×1 + 2×2 + 9×3) / 6 = 32/6 ≈ 5.333
        let result = wma.update(&Ohlcv::new(1110, 9.0, 9.0, 9.0, 9.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 5.333).abs() < 0.01);
        assert_eq!(wma.len(), 3); // Still 3!
    }

    #[test]
    fn wma_temporal_new_period_pushes() {
        // 5-minute timeframe (300 seconds)
        let mut wma = WMA::new(WMAConfig::with_timeframe(3, 300));

        // Period 3: timestamps 900-1199
        wma.update(&Ohlcv::new(1000, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        wma.update(&Ohlcv::new(1100, 1.5, 1.5, 1.5, 1.5, 0.0)).unwrap(); // Same period, updates
        assert_eq!(wma.len(), 1);

        // Period 4: timestamps 1200-1499
        wma.update(&Ohlcv::new(1200, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        assert_eq!(wma.len(), 2);

        // Period 5: timestamps 1500-1799 - should be ready
        // Window: [1.5, 2.0, 3.0], WMA = (1.5×1 + 2.0×2 + 3.0×3) / 6 = 14.5/6 ≈ 2.417
        let result = wma.update(&Ohlcv::new(1500, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.417).abs() < 0.01);
        assert_eq!(wma.len(), 3);

        // Period 6: timestamp 1800 - new period
        // Window: [2.0, 3.0, 4.0], WMA = (2×1 + 3×2 + 4×3) / 6 = 20/6 ≈ 3.333
        let result = wma.update(&Ohlcv::new(1800, 4.0, 4.0, 4.0, 4.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 3.333).abs() < 0.01);
        assert_eq!(wma.len(), 4);
    }

    #[test]
    fn wma_non_temporal_always_pushes() {
        // Non-temporal mode (timeframe = 0)
        let mut wma = WMA::new(WMAConfig::new(3));

        // All at same timestamp - should still push each one
        wma.update(&Ohlcv::new(1000, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        wma.update(&Ohlcv::new(1000, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        wma.update(&Ohlcv::new(1000, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();

        assert_eq!(wma.len(), 3);
        assert!(wma.state().is_ready());
    }

    #[test]
    fn wma_temporal_reset_preserves_config() {
        let mut wma = WMA::new(WMAConfig::with_timeframe(3, 60));

        // Add some data
        wma.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        wma.update(&Ohlcv::new(1020, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        wma.update(&Ohlcv::new(1080, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();

        assert!(wma.is_temporal());
        assert_eq!(wma.timeframe(), 60);

        wma.reset();

        // Temporal config should be preserved
        assert!(wma.is_temporal());
        assert_eq!(wma.timeframe(), 60);
        assert!(wma.state().is_uninitialized());
    }
}
