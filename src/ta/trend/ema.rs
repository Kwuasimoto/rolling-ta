//! Exponential Moving Average (EMA) indicator.

use crate::ta::{
    config::EMAConfig,
    error::{TAError, TAResult},
    math::{ema_multiplier, ema_step, Temporal},
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    Indicator, HistoricalIndicator,
};

/// Exponential Moving Average indicator.
///
/// Gives more weight to recent prices, making it more responsive to new information.
///
/// # Formula
///
/// EMA = (Price - Previous EMA) × Multiplier + Previous EMA
/// Multiplier = 2 / (Period + 1)
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{EMA, EMAConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut ema = EMA::new(EMAConfig::new(3));
/// let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// ema.calc(&data).unwrap();
///
/// assert!(ema.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct EMA {
    config: EMAConfig,
    state: IndicatorState,
    multiplier: f64,
    prev_ema: Option<f64>,
    warmup_sum: f64,
    warmup_count: usize,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Temporal state for candle period detection.
    temporal: Temporal,
    /// EMA value at the start of current candle (for same-candle recalculation).
    ema_at_candle_start: Option<f64>,
}

impl EMA {
    /// Create a new EMA indicator with the given configuration.
    pub fn new(config: EMAConfig) -> Self {
        let multiplier = ema_multiplier(config.period);
        let temporal = if config.timeframe > 0 {
            Temporal::new(config.timeframe)
        } else {
            Temporal::disabled()
        };

        Self {
            config,
            state: IndicatorState::Uninitialized,
            multiplier,
            prev_ema: None,
            warmup_sum: 0.0,
            warmup_count: 0,
            history: Vec::new(),
            latest: None,
            temporal,
            ema_at_candle_start: None,
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
        self.history = Vec::with_capacity(data.len());
        self.warmup_sum = 0.0;
        self.warmup_count = 0;

        let closes = &data.closes;

        // Calculate initial SMA for seeding EMA
        for i in 0..period {
            self.warmup_sum += closes[i];
            self.history.push(f64::NAN);
        }

        let mut ema = self.warmup_sum / period as f64;
        self.history[period - 1] = ema;

        // EMA calculation
        for i in period..closes.len() {
            ema = ema_step(closes[i], ema, self.multiplier);
            self.history.push(ema);
        }

        self.prev_ema = Some(ema);
        self.latest = Some(ema);
        self.warmup_count = period;
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
            // Same candle period - update in place
            if self.state.is_ready() {
                // Restore EMA from candle start and recalculate
                let base_ema = self.ema_at_candle_start.unwrap_or_else(|| self.prev_ema.unwrap());
                let ema = ema_step(close, base_ema, self.multiplier);
                self.prev_ema = Some(ema);
                self.latest = Some(ema);
                // Update last history entry in place
                if let Some(last) = self.history.last_mut() {
                    *last = ema;
                }
                Ok(Some(ema))
            } else {
                // Still warming up - update the last accumulated close
                // Subtract the previous close for this period and add the new one
                if !self.history.is_empty() {
                    // During warmup, history entries are NAN, but we track the close
                    // in warmup_sum. We need to update warmup_sum by replacing the
                    // last close with the new one.
                    // The last close is stored in ema_at_candle_start during warmup
                    if let Some(old_close) = self.ema_at_candle_start {
                        self.warmup_sum = self.warmup_sum - old_close + close;
                        self.ema_at_candle_start = Some(close);
                    }
                }
                Ok(None)
            }
        } else {
            // New candle period - normal update path
            self.temporal.record(timestamp);
            self.warmup_count += 1;

            if self.warmup_count < period {
                // Still warming up - accumulate for SMA seed
                self.warmup_sum += close;
                self.ema_at_candle_start = Some(close); // Track for potential same-candle updates
                self.history.push(f64::NAN);
                self.state = IndicatorState::Warming {
                    count: self.warmup_count,
                };
                Ok(None)
            } else if self.warmup_count == period {
                // First EMA value - use SMA as seed
                self.warmup_sum += close;
                let ema = self.warmup_sum / period as f64;
                self.prev_ema = Some(ema);
                self.ema_at_candle_start = self.prev_ema; // This is the start for next candle
                self.latest = Some(ema);
                self.history.push(ema);
                self.state = IndicatorState::Ready;
                Ok(Some(ema))
            } else {
                // Normal EMA calculation
                // Save current EMA as the base for potential same-candle updates
                self.ema_at_candle_start = self.prev_ema;
                let prev = self.prev_ema.unwrap();
                let ema = ema_step(close, prev, self.multiplier);
                self.prev_ema = Some(ema);
                self.latest = Some(ema);
                self.history.push(ema);
                Ok(Some(ema))
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.prev_ema = None;
        self.warmup_sum = 0.0;
        self.warmup_count = 0;
        self.history.clear();
        self.latest = None;
        self.temporal.reset();
        self.ema_at_candle_start = None;
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

    #[test]
    fn ema_batch_calculation() {
        let mut ema = EMA::new(EMAConfig::new(3));
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        ema.calc(&data).unwrap();

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
    fn ema_streaming_update() {
        let mut ema = EMA::new(EMAConfig::new(3));

        // Warmup
        assert!(ema.update(&Ohlcv::from_close(1.0)).unwrap().is_none());
        assert!(ema.update(&Ohlcv::from_close(2.0)).unwrap().is_none());

        // Third value - should get first EMA (which is SMA)
        let result = ema.update(&Ohlcv::from_close(3.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001);

        // Fourth value
        let result = ema.update(&Ohlcv::from_close(4.0)).unwrap();
        assert!((result.unwrap() - 3.0).abs() < 0.0001);
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
        let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        ema.calc(&data).unwrap();

        assert!(ema.state().is_ready());

        ema.reset();

        assert!(ema.state().is_uninitialized());
        assert!(ema.history.is_empty());
        assert!(ema.latest().is_none());
    }

    // Temporal tests

    #[test]
    fn ema_temporal_config() {
        // Non-temporal (default)
        let ema = EMA::new(EMAConfig::new(14));
        assert!(!ema.is_temporal());
        assert_eq!(ema.timeframe(), 0);

        // Temporal mode
        let ema = EMA::new(EMAConfig::with_timeframe(14, 60));
        assert!(ema.is_temporal());
        assert_eq!(ema.timeframe(), 60);
    }

    #[test]
    fn ema_temporal_same_period_updates_in_place() {
        // 1-minute timeframe (60 seconds), period 3
        let mut ema = EMA::new(EMAConfig::with_timeframe(3, 60));

        // Warmup with different candle periods
        // Period 16: timestamp 960
        ema.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        assert_eq!(ema.len(), 1);

        // Period 17: timestamp 1020
        ema.update(&Ohlcv::new(1020, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        assert_eq!(ema.len(), 2);

        // Period 18: timestamp 1080 - this should be ready
        // First EMA = SMA of (1, 2, 3) = 2.0
        let result = ema.update(&Ohlcv::new(1080, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.0001);
        assert_eq!(ema.len(), 3);

        // Same period (still 18): timestamp 1100 - should update in place
        // Multiplier = 2/(3+1) = 0.5
        // New EMA = (6 - 2) * 0.5 + 2 = 4
        let result = ema.update(&Ohlcv::new(1100, 6.0, 6.0, 6.0, 6.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 4.0).abs() < 0.0001);
        assert_eq!(ema.len(), 3); // Still 3, not 4!

        // Same period again: timestamp 1110 - should update in place again
        // New EMA = (9 - 2) * 0.5 + 2 = 5.5
        let result = ema.update(&Ohlcv::new(1110, 9.0, 9.0, 9.0, 9.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 5.5).abs() < 0.0001);
        assert_eq!(ema.len(), 3); // Still 3!
    }

    #[test]
    fn ema_temporal_new_period_pushes() {
        // 5-minute timeframe (300 seconds)
        let mut ema = EMA::new(EMAConfig::with_timeframe(3, 300));

        // Period 3: timestamps 900-1199
        ema.update(&Ohlcv::new(1000, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        ema.update(&Ohlcv::new(1100, 1.5, 1.5, 1.5, 1.5, 0.0)).unwrap(); // Same period, updates
        assert_eq!(ema.len(), 1);

        // Period 4: timestamps 1200-1499
        ema.update(&Ohlcv::new(1200, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        assert_eq!(ema.len(), 2);

        // Period 5: timestamps 1500-1799 - should be ready
        // First EMA = SMA of (1.5, 2.0, 3.0) = 6.5/3 ≈ 2.1667
        let result = ema.update(&Ohlcv::new(1500, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.1667).abs() < 0.01);
        assert_eq!(ema.len(), 3);

        // Period 6: timestamp 1800 - new period
        // EMA = (4 - 2.1667) * 0.5 + 2.1667 ≈ 3.0833
        let result = ema.update(&Ohlcv::new(1800, 4.0, 4.0, 4.0, 4.0, 0.0)).unwrap();
        assert!(result.is_some());
        assert!((result.unwrap() - 3.0833).abs() < 0.01);
        assert_eq!(ema.len(), 4);
    }

    #[test]
    fn ema_non_temporal_always_pushes() {
        // Non-temporal mode (timeframe = 0)
        let mut ema = EMA::new(EMAConfig::new(3));

        // All at same timestamp - should still push each one
        ema.update(&Ohlcv::new(1000, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        ema.update(&Ohlcv::new(1000, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        ema.update(&Ohlcv::new(1000, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();

        assert_eq!(ema.len(), 3);
        assert!(ema.state().is_ready());
    }

    #[test]
    fn ema_temporal_reset_preserves_config() {
        let mut ema = EMA::new(EMAConfig::with_timeframe(3, 60));

        // Add some data
        ema.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        ema.update(&Ohlcv::new(1020, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap();
        ema.update(&Ohlcv::new(1080, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();

        assert!(ema.is_temporal());
        assert_eq!(ema.timeframe(), 60);

        ema.reset();

        // Temporal config should be preserved
        assert!(ema.is_temporal());
        assert_eq!(ema.timeframe(), 60);
        assert!(ema.state().is_uninitialized());
    }

    #[test]
    fn ema_temporal_warmup_same_period_updates() {
        // Test same-period updates during warmup phase
        let mut ema = EMA::new(EMAConfig::with_timeframe(3, 60));

        // Period 16: multiple ticks, should update in place
        ema.update(&Ohlcv::new(960, 1.0, 1.0, 1.0, 1.0, 0.0)).unwrap();
        ema.update(&Ohlcv::new(990, 2.0, 2.0, 2.0, 2.0, 0.0)).unwrap(); // Same period
        assert_eq!(ema.len(), 1);

        // Period 17
        ema.update(&Ohlcv::new(1020, 3.0, 3.0, 3.0, 3.0, 0.0)).unwrap();
        assert_eq!(ema.len(), 2);

        // Period 18 - should be ready with SMA of (2, 3, 4) = 3.0
        let result = ema.update(&Ohlcv::new(1080, 4.0, 4.0, 4.0, 4.0, 0.0)).unwrap();
        assert!(result.is_some());
        // The window has values: 2.0 (updated), 3.0, 4.0 -> mean = 3.0
        assert!((result.unwrap() - 3.0).abs() < 0.0001);
    }
}
