//! Exponential Moving Average (EMA) indicator.

use crate::ta::{
    config::EMAConfig,
    error::{TAError, TAResult},
    math::{ema_multiplier, ema_step},
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
            warmup_sum: 0.0,
            warmup_count: 0,
            history: Vec::new(),
            latest: None,
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
        let period = self.config.period;

        self.warmup_count += 1;

        if self.warmup_count < period {
            // Still warming up - accumulate for SMA seed
            self.warmup_sum += close;
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
            self.latest = Some(ema);
            self.history.push(ema);
            self.state = IndicatorState::Ready;
            Ok(Some(ema))
        } else {
            // Normal EMA calculation
            let prev = self.prev_ema.unwrap();
            let ema = ema_step(close, prev, self.multiplier);
            self.prev_ema = Some(ema);
            self.latest = Some(ema);
            self.history.push(ema);
            Ok(Some(ema))
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
}
