//! Hull Moving Average (HMA) indicator.
//!
//! HMA is a fast and smooth moving average that reduces lag significantly
//! while maintaining curve smoothness.

use std::collections::VecDeque;

use crate::ta::{
    config::{HMAConfig, WMAConfig},
    error::{TAError, TAResult},
    state::IndicatorState,
    types::{Ohlcv, OhlcvSeries},
    HistoricalIndicator, Indicator,
};

use super::WMA;

/// Hull Moving Average indicator.
///
/// Composes WMA to create a fast, smooth moving average.
///
/// # Formula
///
/// 1. WMA_half = WMA(close, period/2)
/// 2. WMA_full = WMA(close, period)
/// 3. Raw HMA = 2 * WMA_half - WMA_full
/// 4. HMA = WMA(Raw HMA, sqrt(period))
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{HMA, HMAConfig};
/// use rolling_ta::ta::types::OhlcvSeries;
/// use rolling_ta::ta::Indicator;
///
/// let mut hma = HMA::new(HMAConfig::new(16));
/// let closes: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
/// let data = OhlcvSeries::from_closes(&closes);
/// hma.calc(&data).unwrap();
///
/// assert!(hma.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct HMA {
    config: HMAConfig,
    state: IndicatorState,
    wma_full: WMA,
    wma_half: WMA,
    // For final WMA calculation
    sqrt_period: usize,
    weight_sum: usize,
    interim_window: VecDeque<f64>,
    history: Vec<f64>,
    latest: Option<f64>,
}

impl HMA {
    /// Create a new HMA indicator.
    pub fn new(config: HMAConfig) -> Self {
        let half_period = config.half_period().max(1);
        let sqrt_period = config.sqrt_period().max(1);
        let weight_sum = sqrt_period * (sqrt_period + 1) / 2;

        Self {
            config,
            state: IndicatorState::Uninitialized,
            wma_full: WMA::new(WMAConfig::new(config.period)),
            wma_half: WMA::new(WMAConfig::new(half_period)),
            sqrt_period,
            weight_sum,
            interim_window: VecDeque::with_capacity(sqrt_period),
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate WMA over a VecDeque using weights 1, 2, 3, ..., n
    #[inline]
    fn wma_deque(data: &VecDeque<f64>, weight_sum: usize) -> f64 {
        let mut weighted_sum = 0.0;
        for (i, &val) in data.iter().enumerate() {
            weighted_sum += val * (i + 1) as f64;
        }
        weighted_sum / weight_sum as f64
    }
}

impl Default for HMA {
    fn default() -> Self {
        Self::new(HMAConfig::default())
    }
}

impl Indicator for HMA {
    type Output = f64;
    type Config = HMAConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let n = data.len();
        let period = self.config.period;

        // Need period + sqrt_period - 1 for first valid HMA
        let min_required = period + self.sqrt_period - 1;
        if n < min_required {
            return Err(TAError::InsufficientData {
                required: min_required,
                actual: n,
            });
        }

        // Calculate both WMAs
        self.wma_full.calc(data)?;
        self.wma_half.calc(data)?;

        let wma_full_hist = self.wma_full.history();
        let wma_half_hist = self.wma_half.history();

        // Calculate interim values: 2 * wma_half - wma_full
        // Interim is valid from period - 1 (when wma_full becomes valid)
        let mut interim: Vec<f64> = vec![f64::NAN; n];
        for i in (period - 1)..n {
            interim[i] = 2.0 * wma_half_hist[i] - wma_full_hist[i];
        }

        // Reset HMA state
        self.history = vec![f64::NAN; n];

        // Calculate final HMA as WMA of interim with sqrt_period
        // First valid HMA at index: period - 1 + sqrt_period - 1
        let hma_start = period + self.sqrt_period - 2;

        for i in hma_start..n {
            let window_start = i + 1 - self.sqrt_period;
            let window = &interim[window_start..=i];

            let mut weighted_sum = 0.0;
            for (j, &val) in window.iter().enumerate() {
                weighted_sum += val * (j + 1) as f64;
            }
            self.history[i] = weighted_sum / self.weight_sum as f64;
        }

        // Set up interim window for streaming
        self.interim_window.clear();
        let interim_start = n - self.sqrt_period;
        for &v in &interim[interim_start..] {
            self.interim_window.push_back(v);
        }

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        // Update both WMAs
        let full_result = self.wma_full.update(tick)?;
        let half_result = self.wma_half.update(tick)?;

        // Need both WMAs to calculate interim
        match (full_result, half_result) {
            (Some(full), Some(half)) => {
                let interim = 2.0 * half - full;

                // Maintain interim window
                if self.interim_window.len() >= self.sqrt_period {
                    self.interim_window.pop_front();
                }
                self.interim_window.push_back(interim);

                // Need full window for HMA
                if self.interim_window.len() < self.sqrt_period {
                    self.history.push(f64::NAN);
                    self.state = IndicatorState::Warming {
                        count: self.config.period + self.interim_window.len() - 1,
                    };
                    return Ok(None);
                }

                // Calculate HMA
                let hma = Self::wma_deque(&self.interim_window, self.weight_sum);
                self.history.push(hma);
                self.latest = Some(hma);
                self.state = IndicatorState::Ready;

                Ok(Some(hma))
            }
            _ => {
                // Still warming up WMAs
                self.history.push(f64::NAN);
                let count = self.wma_full.history().len().max(self.wma_half.history().len());
                self.state = IndicatorState::Warming { count };
                Ok(None)
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.wma_full.reset();
        self.wma_half.reset();
        self.interim_window.clear();
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period + self.sqrt_period - 1
    }
}

impl HistoricalIndicator for HMA {
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

    fn generate_trend_data(n: usize) -> OhlcvSeries {
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        OhlcvSeries::from_closes(&closes)
    }

    #[test]
    fn hma_batch_calculation() {
        let mut hma = HMA::new(HMAConfig::new(9));
        let data = generate_trend_data(30);

        hma.calc(&data).unwrap();

        assert!(hma.state().is_ready());
        assert_eq!(hma.len(), 30);

        // Warmup period = 9 + 3 - 1 = 11, so first 10 should be NaN
        for i in 0..10 {
            assert!(
                hma.get(i as isize).unwrap().is_nan(),
                "Index {} should be NaN",
                i
            );
        }

        // From index 10, should have valid values
        let value = hma.get(10).unwrap();
        assert!(!value.is_nan(), "Index 10 should be valid");
    }

    #[test]
    fn hma_faster_than_sma() {
        // HMA should respond faster to price changes during transition
        use crate::ta::trend::SMA;

        let mut hma = HMA::new(HMAConfig::new(9));
        let mut sma = SMA::new(crate::ta::config::SMAConfig::new(9));

        // Create data with sudden price jump (check RIGHT after the jump)
        let mut closes: Vec<f64> = vec![100.0; 20];
        closes.extend(vec![110.0; 10]); // Sudden jump
        let data = OhlcvSeries::from_closes(&closes);

        // Right after the jump, HMA should have moved more than SMA
        hma.calc(&data).unwrap();
        sma.calc(&data).unwrap();

        // After the jump, HMA should be closer to 110 than SMA
        let hma_latest = hma.latest().unwrap();
        let sma_latest = sma.latest().unwrap();

        // Both should be between 100 and 110, but HMA should be higher (faster response)
        assert!(
            hma_latest > sma_latest,
            "HMA ({}) should respond faster than SMA ({}) to price jump",
            hma_latest,
            sma_latest
        );
    }

    #[test]
    fn hma_streaming_update() {
        let mut hma = HMA::new(HMAConfig::new(9));

        // Feed data one by one
        for i in 0..30 {
            let tick = Ohlcv::from_close(100.0 + i as f64 * 0.5);
            let _ = hma.update(&tick);
        }

        assert!(hma.state().is_ready());
        let latest = hma.latest();
        assert!(latest.is_some());
        assert!(!latest.unwrap().is_nan());
    }

    #[test]
    fn hma_batch_vs_streaming() {
        let mut hma_batch = HMA::new(HMAConfig::new(9));
        let mut hma_stream = HMA::new(HMAConfig::new(9));

        let data = generate_trend_data(30);

        // Batch
        hma_batch.calc(&data).unwrap();

        // Streaming
        for i in 0..30 {
            let tick = Ohlcv::from_close(data.closes[i]);
            let _ = hma_stream.update(&tick);
        }

        // Compare results
        let batch_latest = hma_batch.latest().unwrap();
        let stream_latest = hma_stream.latest().unwrap();

        assert!(
            (batch_latest - stream_latest).abs() < 1e-10,
            "Batch ({}) and stream ({}) should match",
            batch_latest,
            stream_latest
        );
    }

    #[test]
    fn hma_reset() {
        let mut hma = HMA::new(HMAConfig::new(9));
        let data = generate_trend_data(30);

        hma.calc(&data).unwrap();
        assert!(hma.state().is_ready());

        hma.reset();
        assert!(hma.state().is_uninitialized());
        assert_eq!(hma.len(), 0);
    }
}
