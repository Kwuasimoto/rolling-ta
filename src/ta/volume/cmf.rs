//! Chaikin Money Flow (CMF) indicator.
//!
//! CMF measures the amount of Money Flow Volume over a specific period,
//! helping identify buying and selling pressure.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use std::collections::VecDeque;

use crate::ta::{
    config::CMFConfig,
    error::TAResult,
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Chaikin Money Flow indicator.
///
/// CMF oscillates between -1 and +1, measuring accumulation/distribution
/// pressure over a period.
///
/// # Formula
///
/// 1. Money Flow Multiplier (MFM) = ((Close - Low) - (High - Close)) / (High - Low)
///    = (2 × Close - High - Low) / (High - Low)
/// 2. Money Flow Volume (MFV) = MFM × Volume
/// 3. CMF = Sum of MFV over period / Sum of Volume over period
///
/// # Interpretation
///
/// - CMF > 0: Buying pressure (accumulation)
/// - CMF < 0: Selling pressure (distribution)
/// - CMF near 0: Equilibrium between buyers and sellers
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volume::{CMF, CMFConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut cmf = CMF::new(CMFConfig::new(20));
/// let candles = vec![
///     Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     // ... more candles
/// ];
/// // After warmup period, CMF will produce values
/// ```
#[derive(Debug, Clone)]
pub struct CMF {
    config: CMFConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,

    // Rolling windows for the period
    mfv_window: VecDeque<f64>,
    vol_window: VecDeque<f64>,
    mfv_sum: f64,
    vol_sum: f64,

    // Committed state for same-candle updates
    committed_mfv_window: VecDeque<f64>,
    committed_vol_window: VecDeque<f64>,
    committed_mfv_sum: f64,
    committed_vol_sum: f64,

    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl CMF {
    /// Create a new CMF indicator.
    pub fn new(config: CMFConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            mfv_window: VecDeque::with_capacity(config.period),
            vol_window: VecDeque::with_capacity(config.period),
            mfv_sum: 0.0,
            vol_sum: 0.0,
            committed_mfv_window: VecDeque::with_capacity(config.period),
            committed_vol_window: VecDeque::with_capacity(config.period),
            committed_mfv_sum: 0.0,
            committed_vol_sum: 0.0,
            last_len: 0,
        }
    }

    /// Get the current CMF value.
    #[inline]
    pub fn cmf_value(&self) -> f64 {
        self.latest.unwrap_or(f64::NAN)
    }

    /// Calculate Money Flow Multiplier.
    /// MFM = (2 × Close - High - Low) / (High - Low)
    #[inline]
    fn money_flow_multiplier(candle: &Ohlcv) -> f64 {
        let high = candle.high.0;
        let low = candle.low.0;
        let close = candle.close.0;

        let range = high - low;
        if range > 0.0 {
            (2.0 * close - high - low) / range
        } else {
            0.0 // No range means no directional pressure
        }
    }

    /// Calculate CMF from sums.
    #[inline]
    fn calculate_cmf(mfv_sum: f64, vol_sum: f64) -> f64 {
        if vol_sum > 0.0 {
            mfv_sum / vol_sum
        } else {
            0.0
        }
    }

    /// Add values to rolling windows.
    fn add_to_windows(&mut self, mfv: f64, vol: f64) {
        // Remove oldest if at capacity
        if self.mfv_window.len() >= self.config.period {
            if let Some(old_mfv) = self.mfv_window.pop_front() {
                self.mfv_sum -= old_mfv;
            }
            if let Some(old_vol) = self.vol_window.pop_front() {
                self.vol_sum -= old_vol;
            }
        }

        // Add new values
        self.mfv_window.push_back(mfv);
        self.vol_window.push_back(vol);
        self.mfv_sum += mfv;
        self.vol_sum += vol;
    }
}

impl Default for CMF {
    fn default() -> Self {
        Self::new(CMFConfig::default())
    }
}

impl Indicator for CMF {
    type Output = f64;
    type Config = CMFConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();
        let period = self.config.period;

        // Reset state
        self.history = Vec::with_capacity(n.saturating_sub(period - 1));
        self.mfv_window.clear();
        self.vol_window.clear();
        self.mfv_sum = 0.0;
        self.vol_sum = 0.0;
        self.latest = None;

        if n == 0 {
            self.last_len = 0;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Process all candles
        for i in 0..n {
            let candle = &data[i];
            let mfm = Self::money_flow_multiplier(candle);
            let vol = candle.volume.0;
            let mfv = mfm * vol;

            self.add_to_windows(mfv, vol);

            // After warmup, calculate CMF
            if i >= period - 1 {
                let cmf = Self::calculate_cmf(self.mfv_sum, self.vol_sum);
                self.history.push(cmf);
            }
        }

        self.latest = self.history.last().copied();
        self.last_len = n;

        // Copy to committed state
        self.committed_mfv_window = self.mfv_window.clone();
        self.committed_vol_window = self.vol_window.clone();
        self.committed_mfv_sum = self.mfv_sum;
        self.committed_vol_sum = self.vol_sum;

        self.state = IndicatorState::Ready;
        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len == 0 {
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First time - initialize from scratch
                for i in 0..len {
                    // Before processing the last candle, save committed state
                    if i == len - 1 {
                        self.committed_mfv_window = self.mfv_window.clone();
                        self.committed_vol_window = self.vol_window.clone();
                        self.committed_mfv_sum = self.mfv_sum;
                        self.committed_vol_sum = self.vol_sum;
                    }

                    let candle = &candles[i];
                    let mfm = Self::money_flow_multiplier(candle);
                    let vol = candle.volume.0;
                    let mfv = mfm * vol;

                    self.add_to_windows(mfv, vol);

                    // After warmup, calculate CMF
                    if i >= period - 1 {
                        let cmf = Self::calculate_cmf(self.mfv_sum, self.vol_sum);
                        self.history.push(cmf);
                    }
                }

                self.last_len = len;
                self.latest = self.history.last().copied();
                self.state = IndicatorState::Ready;
                return self.latest;
            }

            // New candle(s) - first commit the previous candle
            self.committed_mfv_window = self.mfv_window.clone();
            self.committed_vol_window = self.vol_window.clone();
            self.committed_mfv_sum = self.mfv_sum;
            self.committed_vol_sum = self.vol_sum;

            // Process new candles
            for i in self.last_len..len {
                let candle = &candles[i];
                let mfm = Self::money_flow_multiplier(candle);
                let vol = candle.volume.0;
                let mfv = mfm * vol;

                self.add_to_windows(mfv, vol);

                // After warmup, calculate CMF
                if i >= period - 1 {
                    let cmf = Self::calculate_cmf(self.mfv_sum, self.vol_sum);
                    self.history.push(cmf);
                }

                // For batch catch-up, commit each completed candle except the last
                if i < len - 1 {
                    self.committed_mfv_window = self.mfv_window.clone();
                    self.committed_vol_window = self.vol_window.clone();
                    self.committed_mfv_sum = self.mfv_sum;
                    self.committed_vol_sum = self.vol_sum;
                }
            }

            self.last_len = len;
            self.latest = self.history.last().copied();
        } else {
            // Same candle - compute tentatively without committing
            let current_candle = &candles[len - 1];
            let mfm = Self::money_flow_multiplier(current_candle);
            let vol = current_candle.volume.0;
            let mfv = mfm * vol;

            // Calculate tentative sums
            let mut tentative_mfv_sum = self.committed_mfv_sum;
            let mut tentative_vol_sum = self.committed_vol_sum;

            // If window is at capacity, account for removal
            if self.committed_mfv_window.len() >= period {
                tentative_mfv_sum -= self.committed_mfv_window.front().copied().unwrap_or(0.0);
                tentative_vol_sum -= self.committed_vol_window.front().copied().unwrap_or(0.0);
            }

            tentative_mfv_sum += mfv;
            tentative_vol_sum += vol;

            // Update working state for next call
            self.mfv_window = self.committed_mfv_window.clone();
            self.vol_window = self.committed_vol_window.clone();
            if self.mfv_window.len() >= period {
                self.mfv_window.pop_front();
                self.vol_window.pop_front();
            }
            self.mfv_window.push_back(mfv);
            self.vol_window.push_back(vol);
            self.mfv_sum = tentative_mfv_sum;
            self.vol_sum = tentative_vol_sum;

            // Calculate CMF if past warmup
            if len >= period {
                let cmf = Self::calculate_cmf(tentative_mfv_sum, tentative_vol_sum);
                // Update last history entry
                if !self.history.is_empty() {
                    *self.history.last_mut().unwrap() = cmf;
                }
                self.latest = Some(cmf);
            }
        }

        self.state = IndicatorState::Ready;
        self.latest
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.mfv_window.clear();
        self.vol_window.clear();
        self.mfv_sum = 0.0;
        self.vol_sum = 0.0;
        self.committed_mfv_window.clear();
        self.committed_vol_window.clear();
        self.committed_mfv_sum = 0.0;
        self.committed_vol_sum = 0.0;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for CMF {
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

    fn create_candles(data: &[(f64, f64, f64, f64, f64)]) -> Vec<Ohlcv> {
        // (open, high, low, close, volume)
        data.iter()
            .enumerate()
            .map(|(i, &(o, h, l, c, v))| Ohlcv::new(i as i64, o, h, l, c, v))
            .collect()
    }

    #[test]
    fn cmf_money_flow_multiplier() {
        // Close at high: MFM = (2*105 - 105 - 100) / (105 - 100) = 105/5 = 1.0
        let candle_at_high = Ohlcv::new(0, 100.0, 105.0, 100.0, 105.0, 1000.0);
        assert!((CMF::money_flow_multiplier(&candle_at_high) - 1.0).abs() < 1e-10);

        // Close at low: MFM = (2*100 - 105 - 100) / (105 - 100) = -5/5 = -1.0
        let candle_at_low = Ohlcv::new(0, 100.0, 105.0, 100.0, 100.0, 1000.0);
        assert!((CMF::money_flow_multiplier(&candle_at_low) - (-1.0)).abs() < 1e-10);

        // Close at midpoint: MFM = (2*102.5 - 105 - 100) / (105 - 100) = 0/5 = 0.0
        let candle_at_mid = Ohlcv::new(0, 100.0, 105.0, 100.0, 102.5, 1000.0);
        assert!((CMF::money_flow_multiplier(&candle_at_mid) - 0.0).abs() < 1e-10);

        // No range (high == low): MFM = 0
        let candle_no_range = Ohlcv::new(0, 100.0, 100.0, 100.0, 100.0, 1000.0);
        assert!((CMF::money_flow_multiplier(&candle_no_range) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn cmf_batch_calculation() {
        let mut cmf = CMF::new(CMFConfig::new(3));

        // All closes near high = strong buying pressure
        let candles = create_candles(&[
            (100.0, 105.0, 100.0, 104.0, 1000.0), // MFM ≈ 0.6
            (104.0, 108.0, 102.0, 107.0, 1000.0), // MFM ≈ 0.67
            (107.0, 110.0, 105.0, 109.0, 1000.0), // MFM = 0.6
        ]);

        cmf.calc(&candles).unwrap();

        assert!(cmf.state().is_ready());
        assert_eq!(cmf.len(), 1); // 3 - 3 + 1 = 1

        let value = cmf.latest().unwrap();
        assert!(value > 0.0, "CMF should be positive with closes near highs: {}", value);
    }

    #[test]
    fn cmf_strong_buying_pressure() {
        let mut cmf = CMF::new(CMFConfig::new(3));

        // All closes at high = maximum buying pressure
        let candles = create_candles(&[
            (100.0, 110.0, 100.0, 110.0, 1000.0), // MFM = 1.0
            (110.0, 120.0, 110.0, 120.0, 1000.0), // MFM = 1.0
            (120.0, 130.0, 120.0, 130.0, 1000.0), // MFM = 1.0
        ]);

        cmf.calc(&candles).unwrap();

        let value = cmf.latest().unwrap();
        assert!((value - 1.0).abs() < 1e-10, "CMF should be 1.0: {}", value);
    }

    #[test]
    fn cmf_strong_selling_pressure() {
        let mut cmf = CMF::new(CMFConfig::new(3));

        // All closes at low = maximum selling pressure
        let candles = create_candles(&[
            (110.0, 110.0, 100.0, 100.0, 1000.0), // MFM = -1.0
            (100.0, 100.0, 90.0, 90.0, 1000.0),   // MFM = -1.0
            (90.0, 90.0, 80.0, 80.0, 1000.0),     // MFM = -1.0
        ]);

        cmf.calc(&candles).unwrap();

        let value = cmf.latest().unwrap();
        assert!((value - (-1.0)).abs() < 1e-10, "CMF should be -1.0: {}", value);
    }

    #[test]
    fn cmf_neutral() {
        let mut cmf = CMF::new(CMFConfig::new(3));

        // All closes at midpoint = neutral
        let candles = create_candles(&[
            (100.0, 110.0, 100.0, 105.0, 1000.0), // MFM = 0.0
            (105.0, 115.0, 105.0, 110.0, 1000.0), // MFM = 0.0
            (110.0, 120.0, 110.0, 115.0, 1000.0), // MFM = 0.0
        ]);

        cmf.calc(&candles).unwrap();

        let value = cmf.latest().unwrap();
        assert!((value - 0.0).abs() < 1e-10, "CMF should be 0.0: {}", value);
    }

    #[test]
    fn cmf_streaming_matches_batch() {
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0, 1000.0),
            (103.0, 108.0, 101.0, 106.0, 1100.0),
            (106.0, 110.0, 104.0, 108.0, 900.0),
            (108.0, 112.0, 106.0, 107.0, 1200.0),
            (107.0, 109.0, 104.0, 105.0, 800.0),
            (105.0, 108.0, 103.0, 107.0, 1000.0),
        ]);

        // Batch calculation
        let mut batch = CMF::new(CMFConfig::new(3));
        batch.calc(&candles).unwrap();

        // Streaming calculation
        let mut stream = CMF::new(CMFConfig::new(3));
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        assert_eq!(batch.len(), stream.len(), "History lengths should match");

        for i in 0..batch.len() {
            let batch_val = batch.get(i as isize).unwrap();
            let stream_val = stream.get(i as isize).unwrap();
            assert!(
                (batch_val - stream_val).abs() < 1e-10,
                "Mismatch at index {}: batch={}, stream={}",
                i, batch_val, stream_val
            );
        }
    }

    #[test]
    fn cmf_same_candle_update() {
        let mut cmf = CMF::new(CMFConfig::new(3));

        // Build up initial state
        let mut candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0, 1000.0),
            (103.0, 108.0, 101.0, 106.0, 1000.0),
            (106.0, 110.0, 104.0, 110.0, 1000.0), // Close at high
        ]);

        // Process initial candles
        for i in 1..=candles.len() {
            cmf.next(&candles[..i]);
        }

        let cmf_initial = cmf.latest().unwrap();

        // Same candle update - close drops to low
        candles[2] = Ohlcv::new(2, 106.0, 110.0, 104.0, 104.0, 1000.0);
        cmf.next(&candles);
        let cmf_after_drop = cmf.latest().unwrap();

        // Same candle update - close back to high
        candles[2] = Ohlcv::new(2, 106.0, 110.0, 104.0, 110.0, 1000.0);
        cmf.next(&candles);
        let cmf_after_recover = cmf.latest().unwrap();

        assert!(cmf_after_drop < cmf_initial,
            "CMF should drop when close moves to low: initial={}, after_drop={}",
            cmf_initial, cmf_after_drop);

        assert!((cmf_after_recover - cmf_initial).abs() < 1e-10,
            "CMF should recover: initial={}, after_recover={}",
            cmf_initial, cmf_after_recover);
    }

    #[test]
    fn cmf_values_in_range() {
        let mut cmf = CMF::new(CMFConfig::new(3));

        // Mixed price movements
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0, 1000.0),
            (103.0, 108.0, 100.0, 101.0, 1100.0),
            (101.0, 106.0, 99.0, 105.0, 900.0),
            (105.0, 110.0, 103.0, 104.0, 1200.0),
            (104.0, 108.0, 102.0, 107.0, 800.0),
        ]);

        cmf.calc(&candles).unwrap();

        for value in cmf.history() {
            assert!(
                *value >= -1.0 && *value <= 1.0,
                "CMF should be between -1 and 1, got {}",
                value
            );
        }
    }

    #[test]
    fn cmf_reset() {
        let mut cmf = CMF::new(CMFConfig::new(3));
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0, 1000.0),
            (103.0, 108.0, 101.0, 106.0, 1000.0),
            (106.0, 110.0, 104.0, 108.0, 1000.0),
        ]);

        cmf.calc(&candles).unwrap();
        assert!(cmf.state().is_ready());
        assert!(!cmf.is_empty());

        cmf.reset();
        assert!(cmf.state().is_uninitialized());
        assert!(cmf.is_empty());
        assert!(cmf.latest().is_none());
    }

    #[test]
    fn cmf_warmup_period() {
        let cmf = CMF::new(CMFConfig::new(20));
        assert_eq!(cmf.warmup_period(), 20);
    }

    #[test]
    fn cmf_empty_data() {
        let mut cmf = CMF::default();
        let candles: Vec<Ohlcv> = vec![];

        cmf.calc(&candles).unwrap();
        assert!(cmf.state().is_ready());
        assert!(cmf.is_empty());
        assert!(cmf.latest().is_none());
    }

    #[test]
    fn cmf_insufficient_data() {
        let mut cmf = CMF::new(CMFConfig::new(20));
        let candles = create_candles(&[
            (100.0, 105.0, 98.0, 103.0, 1000.0),
            (103.0, 108.0, 101.0, 106.0, 1000.0),
        ]);

        cmf.calc(&candles).unwrap();
        assert!(cmf.state().is_ready());
        assert!(cmf.is_empty()); // Not enough data for CMF
    }
}
