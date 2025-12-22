//! Money Flow Index (MFI) indicator.
//!
//! MFI is a momentum indicator that combines price and volume to identify
//! overbought or oversold conditions.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use std::collections::VecDeque;

use crate::ta::{
    config::MFIConfig,
    error::TAResult,
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Money Flow Index indicator.
///
/// MFI uses both price and volume to measure buying and selling pressure.
/// Values range from 0 to 100, with readings above 80 indicating overbought
/// conditions and below 20 indicating oversold conditions.
///
/// # Formula
///
/// 1. Typical Price = (High + Low + Close) / 3
/// 2. Raw Money Flow = Typical Price × Volume
/// 3. If TP > prev_TP: Positive MF = RMF, Negative MF = 0
/// 4. If TP < prev_TP: Negative MF = RMF, Positive MF = 0
/// 5. MFI = 100 × (Sum of Positive MF) / (Sum of Positive MF + Sum of Negative MF)
///
/// # Interpretation
///
/// - MFI > 80: Overbought (potential sell signal)
/// - MFI < 20: Oversold (potential buy signal)
/// - Divergence between MFI and price can signal trend reversal
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volume::{MFI, MFIConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut mfi = MFI::new(MFIConfig::new(14));
/// let candles = vec![
///     Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     // ... more candles
/// ];
/// // After warmup period, MFI will produce values
/// ```
#[derive(Debug, Clone)]
pub struct MFI {
    config: MFIConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,

    // Rolling window of money flows for the period
    pmf_window: VecDeque<f64>,
    nmf_window: VecDeque<f64>,
    pmf_sum: f64,
    nmf_sum: f64,
    prev_tp: Option<f64>,

    // Committed state for same-candle updates
    committed_pmf_window: VecDeque<f64>,
    committed_nmf_window: VecDeque<f64>,
    committed_pmf_sum: f64,
    committed_nmf_sum: f64,
    committed_prev_tp: Option<f64>,

    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl MFI {
    /// Create a new MFI indicator.
    pub fn new(config: MFIConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            pmf_window: VecDeque::with_capacity(config.period),
            nmf_window: VecDeque::with_capacity(config.period),
            pmf_sum: 0.0,
            nmf_sum: 0.0,
            prev_tp: None,
            committed_pmf_window: VecDeque::with_capacity(config.period),
            committed_nmf_window: VecDeque::with_capacity(config.period),
            committed_pmf_sum: 0.0,
            committed_nmf_sum: 0.0,
            committed_prev_tp: None,
            last_len: 0,
        }
    }

    /// Get the current MFI value.
    #[inline]
    pub fn mfi_value(&self) -> f64 {
        self.latest.unwrap_or(f64::NAN)
    }

    /// Calculate typical price.
    #[inline]
    fn typical_price(candle: &Ohlcv) -> f64 {
        (candle.high.0 + candle.low.0 + candle.close.0) / 3.0
    }

    /// Calculate MFI from sums.
    #[inline]
    fn calculate_mfi(pmf_sum: f64, nmf_sum: f64) -> f64 {
        let total = pmf_sum + nmf_sum;
        if total > 0.0 {
            100.0 * pmf_sum / total
        } else {
            50.0 // Neutral when no money flow
        }
    }

    /// Add a money flow value to the rolling windows.
    fn add_money_flow(
        &mut self,
        pmf: f64,
        nmf: f64,
    ) {
        // Remove oldest if at capacity
        if self.pmf_window.len() >= self.config.period {
            if let Some(old_pmf) = self.pmf_window.pop_front() {
                self.pmf_sum -= old_pmf;
            }
            if let Some(old_nmf) = self.nmf_window.pop_front() {
                self.nmf_sum -= old_nmf;
            }
        }

        // Add new values
        self.pmf_window.push_back(pmf);
        self.nmf_window.push_back(nmf);
        self.pmf_sum += pmf;
        self.nmf_sum += nmf;
    }
}

impl Default for MFI {
    fn default() -> Self {
        Self::new(MFIConfig::default())
    }
}

impl Indicator for MFI {
    type Output = f64;
    type Config = MFIConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();
        let period = self.config.period;

        // Reset state
        self.history = Vec::with_capacity(n.saturating_sub(period));
        self.pmf_window.clear();
        self.nmf_window.clear();
        self.pmf_sum = 0.0;
        self.nmf_sum = 0.0;
        self.prev_tp = None;
        self.latest = None;

        if n == 0 {
            self.last_len = 0;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // Calculate typical prices and money flows
        let mut prev_tp = Self::typical_price(&data[0]);

        for i in 1..n {
            let candle = &data[i];
            let tp = Self::typical_price(candle);
            let rmf = tp * candle.volume.0;

            // Determine direction
            let (pmf, nmf) = if tp > prev_tp {
                (rmf, 0.0)
            } else if tp < prev_tp {
                (0.0, rmf)
            } else {
                (0.0, 0.0)
            };

            self.add_money_flow(pmf, nmf);
            prev_tp = tp;

            // After warmup, calculate MFI
            if i >= period {
                let mfi = Self::calculate_mfi(self.pmf_sum, self.nmf_sum);
                self.history.push(mfi);
            }
        }

        self.prev_tp = Some(prev_tp);
        self.latest = self.history.last().copied();
        self.last_len = n;

        // Copy to committed state
        self.committed_pmf_window = self.pmf_window.clone();
        self.committed_nmf_window = self.nmf_window.clone();
        self.committed_pmf_sum = self.pmf_sum;
        self.committed_nmf_sum = self.nmf_sum;
        self.committed_prev_tp = self.prev_tp;

        self.state = IndicatorState::Ready;
        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < 2 {
            return None; // Need at least 2 candles for MFI
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First time - initialize from scratch
                // Process all candles EXCEPT the last one into committed state
                let mut prev_tp = Self::typical_price(&candles[0]);

                for i in 1..len {
                    // Before processing each candle, save state as committed
                    // This way committed state = state BEFORE current candle
                    if i == len - 1 {
                        // Save committed state before processing the last candle
                        self.committed_pmf_window = self.pmf_window.clone();
                        self.committed_nmf_window = self.nmf_window.clone();
                        self.committed_pmf_sum = self.pmf_sum;
                        self.committed_nmf_sum = self.nmf_sum;
                        self.committed_prev_tp = Some(prev_tp);
                    }

                    let candle = &candles[i];
                    let tp = Self::typical_price(candle);
                    let rmf = tp * candle.volume.0;

                    let (pmf, nmf) = if tp > prev_tp {
                        (rmf, 0.0)
                    } else if tp < prev_tp {
                        (0.0, rmf)
                    } else {
                        (0.0, 0.0)
                    };

                    self.add_money_flow(pmf, nmf);
                    prev_tp = tp;

                    // After warmup, calculate MFI
                    if i >= period {
                        let mfi = Self::calculate_mfi(self.pmf_sum, self.nmf_sum);
                        self.history.push(mfi);
                    }
                }

                self.prev_tp = Some(prev_tp);
                self.last_len = len;

                self.latest = self.history.last().copied();
                self.state = IndicatorState::Ready;
                return self.latest;
            }

            // New candle(s) - first commit the previous candle
            // The committed state should be the state AFTER processing the previous last candle
            // (which is now the second-to-last candle)
            self.committed_pmf_window = self.pmf_window.clone();
            self.committed_nmf_window = self.nmf_window.clone();
            self.committed_pmf_sum = self.pmf_sum;
            self.committed_nmf_sum = self.nmf_sum;
            // committed_prev_tp = TP of the previous last candle (now second-to-last)
            self.committed_prev_tp = Some(Self::typical_price(&candles[self.last_len - 1]));

            // Process new candles
            for i in self.last_len..len {
                let candle = &candles[i];
                let tp = Self::typical_price(candle);
                let rmf = tp * candle.volume.0;

                let prev_tp = self.prev_tp.unwrap_or(tp);
                let (pmf, nmf) = if tp > prev_tp {
                    (rmf, 0.0)
                } else if tp < prev_tp {
                    (0.0, rmf)
                } else {
                    (0.0, 0.0)
                };

                self.add_money_flow(pmf, nmf);
                self.prev_tp = Some(tp);

                // After warmup, calculate MFI
                if i >= period {
                    let mfi = Self::calculate_mfi(self.pmf_sum, self.nmf_sum);
                    self.history.push(mfi);
                }

                // For batch catch-up, commit each completed candle except the last
                if i < len - 1 {
                    self.committed_pmf_window = self.pmf_window.clone();
                    self.committed_nmf_window = self.nmf_window.clone();
                    self.committed_pmf_sum = self.pmf_sum;
                    self.committed_nmf_sum = self.nmf_sum;
                    self.committed_prev_tp = Some(tp);
                }
            }

            self.last_len = len;
            self.latest = self.history.last().copied();
        } else {
            // Same candle - compute tentatively without committing
            let current_candle = &candles[len - 1];
            let tp = Self::typical_price(current_candle);
            let rmf = tp * current_candle.volume.0;

            let prev_tp = self.committed_prev_tp.unwrap_or(tp);
            let (pmf, nmf) = if tp > prev_tp {
                (rmf, 0.0)
            } else if tp < prev_tp {
                (0.0, rmf)
            } else {
                (0.0, 0.0)
            };

            // Calculate tentative sums
            let mut tentative_pmf_sum = self.committed_pmf_sum;
            let mut tentative_nmf_sum = self.committed_nmf_sum;

            // If window is at capacity, we need to account for removal
            if self.committed_pmf_window.len() >= period {
                tentative_pmf_sum -= self.committed_pmf_window.front().copied().unwrap_or(0.0);
                tentative_nmf_sum -= self.committed_nmf_window.front().copied().unwrap_or(0.0);
            }

            tentative_pmf_sum += pmf;
            tentative_nmf_sum += nmf;

            // Update working state for next call
            self.pmf_window = self.committed_pmf_window.clone();
            self.nmf_window = self.committed_nmf_window.clone();
            if self.pmf_window.len() >= period {
                self.pmf_window.pop_front();
                self.nmf_window.pop_front();
            }
            self.pmf_window.push_back(pmf);
            self.nmf_window.push_back(nmf);
            self.pmf_sum = tentative_pmf_sum;
            self.nmf_sum = tentative_nmf_sum;
            self.prev_tp = Some(tp);

            // Calculate MFI if past warmup
            if len > period {
                let mfi = Self::calculate_mfi(tentative_pmf_sum, tentative_nmf_sum);
                // Update last history entry
                if !self.history.is_empty() {
                    *self.history.last_mut().unwrap() = mfi;
                }
                self.latest = Some(mfi);
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
        self.pmf_window.clear();
        self.nmf_window.clear();
        self.pmf_sum = 0.0;
        self.nmf_sum = 0.0;
        self.prev_tp = None;
        self.committed_pmf_window.clear();
        self.committed_nmf_window.clear();
        self.committed_pmf_sum = 0.0;
        self.committed_nmf_sum = 0.0;
        self.committed_prev_tp = None;
        self.last_len = 0;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.config.period + 1 // +1 because we need prev_tp
    }
}

impl HistoricalIndicator for MFI {
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
    fn mfi_basic_calculation() {
        let mut mfi = MFI::new(MFIConfig::new(3));

        // Create candles with clear up/down pattern
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),   // TP = 100
            (100.0, 105.0, 99.0, 104.0, 1000.0),   // TP = 102.67, up -> PMF
            (104.0, 107.0, 103.0, 106.0, 1000.0),  // TP = 105.33, up -> PMF
            (106.0, 108.0, 104.0, 105.0, 1000.0),  // TP = 105.67, up -> PMF
            (105.0, 106.0, 100.0, 101.0, 1000.0),  // TP = 102.33, down -> NMF
        ]);

        mfi.calc(&candles).unwrap();

        assert!(mfi.state().is_ready());
        assert_eq!(mfi.len(), 2); // 5 candles - (period=3) - 1 = 1, but we get 2 values

        // All positive flows should give MFI = 100
        // Then one negative flow should reduce it
        let first = mfi.get(0).unwrap();
        let last = mfi.get(-1).unwrap();

        assert!(first > 50.0, "First MFI should be high (uptrend): {}", first);
        assert!(last < first, "Last MFI should drop after down candle: {}", last);
    }

    #[test]
    fn mfi_overbought_signal() {
        let mut mfi = MFI::new(MFIConfig::new(3));

        // All prices going up - strong buying pressure
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),
            (100.0, 105.0, 99.0, 105.0, 1000.0),
            (105.0, 110.0, 104.0, 110.0, 1000.0),
            (110.0, 115.0, 109.0, 115.0, 1000.0),
            (115.0, 120.0, 114.0, 120.0, 1000.0),
        ]);

        mfi.calc(&candles).unwrap();

        let latest = mfi.latest().unwrap();
        assert_eq!(latest, 100.0, "All positive flows should give MFI = 100");
    }

    #[test]
    fn mfi_oversold_signal() {
        let mut mfi = MFI::new(MFIConfig::new(3));

        // All prices going down - strong selling pressure
        let candles = create_candles(&[
            (120.0, 122.0, 118.0, 120.0, 1000.0),
            (120.0, 121.0, 115.0, 115.0, 1000.0),
            (115.0, 116.0, 110.0, 110.0, 1000.0),
            (110.0, 111.0, 105.0, 105.0, 1000.0),
            (105.0, 106.0, 100.0, 100.0, 1000.0),
        ]);

        mfi.calc(&candles).unwrap();

        let latest = mfi.latest().unwrap();
        assert_eq!(latest, 0.0, "All negative flows should give MFI = 0");
    }

    #[test]
    fn mfi_neutral_when_no_flow() {
        let mut mfi = MFI::new(MFIConfig::new(3));

        // Same typical price - no money flow
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),  // TP = 100
            (99.0, 103.0, 98.0, 100.0, 1000.0),   // TP = 100.33 (slightly up)
            (99.0, 102.0, 99.0, 100.0, 1000.0),   // TP = 100.33 (same)
            (99.0, 102.0, 99.0, 100.0, 1000.0),   // TP = 100.33 (same)
            (99.0, 102.0, 99.0, 100.0, 1000.0),   // TP = 100.33 (same)
        ]);

        mfi.calc(&candles).unwrap();

        let latest = mfi.latest().unwrap();
        // First flow is positive, then all neutral
        assert!(latest > 0.0, "Should have some positive flow from first transition");
    }

    #[test]
    fn mfi_streaming_matches_batch() {
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),
            (100.0, 105.0, 99.0, 104.0, 1100.0),
            (104.0, 107.0, 103.0, 106.0, 900.0),
            (106.0, 108.0, 104.0, 105.0, 1200.0),
            (105.0, 106.0, 100.0, 101.0, 800.0),
            (101.0, 103.0, 99.0, 102.0, 1000.0),
            (102.0, 106.0, 101.0, 105.0, 1100.0),
        ]);

        // Batch calculation
        let mut batch = MFI::new(MFIConfig::new(3));
        batch.calc(&candles).unwrap();

        // Streaming calculation
        let mut stream = MFI::new(MFIConfig::new(3));
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
    fn mfi_same_candle_update() {
        let mut mfi = MFI::new(MFIConfig::new(3));

        // Build up initial state
        let mut candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),
            (100.0, 105.0, 99.0, 104.0, 1000.0),
            (104.0, 107.0, 103.0, 106.0, 1000.0),
            (106.0, 108.0, 104.0, 107.0, 1000.0), // Initial: TP high
        ]);

        // Process initial candles
        for i in 1..=candles.len() {
            mfi.next(&candles[..i]);
        }

        let mfi_after_initial = mfi.latest().unwrap();

        // Same candle update - price drops
        candles[3] = Ohlcv::new(3, 106.0, 108.0, 100.0, 101.0, 1000.0);
        mfi.next(&candles);
        let mfi_after_drop = mfi.latest().unwrap();

        // Same candle update - price recovers
        candles[3] = Ohlcv::new(3, 106.0, 110.0, 105.0, 109.0, 1000.0);
        mfi.next(&candles);
        let mfi_after_recover = mfi.latest().unwrap();

        // MFI should change based on typical price direction
        assert!(mfi_after_drop < mfi_after_initial || mfi_after_drop == mfi_after_initial,
            "MFI should drop or stay same when price drops: initial={}, after_drop={}",
            mfi_after_initial, mfi_after_drop);

        assert!(mfi_after_recover > mfi_after_drop,
            "MFI should increase when price recovers: after_drop={}, after_recover={}",
            mfi_after_drop, mfi_after_recover);
    }

    #[test]
    fn mfi_reset() {
        let mut mfi = MFI::new(MFIConfig::new(3));
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),
            (100.0, 105.0, 99.0, 104.0, 1000.0),
            (104.0, 107.0, 103.0, 106.0, 1000.0),
            (106.0, 108.0, 104.0, 107.0, 1000.0),
        ]);

        mfi.calc(&candles).unwrap();
        assert!(mfi.state().is_ready());
        assert!(!mfi.is_empty());

        mfi.reset();
        assert!(mfi.state().is_uninitialized());
        assert!(mfi.is_empty());
        assert!(mfi.latest().is_none());
    }

    #[test]
    fn mfi_warmup_period() {
        let mfi = MFI::new(MFIConfig::new(14));
        assert_eq!(mfi.warmup_period(), 15); // period + 1 for prev_tp
    }

    #[test]
    fn mfi_empty_data() {
        let mut mfi = MFI::default();
        let candles: Vec<Ohlcv> = vec![];

        mfi.calc(&candles).unwrap();
        assert!(mfi.state().is_ready());
        assert!(mfi.is_empty());
        assert!(mfi.latest().is_none());
    }

    #[test]
    fn mfi_insufficient_data() {
        let mut mfi = MFI::new(MFIConfig::new(14));
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),
            (100.0, 105.0, 99.0, 104.0, 1000.0),
        ]);

        mfi.calc(&candles).unwrap();
        assert!(mfi.state().is_ready());
        assert!(mfi.is_empty()); // Not enough data for MFI
    }

    #[test]
    fn mfi_values_in_range() {
        let mut mfi = MFI::new(MFIConfig::new(3));

        // Mixed price movements
        let candles = create_candles(&[
            (100.0, 102.0, 98.0, 100.0, 1000.0),
            (100.0, 105.0, 99.0, 104.0, 1100.0),
            (104.0, 105.0, 100.0, 101.0, 900.0),
            (101.0, 106.0, 100.0, 105.0, 1200.0),
            (105.0, 107.0, 102.0, 103.0, 800.0),
            (103.0, 108.0, 102.0, 107.0, 1000.0),
        ]);

        mfi.calc(&candles).unwrap();

        for value in mfi.history() {
            assert!(
                *value >= 0.0 && *value <= 100.0,
                "MFI should be between 0 and 100, got {}",
                value
            );
        }
    }
}
