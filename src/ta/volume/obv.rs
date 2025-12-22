//! On-Balance Volume (OBV) indicator.
//!
//! OBV is a momentum indicator that uses volume flow to predict price changes.
//!
//! This indicator computes directly from `&[Ohlcv]` slices, making it suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::OBVConfig,
    error::TAResult,
    state::IndicatorState,
    types::Ohlcv,
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
/// # Interpretation
///
/// - Rising OBV with rising price confirms uptrend
/// - Falling OBV with falling price confirms downtrend
/// - Divergence between OBV and price can signal reversal
///
/// # Example
///
/// ```
/// use rolling_ta::ta::volume::{OBV, OBVConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut obv = OBV::default();
/// let candles = vec![
///     Ohlcv::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
///     Ohlcv::new(1, 103.0, 108.0, 101.0, 106.0, 1100.0),
///     Ohlcv::new(2, 106.0, 107.0, 102.0, 104.0, 900.0),
/// ];
/// obv.calc(&candles).unwrap();
///
/// assert!(obv.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct OBV {
    config: OBVConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Committed OBV value (after confirmed candles)
    committed_obv: f64,
    /// Committed previous close for next comparison
    committed_prev_close: f64,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl OBV {
    /// Create a new OBV indicator.
    pub fn new(config: OBVConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
            committed_obv: 0.0,
            committed_prev_close: 0.0,
            last_len: 0,
        }
    }

    /// Get the current OBV value.
    #[inline]
    pub fn obv_value(&self) -> f64 {
        self.latest.unwrap_or(0.0)
    }

    /// Calculate OBV step based on price direction.
    #[inline]
    fn calculate_obv_change(close: f64, prev_close: f64, volume: f64) -> f64 {
        if close > prev_close {
            volume
        } else if close < prev_close {
            -volume
        } else {
            0.0
        }
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

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let n = data.len();

        // Reset state
        self.history = Vec::with_capacity(n);

        if n == 0 {
            self.committed_obv = 0.0;
            self.committed_prev_close = 0.0;
            self.latest = None;
            self.last_len = 0;
            self.state = IndicatorState::Ready;
            return Ok(self);
        }

        // First candle: OBV starts at 0
        let mut obv = 0.0;
        self.history.push(0.0);

        // Subsequent candles
        for i in 1..n {
            let close = data[i].close.0;
            let prev_close = data[i - 1].close.0;
            let volume = data[i].volume.0;

            obv += Self::calculate_obv_change(close, prev_close, volume);
            self.history.push(obv);
        }

        self.committed_obv = obv;
        self.committed_prev_close = data[n - 1].close.0;
        self.latest = self.history.last().copied();
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();

        if len == 0 {
            return None;
        }

        // Determine if this is a new candle or same snapshot
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First candle ever - OBV starts at 0
                self.committed_obv = 0.0;
                self.committed_prev_close = candles[0].close.0;
                self.history.push(0.0);
                self.last_len = 1;
                self.latest = Some(0.0);
                self.state = IndicatorState::Ready;

                // If there are more candles, process them (batch catch-up)
                for i in 1..len {
                    let close = candles[i].close.0;
                    let volume = candles[i].volume.0;

                    self.committed_obv +=
                        Self::calculate_obv_change(close, self.committed_prev_close, volume);
                    self.committed_prev_close = close;
                    self.history.push(self.committed_obv);
                    self.last_len = i + 1;
                }

                self.latest = Some(self.committed_obv);
                return self.latest;
            }

            // New candle(s) added - first commit the previous tentative candle
            // The committed state represents everything BEFORE the last history entry
            // When a new candle arrives, we need to commit the previous one first
            if self.last_len >= 1 && !self.history.is_empty() {
                // Commit the previous candle: update committed state to include it
                self.committed_obv = *self.history.last().unwrap();
                self.committed_prev_close = candles[self.last_len - 1].close.0;
            }

            // Process all new candles since last_len
            for i in self.last_len..len {
                let close = candles[i].close.0;
                let volume = candles[i].volume.0;

                let new_obv = self.committed_obv
                    + Self::calculate_obv_change(close, self.committed_prev_close, volume);
                self.history.push(new_obv);

                // For multiple new candles in batch, commit each as we go
                if i < len - 1 {
                    self.committed_obv = new_obv;
                    self.committed_prev_close = close;
                }
            }

            self.last_len = len;
            self.latest = self.history.last().copied();
        } else {
            // Same candle - compute tentatively without committing
            // The last history entry represents the tentative value
            let current_candle = &candles[len - 1];
            let close = current_candle.close.0;
            let volume = current_candle.volume.0;

            // Calculate tentative OBV from committed state
            // committed_prev_close is the close of the candle BEFORE current
            let tentative_obv = self.committed_obv
                + Self::calculate_obv_change(close, self.committed_prev_close, volume);

            // Update last history entry
            if !self.history.is_empty() {
                *self.history.last_mut().unwrap() = tentative_obv;
            }

            self.latest = Some(tentative_obv);
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
        self.committed_obv = 0.0;
        self.committed_prev_close = 0.0;
        self.last_len = 0;
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

    fn candles_with_volume(data: &[(f64, f64)]) -> Vec<Ohlcv> {
        // (close, volume)
        data.iter()
            .enumerate()
            .map(|(i, &(close, volume))| Ohlcv::new(i as i64, close, close + 2.0, close - 1.0, close, volume))
            .collect()
    }

    #[test]
    fn obv_batch_calculation() {
        let mut obv = OBV::default();
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0), // OBV = 0
            Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0), // close up: OBV = 1100
            Ohlcv::new(2, 105.0, 107.0, 102.0, 103.0, 900.0), // close down: OBV = 1100 - 900 = 200
            Ohlcv::new(3, 103.0, 106.0, 101.0, 103.0, 800.0), // close same: OBV = 200
            Ohlcv::new(4, 103.0, 108.0, 100.0, 107.0, 1200.0), // close up: OBV = 200 + 1200 = 1400
        ];

        obv.calc(&candles).unwrap();

        assert!(obv.state().is_ready());
        assert_eq!(obv.len(), 5);

        assert_eq!(obv.get(0).unwrap(), 0.0);
        assert_eq!(obv.get(1).unwrap(), 1100.0);
        assert_eq!(obv.get(2).unwrap(), 200.0);
        assert_eq!(obv.get(3).unwrap(), 200.0); // Same close
        assert_eq!(obv.get(4).unwrap(), 1400.0);
    }

    #[test]
    fn obv_streaming_next() {
        let mut obv = OBV::default();
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0),
            Ohlcv::new(2, 105.0, 107.0, 102.0, 103.0, 900.0),
        ];

        // Feed growing snapshots
        for i in 1..=candles.len() {
            let snapshot = &candles[..i];
            let result = obv.next(snapshot);
            assert!(result.is_some(), "Should have result at len {}", i);
        }

        assert!(obv.state().is_ready());
        assert_eq!(obv.len(), 3);

        // Verify values
        assert_eq!(obv.get(0).unwrap(), 0.0);
        assert_eq!(obv.get(1).unwrap(), 1100.0);
        assert_eq!(obv.get(2).unwrap(), 200.0);
    }

    #[test]
    fn obv_batch_vs_streaming_equivalence() {
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0),
            Ohlcv::new(2, 105.0, 107.0, 102.0, 103.0, 900.0),
            Ohlcv::new(3, 103.0, 106.0, 101.0, 103.0, 800.0),
            Ohlcv::new(4, 103.0, 108.0, 100.0, 107.0, 1200.0),
            Ohlcv::new(5, 107.0, 110.0, 105.0, 104.0, 1000.0),
            Ohlcv::new(6, 104.0, 106.0, 102.0, 108.0, 1500.0),
        ];

        // Batch
        let mut batch = OBV::default();
        batch.calc(&candles).unwrap();

        // Streaming
        let mut stream = OBV::default();
        for i in 1..=candles.len() {
            stream.next(&candles[..i]);
        }

        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let batch_val = batch.get(i as isize).unwrap();
            let stream_val = stream.get(i as isize).unwrap();
            assert!(
                (batch_val - stream_val).abs() < 1e-10,
                "Mismatch at index {}: batch={}, stream={}",
                i,
                batch_val,
                stream_val
            );
        }
    }

    #[test]
    fn obv_uptrend() {
        let mut obv = OBV::default();
        // Consistent uptrend
        let candles = candles_with_volume(&[
            (101.0, 1000.0),
            (102.0, 1000.0),
            (103.0, 1000.0),
            (104.0, 1000.0),
        ]);

        obv.calc(&candles).unwrap();

        // OBV should be increasing
        let latest = obv.latest().unwrap();
        assert_eq!(latest, 3000.0); // 0 + 1000 + 1000 + 1000
    }

    #[test]
    fn obv_downtrend() {
        let mut obv = OBV::default();
        // Consistent downtrend
        let candles = candles_with_volume(&[
            (104.0, 1000.0),
            (103.0, 1000.0),
            (102.0, 1000.0),
            (101.0, 1000.0),
        ]);

        obv.calc(&candles).unwrap();

        // OBV should be decreasing
        let latest = obv.latest().unwrap();
        assert_eq!(latest, -3000.0); // 0 - 1000 - 1000 - 1000
    }

    #[test]
    fn obv_flat_market() {
        let mut obv = OBV::default();
        // All same closes
        let candles = candles_with_volume(&[
            (100.0, 1000.0),
            (100.0, 1100.0),
            (100.0, 900.0),
            (100.0, 1200.0),
        ]);

        obv.calc(&candles).unwrap();

        // OBV should stay at 0
        let latest = obv.latest().unwrap();
        assert_eq!(latest, 0.0);
    }

    #[test]
    fn obv_same_candle_update() {
        let mut obv = OBV::default();

        // First candle
        let candles1 = vec![Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0)];
        obv.next(&candles1);
        assert_eq!(obv.latest().unwrap(), 0.0);

        // Second candle - initial close at 105 (up)
        let mut candles2 = candles1.clone();
        candles2.push(Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0));
        obv.next(&candles2);
        assert_eq!(obv.latest().unwrap(), 1100.0);

        // Same candle - close drops to 98 (now down from prev 100)
        candles2[1] = Ohlcv::new(1, 100.0, 108.0, 95.0, 98.0, 1100.0);
        obv.next(&candles2);
        assert_eq!(obv.latest().unwrap(), -1100.0);

        // Same candle - close back to 100 (same as prev)
        candles2[1] = Ohlcv::new(1, 100.0, 108.0, 95.0, 100.0, 1100.0);
        obv.next(&candles2);
        assert_eq!(obv.latest().unwrap(), 0.0);
    }

    #[test]
    fn obv_reset() {
        let mut obv = OBV::default();
        let candles = vec![
            Ohlcv::new(0, 100.0, 105.0, 98.0, 100.0, 1000.0),
            Ohlcv::new(1, 100.0, 108.0, 99.0, 105.0, 1100.0),
        ];

        obv.calc(&candles).unwrap();
        assert!(obv.state().is_ready());
        assert_eq!(obv.len(), 2);

        obv.reset();
        assert!(obv.state().is_uninitialized());
        assert_eq!(obv.len(), 0);
        assert_eq!(obv.obv_value(), 0.0);
    }

    #[test]
    fn obv_empty_data() {
        let mut obv = OBV::default();
        let candles: Vec<Ohlcv> = vec![];

        obv.calc(&candles).unwrap();
        assert!(obv.state().is_ready());
        assert_eq!(obv.len(), 0);
        assert!(obv.latest().is_none());
    }
}
