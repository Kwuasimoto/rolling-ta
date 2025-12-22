//! Ichimoku Cloud (Ichimoku Kinko Hyo) indicator.

use crate::ta::{
    config::IchimokuConfig,
    error::{TAError, TAResult},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Output values for the Ichimoku Cloud indicator.
#[derive(Debug, Clone, Copy, Default)]
pub struct IchimokuOutput {
    /// Tenkan-sen (Conversion Line): midpoint of high/low over tenkan_period.
    pub tenkan: f64,
    /// Kijun-sen (Base Line): midpoint of high/low over kijun_period.
    pub kijun: f64,
    /// Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2.
    pub senkou_a: f64,
    /// Senkou Span B (Leading Span B): midpoint of high/low over senkou_b_period.
    pub senkou_b: f64,
}

impl IchimokuOutput {
    /// Create a new IchimokuOutput with NaN values.
    pub fn nan() -> Self {
        Self {
            tenkan: f64::NAN,
            kijun: f64::NAN,
            senkou_a: f64::NAN,
            senkou_b: f64::NAN,
        }
    }

    /// Check if all values are valid (not NaN).
    pub fn is_valid(&self) -> bool {
        !self.tenkan.is_nan()
            && !self.kijun.is_nan()
            && !self.senkou_a.is_nan()
            && !self.senkou_b.is_nan()
    }
}

/// Ichimoku Cloud (Ichimoku Kinko Hyo) indicator.
///
/// The Ichimoku Cloud is a comprehensive indicator that defines support
/// and resistance, identifies trend direction, gauges momentum, and
/// provides trading signals.
///
/// # Components
///
/// - **Tenkan-sen (Conversion Line)**: (Highest High + Lowest Low) / 2 over tenkan_period (default: 9)
/// - **Kijun-sen (Base Line)**: (Highest High + Lowest Low) / 2 over kijun_period (default: 26)
/// - **Senkou Span A (Leading Span A)**: (Tenkan + Kijun) / 2
/// - **Senkou Span B (Leading Span B)**: (Highest High + Lowest Low) / 2 over senkou_b_period (default: 52)
///
/// # Note on Displacement (Chikou Span)
///
/// The traditional Ichimoku includes Chikou Span (lagging close) and displaces
/// Senkou Spans forward. This implementation provides current values without
/// displacement - the caller is responsible for applying any time shifts.
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{Ichimoku, IchimokuConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut ichimoku = Ichimoku::new(IchimokuConfig::default());
/// let candles: Vec<Ohlcv> = (0..60)
///     .map(|i| Ohlcv::new(i as i64, 100.0 + i as f64, 105.0 + i as f64, 95.0 + i as f64, 102.0 + i as f64, 1000.0))
///     .collect();
///
/// ichimoku.calc(&candles).unwrap();
/// assert!(ichimoku.state().is_ready());
/// ```
#[derive(Debug, Clone)]
pub struct Ichimoku {
    config: IchimokuConfig,
    state: IndicatorState,
    history: Vec<IchimokuOutput>,
    latest: Option<IchimokuOutput>,
}

impl Ichimoku {
    /// Create a new Ichimoku indicator with the given configuration.
    pub fn new(config: IchimokuConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            latest: None,
        }
    }

    /// Get the tenkan period.
    #[inline]
    pub fn tenkan_period(&self) -> usize {
        self.config.tenkan_period
    }

    /// Get the kijun period.
    #[inline]
    pub fn kijun_period(&self) -> usize {
        self.config.kijun_period
    }

    /// Get the senkou_b period.
    #[inline]
    pub fn senkou_b_period(&self) -> usize {
        self.config.senkou_b_period
    }

    /// Get the displacement period.
    #[inline]
    pub fn displacement(&self) -> usize {
        self.config.displacement
    }

    /// The warmup period is the maximum of all component periods.
    #[inline]
    fn required_warmup(&self) -> usize {
        self.config
            .tenkan_period
            .max(self.config.kijun_period)
            .max(self.config.senkou_b_period)
    }

    /// Compute the midpoint (highest high + lowest low) / 2 from the last N candles.
    #[inline]
    fn midpoint(candles: &[Ohlcv], period: usize) -> f64 {
        if candles.len() < period {
            return f64::NAN;
        }

        let window = &candles[candles.len() - period..];
        let highest = window.iter().map(|c| c.high.0).fold(f64::NEG_INFINITY, f64::max);
        let lowest = window.iter().map(|c| c.low.0).fold(f64::INFINITY, f64::min);

        (highest + lowest) * 0.5
    }

    /// Compute Ichimoku output from a candle slice at a given position.
    fn compute_at(&self, candles: &[Ohlcv], end_idx: usize) -> IchimokuOutput {
        let slice = &candles[..=end_idx];

        // Tenkan-sen: midpoint over tenkan_period
        let tenkan = if slice.len() >= self.config.tenkan_period {
            Self::midpoint(slice, self.config.tenkan_period)
        } else {
            f64::NAN
        };

        // Kijun-sen: midpoint over kijun_period
        let kijun = if slice.len() >= self.config.kijun_period {
            Self::midpoint(slice, self.config.kijun_period)
        } else {
            f64::NAN
        };

        // Senkou Span A: (Tenkan + Kijun) / 2
        let senkou_a = if !tenkan.is_nan() && !kijun.is_nan() {
            (tenkan + kijun) * 0.5
        } else {
            f64::NAN
        };

        // Senkou Span B: midpoint over senkou_b_period
        let senkou_b = if slice.len() >= self.config.senkou_b_period {
            Self::midpoint(slice, self.config.senkou_b_period)
        } else {
            f64::NAN
        };

        IchimokuOutput {
            tenkan,
            kijun,
            senkou_a,
            senkou_b,
        }
    }
}

impl Default for Ichimoku {
    fn default() -> Self {
        Self::new(IchimokuConfig::default())
    }
}

impl Indicator for Ichimoku {
    type Output = IchimokuOutput;
    type Config = IchimokuConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let warmup = self.required_warmup();

        if data.len() < warmup {
            return Err(TAError::InsufficientData {
                required: warmup,
                actual: data.len(),
            });
        }

        self.history = Vec::with_capacity(data.len());

        // Compute for each candle
        for i in 0..data.len() {
            let output = self.compute_at(data, i);
            self.history.push(output);
        }

        self.latest = self.history.last().copied();
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let warmup = self.required_warmup();

        if candles.len() < warmup {
            self.state = self.state.increment(warmup);
            return None;
        }

        // Compute from the full slice (using all available data up to last candle)
        let output = self.compute_at(candles, candles.len() - 1);

        self.latest = Some(output);
        self.history.push(output);
        self.state = IndicatorState::Ready;

        Some(output)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.history.clear();
        self.latest = None;
        self.state = IndicatorState::Uninitialized;
    }

    fn warmup_period(&self) -> usize {
        self.required_warmup()
    }
}

impl HistoricalIndicator for Ichimoku {
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

    /// Helper to build candles with OHLC values.
    fn candle(ts: i64, open: f64, high: f64, low: f64, close: f64) -> Ohlcv {
        Ohlcv::new(ts, open, high, low, close, 1000.0)
    }

    #[test]
    fn ichimoku_midpoint_calculation() {
        // Verify midpoint formula: (highest high + lowest low) / 2
        let candles = vec![
            candle(0, 100.0, 110.0, 90.0, 105.0),   // H=110, L=90
            candle(1, 105.0, 115.0, 95.0, 110.0),   // H=115, L=95
            candle(2, 110.0, 120.0, 100.0, 115.0),  // H=120, L=100
        ];

        // Midpoint over 3: (120 + 90) / 2 = 105
        let mid = Ichimoku::midpoint(&candles, 3);
        assert!((mid - 105.0).abs() < 0.0001);

        // Midpoint over 2 (last 2): (120 + 95) / 2 = 107.5
        let mid = Ichimoku::midpoint(&candles, 2);
        assert!((mid - 107.5).abs() < 0.0001);
    }

    #[test]
    fn ichimoku_default_config() {
        let ichimoku = Ichimoku::default();
        assert_eq!(ichimoku.tenkan_period(), 9);
        assert_eq!(ichimoku.kijun_period(), 26);
        assert_eq!(ichimoku.senkou_b_period(), 52);
        assert_eq!(ichimoku.displacement(), 26);
    }

    #[test]
    fn ichimoku_warmup_period() {
        // Default: max(9, 26, 52) = 52
        let ichimoku = Ichimoku::default();
        assert_eq!(ichimoku.warmup_period(), 52);

        // Custom config
        let config = IchimokuConfig {
            tenkan_period: 5,
            kijun_period: 10,
            senkou_b_period: 20,
            displacement: 10,
        };
        let ichimoku = Ichimoku::new(config);
        assert_eq!(ichimoku.warmup_period(), 20);
    }

    #[test]
    fn ichimoku_batch_calculation() {
        let config = IchimokuConfig {
            tenkan_period: 3,
            kijun_period: 5,
            senkou_b_period: 7,
            displacement: 5,
        };
        let mut ichimoku = Ichimoku::new(config);

        // Create candles with predictable pattern
        let candles: Vec<Ohlcv> = (0..10)
            .map(|i| candle(i as i64, 100.0, 105.0 + i as f64, 95.0 + i as f64, 100.0 + i as f64))
            .collect();

        ichimoku.calc(&candles).unwrap();

        assert!(ichimoku.state().is_ready());
        assert_eq!(ichimoku.len(), 10);

        // First few should have NaN for components requiring more data
        let first = ichimoku.get(0).unwrap();
        assert!(first.kijun.is_nan()); // Need 5 candles
        assert!(first.senkou_b.is_nan()); // Need 7 candles

        // After warmup, all values should be valid
        let last = ichimoku.get(-1).unwrap();
        assert!(last.is_valid());
    }

    #[test]
    fn ichimoku_streaming_next() {
        let config = IchimokuConfig {
            tenkan_period: 3,
            kijun_period: 5,
            senkou_b_period: 7,
            displacement: 5,
        };
        let mut ichimoku = Ichimoku::new(config);

        // Build up candles incrementally
        let all_candles: Vec<Ohlcv> = (0..10)
            .map(|i| candle(i as i64, 100.0, 105.0 + i as f64, 95.0 + i as f64, 100.0 + i as f64))
            .collect();

        // Not enough candles
        for i in 1..7 {
            let result = ichimoku.next(&all_candles[..i]);
            assert!(result.is_none());
        }

        // Now should produce output
        let result = ichimoku.next(&all_candles[..7]);
        assert!(result.is_some());
        let output = result.unwrap();
        assert!(output.is_valid());
    }

    #[test]
    fn ichimoku_insufficient_data() {
        let mut ichimoku = Ichimoku::default(); // warmup = 52

        let candles: Vec<Ohlcv> = (0..30)
            .map(|i| candle(i as i64, 100.0, 105.0, 95.0, 100.0))
            .collect();

        let result = ichimoku.calc(&candles);
        assert!(matches!(result, Err(TAError::InsufficientData { .. })));
    }

    #[test]
    fn ichimoku_reset() {
        let mut ichimoku = Ichimoku::default();
        let candles: Vec<Ohlcv> = (0..60)
            .map(|i| candle(i as i64, 100.0, 105.0, 95.0, 100.0))
            .collect();

        ichimoku.calc(&candles).unwrap();
        assert!(ichimoku.state().is_ready());

        ichimoku.reset();
        assert!(ichimoku.state().is_uninitialized());
        assert!(ichimoku.history.is_empty());
        assert!(ichimoku.latest().is_none());
    }

    #[test]
    fn ichimoku_senkou_a_formula() {
        // Verify Senkou A = (Tenkan + Kijun) / 2
        let config = IchimokuConfig {
            tenkan_period: 2,
            kijun_period: 3,
            senkou_b_period: 4,
            displacement: 3,
        };
        let mut ichimoku = Ichimoku::new(config);

        let candles = vec![
            candle(0, 100.0, 110.0, 90.0, 100.0),
            candle(1, 100.0, 120.0, 80.0, 100.0),
            candle(2, 100.0, 115.0, 85.0, 100.0),
            candle(3, 100.0, 125.0, 75.0, 100.0),
        ];

        ichimoku.calc(&candles).unwrap();
        let last = ichimoku.get(-1).unwrap();

        // Tenkan (2 periods): last 2 candles -> H=125, L=75 -> (125+75)/2 = 100
        // Kijun (3 periods): last 3 candles -> H=125, L=75 -> (125+75)/2 = 100
        // Senkou A = (100 + 100) / 2 = 100
        assert!((last.tenkan - 100.0).abs() < 0.0001);
        assert!((last.kijun - 100.0).abs() < 0.0001);
        assert!((last.senkou_a - 100.0).abs() < 0.0001);
    }

    #[test]
    fn ichimoku_negative_indexing() {
        let config = IchimokuConfig {
            tenkan_period: 3,
            kijun_period: 5,
            senkou_b_period: 7,
            displacement: 5,
        };
        let mut ichimoku = Ichimoku::new(config);

        let candles: Vec<Ohlcv> = (0..10)
            .map(|i| candle(i as i64, 100.0, 105.0 + i as f64, 95.0 + i as f64, 100.0 + i as f64))
            .collect();

        ichimoku.calc(&candles).unwrap();

        // -1 should be last value
        let last = ichimoku.get(-1).unwrap();
        let also_last = ichimoku.get(9).unwrap();
        assert!((last.tenkan - also_last.tenkan).abs() < 0.0001);
    }
}
