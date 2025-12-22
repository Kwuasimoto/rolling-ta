//! Linear Regression indicators.
//!
//! Provides LinearRegression (fitted value), LinearRegressionR2 (R-squared),
//! and LinearRegressionForecast (predicted next value).
//!
//! These indicators compute directly from `&[Ohlcv]` slices, making them suitable
//! for SharedWindow architecture and parallel computation with Rayon.

use crate::ta::{
    config::LinearRegressionConfig,
    error::{TAError, TAResult},
    math::stats::{calculate_r_squared, linear_forecast, linear_regression, LinearModel},
    state::IndicatorState,
    types::Ohlcv,
    HistoricalIndicator, Indicator,
};

/// Calculate typical price (HLC/3) for a candle.
#[inline]
fn typical_price(candle: &Ohlcv) -> f64 {
    (candle.high.0 + candle.low.0 + candle.close.0) / 3.0
}

// ============================================================================
// LinearRegression
// ============================================================================

/// Linear Regression indicator.
///
/// Computes the fitted value from linear regression on typical prices.
/// Output is the fitted value at x = period - 1 (the current point).
///
/// # Formula
///
/// For a window of N prices, fits y = slope * x + intercept
/// Output = slope * (period - 1) + intercept
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{LinearRegression, LinearRegressionConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut lr = LinearRegression::new(LinearRegressionConfig::new(14));
/// let candles: Vec<Ohlcv> = (0..20)
///     .map(|i| Ohlcv::new(i, 100.0 + i as f64, 105.0 + i as f64, 98.0 + i as f64, 103.0 + i as f64, 1000.0))
///     .collect();
/// lr.calc(&candles).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegression {
    config: LinearRegressionConfig,
    state: IndicatorState,
    model: LinearModel,
    history: Vec<f64>,
    models: Vec<LinearModel>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl LinearRegression {
    /// Create a new LinearRegression indicator.
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            model: LinearModel::default(),
            history: Vec::new(),
            models: Vec::new(),
            latest: None,
            last_len: 0,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Get the current linear model.
    #[inline]
    pub fn model(&self) -> &LinearModel {
        &self.model
    }

    /// Get all historical linear models.
    #[inline]
    pub fn models(&self) -> &[LinearModel] {
        &self.models
    }

    /// Compute regression from a slice of typical prices.
    fn compute_lr(&self, typical_prices: &[f64]) -> TAResult<(f64, LinearModel)> {
        let model = linear_regression(typical_prices)?;
        let lr_value = model.slope * (self.config.period - 1) as f64 + model.intercept;
        Ok((lr_value, model))
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new(LinearRegressionConfig::default())
    }
}

impl Indicator for LinearRegression {
    type Output = f64;
    type Config = LinearRegressionConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        // Calculate typical prices
        let typical_prices: Vec<f64> = data.iter().map(typical_price).collect();

        // Reset state
        self.history = Vec::with_capacity(n);
        self.models = Vec::with_capacity(n);

        // Fill warmup with NaN and default models
        for _ in 0..period - 1 {
            self.history.push(f64::NAN);
            self.models.push(LinearModel::default());
        }

        // Rolling regression from period-1 onwards
        for i in (period - 1)..n {
            let window = &typical_prices[i + 1 - period..=i];
            let (lr_value, model) = self
                .compute_lr(window)
                .map_err(|e| TAError::InvalidData(format!("Linear regression failed at index {}: {}", i, e)))?;
            self.history.push(lr_value);
            self.models.push(model);
        }

        self.model = self.models.last().cloned().unwrap_or_default();
        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < period {
            return None;
        }

        // Compute regression from the last `period` candles
        let window: Vec<f64> = candles[len - period..]
            .iter()
            .map(typical_price)
            .collect();

        let (lr_value, model) = self.compute_lr(&window).ok()?;

        // Determine if this is a new candle
        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                // First time - fill history with warmup NaNs
                self.history = vec![f64::NAN; len - 1];
                self.models = vec![LinearModel::default(); len - 1];
            }
            self.history.push(lr_value);
            self.models.push(model.clone());
            self.last_len = len;
        } else {
            // Same candle - update last entry
            if let Some(last) = self.history.last_mut() {
                *last = lr_value;
            }
            if let Some(last) = self.models.last_mut() {
                *last = model.clone();
            }
        }

        self.model = model;
        self.latest = Some(lr_value);
        self.state = IndicatorState::Ready;

        Some(lr_value)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.models.clear();
        self.model = LinearModel::default();
        self.latest = None;
        self.last_len = 0;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for LinearRegression {
    fn history(&self) -> &[Self::Output] {
        &self.history
    }

    fn get(&self, index: isize) -> Option<Self::Output> {
        let len = self.history.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx >= 0 && idx < len {
            Some(self.history[idx as usize])
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.history.len()
    }
}

// ============================================================================
// LinearRegressionR2
// ============================================================================

/// Linear Regression R² (coefficient of determination) indicator.
///
/// Computes R² which measures how well the regression line fits the data.
/// Values range from 0 to 1, where 1 indicates a perfect fit.
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{LinearRegressionR2, LinearRegressionConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut lr2 = LinearRegressionR2::new(LinearRegressionConfig::new(14));
/// let candles: Vec<Ohlcv> = (0..20)
///     .map(|i| Ohlcv::new(i, 100.0 + i as f64, 105.0 + i as f64, 98.0 + i as f64, 103.0 + i as f64, 1000.0))
///     .collect();
/// lr2.calc(&candles).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegressionR2 {
    config: LinearRegressionConfig,
    state: IndicatorState,
    model: LinearModel,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl LinearRegressionR2 {
    /// Create a new LinearRegressionR2 indicator.
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            model: LinearModel::default(),
            history: Vec::new(),
            latest: None,
            last_len: 0,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate R² from pre-computed LinearRegression models.
    pub fn calc_from_models(&mut self, data: &[Ohlcv], models: &[LinearModel]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        if models.len() != n {
            return Err(TAError::InvalidData(format!(
                "Models length ({}) must match data length ({})",
                models.len(),
                n
            )));
        }

        let typical_prices: Vec<f64> = data.iter().map(typical_price).collect();

        self.history = Vec::with_capacity(n);

        // Fill warmup with NaN
        for _ in 0..period - 1 {
            self.history.push(f64::NAN);
        }

        // Calculate R² using pre-computed models
        for i in (period - 1)..n {
            let window = &typical_prices[i + 1 - period..=i];
            let model_with_r2 = calculate_r_squared(window, models[i].clone())
                .map_err(|e| TAError::InvalidData(format!("R² calculation failed at index {}: {}", i, e)))?;
            self.history.push(model_with_r2.r_squared.unwrap_or(0.0));
        }

        self.model = models.last().cloned().unwrap_or_default();
        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    /// Compute R² from a slice of typical prices.
    fn compute_r2(&self, typical_prices: &[f64]) -> TAResult<(f64, LinearModel)> {
        let base_model = linear_regression(typical_prices)?;
        let model = calculate_r_squared(typical_prices, base_model)?;
        let r2 = model.r_squared.unwrap_or(0.0);
        Ok((r2, model))
    }
}

impl Default for LinearRegressionR2 {
    fn default() -> Self {
        Self::new(LinearRegressionConfig::default())
    }
}

impl Indicator for LinearRegressionR2 {
    type Output = f64;
    type Config = LinearRegressionConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        let typical_prices: Vec<f64> = data.iter().map(typical_price).collect();

        self.history = Vec::with_capacity(n);

        // Fill warmup with NaN
        for _ in 0..period - 1 {
            self.history.push(f64::NAN);
        }

        // Rolling R²
        for i in (period - 1)..n {
            let window = &typical_prices[i + 1 - period..=i];
            let (r2, model) = self
                .compute_r2(window)
                .map_err(|e| TAError::InvalidData(format!("R² calculation failed at index {}: {}", i, e)))?;
            self.history.push(r2);
            if i == n - 1 {
                self.model = model;
            }
        }

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < period {
            return None;
        }

        let window: Vec<f64> = candles[len - period..].iter().map(typical_price).collect();
        let (r2, model) = self.compute_r2(&window).ok()?;

        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                self.history = vec![f64::NAN; len - 1];
            }
            self.history.push(r2);
            self.last_len = len;
        } else {
            if let Some(last) = self.history.last_mut() {
                *last = r2;
            }
        }

        self.model = model;
        self.latest = Some(r2);
        self.state = IndicatorState::Ready;

        Some(r2)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.model = LinearModel::default();
        self.latest = None;
        self.last_len = 0;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for LinearRegressionR2 {
    fn history(&self) -> &[Self::Output] {
        &self.history
    }

    fn get(&self, index: isize) -> Option<Self::Output> {
        let len = self.history.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx >= 0 && idx < len {
            Some(self.history[idx as usize])
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.history.len()
    }
}

// ============================================================================
// LinearRegressionForecast
// ============================================================================

/// Linear Regression Forecast indicator.
///
/// Computes the forecasted value (1 step ahead) from linear regression.
///
/// # Example
///
/// ```
/// use rolling_ta::ta::trend::{LinearRegressionForecast, LinearRegressionConfig};
/// use rolling_ta::ta::types::Ohlcv;
/// use rolling_ta::ta::Indicator;
///
/// let mut lrf = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
/// let candles: Vec<Ohlcv> = (0..20)
///     .map(|i| Ohlcv::new(i, 100.0 + i as f64, 105.0 + i as f64, 98.0 + i as f64, 103.0 + i as f64, 1000.0))
///     .collect();
/// lrf.calc(&candles).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegressionForecast {
    config: LinearRegressionConfig,
    state: IndicatorState,
    model: LinearModel,
    history: Vec<f64>,
    latest: Option<f64>,
    /// Track last snapshot length for next() to detect new candles
    last_len: usize,
}

impl LinearRegressionForecast {
    /// Create a new LinearRegressionForecast indicator.
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            config,
            state: IndicatorState::Uninitialized,
            model: LinearModel::default(),
            history: Vec::new(),
            latest: None,
            last_len: 0,
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate forecast from pre-computed LinearRegression models.
    pub fn calc_from_models(&mut self, models: &[LinearModel]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = models.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        self.history = Vec::with_capacity(n);

        // Fill warmup with NaN
        for _ in 0..period - 1 {
            self.history.push(f64::NAN);
        }

        // Calculate forecast using pre-computed models
        for i in (period - 1)..n {
            let forecast = linear_forecast(models[i].clone(), 1)?;
            self.history.push(forecast);
        }

        self.model = models.last().cloned().unwrap_or_default();
        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    /// Compute forecast from a slice of typical prices.
    fn compute_forecast(&self, typical_prices: &[f64]) -> TAResult<(f64, LinearModel)> {
        let model = linear_regression(typical_prices)?;
        let forecast = linear_forecast(model.clone(), 1)?;
        Ok((forecast, model))
    }
}

impl Default for LinearRegressionForecast {
    fn default() -> Self {
        Self::new(LinearRegressionConfig::default())
    }
}

impl Indicator for LinearRegressionForecast {
    type Output = f64;
    type Config = LinearRegressionConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData {
                required: period,
                actual: n,
            });
        }

        let typical_prices: Vec<f64> = data.iter().map(typical_price).collect();

        self.history = Vec::with_capacity(n);

        // Fill warmup with NaN
        for _ in 0..period - 1 {
            self.history.push(f64::NAN);
        }

        // Rolling forecast
        for i in (period - 1)..n {
            let window = &typical_prices[i + 1 - period..=i];
            let (forecast, model) = self
                .compute_forecast(window)
                .map_err(|e| TAError::InvalidData(format!("Forecast calculation failed at index {}: {}", i, e)))?;
            self.history.push(forecast);
            if i == n - 1 {
                self.model = model;
            }
        }

        self.latest = self.history.last().copied().filter(|v| !v.is_nan());
        self.last_len = n;
        self.state = IndicatorState::Ready;

        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        let len = candles.len();
        let period = self.config.period;

        if len < period {
            return None;
        }

        let window: Vec<f64> = candles[len - period..].iter().map(typical_price).collect();
        let (forecast, model) = self.compute_forecast(&window).ok()?;

        let is_new_candle = len > self.last_len || self.last_len == 0;

        if is_new_candle {
            if self.last_len == 0 {
                self.history = vec![f64::NAN; len - 1];
            }
            self.history.push(forecast);
            self.last_len = len;
        } else {
            if let Some(last) = self.history.last_mut() {
                *last = forecast;
            }
        }

        self.model = model;
        self.latest = Some(forecast);
        self.state = IndicatorState::Ready;

        Some(forecast)
    }

    fn latest(&self) -> Option<Self::Output> {
        self.latest
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.model = LinearModel::default();
        self.latest = None;
        self.last_len = 0;
    }

    fn warmup_period(&self) -> usize {
        self.config.period
    }
}

impl HistoricalIndicator for LinearRegressionForecast {
    fn history(&self) -> &[Self::Output] {
        &self.history
    }

    fn get(&self, index: isize) -> Option<Self::Output> {
        let len = self.history.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx >= 0 && idx < len {
            Some(self.history[idx as usize])
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_candle(ts: i64, open: f64, high: f64, low: f64, close: f64) -> Ohlcv {
        Ohlcv::new(ts, open, high, low, close, 1000.0)
    }

    fn generate_trending_data(n: usize) -> Vec<Ohlcv> {
        (0..n)
            .map(|i| {
                let base = 100.0 + i as f64 * 2.0;
                create_candle(i as i64, base, base + 3.0, base - 1.0, base + 1.5)
            })
            .collect()
    }

    fn generate_noisy_data(n: usize) -> Vec<Ohlcv> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.3).sin() * 5.0;
                create_candle(i as i64, base, base + 2.0, base - 1.0, base + 0.5)
            })
            .collect()
    }

    // LinearRegression tests
    #[test]
    fn lr_batch_calculation() {
        let mut lr = LinearRegression::new(LinearRegressionConfig::new(14));
        let data = generate_trending_data(30);

        lr.calc(&data).unwrap();

        assert!(lr.state().is_ready());
        assert_eq!(lr.len(), 30);

        // First 13 values should be NaN
        for i in 0..13 {
            assert!(lr.get(i as isize).unwrap().is_nan());
        }

        // Values from index 13 onwards should be valid
        assert!(!lr.get(13).unwrap().is_nan());
    }

    #[test]
    fn lr_streaming_matches_batch() {
        let data = generate_trending_data(30);

        let mut batch = LinearRegression::new(LinearRegressionConfig::new(5));
        batch.calc(&data).unwrap();

        let mut stream = LinearRegression::new(LinearRegressionConfig::new(5));
        for i in 1..=data.len() {
            stream.next(&data[..i]);
        }

        let epsilon = 1e-6;
        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let b = batch.get(i as isize).unwrap();
            let s = stream.get(i as isize).unwrap();
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (b - s).abs() < epsilon,
                "Index {}: batch {} != stream {}",
                i,
                b,
                s
            );
        }
    }

    #[test]
    fn lr_same_candle_update() {
        let data = generate_trending_data(20);

        let mut lr = LinearRegression::new(LinearRegressionConfig::new(5));
        for i in 1..=data.len() {
            lr.next(&data[..i]);
        }

        let original = lr.latest().unwrap();

        // Modify last candle
        let mut modified = data.clone();
        let last_idx = modified.len() - 1;
        modified[last_idx] = create_candle(
            modified[last_idx].timestamp.0,
            modified[last_idx].open.0,
            modified[last_idx].high.0 + 5.0,
            modified[last_idx].low.0,
            modified[last_idx].close.0 + 3.0,
        );

        lr.next(&modified);
        let updated = lr.latest().unwrap();

        assert!((original - updated).abs() > 1e-10, "LR should change with modified candle");
        assert_eq!(lr.len(), 20); // Length shouldn't change
    }

    // LinearRegressionR2 tests
    #[test]
    fn lr2_batch_calculation() {
        let mut lr2 = LinearRegressionR2::new(LinearRegressionConfig::new(14));
        let data = generate_trending_data(30);

        lr2.calc(&data).unwrap();

        assert!(lr2.state().is_ready());
        assert_eq!(lr2.len(), 30);

        // R² for trending data should be high (close to 1)
        let r2 = lr2.latest().unwrap();
        assert!(r2 > 0.9, "R² for trending data should be > 0.9, got {}", r2);
    }

    #[test]
    fn lr2_noisy_data() {
        let mut lr2 = LinearRegressionR2::new(LinearRegressionConfig::new(14));
        let data = generate_noisy_data(50);

        lr2.calc(&data).unwrap();

        // R² for noisy data should be lower
        let r2 = lr2.latest().unwrap();
        assert!(r2 >= 0.0 && r2 <= 1.0, "R² should be in [0, 1], got {}", r2);
    }

    #[test]
    fn lr2_streaming_matches_batch() {
        let data = generate_trending_data(30);

        let mut batch = LinearRegressionR2::new(LinearRegressionConfig::new(5));
        batch.calc(&data).unwrap();

        let mut stream = LinearRegressionR2::new(LinearRegressionConfig::new(5));
        for i in 1..=data.len() {
            stream.next(&data[..i]);
        }

        let epsilon = 1e-6;
        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let b = batch.get(i as isize).unwrap();
            let s = stream.get(i as isize).unwrap();
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (b - s).abs() < epsilon,
                "Index {}: batch {} != stream {}",
                i,
                b,
                s
            );
        }
    }

    // LinearRegressionForecast tests
    #[test]
    fn lrf_batch_calculation() {
        let mut lrf = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
        let data = generate_trending_data(30);

        lrf.calc(&data).unwrap();

        assert!(lrf.state().is_ready());
        assert_eq!(lrf.len(), 30);

        // Forecast should be valid
        assert!(!lrf.latest().unwrap().is_nan());
    }

    #[test]
    fn lrf_streaming_matches_batch() {
        let data = generate_trending_data(30);

        let mut batch = LinearRegressionForecast::new(LinearRegressionConfig::new(5));
        batch.calc(&data).unwrap();

        let mut stream = LinearRegressionForecast::new(LinearRegressionConfig::new(5));
        for i in 1..=data.len() {
            stream.next(&data[..i]);
        }

        let epsilon = 1e-6;
        assert_eq!(batch.len(), stream.len());
        for i in 0..batch.len() {
            let b = batch.get(i as isize).unwrap();
            let s = stream.get(i as isize).unwrap();
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (b - s).abs() < epsilon,
                "Index {}: batch {} != stream {}",
                i,
                b,
                s
            );
        }
    }

    // Test calc_from_models optimization
    #[test]
    fn calc_from_models_matches_independent() {
        let data = generate_trending_data(30);

        // Calculate LR once
        let mut lr = LinearRegression::new(LinearRegressionConfig::new(14));
        lr.calc(&data).unwrap();

        // LR2 from models
        let mut lr2_opt = LinearRegressionR2::new(LinearRegressionConfig::new(14));
        lr2_opt.calc_from_models(&data, lr.models()).unwrap();

        // LR2 independent
        let mut lr2_ind = LinearRegressionR2::new(LinearRegressionConfig::new(14));
        lr2_ind.calc(&data).unwrap();

        let epsilon = 1e-10;
        for i in 0..lr2_opt.len() {
            let opt = lr2_opt.get(i as isize).unwrap();
            let ind = lr2_ind.get(i as isize).unwrap();
            if opt.is_nan() && ind.is_nan() {
                continue;
            }
            assert!(
                (opt - ind).abs() < epsilon,
                "Index {}: optimized {} != independent {}",
                i,
                opt,
                ind
            );
        }

        // LRF from models
        let mut lrf_opt = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
        lrf_opt.calc_from_models(lr.models()).unwrap();

        // LRF independent
        let mut lrf_ind = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
        lrf_ind.calc(&data).unwrap();

        for i in 0..lrf_opt.len() {
            let opt = lrf_opt.get(i as isize).unwrap();
            let ind = lrf_ind.get(i as isize).unwrap();
            if opt.is_nan() && ind.is_nan() {
                continue;
            }
            assert!(
                (opt - ind).abs() < epsilon,
                "Index {}: optimized {} != independent {}",
                i,
                opt,
                ind
            );
        }
    }

    #[test]
    fn lr_reset() {
        let mut lr = LinearRegression::new(LinearRegressionConfig::new(14));
        let data = generate_trending_data(30);

        lr.calc(&data).unwrap();
        assert!(lr.state().is_ready());

        lr.reset();
        assert!(lr.state().is_uninitialized());
        assert_eq!(lr.len(), 0);
        assert!(lr.latest().is_none());
    }
}
