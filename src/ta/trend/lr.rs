use crate::ta::{
    Indicator,
    HistoricalIndicator, 
    config::LinearRegressionConfig, 
    error::{ TAError, TAResult }, 
    math::{ 
        LinearModel, 
        RollingWindow,
        stats::{ 
            calculate_r_squared, 
            linear_forecast,
            linear_regression
        }
    }, 
    state::IndicatorState, 
    types::{ Ohlcv, OhlcvSeries }
};

#[derive(Debug, Clone)]
pub struct LinearRegression {
    config: LinearRegressionConfig,
    state: IndicatorState,
    model: LinearModel,
    history: Vec<f64>,
    models: Vec<LinearModel>, // Store models for each point (parallel to history)
    window: RollingWindow
}

impl LinearRegression {
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            window: RollingWindow::new(config.period),
            config,
            model: LinearModel::default(),
            state: IndicatorState::Uninitialized,
            history: Vec::new(),
            models: Vec::new()
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Get the current linear model.
    /// Returns the model fitted to the most recent window.
    #[inline]
    pub fn model(&self) -> &LinearModel {
        &self.model
    }

    /// Get all historical linear models.
    /// Parallel to history - index i contains the model used to calculate history[i].
    /// Models during warmup period are default/invalid.
    #[inline]
    pub fn models(&self) -> &[LinearModel] {
        &self.models
    }
}

impl Indicator for LinearRegression {
    type Output = f64;
    type Config = LinearRegressionConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData { required: period, actual: n });
        }

        // Calculate typical prices (HLC/3)
        let typical_prices: Vec<f64> = data.highs.iter()
            .zip(&data.lows)
            .zip(&data.closes)
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();

        // Reset state
        self.window = RollingWindow::new(period);
        self.history = Vec::with_capacity(n);
        self.models = Vec::with_capacity(n);

        // Fill initial warmup period with NaN and default models
        for i in 0..period - 1 {
            self.window.push(typical_prices[i]);
            self.history.push(f64::NAN);
            self.models.push(LinearModel::default());
        }

        // First valid regression at index period - 1
        self.window.push(typical_prices[period - 1]);
        self.model = linear_regression(&self.window.to_vec())
            .map_err(|e| TAError::InvalidData(format!("Initial linear regression failed: {}", e)))?;

        // Calculate fitted value at x = period - 1 (the current point)
        let lr_value = self.model.slope * (period - 1) as f64 + self.model.intercept;
        self.history.push(lr_value);
        self.models.push(self.model.clone());

        // Rolling regression for remaining data
        for i in period..n {
            self.window.push(typical_prices[i]);
            self.model = linear_regression(&self.window.to_vec())
                .map_err(|e| TAError::InvalidData(format!("Rolling linear regression failed at index {}: {}", i, e)))?;

            // Fitted value at x = period - 1 (the current point in the window)
            let lr_value = self.model.slope * (period - 1) as f64 + self.model.intercept;
            self.history.push(lr_value);
            self.models.push(self.model.clone());
        }

        self.state = IndicatorState::Ready;
        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let period = self.config.period;

        // Calculate typical price
        let typical_price = (tick.high.0 + tick.low.0 + tick.close.0) / 3.0;

        // Add to rolling window
        self.window.push(typical_price);

        match self.state {
            IndicatorState::Uninitialized | IndicatorState::Warming { .. } => {
                self.history.push(f64::NAN);
                self.models.push(LinearModel::default());

                if self.window.len() >= period {
                    // Ready to calculate first regression
                    self.model = linear_regression(&self.window.to_vec())
                        .map_err(|e| TAError::InvalidData(format!("Linear regression failed: {}", e)))?;

                    let lr_value = self.model.slope * (period - 1) as f64 + self.model.intercept;
                    let idx = self.history.len() - 1;
                    self.history[idx] = lr_value;
                    self.models[idx] = self.model.clone();
                    self.state = IndicatorState::Ready;
                    return Ok(Some(lr_value));
                } else {
                    let count = if let IndicatorState::Warming { count } = self.state {
                        count + 1
                    } else {
                        1
                    };
                    self.state = IndicatorState::Warming { count };
                    return Ok(None);
                }
            }
            IndicatorState::Ready => {
                // Recalculate regression on new window
                self.model = linear_regression(&self.window.to_vec())
                    .map_err(|e| TAError::InvalidData(format!("Linear regression failed: {}", e)))?;

                // Fitted value at x = period - 1 (current point in window)
                let lr_value = self.model.slope * (period - 1) as f64 + self.model.intercept;
                self.history.push(lr_value);
                self.models.push(self.model.clone());
                Ok(Some(lr_value))
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.history.last().copied().filter(|v| !v.is_nan())
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.models.clear();
        self.window = RollingWindow::new(self.config.period);
        self.model = LinearModel::default();
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


#[derive(Debug, Clone)]
pub struct LinearRegressionR2 {
    config: LinearRegressionConfig,
    state: IndicatorState,
    model: LinearModel,
    r2_value: f64,
    history: Vec<f64>,
    window: RollingWindow
}

impl LinearRegressionR2 {
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            window: RollingWindow::new(config.period),
            config,
            model: LinearModel::default(),
            r2_value: 0.0,
            state: IndicatorState::Uninitialized,
            history: Vec::new()
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate R² from pre-computed LinearRegression models.
    /// **Recommended**: Use this instead of calc() to avoid recalculating regression.
    ///
    /// # Example
    /// ```ignore
    /// let mut lr = LinearRegression::new(config);
    /// lr.calc(&data)?;
    ///
    /// let mut lr2 = LinearRegressionR2::new(config);
    /// lr2.calc_from_models(&data, lr.models())?;
    /// ```
    pub fn calc_from_models(&mut self, data: &OhlcvSeries, models: &[LinearModel]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData { required: period, actual: n });
        }

        if models.len() != n {
            return Err(TAError::InvalidData(format!(
                "Models length ({}) must match data length ({})",
                models.len(), n
            )));
        }

        // Calculate typical prices (HLC/3)
        let typical_prices: Vec<f64> = data.highs.iter()
            .zip(&data.lows)
            .zip(&data.closes)
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();

        // Reset state
        self.window = RollingWindow::new(period);
        self.history = Vec::with_capacity(n);

        // Fill initial warmup period with NaN
        for i in 0..period - 1 {
            self.window.push(typical_prices[i]);
            self.history.push(f64::NAN);
        }

        // Calculate R² using pre-computed models
        self.window.push(typical_prices[period - 1]);
        let window_data = self.window.to_vec();
        let model_with_r2 = calculate_r_squared(&window_data, models[period - 1].clone())
            .map_err(|e| TAError::InvalidData(format!("R² calculation failed: {}", e)))?;

        self.model = model_with_r2.clone();
        self.r2_value = model_with_r2.r_squared.unwrap_or(0.0);
        self.history.push(self.r2_value);

        // Rolling R² for remaining data
        for i in period..n {
            self.window.push(typical_prices[i]);
            let window_data = self.window.to_vec();

            let model_with_r2 = calculate_r_squared(&window_data, models[i].clone())
                .map_err(|e| TAError::InvalidData(format!("R² calculation failed at index {}: {}", i, e)))?;

            self.model = model_with_r2.clone();
            self.r2_value = model_with_r2.r_squared.unwrap_or(0.0);
            self.history.push(self.r2_value);
        }

        self.state = IndicatorState::Ready;
        Ok(self)
    }
}

impl Indicator for LinearRegressionR2 {
    type Output = f64;
    type Config = LinearRegressionConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData { required: period, actual: n });
        }

        // Calculate typical prices (HLC/3)
        let typical_prices: Vec<f64> = data.highs.iter()
            .zip(&data.lows)
            .zip(&data.closes)
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();

        // Reset state
        self.window = RollingWindow::new(period);
        self.history = Vec::with_capacity(n);

        // Fill initial warmup period with NaN
        for i in 0..period - 1 {
            self.window.push(typical_prices[i]);
            self.history.push(f64::NAN);
        }

        // First valid R² at index period - 1
        self.window.push(typical_prices[period - 1]);
        let window_data = self.window.to_vec();
        let base_model = linear_regression(&window_data)
            .map_err(|e| TAError::InvalidData(format!("Initial linear regression failed: {}", e)))?;

        let model_with_r2 = calculate_r_squared(&window_data, base_model)
            .map_err(|e| TAError::InvalidData(format!("R² calculation failed: {}", e)))?;

        self.model = model_with_r2.clone();
        self.r2_value = model_with_r2.r_squared.unwrap_or(0.0);
        self.history.push(self.r2_value);

        // Rolling regression for remaining data
        for i in period..n {
            self.window.push(typical_prices[i]);
            let window_data = self.window.to_vec();

            let base_model = linear_regression(&window_data)
                .map_err(|e| TAError::InvalidData(format!("Rolling linear regression failed at index {}: {}", i, e)))?;

            let model_with_r2 = calculate_r_squared(&window_data, base_model)
                .map_err(|e| TAError::InvalidData(format!("R² calculation failed at index {}: {}", i, e)))?;

            self.model = model_with_r2.clone();
            self.r2_value = model_with_r2.r_squared.unwrap_or(0.0);
            self.history.push(self.r2_value);
        }

        self.state = IndicatorState::Ready;
        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let period = self.config.period;

        // Calculate typical price
        let typical_price = (tick.high.0 + tick.low.0 + tick.close.0) / 3.0;

        // Add to rolling window
        self.window.push(typical_price);

        match self.state {
            IndicatorState::Uninitialized | IndicatorState::Warming { .. } => {
                self.history.push(f64::NAN);

                if self.window.len() >= period {
                    // Ready to calculate first R²
                    let window_data = self.window.to_vec();
                    let base_model = linear_regression(&window_data)
                        .map_err(|e| TAError::InvalidData(format!("Linear regression failed: {}", e)))?;

                    let model_with_r2 = calculate_r_squared(&window_data, base_model)
                        .map_err(|e| TAError::InvalidData(format!("R² calculation failed: {}", e)))?;

                    self.model = model_with_r2.clone();
                    self.r2_value = model_with_r2.r_squared.unwrap_or(0.0);
                    let idx = self.history.len() - 1;
                    self.history[idx] = self.r2_value;
                    self.state = IndicatorState::Ready;
                    return Ok(Some(self.r2_value));
                } else {
                    let count = if let IndicatorState::Warming { count } = self.state {
                        count + 1
                    } else {
                        1
                    };
                    self.state = IndicatorState::Warming { count };
                    return Ok(None);
                }
            }
            IndicatorState::Ready => {
                // Recalculate R² on new window
                let window_data = self.window.to_vec();
                let base_model = linear_regression(&window_data)
                    .map_err(|e| TAError::InvalidData(format!("Linear regression failed: {}", e)))?;

                let model_with_r2 = calculate_r_squared(&window_data, base_model)
                    .map_err(|e| TAError::InvalidData(format!("R² calculation failed: {}", e)))?;

                self.model = model_with_r2.clone();
                self.r2_value = model_with_r2.r_squared.unwrap_or(0.0);
                self.history.push(self.r2_value);
                Ok(Some(self.r2_value))
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.history.last().copied().filter(|v| !v.is_nan())
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.window = RollingWindow::new(self.config.period);
        self.model = LinearModel::default();
        self.r2_value = 0.0;
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


#[derive(Debug, Clone)]
pub struct LinearRegressionForecast {
    config: LinearRegressionConfig,
    state: IndicatorState,
    model: LinearModel,
    forecast_value: f64,
    history: Vec<f64>,
    window: RollingWindow
}

impl LinearRegressionForecast {
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            window: RollingWindow::new(config.period),
            config,
            model: LinearModel::default(),
            forecast_value: 0.0,
            state: IndicatorState::Uninitialized,
            history: Vec::new()
        }
    }

    /// Get the period.
    #[inline]
    pub fn period(&self) -> usize {
        self.config.period
    }

    /// Calculate forecast from pre-computed LinearRegression models.
    /// **Recommended**: Use this instead of calc() to avoid recalculating regression.
    ///
    /// # Example
    /// ```ignore
    /// let mut lr = LinearRegression::new(config);
    /// lr.calc(&data)?;
    ///
    /// let mut lrf = LinearRegressionForecast::new(config);
    /// lrf.calc_from_models(lr.models())?;
    /// ```
    pub fn calc_from_models(&mut self, models: &[LinearModel]) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = models.len();

        if n < period {
            return Err(TAError::InsufficientData { required: period, actual: n });
        }

        // Reset state
        self.history = Vec::with_capacity(n);

        // Fill initial warmup period with NaN
        for _ in 0..period - 1 {
            self.history.push(f64::NAN);
        }

        // Calculate forecast using pre-computed models
        for i in (period - 1)..n {
            self.model = models[i].clone();

            // Forecast at x = period (next value)
            self.forecast_value = linear_forecast(self.model.clone(), 1)?;
            self.history.push(self.forecast_value);
        }

        self.state = IndicatorState::Ready;
        Ok(self)
    }
}

impl Indicator for LinearRegressionForecast {
    type Output = f64;
    type Config = LinearRegressionConfig;

    fn state(&self) -> IndicatorState {
        self.state
    }

    fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
        let period = self.config.period;
        let n = data.len();

        if n < period {
            return Err(TAError::InsufficientData { required: period, actual: n });
        }

        // Calculate typical prices (HLC/3)
        let typical_prices: Vec<f64> = data.highs.iter()
            .zip(&data.lows)
            .zip(&data.closes)
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();

        // Reset state
        self.window = RollingWindow::new(period);
        self.history = Vec::with_capacity(n);

        // Fill initial warmup period with NaN
        for i in 0..period - 1 {
            self.window.push(typical_prices[i]);
            self.history.push(f64::NAN);
        }

        // First valid forecast at index period - 1
        self.window.push(typical_prices[period - 1]);
        self.model = linear_regression(&self.window.to_vec())
            .map_err(|e| TAError::InvalidData(format!("Initial linear regression failed: {}", e)))?;

        // Forecast at x = period (next value)
        self.forecast_value = linear_forecast(self.model.clone(), 1)?;
        self.history.push(self.forecast_value);

        // Rolling forecast for remaining data
        for i in period..n {
            self.window.push(typical_prices[i]);
            self.model = linear_regression(&self.window.to_vec())
                .map_err(|e| TAError::InvalidData(format!("Rolling linear regression failed at index {}: {}", i, e)))?;

            self.forecast_value = linear_forecast(self.model.clone(), 1)?;
            self.history.push(self.forecast_value);
        }

        self.state = IndicatorState::Ready;
        Ok(self)
    }

    fn update(&mut self, tick: &Ohlcv) -> TAResult<Option<Self::Output>> {
        let period = self.config.period;

        // Calculate typical price
        let typical_price = (tick.high.0 + tick.low.0 + tick.close.0) / 3.0;

        // Add to rolling window
        self.window.push(typical_price);

        match self.state {
            IndicatorState::Uninitialized | IndicatorState::Warming { .. } => {
                self.history.push(f64::NAN);

                if self.window.len() >= period {
                    // Ready to calculate first forecast
                    self.model = linear_regression(&self.window.to_vec())
                        .map_err(|e| TAError::InvalidData(format!("Linear regression failed: {}", e)))?;

                    self.forecast_value = linear_forecast(self.model.clone(), 1)?;
                    let idx = self.history.len() - 1;
                    self.history[idx] = self.forecast_value;
                    self.state = IndicatorState::Ready;
                    return Ok(Some(self.forecast_value));
                } else {
                    let count = if let IndicatorState::Warming { count } = self.state {
                        count + 1
                    } else {
                        1
                    };
                    self.state = IndicatorState::Warming { count };
                    return Ok(None);
                }
            }
            IndicatorState::Ready => {
                // Recalculate forecast on new window
                self.model = linear_regression(&self.window.to_vec())
                    .map_err(|e| TAError::InvalidData(format!("Linear regression failed: {}", e)))?;

                self.forecast_value = linear_forecast(self.model.clone(), 1)?;
                self.history.push(self.forecast_value);
                Ok(Some(self.forecast_value))
            }
        }
    }

    fn latest(&self) -> Option<Self::Output> {
        self.history.last().copied().filter(|v| !v.is_nan())
    }

    fn reset(&mut self) {
        self.state = IndicatorState::Uninitialized;
        self.history.clear();
        self.window = RollingWindow::new(self.config.period);
        self.model = LinearModel::default();
        self.forecast_value = 0.0;
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
