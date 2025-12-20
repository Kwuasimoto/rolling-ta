//! Statistical helper functions.

use crate::prelude::{TAError, TAResult};

/// Calculate mean of a slice.
#[inline]
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate variance of a slice (population variance).
pub fn variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation of a slice (population).
#[inline]
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Calculate mean and standard deviation in single pass (Welford's algorithm).
pub fn mean_std_dev(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let mut count = 0u64;
    let mut mean = 0.0;
    let mut m2 = 0.0;

    for &value in data {
        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    let variance = if count < 2 { 0.0 } else { m2 / count as f64 };
    (mean, variance.sqrt())
}

/// Welford's online algorithm state for running mean/variance.
#[derive(Debug, Clone, Default)]
pub struct WelfordState {
    count: u64,
    mean: f64,
    m2: f64,
}

impl WelfordState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a new value.
    #[inline]
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get current count.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get current mean.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get population variance.
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Get population standard deviation.
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Normalize a value using current stats.
    #[inline]
    pub fn normalize(&self, value: f64, epsilon: f64) -> f64 {
        (value - self.mean) / (self.std_dev() + epsilon)
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Calculate max of a slice.
#[inline]
pub fn max(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Calculate min of a slice.
#[inline]
pub fn min(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::INFINITY, f64::min)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LinearModel {
    pub slope: f64,
    pub intercept: f64,
    data_points: usize,

    pub r_squared: Option<f64>,
    pub forecast: Option<f64>
}

impl LinearModel {
    pub fn new(data_points: usize, slope: f64, intercept: f64, r_squared: Option<f64>, forecast: Option<f64>) -> LinearModel {
        Self { 
            data_points, 
            slope, 
            intercept, 
            r_squared, 
            forecast 
        }
    }
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// Linear regression: returns LinearModel.
pub fn linear_regression(y: &[f64]) -> TAResult<LinearModel> {
    let n = y.len();
    if n < 2 {
        return Err(TAError::InsufficientData {
            actual: n,
            required: 2
        });
    }

    let n_f = n as f64;

    let sum_x = (n * (n - 1) / 2) as f64;
    let sum_xx = (n * (n - 1) * (2 * n - 1) / 6) as f64;

    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = y.iter().enumerate().map(|(i, &yi)| i as f64 * yi).sum();

    let slope: f64 = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n_f;

    Ok(LinearModel {
        slope,
        intercept,
        data_points: n,
        r_squared: None,
        forecast: None
    })
}

pub fn calculate_r_squared(y: &[f64], model: LinearModel) -> TAResult<LinearModel> {
    let n = y.len();
    if n == 0 { return Err(TAError::InvalidData(String::from("y.len == 0, can't calculate r2"))); }

    // R-squared requires a second pass over the data to calculate residuals.
    let sum_y: f64 = y.iter().sum();
    let y_mean = sum_y / n as f64;

    // Calculate Total Sum of Squares (SS_tot) and Residual Sum of Squares (SS_res) in one pass
    let (ss_tot, ss_res) = y
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(mut tot, mut res), (i, &yi)| {
            let pred = model.slope * i as f64 + model.intercept;
            
            tot += (yi - y_mean).powi(2);
            res += (yi - pred).powi(2);
            (tot, res)
        });

    if ss_tot.abs() < f64::EPSILON {
        // If SS_tot is zero, all y-values are the same. R^2 is 1.0 if residuals are also 0.
        return Ok(LinearModel {
            r_squared: Some(1.0),
            ..model
        })
    }
    
    Ok(LinearModel {
        r_squared: Some(1.0 - ss_res / ss_tot),
        ..model
    })
}

#[inline]
pub fn linear_forecast(model: LinearModel, x_index: usize) -> TAResult<f64> {
    return if x_index == 0 {
        Err(TAError::InvalidData(String::from("x_index cannot be 0, failed to get forecast.")))
    } else{
        Ok(model.slope * x_index as f64 + model.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_variance_std_dev() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // Mean = 5, Variance = 4, StdDev = 2
        assert!((mean(&data) - 5.0).abs() < 0.0001);
        assert!((variance(&data) - 4.0).abs() < 0.0001);
        assert!((std_dev(&data) - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_welford() {
        let mut state = WelfordState::new();
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        for &v in &data {
            state.update(v);
        }

        assert!((state.mean() - 5.0).abs() < 0.0001);
        assert!((state.std_dev() - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_mean_std_dev_single_pass() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (m, s) = mean_std_dev(&data);
        assert!((m - 5.0).abs() < 0.0001);
        assert!((s - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_linear_regression() {
        // Perfect line: y = 2x + 1
        let y: [f64; 5] = [1.0, 3.0, 5.0, 7.0, 9.0];
        let model = linear_regression(&y)
                .expect("Test data should produce a valid LinearModel result.");

        let slope = model.slope;
        let intercept = model.intercept;

        // GEMINI: Whats better practice, appending r2 to model internally, or returning and composing the result?
        let model_w_r2 = calculate_r_squared(&y, model)
                .expect("Test data should produce a valid LinearModel result.");

        let r2 = model_w_r2.r_squared
                .expect("Can't compare r2, undefined");

        assert!((slope - 2.0).abs() < 0.0001);
        assert!((intercept - 1.0).abs() < 0.0001);
        assert!((r2 - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_max_min() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert_eq!(max(&data), 9.0);
        assert_eq!(min(&data), 1.0);
    }
}
