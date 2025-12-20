//! Exponential Moving Average calculations.

/// Calculate EMA multiplier for a given period.
///
/// Formula: 2 / (period + 1)
#[inline]
pub fn ema_multiplier(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

/// Calculate next EMA value given previous EMA and new price.
///
/// Formula: (price - prev_ema) * multiplier + prev_ema
#[inline]
pub fn ema_step(price: f64, prev_ema: f64, multiplier: f64) -> f64 {
    (price - prev_ema) * multiplier + prev_ema
}

/// Wilder's smoothing multiplier (used in RSI, ATR, ADX).
///
/// Formula: 1 / period (equivalent to EMA with period = 2*period - 1)
#[inline]
pub fn wilder_multiplier(period: usize) -> f64 {
    1.0 / period as f64
}

/// Calculate next Wilder-smoothed value.
///
/// Formula: prev * (period - 1) / period + current / period
/// Equivalent to: prev + (current - prev) / period
#[inline]
pub fn wilder_step(current: f64, prev_smooth: f64, period: usize) -> f64 {
    let p = period as f64;
    (prev_smooth * (p - 1.0) + current) / p
}

/// Batch calculate EMA over a price series.
///
/// Returns (output_vec, final_ema).
/// First `period - 1` values are NaN.
pub fn ema_batch(prices: &[f64], period: usize) -> (Vec<f64>, f64) {
    let n = prices.len();
    let mut output = vec![f64::NAN; n];

    if n < period || period == 0 {
        return (output, f64::NAN);
    }

    // Initial SMA as seed
    let mut sum = 0.0;
    for i in 0..period {
        sum += prices[i];
    }
    let mut ema = sum / period as f64;
    output[period - 1] = ema;

    // EMA calculation
    let mult = ema_multiplier(period);
    for i in period..n {
        ema = ema_step(prices[i], ema, mult);
        output[i] = ema;
    }

    (output, ema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_multiplier() {
        // Period 14: 2 / 15 ≈ 0.1333
        let mult = ema_multiplier(14);
        assert!((mult - 0.1333).abs() < 0.001);

        // Period 12: 2 / 13 ≈ 0.1538
        let mult = ema_multiplier(12);
        assert!((mult - 0.1538).abs() < 0.001);
    }

    #[test]
    fn test_ema_step() {
        let prev = 100.0;
        let price = 110.0;
        let mult = 0.2; // arbitrary for testing

        // (110 - 100) * 0.2 + 100 = 2 + 100 = 102
        let result = ema_step(price, prev, mult);
        assert!((result - 102.0).abs() < 0.0001);
    }

    #[test]
    fn test_wilder_step() {
        let prev = 10.0;
        let current = 12.0;
        let period = 14;

        // (10 * 13 + 12) / 14 = (130 + 12) / 14 = 142 / 14 ≈ 10.1428
        let result = wilder_step(current, prev, period);
        assert!((result - 10.1428).abs() < 0.001);
    }

    #[test]
    fn test_ema_batch() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (output, final_ema) = ema_batch(&prices, 3);

        // First 2 values should be NaN
        assert!(output[0].is_nan());
        assert!(output[1].is_nan());

        // Index 2: SMA of first 3 = (1+2+3)/3 = 2.0
        assert!((output[2] - 2.0).abs() < 0.0001);

        // Final value should equal returned final_ema
        assert!((output[9] - final_ema).abs() < 0.0001);
    }

    #[test]
    fn test_ema_batch_insufficient_data() {
        let prices = vec![1.0, 2.0];
        let (output, final_ema) = ema_batch(&prices, 5);

        assert!(output.iter().all(|v| v.is_nan()));
        assert!(final_ema.is_nan());
    }
}
