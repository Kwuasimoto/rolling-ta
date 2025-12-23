//! Shared test utilities for reference tests.
//!
//! Provides helpers for loading xlsx fixtures, building OHLCV data,
//! and comparing indicator outputs.

use calamine::{open_workbook, Data, Reader, Xlsx};
use rolling_ta::prelude::*;
use rolling_ta::trend::{ADXOutput, DMIOutput, IchimokuOutput};
use rolling_ta::volatility::{BBOutput, DonchianOutput};
use rolling_ta::momentum::StochRSIOutput;

pub const EPSILON: f64 = 1e-4;

/// Helper to read xlsx file columns by position (NO headers in files).
pub fn read_xlsx_by_position(path: &str, col_indices: &[usize]) -> Vec<Vec<f64>> {
    let mut workbook: Xlsx<_> =
        open_workbook(path).unwrap_or_else(|_| panic!("Failed to open {}", path));

    let sheet_name = workbook.sheet_names()[0].clone();
    let range = workbook
        .worksheet_range(&sheet_name)
        .expect("Failed to read sheet");

    let mut result: Vec<Vec<f64>> = col_indices.iter().map(|_| Vec::new()).collect();

    for row in range.rows() {
        for (result_idx, &col_idx) in col_indices.iter().enumerate() {
            let value = match row.get(col_idx) {
                Some(Data::Float(f)) => *f,
                Some(Data::Int(i)) => *i as f64,
                Some(Data::Empty) => f64::NAN,
                _ => f64::NAN,
            };
            result[result_idx].push(value);
        }
    }

    result
}

/// Build Vec<Ohlcv> from closes only
pub fn build_candles_from_closes(closes: &[f64]) -> Vec<Ohlcv> {
    closes
        .iter()
        .enumerate()
        .map(|(i, &close)| Ohlcv::new(i as i64, close, close, close, close, 0.0))
        .collect()
}

/// Build Vec<Ohlcv> with full OHLCV data from xlsx columns
/// Columns: [timestamp, open, high, low, close, volume]
pub fn build_candles_ohlcv(
    timestamps: &[f64],
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
) -> Vec<Ohlcv> {
    timestamps
        .iter()
        .zip(opens)
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .zip(volumes)
        .map(|(((((t, o), h), l), c), v)| {
            Ohlcv::new(*t as i64, *o, *h, *l, *c, *v)
        })
        .collect()
}

/// Compare two f64 slices, counting matches within epsilon
pub fn compare_values(name: &str, rust_values: &[f64], true_values: &[f64], epsilon: f64) -> usize {
    let mut comparisons = 0;
    for (i, (rust_val, true_val)) in rust_values.iter().zip(true_values.iter()).enumerate() {
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            assert!(
                diff < epsilon,
                "{} mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
                name,
                i,
                rust_val,
                true_val,
                diff
            );
            comparisons += 1;
        }
    }
    comparisons
}

/// Assert two f64 histories are equal within epsilon
pub fn assert_histories_equal(context: &str, a: &[f64], b: &[f64], epsilon: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        context,
        a.len(),
        b.len()
    );

    for i in 0..a.len() {
        let a_val = a[i];
        let b_val = b[i];

        if a_val.is_nan() && b_val.is_nan() {
            continue;
        } else if a_val.is_nan() || b_val.is_nan() {
            panic!(
                "{} at index {}: NaN mismatch ({} vs {})",
                context, i, a_val, b_val
            );
        }

        assert!(
            (a_val - b_val).abs() < epsilon,
            "{} at index {}: {} != {}",
            context,
            i,
            a_val,
            b_val
        );
    }
}

// ============================================================
// Additional Candle Builders
// ============================================================

/// Build Vec<Ohlcv> from high, low, close columns (no open/volume)
pub fn build_candles_hlc(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<Ohlcv> {
    highs
        .iter()
        .zip(lows)
        .zip(closes)
        .enumerate()
        .map(|(i, ((h, l), c))| Ohlcv::new(i as i64, *c, *h, *l, *c, 0.0))
        .collect()
}

/// Build Vec<Ohlcv> from high, low, close, volume columns
pub fn build_candles_hlcv(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
) -> Vec<Ohlcv> {
    highs
        .iter()
        .zip(lows)
        .zip(closes)
        .zip(volumes)
        .enumerate()
        .map(|(i, (((h, l), c), v))| Ohlcv::new(i as i64, *c, *h, *l, *c, *v))
        .collect()
}

/// Build Vec<Ohlcv> from open, high, low, close columns
pub fn build_candles_ohlc(
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) -> Vec<Ohlcv> {
    opens
        .iter()
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .enumerate()
        .map(|(i, (((o, h), l), c))| Ohlcv::new(i as i64, *o, *h, *l, *c, 0.0))
        .collect()
}

/// Build Vec<Ohlcv> from close and volume columns only
pub fn build_candles_cv(closes: &[f64], volumes: &[f64]) -> Vec<Ohlcv> {
    closes
        .iter()
        .zip(volumes)
        .enumerate()
        .map(|(i, (c, v))| Ohlcv::new(i as i64, *c, *c, *c, *c, *v))
        .collect()
}

/// Build Vec<Ohlcv> from timestamp, high, low, close, volume columns
pub fn build_candles_thlcv(
    timestamps: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
) -> Vec<Ohlcv> {
    timestamps
        .iter()
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .zip(volumes)
        .map(|((((t, h), l), c), v)| Ohlcv::new(*t as i64, *c, *h, *l, *c, *v))
        .collect()
}

/// Compare values with warmup offset.
///
/// Skips `warmup` values in `true_values` to align with Rust history
/// which doesn't include warmup NaN/zeros.
///
/// Use this when Python reference data includes zeros for warmup period
/// but Rust indicator only outputs computed values.
pub fn compare_values_with_warmup(
    name: &str,
    rust_values: &[f64],
    true_values: &[f64],
    warmup: usize,
    epsilon: f64,
) -> usize {
    let expected_offset = &true_values[warmup..];
    let mut comparisons = 0;

    for (i, (rust_val, true_val)) in rust_values.iter().zip(expected_offset.iter()).enumerate() {
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            assert!(
                diff < epsilon,
                "{} mismatch at data index {} (rust[{}]): Rust={:.6}, Expected={:.6}, diff={:.6}",
                name,
                i + warmup,
                i,
                rust_val,
                true_val,
                diff
            );
            comparisons += 1;
        }
    }
    comparisons
}

/// Compare using relative error (for large cumulative values like OBV)
pub fn compare_values_relative(
    name: &str,
    rust_values: &[f64],
    true_values: &[f64],
    tolerance: f64,
) -> usize {
    let mut comparisons = 0;
    for (i, (rust_val, true_val)) in rust_values.iter().zip(true_values.iter()).enumerate() {
        if !rust_val.is_nan() && !true_val.is_nan() && true_val.abs() > 1e-10 {
            let rel_diff = (rust_val - true_val).abs() / true_val.abs();
            assert!(
                rel_diff < tolerance,
                "{} mismatch at index {}: Rust={:.6}, Expected={:.6}, rel_diff={:.6}",
                name,
                i,
                rust_val,
                true_val,
                rel_diff
            );
            comparisons += 1;
        }
    }
    comparisons
}

// ============================================================
// Compound Output Comparison Helpers
// ============================================================

/// Assert two BB histories are equal within epsilon
#[allow(dead_code)]
pub fn assert_bb_histories_equal(context: &str, a: &[BBOutput], b: &[BBOutput], epsilon: f64) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());

    for i in 0..a.len() {
        assert_values_equal(&format!("{} upper", context), i, a[i].upper, b[i].upper, epsilon);
        assert_values_equal(&format!("{} middle", context), i, a[i].middle, b[i].middle, epsilon);
        assert_values_equal(&format!("{} lower", context), i, a[i].lower, b[i].lower, epsilon);
    }
}

/// Assert two ADX histories are equal within epsilon
#[allow(dead_code)]
pub fn assert_adx_histories_equal(context: &str, a: &[ADXOutput], b: &[ADXOutput], epsilon: f64) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());

    for i in 0..a.len() {
        assert_values_equal(&format!("{} +DI", context), i, a[i].plus_di, b[i].plus_di, epsilon);
        assert_values_equal(&format!("{} -DI", context), i, a[i].minus_di, b[i].minus_di, epsilon);
        assert_values_equal(&format!("{} DX", context), i, a[i].dx, b[i].dx, epsilon);
        assert_values_equal(&format!("{} ADX", context), i, a[i].adx, b[i].adx, epsilon);
    }
}

/// Assert two DMI histories are equal within epsilon
#[allow(dead_code)]
pub fn assert_dmi_histories_equal(context: &str, a: &[DMIOutput], b: &[DMIOutput], epsilon: f64) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());

    for i in 0..a.len() {
        assert_values_equal(&format!("{} +DI", context), i, a[i].plus_di, b[i].plus_di, epsilon);
        assert_values_equal(&format!("{} -DI", context), i, a[i].minus_di, b[i].minus_di, epsilon);
    }
}

/// Assert two Ichimoku histories are equal within epsilon
#[allow(dead_code)]
pub fn assert_ichimoku_histories_equal(
    context: &str,
    a: &[IchimokuOutput],
    b: &[IchimokuOutput],
    epsilon: f64,
) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());

    for i in 0..a.len() {
        assert_values_equal(&format!("{} tenkan", context), i, a[i].tenkan, b[i].tenkan, epsilon);
        assert_values_equal(&format!("{} kijun", context), i, a[i].kijun, b[i].kijun, epsilon);
        assert_values_equal(&format!("{} senkou_a", context), i, a[i].senkou_a, b[i].senkou_a, epsilon);
        assert_values_equal(&format!("{} senkou_b", context), i, a[i].senkou_b, b[i].senkou_b, epsilon);
    }
}

/// Assert two Donchian histories are equal within epsilon
#[allow(dead_code)]
pub fn assert_donchian_histories_equal(
    context: &str,
    a: &[DonchianOutput],
    b: &[DonchianOutput],
    epsilon: f64,
) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());

    for i in 0..a.len() {
        assert_values_equal(&format!("{} upper", context), i, a[i].upper, b[i].upper, epsilon);
        assert_values_equal(&format!("{} middle", context), i, a[i].middle, b[i].middle, epsilon);
        assert_values_equal(&format!("{} lower", context), i, a[i].lower, b[i].lower, epsilon);
    }
}

/// Assert two StochRSI histories are equal within epsilon
#[allow(dead_code)]
pub fn assert_stochrsi_histories_equal(
    context: &str,
    a: &[StochRSIOutput],
    b: &[StochRSIOutput],
    epsilon: f64,
) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());

    for i in 0..a.len() {
        assert_values_equal(&format!("{} k", context), i, a[i].k, b[i].k, epsilon);
        assert_values_equal(&format!("{} d", context), i, a[i].d, b[i].d, epsilon);
    }
}

/// Helper to compare two values, handling NaN
fn assert_values_equal(context: &str, index: usize, a: f64, b: f64, epsilon: f64) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    if a.is_nan() || b.is_nan() {
        panic!("{} at index {}: NaN mismatch ({} vs {})", context, index, a, b);
    }
    assert!(
        (a - b).abs() < epsilon,
        "{} at index {}: {} != {}",
        context,
        index,
        a,
        b
    );
}