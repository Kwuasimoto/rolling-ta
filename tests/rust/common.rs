//! Shared test utilities for reference tests.
//!
//! Provides helpers for loading xlsx fixtures, building OHLCV data,
//! and comparing indicator outputs.

use calamine::{open_workbook, Data, Reader, Xlsx};
use rolling_ta::prelude::*;

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

/// Build OhlcvSeries from closes only
pub fn build_ohlcv_from_closes(closes: &[f64]) -> OhlcvSeries {
    OhlcvSeries::from_closes(closes)
}

/// Build OhlcvSeries with high, low, close
pub fn build_ohlcv_hlc(highs: &[f64], lows: &[f64], closes: &[f64]) -> OhlcvSeries {
    let mut data = OhlcvSeries::with_capacity(closes.len());
    for i in 0..closes.len() {
        data.timestamps.push(i as i64);
        data.opens.push(closes[i]);
        data.highs.push(highs[i]);
        data.lows.push(lows[i]);
        data.closes.push(closes[i]);
        data.volumes.push(0.0);
    }
    data
}

/// Build OhlcvSeries with close and volume
pub fn build_ohlcv_cv(closes: &[f64], volumes: &[f64]) -> OhlcvSeries {
    let mut data = OhlcvSeries::with_capacity(closes.len());
    for i in 0..closes.len() {
        data.timestamps.push(i as i64);
        data.opens.push(closes[i]);
        data.highs.push(closes[i]);
        data.lows.push(closes[i]);
        data.closes.push(closes[i]);
        data.volumes.push(volumes[i]);
    }
    data
}

/// Build OhlcvSeries with timestamp, high, low, close, volume (for VWAP)
pub fn build_ohlcv_thlcv(
    timestamps: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
) -> OhlcvSeries {
    let mut data = OhlcvSeries::with_capacity(closes.len());
    for i in 0..closes.len() {
        data.timestamps.push(timestamps[i] as i64);
        data.opens.push(closes[i]);
        data.highs.push(highs[i]);
        data.lows.push(lows[i]);
        data.closes.push(closes[i]);
        data.volumes.push(volumes[i]);
    }
    data
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

/// Compare two f64 slices using relative error (for large values like OBV)
pub fn compare_values_relative(name: &str, rust_values: &[f64], true_values: &[f64], rel_tolerance: f64) -> usize {
    let mut comparisons = 0;
    for (i, (rust_val, true_val)) in rust_values.iter().zip(true_values.iter()).enumerate() {
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            let relative_err = if true_val.abs() > 1.0 {
                diff / true_val.abs()
            } else {
                diff
            };
            assert!(
                relative_err < rel_tolerance,
                "{} mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}, rel_err={:.6}",
                name,
                i,
                rust_val,
                true_val,
                diff,
                relative_err
            );
            comparisons += 1;
        }
    }
    comparisons
}

/// Build an Ohlcv tick from OhlcvSeries at given index
pub fn tick_at(data: &OhlcvSeries, i: usize) -> Ohlcv {
    Ohlcv {
        open: data.opens[i].into(),
        high: data.highs[i].into(),
        low: data.lows[i].into(),
        close: data.closes[i].into(),
        volume: data.volumes[i].into(),
        timestamp: data.timestamps[i].into(),
    }
}

/// Slice an OhlcvSeries from start to end (exclusive)
pub fn slice_ohlcv(data: &OhlcvSeries, start: usize, end: usize) -> OhlcvSeries {
    OhlcvSeries {
        timestamps: data.timestamps[start..end].to_vec(),
        opens: data.opens[start..end].to_vec(),
        highs: data.highs[start..end].to_vec(),
        lows: data.lows[start..end].to_vec(),
        closes: data.closes[start..end].to_vec(),
        volumes: data.volumes[start..end].to_vec(),
    }
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

/// Assert two ADX histories are equal within epsilon (compound output: plus_di, minus_di, dx, adx)
pub fn assert_adx_histories_equal(
    context: &str,
    a: &[rolling_ta::trend::ADXOutput],
    b: &[rolling_ta::trend::ADXOutput],
    epsilon: f64,
) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        context,
        a.len(),
        b.len()
    );

    for i in 0..a.len() {
        // Compare plus_di
        let (a_plus, b_plus) = (a[i].plus_di, b[i].plus_di);
        if a_plus.is_nan() && b_plus.is_nan() {
            // ok
        } else if a_plus.is_nan() || b_plus.is_nan() {
            panic!("{} plus_di at index {}: NaN mismatch ({} vs {})", context, i, a_plus, b_plus);
        } else {
            assert!(
                (a_plus - b_plus).abs() < epsilon,
                "{} plus_di at index {}: {} != {}",
                context, i, a_plus, b_plus
            );
        }

        // Compare minus_di
        let (a_minus, b_minus) = (a[i].minus_di, b[i].minus_di);
        if a_minus.is_nan() && b_minus.is_nan() {
            // ok
        } else if a_minus.is_nan() || b_minus.is_nan() {
            panic!("{} minus_di at index {}: NaN mismatch ({} vs {})", context, i, a_minus, b_minus);
        } else {
            assert!(
                (a_minus - b_minus).abs() < epsilon,
                "{} minus_di at index {}: {} != {}",
                context, i, a_minus, b_minus
            );
        }

        // Compare dx
        let (a_dx, b_dx) = (a[i].dx, b[i].dx);
        if a_dx.is_nan() && b_dx.is_nan() {
            // ok
        } else if a_dx.is_nan() || b_dx.is_nan() {
            panic!("{} dx at index {}: NaN mismatch ({} vs {})", context, i, a_dx, b_dx);
        } else {
            assert!(
                (a_dx - b_dx).abs() < epsilon,
                "{} dx at index {}: {} != {}",
                context, i, a_dx, b_dx
            );
        }

        // Compare adx
        let (a_adx, b_adx) = (a[i].adx, b[i].adx);
        if a_adx.is_nan() && b_adx.is_nan() {
            // ok
        } else if a_adx.is_nan() || b_adx.is_nan() {
            panic!("{} adx at index {}: NaN mismatch ({} vs {})", context, i, a_adx, b_adx);
        } else {
            assert!(
                (a_adx - b_adx).abs() < epsilon,
                "{} adx at index {}: {} != {}",
                context, i, a_adx, b_adx
            );
        }
    }
}

/// Assert two BB histories are equal within epsilon (compound output: upper, middle, lower)
pub fn assert_bb_histories_equal(
    context: &str,
    a: &[rolling_ta::volatility::BBOutput],
    b: &[rolling_ta::volatility::BBOutput],
    epsilon: f64,
) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        context,
        a.len(),
        b.len()
    );

    for i in 0..a.len() {
        // Compare upper
        let (a_upper, b_upper) = (a[i].upper, b[i].upper);
        if a_upper.is_nan() && b_upper.is_nan() {
            // ok
        } else if a_upper.is_nan() || b_upper.is_nan() {
            panic!("{} upper at index {}: NaN mismatch ({} vs {})", context, i, a_upper, b_upper);
        } else {
            assert!(
                (a_upper - b_upper).abs() < epsilon,
                "{} upper at index {}: {} != {}",
                context, i, a_upper, b_upper
            );
        }

        // Compare middle
        let (a_middle, b_middle) = (a[i].middle, b[i].middle);
        if a_middle.is_nan() && b_middle.is_nan() {
            // ok
        } else if a_middle.is_nan() || b_middle.is_nan() {
            panic!("{} middle at index {}: NaN mismatch ({} vs {})", context, i, a_middle, b_middle);
        } else {
            assert!(
                (a_middle - b_middle).abs() < epsilon,
                "{} middle at index {}: {} != {}",
                context, i, a_middle, b_middle
            );
        }

        // Compare lower
        let (a_lower, b_lower) = (a[i].lower, b[i].lower);
        if a_lower.is_nan() && b_lower.is_nan() {
            // ok
        } else if a_lower.is_nan() || b_lower.is_nan() {
            panic!("{} lower at index {}: NaN mismatch ({} vs {})", context, i, a_lower, b_lower);
        } else {
            assert!(
                (a_lower - b_lower).abs() < epsilon,
                "{} lower at index {}: {} != {}",
                context, i, a_lower, b_lower
            );
        }
    }
}
