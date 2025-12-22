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