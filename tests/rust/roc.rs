//! ROC reference tests.
//!
//! Verifies ROC (Rate of Change) implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_from_closes, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::momentum::{ROC, ROCConfig};
use rolling_ta::prelude::*;

#[test]
fn roc_batch_vs_reference() {
    // btc-roc.xlsx: timestamp(0), close(1), roc(2)
    let cols = read_xlsx_by_position("resources/data/btc-roc.xlsx", &[1, 2]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    let mut roc = ROC::new(ROCConfig::new(14));
    roc.calc(&candles).unwrap();

    let comparisons = compare_values("ROC batch vs reference", roc.history(), expected, EPSILON);
    println!(
        "ROC: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn roc_streaming_next_vs_batch() {
    // btc-roc.xlsx: timestamp(0), close(1), roc(2)
    let cols = read_xlsx_by_position("resources/data/btc-roc.xlsx", &[1, 2]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    let period = 14;

    // 1. Batch calculation - history includes NaN for warmup period
    let mut batch = ROC::new(ROCConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    let mut stream = ROC::new(ROCConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from batch)
    let batch_computed: Vec<f64> = batch.history().iter().filter(|&v| !v.is_nan()).copied().collect();
    let stream_computed = stream.history();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_histories_equal("ROC batch vs stream", &batch_computed, stream_computed, epsilon);
}

#[test]
fn roc_next_with_fixed_window() {
    // Simulates SharedWindow pattern where snapshot size is limited
    let cols = read_xlsx_by_position("resources/data/btc-roc.xlsx", &[1, 2]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    // ROC needs period + 1 candles minimum
    let period = 14;
    let window_size = period + 1;
    let mut roc = ROC::new(ROCConfig::new(period));

    // Feed snapshots of fixed window size
    for i in window_size..candles.len() {
        let snapshot = &candles[i - window_size + 1..=i];
        let result = roc.next(snapshot);

        assert!(result.is_some(), "Should have result at index {}", i);

        let rust_val = result.unwrap();
        let true_val = expected[i];
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            assert!(
                diff < EPSILON,
                "ROC mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
                i, rust_val, true_val, diff
            );
        }
    }
}
