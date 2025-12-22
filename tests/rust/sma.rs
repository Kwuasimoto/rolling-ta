//! SMA reference tests.
//!
//! Verifies SMA implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_from_closes, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{SMA, SMAConfig};

#[test]
fn sma_batch_vs_reference() {
    // btc-sma.xlsx: timestamp(0), close(1), sma(2)
    let cols = read_xlsx_by_position("resources/data/btc-sma.xlsx", &[1, 2]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    let mut sma = SMA::new(SMAConfig::new(14));
    sma.calc(&candles).unwrap();

    let comparisons = compare_values("SMA batch vs reference", sma.history(), expected, EPSILON);
    println!(
        "SMA: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn sma_streaming_next_vs_batch() {
    // btc-sma.xlsx: timestamp(0), close(1), sma(2)
    let cols = read_xlsx_by_position("resources/data/btc-sma.xlsx", &[1, 2]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    let period = 14;

    // 1. Batch calculation - history includes NaN for warmup period
    let mut batch = SMA::new(SMAConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    // Each call to next() receives a growing snapshot
    // Note: next() only pushes to history when returning Some (not during warmup)
    let mut stream = SMA::new(SMAConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // History formats differ:
    // - batch.history(): [NaN × (period-1), val1, val2, ...] (length = candles.len())
    // - stream.history(): [val1, val2, ...] (length = candles.len() - period + 1)
    //
    // Compare computed values only (skip NaN warmup from batch)
    let batch_computed: Vec<f64> = batch.history().iter().filter(|v| !v.is_nan()).copied().collect();
    let stream_computed = stream.history();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match"
    );

    let epsilon = 1e-6;
    assert_histories_equal("SMA batch vs stream", &batch_computed, stream_computed, epsilon);
}

#[test]
fn sma_next_with_fixed_window() {
    // Simulates SharedWindow pattern where snapshot size is limited
    // This is how IndicatorManager would call next()
    let cols = read_xlsx_by_position("resources/data/btc-sma.xlsx", &[1, 2]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    let period = 14;
    let mut sma = SMA::new(SMAConfig::new(period));

    // Feed snapshots of exactly `period` candles (like SharedWindow.snapshot_last(14))
    for i in period..candles.len() {
        let snapshot = &candles[i - period + 1..=i];
        let result = sma.next(snapshot);

        // Should always produce a value once we have enough data
        assert!(result.is_some(), "Should have result at index {}", i);

        // Compare against Python reference
        let rust_val = result.unwrap();
        let true_val = expected[i];
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            assert!(
                diff < EPSILON,
                "SMA mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
                i, rust_val, true_val, diff
            );
        }
    }
}
