//! HMA reference tests.
//!
//! Verifies HMA implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_from_closes, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{HMA, HMAConfig};

#[test]
fn hma_batch_vs_reference() {
    // btc-hma.xlsx: timestamp(0), close(1), ..., hma(11)
    let cols = read_xlsx_by_position("resources/data/btc-hma.xlsx", &[1, 11]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    let mut hma = HMA::new(HMAConfig::new(14));
    hma.calc(&candles).unwrap();

    let comparisons = compare_values("HMA batch vs reference", hma.history(), expected, EPSILON);
    println!(
        "HMA: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn hma_streaming_next_vs_batch() {
    // btc-hma.xlsx: timestamp(0), close(1), ..., hma(11)
    let cols = read_xlsx_by_position("resources/data/btc-hma.xlsx", &[1, 11]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    let period = 14;

    // 1. Batch calculation - history includes NaN for warmup period
    let mut batch = HMA::new(HMAConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    // Each call to next() receives a growing snapshot
    // Note: next() only pushes to history when returning Some (not during warmup)
    let mut stream = HMA::new(HMAConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // History formats differ:
    // - batch.history(): [NaN × warmup, val1, val2, ...] (length = candles.len())
    // - stream.history(): [val1, val2, ...] (length = candles.len() - warmup)
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
    assert_histories_equal("HMA batch vs stream", &batch_computed, stream_computed, epsilon);
}

#[test]
fn hma_next_with_fixed_window() {
    // Simulates SharedWindow pattern where snapshot grows over time
    let cols = read_xlsx_by_position("resources/data/btc-hma.xlsx", &[1, 11]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    let period = 14;
    let mut hma = HMA::new(HMAConfig::new(period));
    let warmup = hma.warmup_period();

    // Feed growing snapshots starting from warmup point
    for i in warmup..candles.len() {
        let snapshot = &candles[..=i];
        let result = hma.next(snapshot);

        // Should always produce a value once we have enough data
        assert!(result.is_some(), "Should have result at index {}", i);

        // Compare against Python reference
        let rust_val = result.unwrap();
        let true_val = expected[i];
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            assert!(
                diff < EPSILON,
                "HMA mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
                i, rust_val, true_val, diff
            );
        }
    }
}
