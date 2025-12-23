//! Bollinger Bands reference tests.
//!
//! Verifies BB implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! BB has compound output: upper, middle, lower bands.

use crate::common::{
    assert_bb_histories_equal, build_candles_from_closes, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volatility::{BB, BBConfig};

#[test]
fn bb_batch_vs_reference() {
    // btc-bb.xlsx: timestamp(0), close(1), sma/middle(2), upper(3), lower(4)
    let cols = read_xlsx_by_position("resources/data/btc-bb.xlsx", &[1, 2, 3, 4]);
    let closes = &cols[0];
    let expected_middle = &cols[1];
    let expected_upper = &cols[2];
    let expected_lower = &cols[3];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    let mut bb = BB::new(BBConfig::new(20, 2.0));
    bb.calc(&candles).unwrap();

    // Extract component values from BBOutput
    let history = bb.history();
    let rust_upper: Vec<f64> = history.iter().map(|o| o.upper).collect();
    let rust_middle: Vec<f64> = history.iter().map(|o| o.middle).collect();
    let rust_lower: Vec<f64> = history.iter().map(|o| o.lower).collect();

    let upper_cmp = compare_values("BB upper vs reference", &rust_upper, expected_upper, EPSILON);
    let middle_cmp = compare_values("BB middle vs reference", &rust_middle, expected_middle, EPSILON);
    let lower_cmp = compare_values("BB lower vs reference", &rust_lower, expected_lower, EPSILON);

    println!(
        "BB: upper={}, middle={}, lower={} values compared against Python reference",
        upper_cmp, middle_cmp, lower_cmp
    );
    assert!(upper_cmp > 100, "Should have compared many upper values, got {}", upper_cmp);
    assert!(middle_cmp > 100, "Should have compared many middle values, got {}", middle_cmp);
    assert!(lower_cmp > 100, "Should have compared many lower values, got {}", lower_cmp);
}

#[test]
fn bb_streaming_next_vs_batch() {
    // btc-bb.xlsx: timestamp(0), close(1), sma/middle(2), upper(3), lower(4)
    let cols = read_xlsx_by_position("resources/data/btc-bb.xlsx", &[1, 2, 3, 4]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    let period = 20;

    // 1. Batch calculation - history includes NaN-like values for warmup period
    let mut batch = BB::new(BBConfig::new(period, 2.0));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    let mut stream = BB::new(BBConfig::new(period, 2.0));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from both)
    let batch_computed: Vec<_> = batch.history().iter().filter(|o| !o.middle.is_nan()).cloned().collect();
    let stream_computed: Vec<_> = stream.history().iter().filter(|o| !o.middle.is_nan()).cloned().collect();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_bb_histories_equal("BB batch vs stream", &batch_computed, &stream_computed, epsilon);
}

#[test]
fn bb_next_with_fixed_window() {
    // Simulates SharedWindow pattern where snapshot size is limited
    let cols = read_xlsx_by_position("resources/data/btc-bb.xlsx", &[1, 2, 3, 4]);
    let closes = &cols[0];
    let expected_middle = &cols[1];
    let expected_upper = &cols[2];
    let expected_lower = &cols[3];
    let candles = build_candles_from_closes(closes);

    let period = 20;
    let mut bb = BB::new(BBConfig::new(period, 2.0));

    // Feed snapshots of exactly `period` candles
    for i in period..candles.len() {
        let snapshot = &candles[i - period + 1..=i];
        let result = bb.next(snapshot);

        assert!(result.is_some(), "Should have result at index {}", i);

        let output = result.unwrap();
        let true_upper = expected_upper[i];
        let true_middle = expected_middle[i];
        let true_lower = expected_lower[i];

        if !output.middle.is_nan() && !true_middle.is_nan() {
            let diff_upper = (output.upper - true_upper).abs();
            let diff_middle = (output.middle - true_middle).abs();
            let diff_lower = (output.lower - true_lower).abs();

            assert!(
                diff_upper < EPSILON && diff_middle < EPSILON && diff_lower < EPSILON,
                "BB mismatch at index {}: upper=({:.4} vs {:.4}), middle=({:.4} vs {:.4}), lower=({:.4} vs {:.4})",
                i, output.upper, true_upper, output.middle, true_middle, output.lower, true_lower
            );
        }
    }
}
