//! Donchian Channel reference tests.
//!
//! Verifies Donchian implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! Donchian has compound output: upper, middle, lower.

use crate::common::{
    assert_donchian_histories_equal, build_candles_hlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volatility::{Donchian, DonchianConfig};

#[test]
fn donchian_batch_vs_reference() {
    // btc-donchian.xlsx: timestamp(0), high(1), low(2), highs/upper(3), lows/lower(4), centers/middle(5)
    let cols = read_xlsx_by_position("resources/data/btc-donchian.xlsx", &[1, 2, 3, 4, 5]);
    let highs = &cols[0];
    let lows = &cols[1];
    let expected_upper = &cols[2];
    let expected_lower = &cols[3];
    let expected_middle = &cols[4];
    let candles = build_candles_hlc(highs, lows, highs); // use high as close (not used in Donchian)

    // Batch calculation (validates against Python reference)
    // Python default period is 14, not 20
    let mut donchian = Donchian::new(DonchianConfig::new(14));
    donchian.calc(&candles).unwrap();

    let history = donchian.history();
    let rust_upper: Vec<f64> = history.iter().map(|o| o.upper).collect();
    let rust_middle: Vec<f64> = history.iter().map(|o| o.middle).collect();
    let rust_lower: Vec<f64> = history.iter().map(|o| o.lower).collect();

    let upper_cmp = compare_values("Donchian upper vs reference", &rust_upper, expected_upper, EPSILON);
    let middle_cmp = compare_values("Donchian middle vs reference", &rust_middle, expected_middle, EPSILON);
    let lower_cmp = compare_values("Donchian lower vs reference", &rust_lower, expected_lower, EPSILON);

    println!(
        "Donchian: upper={}, middle={}, lower={} values compared against Python reference",
        upper_cmp, middle_cmp, lower_cmp
    );
    assert!(upper_cmp > 100, "Should have compared many upper values, got {}", upper_cmp);
    assert!(middle_cmp > 100, "Should have compared many middle values, got {}", middle_cmp);
    assert!(lower_cmp > 100, "Should have compared many lower values, got {}", lower_cmp);
}

#[test]
fn donchian_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-donchian.xlsx", &[1, 2]);
    let highs = &cols[0];
    let lows = &cols[1];
    let candles = build_candles_hlc(highs, lows, highs);

    // Python default period is 14
    let period = 14;

    // 1. Batch calculation
    let mut batch = Donchian::new(DonchianConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = Donchian::new(DonchianConfig::new(period));
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
    assert_donchian_histories_equal("Donchian batch vs stream", &batch_computed, &stream_computed, epsilon);
}

#[test]
fn donchian_next_with_fixed_window() {
    let cols = read_xlsx_by_position("resources/data/btc-donchian.xlsx", &[1, 2, 3, 4, 5]);
    let highs = &cols[0];
    let lows = &cols[1];
    let expected_upper = &cols[2];
    let expected_lower = &cols[3];
    let expected_middle = &cols[4];
    let candles = build_candles_hlc(highs, lows, highs);

    // Python default period is 14
    let period = 14;
    let mut donchian = Donchian::new(DonchianConfig::new(period));

    // Feed snapshots of exactly period candles (matching Python behavior)
    for i in (period - 1)..candles.len() {
        let snapshot = &candles[i + 1 - period..=i];
        let result = donchian.next(snapshot);

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
                "Donchian mismatch at index {}: upper=({:.4} vs {:.4}), middle=({:.4} vs {:.4}), lower=({:.4} vs {:.4})",
                i, output.upper, true_upper, output.middle, true_middle, output.lower, true_lower
            );
        }
    }
}
