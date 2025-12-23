//! CMF reference tests.
//!
//! Verifies CMF (Chaikin Money Flow) implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_hlcv, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volume::{CMF, CMFConfig};

#[test]
fn cmf_batch_vs_reference() {
    // btc-cmf.xlsx: timestamp(0), high(1), low(2), close(3), volume(4), mfm(5), mfv(6), cmf(7)
    let cols = read_xlsx_by_position("resources/data/btc-cmf.xlsx", &[1, 2, 3, 4, 7]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let volumes = &cols[3];
    let expected_cmf = &cols[4];
    let candles = build_candles_hlcv(highs, lows, closes, volumes);

    // Batch calculation (validates against Python reference)
    let mut cmf = CMF::new(CMFConfig::new(20));
    cmf.calc(&candles).unwrap();

    let comparisons = compare_values("CMF batch vs reference", cmf.history(), expected_cmf, EPSILON);
    println!(
        "CMF: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn cmf_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-cmf.xlsx", &[1, 2, 3, 4]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let volumes = &cols[3];
    let candles = build_candles_hlcv(highs, lows, closes, volumes);

    let period = 20;

    // 1. Batch calculation
    let mut batch = CMF::new(CMFConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = CMF::new(CMFConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only
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
    assert_histories_equal("CMF batch vs stream", &batch_computed, stream_computed, epsilon);
}

#[test]
fn cmf_next_with_fixed_window() {
    let cols = read_xlsx_by_position("resources/data/btc-cmf.xlsx", &[1, 2, 3, 4, 7]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let volumes = &cols[3];
    let expected_cmf = &cols[4];
    let candles = build_candles_hlcv(highs, lows, closes, volumes);

    let period = 20;
    let mut cmf = CMF::new(CMFConfig::new(period));

    // Feed snapshots of fixed window size
    for i in period..candles.len() {
        let snapshot = &candles[i - period + 1..=i];
        let result = cmf.next(snapshot);

        assert!(result.is_some(), "Should have result at index {}", i);

        let rust_val = result.unwrap();
        let true_val = expected_cmf[i];
        if !rust_val.is_nan() && !true_val.is_nan() {
            let diff = (rust_val - true_val).abs();
            assert!(
                diff < EPSILON,
                "CMF mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
                i, rust_val, true_val, diff
            );
        }
    }
}
