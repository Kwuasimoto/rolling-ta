//! OBV reference tests.
//!
//! Verifies OBV implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! OBV uses relative error tolerance due to large cumulative values.

use crate::common::{
    assert_histories_equal, build_candles_cv, compare_values_relative, read_xlsx_by_position,
};
use rolling_ta::prelude::*;
use rolling_ta::volume::OBV;

#[test]
fn obv_batch_vs_reference() {
    // btc-obv.xlsx: timestamp(0), close(1), volume(2), up(3), down(4), obv(5)
    let cols = read_xlsx_by_position("resources/data/btc-obv.xlsx", &[1, 2, 5]);
    let closes = &cols[0];
    let volumes = &cols[1];
    let expected = &cols[2];
    let candles = build_candles_cv(closes, volumes);

    // Batch calculation (validates against Python reference)
    let mut obv = OBV::default();
    obv.calc(&candles).unwrap();

    let comparisons = compare_values_relative(
        "OBV batch vs reference",
        obv.history(),
        expected,
        0.001, // 0.1% tolerance for large cumulative values
    );
    println!(
        "OBV: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn obv_streaming_next_vs_batch() {
    // btc-obv.xlsx: timestamp(0), close(1), volume(2), up(3), down(4), obv(5)
    let cols = read_xlsx_by_position("resources/data/btc-obv.xlsx", &[1, 2, 5]);
    let closes = &cols[0];
    let volumes = &cols[1];
    let candles = build_candles_cv(closes, volumes);

    // 1. Batch calculation
    let mut batch = OBV::default();
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    let mut stream = OBV::default();
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // OBV doesn't have warmup period - compare full histories
    let epsilon = 1e-6;
    assert_histories_equal("OBV batch vs stream", batch.history(), stream.history(), epsilon);
}

