//! ATR reference tests.
//!
//! Verifies ATR and TR implementations against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_hlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volatility::{ATR, ATRConfig, TR};

#[test]
fn tr_batch_vs_reference() {
    // btc-atr.xlsx: timestamp(0), high(1), low(2), close(3), tr(4), atr(5)
    let cols = read_xlsx_by_position("resources/data/btc-atr.xlsx", &[1, 2, 3, 4]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_tr = &cols[3];
    let candles = build_candles_hlc(highs, lows, closes);

    // Batch calculation (validates TR against Python reference)
    let mut tr = TR::default();
    tr.calc(&candles).unwrap();

    let comparisons = compare_values("TR batch vs reference", tr.history(), expected_tr, EPSILON);
    println!("TR: {} values compared against Python reference", comparisons);
    assert!(comparisons > 100, "Should have compared many TR values, got {}", comparisons);
}

#[test]
fn atr_batch_vs_reference() {
    // btc-atr.xlsx: timestamp(0), high(1), low(2), close(3), tr(4), atr(5)
    let cols = read_xlsx_by_position("resources/data/btc-atr.xlsx", &[1, 2, 3, 4, 5]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_atr = &cols[4];
    let candles = build_candles_hlc(highs, lows, closes);

    // Batch calculation (validates ATR against Python reference)
    let mut atr = ATR::new(ATRConfig::new(14));
    atr.calc(&candles).unwrap();

    let comparisons = compare_values("ATR batch vs reference", atr.history(), expected_atr, EPSILON);
    println!("ATR: {} values compared against Python reference", comparisons);
    assert!(comparisons > 100, "Should have compared many ATR values, got {}", comparisons);
}

#[test]
fn atr_streaming_next_vs_batch() {
    // btc-atr.xlsx: timestamp(0), high(1), low(2), close(3), tr(4), atr(5)
    let cols = read_xlsx_by_position("resources/data/btc-atr.xlsx", &[1, 2, 3]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let candles = build_candles_hlc(highs, lows, closes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = ATR::new(ATRConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    let mut stream = ATR::new(ATRConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from both)
    let batch_computed: Vec<f64> = batch.history().iter().filter(|&v| !v.is_nan()).copied().collect();
    let stream_computed: Vec<f64> = stream.history().iter().filter(|&v| !v.is_nan()).copied().collect();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_histories_equal("ATR batch vs stream", &batch_computed, &stream_computed, epsilon);
}
