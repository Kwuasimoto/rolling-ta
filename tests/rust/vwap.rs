//! VWAP reference tests.
//!
//! Verifies VWAP implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! VWAP uses timestamps for daily reset interval (86400 seconds).

use crate::common::{
    assert_histories_equal, build_candles_thlcv, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volume::{VWAP, VWAPConfig};

#[test]
fn vwap_batch_vs_reference() {
    // btc-vwap.xlsx: timestamp(0), timestamp_mod(1), high(2), low(3), close(4),
    //                typical(5), volume(6), raw_accum(7), vol_accum(8), vwap(9)
    let cols = read_xlsx_by_position(
        "resources/data/btc-vwap.xlsx",
        &[0, 2, 3, 4, 6, 9], // timestamp, high, low, close, volume, vwap
    );
    let timestamps = &cols[0];
    let highs = &cols[1];
    let lows = &cols[2];
    let closes = &cols[3];
    let volumes = &cols[4];
    let expected_vwap = &cols[5];
    let candles = build_candles_thlcv(timestamps, highs, lows, closes, volumes);

    // Batch calculation (validates against Python reference)
    // Python uses reset_interval = 1440 * 60 = 86400 (daily)
    let mut vwap = VWAP::new(VWAPConfig::new(86400));
    vwap.calc(&candles).unwrap();

    let comparisons = compare_values("VWAP batch vs reference", vwap.history(), expected_vwap, EPSILON);
    println!("VWAP: {} values compared against Python reference", comparisons);
    assert!(comparisons > 100, "Should have compared many VWAP values, got {}", comparisons);
}

#[test]
fn vwap_streaming_next_vs_batch() {
    // btc-vwap.xlsx: timestamp(0), timestamp_mod(1), high(2), low(3), close(4),
    //                typical(5), volume(6), raw_accum(7), vol_accum(8), vwap(9)
    let cols = read_xlsx_by_position(
        "resources/data/btc-vwap.xlsx",
        &[0, 2, 3, 4, 6], // timestamp, high, low, close, volume
    );
    let timestamps = &cols[0];
    let highs = &cols[1];
    let lows = &cols[2];
    let closes = &cols[3];
    let volumes = &cols[4];
    let candles = build_candles_thlcv(timestamps, highs, lows, closes, volumes);

    // 1. Batch calculation
    let mut batch = VWAP::new(VWAPConfig::new(86400));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    let mut stream = VWAP::new(VWAPConfig::new(86400));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // VWAP doesn't have warmup period - compare full histories
    let epsilon = 1e-6;
    assert_histories_equal("VWAP batch vs stream", batch.history(), stream.history(), epsilon);
}
