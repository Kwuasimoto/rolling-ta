//! VWAP reference tests.
//!
//! Verifies VWAP implementation against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.
//! VWAP uses timestamps for daily reset interval (86400 seconds).

use crate::common::{
    assert_histories_equal, build_ohlcv_thlcv, compare_values, read_xlsx_by_position,
    slice_ohlcv, tick_at, EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volume::{VWAP, VWAPConfig};

#[test]
fn vwap_modes_equivalent() {
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
    let data = build_ohlcv_thlcv(timestamps, highs, lows, closes, volumes);

    // 1. Batch calculation - validate VWAP against Python reference
    // Python uses reset_interval = 1440 * 60 = 86400 (daily)
    let mut batch = VWAP::new(VWAPConfig::new(86400));
    batch.calc(&data).unwrap();
    let vwap_comparisons = compare_values("VWAP batch vs reference", batch.history(), expected_vwap, EPSILON);
    println!("VWAP: {} values compared against Python reference", vwap_comparisons);
    assert!(vwap_comparisons > 100, "Should have compared many VWAP values, got {}", vwap_comparisons);

    // 2. Full streaming (tests update() from scratch, including warmup)
    let mut stream = VWAP::new(VWAPConfig::new(86400));
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 3. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = VWAP::new(VWAPConfig::new(86400));
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_histories_equal("VWAP batch vs stream", batch.history(), stream.history(), epsilon);
    assert_histories_equal("VWAP batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
