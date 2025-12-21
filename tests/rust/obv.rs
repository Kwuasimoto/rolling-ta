//! OBV reference tests.
//!
//! Verifies OBV implementation against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.
//! OBV uses relative error tolerance due to large cumulative values.

use crate::common::{
    assert_histories_equal, build_ohlcv_cv, compare_values_relative, read_xlsx_by_position,
    slice_ohlcv, tick_at,
};
use rolling_ta::prelude::*;
use rolling_ta::volume::OBV;

#[test]
fn obv_modes_equivalent() {
    // btc-obv.xlsx: timestamp(0), close(1), volume(2), up(3), down(4), obv(5)
    let cols = read_xlsx_by_position("resources/data/btc-obv.xlsx", &[1, 2, 5]);
    let closes = &cols[0];
    let volumes = &cols[1];
    let expected_obv = &cols[2];
    let data = build_ohlcv_cv(closes, volumes);

    // 1. Batch calculation - validate OBV against Python reference (relative error)
    let mut batch = OBV::default();
    batch.calc(&data).unwrap();
    let obv_comparisons = compare_values_relative(
        "OBV batch vs reference",
        batch.history(),
        expected_obv,
        0.001, // 0.1% tolerance for large cumulative values
    );
    println!("OBV: {} values compared against Python reference", obv_comparisons);
    assert!(obv_comparisons > 100, "Should have compared many OBV values, got {}", obv_comparisons);

    // 2. Full streaming (tests update() from scratch, including warmup)
    let mut stream = OBV::default();
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 3. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = OBV::default();
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_histories_equal("OBV batch vs stream", batch.history(), stream.history(), epsilon);
    assert_histories_equal("OBV batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
