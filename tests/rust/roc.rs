//! ROC reference tests.
//!
//! Verifies ROC (Rate of Change) implementation against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.

use crate::common::{
    assert_histories_equal, build_ohlcv_from_closes, compare_values, read_xlsx_by_position,
    slice_ohlcv, tick_at, EPSILON,
};
use rolling_ta::momentum::{ROC, ROCConfig};
use rolling_ta::prelude::*;

#[test]
fn roc_modes_equivalent() {
    // btc-roc.xlsx: timestamp(0), open(1), high(2), low(3), close(4), volume(5), roc(6)
    let cols = read_xlsx_by_position("resources/data/btc-roc.xlsx", &[4, 6]);
    let closes = &cols[0];
    let expected_roc = &cols[1];
    let data = build_ohlcv_from_closes(closes);

    // 1. Batch calculation - validate ROC against Python reference
    let mut batch = ROC::new(ROCConfig::new(14));
    batch.calc(&data).unwrap();
    let roc_comparisons = compare_values("ROC batch vs reference", batch.history(), expected_roc, EPSILON);
    println!("ROC: {} values compared against Python reference", roc_comparisons);
    assert!(roc_comparisons > 100, "Should have compared many ROC values, got {}", roc_comparisons);

    // 2. Full streaming (tests update() from scratch, including warmup)
    let mut stream = ROC::new(ROCConfig::new(14));
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 3. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = ROC::new(ROCConfig::new(14));
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_histories_equal("ROC batch vs stream", batch.history(), stream.history(), epsilon);
    assert_histories_equal("ROC batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
