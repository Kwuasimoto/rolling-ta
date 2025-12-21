//! ATR reference tests.
//!
//! Verifies ATR and TR implementations against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.

use crate::common::{
    assert_histories_equal, build_ohlcv_hlc, compare_values, read_xlsx_by_position,
    slice_ohlcv, tick_at, EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volatility::{ATR, ATRConfig, TR};

#[test]
fn atr_modes_equivalent() {
    // btc-atr.xlsx: timestamp(0), high(1), low(2), close(3), tr(4), atr(5)
    let cols = read_xlsx_by_position("resources/data/btc-atr.xlsx", &[1, 2, 3, 4, 5]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_tr = &cols[3];
    let expected_atr = &cols[4];
    let data = build_ohlcv_hlc(highs, lows, closes);

    // 1. Batch calculation - validate TR against Python reference
    let mut tr = TR::default();
    tr.calc(&data).unwrap();
    let tr_comparisons = compare_values("TR batch vs reference", tr.history(), expected_tr, EPSILON);
    println!("TR: {} values compared against Python reference", tr_comparisons);
    assert!(tr_comparisons > 100, "Should have compared many TR values, got {}", tr_comparisons);

    // 2. Batch calculation - validate ATR against Python reference
    let mut batch = ATR::new(ATRConfig::new(14));
    batch.calc(&data).unwrap();
    let atr_comparisons = compare_values("ATR batch vs reference", batch.history(), expected_atr, EPSILON);
    println!("ATR: {} values compared against Python reference", atr_comparisons);
    assert!(atr_comparisons > 100, "Should have compared many ATR values, got {}", atr_comparisons);

    // 3. Full streaming (tests update() from scratch, including warmup)
    let mut stream = ATR::new(ATRConfig::new(14));
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 4. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = ATR::new(ATRConfig::new(14));
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_histories_equal("ATR batch vs stream", batch.history(), stream.history(), epsilon);
    assert_histories_equal("ATR batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
