//! EMA reference tests.
//!
//! Verifies EMA implementation against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.

use crate::common::{
    assert_histories_equal, build_ohlcv_from_closes, compare_values, read_xlsx_by_position,
    slice_ohlcv, tick_at, EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{EMA, EMAConfig};

#[test]
fn ema_modes_equivalent() {
    // btc-ema.xlsx: timestamp(0), close(1), ema(2)
    let cols = read_xlsx_by_position("resources/data/btc-ema.xlsx", &[1, 2]);
    let closes = &cols[0];
    let expected = &cols[1];
    let data = build_ohlcv_from_closes(closes);

    // 1. Batch calculation (also validates against Python reference)
    let mut batch = EMA::new(EMAConfig::new(14));
    batch.calc(&data).unwrap();

    let comparisons = compare_values("EMA batch vs reference", batch.history(), expected, EPSILON);
    println!(
        "EMA: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );

    // 2. Full streaming (tests update() from scratch, including warmup)
    let mut stream = EMA::new(EMAConfig::new(14));
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 3. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = EMA::new(EMAConfig::new(14));
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_histories_equal("EMA batch vs stream", batch.history(), stream.history(), epsilon);
    assert_histories_equal("EMA batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
