//! Bollinger Bands reference tests.
//!
//! Verifies BB implementation against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.
//! BB has compound output: upper, middle, lower bands.

use crate::common::{
    assert_bb_histories_equal, build_ohlcv_from_closes, compare_values, read_xlsx_by_position,
    slice_ohlcv, tick_at, EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volatility::{BB, BBConfig};

#[test]
fn bb_modes_equivalent() {
    // btc-bb.xlsx: timestamp(0), close(1), sma(2), upper(3), lower(4)
    let cols = read_xlsx_by_position("resources/data/btc-bb.xlsx", &[1, 2, 3, 4]);
    let closes = &cols[0];
    let expected_middle = &cols[1];
    let expected_upper = &cols[2];
    let expected_lower = &cols[3];
    let data = build_ohlcv_from_closes(closes);

    // 1. Batch calculation (also validates against Python reference)
    let mut batch = BB::new(BBConfig::new(20, 2.0));
    batch.calc(&data).unwrap();

    // Extract component values from BBOutput
    let history = batch.history();
    let rust_upper: Vec<f64> = history.iter().map(|o| o.upper).collect();
    let rust_middle: Vec<f64> = history.iter().map(|o| o.middle).collect();
    let rust_lower: Vec<f64> = history.iter().map(|o| o.lower).collect();

    let upper_cmp = compare_values("BB upper vs reference", &rust_upper, expected_upper, EPSILON);
    let middle_cmp = compare_values("BB middle vs reference", &rust_middle, expected_middle, EPSILON);
    let lower_cmp = compare_values("BB lower vs reference", &rust_lower, expected_lower, EPSILON);

    println!(
        "BB: upper={}, middle={}, lower={} values compared against Python reference",
        upper_cmp, middle_cmp, lower_cmp
    );
    assert!(upper_cmp > 100, "Should have compared many upper values, got {}", upper_cmp);
    assert!(middle_cmp > 100, "Should have compared many middle values, got {}", middle_cmp);
    assert!(lower_cmp > 100, "Should have compared many lower values, got {}", lower_cmp);

    // 2. Full streaming (tests update() from scratch, including warmup)
    let mut stream = BB::new(BBConfig::new(20, 2.0));
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 3. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = BB::new(BBConfig::new(20, 2.0));
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_bb_histories_equal("BB batch vs stream", batch.history(), stream.history(), epsilon);
    assert_bb_histories_equal("BB batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
