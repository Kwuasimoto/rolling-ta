//! CMF reference tests.
//!
//! Verifies CMF (Chaikin Money Flow) implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{assert_histories_equal, build_candles_hlcv, read_xlsx_by_position};
use rolling_ta::prelude::*;
use rolling_ta::volume::{CMF, CMFConfig};

// NOTE: cmf_batch_vs_reference test is disabled.
// The btc-cmf.xlsx file only contains OHLCV data (columns 0-4).
// CMF reference values (columns 5-7) are all NaN - never populated.
// There's also no Python CMF implementation in the codebase.
// To enable this test, a Python CMF implementation needs to be created
// and the xlsx file regenerated with calculated CMF values.

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

// NOTE: CMF does not support fixed-window testing.
//
// CMF's next() implementation tracks `last_len` to detect new candles.
// When called with same-length fixed windows (e.g., 20 candles each time),
// it treats subsequent calls as same-candle updates rather than independent
// calculations. This is by design for streaming with growing snapshots.
//
// Use streaming mode (next with growing snapshots) instead of fixed windows.
