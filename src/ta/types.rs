//! Domain types for technical analysis.
//!
//! NewType wrappers prevent primitive obsession and provide type safety.

use std::ops::{Add, Div, Mul, Sub};

// ============================================================
// NewType Wrappers
// ============================================================

/// Price value wrapper.
///
/// Prevents confusion between price and volume values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Price(pub f64);

impl Price {
    pub const ZERO: Price = Price(0.0);
    pub const NAN: Price = Price(f64::NAN);

    #[inline]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.0.is_nan()
    }
}

impl From<f64> for Price {
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<Price> for f64 {
    fn from(p: Price) -> Self {
        p.0
    }
}

/// Volume value wrapper.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Volume(pub f64);

impl Volume {
    pub const ZERO: Volume = Volume(0.0);

    #[inline]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl From<f64> for Volume {
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<Volume> for f64 {
    fn from(v: Volume) -> Self {
        v.0
    }
}

/// Unix timestamp in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Timestamp(pub i64);

impl Timestamp {
    #[inline]
    pub fn new(value: i64) -> Self {
        Self(value)
    }

    #[inline]
    pub fn value(self) -> i64 {
        self.0
    }
}

impl From<i64> for Timestamp {
    fn from(v: i64) -> Self {
        Self(v)
    }
}

// ============================================================
// OHLCV Types
// ============================================================

/// Single OHLCV candle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ohlcv {
    pub timestamp: Timestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Volume,
}

impl Ohlcv {
    /// Create a new OHLCV candle.
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp: Timestamp(timestamp),
            open: Price(open),
            high: Price(high),
            low: Price(low),
            close: Price(close),
            volume: Volume(volume),
        }
    }

    /// Create from just a close price (for simple testing).
    #[inline]
    pub fn from_close(close: f64) -> Self {
        Self {
            close: Price(close),
            high: Price(close),
            low: Price(close),
            open: Price(close),
            ..Default::default()
        }
    }

    /// Typical price: (high + low + close) / 3
    #[inline]
    pub fn typical_price(&self) -> f64 {
        (self.high.0 + self.low.0 + self.close.0) / 3.0
    }

    /// True range for this candle (requires previous close).
    #[inline]
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high.0 - self.low.0;
        let hc = (self.high.0 - prev_close).abs();
        let lc = (self.low.0 - prev_close).abs();
        hl.max(hc).max(lc)
    }
}

/// Historical OHLCV data series.
///
/// Stored as separate vectors for cache-friendly access in batch calculations.
#[derive(Debug, Clone, Default)]
pub struct OhlcvSeries {
    pub timestamps: Vec<i64>,
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
}

impl OhlcvSeries {
    /// Create empty series with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            opens: Vec::with_capacity(capacity),
            highs: Vec::with_capacity(capacity),
            lows: Vec::with_capacity(capacity),
            closes: Vec::with_capacity(capacity),
            volumes: Vec::with_capacity(capacity),
        }
    }

    /// Create from just close prices (for simple testing).
    pub fn from_closes(closes: &[f64]) -> Self {
        Self {
            timestamps: (0..closes.len() as i64).collect(),
            opens: closes.to_vec(),
            highs: closes.to_vec(),
            lows: closes.to_vec(),
            closes: closes.to_vec(),
            volumes: vec![0.0; closes.len()],
        }
    }

    /// Create from OHLCV tuples.
    pub fn from_tuples(data: &[(i64, f64, f64, f64, f64, f64)]) -> Self {
        let mut series = Self::with_capacity(data.len());
        for &(ts, o, h, l, c, v) in data {
            series.timestamps.push(ts);
            series.opens.push(o);
            series.highs.push(h);
            series.lows.push(l);
            series.closes.push(c);
            series.volumes.push(v);
        }
        series
    }

    /// Number of candles in the series.
    #[inline]
    pub fn len(&self) -> usize {
        self.closes.len()
    }

    /// Check if series is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.closes.is_empty()
    }

    /// Push a new candle.
    pub fn push(&mut self, candle: &Ohlcv) {
        self.timestamps.push(candle.timestamp.0);
        self.opens.push(candle.open.0);
        self.highs.push(candle.high.0);
        self.lows.push(candle.low.0);
        self.closes.push(candle.close.0);
        self.volumes.push(candle.volume.0);
    }

    /// Get candle at index.
    pub fn get(&self, index: usize) -> Option<Ohlcv> {
        if index >= self.len() {
            return None;
        }
        Some(Ohlcv {
            timestamp: Timestamp(self.timestamps[index]),
            open: Price(self.opens[index]),
            high: Price(self.highs[index]),
            low: Price(self.lows[index]),
            close: Price(self.closes[index]),
            volume: Volume(self.volumes[index]),
        })
    }
}

// ============================================================
// Arithmetic Implementations for Price
// ============================================================

impl Add for Price {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Price(self.0 + rhs.0)
    }
}

impl Sub for Price {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Price(self.0 - rhs.0)
    }
}

impl Mul<f64> for Price {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Price(self.0 * rhs)
    }
}

impl Div<f64> for Price {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        Price(self.0 / rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ohlcv_from_close() {
        let candle = Ohlcv::from_close(100.0);
        assert_eq!(candle.close.0, 100.0);
        assert_eq!(candle.high.0, 100.0);
        assert_eq!(candle.low.0, 100.0);
    }

    #[test]
    fn typical_price_calculation() {
        let candle = Ohlcv::new(0, 100.0, 110.0, 90.0, 105.0, 1000.0);
        let tp = candle.typical_price();
        // (110 + 90 + 105) / 3 = 101.666...
        assert!((tp - 101.666666).abs() < 0.001);
    }

    #[test]
    fn true_range_calculation() {
        let candle = Ohlcv::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);
        let tr = candle.true_range(100.0);
        // max(110-95, |110-100|, |95-100|) = max(15, 10, 5) = 15
        assert_eq!(tr, 15.0);
    }

    #[test]
    fn series_from_closes() {
        let series = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0]);
        assert_eq!(series.len(), 3);
        assert_eq!(series.closes, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn price_arithmetic() {
        let p1 = Price(100.0);
        let p2 = Price(50.0);
        assert_eq!((p1 + p2).0, 150.0);
        assert_eq!((p1 - p2).0, 50.0);
        assert_eq!((p1 * 2.0).0, 200.0);
        assert_eq!((p1 / 2.0).0, 50.0);
    }
}
