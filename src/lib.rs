//! # rolling-ta
//!
//! High-performance technical analysis library with streaming support.
//!
//! ## Features
//!
//! - **Batch calculation**: Process historical data with `calc()`
//! - **Streaming updates**: Process snapshots with `next()` (parallel-safe)
//! - **SharedWindow architecture**: Single window shared across indicators
//! - **Type-safe**: NewType wrappers prevent primitive obsession
//!
//! ## Example
//!
//! ```rust
//! use rolling_ta::prelude::*;
//! use rolling_ta::trend::{SMA, SMAConfig};
//!
//! // Create indicator with period 3
//! let mut sma = SMA::new(SMAConfig::new(3));
//!
//! // Batch calculate over historical data
//! let candles: Vec<Ohlcv> = vec![
//!     Ohlcv::from_close(1.0),
//!     Ohlcv::from_close(2.0),
//!     Ohlcv::from_close(3.0),
//!     Ohlcv::from_close(4.0),
//!     Ohlcv::from_close(5.0),
//! ];
//! sma.calc(&candles).unwrap();
//!
//! // Stream from snapshot (parallel-safe with SharedWindow)
//! let snapshot = vec![
//!     Ohlcv::from_close(4.0),
//!     Ohlcv::from_close(5.0),
//!     Ohlcv::from_close(6.0),
//! ];
//! if let Some(value) = sma.next(&snapshot) {
//!     println!("SMA: {}", value);
//! }
//! ```

pub mod ta;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::ta::{
        error::{TAError, TAResult},
        state::IndicatorState,
        temporal::{CandleBuilder, Tick, Timeframe},
        types::{Ohlcv, Price, Timestamp, Volume},
        Indicator, HistoricalIndicator,
    };
}

// Re-export top-level modules for ergonomic access
pub use ta::trend;
// TODO: Uncomment after migration to new Indicator trait
// pub use ta::momentum;
// pub use ta::volatility;
// pub use ta::volume;
