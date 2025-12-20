//! # rolling-ta
//!
//! High-performance technical analysis library with streaming support.
//!
//! ## Features
//!
//! - **Batch calculation**: Process historical data with `calc()`
//! - **Streaming updates**: Update indicators tick-by-tick with `update()`
//! - **Zero-copy design**: Minimal allocations in hot paths
//! - **Type-safe**: NewType wrappers prevent primitive obsession
//!
//! ## Example
//!
//! ```rust
//! use rolling_ta::prelude::*;
//! use rolling_ta::trend::SMA;
//!
//! // Create indicator with period 3
//! use rolling_ta::trend::SMAConfig;
//! let mut sma = SMA::new(SMAConfig::new(3));
//!
//! // Batch calculate over historical data
//! let data = OhlcvSeries::from_closes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! sma.calc(&data).unwrap();
//!
//! // Stream new ticks
//! let tick = Ohlcv::from_close(6.0);
//! if let Some(value) = sma.update(&tick).unwrap() {
//!     println!("SMA: {}", value);
//! }
//! ```

pub mod ta;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::ta::{
        error::{TAError, TAResult},
        state::IndicatorState,
        types::{Ohlcv, OhlcvSeries, Price, Timestamp, Volume},
        Indicator, HistoricalIndicator,
    };
}

// Re-export top-level modules for ergonomic access
pub use ta::trend;
pub use ta::momentum;
pub use ta::volatility;
pub use ta::volume;
