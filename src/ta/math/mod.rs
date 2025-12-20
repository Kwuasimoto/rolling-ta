//! Mathematical helper functions for indicator calculations.
//!
//! These are pure functions with no state - they implement the core algorithms.

pub mod ema;
pub mod rolling;
pub mod stats;

// Re-export commonly used functions
pub use ema::*;
pub use rolling::*;
pub use stats::*;
