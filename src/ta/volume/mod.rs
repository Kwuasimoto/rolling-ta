//! Volume indicators.
//!
//! Indicators that incorporate trading volume into analysis.

mod obv;
mod vwap;

pub use obv::OBV;
pub use vwap::VWAP;

// Re-export configs for convenience
pub use crate::ta::config::{OBVConfig, VWAPConfig};
