//! Error types for technical analysis operations.

use thiserror::Error;

/// Errors that can occur during indicator calculations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TAError {
    /// Indicator has not been initialized with `calc()`.
    #[error("indicator not initialized: call calc() first")]
    NotInitialized,

    /// Insufficient data for the requested calculation.
    #[error("insufficient data: need {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid configuration parameter.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Period must be greater than zero.
    #[error("period must be > 0, got {0}")]
    InvalidPeriod(usize),

    /// Division by zero would occur.
    #[error("division by zero in {context}")]
    DivisionByZero { context: &'static str },

    /// Data contains NaN or infinity.
    #[error("invalid data: {0}")]
    InvalidData(String),

    /// Index out of bounds.
    #[error("index {index} out of bounds (len: {len})")]
    IndexOutOfBounds { index: isize, len: usize },

    /// Invalid indicator ID.
    #[error("invalid indicator ID: indicator not found")]
    InvalidId,
}

/// Result type alias for TA operations.
pub type TAResult<T> = Result<T, TAError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = TAError::InsufficientData {
            required: 14,
            actual: 5,
        };
        assert_eq!(
            err.to_string(),
            "insufficient data: need 14 points, got 5"
        );
    }

    #[test]
    fn error_equality() {
        let err1 = TAError::InvalidPeriod(0);
        let err2 = TAError::InvalidPeriod(0);
        assert_eq!(err1, err2);
    }
}
