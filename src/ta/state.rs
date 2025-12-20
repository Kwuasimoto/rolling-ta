//! Indicator state management.

/// Indicator calculation state.
///
/// Tracks whether an indicator has been initialized and is ready to produce output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndicatorState {
    /// Indicator has not been initialized. Call `calc()` or start feeding ticks.
    #[default]
    Uninitialized,

    /// Indicator is warming up. Has processed `count` ticks but needs more.
    Warming { count: usize },

    /// Indicator is ready to produce valid output.
    Ready,
}

impl IndicatorState {
    /// Check if indicator is ready to produce output.
    #[inline]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if indicator is still warming up.
    #[inline]
    pub fn is_warming(&self) -> bool {
        matches!(self, Self::Warming { .. })
    }

    /// Check if indicator is uninitialized.
    #[inline]
    pub fn is_uninitialized(&self) -> bool {
        matches!(self, Self::Uninitialized)
    }

    /// Get warmup count if warming, None otherwise.
    #[inline]
    pub fn warmup_count(&self) -> Option<usize> {
        match self {
            Self::Warming { count } => Some(*count),
            _ => None,
        }
    }

    /// Transition to warming state with incremented count.
    /// Returns new state (Warming or Ready if threshold met).
    #[inline]
    pub fn increment(self, threshold: usize) -> Self {
        match self {
            Self::Uninitialized => {
                if threshold <= 1 {
                    Self::Ready
                } else {
                    Self::Warming { count: 1 }
                }
            }
            Self::Warming { count } => {
                let new_count = count + 1;
                if new_count >= threshold {
                    Self::Ready
                } else {
                    Self::Warming { count: new_count }
                }
            }
            Self::Ready => Self::Ready,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_uninitialized() {
        let state = IndicatorState::default();
        assert!(state.is_uninitialized());
        assert!(!state.is_ready());
    }

    #[test]
    fn increment_transitions() {
        let state = IndicatorState::Uninitialized;

        // First tick with threshold 3
        let state = state.increment(3);
        assert!(state.is_warming());
        assert_eq!(state.warmup_count(), Some(1));

        // Second tick
        let state = state.increment(3);
        assert!(state.is_warming());
        assert_eq!(state.warmup_count(), Some(2));

        // Third tick - should be ready
        let state = state.increment(3);
        assert!(state.is_ready());
    }

    #[test]
    fn ready_stays_ready() {
        let state = IndicatorState::Ready;
        let state = state.increment(100);
        assert!(state.is_ready());
    }

    #[test]
    fn threshold_one_immediate_ready() {
        let state = IndicatorState::Uninitialized;
        let state = state.increment(1);
        assert!(state.is_ready());
    }
}
