use std::time::Instant;

/// A type-safe wrapper around `std::time::Instant` to measure execution duration.
///
/// This adheres to the NewType Pattern to enforce
/// domain-specific type safety over raw primitives.
/// It avoids primitive obsession.
pub struct ExecutionTimer {
    start_time: Instant,
}

impl ExecutionTimer {
    /// Constructs a new timer and immediately records the starting time.
    /// This is the "start" function.
    pub fn start() -> Self {
        ExecutionTimer {
            start_time: Instant::now(),
        }
    }

    /// Stops the timer and returns the elapsed duration.
    ///
    /// This method is focused solely on calculating and returning the duration,
    /// adhering to the Single Responsibility Principle (SRP).
    pub fn stop(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}