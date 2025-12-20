//! Rolling window utilities.

use std::collections::VecDeque;

/// Fixed-capacity rolling window.
///
/// Efficiently maintains a sliding window of values with O(1) push/pop.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    buffer: VecDeque<f64>,
    capacity: usize,
    sum: f64,
}

impl RollingWindow {
    /// Create a new rolling window with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    /// Push a value, removing oldest if at capacity.
    /// Returns the removed value if any.
    #[inline]
    pub fn push(&mut self, value: f64) -> Option<f64> {
        self.sum += value;

        if self.buffer.len() >= self.capacity {
            let removed = self.buffer.pop_front().unwrap();
            self.sum -= removed;
            self.buffer.push_back(value);
            Some(removed)
        } else {
            self.buffer.push_back(value);
            None
        }
    }

    /// Get current sum of all values.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get current mean of all values.
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.buffer.is_empty() {
            f64::NAN
        } else {
            self.sum / self.buffer.len() as f64
        }
    }

    /// Number of values currently in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if window is full (at capacity).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the oldest value.
    #[inline]
    pub fn front(&self) -> Option<f64> {
        self.buffer.front().copied()
    }

    /// Get the newest value.
    #[inline]
    pub fn back(&self) -> Option<f64> {
        self.buffer.back().copied()
    }

    /// Get value at index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<f64> {
        self.buffer.get(index).copied()
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }

    /// Iterate over values from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.buffer.iter().copied()
    }

    /// Get as slice (may not be contiguous).
    pub fn as_slices(&self) -> (&[f64], &[f64]) {
        self.buffer.as_slices()
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.buffer.iter().cloned().collect()
    }
}

/// High-low tracking window for Donchian-style calculations.
#[derive(Debug, Clone)]
pub struct HighLowWindow {
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    capacity: usize,
}

impl HighLowWindow {
    pub fn new(capacity: usize) -> Self {
        Self {
            highs: VecDeque::with_capacity(capacity),
            lows: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push high and low values.
    pub fn push(&mut self, high: f64, low: f64) {
        if self.highs.len() >= self.capacity {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        self.highs.push_back(high);
        self.lows.push_back(low);
    }

    /// Get highest high in the window.
    #[inline]
    pub fn highest(&self) -> f64 {
        self.highs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get lowest low in the window.
    #[inline]
    pub fn lowest(&self) -> f64 {
        self.lows
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get midpoint (highest + lowest) / 2.
    #[inline]
    pub fn midpoint(&self) -> f64 {
        (self.highest() + self.lowest()) / 2.0
    }

    /// Check if window is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.highs.len() >= self.capacity
    }

    /// Number of values in window.
    #[inline]
    pub fn len(&self) -> usize {
        self.highs.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.highs.is_empty()
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.highs.clear();
        self.lows.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_window_basic() {
        let mut window = RollingWindow::new(3);

        assert!(window.push(1.0).is_none());
        assert!(window.push(2.0).is_none());
        assert!(window.push(3.0).is_none());
        assert!(window.is_full());
        assert_eq!(window.sum(), 6.0);
        assert_eq!(window.mean(), 2.0);

        // Push beyond capacity
        let removed = window.push(4.0);
        assert_eq!(removed, Some(1.0));
        assert_eq!(window.sum(), 9.0); // 2 + 3 + 4
        assert_eq!(window.mean(), 3.0);
    }

    #[test]
    fn rolling_window_empty() {
        let window = RollingWindow::new(3);
        assert!(window.mean().is_nan());
        assert!(window.is_empty());
    }

    #[test]
    fn high_low_window() {
        let mut window = HighLowWindow::new(3);

        window.push(10.0, 5.0);
        window.push(12.0, 6.0);
        window.push(8.0, 4.0);

        assert_eq!(window.highest(), 12.0);
        assert_eq!(window.lowest(), 4.0);
        assert_eq!(window.midpoint(), 8.0);

        // Push beyond capacity
        window.push(9.0, 7.0);
        assert_eq!(window.highest(), 12.0); // 12, 8, 9 -> 12
        assert_eq!(window.lowest(), 4.0);   // 6, 4, 7 -> 4
    }
}
