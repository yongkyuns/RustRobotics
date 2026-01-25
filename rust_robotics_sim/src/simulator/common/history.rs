//! History management with circular buffers for trajectory visualization.

use std::collections::VecDeque;

/// Default maximum history length for trajectory visualization
pub const DEFAULT_HISTORY_LEN: usize = 1000;

/// Manages history of states (true, estimated, dead reckoning) with circular buffers.
///
/// This eliminates duplicated history management code between localization and SLAM demos.
#[derive(Debug, Clone)]
pub struct HistoryManager<T: Clone> {
    /// History of true states
    true_history: VecDeque<T>,
    /// History of estimated states
    estimated_history: VecDeque<T>,
    /// History of dead reckoning states
    dr_history: VecDeque<T>,
    /// Maximum history length
    max_len: usize,
    /// Total step count (for time-based plotting)
    step_count: usize,
}

impl<T: Clone> HistoryManager<T> {
    /// Create a new history manager with the specified maximum length.
    pub fn new(max_len: usize) -> Self {
        Self {
            true_history: VecDeque::with_capacity(max_len),
            estimated_history: VecDeque::with_capacity(max_len),
            dr_history: VecDeque::with_capacity(max_len),
            max_len,
            step_count: 0,
        }
    }

    /// Create a new history manager with default maximum length.
    pub fn with_default_len() -> Self {
        Self::new(DEFAULT_HISTORY_LEN)
    }

    /// Push new states to history and increment step count.
    pub fn push(&mut self, true_state: T, est_state: T, dr_state: T) {
        self.true_history.push_back(true_state);
        self.estimated_history.push_back(est_state);
        self.dr_history.push_back(dr_state);
        self.step_count += 1;

        // Maintain circular buffer behavior
        if self.true_history.len() > self.max_len {
            self.true_history.pop_front();
            self.estimated_history.pop_front();
            self.dr_history.pop_front();
        }
    }

    /// Get the true state history as a slice.
    pub fn get_true(&self) -> impl Iterator<Item = &T> {
        self.true_history.iter()
    }

    /// Get the estimated state history as a slice.
    pub fn get_estimated(&self) -> impl Iterator<Item = &T> {
        self.estimated_history.iter()
    }

    /// Get the dead reckoning state history as a slice.
    pub fn get_dr(&self) -> impl Iterator<Item = &T> {
        self.dr_history.iter()
    }

    /// Get the total step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the current history length.
    pub fn len(&self) -> usize {
        self.true_history.len()
    }

    /// Check if history is empty.
    pub fn is_empty(&self) -> bool {
        self.true_history.is_empty()
    }

    /// Get the maximum history length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get the start step for time-based plotting.
    /// This is the step count at the beginning of the current history window.
    pub fn start_step(&self) -> usize {
        if self.step_count > self.max_len {
            self.step_count - self.max_len
        } else {
            0
        }
    }

    /// Clear all history and reset step count.
    pub fn clear(&mut self) {
        self.true_history.clear();
        self.estimated_history.clear();
        self.dr_history.clear();
        self.step_count = 0;
    }

    /// Clear history but keep step count (for partial resets).
    pub fn clear_history(&mut self) {
        self.true_history.clear();
        self.estimated_history.clear();
        self.dr_history.clear();
    }

    /// Initialize with a single state for all three histories.
    pub fn init_with(&mut self, initial: T) {
        self.clear();
        self.true_history.push_back(initial.clone());
        self.estimated_history.push_back(initial.clone());
        self.dr_history.push_back(initial);
    }
}

impl<T: Clone + Default> Default for HistoryManager<T> {
    fn default() -> Self {
        let mut manager = Self::with_default_len();
        manager.init_with(T::default());
        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_push_and_get() {
        let mut history: HistoryManager<f32> = HistoryManager::new(5);

        history.push(1.0, 1.1, 1.2);
        history.push(2.0, 2.1, 2.2);
        history.push(3.0, 3.1, 3.2);

        assert_eq!(history.len(), 3);
        assert_eq!(history.step_count(), 3);

        let true_vals: Vec<_> = history.get_true().copied().collect();
        assert_eq!(true_vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_history_circular_buffer() {
        let mut history: HistoryManager<i32> = HistoryManager::new(3);

        for i in 1..=5 {
            history.push(i, i * 10, i * 100);
        }

        assert_eq!(history.len(), 3);
        assert_eq!(history.step_count(), 5);

        let true_vals: Vec<_> = history.get_true().copied().collect();
        assert_eq!(true_vals, vec![3, 4, 5]);
    }

    #[test]
    fn test_history_start_step() {
        let mut history: HistoryManager<i32> = HistoryManager::new(3);

        // Before exceeding max_len
        history.push(1, 10, 100);
        history.push(2, 20, 200);
        assert_eq!(history.start_step(), 0);

        // After exceeding max_len
        history.push(3, 30, 300);
        history.push(4, 40, 400);
        history.push(5, 50, 500);
        assert_eq!(history.start_step(), 2);
    }

    #[test]
    fn test_history_clear() {
        let mut history: HistoryManager<f32> = HistoryManager::new(5);

        history.push(1.0, 1.1, 1.2);
        history.push(2.0, 2.1, 2.2);

        history.clear();

        assert!(history.is_empty());
        assert_eq!(history.step_count(), 0);
    }
}
