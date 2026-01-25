//! Error tracking for position estimation accuracy.

use std::collections::VecDeque;

use super::history::DEFAULT_HISTORY_LEN;

/// Tracks position errors over time for estimation and dead reckoning.
///
/// Provides Euclidean distance error calculation and history management.
#[derive(Debug, Clone)]
pub struct ErrorTracker {
    /// History of estimation errors
    est_errors: VecDeque<f32>,
    /// History of dead reckoning errors
    dr_errors: VecDeque<f32>,
    /// Maximum history length
    max_len: usize,
}

impl ErrorTracker {
    /// Create a new error tracker with the specified maximum history length.
    pub fn new(max_len: usize) -> Self {
        Self {
            est_errors: VecDeque::with_capacity(max_len),
            dr_errors: VecDeque::with_capacity(max_len),
            max_len,
        }
    }

    /// Create a new error tracker with default maximum length.
    pub fn with_default_len() -> Self {
        Self::new(DEFAULT_HISTORY_LEN)
    }

    /// Calculate 2D Euclidean position error between two points.
    pub fn position_error_2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
        ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
    }

    /// Track errors for this timestep.
    pub fn track(&mut self, est_err: f32, dr_err: f32) {
        self.est_errors.push_back(est_err);
        self.dr_errors.push_back(dr_err);

        // Maintain circular buffer behavior
        if self.est_errors.len() > self.max_len {
            self.est_errors.pop_front();
            self.dr_errors.pop_front();
        }
    }

    /// Track errors by computing position differences.
    ///
    /// Convenience method that calculates errors from positions.
    pub fn track_positions(
        &mut self,
        true_x: f32,
        true_y: f32,
        est_x: f32,
        est_y: f32,
        dr_x: f32,
        dr_y: f32,
    ) {
        let est_err = Self::position_error_2d(true_x, true_y, est_x, est_y);
        let dr_err = Self::position_error_2d(true_x, true_y, dr_x, dr_y);
        self.track(est_err, dr_err);
    }

    /// Get the estimation error history.
    pub fn get_est_errors(&self) -> impl Iterator<Item = &f32> {
        self.est_errors.iter()
    }

    /// Get the dead reckoning error history.
    pub fn get_dr_errors(&self) -> impl Iterator<Item = &f32> {
        self.dr_errors.iter()
    }

    /// Get the current estimation error (last value).
    pub fn current_est_error(&self) -> Option<f32> {
        self.est_errors.back().copied()
    }

    /// Get the current dead reckoning error (last value).
    pub fn current_dr_error(&self) -> Option<f32> {
        self.dr_errors.back().copied()
    }

    /// Get the current history length.
    pub fn len(&self) -> usize {
        self.est_errors.len()
    }

    /// Check if error history is empty.
    pub fn is_empty(&self) -> bool {
        self.est_errors.is_empty()
    }

    /// Clear all error history.
    pub fn clear(&mut self) {
        self.est_errors.clear();
        self.dr_errors.clear();
    }

    /// Initialize with zero errors.
    pub fn init_with_zero(&mut self) {
        self.clear();
        self.est_errors.push_back(0.0);
        self.dr_errors.push_back(0.0);
    }
}

impl Default for ErrorTracker {
    fn default() -> Self {
        let mut tracker = Self::with_default_len();
        tracker.init_with_zero();
        tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_error_2d() {
        let err = ErrorTracker::position_error_2d(0.0, 0.0, 3.0, 4.0);
        assert!((err - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_error_tracking() {
        let mut tracker = ErrorTracker::new(5);

        tracker.track(1.0, 2.0);
        tracker.track(1.5, 2.5);
        tracker.track(2.0, 3.0);

        assert_eq!(tracker.len(), 3);
        assert_eq!(tracker.current_est_error(), Some(2.0));
        assert_eq!(tracker.current_dr_error(), Some(3.0));
    }

    #[test]
    fn test_track_positions() {
        let mut tracker = ErrorTracker::new(5);

        // True at origin, est at (3,4), dr at (0,5)
        tracker.track_positions(0.0, 0.0, 3.0, 4.0, 0.0, 5.0);

        assert!((tracker.current_est_error().unwrap() - 5.0).abs() < 1e-6);
        assert!((tracker.current_dr_error().unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_circular_buffer() {
        let mut tracker = ErrorTracker::new(3);

        for i in 1..=5 {
            tracker.track(i as f32, i as f32 * 2.0);
        }

        assert_eq!(tracker.len(), 3);

        let est_errors: Vec<_> = tracker.get_est_errors().copied().collect();
        assert_eq!(est_errors, vec![3.0, 4.0, 5.0]);
    }
}
