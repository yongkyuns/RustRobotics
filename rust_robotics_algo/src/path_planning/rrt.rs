//! RRT (Rapidly-exploring Random Tree) path planning algorithm
//!
//! Works in continuous space with circular obstacles.

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A node in the RRT tree
#[derive(Debug, Clone)]
pub struct RrtNode {
    /// X position
    pub x: f32,
    /// Y position
    pub y: f32,
    /// Index of parent node in the tree (None for root)
    pub parent: Option<usize>,
}

impl RrtNode {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y, parent: None }
    }

    fn with_parent(x: f32, y: f32, parent: usize) -> Self {
        Self {
            x,
            y,
            parent: Some(parent),
        }
    }

    fn distance_to(&self, x: f32, y: f32) -> f32 {
        let dx = self.x - x;
        let dy = self.y - y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A circular obstacle
#[derive(Debug, Clone, Copy)]
pub struct CircleObstacle {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

impl CircleObstacle {
    pub fn new(x: f32, y: f32, radius: f32) -> Self {
        Self { x, y, radius }
    }

    fn contains(&self, px: f32, py: f32) -> bool {
        let dx = px - self.x;
        let dy = py - self.y;
        (dx * dx + dy * dy).sqrt() <= self.radius
    }
}

/// Result of RRT path planning
#[derive(Debug, Clone)]
pub struct RrtResult {
    /// Path in world coordinates (from start to goal)
    pub path: Vec<(f32, f32)>,
    /// All nodes in the tree (for visualization)
    pub tree: Vec<RrtNode>,
    /// Whether a path was found
    pub success: bool,
    /// Number of iterations (main loop iterations)
    pub iterations: usize,
}

/// Configuration for RRT planner
#[derive(Debug, Clone)]
pub struct RrtConfig {
    /// Step size for tree expansion
    pub expand_distance: f32,
    /// Probability of sampling the goal (0.0 to 1.0)
    pub goal_sample_rate: f32,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Distance threshold to consider goal reached
    pub goal_threshold: f32,
    /// Random seed (None for random)
    pub seed: Option<u64>,
}

impl Default for RrtConfig {
    fn default() -> Self {
        Self {
            expand_distance: 1.0,
            goal_sample_rate: 0.1,
            max_iterations: 500,
            goal_threshold: 1.0,
            seed: None,
        }
    }
}

/// RRT path planner
pub struct RrtPlanner {
    /// Minimum x bound of search space
    pub min_x: f32,
    /// Maximum x bound of search space
    pub max_x: f32,
    /// Minimum y bound of search space
    pub min_y: f32,
    /// Maximum y bound of search space
    pub max_y: f32,
    /// List of circular obstacles
    pub obstacles: Vec<CircleObstacle>,
    /// Configuration
    pub config: RrtConfig,
}

impl RrtPlanner {
    /// Create a new RRT planner
    pub fn new(min_x: f32, max_x: f32, min_y: f32, max_y: f32, config: RrtConfig) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            obstacles: Vec::new(),
            config,
        }
    }

    /// Add a circular obstacle
    pub fn add_obstacle(&mut self, obstacle: CircleObstacle) {
        self.obstacles.push(obstacle);
    }

    /// Clear all obstacles
    pub fn clear_obstacles(&mut self) {
        self.obstacles.clear();
    }

    /// Plan a path from start to goal
    pub fn plan(&self, start: (f32, f32), goal: (f32, f32)) -> RrtResult {
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let x_dist = Uniform::new(self.min_x, self.max_x);
        let y_dist = Uniform::new(self.min_y, self.max_y);
        let goal_dist = Uniform::new(0.0f32, 1.0);

        // Initialize tree with start node
        let mut tree = vec![RrtNode::new(start.0, start.1)];
        let mut iterations = 0usize;

        for _ in 0..self.config.max_iterations {
            iterations += 1;

            // Sample random point (with goal bias)
            let (sample_x, sample_y) = if goal_dist.sample(&mut rng) < self.config.goal_sample_rate
            {
                (goal.0, goal.1)
            } else {
                (x_dist.sample(&mut rng), y_dist.sample(&mut rng))
            };

            // Find nearest node in tree
            let nearest_idx = self.find_nearest(&tree, sample_x, sample_y);
            let nearest = &tree[nearest_idx];

            // Steer toward sample
            let (new_x, new_y) = self.steer(nearest.x, nearest.y, sample_x, sample_y);

            // Check collision
            if self.is_collision_free(nearest.x, nearest.y, new_x, new_y) {
                let new_node = RrtNode::with_parent(new_x, new_y, nearest_idx);
                tree.push(new_node);

                // Check if goal reached
                let dist_to_goal = ((new_x - goal.0).powi(2) + (new_y - goal.1).powi(2)).sqrt();
                if dist_to_goal <= self.config.goal_threshold {
                    // Add goal node
                    let goal_node = RrtNode::with_parent(goal.0, goal.1, tree.len() - 1);
                    tree.push(goal_node);

                    // Extract path
                    let path = self.extract_path(&tree);
                    return RrtResult {
                        path,
                        tree,
                        success: true,
                        iterations,
                    };
                }
            }
        }

        // No path found within max iterations
        RrtResult {
            path: vec![],
            tree,
            success: false,
            iterations,
        }
    }

    /// Find the nearest node in the tree to a point
    fn find_nearest(&self, tree: &[RrtNode], x: f32, y: f32) -> usize {
        let mut min_dist = f32::INFINITY;
        let mut nearest_idx = 0;

        for (i, node) in tree.iter().enumerate() {
            let dist = node.distance_to(x, y);
            if dist < min_dist {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        nearest_idx
    }

    /// Steer from one point toward another, limited by expand_distance
    fn steer(&self, from_x: f32, from_y: f32, to_x: f32, to_y: f32) -> (f32, f32) {
        let dx = to_x - from_x;
        let dy = to_y - from_y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= self.config.expand_distance {
            (to_x, to_y)
        } else {
            let theta = dy.atan2(dx);
            let new_x = from_x + self.config.expand_distance * theta.cos();
            let new_y = from_y + self.config.expand_distance * theta.sin();
            (new_x, new_y)
        }
    }

    /// Check if path between two points is collision-free
    fn is_collision_free(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
        // Check endpoints
        if self.point_in_obstacle(x1, y1) || self.point_in_obstacle(x2, y2) {
            return false;
        }

        // Check along the path
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();
        let steps = (dist / (self.config.expand_distance * 0.1)).ceil() as usize;

        for i in 1..steps {
            let t = i as f32 / steps as f32;
            let x = x1 + t * dx;
            let y = y1 + t * dy;
            if self.point_in_obstacle(x, y) {
                return false;
            }
        }

        true
    }

    /// Check if a point is inside any obstacle
    fn point_in_obstacle(&self, x: f32, y: f32) -> bool {
        for obs in &self.obstacles {
            if obs.contains(x, y) {
                return true;
            }
        }
        false
    }

    /// Extract path from tree (backtrack from goal to start)
    fn extract_path(&self, tree: &[RrtNode]) -> Vec<(f32, f32)> {
        let mut path = Vec::new();
        let mut current_idx = tree.len() - 1;

        loop {
            let node = &tree[current_idx];
            path.push((node.x, node.y));

            match node.parent {
                Some(parent_idx) => current_idx = parent_idx,
                None => break,
            }
        }

        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_path() {
        let config = RrtConfig {
            seed: Some(42),
            max_iterations: 1000,
            ..Default::default()
        };
        let planner = RrtPlanner::new(-10.0, 10.0, -10.0, 10.0, config);
        let result = planner.plan((-5.0, -5.0), (5.0, 5.0));
        assert!(result.success);
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_path_with_obstacles() {
        let config = RrtConfig {
            seed: Some(42),
            max_iterations: 2000,
            ..Default::default()
        };
        let mut planner = RrtPlanner::new(-10.0, 10.0, -10.0, 10.0, config);
        // Add obstacle in the middle
        planner.add_obstacle(CircleObstacle::new(0.0, 0.0, 3.0));
        let result = planner.plan((-5.0, 0.0), (5.0, 0.0));
        assert!(result.success);
        // Path should go around the obstacle
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_tree_growth() {
        let config = RrtConfig {
            seed: Some(42),
            max_iterations: 100,
            ..Default::default()
        };
        let planner = RrtPlanner::new(-10.0, 10.0, -10.0, 10.0, config);
        let result = planner.plan((-5.0, -5.0), (5.0, 5.0));
        // Tree should have grown
        assert!(result.tree.len() > 1);
    }
}
