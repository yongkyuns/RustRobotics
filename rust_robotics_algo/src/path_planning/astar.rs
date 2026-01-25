//! A* path planning algorithm

use super::grid::Grid;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Result of A* path planning
#[derive(Debug, Clone)]
pub struct AStarResult {
    /// Path in world coordinates (from start to goal)
    pub path: Vec<(f32, f32)>,
    /// Visited cells during search (for visualization)
    pub visited: Vec<(usize, usize)>,
    /// Whether a path was found
    pub success: bool,
    /// Number of iterations (nodes expanded)
    pub iterations: usize,
}

/// A* path planner
pub struct AStarPlanner<'a> {
    grid: &'a Grid,
}

/// Node for A* search
#[derive(Clone)]
struct Node {
    x: usize,
    y: usize,
    g: f32, // Cost from start
    h: f32, // Heuristic to goal
}

impl Node {
    fn f(&self) -> f32 {
        self.g + self.h
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (lower f = higher priority)
        other.f().partial_cmp(&self.f()).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> AStarPlanner<'a> {
    /// Create a new A* planner for the given grid
    pub fn new(grid: &'a Grid) -> Self {
        Self { grid }
    }

    /// Plan a path from start to goal (world coordinates)
    pub fn plan(&self, start: (f32, f32), goal: (f32, f32)) -> AStarResult {
        // Convert to grid coordinates
        let start_grid = match self.grid.world_to_grid(start.0, start.1) {
            Some(pos) => pos,
            None => {
                return AStarResult {
                    path: vec![],
                    visited: vec![],
                    success: false,
                    iterations: 0,
                }
            }
        };

        let goal_grid = match self.grid.world_to_grid(goal.0, goal.1) {
            Some(pos) => pos,
            None => {
                return AStarResult {
                    path: vec![],
                    visited: vec![],
                    success: false,
                    iterations: 0,
                }
            }
        };

        // Check if start or goal is an obstacle
        if self.grid.is_obstacle(start_grid.0, start_grid.1)
            || self.grid.is_obstacle(goal_grid.0, goal_grid.1)
        {
            return AStarResult {
                path: vec![],
                visited: vec![],
                success: false,
                iterations: 0,
            };
        }

        // A* search
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut visited_order = Vec::new();
        let mut iterations = 0usize;
        let mut came_from: std::collections::HashMap<(usize, usize), (usize, usize)> =
            std::collections::HashMap::new();
        let mut g_scores: std::collections::HashMap<(usize, usize), f32> =
            std::collections::HashMap::new();

        let start_node = Node {
            x: start_grid.0,
            y: start_grid.1,
            g: 0.0,
            h: self.heuristic(start_grid, goal_grid),
        };

        g_scores.insert((start_node.x, start_node.y), 0.0);
        open_set.push(start_node);

        // 8-directional movement
        let directions: [(i32, i32); 8] = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),  // Cardinal
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1), // Diagonal
        ];

        while let Some(current) = open_set.pop() {
            iterations += 1;
            let current_pos = (current.x, current.y);

            if closed_set.contains(&current_pos) {
                continue;
            }

            visited_order.push(current_pos);
            closed_set.insert(current_pos);

            // Check if we reached the goal
            if current_pos == goal_grid {
                let path = self.reconstruct_path(&came_from, goal_grid, start_grid);
                return AStarResult {
                    path,
                    visited: visited_order,
                    success: true,
                    iterations,
                };
            }

            // Explore neighbors
            for (dx, dy) in directions.iter() {
                let nx = current.x as i32 + dx;
                let ny = current.y as i32 + dy;

                if nx < 0 || ny < 0 {
                    continue;
                }

                let nx = nx as usize;
                let ny = ny as usize;

                if !self.grid.is_valid(nx, ny) || self.grid.is_obstacle(nx, ny) {
                    continue;
                }

                let neighbor_pos = (nx, ny);

                if closed_set.contains(&neighbor_pos) {
                    continue;
                }

                // Movement cost (1.0 for cardinal, sqrt(2) for diagonal)
                let move_cost = if *dx != 0 && *dy != 0 {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };

                let tentative_g = current.g + move_cost;

                let current_g = g_scores.get(&neighbor_pos).copied().unwrap_or(f32::INFINITY);

                if tentative_g < current_g {
                    came_from.insert(neighbor_pos, current_pos);
                    g_scores.insert(neighbor_pos, tentative_g);

                    let neighbor_node = Node {
                        x: nx,
                        y: ny,
                        g: tentative_g,
                        h: self.heuristic(neighbor_pos, goal_grid),
                    };

                    open_set.push(neighbor_node);
                }
            }
        }

        // No path found
        AStarResult {
            path: vec![],
            visited: visited_order,
            success: false,
            iterations,
        }
    }

    /// Euclidean heuristic
    fn heuristic(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        let dx = (a.0 as f32 - b.0 as f32).abs();
        let dy = (a.1 as f32 - b.1 as f32).abs();
        (dx * dx + dy * dy).sqrt()
    }

    /// Reconstruct path from came_from map
    fn reconstruct_path(
        &self,
        came_from: &std::collections::HashMap<(usize, usize), (usize, usize)>,
        goal: (usize, usize),
        start: (usize, usize),
    ) -> Vec<(f32, f32)> {
        let mut path = Vec::new();
        let mut current = goal;

        while current != start {
            let world_pos = self.grid.grid_to_world(current.0, current.1);
            path.push(world_pos);

            match came_from.get(&current) {
                Some(&parent) => current = parent,
                None => break,
            }
        }

        // Add start position
        let start_world = self.grid.grid_to_world(start.0, start.1);
        path.push(start_world);

        // Reverse to get path from start to goal
        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_path() {
        let grid = Grid::new(10, 10, 1.0);
        let planner = AStarPlanner::new(&grid);
        let result = planner.plan((-4.0, -4.0), (4.0, 4.0));
        assert!(result.success);
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_path_with_obstacles() {
        let mut grid = Grid::new(10, 10, 1.0);
        // Create a wall
        for y in 0..8 {
            grid.set_obstacle(5, y);
        }
        let planner = AStarPlanner::new(&grid);
        let result = planner.plan((-4.0, 0.0), (4.0, 0.0));
        assert!(result.success);
        // Path should go around the wall
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_no_path() {
        let mut grid = Grid::new(10, 10, 1.0);
        // Create a complete barrier
        for y in 0..10 {
            grid.set_obstacle(5, y);
        }
        let planner = AStarPlanner::new(&grid);
        let result = planner.plan((-4.0, 0.0), (4.0, 0.0));
        assert!(!result.success);
    }
}
