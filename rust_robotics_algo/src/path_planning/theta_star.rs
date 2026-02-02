//! Theta* path planning algorithm
//!
//! Theta* is an any-angle path planning algorithm that extends A* by allowing
//! the parent of a node to be any other node in the grid, as long as there is
//! a line of sight between them. This produces smoother, shorter paths than A*.

use super::grid::Grid;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Result of Theta* path planning
#[derive(Debug, Clone)]
pub struct ThetaStarResult {
    /// Path in world coordinates (from start to goal)
    pub path: Vec<(f32, f32)>,
    /// Visited cells during search (for visualization)
    pub visited: Vec<(usize, usize)>,
    /// Whether a path was found
    pub success: bool,
    /// Number of iterations (nodes expanded)
    pub iterations: usize,
}

/// Theta* path planner
pub struct ThetaStarPlanner<'a> {
    grid: &'a Grid,
}

/// Node for Theta* search
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

impl<'a> ThetaStarPlanner<'a> {
    /// Create a new Theta* planner for the given grid
    pub fn new(grid: &'a Grid) -> Self {
        Self { grid }
    }

    /// Plan a path from start to goal (world coordinates)
    pub fn plan(&self, start: (f32, f32), goal: (f32, f32)) -> ThetaStarResult {
        // Convert to grid coordinates
        let start_grid = match self.grid.world_to_grid(start.0, start.1) {
            Some(pos) => pos,
            None => return self.fail(),
        };

        let goal_grid = match self.grid.world_to_grid(goal.0, goal.1) {
            Some(pos) => pos,
            None => return self.fail(),
        };

        // Check if start or goal is an obstacle
        if self.grid.is_obstacle(start_grid.0, start_grid.1)
            || self.grid.is_obstacle(goal_grid.0, goal_grid.1)
        {
            return self.fail();
        }

        // Search structures
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut visited_order = Vec::new();
        let mut iterations = 0usize;
        
        // Maps current node -> parent node
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        
        // Cost from start to node
        let mut g_scores: HashMap<(usize, usize), f32> = HashMap::new();

        // Initialize start node
        let start_node = Node {
            x: start_grid.0,
            y: start_grid.1,
            g: 0.0,
            h: self.heuristic(start_grid, goal_grid),
        };

        g_scores.insert((start_node.x, start_node.y), 0.0);
        // Start node is its own parent (sentinel)
        came_from.insert((start_node.x, start_node.y), (start_node.x, start_node.y)); 
        open_set.push(start_node);

        // 8-directional movement
        let directions: [(i32, i32); 8] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  // Cardinal
            (1, 1), (1, -1), (-1, 1), (-1, -1), // Diagonal
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
                return ThetaStarResult {
                    path,
                    visited: visited_order,
                    success: true,
                    iterations,
                };
            }

            // Get parent of current node
            let parent_pos = *came_from.get(&current_pos).unwrap_or(&current_pos);

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

                // Theta* Logic:
                // Check if we can go straight from Parent(Current) to Neighbor
                let (new_parent, new_g) = if self.has_line_of_sight(parent_pos, neighbor_pos) {
                    // Path 2: Parent(Current) -> Neighbor
                    let g_parent = *g_scores.get(&parent_pos).unwrap_or(&f32::INFINITY);
                    let dist = self.euclidean_dist(parent_pos, neighbor_pos);
                    (parent_pos, g_parent + dist)
                } else {
                    // Path 1: Current -> Neighbor (Standard A*)
                    let g_current = *g_scores.get(&current_pos).unwrap_or(&f32::INFINITY);
                    let dist = self.euclidean_dist(current_pos, neighbor_pos);
                    (current_pos, g_current + dist)
                };

                let neighbor_old_g = *g_scores.get(&neighbor_pos).unwrap_or(&f32::INFINITY);

                if new_g < neighbor_old_g {
                    came_from.insert(neighbor_pos, new_parent);
                    g_scores.insert(neighbor_pos, new_g);

                    let neighbor_node = Node {
                        x: nx,
                        y: ny,
                        g: new_g,
                        h: self.heuristic(neighbor_pos, goal_grid),
                    };

                    open_set.push(neighbor_node);
                }
            }
        }

        self.fail_with_visited(visited_order, iterations)
    }

    fn fail(&self) -> ThetaStarResult {
        ThetaStarResult {
            path: vec![],
            visited: vec![],
            success: false,
            iterations: 0,
        }
    }

    fn fail_with_visited(&self, visited: Vec<(usize, usize)>, iterations: usize) -> ThetaStarResult {
        ThetaStarResult {
            path: vec![],
            visited,
            success: false,
            iterations,
        }
    }

    /// Euclidean heuristic
    fn heuristic(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        self.euclidean_dist(a, b)
    }

    /// Euclidean distance between two grid cells
    fn euclidean_dist(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        let dx = (a.0 as f32 - b.0 as f32).abs();
        let dy = (a.1 as f32 - b.1 as f32).abs();
        (dx * dx + dy * dy).sqrt()
    }

    /// Bresenham's Line Algorithm for Line-of-Sight check
    /// Returns true if there is a clear line of sight between s and e
    fn has_line_of_sight(&self, s: (usize, usize), e: (usize, usize)) -> bool {
        let x0 = s.0 as i32;
        let y0 = s.1 as i32;
        let x1 = e.0 as i32;
        let y1 = e.1 as i32;

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            // Check if current cell is obstacle
            // (Note: s and e are start and end, we usually allow start to be whatever, 
            // but for LOS check in path planning, we need strictly free path. 
            // However, usually the start and end nodes themselves are free if we got here.)
            if self.grid.is_obstacle(x as usize, y as usize) {
                return false;
            }

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }

        true
    }

    /// Reconstruct path from came_from map
    fn reconstruct_path(
        &self,
        came_from: &HashMap<(usize, usize), (usize, usize)>,
        goal: (usize, usize),
        start: (usize, usize),
    ) -> Vec<(f32, f32)> {
        let mut path = Vec::new();
        let mut current = goal;

        // In Theta*, parent can be far away, so we just add the waypoints.
        // The path is the sequence of parents.
        
        path.push(self.grid.grid_to_world(current.0, current.1));

        while current != start {
            match came_from.get(&current) {
                Some(&parent) => {
                    if parent == current {
                        // Should not happen unless it's start, but start is handled by loop condition
                        break; 
                    }
                    current = parent;
                    path.push(self.grid.grid_to_world(current.0, current.1));
                },
                None => break, // Should not happen if path found
            }
        }

        // Add start position (already added if loop worked correctly?)
        // Wait, loop stops when current == start. So start is NOT added inside the loop.
        // path.push(self.grid.grid_to_world(start.0, start.1)); // The logic above effectively does this?
        // Let's trace: 
        // Goal -> Parent(Goal) -> ... -> Start.
        // Loop condition: current != start.
        // Last iteration: current is some node X where Parent(X) == Start.
        // We push X.
        // We set current = Start.
        // Loop terminates.
        // Start is NOT pushed.
        
        // Wait, checking A* implementation:
        /*
        while current != start {
            // ... push current ...
            current = parent;
        }
        path.push(start_world);
        */
        // Yes, need to push start.
        
        // Check duplication: If start == goal?
        // path has goal. loop doesn't run. we push start. path = [goal, start].
        // distinct? yes. (x,y) are same but floats.
        
        // To be safe and clean:
        if path.last().map(|p| *p != self.grid.grid_to_world(start.0, start.1)).unwrap_or(true) {
             path.push(self.grid.grid_to_world(start.0, start.1));
        }

        // Reverse to get path from start to goal
        path.reverse();
        path
    }
}
