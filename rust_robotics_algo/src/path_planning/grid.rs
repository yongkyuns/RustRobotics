//! Grid representation for path planning

use std::collections::HashSet;

/// A 2D grid for path planning
#[derive(Debug, Clone)]
pub struct Grid {
    /// Width of the grid in cells
    pub width: usize,
    /// Height of the grid in cells
    pub height: usize,
    /// Resolution (meters per cell)
    pub resolution: f32,
    /// Set of obstacle cell indices (x, y)
    obstacles: HashSet<(usize, usize)>,
    /// Origin offset in world coordinates (center of grid)
    origin_x: f32,
    origin_y: f32,
}

impl Grid {
    /// Create a new grid with given dimensions and resolution
    ///
    /// The grid is centered at the world origin.
    pub fn new(width: usize, height: usize, resolution: f32) -> Self {
        let origin_x = (width as f32 * resolution) / 2.0;
        let origin_y = (height as f32 * resolution) / 2.0;
        Self {
            width,
            height,
            resolution,
            obstacles: HashSet::new(),
            origin_x,
            origin_y,
        }
    }

    /// Set a cell as an obstacle
    pub fn set_obstacle(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            self.obstacles.insert((x, y));
        }
    }

    /// Clear an obstacle from a cell
    pub fn clear_obstacle(&mut self, x: usize, y: usize) {
        self.obstacles.remove(&(x, y));
    }

    /// Toggle obstacle state at a cell
    pub fn toggle_obstacle(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            if self.obstacles.contains(&(x, y)) {
                self.obstacles.remove(&(x, y));
            } else {
                self.obstacles.insert((x, y));
            }
        }
    }

    /// Check if a cell index is within bounds
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Check if a cell is an obstacle
    pub fn is_obstacle(&self, x: usize, y: usize) -> bool {
        self.obstacles.contains(&(x, y))
    }

    /// Check if a cell is free (valid and not an obstacle)
    pub fn is_free(&self, x: usize, y: usize) -> bool {
        self.is_valid(x, y) && !self.is_obstacle(x, y)
    }

    /// Convert world coordinates to grid cell indices
    ///
    /// Returns None if the position is outside the grid
    pub fn world_to_grid(&self, wx: f32, wy: f32) -> Option<(usize, usize)> {
        let gx = ((wx + self.origin_x) / self.resolution).floor();
        let gy = ((wy + self.origin_y) / self.resolution).floor();

        if gx >= 0.0 && gy >= 0.0 {
            let gx = gx as usize;
            let gy = gy as usize;
            if gx < self.width && gy < self.height {
                return Some((gx, gy));
            }
        }
        None
    }

    /// Convert grid cell indices to world coordinates (cell center)
    pub fn grid_to_world(&self, gx: usize, gy: usize) -> (f32, f32) {
        let wx = (gx as f32 + 0.5) * self.resolution - self.origin_x;
        let wy = (gy as f32 + 0.5) * self.resolution - self.origin_y;
        (wx, wy)
    }

    /// Get the world bounds of the grid
    pub fn world_bounds(&self) -> (f32, f32, f32, f32) {
        let min_x = -self.origin_x;
        let min_y = -self.origin_y;
        let max_x = self.width as f32 * self.resolution - self.origin_x;
        let max_y = self.height as f32 * self.resolution - self.origin_y;
        (min_x, min_y, max_x, max_y)
    }

    /// Get all obstacle positions
    pub fn obstacles(&self) -> &HashSet<(usize, usize)> {
        &self.obstacles
    }

    /// Clear all obstacles
    pub fn clear_all_obstacles(&mut self) {
        self.obstacles.clear();
    }
}

impl Default for Grid {
    fn default() -> Self {
        Self::new(20, 20, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = Grid::new(10, 10, 0.5);
        assert_eq!(grid.width, 10);
        assert_eq!(grid.height, 10);
        assert_eq!(grid.resolution, 0.5);
    }

    #[test]
    fn test_world_grid_conversion() {
        let grid = Grid::new(10, 10, 1.0);
        // Center of grid should map to center cell
        let (gx, gy) = grid.world_to_grid(0.0, 0.0).unwrap();
        assert_eq!(gx, 5);
        assert_eq!(gy, 5);

        // Convert back
        let (wx, wy) = grid.grid_to_world(gx, gy);
        assert!((wx - 0.5).abs() < 0.01);
        assert!((wy - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_obstacles() {
        let mut grid = Grid::new(10, 10, 1.0);
        assert!(!grid.is_obstacle(5, 5));
        grid.set_obstacle(5, 5);
        assert!(grid.is_obstacle(5, 5));
        grid.clear_obstacle(5, 5);
        assert!(!grid.is_obstacle(5, 5));
    }
}
