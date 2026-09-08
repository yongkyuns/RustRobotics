//! Independent finite-grid shortest-path oracles for issue #4.
//!
//! The reference owns occupancy and coordinates separately from production Grid,
//! and uses f64 Floyd-Warshall rather than either planner's heap or heuristic.
//! Destination-only diagonal collision semantics intentionally allow corners.

use rust_robotics_algo::path_planning::{AStarPlanner, DijkstraPlanner, Grid};

struct Map {
    width: usize,
    height: usize,
    resolution: f32,
    blocked: Vec<bool>,
}

impl Map {
    fn center(&self, cell: usize) -> (f32, f32) {
        // Power-of-two resolutions make all fixture centers exactly representable.
        let x = (cell % self.width) as f64 + 0.5 - self.width as f64 / 2.0;
        let y = (cell / self.width) as f64 + 0.5 - self.height as f64 / 2.0;
        let scale = f64::from(self.resolution);
        ((x * scale) as f32, (y * scale) as f32)
    }

    fn grid(&self) -> Grid {
        assert_eq!(self.blocked.len(), self.width * self.height);
        let mut grid = Grid::new(self.width, self.height, self.resolution);
        for (cell, blocked) in self.blocked.iter().enumerate() {
            if *blocked {
                grid.set_obstacle(cell % self.width, cell / self.width);
            }
        }
        grid
    }

    fn distances(&self) -> Vec<f64> {
        let n = self.blocked.len();
        let mut distances = vec![f64::INFINITY; n * n];
        for from in 0..n {
            for to in 0..n {
                if self.blocked[from] || self.blocked[to] {
                    continue;
                }
                let dx = (from % self.width).abs_diff(to % self.width);
                let dy = (from / self.width).abs_diff(to / self.width);
                if dx <= 1 && dy <= 1 {
                    distances[from * n + to] = ((dx * dx + dy * dy) as f64).sqrt();
                }
            }
        }
        for via in 0..n {
            for from in 0..n {
                for to in 0..n {
                    let through = distances[from * n + via] + distances[via * n + to];
                    distances[from * n + to] = distances[from * n + to].min(through);
                }
            }
        }
        distances
    }

    fn check(&self, grid: &Grid, distances: &[f64], start: usize, goal: usize, case: &str) {
        let a = AStarPlanner::new(grid).plan(self.center(start), self.center(goal));
        let d = DijkstraPlanner::new(grid).plan(self.center(start), self.center(goal));
        let expected = distances[start * self.blocked.len() + goal];
        for (name, success, path) in [("A*", a.success, a.path), ("Dijkstra", d.success, d.path)] {
            let context = format!("{case}, {name}, {start}->{goal}, scale={}", self.resolution);
            assert_eq!(success, expected.is_finite(), "{context}: reachability");
            if !success {
                assert!(path.is_empty(), "{context}: failed search returned a path");
                continue;
            }
            assert_eq!(path.first(), Some(&self.center(start)), "{context}: start");
            assert_eq!(path.last(), Some(&self.center(goal)), "{context}: goal");
            assert!(path.len() <= self.blocked.len(), "{context}: path cycle");
            let mut cells = Vec::new();
            for point in &path {
                assert!(
                    point.0.is_finite() && point.1.is_finite(),
                    "{context}: non-finite path"
                );
                let cell = (0..self.blocked.len())
                    .find(|&cell| self.center(cell) == *point)
                    .expect("path point must be an in-bounds cell center");
                assert!(!self.blocked[cell], "{context}: path intersects obstacle");
                cells.push(cell);
            }
            let mut actual = 0.0_f64;
            for pair in cells.windows(2) {
                let dx = (pair[0] % self.width).abs_diff(pair[1] % self.width);
                let dy = (pair[0] / self.width).abs_diff(pair[1] / self.width);
                actual += match (dx, dy) {
                    (1, 0) | (0, 1) => 1.0,
                    (1, 1) => std::f64::consts::SQRT_2,
                    _ => panic!("{context}: illegal path segment"),
                };
            }
            let scale = f64::from(self.resolution);
            let (actual, expected) = (actual * scale, expected * scale);
            // Same error budget as the existing A*/Dijkstra comparison: permits
            // f32 path-cost ranking roundoff, not an extra grid step or detour.
            let tolerance = 1e-5 * expected.max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{context}: optimal cost: actual={actual}, expected={expected}, tol={tolerance}"
            );
        }
    }
}

#[test]
fn exhaustive_small_maps_match_independent_oracle() {
    // All 512 occupancy masks and all 81 ordered endpoint pairs: includes blocked
    // endpoints, start==goal, no-path cases, and diagonal-only corner passages.
    for mask in 0_u32..512 {
        let map = Map {
            width: 3,
            height: 3,
            resolution: 1.0,
            blocked: (0..9).map(|bit| mask & (1 << bit) != 0).collect(),
        };
        let grid = map.grid();
        let distances = map.distances();
        for start in 0..9 {
            for goal in 0..9 {
                map.check(&grid, &distances, start, goal, &format!("mask={mask}"));
            }
        }
    }
}

fn next_word(state: &mut u32) -> u32 {
    // Fixed xorshift32 sequence, independent of rand crate version/platform.
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn seeded_rectangular_maps_match_independent_oracle() {
    for seed in 1_u32..=24 {
        let mut state = seed ^ 0xa341_316c;
        let blocked: Vec<bool> = (0..56).map(|_| next_word(&mut state) % 5 == 0).collect();
        let mut pairs = vec![(0, 55), (7, 48)];
        for _ in 0..6 {
            let start = (next_word(&mut state) % 56) as usize;
            let goal = (next_word(&mut state) % 56) as usize;
            pairs.push((start, goal));
        }
        for resolution in [0.25, 1.0, 2.0] {
            let map = Map {
                width: 8,
                height: 7,
                resolution,
                blocked: blocked.clone(),
            };
            let grid = map.grid();
            let distances = map.distances();
            for &(start, goal) in &pairs {
                map.check(&grid, &distances, start, goal, &format!("seed={seed}"));
            }
        }
    }
}

#[test]
fn oracle_matches_closed_form_and_documented_corner_convention() {
    let mut map = Map {
        width: 3,
        height: 3,
        resolution: 0.25,
        blocked: vec![false; 9],
    };
    let distances = map.distances();
    for from in 0_usize..9 {
        for to in 0_usize..9 {
            let dx = (from % 3).abs_diff(to % 3);
            let dy = (from / 3).abs_diff(to / 3);
            let expected = dx.abs_diff(dy) as f64 + dx.min(dy) as f64 * std::f64::consts::SQRT_2;
            assert!((distances[from * 9 + to] - expected).abs() < 1e-12);
        }
    }
    map.blocked = vec![true; 9];
    for cell in [0, 4, 8] {
        map.blocked[cell] = false;
    }
    let distances = map.distances();
    assert!((distances[8] - 2.0 * std::f64::consts::SQRT_2).abs() < 1e-12);
    map.check(&map.grid(), &distances, 0, 8, "diagonal-only passage");
}

#[test]
fn out_of_bounds_endpoints_are_rejected() {
    let map = Map {
        width: 8,
        height: 6,
        resolution: 0.25,
        blocked: vec![false; 48],
    };
    let grid = map.grid();
    // Upper edges are exclusive; lower-edge probes lie strictly outside.
    for outside in [(1.0, 0.0), (0.0, 0.75), (-1.25, 0.0), (0.0, -1.0)] {
        for (start, goal) in [(outside, map.center(20)), (map.center(20), outside)] {
            let a = AStarPlanner::new(&grid).plan(start, goal);
            let d = DijkstraPlanner::new(&grid).plan(start, goal);
            assert!(!a.success && a.path.is_empty());
            assert!(!d.success && d.path.is_empty());
        }
    }
}
