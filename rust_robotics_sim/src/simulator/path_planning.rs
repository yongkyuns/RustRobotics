//! Path planning simulation with multiple algorithms

use egui::*;
use egui_plot::{Line, PlotPoints, PlotUi, Polygon};
use rust_robotics_algo::path_planning::{
    AStarPlanner, CircleObstacle, DijkstraPlanner, Grid, RrtConfig, RrtNode, RrtPlanner,
};

use super::{Draw, Simulate};

/// Available path planning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    AStar,
    Dijkstra,
    Rrt,
}

impl Algorithm {
    fn label(&self) -> &'static str {
        match self {
            Algorithm::AStar => "A*",
            Algorithm::Dijkstra => "Dijkstra",
            Algorithm::Rrt => "RRT",
        }
    }

}

/// State of the path planning process
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanningState {
    /// Waiting for start point
    WaitingForStart,
    /// Waiting for goal point
    WaitingForGoal,
    /// Planning complete, showing result
    ShowingResult,
}

/// Unified planning result
#[derive(Debug, Clone)]
struct PlanningResult {
    /// Path in world coordinates
    path: Vec<(f32, f32)>,
    /// Visited cells for grid-based algorithms
    visited: Vec<(usize, usize)>,
    /// Tree nodes for RRT
    tree: Vec<RrtNode>,
    /// Whether planning succeeded
    success: bool,
    /// Number of iterations (main loop iterations)
    iterations: usize,
    /// Path length in world units
    path_length: f32,
    /// Euclidean distance from start to goal (for optimality ratio)
    euclidean_distance: f32,
}

/// Path planning simulation
pub struct PathPlanning {
    /// Selected algorithm
    algorithm: Algorithm,
    /// The grid for planning (used by all algorithms)
    grid: Grid,
    /// Start position in world coordinates
    start: Option<(f32, f32)>,
    /// Goal position in world coordinates
    goal: Option<(f32, f32)>,
    /// Planning result
    result: Option<PlanningResult>,
    /// Current planning state
    state: PlanningState,
    /// Unique identifier
    id: usize,
    /// Grid width (in cells)
    grid_width: usize,
    /// Grid height (in cells)
    grid_height: usize,
    /// Grid resolution (meters per cell)
    grid_resolution: f32,
    /// Show visited cells / tree
    show_visited: bool,
    /// Animation progress for visited cells (0.0 to 1.0)
    animation_progress: f32,
    /// Whether animation is complete
    animation_complete: bool,
    /// Last cell toggled during drag (to avoid re-toggling same cell)
    last_toggled_cell: Option<(usize, usize)>,
    /// Whether we're adding or removing obstacles during drag
    drag_adding_obstacle: bool,
    /// RRT expand distance
    rrt_expand_dist: f32,
    /// RRT goal sample rate
    rrt_goal_sample_rate: f32,
    /// RRT max iterations
    rrt_max_iter: usize,
}

impl PathPlanning {
    /// Create a new path planning simulation
    pub fn new(id: usize, _time: f32) -> Self {
        let grid_width = 40;
        let grid_height = 40;
        let grid_resolution = 1.0;

        Self {
            algorithm: Algorithm::AStar,
            grid: Grid::new(grid_width, grid_height, grid_resolution),
            start: None,
            goal: None,
            result: None,
            state: PlanningState::WaitingForStart,
            id,
            grid_width,
            grid_height,
            grid_resolution,
            show_visited: true,
            animation_progress: 0.0,
            animation_complete: false,
            last_toggled_cell: None,
            drag_adding_obstacle: true,
            rrt_expand_dist: 1.0,
            rrt_goal_sample_rate: 0.1,
            rrt_max_iter: 500,
        }
    }

    /// Run the selected algorithm
    fn run_planner(&mut self) {
        if let (Some(start), Some(goal)) = (self.start, self.goal) {
            self.animation_progress = 0.0;
            self.animation_complete = false;

            // Compute euclidean distance from start to goal
            let dx = goal.0 - start.0;
            let dy = goal.1 - start.1;
            let euclidean_distance = (dx * dx + dy * dy).sqrt();

            let mut result = match self.algorithm {
                Algorithm::AStar => {
                    let planner = AStarPlanner::new(&self.grid);
                    let res = planner.plan(start, goal);
                    PlanningResult {
                        path: res.path,
                        visited: res.visited,
                        tree: Vec::new(),
                        success: res.success,
                        iterations: res.iterations,
                        path_length: 0.0,
                        euclidean_distance,
                    }
                }
                Algorithm::Dijkstra => {
                    let planner = DijkstraPlanner::new(&self.grid);
                    let res = planner.plan(start, goal);
                    PlanningResult {
                        path: res.path,
                        visited: res.visited,
                        tree: Vec::new(),
                        success: res.success,
                        iterations: res.iterations,
                        path_length: 0.0,
                        euclidean_distance,
                    }
                }
                Algorithm::Rrt => {
                    let (min_x, min_y, max_x, max_y) = self.grid.world_bounds();
                    let config = RrtConfig {
                        expand_distance: self.rrt_expand_dist,
                        goal_sample_rate: self.rrt_goal_sample_rate,
                        max_iterations: self.rrt_max_iter,
                        goal_threshold: self.rrt_expand_dist,
                        seed: None,
                    };
                    let mut planner = RrtPlanner::new(min_x, max_x, min_y, max_y, config);

                    // Convert grid obstacles to circles for RRT
                    // Use radius that covers the grid cell (half diagonal)
                    let radius = self.grid.resolution * 0.5 * std::f32::consts::SQRT_2;
                    for &(gx, gy) in self.grid.obstacles() {
                        let (cx, cy) = self.grid.grid_to_world(gx, gy);
                        planner.add_obstacle(CircleObstacle::new(cx, cy, radius));
                    }

                    let res = planner.plan(start, goal);
                    PlanningResult {
                        path: res.path,
                        visited: Vec::new(),
                        tree: res.tree,
                        success: res.success,
                        iterations: res.iterations,
                        path_length: 0.0,
                        euclidean_distance,
                    }
                }
            };

            result.path_length = Self::compute_path_length(&result.path);

            self.result = Some(result);
            self.state = PlanningState::ShowingResult;
        }
    }

    /// Compute total path length in world units
    fn compute_path_length(path: &[(f32, f32)]) -> f32 {
        if path.len() < 2 {
            return 0.0;
        }
        path.windows(2)
            .map(|w| {
                let dx = w[1].0 - w[0].0;
                let dy = w[1].1 - w[0].1;
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// Clear the current planning state
    fn clear(&mut self) {
        self.start = None;
        self.goal = None;
        self.result = None;
        self.state = PlanningState::WaitingForStart;
        self.animation_progress = 0.0;
        self.animation_complete = false;
    }

    /// Copy start, goal, obstacles, and grid settings from another planner
    /// and run the algorithm if both start and goal are set
    pub fn copy_state_from(&mut self, other: &PathPlanning) {
        // Copy grid settings
        self.grid_width = other.grid_width;
        self.grid_height = other.grid_height;
        self.grid_resolution = other.grid_resolution;
        self.grid = Grid::new(self.grid_width, self.grid_height, self.grid_resolution);

        // Copy obstacles
        for &(x, y) in other.grid.obstacles() {
            self.grid.set_obstacle(x, y);
        }

        // Copy start and goal
        self.start = other.start;
        self.goal = other.goal;

        // Update state based on what we have
        if self.start.is_some() && self.goal.is_some() {
            self.state = PlanningState::ShowingResult;
            self.run_planner();
        } else if self.start.is_some() {
            self.state = PlanningState::WaitingForGoal;
        } else {
            self.state = PlanningState::WaitingForStart;
        }
    }

    /// Rebuild grid with new dimensions
    fn rebuild_grid(&mut self) {
        let old_obstacles: Vec<_> = self.grid.obstacles().iter().copied().collect();
        self.grid = Grid::new(self.grid_width, self.grid_height, self.grid_resolution);

        // Restore obstacles that are still within bounds
        for (x, y) in old_obstacles {
            if x < self.grid_width && y < self.grid_height {
                self.grid.set_obstacle(x, y);
            }
        }

        // Clear planning state since grid changed
        self.clear();
    }

    /// Update grid settings from external configuration
    pub fn update_grid_settings(&mut self, width: usize, height: usize, resolution: f32) {
        if self.grid_width != width || self.grid_height != height || self.grid_resolution != resolution {
            self.grid_width = width;
            self.grid_height = height;
            self.grid_resolution = resolution;
            self.rebuild_grid();
        }
    }

    /// Handle mouse interaction for placing points and obstacles
    pub fn handle_mouse(&mut self, plot_response: &egui_plot::PlotResponse<()>) {
        // Reset drag state when right button is released
        if !plot_response
            .response
            .ctx
            .input(|i| i.pointer.secondary_down())
        {
            self.last_toggled_cell = None;
        }

        if let Some(pos) = plot_response.response.hover_pos() {
            let plot_pos = plot_response.transform.value_from_position(pos);
            let wx = plot_pos.x as f32;
            let wy = plot_pos.y as f32;

            // All algorithms use grid obstacles
            if let Some((gx, gy)) = self.grid.world_to_grid(wx, wy) {
                let current_cell = (gx, gy);

                // Right-click to add/remove obstacles
                if plot_response.response.secondary_clicked() {
                    // On initial click, determine mode and toggle the first cell
                    self.drag_adding_obstacle = !self.grid.is_obstacle(gx, gy);
                    if self.drag_adding_obstacle {
                        self.grid.set_obstacle(gx, gy);
                    } else {
                        self.grid.clear_obstacle(gx, gy);
                    }
                    self.last_toggled_cell = Some(current_cell);

                    if self.start.is_some() && self.goal.is_some() {
                        self.run_planner();
                    }
                }
                // Right-click drag to continue adding/removing
                else if plot_response
                    .response
                    .ctx
                    .input(|i| i.pointer.secondary_down())
                {
                    if self.last_toggled_cell != Some(current_cell) {
                        if self.drag_adding_obstacle {
                            self.grid.set_obstacle(gx, gy);
                        } else {
                            self.grid.clear_obstacle(gx, gy);
                        }
                        self.last_toggled_cell = Some(current_cell);

                        if self.start.is_some() && self.goal.is_some() {
                            self.run_planner();
                        }
                    }
                }
                // Left-click to set start/goal
                else if plot_response.response.clicked() {
                    if !self.grid.is_obstacle(gx, gy) {
                        let world_pos = self.grid.grid_to_world(gx, gy);
                        self.handle_left_click(world_pos);
                    }
                }
            }
        }
    }

    fn handle_left_click(&mut self, world_pos: (f32, f32)) {
        match self.state {
            PlanningState::WaitingForStart => {
                self.start = Some(world_pos);
                self.state = PlanningState::WaitingForGoal;
            }
            PlanningState::WaitingForGoal => {
                self.goal = Some(world_pos);
                self.run_planner();
            }
            PlanningState::ShowingResult => {
                self.start = Some(world_pos);
                self.goal = None;
                self.result = None;
                self.state = PlanningState::WaitingForGoal;
            }
        }
    }

    /// Draw the grid lines
    fn draw_grid(&self, plot_ui: &mut PlotUi<'_>) {
        let (min_x, min_y, max_x, max_y) = self.grid.world_bounds();

        // Draw vertical lines
        for i in 0..=self.grid.width {
            let x = min_x + i as f32 * self.grid.resolution;
            let points = PlotPoints::new(vec![[x as f64, min_y as f64], [x as f64, max_y as f64]]);
            plot_ui.line(
                Line::new("", points)
                    .color(Color32::from_gray(200))
                    .width(0.5),
            );
        }

        // Draw horizontal lines
        for i in 0..=self.grid.height {
            let y = min_y + i as f32 * self.grid.resolution;
            let points = PlotPoints::new(vec![[min_x as f64, y as f64], [max_x as f64, y as f64]]);
            plot_ui.line(
                Line::new("", points)
                    .color(Color32::from_gray(200))
                    .width(0.5),
            );
        }
    }

    /// Draw grid obstacles as filled cells
    fn draw_grid_obstacles(&self, plot_ui: &mut PlotUi<'_>) {
        for &(gx, gy) in self.grid.obstacles() {
            let (cx, cy) = self.grid.grid_to_world(gx, gy);
            let half = self.grid.resolution / 2.0;

            let points = PlotPoints::new(vec![
                [(cx - half) as f64, (cy - half) as f64],
                [(cx + half) as f64, (cy - half) as f64],
                [(cx + half) as f64, (cy + half) as f64],
                [(cx - half) as f64, (cy + half) as f64],
            ]);

            plot_ui.polygon(
                Polygon::new("", points)
                    .fill_color(Color32::from_gray(80))
                    .stroke(egui::Stroke::new(1.0, Color32::from_gray(60))),
            );
        }
    }

    /// Draw visited cells for grid-based algorithms
    fn draw_visited(&self, plot_ui: &mut PlotUi<'_>) {
        if !self.show_visited {
            return;
        }

        if let Some(result) = &self.result {
            let num_cells = if self.animation_complete {
                result.visited.len()
            } else {
                (result.visited.len() as f32 * self.animation_progress) as usize
            };

            for &(gx, gy) in result.visited.iter().take(num_cells) {
                let (cx, cy) = self.grid.grid_to_world(gx, gy);
                let half = self.grid.resolution / 2.0 * 0.8;

                let points = PlotPoints::new(vec![
                    [(cx - half) as f64, (cy - half) as f64],
                    [(cx + half) as f64, (cy - half) as f64],
                    [(cx + half) as f64, (cy + half) as f64],
                    [(cx - half) as f64, (cy + half) as f64],
                ]);

                plot_ui.polygon(
                    Polygon::new("", points)
                        .fill_color(Color32::from_rgba_unmultiplied(100, 180, 255, 60))
                        .stroke(egui::Stroke::NONE),
                );
            }
        }
    }

    /// Draw RRT tree
    fn draw_tree(&self, plot_ui: &mut PlotUi<'_>) {
        if !self.show_visited {
            return;
        }

        if let Some(result) = &self.result {
            let num_edges = if self.animation_complete {
                result.tree.len()
            } else {
                (result.tree.len() as f32 * self.animation_progress) as usize
            };

            for node in result.tree.iter().take(num_edges) {
                if let Some(parent_idx) = node.parent {
                    let parent = &result.tree[parent_idx];
                    let points = PlotPoints::new(vec![
                        [parent.x as f64, parent.y as f64],
                        [node.x as f64, node.y as f64],
                    ]);
                    plot_ui.line(
                        Line::new("", points)
                            .color(Color32::from_rgba_unmultiplied(100, 180, 255, 100))
                            .width(1.0),
                    );
                }
            }
        }
    }

    /// Draw the final path
    fn draw_path(&self, plot_ui: &mut PlotUi<'_>) {
        if let Some(result) = &self.result {
            if result.success && result.path.len() >= 2 {
                // Animate path based on progress
                let num_points = if self.animation_complete {
                    result.path.len()
                } else {
                    ((result.path.len() as f32 * self.animation_progress) as usize).max(2)
                };

                let points: Vec<[f64; 2]> = result
                    .path
                    .iter()
                    .take(num_points)
                    .map(|(x, y)| [*x as f64, *y as f64])
                    .collect();

                plot_ui.line(
                    Line::new("Path", PlotPoints::new(points))
                        .color(Color32::from_rgb(50, 100, 255))
                        .width(3.0),
                );
            }
        }
    }

    /// Draw start marker (green circle)
    fn draw_start(&self, plot_ui: &mut PlotUi<'_>) {
        if let Some((x, y)) = self.start {
            let radius = self.grid.resolution / 3.0;
            let points = self.circle_points(x, y, radius, 16);
            plot_ui.polygon(
                Polygon::new("Start", PlotPoints::new(points))
                    .fill_color(Color32::from_rgb(50, 200, 50))
                    .stroke(egui::Stroke::new(2.0, Color32::from_rgb(30, 150, 30))),
            );
        }
    }

    /// Draw goal marker (red circle)
    fn draw_goal(&self, plot_ui: &mut PlotUi<'_>) {
        if let Some((x, y)) = self.goal {
            let radius = self.grid.resolution / 3.0;
            let points = self.circle_points(x, y, radius, 16);
            plot_ui.polygon(
                Polygon::new("Goal", PlotPoints::new(points))
                    .fill_color(Color32::from_rgb(200, 50, 50))
                    .stroke(egui::Stroke::new(2.0, Color32::from_rgb(150, 30, 30))),
            );
        }
    }

    /// Generate points for a circle
    fn circle_points(&self, cx: f32, cy: f32, radius: f32, segments: usize) -> Vec<[f64; 2]> {
        (0..segments)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
                [
                    (cx + radius * angle.cos()) as f64,
                    (cy + radius * angle.sin()) as f64,
                ]
            })
            .collect()
    }
}

impl Default for PathPlanning {
    fn default() -> Self {
        Self::new(1, 0.0)
    }
}

impl Simulate for PathPlanning {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.id
    }

    fn match_state_with(&mut self, _other: &dyn Simulate) {
        // No state synchronization needed for path planning
    }

    fn step(&mut self, dt: f32) {
        // Animate visited cells / tree visualization
        if self.result.is_some() && !self.animation_complete {
            self.animation_progress += dt * 1.0;
            if self.animation_progress >= 1.0 {
                self.animation_progress = 1.0;
                self.animation_complete = true;
            }
        }
    }

    fn reset_state(&mut self) {
        // Rerun the planner with current start/end (restart animation)
        if self.start.is_some() && self.goal.is_some() {
            self.run_planner();
        }
    }

    fn reset_all(&mut self) {
        // Clear obstacles
        self.grid = Grid::new(self.grid_width, self.grid_height, self.grid_resolution);
        // Clear start/end and result
        self.clear();
    }
}

impl Draw for PathPlanning {
    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        self.draw_grid(plot_ui);
        self.draw_grid_obstacles(plot_ui);

        // Draw algorithm-specific visualizations
        if self.algorithm == Algorithm::Rrt {
            self.draw_tree(plot_ui);
        } else {
            self.draw_visited(plot_ui);
        }

        self.draw_path(plot_ui);
        self.draw_start(plot_ui);
        self.draw_goal(plot_ui);
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        let mut keep = true;

        ui.push_id(self.id, |ui| {
            ui.group(|ui| {
                ui.set_width(220.0); // Fixed width for each planner panel
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading(format!("Path Planning {}", self.id));
                        if ui.small_button("x").clicked() {
                            keep = false;
                        }
                    });

                ui.separator();

                // Algorithm selector
                ui.horizontal(|ui| {
                    ui.label("Algorithm:");
                    for algo in [Algorithm::AStar, Algorithm::Dijkstra, Algorithm::Rrt] {
                        if ui
                            .selectable_label(self.algorithm == algo, algo.label())
                            .clicked()
                        {
                            self.algorithm = algo;
                            // Re-run planner with new algorithm if start/goal exist
                            if self.start.is_some() && self.goal.is_some() {
                                self.run_planner();
                            } else {
                                self.result = None;
                            }
                        }
                    }
                });

                // RRT-specific settings
                if self.algorithm == Algorithm::Rrt {
                    ui.collapsing("RRT Settings", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Expand dist:");
                            ui.add(
                                DragValue::new(&mut self.rrt_expand_dist)
                                    .range(0.1..=5.0)
                                    .speed(0.1),
                            );
                        });

                        ui.horizontal(|ui| {
                            ui.label("Goal bias:");
                            ui.add(
                                DragValue::new(&mut self.rrt_goal_sample_rate)
                                    .range(0.0..=1.0)
                                    .speed(0.01),
                            );
                        });

                        ui.horizontal(|ui| {
                            ui.label("Max iter:");
                            ui.add(DragValue::new(&mut self.rrt_max_iter).range(100..=5000));
                        });
                    });
                }

                let show_label = if self.algorithm == Algorithm::Rrt {
                    "Show tree"
                } else {
                    "Show visited cells"
                };
                ui.checkbox(&mut self.show_visited, show_label);

                ui.separator();

                // Status
                let status_text = match self.state {
                    PlanningState::WaitingForStart => "Click to set start point",
                    PlanningState::WaitingForGoal => "Click to set goal point",
                    PlanningState::ShowingResult => {
                        if let Some(result) = &self.result {
                            if result.success {
                                "Path found!"
                            } else {
                                "No path found"
                            }
                        } else {
                            "Ready"
                        }
                    }
                };
                ui.label(format!("Status: {}", status_text));

                // Result info / Performance metrics
                if let Some(result) = &self.result {
                    ui.separator();
                    ui.label("Performance:");

                    // Iteration count
                    ui.label(format!("  Iterations: {}", result.iterations));

                    // Algorithm-specific stats
                    if self.algorithm == Algorithm::Rrt {
                        ui.label(format!("  Tree nodes: {}", result.tree.len()));
                    } else {
                        ui.label(format!("  Cells visited: {}", result.visited.len()));
                    }

                    // Path stats
                    if result.success {
                        ui.label(format!("  Path waypoints: {}", result.path.len()));
                        ui.label(format!("  Path length: {:.2} units", result.path_length));

                        // Optimality ratio (path_length / euclidean_distance)
                        // 1.0 = optimal (straight line), higher = less optimal
                        if result.euclidean_distance > 0.0 {
                            let optimality_ratio = result.path_length / result.euclidean_distance;
                            ui.label(format!("  Optimality ratio: {:.2}", optimality_ratio));
                        }
                    }
                }

                ui.separator();
                ui.label("Left-click: Set start/goal");
                ui.label("Right-click drag: Paint obstacles");
                }); // end vertical
            }); // end group
        }); // end push_id

        keep
    }
}
