//! Path planning simulation with multiple algorithms

use egui::*;
use egui_plot::{Line, PlotPoints, PlotUi, Polygon};
use rust_robotics_algo::path_planning::{
    AStarPlanner, CircleObstacle, DijkstraPlanner, Grid, RrtConfig, RrtNode, RrtPlanner,
    ThetaStarPlanner,
};

use super::{Draw, Simulate};

/// Environment mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentMode {
    Grid,
    Continuous,
}

/// Available path planning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    AStar,
    ThetaStar,
    Dijkstra,
    Rrt,
}

impl Algorithm {
    fn label(&self) -> &'static str {
        match self {
            Algorithm::AStar => "A*",
            Algorithm::ThetaStar => "Theta*",
            Algorithm::Dijkstra => "Dijkstra",
            Algorithm::Rrt => "RRT",
        }
    }
    
    fn is_grid(&self) -> bool {
        match self {
            Algorithm::AStar | Algorithm::Dijkstra => true,
            Algorithm::ThetaStar | Algorithm::Rrt => false,
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
    /// Execution time
    execution_time: std::time::Duration,
}

struct PlannerInstance {
    id: usize,
    algorithm: Algorithm,
    result: Option<PlanningResult>,
    rrt_expand_dist: f32,
    rrt_goal_sample_rate: f32,
    rrt_max_iter: usize,
    show_visited: bool,
    color: Color32,
}

impl PlannerInstance {
    fn new(id: usize, algorithm: Algorithm, color: Color32) -> Self {
        Self {
            id,
            algorithm,
            result: None,
            rrt_expand_dist: 1.0,
            rrt_goal_sample_rate: 0.1,
            rrt_max_iter: 500,
            show_visited: true,
            color,
        }
    }
}

/// Path planning simulation
pub struct PathPlanning {
    /// Current Environment Mode
    mode: EnvironmentMode,
    /// The grid for planning (used by all algorithms)
    grid: Grid,
    /// Continuous obstacles (used for continuous mode)
    continuous_obstacles: Vec<CircleObstacle>,
    /// Start position in world coordinates
    start: Option<(f32, f32)>,
    /// Goal position in world coordinates
    goal: Option<(f32, f32)>,
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
    /// Animation progress for visited cells (0.0 to 1.0)
    animation_progress: f32,
    /// Whether animation is complete
    animation_complete: bool,
    /// Last cell toggled during drag (to avoid re-toggling same cell)
    last_toggled_cell: Option<(usize, usize)>,
    /// Whether we're adding or removing obstacles during drag
    drag_adding_obstacle: bool,
    /// Continuous obstacle radius for drawing
    continuous_obstacle_radius: f32,
    /// List of active planners
    planners: Vec<PlannerInstance>,
    /// Next planner ID generator
    next_planner_id: usize,
}

impl PathPlanning {
    /// Create a new path planning simulation
    pub fn new(id: usize, _time: f32) -> Self {
        let grid_width = 40;
        let grid_height = 40;
        let grid_resolution = 1.0;

        let mut app = Self {
            mode: EnvironmentMode::Grid,
            grid: Grid::new(grid_width, grid_height, grid_resolution),
            continuous_obstacles: Vec::new(),
            start: None,
            goal: None,
            state: PlanningState::WaitingForStart,
            id,
            grid_width,
            grid_height,
            grid_resolution,
            animation_progress: 0.0,
            animation_complete: false,
            last_toggled_cell: None,
            drag_adding_obstacle: true,
            continuous_obstacle_radius: 1.0,
            planners: Vec::new(),
            next_planner_id: 0,
        };
        
        // Add a default planner
        app.add_planner(Algorithm::AStar);
        app
    }
    
    fn get_color(sim_id: usize, planner_id: usize) -> Color32 {
        let colors = [
            Color32::from_rgb(50, 100, 255),  // Blue
            Color32::from_rgb(255, 100, 50),  // Red
            Color32::from_rgb(50, 255, 100),  // Green
            Color32::from_rgb(255, 200, 50),  // Yellow
            Color32::from_rgb(200, 50, 255),  // Purple
            Color32::from_rgb(50, 255, 255),  // Cyan
        ];
        colors[(sim_id + planner_id) % colors.len()]
    }
    
    fn add_planner(&mut self, algorithm: Algorithm) {
        let color = Self::get_color(self.id, self.next_planner_id);
        self.planners.push(PlannerInstance::new(self.next_planner_id, algorithm, color));
        self.next_planner_id += 1;
        
        // If we have start/goal, run this new planner immediately
        if self.start.is_some() && self.goal.is_some() {
            let last_idx = self.planners.len() - 1;
            self.run_single_planner(last_idx);
        }
    }
    
    fn remove_planner(&mut self, planner_id: usize) {
        if let Some(pos) = self.planners.iter().position(|p| p.id == planner_id) {
            self.planners.remove(pos);
        }
    }

    /// Run all planners
    fn run_all_planners(&mut self) {
        if self.start.is_some() && self.goal.is_some() {
            self.animation_progress = 0.0;
            self.animation_complete = false;
            
            // Need to compute this once but borrow checker makes it hard to share euclidean_dist
            // So we'll just let each planner compute it or compute it inside the loop
            
            for i in 0..self.planners.len() {
                self.run_single_planner(i);
            }
            self.state = PlanningState::ShowingResult;
        }
    }
    
    fn run_single_planner(&mut self, idx: usize) {
        if let (Some(start), Some(goal)) = (self.start, self.goal) {
             let algo = self.planners[idx].algorithm;
             
             // Compute euclidean distance
            let dx = goal.0 - start.0;
            let dy = goal.1 - start.1;
            let euclidean_distance = (dx * dx + dy * dy).sqrt();
            
            // If in continuous mode and using a grid-based algo (or Theta*), rasterize first
            if self.mode == EnvironmentMode::Continuous {
                match algo {
                    Algorithm::AStar | Algorithm::ThetaStar | Algorithm::Dijkstra => {
                        self.rasterize_continuous_to_grid();
                    }
                    _ => {}
                }
            }
            
            // Re-borrow for RRT config access if needed
            // But we need to be careful. RRT Configs are stored in planner_inst.
            
            let start_time = std::time::Instant::now();

            let mut result = match algo {
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
                        execution_time: std::time::Duration::default(),
                    }
                }
                Algorithm::ThetaStar => {
                    let planner = ThetaStarPlanner::new(&self.grid);
                    let res = planner.plan(start, goal);
                    PlanningResult {
                        path: res.path,
                        visited: res.visited,
                        tree: Vec::new(),
                        success: res.success,
                        iterations: res.iterations,
                        path_length: 0.0,
                        euclidean_distance,
                        execution_time: std::time::Duration::default(),
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
                        execution_time: std::time::Duration::default(),
                    }
                }
                Algorithm::Rrt => {
                    let (min_x, min_y, max_x, max_y) = self.grid.world_bounds();
                    let config = {
                         let p = &self.planners[idx];
                         RrtConfig {
                            expand_distance: p.rrt_expand_dist,
                            goal_sample_rate: p.rrt_goal_sample_rate,
                            max_iterations: p.rrt_max_iter,
                            goal_threshold: p.rrt_expand_dist,
                            seed: None,
                        }
                    };
                    
                    let mut planner = RrtPlanner::new(min_x, max_x, min_y, max_y, config);

                    if self.mode == EnvironmentMode::Grid {
                        // Convert grid obstacles to circles for RRT in Grid mode
                        let radius = self.grid.resolution * 0.5 * std::f32::consts::SQRT_2;
                        for &(gx, gy) in self.grid.obstacles() {
                            let (cx, cy) = self.grid.grid_to_world(gx, gy);
                            planner.add_obstacle(CircleObstacle::new(cx, cy, radius));
                        }
                    } else {
                        // Use continuous obstacles
                        for obs in &self.continuous_obstacles {
                            planner.add_obstacle(obs.clone());
                        }
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
                        execution_time: std::time::Duration::default(),
                    }
                }
            };
            
            result.execution_time = start_time.elapsed();
            result.path_length = Self::compute_path_length(&result.path);
            self.planners[idx].result = Some(result);
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
        for p in &mut self.planners {
            p.result = None;
        }
        self.state = PlanningState::WaitingForStart;
        self.animation_progress = 0.0;
        self.animation_complete = false;
    }

    /// Copy state from another planner (Used for persistent state across hot reloads or resets)
    pub fn copy_state_from(&mut self, other: &PathPlanning) {
        self.mode = other.mode;
        
        // Copy grid settings
        self.grid_width = other.grid_width;
        self.grid_height = other.grid_height;
        self.grid_resolution = other.grid_resolution;
        self.grid = Grid::new(self.grid_width, self.grid_height, self.grid_resolution);

        // Copy grid obstacles
        for &(x, y) in other.grid.obstacles() {
            self.grid.set_obstacle(x, y);
        }
        
        // Copy continuous obstacles
        self.continuous_obstacles = other.continuous_obstacles.clone();

        // Copy start and goal
        self.start = other.start;
        self.goal = other.goal;
        
        // Copy planners configs?
        // Since Planners are dynamic now, we try to preserve them if possible
        // But for simplicity in this context, we just keep defaults or copy if struct matches.
        // Given complexity, we'll reset planners list but respect the mode.
        // (A robust implementation would serialize/deserialize or clone configs)
        self.planners.clear();
        if !other.planners.is_empty() {
            // Re-create planners with same algos
             for p in &other.planners {
                let color = Self::get_color(self.id, p.id);
                let mut new_p = PlannerInstance::new(p.id, p.algorithm, color);
                new_p.rrt_expand_dist = p.rrt_expand_dist;
                new_p.rrt_goal_sample_rate = p.rrt_goal_sample_rate;
                new_p.rrt_max_iter = p.rrt_max_iter;
                new_p.show_visited = p.show_visited;
                self.planners.push(new_p);
             }
             self.next_planner_id = other.next_planner_id;
        } else {
             // Fallback default
             let default_algo = if self.mode == EnvironmentMode::Grid { Algorithm::AStar } else { Algorithm::Rrt };
             self.add_planner(default_algo);
        }

        // Update state based on what we have
        if self.start.is_some() && self.goal.is_some() {
            self.state = PlanningState::ShowingResult;
            self.run_all_planners();
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

    /// Set environment mode
    pub fn set_env_mode(&mut self, mode: EnvironmentMode) {
        if self.mode != mode {
            self.mode = mode;
            // Reset planners to defaults suitable for new mode
            self.planners.clear();
            match self.mode {
                EnvironmentMode::Grid => self.add_planner(Algorithm::AStar),
                EnvironmentMode::Continuous => self.add_planner(Algorithm::Rrt),
            }
            self.clear();
        }
    }

    pub fn set_continuous_obstacle_radius(&mut self, radius: f32) {
        self.continuous_obstacle_radius = radius;
    }

    /// Rasterize continuous obstacles into grid
    fn rasterize_continuous_to_grid(&mut self) {
        // Clear all obstacles first
        self.grid.clear_all_obstacles();
        
        let radius_sq = self.continuous_obstacle_radius * self.continuous_obstacle_radius;
        // Padding to ensure we cover all cells that might overlap
        let check_radius_cells = (self.continuous_obstacle_radius / self.grid.resolution).ceil() as usize;

        for obs in &self.continuous_obstacles {
            if let Some((cx, cy)) = self.grid.world_to_grid(obs.x, obs.y) {
                // Determine bounding box in grid
                let min_x = cx.saturating_sub(check_radius_cells);
                let min_y = cy.saturating_sub(check_radius_cells);
                let max_x = (cx + check_radius_cells).min(self.grid.width - 1);
                let max_y = (cy + check_radius_cells).min(self.grid.height - 1);
                
                let obs_r_sq = obs.radius * obs.radius;

                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let (wx, wy) = self.grid.grid_to_world(x, y);
                        let dx = wx - obs.x;
                        let dy = wy - obs.y;
                        if dx*dx + dy*dy <= obs_r_sq {
                            self.grid.set_obstacle(x, y);
                        }
                    }
                }
            }
        }
    }

    /// Handle mouse interaction
    pub fn handle_mouse(&mut self, plot_response: &egui_plot::PlotResponse<()>) {
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

            if self.mode == EnvironmentMode::Grid {
                self.handle_mouse_grid(wx, wy, plot_response);
            } else {
                self.handle_mouse_continuous(wx, wy, plot_response);
            }
        }
    }
    
    fn handle_mouse_grid(&mut self, wx: f32, wy: f32, plot_response: &egui_plot::PlotResponse<()>) {
        if let Some((gx, gy)) = self.grid.world_to_grid(wx, wy) {
            let current_cell = (gx, gy);

            // Right-click to add/remove obstacles
            if plot_response.response.secondary_clicked() {
                self.drag_adding_obstacle = !self.grid.is_obstacle(gx, gy);
                if self.drag_adding_obstacle {
                    self.grid.set_obstacle(gx, gy);
                } else {
                    self.grid.clear_obstacle(gx, gy);
                }
                self.last_toggled_cell = Some(current_cell);
                if self.start.is_some() && self.goal.is_some() { self.run_all_planners(); }
            }
            // Right-click drag
            else if plot_response.response.ctx.input(|i| i.pointer.secondary_down()) {
                if self.last_toggled_cell != Some(current_cell) {
                    if self.drag_adding_obstacle {
                        self.grid.set_obstacle(gx, gy);
                    } else {
                        self.grid.clear_obstacle(gx, gy);
                    }
                    self.last_toggled_cell = Some(current_cell);
                    if self.start.is_some() && self.goal.is_some() { self.run_all_planners(); }
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

    fn handle_mouse_continuous(&mut self, wx: f32, wy: f32, plot_response: &egui_plot::PlotResponse<()>) {
        // Right-click to add/remove obstacles
        if plot_response.response.secondary_clicked() {
            // Check if clicking existing obstacle
            let mut removed = false;
            let click_radius_sq = self.continuous_obstacle_radius * self.continuous_obstacle_radius;
            
            // Iterate backwards
            if let Some(idx) = self.continuous_obstacles.iter().position(|obs| {
                let dx = obs.x - wx;
                let dy = obs.y - wy;
                (dx*dx + dy*dy) < click_radius_sq
            }) {
                self.continuous_obstacles.remove(idx);
                removed = true;
            }

            if !removed {
                self.continuous_obstacles.push(CircleObstacle::new(wx, wy, self.continuous_obstacle_radius));
            }
            
            if self.start.is_some() && self.goal.is_some() { self.run_all_planners(); }
        }
        // Left-click to set start/goal
        else if plot_response.response.clicked() {
            // Check collision with obstacles
            let mut collision = false;
            for obs in &self.continuous_obstacles {
                 let dx = obs.x - wx;
                 let dy = obs.y - wy;
                 if (dx*dx + dy*dy) < obs.radius * obs.radius {
                     collision = true;
                     break;
                 }
            }
            
            if !collision {
                 self.handle_left_click((wx, wy));
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
                self.run_all_planners();
            }
            PlanningState::ShowingResult => {
                self.start = Some(world_pos);
                self.goal = None;
                for p in &mut self.planners { p.result = None; }
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
    
    /// Draw continuous obstacles (circles)
    fn draw_continuous_obstacles(&self, plot_ui: &mut PlotUi<'_>) {
        for obs in &self.continuous_obstacles {
            let points = self.circle_points(obs.x, obs.y, obs.radius, 32);
             plot_ui.polygon(
                Polygon::new("", points)
                    .fill_color(Color32::from_gray(80))
                    .stroke(egui::Stroke::new(1.0, Color32::from_gray(60))),
            );
        }
    }

    /// Draw visited cells for grid-based algorithms
    fn draw_visited(&self, plot_ui: &mut PlotUi<'_>, planner: &PlannerInstance) {
        if !planner.show_visited { return; }

        if let Some(result) = &planner.result {
            let num_cells = if self.animation_complete {
                result.visited.len()
            } else {
                (result.visited.len() as f32 * self.animation_progress) as usize
            };

            // To avoid z-fighting or mess with multiple planners showing visited, 
            // we should probably make them more transparent or only show for one?
            // For now, let's just make them very transparent and tinted with planner color.
            let base_color = planner.color;
            let fill_color = Color32::from_rgba_unmultiplied(base_color.r(), base_color.g(), base_color.b(), 30);

            for &(gx, gy) in result.visited.iter().take(num_cells) {
                let (cx, cy) = self.grid.grid_to_world(gx, gy);
                // Make them slightly smaller to see overlaps?
                let half = self.grid.resolution / 2.0 * 0.8;

                let points = PlotPoints::new(vec![
                    [(cx - half) as f64, (cy - half) as f64],
                    [(cx + half) as f64, (cy - half) as f64],
                    [(cx + half) as f64, (cy + half) as f64],
                    [(cx - half) as f64, (cy + half) as f64],
                ]);

                plot_ui.polygon(
                    Polygon::new("", points)
                        .fill_color(fill_color)
                        .stroke(egui::Stroke::NONE),
                );
            }
        }
    }

    /// Draw RRT tree
    fn draw_tree(&self, plot_ui: &mut PlotUi<'_>, planner: &PlannerInstance) {
        if !planner.show_visited { return; }

        if let Some(result) = &planner.result {
            let num_edges = if self.animation_complete {
                result.tree.len()
            } else {
                (result.tree.len() as f32 * self.animation_progress) as usize
            };
            
            let base_color = planner.color;
            let line_color = Color32::from_rgba_unmultiplied(base_color.r(), base_color.g(), base_color.b(), 100);

            for node in result.tree.iter().take(num_edges) {
                if let Some(parent_idx) = node.parent {
                    let parent = &result.tree[parent_idx];
                    let points = PlotPoints::new(vec![
                        [parent.x as f64, parent.y as f64],
                        [node.x as f64, node.y as f64],
                    ]);
                    plot_ui.line(
                        Line::new("", points)
                            .color(line_color)
                            .width(1.0),
                    );
                }
            }
        }
    }

    /// Draw the final path
    fn draw_path(&self, plot_ui: &mut PlotUi<'_>, planner: &PlannerInstance) {
        if let Some(result) = &planner.result {
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
                    Line::new(format!("{} (Sim {})", planner.algorithm.label(), self.id), PlotPoints::new(points))
                        .color(planner.color)
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
        // Check if ANY planner has a result
        if self.planners.iter().any(|p| p.result.is_some()) && !self.animation_complete {
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
            self.run_all_planners();
        }
    }

    fn reset_all(&mut self) {
        // Clear obstacles
        self.grid = Grid::new(self.grid_width, self.grid_height, self.grid_resolution);
        self.continuous_obstacles.clear();
        // Clear start/end and result
        self.clear();
    }
}

impl Draw for PathPlanning {
    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        if self.mode == EnvironmentMode::Grid {
            self.draw_grid(plot_ui);
            self.draw_grid_obstacles(plot_ui);
        } else {
             let (min_x, min_y, max_x, max_y) = self.grid.world_bounds();
             // Draw boundary box
             let points = PlotPoints::new(vec![
                 [min_x as f64, min_y as f64],
                 [max_x as f64, min_y as f64],
                 [max_x as f64, max_y as f64],
                 [min_x as f64, max_y as f64],
                 [min_x as f64, min_y as f64],
             ]);
             plot_ui.line(Line::new("", points).color(Color32::from_gray(100)).width(1.0));
             
             self.draw_continuous_obstacles(plot_ui);
        }

        // Draw planners
        for planner in &self.planners {
            if planner.algorithm == Algorithm::Rrt {
                self.draw_tree(plot_ui, planner);
            } else if self.mode == EnvironmentMode::Grid {
                 self.draw_visited(plot_ui, planner);
            }
            self.draw_path(plot_ui, planner);
        }

        self.draw_start(plot_ui);
        self.draw_goal(plot_ui);
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        let mut keep = true;

        ui.push_id(self.id, |ui| {
            ui.group(|ui| {
                ui.set_width(240.0); // Widen for better layout
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading(format!("Planner {}", self.id));
                        if ui.small_button("x").clicked() {
                            keep = false;
                        }
                    });

                ui.separator();
                
                // We need to collect IDs to remove to avoid mutable borrow error
                // let mut remove_ids = Vec::new(); // Unused
                let mut rerun_indices = Vec::new();
                
                for (idx, planner) in self.planners.iter_mut().enumerate() {
                    ui.push_id(planner.id, |ui| {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                // Colored indicator
                                let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
                                ui.painter().circle_filled(rect.center(), 4.0, planner.color);
                                
                                // Algorithm Selector
                                ComboBox::from_id_salt("algo")
                                    .selected_text(planner.algorithm.label())
                                    .show_ui(ui, |ui| {
                                        let available_algos = match self.mode {
                                            EnvironmentMode::Grid => vec![Algorithm::AStar, Algorithm::ThetaStar, Algorithm::Dijkstra],
                                            EnvironmentMode::Continuous => vec![Algorithm::Rrt, Algorithm::ThetaStar],
                                        };
                                        for algo in available_algos {
                                            if ui.selectable_value(&mut planner.algorithm, algo, algo.label()).clicked() {
                                                // If start/goal set, re-run?
                                                planner.result = None;
                                                rerun_indices.push(idx);
                                            }
                                        }
                                    });
                            });
                            
                            // Planner specific settings
                            if planner.algorithm == Algorithm::Rrt {
                                ui.collapsing("Settings", |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Expand:");
                                        ui.add(DragValue::new(&mut planner.rrt_expand_dist).range(0.1..=5.0).speed(0.1));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Goal Bias:");
                                        ui.add(DragValue::new(&mut planner.rrt_goal_sample_rate).range(0.0..=1.0).speed(0.01));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Max Iter:");
                                        ui.add(DragValue::new(&mut planner.rrt_max_iter).range(100..=5000));
                                    });
                                });
                            }
                            
                            let show_label = if planner.algorithm == Algorithm::Rrt { "Show Tree" } else { "Show Visited" };
                            if self.mode == EnvironmentMode::Grid || planner.algorithm == Algorithm::Rrt {
                                ui.checkbox(&mut planner.show_visited, show_label);
                            }
                            
                            // Results
                            if let Some(res) = &planner.result {
                                if res.success {
                                    ui.label(format!("Path: {:.2}", res.path_length));
                                    if res.euclidean_distance > 0.0 {
                                        ui.label(format!("Ratio: {:.2}", res.path_length / res.euclidean_distance));
                                    }
                                } else {
                                    ui.label("No path found");
                                }
                                ui.label(format!("Iter: {}", res.iterations));
                                ui.label(format!("Time: {:.2?}", res.execution_time));
                            }
                        });
                    });
                }
                
                for idx in rerun_indices {
                    if self.start.is_some() && self.goal.is_some() {
                        self.run_single_planner(idx);
                        // Also set state to showing result to trigger animation if not already
                        self.state = PlanningState::ShowingResult;
                        self.animation_complete = false;
                        self.animation_progress = 0.0;
                    }
                }
                
                ui.separator();

                // Status
                let status_text = match self.state {
                    PlanningState::WaitingForStart => "Click to set start point",
                    PlanningState::WaitingForGoal => "Click to set goal point",
                    PlanningState::ShowingResult => "Showing results",
                };
                ui.label(format!("Status: {}", status_text));
                
                if self.state == PlanningState::ShowingResult {
                    if ui.button("Re-run All").clicked() {
                        self.run_all_planners();
                    }
                }

                }); // end vertical
            }); // end group
        }); // end push_id

        keep
    }
}
