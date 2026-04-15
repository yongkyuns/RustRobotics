//! Guided help-tour definitions and overlay rendering.
//!
//! The simulator intentionally includes a lightweight walkthrough layer because
//! the application mixes several distinct robotics domains behind one shared
//! shell. This module owns:
//!
//! - the ordered help steps per mode
//! - which UI region each step should highlight
//! - popup progression state
//! - dimming / highlight rendering
use super::{SimMode, Simulator};
use egui::*;

/// Logical UI region that a help step can point to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HelpTarget {
    ModeSelector,
    Controls,
    Options,
    Scene,
}

/// One instructional step in the guided walkthrough.
#[derive(Debug, Clone, Copy)]
struct HelpStep {
    title: &'static str,
    body: &'static str,
    target: Option<HelpTarget>,
}

impl Simulator {
    /// Returns the full help-step sequence for the currently selected mode.
    fn help_steps(&self) -> &'static [HelpStep] {
        const INVERTED_PENDULUM_STEPS: [HelpStep; 9] = [
            HelpStep {
                title: "Mode Selector",
                body: "Choose which simulator is active here. Switching modes resets the local timebase and opens the relevant controls.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Play And Reset",
                body: "Use Play or Pause to freeze the current response, Restart to re-run with the same parameters, and Reset All to return this mode to its defaults. That makes it easy to test one change at a time and replay from the same starting condition.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed And Noise",
                body: "Simulation speed changes how quickly simulated time advances. The Noise toggle injects sensor noise and actuator noise. As you raise it, expect noisier measurements, rougher control effort, and more visible jitter in sensitive controllers.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Comparing Pendulums",
                body: "Add Pendulum creates another copy of the same task so you can compare controllers or parameter choices side by side. This is the easiest way to build intuition, because you can keep one baseline untouched while you experiment on the other.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Physical Model",
                body: "The Cart section changes the plant itself. Increasing Beam Length slows the pendulum and gives the controller more leverage. Increasing Ball Mass makes the top heavier and usually harder to catch after a disturbance. Increasing Cart Mass makes lateral correction more sluggish for the same controller.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Controller Choice",
                body: "The Controller selector changes the control strategy, not just the tuning. LQR is a clean linear stabilizer near upright, PID reacts directly to angle error, and MPC looks ahead but can cost more computation. If behavior changes a lot after a plant change, switching controllers helps show whether the limitation is the tuning or the control law itself.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Tuning Intuition",
                body: "For LQR, increasing the Rod Angle or Rod Angular Velocity weights makes the controller fight tipping more aggressively. Increasing the Lateral Position or Velocity weights makes it care more about keeping the cart centered. Increasing the Control Input weight makes it more conservative. For PID, higher P reacts harder to tilt, higher I removes steady bias but can add overshoot, and higher D damps oscillation.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Scene And Plot",
                body: "Watch the cart motion and rod angle here, then open the Signal Plot to compare one channel across multiple pendulums. If a heavier ball causes larger swings or slower recovery, you should see that immediately both in the scene and in the angle or control traces.",
                target: Some(HelpTarget::Scene),
            },
            HelpStep {
                title: "Scene Overview",
                body: "This is the live pendulum scene. Use it to connect parameter changes to visible behavior: quicker recovery, more oscillation, slower cart motion, or loss of stability.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const LOCALIZATION_STEPS: [HelpStep; 7] = [
            HelpStep {
                title: "Mode Selector",
                body: "Use this row to switch between localization, SLAM, pendulum, robot, and planning demos.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback Controls",
                body: "Play or Pause freezes the estimation process, Restart replays the same setup, and Add Vehicle creates another estimator so you can compare settings side by side.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed",
                body: "Simulation speed changes how fast the vehicle and filter move through time. Slowing the demo down is useful when you want to watch drift build up or see how the filter recovers after a turn.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Drive Mode",
                body: "Kinematic mode uses simple commanded velocity and yaw rate, so it is easier to reason about. Dynamic mode adds tire and body dynamics, so the vehicle can slip or lag. Dynamic mode is where model mismatch becomes more visible.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Motion And Vehicle Knobs",
                body: "In kinematic mode, increasing speed makes dead-reckoning drift accumulate faster, and increasing yaw rate creates tighter turns that are harder to estimate cleanly. In dynamic mode, mass, axle distances, and tire parameters change how quickly the car rotates and settles, which directly affects how hard localization becomes.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "What To Compare",
                body: "Compare the true pose, dead-reckoning path, and particle-filter estimate. If the estimate stays close to truth while dead reckoning drifts away, the filter is using observations effectively. If all curves separate, the motion or sensor assumptions are too weak for the chosen settings.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Localization Scene",
                body: "Drive and inspect the localization behavior in this scene. Arrow keys control dynamic vehicle modes. The most informative moments are hard turns and higher-speed segments, where model error is easiest to see.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const MUJOCO_STEPS: [HelpStep; 6] = [
            HelpStep {
                title: "Mode Selector",
                body: "Use this selector to enter the Robot mode or return to the other simulators. Each mode focuses on a different robotics demo.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback Controls",
                body: "Play or Pause freezes the live robot, Restart resets the current run, and Reset All restores the mode defaults. Use these when you want a clean comparison after changing the target or switching robots.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed",
                body: "Simulation speed changes how quickly robot time advances. Slowing it down is useful when you want to study gait timing, recovery steps, or how the controller responds after you move the target.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Robot Panel",
                body: "Choose which robot to run here. Different robots expose different control behaviors and policy styles, so switching robots is really switching the task and controller together. Read the description text as a clue for what motion you should expect to see.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Setpoint Interaction",
                body: "For robots that support setpoint control, dragging the red ball changes the commanded target in the scene. Move it farther away and expect the controller to lean, step, or reorient more aggressively to chase it. Small target changes should produce small posture corrections; large jumps expose controller limits much more clearly.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Viewport Note",
                body: "The viewport is where you judge stability and tracking quality. Watch whether the robot approaches the target smoothly, oscillates around it, or fails to recover after a disturbance. Those visual cues are often more informative than a single scalar metric.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const PATH_PLANNING_STEPS: [HelpStep; 7] = [
            HelpStep {
                title: "Mode Selector",
                body: "Switch between planners and the other demos from this selector.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback And Comparison",
                body: "Use Restart to replay the same planning scene, Add Planner to create another planner panel, and Show Graph if you want plotted data later. The main value here is comparing multiple planners in the same environment.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed",
                body: "Simulation speed mainly affects how quickly animations and repeated replans advance. Slowing it down is useful when you want to watch visited cells or RRT growth more carefully.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Environment Controls",
                body: "Grid mode gives discrete cells and naturally fits A*, Dijkstra, and Theta*. Continuous mode gives free space with circular obstacles and is better for RRT. Changing grid size or resolution changes both difficulty and cost: finer grids can produce cleaner paths, but they usually make the search do more work.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Planner Choice",
                body: "A* is a strong baseline when you want efficient shortest-path search with a heuristic. Dijkstra ignores the heuristic and usually explores more broadly. Theta* can cut across line of sight and often produces shorter-looking paths. RRT is sampling-based, so it is better matched to continuous spaces than dense grids.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Planner Knobs",
                body: "For RRT, larger Expand values make the tree jump farther each step, which can explore faster but miss narrow passages. Higher Goal Bias pulls the tree toward the goal more often, which can speed up easy scenes but reduce exploration. Max Iter controls how long the planner is allowed to keep searching before it gives up.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Planning Scene",
                body: "Set a green start and a red goal in the scene, then compare how each planner moves through the same environment. If you enable visited cells or the RRT tree, you can study not just the final path but the search strategy that produced it.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const SLAM_STEPS: [HelpStep; 7] = [
            HelpStep {
                title: "Mode Selector",
                body: "Use the selector to move between SLAM and the other simulations.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback Controls",
                body: "Play or Pause freezes the current trajectory, Restart replays the run from the same map, and simulation speed changes how quickly observations accumulate. Slowing the demo down makes it easier to see when estimates start drifting apart.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Algorithm Toggles",
                body: "Enable EKF, Graph SLAM, or both. This lets you compare filtering versus optimization on the same sensor stream. If one estimate drifts less or recovers better after revisiting a place, that difference is exactly the lesson this demo is meant to show.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Graph SLAM Options",
                body: "Robust kernels reduce the influence of bad measurements, Sparse solves the optimization more efficiently, and Loop Closure allows the estimate to use revisited places to correct accumulated drift. When loop closure is on, expect the map and trajectory to tighten up after revisits.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Motion And Landmarks",
                body: "Increasing linear or angular speed makes localization harder because error accumulates faster between observations. The landmark count controls how much information the world provides; more landmarks usually give the estimators more chances to correct drift.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Display And Error",
                body: "Use the display toggles to show covariance, observations, dead reckoning, and landmarks. The error box is the quickest summary of estimator quality: if EKF or Graph error stays below dead reckoning, the estimator is adding useful information instead of just replaying odometry.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "SLAM Scene",
                body: "Drive the robot in the scene and compare how the map and pose estimate evolve over time. The most instructive moments are turns, revisits, and longer runs where drift has time to build up.",
                target: Some(HelpTarget::Scene),
            },
        ];

        match self.mode {
            SimMode::InvertedPendulum => &INVERTED_PENDULUM_STEPS,
            SimMode::Localization => &LOCALIZATION_STEPS,
            SimMode::Mujoco => &MUJOCO_STEPS,
            SimMode::PathPlanning => &PATH_PLANNING_STEPS,
            SimMode::Slam => &SLAM_STEPS,
        }
    }

    /// Returns the active help sequence after removing already-explained shared
    /// introduction steps.
    fn active_help_steps(&self) -> Vec<HelpStep> {
        self.help_steps()
            .iter()
            .copied()
            .filter(|step| {
                !self.help_state.shared_help_intro_completed
                    || !matches!(
                        step.target,
                        Some(HelpTarget::ModeSelector | HelpTarget::Controls)
                    )
            })
            .collect()
    }

    /// Rewinds the current guided tour to its first step.
    pub(super) fn reset_help_tour(&mut self) {
        self.help_state.help_step_index = 0;
    }

    /// Resolves the screen rectangle associated with a logical help target.
    fn help_target_rect(&self, target: HelpTarget) -> Option<Rect> {
        match target {
            HelpTarget::ModeSelector => self.help_state.help_mode_selector_rect,
            HelpTarget::Controls => self.help_state.help_controls_rect,
            HelpTarget::Options => self.help_state.help_options_rect,
            HelpTarget::Scene => self.help_state.help_scene_rect,
        }
    }

    /// Paints the current highlight overlay around the active help target.
    pub(super) fn paint_help_highlight(&self, ui: &Ui) {
        if !self.help_state.show_help_popup {
            return;
        }

        let steps = self.active_help_steps();
        let Some(step) = steps.get(self.help_state.help_step_index) else {
            return;
        };
        let Some(target) = step.target else {
            return;
        };
        let Some(rect) = self.help_target_rect(target) else {
            return;
        };

        let pulse = (ui.ctx().input(|i| i.time) as f32 * 2.5).sin().abs();
        let highlight_rect = rect.expand(4.0);
        let stroke_color = Color32::from_rgb(255, 210, 64);
        let fill_color = stroke_color.linear_multiply(0.08 + pulse * 0.05);
        let painter = ui.ctx().layer_painter(LayerId::new(
            Order::Background,
            Id::new("simulation_help_highlight"),
        ));
        painter.rect_filled(highlight_rect, 8.0, fill_color);
        painter.rect_stroke(
            highlight_rect,
            8.0,
            Stroke::new(1.5 + pulse * 0.5, stroke_color),
            StrokeKind::Outside,
        );
    }

    pub(super) fn show_help_windows(&mut self, ctx: &Context) -> Vec<Rect> {
        let mut overlay_occlusions = Vec::new();

        let mut start_tutorial = false;
        let mut skip_tutorial = false;
        if self.help_state.show_tutorial_prompt {
            let mut show_tutorial_prompt = self.help_state.show_tutorial_prompt;
            if let Some(window) = Window::new("Welcome")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .open(&mut show_tutorial_prompt)
                .show(ctx, |ui| {
                    ui.heading("Welcome to Rust Robotics");
                    ui.separator();
                    ui.label("This app includes several robotics demos, including pendulum control, localization, path planning, SLAM, and robot simulation.");
                    ui.add_space(6.0);
                    ui.label("Would you like a short guided tutorial when you open each simulator for the first time?");
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Start Tutorial").clicked() {
                            start_tutorial = true;
                        }
                        if ui.button("Skip").clicked() {
                            skip_tutorial = true;
                        }
                    });
                })
            {
                overlay_occlusions.push(window.response.rect);
            }
            self.help_state.show_tutorial_prompt = show_tutorial_prompt;
        }
        if start_tutorial {
            self.help_state.tutorial_enabled = Some(true);
            self.help_state.show_tutorial_prompt = false;
            self.help_state.visited_modes.insert(self.mode);
            self.help_state.show_help_popup = true;
            self.reset_help_tour();
        }
        if skip_tutorial
            || !self.help_state.show_tutorial_prompt && self.help_state.tutorial_enabled.is_none()
        {
            self.help_state.tutorial_enabled = Some(false);
            self.help_state.show_tutorial_prompt = false;
            self.help_state.show_help_popup = false;
        }

        let mut close_popup = false;
        if self.help_state.show_help_popup {
            let steps = self.active_help_steps();
            let step_index = self
                .help_state
                .help_step_index
                .min(steps.len().saturating_sub(1));
            let step = steps[step_index];
            let mode = self.mode;
            let mut next_help_step = step_index;
            let mut show_help_popup = self.help_state.show_help_popup;
            if let Some(window) = Window::new("Simulation Help")
                .collapsible(false)
                .resizable(false)
                .fixed_size(vec2(560.0, 320.0))
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .open(&mut show_help_popup)
                .show(ctx, |ui| {
                    ui.set_min_size(vec2(520.0, 280.0));
                    ui.heading(format!("{} Instructions", mode.label()));
                    ui.separator();
                    ScrollArea::vertical()
                        .max_height(210.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!("Step {} of {}", step_index + 1, steps.len()))
                                    .small()
                                    .weak(),
                            );
                            ui.heading(step.title);
                            ui.label(step.body);

                            if mode == SimMode::Localization || mode == SimMode::Slam {
                                ui.add_space(6.0);
                                ui.label("Keyboard: arrows drive, `Space` pauses, `Enter` restarts.");
                            }
                            if mode == SimMode::PathPlanning && step_index == steps.len() - 1 {
                                ui.add_space(6.0);
                                ui.label("Left-click sets start/goal. Right-click paints or places obstacles depending on environment mode.");
                            }
                            if mode == SimMode::Mujoco && step_index == steps.len() - 1 {
                                ui.add_space(6.0);
                                ui.label("Try moving the setpoint a short distance first, then make larger moves to see how the controller transitions from fine tracking to more aggressive recovery.");
                            }
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        let button_size = vec2(96.0, 28.0);

                        if ui
                            .add_enabled(step_index > 0, Button::new("Back").min_size(button_size))
                            .clicked()
                        {
                            next_help_step = step_index.saturating_sub(1);
                        }

                        let forward_label = if step_index + 1 < steps.len() {
                            "Next"
                        } else {
                            "Done"
                        };
                        if ui
                            .add(Button::new(forward_label).min_size(button_size))
                            .clicked()
                        {
                            if step_index + 1 < steps.len() {
                                next_help_step = step_index + 1;
                            } else {
                                close_popup = true;
                            }
                        }

                        ui.add_space((ui.available_width() - button_size.x).max(0.0));

                        if ui
                            .add(Button::new("Close Tour").min_size(button_size))
                            .clicked()
                        {
                            close_popup = true;
                        }
                    });
                })
            {
                overlay_occlusions.push(window.response.rect);
            }
            self.help_state.show_help_popup = show_help_popup;
            self.help_state.help_step_index = next_help_step;
        }
        if close_popup {
            self.help_state.show_help_popup = false;
            self.help_state.shared_help_intro_completed = true;
        }

        overlay_occlusions
    }
}
