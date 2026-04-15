use egui::Ui;

pub(super) struct SharedPanelOutcome {
    pub selected_index: usize,
    pub reset_view: bool,
}

pub(super) fn show_shared_panel(
    ui: &mut Ui,
    selected_index: usize,
    robot_labels: &[&str],
    robot_description: &str,
    mut details: impl FnMut(&mut Ui),
) -> SharedPanelOutcome {
    let mut selected_index = selected_index.min(robot_labels.len().saturating_sub(1));

    ui.heading("Robot");
    ui.label("Robot simulation running inside RustRobotics.");
    egui::ComboBox::from_label("Robot")
        .selected_text(
            robot_labels
                .get(selected_index)
                .copied()
                .unwrap_or("Unknown"),
        )
        .show_ui(ui, |ui| {
            for (index, label) in robot_labels.iter().copied().enumerate() {
                ui.selectable_value(&mut selected_index, index, label);
            }
        });
    ui.label(robot_description);
    ui.label("Control: drag the red setpoint ball in the viewport.");
    let reset_view = ui.button("Reset view").clicked();
    ui.separator();
    details(ui);

    SharedPanelOutcome {
        selected_index,
        reset_view,
    }
}
