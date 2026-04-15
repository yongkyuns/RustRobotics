use egui::Ui;

pub(super) fn show_stacked_layout<T>(
    ui: &mut Ui,
    state: &mut T,
    mut controls: impl FnMut(&mut T, &mut Ui),
    mut viewport: impl FnMut(&mut T, &mut Ui),
) {
    ui.vertical(|ui| {
        controls(state, ui);
        ui.separator();
        viewport(state, ui);
    });
}
