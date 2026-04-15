use egui::{Pos2, Response, Ui, Vec2};

pub(super) struct ViewportInteraction {
    pub reset_view: bool,
    pub setpoint_world: Option<[f32; 3]>,
    pub primary_orbit_delta: Option<Vec2>,
    pub secondary_orbit_delta: Option<Vec2>,
    pub zoom_factor: Option<f32>,
}

pub(super) fn gather_viewport_interaction(
    uses_setpoint_ball: bool,
    ui: &Ui,
    response: &Response,
    intersect_plane_z: impl Fn(Pos2) -> Option<[f32; 3]>,
) -> ViewportInteraction {
    let mut out = ViewportInteraction {
        reset_view: response.double_clicked(),
        setpoint_world: None,
        primary_orbit_delta: None,
        secondary_orbit_delta: None,
        zoom_factor: None,
    };

    if uses_setpoint_ball {
        if response.clicked_by(egui::PointerButton::Primary) {
            if let Some(pointer) = response.interact_pointer_pos() {
                out.setpoint_world = intersect_plane_z(pointer);
            }
        }
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pointer) = response.interact_pointer_pos() {
                out.setpoint_world = intersect_plane_z(pointer);
            }
        }
    } else if response.dragged_by(egui::PointerButton::Primary) {
        out.primary_orbit_delta = Some(response.drag_delta());
    }

    if response.dragged_by(egui::PointerButton::Secondary) {
        out.secondary_orbit_delta = Some(response.drag_delta());
    }

    if response.hovered() {
        let scroll_y = ui.input(|input| input.raw_scroll_delta.y);
        if scroll_y.abs() > f32::EPSILON {
            out.zoom_factor = Some((1.0 - scroll_y * 0.0015).clamp(0.8, 1.25));
        }
    }

    out
}
