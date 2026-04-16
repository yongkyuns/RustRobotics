use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Stroke, Style, TextStyle, Theme, Visuals,
};

pub fn install(ctx: &egui::Context) {
    ctx.set_theme(Theme::Dark);
    ctx.set_style(build_style(UiDensity::Comfortable));
    ctx.set_visuals(build_visuals());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiDensity {
    Comfortable,
    Compact,
}

pub fn apply_density(ctx: &egui::Context, density: UiDensity) {
    ctx.set_style(build_style(density));
    ctx.set_visuals(build_visuals());
}

fn build_style(density: UiDensity) -> Style {
    let mut style = Style::default();
    let scale = match density {
        UiDensity::Comfortable => 1.0,
        UiDensity::Compact => 0.82,
    };

    style.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(22.0 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Name("Heading2".into()),
            FontId::new(20.0 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Body,
            FontId::new(15.5 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(14.0 * scale, FontFamily::Monospace),
        ),
        (
            TextStyle::Button,
            FontId::new(15.0 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(13.0 * scale, FontFamily::Proportional),
        ),
    ]
    .into();

    style.spacing.item_spacing = egui::vec2(10.0 * scale, 10.0 * scale);
    style.spacing.button_padding = egui::vec2(12.0 * scale, 8.0 * scale);
    style.spacing.menu_margin = egui::Margin::same((10.0 * scale).round() as i8);
    style.spacing.window_margin = egui::Margin::same((8.0 * scale).round() as i8);
    style.spacing.indent = 18.0 * scale;
    style.spacing.combo_width = 148.0 * scale;
    style.spacing.slider_width = 180.0 * scale;
    style.spacing.interact_size = egui::vec2(42.0 * scale, 28.0 * scale);
    style.visuals = build_visuals();

    style
}

fn build_visuals() -> Visuals {
    let mut visuals = Visuals::dark();

    let accent = Color32::from_rgb(230, 122, 76);
    let accent_soft = Color32::from_rgb(198, 97, 59);
    let panel = Color32::from_rgb(17, 21, 28);
    let panel_alt = Color32::from_rgb(24, 29, 38);
    let panel_strong = Color32::from_rgb(31, 37, 48);
    let text = Color32::from_rgb(233, 238, 245);
    let text_muted = Color32::from_rgb(153, 165, 180);
    let border = Color32::from_rgb(58, 69, 84);

    visuals.override_text_color = Some(text);
    visuals.panel_fill = panel;
    visuals.window_fill = panel_alt;
    visuals.faint_bg_color = panel_alt;
    visuals.extreme_bg_color = panel_strong;
    visuals.code_bg_color = Color32::from_rgb(20, 25, 33);
    visuals.warn_fg_color = Color32::from_rgb(255, 196, 107);
    visuals.error_fg_color = Color32::from_rgb(255, 108, 108);
    visuals.hyperlink_color = Color32::from_rgb(120, 188, 255);
    visuals.selection.bg_fill = accent;
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(255, 221, 209));

    visuals.window_corner_radius = CornerRadius::same(16);
    visuals.menu_corner_radius = CornerRadius::same(12);
    visuals.window_stroke = Stroke::new(1.0, border);
    visuals.widgets.noninteractive.bg_fill = panel_alt;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text_muted);

    visuals.widgets.inactive.bg_fill = panel_strong;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, text);

    visuals.widgets.hovered.bg_fill = Color32::from_rgb(42, 50, 64);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, text);
    visuals.widgets.hovered.expansion = 0.0;

    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);

    visuals.widgets.open.bg_fill = panel_strong;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, accent_soft);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, text);

    visuals
}
