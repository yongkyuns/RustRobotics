#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct GlVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl GlVertex {
    pub fn world(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            color,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SharedGeomSnapshot {
    pub type_id: i32,
    pub data_id: i32,
    pub size: [f32; 3],
    pub rgba: [f32; 4],
    pub pos: [f32; 3],
    pub mat: [f32; 9],
}

pub(super) const GEOM_PLANE: i32 = 0;
pub(super) const GEOM_SPHERE: i32 = 2;
pub(super) const GEOM_CAPSULE: i32 = 3;
pub(super) const GEOM_CYLINDER: i32 = 5;
pub(super) const GEOM_BOX: i32 = 6;
pub(super) const GEOM_MESH: i32 = 7;
pub(super) const GEOM_LINE: i32 = 9;

pub(super) fn append_grid_lines(lines: &mut Vec<GlVertex>) {
    let color = [0.22, 0.28, 0.34, 1.0];
    for i in -16..=16 {
        let offset = i as f32 * 0.25;
        push_line(lines, [-4.0, offset, 0.0], [4.0, offset, 0.0], color);
        push_line(lines, [offset, -4.0, 0.0], [offset, 4.0, 0.0], color);
    }
}

pub(super) fn append_primitive_geom(
    triangles: &mut Vec<GlVertex>,
    lines: &mut Vec<GlVertex>,
    geom: &SharedGeomSnapshot,
    diagnostic_colors: bool,
) {
    match geom.type_id {
        GEOM_PLANE | GEOM_MESH => {}
        GEOM_SPHERE => append_sphere_geom(triangles, geom, 12, 20, diagnostic_colors),
        GEOM_CAPSULE => append_capsule_geom(triangles, geom, 10, 18, diagnostic_colors),
        GEOM_CYLINDER => append_cylinder_geom(triangles, geom, 18, diagnostic_colors),
        GEOM_BOX => append_box_geom(triangles, geom, diagnostic_colors),
        GEOM_LINE => append_line_geom(lines, geom, diagnostic_colors),
        _ => {}
    }
}

pub(super) fn append_world_sphere(
    triangles: &mut Vec<GlVertex>,
    pos: [f32; 3],
    radius: f32,
    color: [f32; 4],
) {
    let geom = SharedGeomSnapshot {
        type_id: GEOM_SPHERE,
        data_id: -1,
        size: [radius, radius, radius],
        rgba: color,
        pos,
        mat: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };
    append_sphere_geom(triangles, &geom, 10, 16, false);
}

pub(super) fn display_geom_color(geom: &SharedGeomSnapshot, diagnostic_colors: bool) -> [f32; 4] {
    if diagnostic_colors {
        diagnostic_geom_color(geom)
    } else {
        [
            geom.rgba[0].clamp(0.0, 1.0),
            geom.rgba[1].clamp(0.0, 1.0),
            geom.rgba[2].clamp(0.0, 1.0),
            1.0,
        ]
    }
}

pub(super) fn transform_geom_point(geom: &SharedGeomSnapshot, local: [f32; 3]) -> [f32; 3] {
    [
        geom.pos[0] + geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.pos[1] + geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.pos[2] + geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ]
}

pub(super) fn transform_geom_vector(geom: &SharedGeomSnapshot, local: [f32; 3]) -> [f32; 3] {
    normalize3([
        geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ])
}

pub(super) fn geom_axes(geom: &SharedGeomSnapshot) -> [[f32; 3]; 3] {
    [
        [geom.mat[0], geom.mat[3], geom.mat[6]],
        [geom.mat[1], geom.mat[4], geom.mat[7]],
        [geom.mat[2], geom.mat[5], geom.mat[8]],
    ]
}

pub(super) fn geom_model_matrix(geom: &SharedGeomSnapshot) -> [[f32; 4]; 4] {
    [
        [geom.mat[0], geom.mat[3], geom.mat[6], 0.0],
        [geom.mat[1], geom.mat[4], geom.mat[7], 0.0],
        [geom.mat[2], geom.mat[5], geom.mat[8], 0.0],
        [geom.pos[0], geom.pos[1], geom.pos[2], 1.0],
    ]
}

fn append_line_geom(lines: &mut Vec<GlVertex>, geom: &SharedGeomSnapshot, diagnostic_colors: bool) {
    let axes = geom_axes(geom);
    let half = scale3(axes[2], geom.size[1].max(0.02));
    let a = [
        geom.pos[0] - half[0],
        geom.pos[1] - half[1],
        geom.pos[2] - half[2],
    ];
    let b = [
        geom.pos[0] + half[0],
        geom.pos[1] + half[1],
        geom.pos[2] + half[2],
    ];
    push_line(lines, a, b, display_geom_color(geom, diagnostic_colors));
}

fn append_box_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &SharedGeomSnapshot,
    diagnostic_colors: bool,
) {
    let axes = geom_axes(geom);
    let sx = geom.size[0];
    let sy = geom.size[1];
    let sz = geom.size[2];
    let corners = [
        transform_geom_point(geom, [sx, sy, sz]),
        transform_geom_point(geom, [sx, sy, -sz]),
        transform_geom_point(geom, [sx, -sy, sz]),
        transform_geom_point(geom, [sx, -sy, -sz]),
        transform_geom_point(geom, [-sx, sy, sz]),
        transform_geom_point(geom, [-sx, sy, -sz]),
        transform_geom_point(geom, [-sx, -sy, sz]),
        transform_geom_point(geom, [-sx, -sy, -sz]),
    ];
    let color = display_geom_color(geom, diagnostic_colors);
    let faces = [
        ([0, 2, 3, 1], axes[0]),
        ([4, 5, 7, 6], scale3(axes[0], -1.0)),
        ([0, 1, 5, 4], axes[1]),
        ([2, 6, 7, 3], scale3(axes[1], -1.0)),
        ([0, 4, 6, 2], axes[2]),
        ([1, 3, 7, 5], scale3(axes[2], -1.0)),
    ];
    for (indices, normal) in faces {
        push_triangle(
            triangles,
            corners[indices[0]],
            corners[indices[1]],
            corners[indices[2]],
            normalize3(normal),
            color,
        );
        push_triangle(
            triangles,
            corners[indices[0]],
            corners[indices[2]],
            corners[indices[3]],
            normalize3(normal),
            color,
        );
    }
}

fn append_cylinder_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &SharedGeomSnapshot,
    segments: usize,
    diagnostic_colors: bool,
) {
    let color = display_geom_color(geom, diagnostic_colors);
    let half = geom.size[1].max(0.01);
    let radius = geom.size[0].max(0.01);
    for i in 0..segments {
        let a0 = i as f32 / segments as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
        let p0 = [radius * a0.cos(), radius * a0.sin(), -half];
        let p1 = [radius * a1.cos(), radius * a1.sin(), -half];
        let p2 = [radius * a1.cos(), radius * a1.sin(), half];
        let p3 = [radius * a0.cos(), radius * a0.sin(), half];

        let w0 = transform_geom_point(geom, p0);
        let w1 = transform_geom_point(geom, p1);
        let w2 = transform_geom_point(geom, p2);
        let w3 = transform_geom_point(geom, p3);
        let n0 = transform_geom_vector(geom, normalize3([a0.cos(), a0.sin(), 0.0]));
        let n1 = transform_geom_vector(geom, normalize3([a1.cos(), a1.sin(), 0.0]));
        let n_mid = normalize3(add3(n0, n1));

        push_triangle(triangles, w0, w1, w2, n_mid, color);
        push_triangle(triangles, w0, w2, w3, n_mid, color);

        let top_center = transform_geom_point(geom, [0.0, 0.0, half]);
        let bottom_center = transform_geom_point(geom, [0.0, 0.0, -half]);
        push_triangle(
            triangles,
            top_center,
            w3,
            w2,
            transform_geom_vector(geom, [0.0, 0.0, 1.0]),
            color,
        );
        push_triangle(
            triangles,
            bottom_center,
            w1,
            w0,
            transform_geom_vector(geom, [0.0, 0.0, -1.0]),
            color,
        );
    }
}

fn append_capsule_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &SharedGeomSnapshot,
    hemi_rings: usize,
    segments: usize,
    diagnostic_colors: bool,
) {
    append_cylinder_geom(triangles, geom, segments, diagnostic_colors);
    let radius = geom.size[0].max(0.01);
    let half = geom.size[1].max(0.01);
    let color = display_geom_color(geom, diagnostic_colors);
    for hemisphere in [-1.0f32, 1.0] {
        for ring in 0..hemi_rings {
            let v0 = ring as f32 / hemi_rings as f32 * std::f32::consts::FRAC_PI_2;
            let v1 = (ring + 1) as f32 / hemi_rings as f32 * std::f32::consts::FRAC_PI_2;
            let z0 = hemisphere * (half + radius * v0.sin());
            let z1 = hemisphere * (half + radius * v1.sin());
            let r0 = radius * v0.cos();
            let r1 = radius * v1.cos();

            for seg in 0..segments {
                let a0 = seg as f32 / segments as f32 * std::f32::consts::TAU;
                let a1 = (seg + 1) as f32 / segments as f32 * std::f32::consts::TAU;

                let p00 = [r0 * a0.cos(), r0 * a0.sin(), z0];
                let p01 = [r0 * a1.cos(), r0 * a1.sin(), z0];
                let p10 = [r1 * a0.cos(), r1 * a0.sin(), z1];
                let p11 = [r1 * a1.cos(), r1 * a1.sin(), z1];

                let n00 = transform_geom_vector(
                    geom,
                    normalize3([p00[0], p00[1], hemisphere * (p00[2] - hemisphere * half)]),
                );
                let n01 = transform_geom_vector(
                    geom,
                    normalize3([p01[0], p01[1], hemisphere * (p01[2] - hemisphere * half)]),
                );
                let n10 = transform_geom_vector(
                    geom,
                    normalize3([p10[0], p10[1], hemisphere * (p10[2] - hemisphere * half)]),
                );
                let n11 = transform_geom_vector(
                    geom,
                    normalize3([p11[0], p11[1], hemisphere * (p11[2] - hemisphere * half)]),
                );

                push_triangle(
                    triangles,
                    transform_geom_point(geom, p00),
                    transform_geom_point(geom, p01),
                    transform_geom_point(geom, p11),
                    normalize3(add3(add3(n00, n01), n11)),
                    color,
                );
                push_triangle(
                    triangles,
                    transform_geom_point(geom, p00),
                    transform_geom_point(geom, p11),
                    transform_geom_point(geom, p10),
                    normalize3(add3(add3(n00, n11), n10)),
                    color,
                );
            }
        }
    }
}

fn append_sphere_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &SharedGeomSnapshot,
    rings: usize,
    segments: usize,
    diagnostic_colors: bool,
) {
    let radius = geom.size[0].max(0.01);
    let color = display_geom_color(geom, diagnostic_colors);
    for ring in 0..rings {
        let v0 = ring as f32 / rings as f32 * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let v1 =
            (ring + 1) as f32 / rings as f32 * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let z0 = radius * v0.sin();
        let z1 = radius * v1.sin();
        let r0 = radius * v0.cos();
        let r1 = radius * v1.cos();

        for seg in 0..segments {
            let a0 = seg as f32 / segments as f32 * std::f32::consts::TAU;
            let a1 = (seg + 1) as f32 / segments as f32 * std::f32::consts::TAU;
            let p00 = [r0 * a0.cos(), r0 * a0.sin(), z0];
            let p01 = [r0 * a1.cos(), r0 * a1.sin(), z0];
            let p10 = [r1 * a0.cos(), r1 * a0.sin(), z1];
            let p11 = [r1 * a1.cos(), r1 * a1.sin(), z1];

            push_triangle(
                triangles,
                transform_geom_point(geom, p00),
                transform_geom_point(geom, p01),
                transform_geom_point(geom, p11),
                normalize3(transform_geom_vector(
                    geom,
                    normalize3(add3(add3(p00, p01), p11)),
                )),
                color,
            );
            push_triangle(
                triangles,
                transform_geom_point(geom, p00),
                transform_geom_point(geom, p11),
                transform_geom_point(geom, p10),
                normalize3(transform_geom_vector(
                    geom,
                    normalize3(add3(add3(p00, p11), p10)),
                )),
                color,
            );
        }
    }
}

fn push_triangle(
    triangles: &mut Vec<GlVertex>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
) {
    triangles.push(GlVertex::world(a, normal, color));
    triangles.push(GlVertex::world(b, normal, color));
    triangles.push(GlVertex::world(c, normal, color));
}

fn push_line(lines: &mut Vec<GlVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    let normal = [0.0, 0.0, 1.0];
    lines.push(GlVertex::world(a, normal, color));
    lines.push(GlVertex::world(b, normal, color));
}

fn diagnostic_geom_color(geom: &SharedGeomSnapshot) -> [f32; 4] {
    let seed = (geom.data_id as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((geom.type_id as u32).wrapping_mul(0x85EB_CA6B));
    let hue = (seed % 360) as f32;
    let sat = 0.72;
    let val = 0.92;
    let chroma = val * sat;
    let h = hue / 60.0;
    let x = chroma * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h as i32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = val - chroma;
    [r1 + m, g1 + m, b1 + m, 1.0]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = length3(v);
    if len <= 1e-6 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

#[allow(dead_code)]
fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize3(cross3(
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
        [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
    ))
}
