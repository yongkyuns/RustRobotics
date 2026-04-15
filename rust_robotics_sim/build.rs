use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MUJOCO_HOME");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "wasm32" {
        return;
    }

    let mujoco_home = resolve_mujoco_home();
    let include_dir = mujoco_home.join("include");
    let lib_dir = mujoco_home.join("lib");
    let header = include_dir.join("mujoco/mujoco.h");

    assert!(
        header.exists(),
        "MuJoCo header not found at {}. Set MUJOCO_HOME to a directory containing include/mujoco/mujoco.h and lib/.",
        header.display()
    );
    assert!(
        lib_dir.exists(),
        "MuJoCo lib directory not found at {}. Set MUJOCO_HOME to a directory containing include/ and lib/.",
        lib_dir.display()
    );

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mujoco");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(target_os.as_str(), "linux" | "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("mj.*")
        .allowlist_function("mju_.*")
        .allowlist_function("mjr_.*")
        .allowlist_function("mjv_.*")
        .allowlist_function("mjt_.*")
        .allowlist_type("mj.*")
        .allowlist_type("mjt.*")
        .allowlist_var("mj.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("failed to generate MuJoCo bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    bindings
        .write_to_file(out_path.join("mujoco_bindings.rs"))
        .expect("failed to write MuJoCo bindings");
}

fn resolve_mujoco_home() -> PathBuf {
    if let Some(path) = env::var_os("MUJOCO_HOME") {
        return PathBuf::from(path);
    }

    for candidate in default_mujoco_candidates() {
        if candidate.join("include/mujoco/mujoco.h").exists() {
            return candidate;
        }
    }

    panic!(
        "MUJOCO_HOME is not set. Set it to a MuJoCo installation containing include/mujoco/mujoco.h and lib/."
    );
}

fn default_mujoco_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(home) = home_dir() {
        candidates.push(home.join(".mujoco/mujoco-3.4.0"));
        candidates.push(home.join("mujoco-3.4.0"));
    }
    candidates
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}
