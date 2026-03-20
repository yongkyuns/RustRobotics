use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MUJOCO_HOME");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "wasm32" {
        return;
    }

    let mujoco_home = env::var("MUJOCO_HOME")
        .unwrap_or_else(|_| "/Users/ykshin/Dev/me/mujoco_340_env".to_string());
    let include_dir = PathBuf::from(&mujoco_home).join("include");
    let lib_dir = PathBuf::from(&mujoco_home).join("lib");
    let header = include_dir.join("mujoco/mujoco.h");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mujoco");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

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
