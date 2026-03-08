//! Build script for mnemefusion-core.
//!
//! When the `entity-extraction` feature is enabled, validates that required
//! patches have been applied to llama-cpp-sys-2 in the cargo registry.
//! Without these patches, the build will succeed but the resulting binary
//! will fail at runtime (backends won't load, or build panics on Linux).

fn main() {
    // Only check patches when entity-extraction feature is active
    #[cfg(feature = "entity-extraction")]
    check_llama_patches();
}

#[cfg(feature = "entity-extraction")]
fn check_llama_patches() {
    use std::path::{Path, PathBuf};

    // Find llama-cpp-sys-2 source in cargo registry
    let cargo_home = std::env::var("CARGO_HOME")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_default();
            format!("{}/.cargo", home)
        });

    let registry_src = PathBuf::from(&cargo_home).join("registry").join("src");
    if !registry_src.exists() {
        // Can't check — might be a vendor or git dependency
        return;
    }

    let mut llama_dir: Option<PathBuf> = None;
    if let Ok(registries) = std::fs::read_dir(&registry_src) {
        for registry in registries.flatten() {
            if let Ok(packages) = std::fs::read_dir(registry.path()) {
                for pkg in packages.flatten() {
                    let name = pkg.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("llama-cpp-sys-2-") {
                        llama_dir = Some(pkg.path());
                    }
                }
            }
        }
    }

    let Some(llama_dir) = llama_dir else { return };

    // Check 1: GGML_BACKEND_DL must be ON
    let cmake_path = llama_dir
        .join("llama.cpp")
        .join("ggml")
        .join("CMakeLists.txt");
    if cmake_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&cmake_path) {
            if content.contains("GGML_BACKEND_DL") && content.contains("OFF)") {
                println!("cargo:warning=╔══════════════════════════════════════════════════════════════╗");
                println!("cargo:warning=║  GGML_BACKEND_DL is OFF — backends won't load at runtime!  ║");
                println!("cargo:warning=║  Run: python3 scripts/apply_patches.py                     ║");
                println!("cargo:warning=║  Or:  bash scripts/setup_linux.sh                          ║");
                println!("cargo:warning=╚══════════════════════════════════════════════════════════════╝");
            }
        }
    }

    // Check 2: hard_link().unwrap() should be .ok() on Linux
    #[cfg(target_os = "linux")]
    {
        let build_rs = llama_dir.join("build.rs");
        if build_rs.exists() {
            if let Ok(content) = std::fs::read_to_string(&build_rs) {
                if content.contains("hard_link(") && content.contains(".unwrap()") {
                    println!("cargo:warning=╔══════════════════════════════════════════════════════════════╗");
                    println!("cargo:warning=║  hard_link().unwrap() will panic on Linux!                  ║");
                    println!("cargo:warning=║  Run: python3 scripts/apply_patches.py                     ║");
                    println!("cargo:warning=╚══════════════════════════════════════════════════════════════╝");
                }
            }
        }
    }
}
