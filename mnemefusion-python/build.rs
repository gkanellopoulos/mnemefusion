// Build script for mnemefusion-python to configure CUDA properly

use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // Get build features
    let cuda_enabled = cfg!(feature = "entity-extraction-cuda");

    if cuda_enabled {
        println!("cargo:warning=Building with CUDA support enabled");

        // Detect CUDA path: env var > platform defaults
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

        let cuda_arch = env::var("CMAKE_CUDA_ARCHITECTURES").unwrap_or_else(|_| "75".to_string());

        // Set link search paths (these affect the linker, not cmake)
        if cfg!(target_os = "windows") {
            println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        }

        // Link CUDA libraries
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        if cfg!(target_os = "windows") {
            println!("cargo:rustc-link-lib=dylib=cudart_static");
        } else {
            println!("cargo:rustc-link-lib=static=cudart_static");
        }

        println!("cargo:warning=CUDA environment configured for architecture sm_{}", cuda_arch);
    } else {
        println!("cargo:warning=Building without CUDA support (CPU only)");
    }
}
