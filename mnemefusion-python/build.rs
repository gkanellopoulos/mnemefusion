// Build script for mnemefusion-python to configure CUDA properly

use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // Get build features
    let cuda_enabled = cfg!(feature = "entity-extraction-cuda");

    if cuda_enabled {
        println!("cargo:warning=Building with CUDA support enabled");

        // Set CUDA-related environment variables for CMake (used by llama-cpp-2)
        let cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1";

        println!("cargo:rustc-env=CUDA_PATH={}", cuda_path);
        println!("cargo:rustc-env=CUDA_TOOLKIT_ROOT_DIR={}", cuda_path);
        println!("cargo:rustc-env=CMAKE_CUDA_COMPILER={}\\bin\\nvcc.exe", cuda_path);
        println!("cargo:rustc-env=CMAKE_CUDA_ARCHITECTURES=75");

        // Add CUDA library path to linker
        println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);

        // Link CUDA libraries
        println!("cargo:rustc-link-lib=dylib=cudart_static");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");

        println!("cargo:warning=CUDA environment configured for architecture sm_75 (GTX 1650 Ti)");
    } else {
        println!("cargo:warning=Building without CUDA support (CPU only)");
    }
}
