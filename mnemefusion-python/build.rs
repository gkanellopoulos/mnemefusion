// Build script for mnemefusion-python to configure CUDA properly
//
// Multi-arch CUDA support:
//   CMAKE_CUDA_ARCHITECTURES controls which GPU architectures are compiled.
//   Set to a semicolon-separated list for multi-arch: "75;86;89"
//   Or "native" to auto-detect the current GPU.
//   Default: auto-detect via nvidia-smi, fallback to "75;86;89" (covers most GPUs).

use std::env;
use std::process::Command;

fn main() {
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

        // Multi-arch CUDA: detect or use env var
        let cuda_arch = env::var("CMAKE_CUDA_ARCHITECTURES").unwrap_or_else(|_| {
            detect_cuda_architectures()
        });

        // Pass architecture to cmake via env var (llama-cpp-sys-2's build.rs reads this)
        println!("cargo:rustc-env=CMAKE_CUDA_ARCHITECTURES={}", cuda_arch);

        // Set link search paths
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

        println!("cargo:warning=CUDA architectures: sm_{}", cuda_arch.replace(';', ", sm_"));
    } else {
        println!("cargo:warning=Building without CUDA support (CPU only)");
    }
}

/// Auto-detect GPU compute capability using nvidia-smi.
/// Returns semicolon-separated list of unique architectures.
/// Falls back to common multi-arch set if detection fails.
fn detect_cuda_architectures() -> String {
    // Try nvidia-smi to detect installed GPUs
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut archs: Vec<String> = stdout
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed == "[N/A]" {
                        return None;
                    }
                    // "7.5" -> "75", "8.6" -> "86"
                    Some(trimmed.replace('.', ""))
                })
                .collect();
            archs.sort();
            archs.dedup();
            if !archs.is_empty() {
                return archs.join(";");
            }
        }
    }

    // Fallback: build for common architectures with -real suffix for native code.
    // -real = compile optimized native kernels (not just PTX for JIT).
    // 75 = Turing (GTX 1650 Ti), 86 = Ampere (RTX 3050, A40), 89 = Ada Lovelace (RTX 4090)
    println!("cargo:warning=Could not detect GPU — building for sm_75, sm_86, sm_89 (fat binary)");
    "75-real;86-real;89-real".to_string()
}
