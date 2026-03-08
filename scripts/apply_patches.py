#!/usr/bin/env python3
"""Apply required patches to cargo registry dependencies.

MnemeFusion depends on llama-cpp-sys-2 and usearch, both of which need
small patches to build correctly on Linux. This script automates applying
those patches after `cargo fetch`.

Usage:
    python3 scripts/apply_patches.py          # Apply all patches
    python3 scripts/apply_patches.py --check  # Check if patches are needed
    python3 scripts/apply_patches.py --clean  # Clean build caches after patching

Patches applied:
1. GGML_BACKEND_DL: OFF -> ON (llama-cpp-sys-2)
   Required for dynamic backend loading (ggml-cpu.so, ggml-cuda.so).
   Without this, backends don't export ggml_backend_init symbol.

2. hard_link().unwrap() -> .ok() (llama-cpp-sys-2 build.rs)
   Prevents EEXIST panic when cmake install creates symlinks that
   already exist in the output directory.

3. SIMSIMD removal (usearch build.rs)
   SIMSIMD fails to compile on Linux with GCC 11 (_mm512_reduce_add_ph).
   We remove the SIMSIMD source compilation entirely.
"""

import os
import re
import sys
import glob
import shutil
import argparse
from pathlib import Path


def find_cargo_registry():
    """Find the cargo registry source directory."""
    cargo_home = os.environ.get("CARGO_HOME", os.path.expanduser("~/.cargo"))
    registry_src = os.path.join(cargo_home, "registry", "src")
    if not os.path.isdir(registry_src):
        return None
    return registry_src


def find_package_dir(registry_src, package_prefix):
    """Find the latest version of a package in the cargo registry."""
    matches = []
    for registry in os.listdir(registry_src):
        registry_path = os.path.join(registry_src, registry)
        if not os.path.isdir(registry_path):
            continue
        for entry in os.listdir(registry_path):
            if entry.startswith(package_prefix):
                matches.append(os.path.join(registry_path, entry))
    # Return the latest (sorted alphabetically, which works for semver)
    return sorted(matches)[-1] if matches else None


def patch_ggml_backend_dl(llama_dir):
    """Patch 1: GGML_BACKEND_DL OFF -> ON."""
    cmake_path = os.path.join(llama_dir, "llama.cpp", "ggml", "CMakeLists.txt")
    if not os.path.exists(cmake_path):
        print(f"  SKIP: {cmake_path} not found")
        return False

    with open(cmake_path, "r") as f:
        content = f.read()

    if "GGML_BACKEND_DL" not in content:
        print("  SKIP: GGML_BACKEND_DL not found in CMakeLists.txt")
        return False

    if re.search(r'GGML_BACKEND_DL.*OFF\)', content):
        new_content = re.sub(
            r'(GGML_BACKEND_DL\s+"[^"]*")\s+OFF\)',
            r'\1 ON)',
            content
        )
        with open(cmake_path, "w") as f:
            f.write(new_content)
        print("  APPLIED: GGML_BACKEND_DL OFF -> ON")
        return True
    else:
        print("  OK: GGML_BACKEND_DL already ON")
        return False


def patch_hard_link(llama_dir):
    """Patch 2: hard_link().unwrap() -> .ok() in build.rs."""
    build_rs = os.path.join(llama_dir, "build.rs")
    if not os.path.exists(build_rs):
        print(f"  SKIP: {build_rs} not found")
        return False

    with open(build_rs, "r") as f:
        content = f.read()

    # Replace .unwrap() with .ok() on lines containing hard_link
    lines = content.split('\n')
    unwrap_count = 0
    new_lines = []
    for line in lines:
        if 'hard_link(' in line and '.unwrap()' in line:
            new_lines.append(line.replace('.unwrap()', '.ok()'))
            unwrap_count += 1
        else:
            new_lines.append(line)
    new_content = '\n'.join(new_lines)

    if unwrap_count > 0:
        with open(build_rs, "w") as f:
            f.write(new_content)
        print(f"  APPLIED: {unwrap_count} hard_link().unwrap() -> .ok()")
        return True
    else:
        print("  OK: hard_link patch already applied")
        return False


def patch_usearch_simsimd(usearch_dir):
    """Patch 3: Remove SIMSIMD compilation from usearch build.rs."""
    build_rs = os.path.join(usearch_dir, "build.rs")
    if not os.path.exists(build_rs):
        print(f"  SKIP: {build_rs} not found")
        return False

    with open(build_rs, "r") as f:
        content = f.read()

    changed = False

    # Remove simsimd/c/lib.c compilation
    if "simsimd/c/lib.c" in content:
        lines = content.split("\n")
        new_lines = [l for l in lines if "simsimd/c/lib.c" not in l]
        content = "\n".join(new_lines)
        changed = True
        print("  APPLIED: Removed simsimd/c/lib.c compilation")

    # Replace flags_to_try with empty vec
    if re.search(r'flags_to_try\s*=\s*match\s+target_arch', content):
        content = re.sub(
            r'flags_to_try\s*=\s*match\s+target_arch[^;]*;',
            'flags_to_try = Vec::<&str>::new();',
            content,
            flags=re.DOTALL
        )
        changed = True
        print("  APPLIED: Replaced flags_to_try with empty Vec")

    if changed:
        with open(build_rs, "w") as f:
            f.write(content)
        return True
    else:
        print("  OK: usearch SIMSIMD patch already applied or not needed")
        return False


def clean_build_caches(workspace_root):
    """Clean stale build caches after patching."""
    target_dir = os.path.join(workspace_root, "target")
    cleaned = 0
    for profile in ["release", "debug"]:
        for prefix in ["llama-cpp-sys-2", "usearch"]:
            for subdir in ["build", ".fingerprint"]:
                pattern = os.path.join(target_dir, profile, subdir, f"{prefix}-*")
                for d in glob.glob(pattern):
                    shutil.rmtree(d)
                    cleaned += 1
    if cleaned:
        print(f"Cleaned {cleaned} stale build cache directories")
    else:
        print("No stale caches to clean")


def main():
    parser = argparse.ArgumentParser(description="Apply MnemeFusion dependency patches")
    parser.add_argument("--check", action="store_true", help="Check if patches are needed")
    parser.add_argument("--clean", action="store_true", help="Clean build caches after patching")
    args = parser.parse_args()

    registry_src = find_cargo_registry()
    if not registry_src:
        print("ERROR: Cargo registry not found. Run 'cargo fetch' first.")
        sys.exit(1)

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    any_applied = False

    # Patch 1 & 2: llama-cpp-sys-2
    llama_dir = find_package_dir(registry_src, "llama-cpp-sys-2-")
    if llama_dir:
        print(f"\nllama-cpp-sys-2: {os.path.basename(llama_dir)}")
        if not args.check:
            any_applied |= patch_ggml_backend_dl(llama_dir)
            any_applied |= patch_hard_link(llama_dir)
        else:
            # Just check
            cmake = os.path.join(llama_dir, "llama.cpp", "ggml", "CMakeLists.txt")
            if os.path.exists(cmake):
                with open(cmake) as f:
                    if "GGML_BACKEND_DL" in f.read() and "OFF)" in f.read():
                        print("  NEEDS PATCH: GGML_BACKEND_DL")
            build_rs = os.path.join(llama_dir, "build.rs")
            if os.path.exists(build_rs):
                with open(build_rs) as f:
                    if "hard_link(" in f.read() and ".unwrap()" in f.read():
                        print("  NEEDS PATCH: hard_link")
    else:
        print("WARNING: llama-cpp-sys-2 not found in cargo registry")

    # Patch 3: usearch
    usearch_dir = find_package_dir(registry_src, "usearch-")
    if usearch_dir:
        print(f"\nusearch: {os.path.basename(usearch_dir)}")
        if not args.check:
            any_applied |= patch_usearch_simsimd(usearch_dir)
        else:
            build_rs = os.path.join(usearch_dir, "build.rs")
            if os.path.exists(build_rs):
                with open(build_rs) as f:
                    if "simsimd/c/lib.c" in f.read():
                        print("  NEEDS PATCH: SIMSIMD")
    else:
        print("NOTE: usearch not found in cargo registry (may use [patch.crates-io])")

    # Clean build caches if any patches were applied
    if any_applied or args.clean:
        print()
        clean_build_caches(workspace_root)

    if any_applied:
        print("\nPatches applied. Rebuild with: maturin develop --release --features entity-extraction-cuda")


if __name__ == "__main__":
    main()
