#!/usr/bin/env python3
"""Apply required patches to cargo registry dependencies.

HISTORY: This script previously applied 3 patches:
1. GGML_BACKEND_DL: OFF -> ON (llama-cpp-sys-2) — now baked into mnemefusion-llama-cpp-sys-2
2. hard_link().unwrap() -> .ok() (llama-cpp-sys-2) — now baked into mnemefusion-llama-cpp-sys-2
3. SIMSIMD removal (usearch) — now handled via default-features = false in Cargo.toml

All patches are now obsolete. The llama-cpp patches are published as forked crates
(mnemefusion-llama-cpp-2 and mnemefusion-llama-cpp-sys-2 on crates.io). The usearch
SIMSIMD issue is avoided by disabling the simsimd feature at the Cargo.toml level.

This script now only provides build cache cleaning for development use.

Usage:
    python3 scripts/apply_patches.py --clean  # Clean build caches
"""

import os
import glob
import shutil
import argparse


def clean_build_caches(workspace_root):
    """Clean stale build caches after dependency changes."""
    target_dir = os.path.join(workspace_root, "target")
    cleaned = 0
    for profile in ["release", "debug"]:
        for prefix in ["llama-cpp-sys-2", "mnemefusion-llama-cpp-sys-2", "usearch"]:
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
    parser = argparse.ArgumentParser(description="MnemeFusion build utilities")
    parser.add_argument("--clean", action="store_true", help="Clean build caches")
    args = parser.parse_args()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.clean:
        clean_build_caches(workspace_root)
    else:
        print("All dependency patches are now obsolete.")
        print("llama-cpp patches: baked into mnemefusion-llama-cpp-{sys-}2 on crates.io")
        print("usearch SIMSIMD:   disabled via default-features = false in Cargo.toml")
        print()
        print("Use --clean to clear stale build caches after dependency changes.")


if __name__ == "__main__":
    main()
