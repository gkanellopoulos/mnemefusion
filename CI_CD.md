# CI/CD Documentation

This document describes the continuous integration and continuous deployment setup for MnemeFusion.

## Overview

MnemeFusion uses GitHub Actions for automated testing, benchmarking, and code quality checks. All workflows run on Linux (Ubuntu) for optimal tooling support.

## Workflows

### 1. Test Workflow (`.github/workflows/test.yml`)

**Triggers**:
- Every push to `main`
- Every pull request to `main`

**Jobs**:

#### Test Suite Job
Runs comprehensive test suite including:
- Code formatting check (`cargo fmt`)
- Linting with Clippy (`cargo clippy`)
- All unit tests
- All integration tests
- Property-based tests (48 properties, 100 iterations each)
- Documentation tests

**Caching**:
- Cargo registry
- Cargo index
- Build artifacts

#### Coverage Job
Generates code coverage reports:
- Uses `cargo-llvm-cov` (Linux only)
- Uploads coverage report as artifact
- Displays coverage summary in logs
- Optional Codecov integration (commented out)

**Target**: >80% line coverage

### 2. Benchmark Workflow (`.github/workflows/benchmark.yml`)

**Triggers**:
- Pull requests to `main`
- Manual workflow dispatch

**Purpose**:
- Detects performance regressions
- Compares PR performance against `main` branch
- Runs both `core_operations` and `profiling` benchmarks

**Baseline Comparison**:
```bash
# PR branch benchmarks saved as baseline
cargo bench -- --save-baseline pr

# Main branch benchmarks compared against PR
cargo bench -- --baseline pr
```

**Artifacts**:
- Full Criterion benchmark results uploaded
- Viewable in GitHub Actions artifacts

## Status Badges

The README displays real-time CI/CD status:

```markdown
[![Tests](https://github.com/gkanellopoulos/mnemefusion/actions/workflows/test.yml/badge.svg)](...)
[![Benchmarks](https://github.com/gkanellopoulos/mnemefusion/actions/workflows/benchmark.yml/badge.svg)](...)
```

## Running Tests Locally

### All Tests
```bash
cargo test --all-features --verbose
```

### Property Tests
```bash
cargo test --test property_tests --release
```

### Benchmarks
```bash
cargo bench --bench core_operations
cargo bench --bench profiling
```

### Code Coverage (Linux only)
```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate coverage
cargo llvm-cov --all-features --workspace --html

# View report
open target/llvm-cov/html/index.html
```

### Formatting
```bash
cargo fmt --all -- --check
```

### Linting
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

## Performance Regression Thresholds

**Warning**: >5% slowdown on critical paths
**Failure**: >10% slowdown on critical paths

Critical paths:
- Memory add operation (target: <10ms)
- Memory get operation (target: <1ms)
- Vector search (target: <50ms for 100k memories)
- Query fusion (target: <100ms)

## Coverage Goals

**Target**: >80% line coverage

**Priority areas**:
- Core memory operations (add, get, delete)
- Query planning and fusion
- Error handling paths
- Edge cases in graph traversal

**Current status**: Measured automatically in CI/CD

## Troubleshooting

### Coverage Tool Not Working on Windows

**Issue**: `cargo-llvm-cov` requires Rust 1.87+ or may have Windows compatibility issues.

**Solution**: Coverage is measured in CI/CD on Linux runners. Local Windows development can skip coverage measurement.

### Benchmark Results Unavailable

**Issue**: Benchmarks only run on PRs, not on direct pushes to main.

**Solution**:
- Create a PR to run benchmarks
- Or manually trigger: Go to Actions → Benchmarks → Run workflow

### CI Failures on Formatting

**Issue**: Code not formatted correctly.

**Solution**:
```bash
cargo fmt --all
git commit -am "fix: format code"
git push
```

### CI Failures on Clippy

**Issue**: Linting warnings treated as errors.

**Solution**: Fix the warnings or add `#[allow(...)]` if intentional:
```rust
#[allow(dead_code)]
fn unused_helper() { }
```

## Future Enhancements

### Planned
- [ ] Codecov.io integration for public coverage reports
- [ ] Automatic PR comments with benchmark results
- [ ] Platform matrix: Linux, macOS, Windows
- [ ] Rust version matrix: stable, beta, nightly
- [ ] Automated regression detection with thresholds
- [ ] Performance tracking over time

### Optional
- [ ] Scheduled nightly benchmarks
- [ ] Fuzzing integration
- [ ] Security audit workflow (cargo-audit)
- [ ] Dependency update automation (Dependabot)

## Contributing

When submitting a PR:

1. **Ensure tests pass locally**: `cargo test --all-features`
2. **Format code**: `cargo fmt --all`
3. **Check lints**: `cargo clippy --all-targets --all-features`
4. **Run property tests**: `cargo test --test property_tests --release`
5. **Check benchmarks** (CI will run automatically)

CI must pass before merge.

## Contact

For CI/CD issues or questions, open an issue or contact the maintainers.

---

**Last Updated**: January 25, 2026
**Sprint**: Sprint 15 - Comprehensive Testing
