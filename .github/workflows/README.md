# GitHub Actions Workflows

## Status: DISABLED (January 28, 2026)

The CI/CD workflows have been temporarily disabled until we finalize the CI/CD flow.

**Disabled workflows:**
- `test.yml.disabled` - Test suite and code coverage
- `benchmark.yml.disabled` - Benchmark regression detection

**Reason for disabling:**
- Workflows were created in previous sessions for Linux testing
- Currently failing on every push to main
- Need to be updated and tested before re-enabling

## How to Re-enable

When ready to finalize CI/CD:

1. **Rename workflows back:**
   ```bash
   git mv .github/workflows/test.yml.disabled .github/workflows/test.yml
   git mv .github/workflows/benchmark.yml.disabled .github/workflows/benchmark.yml
   ```

2. **Test workflows locally:**
   - Install act (https://github.com/nektos/act)
   - Run: `act -j test` to test locally
   - Fix any issues before pushing

3. **Update workflows if needed:**
   - Review dependencies and steps
   - Update Rust version if needed
   - Test on fresh Linux environment

4. **Commit and push:**
   ```bash
   git add .github/workflows/
   git commit -m "chore: re-enable CI/CD workflows"
   git push
   ```

## Future Work

When re-enabling, consider:
- Adding Windows and macOS runners (currently Linux only)
- Splitting into more granular jobs
- Adding benchmark result publishing
- Setting up coverage reporting (codecov.io)
- Adding status badges back to README

## References

- Sprint 15 implemented initial CI/CD setup
- Documentation: `docs/CI_CD.md` (if exists)
- Test workflow was running successfully in Sprint 15
