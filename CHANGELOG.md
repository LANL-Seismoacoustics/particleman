# Changelog

## 0.4.0 - 2026-07-20

### Major Changes

* **Migration to external stockwell library**: Particleman now depends on the external [stockwell](https://github.com/claudiodsf/stockwell) library instead of shipping its own C implementation. This provides:
  - Simplified installation (no C compiler or FFTW required)
  - Pre-built wheels for Windows, macOS, and Linux
  - Better cross-platform support
  - Automatic memory cleanup
  - Active external maintenance

### New Features

* **Gamma parameter for resolution tuning**: `stransform()` now accepts a `gamma` parameter (default=1.0) to control the time-frequency resolution tradeoff:
  - `gamma < 1`: Better time resolution, worse frequency resolution
  - `gamma = 1`: Standard Stockwell transform (default)
  - `gamma > 1`: Better frequency resolution, worse time resolution

* **Alternative window functions**: `stransform()` now accepts a `win_type` parameter to select the window function:
  - `'gauss'`: Standard Gaussian window (default)
  - `'kazemi'`: Alternative window from Kazemi (2014)

### Bug Fixes

* Fixed internal default handling in `st()` function (was using `N % 2`, now correctly uses `N // 2`)
* Improved memory management with automatic cleanup on exit

### Backward Compatibility

* ✅ All existing code continues to work without changes
* ✅ New parameters are optional with sensible defaults
* ✅ All existing tests pass

### Breaking Changes

* **Dependency change**: Now requires `stockwell>=1.0` package
* **Version bump**: 0.3.0 → 0.4.0 to indicate new features and dependency change

### Removed

* Internal C implementation (`src/particleman/st.c` - no longer built or used)
* Direct FFTW dependency at build time
* Custom C extension building in `setup.py`

### For Developers

* Simplified build process (no C compilation needed)
* Reduced maintenance burden (~480 lines of C/Python code removed)
* See [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for detailed migration information

---

## 0.3.0

* The order of arguments to `particleman.plotting.NIP_filter_plots` has changed,
  such that filtered radial is required, but filtered vertical and transverse
  are optional.  This allows Love polarization to be plotted.
* The location of the radial and vertical tiles in
  `particleman.plotting.NIP_filter_plots` have been swapped.
