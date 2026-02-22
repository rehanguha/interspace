# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.11] - 2025-02-22

### Added
- Added `__version__` attribute to the module
- Added comprehensive docstrings with parameters, returns, and examples for all functions
- Added type hints for better IDE support
- Added `_validate_same_shape` helper function for consistent shape validation
- Added shape validation to `euclidean`, `manhattan`, and other vector functions

### Changed
- Migrated from `setup.py` to modern `pyproject.toml` configuration
- Updated Python version requirement from `>=2.7` to `>=3.8`
- Improved `haversine` function to use `np.radians` instead of `math.radians`
- Replaced `math` module functions with `numpy` equivalents for consistency
- Improved error messages for shape validation

### Fixed
- Fixed inconsistent shape validation across distance functions
- Fixed redundant array conversions in several functions

### Documentation
- Completely rewrote README with badges, tables, and comprehensive examples
- Added API reference table for all functions
- Added installation instructions for development

### Tests
- Added comprehensive test coverage for all functions
- Added test classes for organization
- Added edge case tests (zero vectors, empty sets, shape mismatches)
- Added tests for `__version__` and `__all__` exports

## [0.0.10] - Previous Release

### Added
- Initial release with basic distance functions
- Euclidean, Manhattan, Minkowski distances
- Cosine similarity and distance
- Haversine distance
- Hamming distance
- Mahalanobis distance
- Chebyshev distance
- Jaccard, Dice, Matching, Overlap distances
- Canberra, Bray-Curtis distances
- Correlation and Pearson distances
- Squared chord distance

[0.0.11]: https://github.com/rehanguha/interspace/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/rehanguha/interspace/releases/tag/v0.0.10