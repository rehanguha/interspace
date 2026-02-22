# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-22

### Added

#### Modular Architecture
- Complete restructure into modular package architecture
- Organized functions into categories: `distances/`, `information/`, `metrics/`, `misc/`
- Support for both flat API (`interspace.euclidean()`) and categorized access (`interspace.distances.vector.euclidean()`)

#### New Distance Functions (35+ new functions)

**Weighted Distances**
- `weighted_euclidean()` - Weighted Euclidean distance
- `weighted_manhattan()` - Weighted Manhattan distance  
- `weighted_minkowski()` - Weighted Minkowski distance

**Probability Distances**
- `kl_divergence()` - Kullback-Leibler divergence
- `js_distance()` - Jensen-Shannon distance
- `bhattacharyya_distance()` - Bhattacharyya distance
- `hellinger_distance()` - Hellinger distance
- `total_variation_distance()` - Total variation distance
- `wasserstein_distance()` - Earth Mover's Distance (1D)

**String Distances**
- `levenshtein_distance()` - Edit distance
- `damerau_levenshtein_distance()` - Edit + transpositions
- `jaro_distance()` - Jaro similarity
- `jaro_winkler_distance()` - Jaro with prefix weighting
- `hamming_distance_normalized()` - Normalized Hamming distance

**Geographic Distances**
- `vincenty_distance()` - Geodesic distance on ellipsoid (WGS-84)
- `bearing()` - Direction between two coordinates
- `midpoint()` - Geographic midpoint between coordinates
- `destination_point()` - Point at distance along bearing

**Time Series Distances**
- `dtw_distance()` - Dynamic Time Warping
- `euclidean_distance_1d()` - Optimized 1D Euclidean
- `longest_common_subsequence()` - LCS length

**Matrix Distances**
- `frobenius_distance()` - Frobenius norm distance
- `spectral_distance()` - Spectral norm distance
- `trace_distance()` - Nuclear norm distance

**Binary Distances**
- `russell_rao_distance()` - Russell-Rao distance
- `sokal_sneath_distance()` - Sokal-Sneath distance
- `kulczynski_distance()` - Kulczynski distance

**Normalized Distances**
- `normalized_euclidean()` - Dimension-normalized Euclidean
- `standardized_euclidean()` - Variance-weighted Euclidean
- `seuclidean()` - Alias for standardized_euclidean
- `chi2_distance()` - Chi-squared distance
- `gower_distance()` - Mixed variable types distance

**Physics Distances**
- `angular_distance()` - Shortest angular distance
- `spherical_law_of_cosines()` - Alternative great-circle formula
- `euclidean_3d()` - 3D Euclidean distance

**Set Distances**
- `tanimoto_distance()` - Extended Jaccard for vectors

**Information Theory**
- `entropy()` - Shannon entropy
- `cross_entropy()` - Cross-entropy
- `mutual_information()` - Mutual information

**Metrics Utilities**
- `pairwise_distance()` - Compute distance matrices
- `is_distance_metric()` - Validate metric properties

### Changed
- Version bumped from 0.0.11 to 0.1.0
- Improved documentation with formulas and examples for all functions
- Better error handling and validation

### Documentation
- Comprehensive README with all function tables
- Usage examples for each category
- Module structure documentation

---

## [0.0.11] - Previous Release

### Added
- Basic vector distances: euclidean, manhattan, minkowski, chebyshev_distance, cosine_similarity, cosine_distance, mahalanobis
- Geographic distance: haversine
- Set distances: jaccard_distance, dice_distance, matching_distance, overlap_distance
- Distribution distances: canberra_distance, braycurtis_distance, correlation_distance, pearson_distance, squared_chord_distance
- Hamming distance for integers, strings, and arrays

---

[0.1.0]: https://github.com/rehanguha/interspace/compare/v0.0.11...v0.1.0
[0.0.11]: https://github.com/rehanguha/interspace/releases/tag/v0.0.11