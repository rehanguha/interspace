# Interspace

[![PyPI version](https://badge.fury.io/py/interspace.svg)](https://badge.fury.io/py/interspace)
[![Python](https://img.shields.io/pypi/pyversions/interspace.svg)](https://pypi.org/project/interspace/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/rehanguha/interspace/actions/workflows/test.yml/badge.svg)](https://github.com/rehanguha/interspace/actions/workflows/test.yml)

A comprehensive collection of distance and similarity functions for vectors, sequences, and distributions. Designed for machine learning, data science, and scientific computing applications.

## Features

- **50+ Distance Functions** across multiple categories
- **Pure Python + NumPy** - no external dependencies
- **Comprehensive Documentation** with examples and formulas
- **Type Hints** for better IDE support
- **Modular Architecture** - import by category or use flat API

## Installation

### From PyPI

```bash
pip install interspace
```

### From Source

```bash
git clone https://github.com/rehanguha/interspace.git
cd interspace
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import interspace

# Direct access (flat API)
interspace.euclidean([1, 2, 3], [4, 5, 6])
# 5.196152422706632

interspace.levenshtein_distance("kitten", "sitting")
# 3

interspace.haversine((42.52, 15.28), (51.51, -0.13))
# 1231910.73... (Zagreb to London in meters)

# Categorized access
interspace.distances.vector.euclidean([1, 2], [3, 4])
interspace.distances.string.levenshtein_distance("hello", "hallo")
interspace.distances.geographic.haversine((0, 0), (1, 1))
```

---

## Available Functions

### Vector Distances

| Function | Description | Formula |
|----------|-------------|---------|
| `euclidean(x, y)` | L2 norm distance | `√Σ(xᵢ - yᵢ)²` |
| `manhattan(x, y)` | L1 norm / cityblock distance | `Σ|xᵢ - yᵢ|` |
| `minkowski(x, y, p)` | Generalized p-norm distance | `(Σ|xᵢ - yᵢ|ᵖ)^(1/p)` |
| `chebyshev_distance(x, y)` | L∞ norm / maximum distance | `max|xᵢ - yᵢ|` |
| `cosine_similarity(x, y)` | Angular similarity | `x·y / (‖x‖‖y‖)` |
| `cosine_distance(x, y)` | 1 - cosine_similarity | `1 - (x·y / (‖x‖‖y‖))` |
| `mahalanobis(u, v, VI)` | Distance with covariance | `√((u-v)ᵀVI(u-v))` |

```python
>>> interspace.euclidean([1, 2, 3], [4, 5, 6])
5.196152422706632

>>> interspace.minkowski([1, 2], [4, 6], p=1)  # Manhattan
7.0

>>> interspace.cosine_similarity([1, 0], [0, 1])
0.0
```

### Weighted Distances

| Function | Description |
|----------|-------------|
| `weighted_euclidean(x, y, w)` | Weighted Euclidean distance |
| `weighted_manhattan(x, y, w)` | Weighted Manhattan distance |
| `weighted_minkowski(x, y, w, p)` | Weighted Minkowski distance |

```python
>>> interspace.weighted_euclidean([1, 2], [4, 6], [1, 0.5])
4.301162633521313
```

### Set Distances

| Function | Description |
|----------|-------------|
| `jaccard_distance(x, y)` | 1 - |X ∩ Y| / |X ∪ Y| |
| `dice_distance(x, y)` | 1 - 2|X ∩ Y| / (|X| + |Y|) |
| `matching_distance(x, y)` | Proportion of mismatched positions |
| `overlap_distance(x, y)` | 1 - |X ∩ Y| / min(|X|, |Y|) |
| `tanimoto_distance(x, y)` | Extended Jaccard for vectors |

```python
>>> interspace.jaccard_distance([1, 2, 3], [2, 3, 4])
0.5

>>> interspace.dice_distance([1, 2, 3], [2, 3, 4])
0.4
```

### Distribution Distances

| Function | Description |
|----------|-------------|
| `canberra_distance(x, y)` | Weighted Manhattan distance |
| `braycurtis_distance(x, y)` | Bray-Curtis dissimilarity |
| `correlation_distance(x, y)` | 1 - Pearson correlation |
| `pearson_distance(x, y)` | Alias for correlation_distance |
| `squared_chord_distance(x, y)` | Squared chord distance |

```python
>>> interspace.canberra_distance([1, 2, 3], [2, 2, 4])
0.47619047619047616

>>> interspace.correlation_distance([1, 2, 3], [3, 2, 1])
2.0
```

### Probability Distances

| Function | Description |
|----------|-------------|
| `kl_divergence(p, q)` | Kullback-Leibler divergence |
| `js_distance(p, q)` | Jensen-Shannon distance |
| `bhattacharyya_distance(p, q)` | Distribution overlap measure |
| `hellinger_distance(p, q)` | Fidelity-based distance |
| `total_variation_distance(p, q)` | L1 distribution distance |
| `wasserstein_distance(p, q)` | Earth Mover's Distance (1D) |

```python
>>> interspace.kl_divergence([0.5, 0.5], [0.5, 0.5])
0.0

>>> interspace.js_distance([1.0, 0.0], [0.5, 0.5])
0.4645034044881785
```

### String Distances

| Function | Description |
|----------|-------------|
| `hamming(a, b)` | Bitwise or per-position mismatches |
| `hamming_distance_normalized(a, b)` | Normalized Hamming distance |
| `levenshtein_distance(s1, s2)` | Edit distance |
| `damerau_levenshtein_distance(s1, s2)` | Edit + transpositions |
| `jaro_distance(s1, s2)` | String similarity |
| `jaro_winkler_distance(s1, s2)` | Jaro with prefix weighting |

```python
>>> interspace.levenshtein_distance("kitten", "sitting")
3

>>> interspace.jaro_winkler_distance("MARTHA", "MARHTA")
0.9666666666666667

>>> interspace.hamming(0b1010, 0b0011)
2
```

### Geographic Distances

| Function | Description |
|----------|-------------|
| `haversine(coord1, coord2, R)` | Great-circle distance |
| `vincenty_distance(coord1, coord2)` | Geodesic on ellipsoid |
| `bearing(coord1, coord2)` | Direction between points |
| `midpoint(coord1, coord2)` | Geographic midpoint |
| `destination_point(coord, bearing, distance)` | Point along bearing |

```python
>>> zagreb = (45.8150, 15.9819)
>>> london = (51.5074, -0.1278)
>>> interspace.haversine(zagreb, london)
1230000.0  # meters

>>> interspace.bearing((0, 0), (1, 0))
0.0  # North
```

### Time Series Distances

| Function | Description |
|----------|-------------|
| `dtw_distance(x, y)` | Dynamic Time Warping |
| `euclidean_distance_1d(x, y)` | 1D Euclidean distance |
| `longest_common_subsequence(x, y)` | LCS length |

```python
>>> interspace.dtw_distance([1, 2, 3], [1, 2, 2, 3])
0.0

>>> interspace.longest_common_subsequence([1, 2, 3, 4], [2, 3, 5])
2
```

### Matrix Distances

| Function | Description |
|----------|-------------|
| `frobenius_distance(A, B)` | Frobenius norm distance |
| `spectral_distance(A, B)` | Largest singular value |
| `trace_distance(A, B)` | Nuclear norm distance / 2 |

```python
>>> A = [[1, 0], [0, 1]]
>>> B = [[1, 0], [0, 2]]
>>> interspace.spectral_distance(A, B)
1.0
```

### Binary Distances

| Function | Description |
|----------|-------------|
| `russell_rao_distance(x, y)` | Russell-Rao distance |
| `sokal_sneath_distance(x, y)` | Sokal-Sneath distance |
| `kulczynski_distance(x, y)` | Kulczynski distance |

```python
>>> interspace.russell_rao_distance([1, 0, 1, 0], [1, 1, 0, 0])
0.75
```

### Normalized Distances

| Function | Description |
|----------|-------------|
| `normalized_euclidean(x, y)` | Euclidean / √n |
| `standardized_euclidean(x, y, variances)` | Variance-weighted |
| `seuclidean(x, y, V)` | Alias for standardized_euclidean |
| `chi2_distance(x, y)` | Chi-squared distance |
| `gower_distance(x, y, types, ranges)` | Mixed variable types |

```python
>>> interspace.chi2_distance([1, 2, 3], [2, 3, 4])
0.2777777777777778
```

### Physics Distances

| Function | Description |
|----------|-------------|
| `angular_distance(angle1, angle2)` | Shortest angular distance |
| `spherical_law_of_cosines(coord1, coord2)` | Alternative great-circle |
| `euclidean_3d(point1, point2)` | 3D Euclidean distance |

```python
>>> interspace.angular_distance(10, 350)
20.0

>>> interspace.euclidean_3d([0, 0, 0], [1, 2, 2])
3.0
```

### Information Theory

| Function | Description |
|----------|-------------|
| `entropy(p, base)` | Shannon entropy |
| `cross_entropy(p, q, base)` | Cross-entropy |
| `mutual_information(x, y, base)` | Mutual information |

```python
>>> interspace.entropy([0.5, 0.5])
1.0

>>> interspace.mutual_information([0, 0, 1, 1], [0, 0, 1, 1])
1.0
```

### Metrics Utilities

| Function | Description |
|----------|-------------|
| `pairwise_distance(X, Y, metric)` | Compute distance matrix |
| `is_distance_metric(func)` | Validate metric properties |

```python
>>> X = [[1, 2], [3, 4], [5, 6]]
>>> interspace.pairwise_distance(X, metric="euclidean")
array([[0.        , 2.82842712, 5.65685425],
       [2.82842712, 0.        , 2.82842712],
       [5.65685425, 2.82842712, 0.        ]])
```

---

## Module Structure

```
interspace/
├── __init__.py          # Main exports (flat API)
├── _validators.py       # Internal validation helpers
├── distances/
│   ├── vector.py        # Euclidean, Manhattan, Minkowski, etc.
│   ├── weighted.py      # Weighted distance functions
│   ├── set.py           # Jaccard, Dice, Tanimoto, etc.
│   ├── distribution.py  # Canberra, Bray-Curtis, etc.
│   ├── probability.py   # KL, JS, Bhattacharyya, etc.
│   ├── string.py        # Levenshtein, Jaro, Hamming, etc.
│   ├── geographic.py    # Haversine, Vincenty, Bearing, etc.
│   ├── time_series.py   # DTW, LCS, etc.
│   ├── matrix.py        # Frobenius, Spectral, Trace
│   ├── binary.py        # Russell-Rao, Sokal-Sneath, etc.
│   ├── normalized.py    # Chi-squared, Gower, etc.
│   └── physics.py       # Angular, 3D Euclidean, etc.
├── information/
│   └── theory.py        # Entropy, Cross-entropy, MI
├── metrics/
│   ├── pairwise.py      # Pairwise distance matrix
│   └── validation.py    # Metric property validation
└── misc/
    └── misc.py          # Experimental functions
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=interspace --cov-report=html
```

### Code Quality

```bash
# Format code
black interspace/

# Lint
ruff check interspace/

# Type check
mypy interspace/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rehan Guha**
- Email: rehanguha29@gmail.com
- GitHub: [@rehanguha](https://github.com/rehanguha)

## Acknowledgments

Inspired by `scipy.spatial.distance` and designed for simplicity and ease of use.