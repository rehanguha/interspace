# Interspace

[![PyPI version](https://badge.fury.io/py/interspace.svg)](https://badge.fury.io/py/interspace)
[![Python](https://img.shields.io/pypi/pyversions/interspace.svg)](https://pypi.org/project/interspace/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/rehanguha/interspace/actions/workflows/test.yml/badge.svg)](https://github.com/rehanguha/interspace/actions/workflows/test.yml)

A comprehensive collection of distance and similarity functions for vectors, sequences, and distributions. Designed for machine learning, data science, and scientific computing applications.

## Features

- **Vector Distances**: Euclidean, Manhattan, Minkowski, Chebyshev, Cosine, Mahalanobis
- **Geographic Distances**: Haversine (great-circle distance)
- **Sequence Distances**: Hamming, Jaccard, Dice, Matching, Overlap
- **Distribution Distances**: Canberra, Bray-Curtis, Squared Chord, Correlation
- **Type hints** for better IDE support
- **Comprehensive documentation** with examples
- **100% test coverage**

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

# Euclidean distance
print(interspace.euclidean([1, 2, 3], [4, 5, 6]))
# 5.196152422706632

# Manhattan distance
print(interspace.manhattan([1, 2, 3], [4, 5, 6]))
# 9.0

# Cosine similarity
print(interspace.cosine_similarity([1, 0, 0], [0, 1, 0]))
# 0.0

# Haversine distance (in meters)
print(interspace.haversine((42.5170365, 15.2778599), (51.5073219, -0.1276474)))
# 1231910.73... (Zagreb to London)
```

## Available Functions

### Vector Distances

| Function | Description | Formula |
|----------|-------------|---------|
| `euclidean(x, y)` | L2 norm distance | √Σ(xᵢ - yᵢ)² |
| `manhattan(x, y)` | L1 norm / cityblock distance | Σ\|xᵢ - yᵢ\| |
| `minkowski(x, y, p)` | Generalized p-norm distance | (Σ\|xᵢ - yᵢ\|ᵖ)^(1/p) |
| `chebyshev_distance(x, y)` | L∞ norm / maximum distance | max\|xᵢ - yᵢ\| |
| `cosine_similarity(x, y)` | Angular similarity | x·y / (‖x‖‖y‖) |
| `cosine_distance(x, y)` | 1 - cosine_similarity | 1 - (x·y / (‖x‖‖y‖)) |
| `mahalanobis(u, v, VI)` | Distance with covariance | √((u-v)ᵀVI(u-v)) |

### Geographic Distances

| Function | Description |
|----------|-------------|
| `haversine(coord1, coord2, R=6372800)` | Great-circle distance between (lat, lon) pairs |

### Sequence/Set Distances

| Function | Description |
|----------|-------------|
| `hamming(a, b)` | Bitwise (integers) or positional mismatches (strings/arrays) |
| `jaccard_distance(x, y)` | 1 - \|X ∩ Y\| / \|X ∪ Y\| |
| `dice_distance(x, y)` | 1 - 2\|X ∩ Y\| / (\|X\| + \|Y\|) |
| `matching_distance(x, y)` | Proportion of mismatched positions |
| `overlap_distance(x, y)` | 1 - \|X ∩ Y\| / min(\|X\|, \|Y\|) |

### Distribution Distances

| Function | Description |
|----------|-------------|
| `canberra_distance(x, y)` | Weighted Manhattan distance |
| `braycurtis_distance(x, y)` | Bray-Curtis dissimilarity |
| `correlation_distance(x, y)` | 1 - Pearson correlation |
| `pearson_distance(x, y)` | Alias for correlation_distance |
| `squared_chord_distance(x, y)` | Squared chord distance |

## Usage Examples

### Basic Distances

```python
import interspace

# Euclidean distance
result = interspace.euclidean([1, 2, 3], [4, 5, 6])
# 5.196152422706632

# Minkowski with different p values
interspace.minkowski([1, 2], [4, 6], p=1)  # Manhattan: 7.0
interspace.minkowski([1, 2], [4, 6], p=2)  # Euclidean: 5.0
interspace.minkowski([1, 2], [4, 6], p=3)  # ~5.04

# Chebyshev (maximum) distance
interspace.chebyshev_distance([1, 2, 3], [4, 5, 6])
# 3.0
```

### Cosine Similarity & Distance

```python
# Orthogonal vectors
interspace.cosine_similarity([1, 0], [0, 1])  # 0.0
interspace.cosine_distance([1, 0], [0, 1])    # 1.0

# Identical vectors
interspace.cosine_similarity([1, 1], [1, 1])  # 1.0
interspace.cosine_distance([1, 1], [1, 1])    # 0.0

# Opposite vectors
interspace.cosine_similarity([1, 0], [-1, 0])  # -1.0
interspace.cosine_distance([1, 0], [-1, 0])    # 2.0
```

### Geographic Distance (Haversine)

```python
# Distance between two cities (in meters)
zagreb = (45.8150, 15.9819)
london = (51.5074, -0.1278)
distance = interspace.haversine(zagreb, london)
# ~1230000 meters (1230 km)

# Use a different radius (e.g., Mars)
mars_distance = interspace.haversine((0, 0), (45, 90), R=3389000)
```

### Hamming Distance

```python
# For integers (bitwise)
interspace.hamming(0b1010, 0b0011)  # 2

# For strings
interspace.hamming("karolin", "kathrin")  # 3

# For arrays
interspace.hamming([1, 2, 3, 4], [1, 0, 3, 5])  # 2
```

### Set-based Distances

```python
# Jaccard distance
interspace.jaccard_distance([1, 2, 3], [2, 3, 4])  # 0.5

# Dice distance
interspace.dice_distance([1, 2, 3], [2, 3, 4])  # 0.4

# Overlap distance
interspace.overlap_distance([1, 2, 3], [2, 3, 4])  # 0.5
```

### Distribution Distances

```python
# Canberra distance
interspace.canberra_distance([1, 2, 3], [2, 2, 4])

# Bray-Curtis distance
interspace.braycurtis_distance([1, 2, 3], [2, 2, 4])

# Correlation distance
interspace.correlation_distance([1, 2, 3], [3, 2, 1])  # 2.0
```

### Mahalanobis Distance

```python
import numpy as np

# Inverse covariance matrix
VI = np.array([[1, 0.5], [0.5, 1]])

# Distance between two points
interspace.mahalanobis([0, 0], [1, 1], VI)
```

## API Reference

For detailed API documentation, see the [API Reference](docs/api.md) or use Python's help:

```python
import interspace
help(interspace.euclidean)
```

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