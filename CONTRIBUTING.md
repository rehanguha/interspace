# Contributing to Interspace

Thank you for your interest in contributing to Interspace! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainer.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/interspace.git
   cd interspace
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify the installation:
   ```bash
   python -c "import interspace; print(interspace.__version__)"
   ```

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. A clear title and description
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Python version and OS

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:

1. A clear description of the feature
2. Use cases and examples
3. Any relevant references or implementations

### Adding New Distance Functions

When adding a new distance function:

1. Add the function to `interspace/interspace.py`
2. Add comprehensive docstring following NumPy style
3. Add type hints
4. Add the function to `__all__` list
5. Add tests in `interspace/test_interspace.py`
6. Update README.md with documentation
7. Update CHANGELOG.md

Example function template:

```python
def new_distance(x: VectorLike, y: VectorLike) -> float:
    """Compute New distance between two vectors.
    
    Brief description of the distance metric.
    
    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    
    Returns
    -------
    float
        New distance.
    
    Raises
    ------
    ValueError
        If vectors have different shapes.
    
    Examples
    --------
    >>> new_distance([1, 2, 3], [4, 5, 6])
    3.0
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    # Implementation here
    return result
```

## Coding Standards

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints for all public functions

### Formatting

We use `black` for code formatting:

```bash
black interspace/
```

### Linting

We use `ruff` for linting:

```bash
ruff check interspace/
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy interspace/
```

### Docstrings

Use NumPy-style docstrings for all public functions:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """Brief description of the function.
    
    Longer description if needed.
    
    Parameters
    ----------
    arg1 : type
        Description of arg1.
    arg2 : type, optional
        Description of arg2. Default is value.
    
    Returns
    -------
    return_type
        Description of the return value.
    
    Raises
    ------
    ExceptionType
        When this exception is raised.
    
    Examples
    --------
    >>> function_name(1, 2)
    3
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest interspace/test_interspace.py

# Run specific test class
pytest interspace/test_interspace.py::TestEuclidean

# Run specific test
pytest interspace/test_interspace.py::TestEuclidean::test_basic_distance
```

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov=interspace --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=interspace --cov-report=html
# Open htmlcov/index.html in browser
```

### Writing Tests

- Use pytest framework
- Organize tests into classes by function
- Write descriptive test names
- Test both success and failure cases
- Test edge cases

Example test class:

```python
class TestNewDistance:
    """Tests for new_distance function."""

    def test_basic_distance(self):
        assert interspace.new_distance([1, 2], [3, 4]) == pytest.approx(expected)

    def test_identical_vectors(self):
        assert interspace.new_distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.new_distance([1], [1, 2])
```

## Pull Request Process

1. **Create a branch**: Create a feature branch from `main`

2. **Make changes**: Implement your changes following coding standards

3. **Add tests**: Ensure all new code has tests

4. **Update documentation**: Update README.md, docstrings, and CHANGELOG.md as needed

5. **Run tests and linting**:
   ```bash
   pytest
   black interspace/
   ruff check interspace/
   mypy interspace/
   ```

6. **Commit changes**: Write clear commit messages
   ```
   feat: add new_distance function
   fix: correct shape validation in euclidean
   docs: update README with examples
   test: add edge case tests for cosine_distance
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI checks pass

### PR Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] All new code has type hints
- [ ] All new functions have NumPy-style docstrings
- [ ] All new code has tests
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Contact the maintainer at rehanguha29@gmail.com

Thank you for contributing to Interspace!