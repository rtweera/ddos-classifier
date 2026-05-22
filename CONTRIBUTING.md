# Contributing to DDoS Classifier

Thank you for your interest in contributing to the DDoS Classifier project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:
- Be respectful of different viewpoints and experiences
- Use inclusive language
- Focus on constructive feedback
- Report inappropriate behavior to project maintainers

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- GitHub account
- Poetry (for dependency management)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ddos-classifier.git
   cd ddos-classifier
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/rtweera/ddos-classifier.git
   ```

## Development Setup

### 1. Install Dependencies

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Activate Virtual Environment

```bash
# Start Poetry shell
poetry shell

# Verify installation
python --version  # Should be 3.12+
poetry --version
```

### 3. Set Up Development Tools

The project uses standard Python development tools:

```bash
# If you want to use linting/formatting tools manually
# (These can be added to poetry dependencies if needed)
pip install black flake8 pytest
```

### 4. Verify Setup

```bash
# Test that the preprocessing pipeline imports correctly
python -c "from preprocessing import DataPreprocessingPipeline; print('✓ Setup successful')"
```

## Making Changes

### Create a Feature Branch

Always create a new branch for your work:

```bash
# Update your local main
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming Conventions

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation improvements
- `refactor/description` - Code refactoring
- `test/description` - Test additions

### Commit Messages

Write clear, descriptive commit messages:

```bash
# Good examples:
git commit -m "Add IP octet feature in composite splitter"
git commit -m "Fix NaN handling in numerical standardiser"
git commit -m "Document autoencoder architecture"

# Format: [Type] Brief description (50 chars max)
# Types: feat, fix, docs, refactor, test, chore
```

## Code Style

### Python Code Style

The project aims to follow PEP 8 conventions. Key guidelines:

#### Naming Conventions
```python
# Classes: PascalCase
class DataPreprocessingPipeline:
    pass

# Functions/methods: snake_case
def transform_data():
    pass

# Constants: UPPER_CASE
MAX_FEATURES = 100
DEFAULT_STRATEGY = 'mean'

# Private/internal: leading underscore
def _internal_helper():
    pass
```

#### Code Organization
```python
# 1. Imports (standard library, third-party, local)
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from preprocessing.feature_dropper import FeatureDropper

# 2. Constants
DEFAULT_FEATURES = ['Unnamed: 0', 'Flow ID', 'Timestamp']

# 3. Classes/Functions
class MyTransformer(BaseEstimator, TransformerMixin):
    """Docstring here"""
    pass
```

#### Docstring Format
```python
def transform(self, X):
    """
    Transform input data using the preprocessing pipeline.

    Parameters:
    X (pandas.DataFrame): Input data to be preprocessed.

    Returns:
    pandas.DataFrame: Transformed data.

    Raises:
    ValueError: If input is not a DataFrame.
    """
    pass
```

#### Line Length
- Maximum 100 characters per line
- Break long lines at logical points
- Use implicit line continuation in lists/dicts

### Example: Well-Formatted Code

```python
class NumericalStandardiser(BaseEstimator, TransformerMixin):
    """
    Standardizes numerical features to have mean=0 and std=1.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fit the scaler on the input data.

        Parameters:
        X (pandas.DataFrame): Input data.
        y: Ignored, present for API consistency.

        Returns:
        self: Returns the fitted transformer.
        """
        # Select numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numerical_cols])
        return self

    def transform(self, X):
        """
        Transform data using the fitted scaler.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        pandas.DataFrame: Transformed data.
        """
        df = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.transform(X[numerical_cols])
        return df
```

## Testing

### Running Tests

Currently, the project uses exploratory notebooks for validation. As you add features:

```bash
# Test imports work
python -c "from preprocessing import DataPreprocessingPipeline"

# Test with notebooks
jupyter notebook notebooks/test_your_feature.ipynb

# Validate data pipeline
python -c "
import pandas as pd
from preprocessing import DataPreprocessingPipeline
df = pd.read_csv('data/sample.csv')
pipeline = DataPreprocessingPipeline()
pipeline.fit(df)
result = pipeline.transform(df)
print(f'Input shape: {df.shape}, Output shape: {result.shape}')
"
```

### Testing Guidelines

When adding new code:

1. **Unit Testing** - Test individual functions/methods
2. **Integration Testing** - Test with the full pipeline
3. **Notebook Testing** - Create/update notebooks to demonstrate usage
4. **Edge Cases** - Test with unusual inputs
   - Empty dataframes
   - Single row inputs
   - Missing values
   - All zeros/constants
   - Very large datasets

### Creating Tests

If adding substantial features, create a test notebook:

```bash
# Create new test notebook
jupyter notebook notebooks/test_feature_name.ipynb
```

In the notebook:
```python
import pandas as pd
from preprocessing import DataPreprocessingPipeline

# Test 1: Basic functionality
df = pd.read_csv('data/sample.csv')
pipeline = DataPreprocessingPipeline()
pipeline.fit(df)
result = pipeline.transform(df)
assert result.shape[1] > 0, "Transform should return data"

# Test 2: Edge cases
empty_df = df.iloc[:0]
try:
    pipeline.transform(empty_df)
    print("✓ Handles empty data")
except Exception as e:
    print(f"✗ Failed on empty data: {e}")

# Test 3: Performance
import time
start = time.time()
for _ in range(10):
    pipeline.transform(df)
elapsed = time.time() - start
print(f"Time for 10 transforms: {elapsed:.2f}s")
```

## Documentation

### Updating Documentation

When making changes, update relevant documentation:

1. **README.md** - For major features or setup changes
2. **PREPROCESSING.md** - For preprocessing pipeline changes
3. **MODELS.md** - For model updates
4. **Dataset.md** - For dataset-related information
5. **Code Comments** - For complex logic within code
6. **Docstrings** - For all public functions/classes

### Documentation Standards

- Use clear, accessible language
- Include code examples for features
- Keep formatting consistent
- Link to related sections
- Update table of contents if needed

Example documentation update:

```markdown
## New Feature Name

### Overview
Brief description of what it does.

### Usage

```python
from preprocessing import NewTransformer

transformer = NewTransformer(param1=value1)
transformer.fit(data)
result = transformer.transform(data)
```

### Parameters
- `param1`: Description of parameter 1
- `param2`: Description of parameter 2

### Example
Real-world usage example
```

## Submitting Changes

### Before Submitting

1. **Verify your code works**:
   ```bash
   # Test imports
   python -c "from preprocessing import DataPreprocessingPipeline"
   
   # Run notebooks to test functionality
   jupyter notebook notebook_name.ipynb
   ```

2. **Check code style**:
   - Follow PEP 8 conventions
   - Use meaningful variable names
   - Add comments for complex logic
   - Keep functions small and focused

3. **Update tests/notebooks**:
   - Add or update tests for your changes
   - Ensure existing notebooks still work
   - Document your changes

4. **Update documentation**:
   - Update relevant .md files
   - Add docstrings to new functions
   - Include usage examples

### Commit and Push

```bash
# Commit your changes
git add .
git commit -m "feat: Add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Submit Pull Request

1. Go to GitHub and create a Pull Request
2. Use a descriptive title (e.g., "Add IP octet splitting in composite splitter")
3. In the description, include:
   - What changes were made
   - Why these changes were needed
   - How to test the changes
   - Any breaking changes
   - Relevant issue numbers (if any)

**PR Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How to test these changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Tests pass
- [ ] No breaking changes
```

### Code Review

- Respond to review comments promptly
- Make requested changes in new commits (don't force push)
- Ask for clarification if feedback is unclear
- Be respectful and constructive

## Tips for Successful Contributions

1. **Start Small** - Begin with documentation or small features
2. **Read Existing Code** - Understand the current implementation
3. **Ask Questions** - Open an issue to discuss major changes
4. **Test Thoroughly** - Ensure your changes work
5. **Document Well** - Clear documentation helps maintainers
6. **Be Patient** - Reviews may take time

## Common Contribution Areas

### 1. Documentation Improvements
- Clarify confusing sections
- Add more examples
- Fix typos and formatting
- Add missing information

### 2. Code Improvements
- Bug fixes
- Performance optimizations
- Code cleanup and refactoring
- Add type hints (if applicable)

### 3. New Features
- Additional preprocessing transformers
- New ML models
- Experiment notebooks
- Utility functions

### 4. Tests and Validation
- Create comprehensive test notebooks
- Add edge case testing
- Performance benchmarking

## Questions?

- Open an issue for questions or discussions
- Check existing issues for similar topics
- Review documentation before asking

## Recognition

Contributors will be acknowledged in:
- Project README (contributor list)
- Release notes
- Commit history

Thank you for contributing to DDoS Classifier! 🎉
