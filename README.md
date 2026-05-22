# DDoS Classifier

A machine learning project for classifying DDoS (Distributed Denial of Service) attacks using advanced data preprocessing and deep learning models. This repository provides reusable preprocessing pipelines and explores multiple machine learning approaches to accurately identify DDoS traffic patterns.

## Features

- **Production-ready preprocessing pipeline** - Modular, sklearn-compatible transformers for feature engineering
- **Multiple ML models** - Logistic Regression, Autoencoders, and clustering algorithms
- **Comprehensive data analysis** - Exploratory notebooks demonstrating dataset characteristics and model performance
- **Easy experimentation** - Jupyter notebooks for iterative development and model evaluation
- **Reproducible results** - Poetry-managed dependencies and standardized preprocessing

## Quick Links

- 📚 [Quick Start Guide](QUICKSTART.md) - Get up and running in 5 minutes
- 🔧 [Preprocessing Documentation](PREPROCESSING.md) - Data pipeline API and examples
- 🤖 [Models Documentation](MODELS.md) - Algorithm descriptions and use cases
- 📊 [Dataset Documentation](Dataset.md) - Dataset structure and features
- 🤝 [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

## Tech Stack

- **Python 3.12** - Programming language
- **scikit-learn** - ML algorithms and utilities
- **pandas / numpy** - Data manipulation and numerical computing
- **matplotlib / seaborn** - Data visualization
- **PyTorch** - Deep learning framework (CPU optimized)
- **Poetry** - Dependency management and packaging

## Project Structure

```
├── preprocessing/              # Data preprocessing pipeline and transformers
│   ├── training_pipeline.py    # DataPreprocessingPipeline class
│   ├── feature_dropper.py      # Remove unwanted features
│   ├── composite_splitter.py   # Split IP addresses into octets
│   ├── numerical_standardiser.py # Standardize numerical features
│   ├── category_encoder.py     # Encode categorical variables
│   └── feature_imputer.py      # Handle missing values
├── models/                     # Trained model files
│   ├── logistic_regression.pkl # Trained logistic regression model
│   └── autoencoder_model.pt    # Trained autoencoder model
├── logistic_regression/        # Logistic regression experiments
├── autoencoder/                # Autoencoder experiments
├── notebooks/                  # Additional exploratory notebooks
├── data/                       # Dataset directory (not in repo)
├── data analysis.ipynb         # Main data exploration notebook
├── pipeline usage.ipynb        # Preprocessing pipeline examples
└── README.md                   # This file
```

## Setup

### Prerequisites

- Python 3.12 or higher
- Poetry (install from https://python-poetry.org)

### Installation

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/rtweera/ddos-classifier.git
   cd ddos-classifier
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the Poetry shell**:
   ```bash
   poetry shell
   ```

## Usage

### Using the Preprocessing Pipeline

The main preprocessing pipeline is `DataPreprocessingPipeline` in `preprocessing/training_pipeline.py`. See [PREPROCESSING.md](PREPROCESSING.md) for detailed API documentation.

Quick example:
```python
from preprocessing import DataPreprocessingPipeline
import pandas as pd

# Load your data
df = pd.read_csv('path/to/data.csv')

# Create and fit the pipeline
pipeline = DataPreprocessingPipeline()
pipeline.fit(df)

# Transform new data
transformed = pipeline.transform(df)

# Save for later use
pipeline.save('pipeline.pkl')
```

### Running Experiments

Start exploring with the included notebooks:

```bash
# Main data analysis
jupyter notebook "data analysis.ipynb"

# Pipeline usage examples
jupyter notebook "pipeline usage.ipynb"

# Model-specific experiments
jupyter notebook notebooks/logistic_regression.ipynb
jupyter notebook notebooks/autoencoder.ipynb
```

## Pipeline Overview

The preprocessing pipeline includes the following stages:

1. **Feature Dropping** - Removes irrelevant features (index, Flow ID, Timestamp)
2. **Composite Splitting** - Splits IP addresses into octets for better feature representation
3. **Numerical Standardization** - Standardizes numerical features to mean=0, std=1

See [PREPROCESSING.md](PREPROCESSING.md) for more details on each transformer and customization options.

## Models

The project explores three main approaches:

- **Logistic Regression** - Fast baseline model for binary classification
- **Autoencoder** - Deep learning model for feature learning and anomaly detection
- **DBSCAN** - Clustering algorithm for unsupervised DDoS detection

See [MODELS.md](MODELS.md) for detailed algorithm descriptions and use cases.

## Dataset

This project uses the [Kaggle DDoS Dataset](https://www.kaggle.com/datasets/devendra416/ddos-datasets). See [Dataset.md](Dataset.md) for:
- Dataset structure and schema
- Feature descriptions
- Download instructions
- Data characteristics and statistics

## Troubleshooting

### Issues with PyTorch Installation

If you encounter issues installing PyTorch CPU version, ensure you're using Python 3.12:
```bash
python --version
```

### Memory Issues with Large Datasets

The preprocessing pipeline supports incremental processing:
```python
pipeline = DataPreprocessingPipeline()
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    pipeline.partial_fit(chunk)
```

### Missing Data Files

Download the DDoS dataset from [Kaggle](https://www.kaggle.com/datasets/devendra416/ddos-datasets) and place it in the `data/` directory.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing procedures
- Pull request process

## License

This project is open source. See LICENSE file for details.

## References

- Dataset: [Kaggle DDoS Datasets](https://www.kaggle.com/datasets/devendra416/ddos-datasets)
- scikit-learn: https://scikit-learn.org
- PyTorch: https://pytorch.org

