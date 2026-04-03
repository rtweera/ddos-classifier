# DDoS Classifier

A machine learning project for preprocessing and experimenting with DDoS traffic classification workflows.

## Dataset

- [Kaggle DDoS Dataset](https://www.kaggle.com/datasets/devendra416/ddos-datasets)

## Tech Stack

- Python 3.12
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- PyTorch (CPU)
- Poetry

## Project Structure

- `preprocessing/` - reusable preprocessing transformers and pipeline
  - `training_pipeline.py` (`DataPreprocessingPipeline`)
  - `feature_dropper.py`
  - `composite_splitter.py`
  - `numerical_standardiser.py`
- `notebooks/` and root notebooks - experimentation and analysis
- `data/` - dataset-related package directory
- `models/` - model-related package directory
- `Dataset.md` - dataset link reference

## Setup

1. Install Poetry
2. Install dependencies:

```bash
poetry install
```

3. Start a shell:

```bash
poetry shell
```

## Usage Notes

- The preprocessing flow is centered on `DataPreprocessingPipeline` in `preprocessing/training_pipeline.py`.
- Current pipeline stages include feature dropping, composite feature splitting (IP columns), and numerical standardization.
- Several exploratory notebooks in the repository demonstrate data analysis and modeling experiments.

