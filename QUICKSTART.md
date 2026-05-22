# Quick Start Guide

Get started with the DDoS Classifier in 5 minutes!

## Installation (2 minutes)

### 1. Prerequisites

- Python 3.12+
- Git
- Poetry (install from https://python-poetry.org)

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/rtweera/ddos-classifier.git
cd ddos-classifier

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

### 3. Verify Installation

```bash
# Test that everything is set up correctly
python -c "from preprocessing import DataPreprocessingPipeline; print('✓ Ready to go!')"
```

## Quick Example 1: Using the Preprocessing Pipeline (2 minutes)

The preprocessing pipeline cleans and transforms raw DDoS traffic data:

```python
import pandas as pd
from preprocessing import DataPreprocessingPipeline

# Step 1: Load your data
df = pd.read_csv('data/ddos_traffic.csv')
print(f"Loaded {len(df)} network flows with {len(df.columns)} features")

# Step 2: Create the preprocessing pipeline
pipeline = DataPreprocessingPipeline()

# Step 3: Fit on your data (learns the transformation parameters)
pipeline.fit(df)

# Step 4: Transform the data
transformed_df = pipeline.transform(df)
print(f"Transformed data shape: {transformed_df.shape}")

# Step 5: Transform new data using the fitted pipeline
new_data = pd.read_csv('data/new_traffic.csv')
transformed_new = pipeline.transform(new_data)
```

## Quick Example 2: Training a DDoS Classifier (3 minutes)

Build a simple DDoS detection model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import DataPreprocessingPipeline

# Load data
df = pd.read_csv('data/ddos_traffic.csv')
X = df.drop('Label', axis=1)
y = (df['Label'] == 'DDoS').astype(int)  # 1 for DDoS, 0 for Benign

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess
pipeline = DataPreprocessingPipeline()
pipeline.fit(X_train)
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
print(classification_report(y_test, y_pred, 
                          target_names=['Benign', 'DDoS']))
```

## Quick Example 3: Making Predictions on New Data (1 minute)

Use a trained model to detect DDoS attacks:

```python
import pandas as pd
import joblib

# Load trained models
pipeline = joblib.load('models/preprocessing_pipeline.pkl')
model = joblib.load('models/logistic_regression.pkl')

# New network flow data arrives
new_flow = pd.read_csv('data/incoming_traffic.csv')

# Preprocess
processed = pipeline.transform(new_flow)

# Predict
is_ddos = model.predict(processed)
confidence = model.predict_proba(processed)[:, 1]

# Display results
for i, (flow, pred, conf) in enumerate(zip(new_flow.iterrows(), is_ddos, confidence)):
    status = "🚨 DDoS DETECTED" if pred else "✓ Normal"
    print(f"Flow {i}: {status} (confidence: {conf:.2%})")
```

## Running Exploratory Notebooks

Learn by exploring with Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Open and run these notebooks:
# - "data analysis.ipynb" - Explore the dataset
# - "pipeline usage.ipynb" - Preprocessing pipeline examples
# - "notebooks/logistic_regression.ipynb" - Model training
# - "notebooks/autoencoder.ipynb" - Deep learning approach
```

## Next Steps

1. **Preprocessing Details** → Read [PREPROCESSING.md](PREPROCESSING.md)
2. **Model Information** → Read [MODELS.md](MODELS.md)
3. **Dataset Details** → Read [Dataset.md](Dataset.md)
4. **Contributing** → See [CONTRIBUTING.md](CONTRIBUTING.md)
5. **Full README** → Check [README.md](README.md)

## Common Tasks

### Task 1: Download Dataset

```bash
# Download from Kaggle DDoS Dataset
# https://www.kaggle.com/datasets/devendra416/ddos-datasets
# Extract to data/ directory
unzip ddos_dataset.zip -d data/
```

### Task 2: Save Trained Pipeline for Later Use

```python
from preprocessing import DataPreprocessingPipeline

pipeline = DataPreprocessingPipeline()
pipeline.fit(X_train)

# Save
pipeline.save('my_trained_pipeline.pkl')

# Later: Load and use
pipeline = DataPreprocessingPipeline()
pipeline.load('my_trained_pipeline.pkl')
result = pipeline.transform(new_data)
```

### Task 3: Handle Large Datasets

```python
from preprocessing import DataPreprocessingPipeline

pipeline = DataPreprocessingPipeline()

# Process file in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=50000):
    pipeline.partial_fit(chunk)

# Now use for transformation
processed = pipeline.transform(new_chunk)
```

### Task 4: Evaluate Model Performance

```python
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'preprocessing'"

**Solution:** Make sure you're in the Poetry shell:
```bash
poetry shell
```

### Issue: "FileNotFoundError: data/ddos_traffic.csv"

**Solution:** Download the dataset:
1. Visit [Kaggle DDoS Dataset](https://www.kaggle.com/datasets/devendra416/ddos-datasets)
2. Download and extract to `data/` directory
3. See [Dataset.md](Dataset.md) for details

### Issue: "Python 3.12 not found"

**Solution:** Update Python or check your version:
```bash
python --version  # Should be 3.12 or higher
poetry env info    # Check Poetry environment details
```

### Issue: "The kernel has crashed" in Jupyter

**Solution:** Restart kernel and clear cell outputs:
```bash
# In Jupyter: Kernel → Restart Kernel
# Then: Cell → All Output → Clear
```

## Tips for Success

1. **Start Simple** - Begin with the logistic regression examples
2. **Explore Data** - Run the data analysis notebook first
3. **Check Shapes** - Always verify data shapes match expectations
4. **Save Models** - Save trained pipelines and models for reuse
5. **Read Docs** - Check PREPROCESSING.md and MODELS.md for details
6. **Use Notebooks** - Jupyter is great for experimentation

## Code Examples Cheat Sheet

### Import Everything You Need
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing import DataPreprocessingPipeline
import joblib
```

### Full Workflow
```python
# Load
df = pd.read_csv('data.csv')
X, y = df.drop('Label', axis=1), df['Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
pipe = DataPreprocessingPipeline()
pipe.fit(X_train)
X_train, X_test = pipe.transform(X_train), pipe.transform(X_test)

# Train
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(pipe, 'pipeline.pkl')

# Load later
model = joblib.load('model.pkl')
pipe = joblib.load('pipeline.pkl')
```

## Getting Help

- 📖 Read [PREPROCESSING.md](PREPROCESSING.md) for pipeline details
- 🤖 Check [MODELS.md](MODELS.md) for model information
- 📊 See [Dataset.md](Dataset.md) for data details
- 🤝 Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- 💬 Open an issue on GitHub for questions
- 📚 Check the notebook examples for more code samples

## Performance Expectations

On typical DDoS datasets:
- **Logistic Regression**: ~95-97% accuracy, <1 second to train
- **Autoencoder**: ~96-98% accuracy, 1-5 minutes to train
- **DBSCAN**: ~90-96% accuracy (unsupervised), <1 second

*Actual performance depends on your data and hyperparameters*

---

**Happy classifying! 🎉** If you have questions, check the [README](README.md) or other documentation files.
