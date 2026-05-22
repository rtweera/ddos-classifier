# Preprocessing Pipeline Documentation

## Overview

The preprocessing pipeline is the core component for preparing raw DDoS traffic data for machine learning models. It provides a modular, scikit-learn compatible pipeline that handles feature engineering, missing value imputation, categorical encoding, and numerical standardization.

## DataPreprocessingPipeline Class

The main entry point is the `DataPreprocessingPipeline` class located in `preprocessing/training_pipeline.py`.

### Initialization

```python
from preprocessing import DataPreprocessingPipeline

# Default configuration
pipeline = DataPreprocessingPipeline()

# Custom imputation strategies
pipeline = DataPreprocessingPipeline(
    numerical_imputer_strategy='median',      # 'mean', 'median', or 'most_frequent'
    categorical_imputer_strategy='most_frequent'
)
```

### Basic Usage

```python
import pandas as pd
from preprocessing import DataPreprocessingPipeline

# Load your data
df = pd.read_csv('data/ddos_traffic.csv')

# Create pipeline
pipeline = DataPreprocessingPipeline()

# Fit on training data
pipeline.fit(df)

# Transform data
transformed_df = pipeline.transform(df)

# Transform new data (test/validation)
new_data = pd.read_csv('data/test_data.csv')
transformed_test = pipeline.transform(new_data)
```

### API Reference

#### `fit(X)`
Fit the preprocessing pipeline on input data.

**Parameters:**
- `X` (pandas.DataFrame): Input data to be preprocessed

**Returns:**
- self: Returns the fitted pipeline for method chaining

**Example:**
```python
pipeline.fit(training_data)
```

#### `transform(X)`
Transform input data using the fitted pipeline.

**Parameters:**
- `X` (pandas.DataFrame): Input data to be preprocessed

**Returns:**
- pandas.DataFrame: Transformed data

**Example:**
```python
transformed = pipeline.transform(new_data)
```

#### `partial_fit(X)`
Partially fit the pipeline on new data (useful for streaming/incremental learning).

**Parameters:**
- `X` (pandas.DataFrame): Data chunk to fit

**Returns:**
- None

**Example:**
```python
# Process large file in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    pipeline.partial_fit(chunk)
```

#### `save(path)`
Save the fitted pipeline to a file for later use.

**Parameters:**
- `path` (str): File path to save the pipeline

**Raises:**
- ValueError: If no path is provided

**Example:**
```python
pipeline.save('models/preprocessing_pipeline.pkl')
```

#### `load(path)`
Load a previously saved pipeline.

**Parameters:**
- `path` (str): File path to load the pipeline from

**Returns:**
- self: Returns the loaded pipeline for method chaining

**Raises:**
- ValueError: If no path is provided

**Example:**
```python
pipeline = DataPreprocessingPipeline()
pipeline.load('models/preprocessing_pipeline.pkl')
transformed = pipeline.transform(new_data)
```

## Pipeline Stages

The pipeline executes the following transformations in order:

### 1. FeatureDropper
Removes non-predictive or irrelevant features from the dataset.

**Default Features Dropped:**
- `Unnamed: 0` - Index column from CSV
- `Flow ID` - Unique flow identifier (not predictive)
- `Timestamp` - Temporal information (often removed for generalization)

**Class:** `preprocessing.feature_dropper.FeatureDropper`

**Customization:**
```python
from preprocessing.feature_dropper import FeatureDropper

# Use custom feature list
dropper = FeatureDropper(features_to_drop=['col1', 'col2', 'col3'])

# Or as part of pipeline modification:
pipeline.preprocessor.named_steps['feature_dropper'].features_to_drop = ['custom_features']
```

### 2. CompositeSplitter
Splits composite features (IP addresses) into individual octets for better numerical representation.

**Default Composite Features Split:**
- `Src IP` - Split into `Src IP_octet1`, `Src IP_octet2`, `Src IP_octet3`, `Src IP_octet4`
- `Dst IP` - Split into `Dst IP_octet1`, `Dst IP_octet2`, `Dst IP_octet3`, `Dst IP_octet4`

**Class:** `preprocessing.composite_splitter.CompositeSplitter`

**Why This Matters:**
- IP addresses are composite features containing network information
- Splitting into octets preserves network topology information
- Each octet (0-255) represents different network segments
- Helps ML models learn network-based patterns in DDoS attacks

**Customization:**
```python
from preprocessing.composite_splitter import CompositeSplitter

# Specify which IP columns to split
splitter = CompositeSplitter(composite_features=['Src IP', 'Dst IP'])

# The class also includes a timestamp_splitter method (not used in default pipeline):
# Splits timestamp into: DT Year, DT Month, DT Day, DT Hour, DT Minute, DT Second
```

**Example Output:**
```
Original:
Src IP            Dst IP
192.168.1.100     10.0.0.50

After splitting:
Src IP_octet1  Src IP_octet2  Src IP_octet3  Src IP_octet4  Dst IP_octet1  Dst IP_octet2  Dst IP_octet3  Dst IP_octet4
192            168            1              100            10             0              0              50
```

### 3. NumericalStandardiser
Standardizes numerical features to have mean=0 and standard deviation=1.

**Class:** `preprocessing.numerical_standardiser.NumericalStandardiser`

**Why Standardization Matters:**
- Many ML algorithms perform better with standardized features
- Prevents features with large scales from dominating learning
- Required for distance-based algorithms (KNN, K-Means)
- Improves convergence in gradient descent optimization

**What It Does:**
- For each numerical feature: `(x - mean) / std_dev`
- Fitted on training data, applied to all datasets
- Uses scikit-learn's StandardScaler internally

**Example:**
```
Before standardization:
Flow Duration: 0 to 10,000,000 microseconds
Total Fwd Packets: 0 to 10,000 packets

After standardization:
Flow Duration: -0.5 to 2.3 (approximately)
Total Fwd Packets: -0.3 to 1.8 (approximately)
```

### Optional Stages (Currently Disabled)

These stages are included in the code but disabled by default because the dataset is clean:

#### FeatureImputer
Handles missing values in the dataset.

**Strategies:**
- `mean` - Replace with mean value (numerical features)
- `median` - Replace with median value (numerical features)
- `most_frequent` - Replace with mode (categorical features)

**To Enable:**
```python
# Modify the preprocessing pipeline in training_pipeline.py
self.preprocessor = Pipeline([
    ('feature_dropper', self.feature_dropper),
    ('composite_splitter', CompositeSplitter()),
    ('feature_imputer', self.feature_imputer),  # Uncomment this line
    ('numerical_standardiser', self.numerical_standardiser)
])
```

#### CategoryEncoder
Encodes categorical variables to numerical format.

**Encoding Methods:**
- One-hot encoding for low-cardinality features
- Label encoding for high-cardinality features

**To Enable:**
```python
self.preprocessor = Pipeline([
    ('feature_dropper', self.feature_dropper),
    ('composite_splitter', CompositeSplitter()),
    ('category_encoder', self.category_encoder),  # Uncomment this line
    ('numerical_standardiser', self.numerical_standardiser)
])
```

## Advanced Usage

### Complete Training Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import DataPreprocessingPipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('data/ddos_traffic.csv')
X = df.drop('Label', axis=1)
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit preprocessing pipeline
pipeline = DataPreprocessingPipeline()
pipeline.fit(X_train)

# Transform data
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train classifier
clf = LogisticRegression(random_state=42)
clf.fit(X_train_processed, y_train)

# Evaluate
score = clf.score(X_test_processed, y_test)
print(f"Accuracy: {score:.4f}")

# Save for deployment
pipeline.save('models/ddos_preprocessing_pipeline.pkl')
```

### Processing Large Datasets with Partial Fit

```python
from preprocessing import DataPreprocessingPipeline

pipeline = DataPreprocessingPipeline()

# Process file in chunks
for chunk in pd.read_csv('large_ddos_dataset.csv', chunksize=50000):
    pipeline.partial_fit(chunk)

# Now use for transformation
result = pipeline.transform(chunk)
```

### Pipeline Integration with scikit-learn GridSearchCV

```python
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from preprocessing import DataPreprocessingPipeline

# Create combined pipeline
combined_pipeline = SKPipeline([
    ('preprocessing', DataPreprocessingPipeline()),
    ('classifier', LogisticRegression())
])

# Grid search
params = {
    'preprocessing__numerical_imputer_strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10.0]
}

gs = GridSearchCV(combined_pipeline, params, cv=5)
gs.fit(X_train, y_train)
print(f"Best parameters: {gs.best_params_}")
```

### Loading and Using Saved Pipeline

```python
from preprocessing import DataPreprocessingPipeline

# Load the saved pipeline
pipeline = DataPreprocessingPipeline()
pipeline.load('models/ddos_preprocessing_pipeline.pkl')

# Use it on new data
new_data = pd.read_csv('new_traffic_data.csv')
processed_data = pipeline.transform(new_data)

# Now can feed to trained models
predictions = trained_model.predict(processed_data)
```

## Implementation Details

### Base Classes

All custom transformers inherit from scikit-learn base classes:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """Learn parameters from X"""
        return self
    
    def transform(self, X):
        """Transform X using learned parameters"""
        return X
```

This ensures compatibility with scikit-learn's `Pipeline`, `GridSearchCV`, and other tools.

## Troubleshooting

### Issue: Pipeline throws error on missing features

**Cause:** Transform data has different columns than fit data

**Solution:**
```python
# Ensure column order and selection matches
X_train_cols = preprocessing.preprocessor.named_steps['feature_dropper'].features_to_drop
X_test = X_test[X_train.columns]
```

### Issue: Standardization causes NaN values

**Cause:** Feature has zero standard deviation (constant value)

**Solution:**
```python
# Drop constant features before pipeline
X = X.loc[:, (X.std() > 0)]
```

### Issue: Memory error with large datasets

**Solution:** Use partial_fit with smaller chunks:
```python
for chunk in pd.read_csv('file.csv', chunksize=25000):
    pipeline.partial_fit(chunk)
```

## Related Files

- `preprocessing/training_pipeline.py` - Main pipeline implementation
- `preprocessing/feature_dropper.py` - Feature removal transformer
- `preprocessing/composite_splitter.py` - IP address splitting transformer
- `preprocessing/numerical_standardiser.py` - Feature standardization
- `preprocessing/feature_imputer.py` - Missing value handling
- `preprocessing/category_encoder.py` - Categorical encoding
- `pipeline usage.ipynb` - Interactive usage examples
- `data analysis.ipynb` - Data exploration with preprocessing

## References

- [scikit-learn Pipeline documentation](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [Feature Scaling and Standardization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-centering-and-scaling)
- [BaseEstimator and TransformerMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)
