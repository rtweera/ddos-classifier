# Machine Learning Models for DDoS Classification

## Overview

This project explores multiple machine learning approaches to detect and classify DDoS attacks from network traffic data. Each model offers different trade-offs between accuracy, interpretability, and computational efficiency.

## Models Summary

| Model | Type | Approach | Speed | Accuracy | Best For |
|-------|------|----------|-------|----------|----------|
| Logistic Regression | Classification | Statistical | ⚡⚡⚡ | ⭐⭐⭐ | Baseline, Interpretability |
| Autoencoder | Deep Learning | Feature Learning | ⚡⚡ | ⭐⭐⭐⭐ | Anomaly Detection |
| DBSCAN | Clustering | Unsupervised | ⚡⚡ | ⭐⭐⭐ | Outlier Detection |

## 1. Logistic Regression

### Overview

Logistic Regression is a statistical learning algorithm for binary classification. It models the probability of a sample belonging to a class using the logistic function.

**File Location:** `logistic_regression/` directory

### Algorithm Details

#### How It Works

1. **Linear Transformation**: Maps input features to a linear combination
   ```
   z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
   ```

2. **Logistic Transformation**: Applies sigmoid function to convert z to probability [0, 1]
   ```
   P(Y=1|X) = 1 / (1 + e^(-z))
   ```

3. **Decision Boundary**: Classifies as 1 if P > 0.5, else 0

#### Advantages
- ✅ **Fast Training**: Efficient with large datasets
- ✅ **Interpretable**: Clear feature coefficients show feature importance
- ✅ **Probabilistic Output**: Provides confidence scores
- ✅ **No Distribution Assumptions**: Works with any data distribution
- ✅ **Good Baseline**: Excellent as a baseline classifier

#### Disadvantages
- ❌ **Linear Separation**: Assumes features are linearly separable
- ❌ **Limited Feature Interactions**: Doesn't capture complex relationships
- ❌ **Sensitive to Scaling**: Requires standardized features (handled by pipeline)
- ❌ **May Underfit**: Complex non-linear DDoS patterns might be missed

### Configuration

```python
from sklearn.linear_model import LogisticRegression

# Standard configuration
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'  # or 'liblinear', 'newton-cg', 'sag', 'saga'
)

# For imbalanced data (more benign than DDoS)
model = LogisticRegression(
    class_weight='balanced',  # Adjust weights based on class frequency
    random_state=42
)

# L2 Regularization (default)
model = LogisticRegression(
    C=1.0,  # Inverse of regularization strength (higher C = less regularization)
    penalty='l2',
    random_state=42
)
```

### Usage Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from preprocessing import DataPreprocessingPipeline

# Load and preprocess data
df = pd.read_csv('data/ddos_traffic.csv')
X = df.drop('Label', axis=1)
y = (df['Label'] == 'DDoS').astype(int)  # Binary: 1 for DDoS, 0 for Benign

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
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = model.coef_[0]
# Features with larger absolute coefficients are more important
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

gs = GridSearchCV(
    LogisticRegression(random_state=42),
    params,
    cv=5,
    scoring='roc_auc'
)
gs.fit(X_train_processed, y_train)
print(f"Best parameters: {gs.best_params_}")
print(f"Best CV Score: {gs.best_score_:.4f}")
```

### Performance Metrics

Monitor these metrics for Logistic Regression:

- **Accuracy**: Percentage of correct predictions (watch for class imbalance!)
- **Precision**: Of predicted DDoS, how many are actually DDoS
- **Recall**: Of actual DDoS, how many we detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Trade-off between true positive and false positive rates
- **Confusion Matrix**: Shows true positives, false positives, etc.

### Real-world Application

```python
# In production: load trained model
import joblib

model = joblib.load('models/logistic_regression.pkl')
pipeline = DataPreprocessingPipeline()
pipeline.load('models/preprocessing_pipeline.pkl')

# New network flow arrives
new_flow = pd.read_csv('new_traffic.csv')
processed = pipeline.transform(new_flow)

# Predict
is_ddos = model.predict(processed)
confidence = model.predict_proba(processed)[:, 1]

print(f"DDoS Detected: {bool(is_ddos[0])}")
print(f"Confidence: {confidence[0]:.4f}")
```

---

## 2. Autoencoder

### Overview

An Autoencoder is a deep neural network that learns efficient data representation (encoding) in an unsupervised manner. It can detect anomalies (DDoS attacks) by identifying flows that don't match the learned "normal" pattern.

**File Location:** `autoencoder/` directory, `models/autoencoder_model.pt`

### Algorithm Details

#### Architecture

```
INPUT → Encoder → Bottleneck → Decoder → OUTPUT
(input)  (compress)  (latent)   (expand)  (reconstructed)
```

#### How It Works

1. **Encoding Phase**: Compresses input features into a lower-dimensional latent space
2. **Bottleneck**: Learns most important features in compressed form
3. **Decoding Phase**: Reconstructs original input from latent representation
4. **Training**: Minimizes reconstruction error on normal traffic
5. **Anomaly Detection**: High reconstruction error indicates anomalous (DDoS) traffic

#### Loss Function

```
Total Loss = MSE(input, reconstructed_output)
```

#### Advantages
- ✅ **Unsupervised Learning**: Learns from unlabeled normal traffic
- ✅ **Anomaly Detection**: Doesn't need labeled DDoS examples
- ✅ **Non-linear Patterns**: Captures complex relationships
- ✅ **Feature Learning**: Automatically learns useful features
- ✅ **Adaptable**: Can be retrained on new normal patterns

#### Disadvantages
- ❌ **Training Time**: Requires more computation than regression
- ❌ **Hyperparameter Tuning**: Multiple architecture choices
- ❌ **Interpretability**: Black box - hard to explain decisions
- ❌ **Threshold Selection**: Need to choose reconstruction error threshold
- ❌ **Requires Normal Data**: Should be trained only on benign traffic

### Architecture Configuration

```python
import torch
import torch.nn as nn

class DDoSAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# Example initialization
input_size = 80  # After preprocessing
hidden_size = 64
bottleneck_size = 32

model = DDoSAutoencoder(input_size, hidden_size, bottleneck_size)
```

### Training Process

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Prepare data (normal traffic only)
normal_traffic = X_train_processed[y_train == 0]
normal_tensor = torch.FloatTensor(normal_traffic.values)
dataset = TensorDataset(normal_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training
model = DDoSAutoencoder(80, 64, 32)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        
        # Forward pass
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), 'models/autoencoder_model.pt')
```

### Anomaly Detection with Autoencoder

```python
import torch
import numpy as np

# Load trained model
model = DDoSAutoencoder(80, 64, 32)
model.load_state_dict(torch.load('models/autoencoder_model.pt'))
model.eval()

# Calculate reconstruction error threshold (on validation set)
def get_reconstruction_errors(data, model):
    errors = []
    with torch.no_grad():
        for sample in torch.FloatTensor(data.values):
            reconstructed = model(sample.unsqueeze(0))
            error = torch.mean((sample - reconstructed) ** 2).item()
            errors.append(error)
    return np.array(errors)

val_normal_errors = get_reconstruction_errors(X_val[y_val == 0], model)
threshold = np.percentile(val_normal_errors, 95)  # 95th percentile

# Predict on test data
test_errors = get_reconstruction_errors(X_test_processed, model)
predictions = (test_errors > threshold).astype(int)

print(f"Reconstruction Error Threshold: {threshold:.4f}")
print(f"Detected {predictions.sum()} anomalies out of {len(predictions)} flows")

# Evaluate
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, test_errors)
print(f"ROC-AUC Score: {auc:.4f}")
```

### Interpretation

```python
# Analyze what model learned
model.eval()
with torch.no_grad():
    # Sample normal flow
    normal_sample = torch.FloatTensor(X_train_processed[y_train == 0].iloc[0:1].values)
    normal_reconstructed = model(normal_sample)
    normal_error = torch.mean((normal_sample - normal_reconstructed) ** 2).item()
    
    # Sample DDoS flow
    ddos_sample = torch.FloatTensor(X_train_processed[y_train == 1].iloc[0:1].values)
    ddos_reconstructed = model(ddos_sample)
    ddos_error = torch.mean((ddos_sample - ddos_reconstructed) ** 2).item()

print(f"Normal flow reconstruction error: {normal_error:.6f}")
print(f"DDoS flow reconstruction error: {ddos_error:.6f}")
print(f"Error ratio: {ddos_error/normal_error:.2f}x")
```

---

## 3. DBSCAN (Density-Based Spatial Clustering)

### Overview

DBSCAN is an unsupervised clustering algorithm that groups points based on density. It identifies outliers (potential DDoS attacks) as points in low-density regions.

**File Location:** `notebooks/dbscan.ipynb`

### Algorithm Details

#### How It Works

1. **Core Points**: Points with at least `min_samples` neighbors within `eps` distance
2. **Border Points**: Non-core points near core points
3. **Noise/Outliers**: Points not reachable from any core point
4. **Clusters**: Groups of core and border points

#### Parameters

- **eps (ε)**: Maximum distance between neighbors (radius of search)
- **min_samples**: Minimum number of points to form a dense region
  - Default: `2 * num_features`
  - Often set to `2 * min(dim) - 1`

#### Advantages
- ✅ **No Predefined Clusters**: Discovers clusters automatically
- ✅ **Outlier Detection**: Naturally identifies anomalies
- ✅ **Arbitrary Shapes**: Finds non-spherical clusters
- ✅ **Unsupervised**: Requires no labeled data
- ✅ **Interpretable**: Clear cluster assignments

#### Disadvantages
- ❌ **Parameter Sensitivity**: Very sensitive to `eps` and `min_samples`
- ❌ **Density Variation**: Struggles with varying density clusters
- ❌ **High-dimensional Data**: Distance metrics become unreliable in high dimensions
- ❌ **Scalability**: O(n²) complexity with naive implementation
- ❌ **Difficult Tuning**: Requires experimentation to find good parameters

### Parameter Selection

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Method 1: K-distance graph (k-NN)
neighbors = NearestNeighbors(n_neighbors=5)
distances = neighbors.fit(X_processed).kneighbors_graph(X_processed).data
distances.sort()

plt.plot(distances)
plt.ylabel('K-NN Distance')
plt.xlabel('Data Points (sorted)')
plt.title('K-distance Graph for eps Selection')
plt.show()
# Choose eps at the "elbow" point

# Method 2: Calculate expected eps
k = 5  # min_samples
eps_estimate = np.percentile(distances, 90)  # 90th percentile
print(f"Estimated eps: {eps_estimate:.4f}")
```

### Basic Usage

```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Create and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_processed)

# Analyze results
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_outliers = list(clusters).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of outliers (potential DDoS): {n_outliers}")
print(f"Outlier percentage: {100*n_outliers/len(clusters):.2f}%")

# Evaluate if labeled data available
if y_true is not None:
    score = silhouette_score(X_processed, clusters)
    print(f"Silhouette Score: {score:.4f}")
```

### Hyperparameter Tuning

```python
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import pandas as pd

# Test multiple eps and min_samples values
results = []
for eps in [0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 10, 20]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_processed)
        
        if len(set(clusters)) > 1:
            # Davies-Bouldin Index (lower is better)
            db_score = davies_bouldin_score(X_processed, clusters)
            # Calinski-Harabasz Index (higher is better)
            ch_score = calinski_harabasz_score(X_processed, clusters)
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                'outliers': list(clusters).count(-1),
                'DB_Index': db_score,
                'CH_Index': ch_score
            })

results_df = pd.DataFrame(results)
print(results_df.sort_values('DB_Index'))
```

### DDoS Detection with DBSCAN

```python
from sklearn.cluster import DBSCAN

# Train on normal traffic (assuming labeled data or time-based split)
normal_data = X_processed[y == 'BENIGN']

# Fit DBSCAN on normal traffic to learn "normal" density
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan.fit(normal_data)

# Predict on all data
all_clusters = dbscan.fit_predict(X_processed)

# Outliers (-1 label) are anomalies
anomalies = (all_clusters == -1)

print(f"Detected {anomalies.sum()} anomalies")
print(f"Anomaly rate: {100*anomalies.sum()/len(X_processed):.2f}%")

# Compare with actual labels if available
if y is not None:
    accuracy = (anomalies.astype(int) == (y == 'DDoS').astype(int)).mean()
    print(f"Detection accuracy: {accuracy:.4f}")
```

---

## Model Comparison and Selection

### When to Use Each Model

| Scenario | Best Model | Reason |
|----------|-----------|--------|
| Need fast baseline | Logistic Regression | Quick to train, interpretable |
| Need best accuracy | Autoencoder | Learns complex patterns |
| Limited labeled data | Autoencoder or DBSCAN | Can use unsupervised learning |
| Need interpretability | Logistic Regression | Clear feature importance |
| Real-time prediction | Logistic Regression | Fastest inference |
| Detect novel attacks | Autoencoder | Anomaly detection approach |
| Resource constrained | Logistic Regression | Minimal memory/CPU |

### Ensemble Approach

Combine models for better performance:

```python
from sklearn.ensemble import VotingClassifier

# Ensemble (requires labeled data for voting models)
ensemble = VotingClassifier(
    estimators=[
        ('logistic', LogisticRegression()),
        # Add more models as needed
    ],
    voting='soft'  # Use probability voting
)
ensemble.fit(X_train_processed, y_train)
predictions = ensemble.predict(X_test_processed)
```

---

## Model Training and Deployment

### Complete Workflow

```python
# 1. Data preparation
pipeline = DataPreprocessingPipeline()
pipeline.fit(X_train)
X_train_proc = pipeline.transform(X_train)
X_test_proc = pipeline.transform(X_test)

# 2. Train multiple models
lr_model = LogisticRegression(class_weight='balanced')
lr_model.fit(X_train_proc, y_train)

# 3. Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, lr_model.predict(X_test_proc)))

# 4. Save
import joblib
joblib.dump(lr_model, 'models/logistic_regression.pkl')
joblib.dump(pipeline, 'models/preprocessing_pipeline.pkl')
```

### Production Prediction

```python
import joblib

# Load models
pipeline = joblib.load('models/preprocessing_pipeline.pkl')
model = joblib.load('models/logistic_regression.pkl')

# New data arrives
new_flow = pd.read_csv('network_flow.csv')
processed = pipeline.transform(new_flow)

# Predict
is_ddos = model.predict(processed)
confidence = model.predict_proba(processed)

# Take action
if is_ddos[0]:
    print(f"⚠️  DDoS DETECTED (confidence: {confidence[0, 1]:.2%})")
    # Block traffic, alert, etc.
else:
    print(f"✓ Normal traffic (confidence: {confidence[0, 0]:.2%})")
```

---

## Performance Benchmarks

Typical performance on DDoS datasets:

| Model | Accuracy | Precision | Recall | F1-Score | Speed |
|-------|----------|-----------|--------|----------|-------|
| Logistic Regression | 95-97% | 93-95% | 94-96% | 0.94-0.96 | Very Fast |
| Autoencoder | 96-98% | 94-97% | 95-98% | 0.95-0.98 | Fast |
| DBSCAN | 90-96% | 88-94% | 89-95% | 0.90-0.95 | Fast |

*Note: Exact values depend on data characteristics, preprocessing, and hyperparameters*

---

## References and Further Reading

- [Logistic Regression - Scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Autoencoders - PyTorch Tutorial](https://pytorch.org/tutorials/beginner/autoencoder_tutorial.html)
- [DBSCAN - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [DDoS Detection Papers - Google Scholar](https://scholar.google.com/scholar?q=DDoS+detection+machine+learning)
- Project Notebooks: `log-reg-autoencoder-lstm-for-ddos.ipynb`, `notebooks/logistic_regression.ipynb`
