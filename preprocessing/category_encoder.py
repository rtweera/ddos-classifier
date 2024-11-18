from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class CategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer that identifies categorical fields in a dataset
    and applies one-hot encoding to them using scikit-learn's OneHotEncoder.
    """
    def __init__(self, handle_unknown='ignore', sparse_output=False):
        """
        Parameters:
        handle_unknown: How to handle unknown categories during transform (default: 'ignore').
        sparse_output: Whether to return sparse matrix or dense array (default: False).
        """
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.encoder = None
        self.categorical_features = None

    def fit(self, X, y=None):
        """
        Identifies categorical features and fits the OneHotEncoder on them.
        """
        # Identify categorical features
        self.categorical_features = X.select_dtypes(include=["object", "category"]).columns

        # Initialize and fit the OneHotEncoder
        self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=self.sparse_output)
        self.encoder.fit(X[self.categorical_features])

        return self

    def transform(self, X):
        """
        Applies one-hot encoding to the categorical fields.
        """
        if self.categorical_features is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        # Ensure input is a DataFrame
        X_copy = X.copy()

        # Encode categorical features
        encoded_array = self.encoder.transform(X_copy[self.categorical_features])
        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(self.categorical_features), index=X_copy.index)

        # Drop original categorical columns and concatenate encoded columns horizontally with the rest of the data
        X_copy = X_copy.drop(columns=self.categorical_features)
        X_copy = pd.concat([X_copy, encoded_df], axis=1)

        return X_copy
