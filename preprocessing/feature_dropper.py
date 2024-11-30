from sklearn.base import BaseEstimator, TransformerMixin

class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    A custom transformer that drops specified features from the input data.
    """
    def __init__(self, features_to_drop=None):
        if features_to_drop is None:
            features_to_drop = ['Unnamed: 0', 'Flow ID', 'Timestamp']    # INFO: possibly need to drop Timestamp as well
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        """
        Fit the transformer by doing nothing.
        """
        return self

    def transform(self, X):
        """
        Drop specified features from the input data.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        pandas.DataFrame: Transformed data with specified features dropped.
        """
        df = X.copy()
        return df.drop(columns=self.features_to_drop)

    def partial_fit(self, X, y=None):
        """
        Partially fit the transformer by doing nothing.
        """
        return self