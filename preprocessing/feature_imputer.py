from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class FeatureImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that imputes missing values in the input data.
    """
    def __init__(self, num_strategy="mean", cat_strategy="most_frequent"):
        """
        Initialize the transformer with the imputation strategies for numerical and categorical features.

        Parameters:
        num_strategy (str): Imputation strategy for numerical features.
        cat_strategy (str): Imputation strategy for categorical features.
        """
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_imputer = None
        self.cat_imputer = None
        self.numerical_features = None
        self.categorical_features = None

    def fit(self, X, y=None):
        """
        Identifies numerical and categorical fields and fits imputers to them.
        """
        # Identify numerical and categorical features
        self.numerical_features = X.select_dtypes(include="number").columns
        self.categorical_features = X.select_dtypes(include=["object", "category"]).columns

        # Create and fit imputers
        self.num_imputer = SimpleImputer(strategy=self.num_strategy)
        self.cat_imputer = SimpleImputer(strategy=self.cat_strategy)

        if not self.numerical_features.empty:
            self.num_imputer.fit(X[self.numerical_features])
        if not self.categorical_features.empty:
            self.cat_imputer.fit(X[self.categorical_features])

        return self

    def transform(self, X):
        """
        Applies the imputations to the respective fields.
        """
        if self.numerical_features is None or self.categorical_features is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        X_copy = X.copy()

        # Impute numerical features
        if not self.numerical_features.empty:
            X_copy[self.numerical_features] = self.num_imputer.transform(X[self.numerical_features])

        # Impute categorical features
        if not self.categorical_features.empty:
            X_copy[self.categorical_features] = self.cat_imputer.transform(X[self.categorical_features])

        return X_copy