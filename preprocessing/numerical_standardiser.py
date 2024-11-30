import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalStandardiser(BaseEstimator, TransformerMixin):
    """
    A custom transformer that standardises numerical features in the input data.
    """
    def __init__(self):
        self.numerical_columns = None
        self.scaler = StandardScaler()
        self.means = {}
        self.vars = {}
        self.counts = {}
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Identifies numerical features and fits the scaler on them.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        """
        # select only numerical cols
        self.numerical_columns = X.select_dtypes(include=['number']).columns

        # fit the scaler
        self.scaler.fit(X[self.numerical_columns])

        self.is_fitted = True

        return self

    def partial_fit(self, X, y=None):
        """
        Partially fits the scaler on the input data.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        """

        for col in X.select_dtypes(include=[np.number]).columns:
            col_data = X[col].to_numpy()

            if col not in self.means:
                self.means[col] = np.mean(col_data)
                self.vars[col] = np.var(col_data)
                self.counts[col] = 1

            n = len(col_data)
            new_mean = col_data.mean()
            new_var = col_data.var(ddof=0)

            total_count = self.counts[col] + n
            delta = new_mean - self.means[col]
            self.vars[col] = (self.counts[col] * self.vars[col] + n * new_var + (delta ** 2) * self.counts[col] * n / total_count) / total_count

            self.means[col] = (self.counts[col] * self.means[col] + n * new_mean) / total_count
            self.counts[col] = total_count
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Applies standardization to the identified numerical features.
        """
        print("testing")
        if not self.is_fitted:
            raise RuntimeError("The transformer has not been fitted yet.")

        # to avoid modifying the original df.
        df = X.copy()
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        return df
