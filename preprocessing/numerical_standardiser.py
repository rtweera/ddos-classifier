from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalStandardiser(BaseEstimator, TransformerMixin):
    """
    A custom transformer that standardises numerical features in the input data.
    """
    def __init__(self):
        self.numerical_columns = None
        self.scaler = StandardScaler()

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

        return self

    def transform(self, X):
        """
        Applies standardization to the identified numerical features.
        """
        if self.numerical_columns is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        # to avoid modifying the original df.
        df = X.copy()
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        return df
