import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class CompositeSplitter(BaseEstimator, TransformerMixin):
    """
    A custom transformer that transforms the input data's composite features into separate features.
    """
    def __init__(self, composite_features=None):
        if composite_features is None:
            composite_features = ['Src IP', 'Dst IP']    # INFO: possibly need to drop Timestamp as well
        self.composite_features = composite_features

    def fit(self, X, y=None):
        """
        Fit the transformer by doing nothing.
        """
        return self

    def partial_fit(self, X, y=None):
        """
        Partially fit the transformer by doing nothing.
        """
        return self

    def transform(self, X):
        """
        Split the input data's composite features into separate features.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        pandas.DataFrame: Transformed data with composite features split into separate features.
        """
        df = X.copy()
        for feature in self.composite_features:
            df = self.ip_splitter(df, feature)

        return df


    @staticmethod
    def ip_splitter(df, ip_col):
        """
        Split the input data's IP address into separate octets.

        Parameters:
        df (pandas.DataFrame): Input data.
        ip_col (str): Name of the IP address column.

        Returns:
        pandas.DataFrame: Transformed data with IP address split into separate octets.
        """
        # Split the IP address into separate octets
        ip_octets = df[ip_col].str.split('.', expand=True)
        ip_octets.columns = [f'{ip_col}_octet1', f'{ip_col}_octet2', f'{ip_col}_octet3', f'{ip_col}_octet4']
        ip_octets = ip_octets.apply(pd.to_numeric, errors='coerce')   # convert to int from object
        print("hi")

        # Concatenate the octets back to the original data
        df = pd.concat([df, ip_octets], axis=1)

        return df.drop(columns=ip_col)

    @staticmethod
    def timestamp_splitter(df, timestamp_col):
        """
        Split the input data's timestamp into separate date and time components.

        Parameters:
        df (pandas.DataFrame): Input data.
        timestamp_col (str): Name of the timestamp column.

        Returns:
        pandas.DataFrame: Transformed data with timestamp split into separate components.
        """
        # Split the timestamp into separate individual components
        df["DT Year"] = pd.to_datetime(df[timestamp_col]).dt.year
        df["DT Month"] = pd.to_datetime(df[timestamp_col]).dt.month
        df["DT Day"] = pd.to_datetime(df[timestamp_col]).dt.day
        df["DT Hour"] = pd.to_datetime(df[timestamp_col]).dt.hour
        df["DT Minute"] = pd.to_datetime(df[timestamp_col]).dt.minute
        df["DT Second"] = pd.to_datetime(df[timestamp_col]).dt.second

        return df.drop(columns=timestamp_col)