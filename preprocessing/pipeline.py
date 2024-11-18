# DEPRECATED: This file is deprecated. Please use the `training_pipeline.py` file instead.
# WARNING: This file will be removed in a future release.

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataPreprocessingPipeline:
    """
    A reusable data preprocessing pipeline that handles:
    - Identification of categorical and numerical variables
    - Missing value imputation
    - Categorical variable encoding
    """

    def __init__(self, numerical_imputer_strategy='mean', categorical_imputer_strategy='most_frequent'):
        self.numerical_imputer_strategy = numerical_imputer_strategy
        self.categorical_imputer_strategy = categorical_imputer_strategy

        # Create reusable preprocessing steps
        self.num_imputer = SimpleImputer(strategy=self.numerical_imputer_strategy)
        self.cat_imputer = SimpleImputer(strategy=self.categorical_imputer_strategy)
        self.label_encoder = LabelEncoder() # TODO: how to change to this label_encoder instead of OH? automatically.
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        # Combine preprocessing steps into a pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num_imputer', self.num_imputer, self.get_numerical_features),
                ('cat_imputer', self.cat_imputer, self.get_categorical_features),
                ('one_hot', self.one_hot_encoder, self.get_categorical_features)
            ],
            remainder='passthrough'
        )

    def fit_transform(self, X):
        """
        Fit and transform the data through the preprocessing pipeline.

        Parameters:
        X (pandas.DataFrame): Input data to be preprocessed.

        Returns:
        pandas.DataFrame: Preprocessed data.
        """
        return self.preprocessor.fit_transform(X)

    @staticmethod
    def get_numerical_features(X):
        """
        Identify numerical features in the input data.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        list: List of numerical feature names.
        """
        return X.select_dtypes(include='number').columns.tolist()

    @staticmethod
    def get_categorical_features(X):
        """
        Identify categorical features in the input data.

        Parameters:
        X (pandas.DataFrame): Input data.

        Returns:
        list: List of categorical feature names.
        """
        return X.select_dtypes(exclude='number').columns.tolist()


