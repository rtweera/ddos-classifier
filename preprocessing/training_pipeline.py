from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from preprocessing.feature_dropper import FeatureDropper
from preprocessing.category_encoder import CategoryEncoder
from preprocessing.numerical_standardiser import NumericalStandardiser
from preprocessing.feature_imputer import FeatureImputer
from preprocessing.composite_splitter import CompositeSplitter

class DataPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    A reusable data preprocessing pipeline that handles:
    - Dropping unwanted features
    - Identification of categorical and numerical variables
    - Missing value imputation
    - Categorical variable encoding
    - Standardisation of numerical features
    """

    def __init__(self, numerical_imputer_strategy='mean', categorical_imputer_strategy='most_frequent'):
        self.numerical_imputer_strategy = numerical_imputer_strategy
        self.categorical_imputer_strategy = categorical_imputer_strategy

        self.feature_dropper = FeatureDropper()
        self.feature_imputer = FeatureImputer(num_strategy=self.numerical_imputer_strategy, cat_strategy=self.categorical_imputer_strategy)
        self.category_encoder = CategoryEncoder()
        self.numerical_standardiser = NumericalStandardiser()

        # Combine preprocessing steps into a pipeline
        self.preprocessor = Pipeline([
            ('feature_dropper', self.feature_dropper),
            ('composite_splitter', CompositeSplitter()),
            # ('feature_imputer', self.feature_imputer),    # INFO: Not needed for the dataset
            # ('category_encoder', self.category_encoder),  # INFO: Not needed for the dataset
            ('numerical_standardiser', self.numerical_standardiser)
        ])

    def fit(self, X):
        """
        Fit the preprocessing pipeline on the input data.

        Parameters:
        X (pandas.DataFrame): Input data to be preprocessed.
        """
        self.preprocessor.fit(X)
        return self

    def partial_fit(self, X):
        """
        Partially fit the preprocessing pipeline on the input data.

        Parameters:
        X (pandas.DataFrame): Input data to be preprocessed.
        """
        self.preprocessor.partial_fit(X)
        return

    def transform(self, X):
        """
        Transform the input data using the preprocessing pipeline.

        Parameters:
        X (pandas.DataFrame): Input data to be preprocessed.

        Returns:
        pandas.DataFrame: Preprocessed data.
        """
        return self.preprocessor.transform(X)

    def save(self, path=None):
        """
        Save the preprocessing pipeline to a file.

        Parameters:
        path: File path to save the pipeline.

        Exceptions:
        ValueError: If no file path is provided.
        """
        if path is None:
            raise ValueError("Please provide a file path to save the pipeline.")
        joblib.dump(self.preprocessor, path)
        print(f"Preprocessing pipeline saved to {path}")

    def load(self, path):
        """
        Load the preprocessing pipeline from a file.

        Parameters:
        path: File path to load the pipeline.

        """
        if path is None:
            raise ValueError("Please provide a file path to load the pipeline.")

        self.preprocessor = joblib.load(path)
        print(f"Preprocessing pipeline loaded from {path}")
        return self