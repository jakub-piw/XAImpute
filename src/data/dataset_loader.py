import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import MinMaxScaler
import os
from .missing_values_injector import MissingDataInjector
from typing import Self

class DatasetConfig(BaseModel):
    """Configuration for loading and splitting a dataset."""
    filepath: str = Field(..., description="Path to the preprocessed CSV file.")
    target_column: str = Field(..., description="Name of the target column in the dataset.")
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Proportion of the dataset to include in the test split.")
    random_state: int = Field(default=42, description="Random state")

    
    @field_validator("filepath")
    def file_must_exist(cls, v):
        if not os.path.isfile(v):
            raise FileNotFoundError(f"File not found: {v}")
        if not v.endswith(".csv"):
            raise ValueError("Dataset must be a CSV file.")
        return v


class DatasetLoader:
    """Loads a cleaned dataset from CSV and splits it into train/test sets."""

    def __init__(self, config: DatasetConfig):
        """
        Initializes the DatasetLoader.

        Args:
            config (DatasetConfig): Configuration for data loading and splitting.
        """
        self.config = config
        self.df: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.injector: MissingDataInjector = None
        self.X_train_missing: pd.DataFrame = None
        self.X_test_missing: pd.DataFrame = None

    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from a CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        self.df = pd.read_csv(self.config.filepath)
        if self.config.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in dataset.")
        return self.df

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and test sets using a given random seed.

        Args:
            random_state (int, optional): Random seed for reproducibility. Overrides the default if provided.

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X = self.df.drop(columns=[self.config.target_column])
        y = self.df[self.config.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def validate_and_scale(self):
        """
        Validates that all features are numeric and applies MinMax scaling in-place.
        """
        if not all(self.X_train.dtypes.apply(pd.api.types.is_numeric_dtype)):
            raise TypeError("All features in X_train must be numeric.")

        if not all(self.X_test.dtypes.apply(pd.api.types.is_numeric_dtype)):
            raise TypeError("All features in X_test must be numeric.")

        scaler = MinMaxScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
    
    def inject_missing_values(self, proportion: float):
        self.injector = MissingDataInjector(proportion, self.config.random_state)
        if self.X_test is not None and self.X_train is not None:
            self.X_train_missing = self.injector.inject(self.X_train)
            self.X_test_missing = self.injector.inject(self.X_test)
            return self.X_train_missing, self.X_test_missing
        else:
            raise RuntimeError("Dataset must be split before injecting missing values.")

    def run(self, proportion: float, scaling: bool = False) -> Self:
        """
        Runs the full loading and splitting pipeline.

        Args:
            proportion (float): Proportion of missing values to inject.
            scaling (bool): Whether to scale features to [0, 1].

        Returns:
            Self
        """
        self.load()
        self.split()
        if scaling:
            self.validate_and_scale()
        self.inject_missing_values(proportion)
        return self
