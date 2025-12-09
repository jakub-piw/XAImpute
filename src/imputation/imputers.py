from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from src.imputation.base import BaseImputer
from sklearn.experimental import enable_iterative_imputer
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Self, Dict
from fancyimpute import SoftImpute as FancySoftImpute

class kNNImputerConfig(BaseModel):
    k_neighbors: int = 5 

class kNNImputer(BaseImputer):
    def __init__(self, config: kNNImputerConfig):
        self.imputer = KNNImputer(n_neighbors=config.k_neighbors)

    def fit(self, X: pd.DataFrame) -> Self:
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
class MeanImputer(BaseImputer):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="mean")

    def fit(self, X: pd.DataFrame) -> Self:
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
class MICEImputerConfig(BaseModel):
    max_iter: int = 20 
    random_state: int = 42

class MICEImputer(BaseImputer):
    def __init__(self, config: MICEImputerConfig):
        self.imputer = IterativeImputer(
            max_iter=config.max_iter,
            random_state=config.random_state
        )

    def fit(self, X: pd.DataFrame) -> Self:
        """Fits the MICE imputer on the provided DataFrame."""
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the provided DataFrame using the fitted imputer."""
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

class RandomImputerConfig(BaseModel):
    random_state: int = 42


class RandomImputer(BaseImputer):
    """
    Randomly imputes missing values by sampling from observed non-missing values.
    """

    def __init__(self, config: RandomImputerConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        self.non_missing_values: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame) -> Self:
        """Stores non-missing values for each column."""
        self.non_missing_values = {
            column: X[column].dropna().values
            for column in X.columns
            if X[column].notna().any()
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values by random sampling."""
        X_copy = X.copy()

        for column, available_values in self.non_missing_values.items():
            if column in X_copy.columns:
                missing_mask = X_copy[column].isnull()
                if missing_mask.any():
                    X_copy.loc[missing_mask, column] = self.rng.choice(
                        available_values,
                        size=missing_mask.sum(),
                        replace=True
                    )

        return X_copy

class SoftImputerConfig(BaseModel):
    shrinkage_value: float = 0.5
    max_iters: int = 100
    convergence_threshold: float = 1e-5

class SoftImputer(BaseImputer):
    def __init__(self, config: SoftImputerConfig):
        self.config = config
        self.imputer = FancySoftImpute(
            shrinkage_value=config.shrinkage_value,
            max_iters=config.max_iters,
            convergence_threshold=config.convergence_threshold
        )
        self.X_train_shape = None

    def fit(self, X: pd.DataFrame) -> Self:
        """SoftImpute fits and imputes at the same time, so here we only store training shape."""
        self.X_train_shape = X.shape
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms data using SoftImpute. If X includes both train and test, split manually."""
        X_values = X.to_numpy()
        X_imputed = self.imputer.fit_transform(X_values)
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    def fit_transform_split(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Special method to handle stacking train and test together."""
        X_train = X_train.copy()
        X_test = X_test.copy()

        combined = np.vstack([X_train.values, X_test.values])
        imputed = self.imputer.fit_transform(combined)

        X_train_imputed = imputed[:len(X_train)]
        X_test_imputed = imputed[len(X_train):]

        return (
            pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index),
            pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index),
        )