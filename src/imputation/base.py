from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel
from typing import Any

class BaseImputer(ABC):
    """
    Abstract base class for imputers with Pydantic validation.
    """

    config: BaseModel

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseImputer":
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
