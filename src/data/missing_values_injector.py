import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class MissingDataInjector:
    """
    Injects missing values into continuous columns of a dataset using MCAR (missing completely at random).
    """

    def __init__(
        self,
        proportion: float = 0.1,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            proportion (float): Proportion of missing values to inject per column.
            random_state (int, optional): Seed for reproducibility.
        """
        self.proportion = proportion
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.missing_indices_: Dict[str, List[int]] = {}

    def _get_continuous_columns(self, df: pd.DataFrame, threshold: int = 5) -> List[str]:
        """
        Detect continuous (numeric) columns, excluding columns with too few unique values.

        Args:
            df (pd.DataFrame): Input dataframe.
            threshold (int): Minimum number of unique values required to treat as continuous.

        Returns:
            List[str]: Names of continuous columns.
        """
        continuous_columns = df.select_dtypes(include=["float", "int"]).columns.tolist()

        continuous_columns = [
            col for col in continuous_columns if df[col].nunique() > threshold
        ]

        return continuous_columns


    def inject(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Injects missing values into the dataframe.

        Args:
            df (pd.DataFrame): Data to modify (copy will be returned).
            columns (List[str], optional): Specific columns to inject. If None, all continuous columns will be used.

        Returns:
            pd.DataFrame: Modified dataframe with NaNs injected.
        """
        df_modified = df.copy()
        if columns is None:
            columns = self._get_continuous_columns(df)

        self.missing_indices_.clear()

        for col in columns:
            n = df.shape[0]
            k = int(np.floor(self.proportion * n))
            indices = self.rng.choice(n, size=k, replace=False)
            self.missing_indices_[col] = indices.tolist()
            df_modified.iloc[indices, df.columns.get_loc(col)] = np.nan

        return df_modified

    def get_missing_indices(self) -> Dict[str, List[int]]:
        """
        Returns the indices where missing values were injected.

        Returns:
            Dict[str, List[int]]: Dictionary with column names as keys and list of row indices as values.
        """
        return self.missing_indices_


    def get_missing_values(
        self, 
        df: pd.DataFrame, 
    ) -> Dict[str, np.ndarray]:
        """
        Retrieves values at the missing indices for each column.

        Args:
            df (pd.DataFrame): Original dataframe before injection.

        Returns:
            Dict[str, np.ndarray]: 
                {column_name: values}
        """
        results = {}
        for col, indices in self.missing_indices_.items():
            selected_values = df.iloc[indices][col].values
            results[col] = selected_values
        return results