import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from typing import Optional

class PermutationImportanceExplainer:
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, task_type: str):
        """
        Args:
            model: Trained model.
            X_test (pd.DataFrame): Features for permutation importance calculation.
            y_test (pd.Series): True labels for scoring.
            task_type (str): 'classification' or 'regression'.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type.lower()
        self.importance_df: Optional[pd.DataFrame] = None

        if self.task_type == "classification":
            self.scoring = "accuracy"
        elif self.task_type == "regression":
            self.scoring = "r2"
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Use 'classification' or 'regression'.")

    def compute_importance(self) -> pd.DataFrame:
        """Computes permutation feature importance with fixed n_repeats = 10."""
        result = permutation_importance(
            self.model, 
            self.X_test, 
            self.y_test, 
            scoring=self.scoring, 
            n_repeats=10,  
            random_state=42
        )

        self.importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

        return self.importance_df

    def plot_importance(self) -> Optional[plt.Figure]:
        """Plots the feature importances."""
        if self.importance_df is None:
            raise ValueError("You must call compute_importance() first!")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(self.importance_df['Feature'], self.importance_df['Importance'], color='royalblue')
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Features")
        ax.set_title("Permutation Feature Importance")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def compare_importances(self, other_importance: pd.DataFrame) -> pd.DataFrame:
        """
        Compares self.importance_df to another importance dataframe.

        Args:
            other_importance (pd.DataFrame): DataFrame with columns ['Feature', 'Importance']

        Returns:
            pd.DataFrame: Merged DataFrame with differences computed.
        """
        if self.importance_df is None:
            raise ValueError("You must call compute_importance() first!")

        required_cols = {'Feature', 'Importance'}
        if not required_cols.issubset(other_importance.columns):
            raise ValueError(f"other_importance must contain columns {required_cols}")

        if other_importance['Feature'].duplicated().any() or self.importance_df['Feature'].duplicated().any():
            raise ValueError("Duplicate feature names found in importance DataFrames!")

        self_df = self.importance_df.copy()
        other_df = other_importance.copy()
        merged = pd.merge(
            self_df,
            other_df,
            on="Feature",
            suffixes=('_self', '_other'),
            how="inner" 
        )

        merged['Difference'] = merged['Importance_other'] - merged['Importance_self']
        return merged.sort_values(by='Difference', ascending=False).reset_index(drop=True)


    @staticmethod
    def plot_difference(diff_df: pd.DataFrame) -> plt.Figure:
        """Plots difference in feature importance between two models."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if x > 0 else 'red' for x in diff_df['Difference']]
        ax.barh(diff_df['Feature'], diff_df['Difference'], color=colors)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlabel("Importance Difference (Imputed - Original)")
        ax.set_ylabel("Features")
        ax.set_title("Change in Feature Importance Due to Imputation")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig
