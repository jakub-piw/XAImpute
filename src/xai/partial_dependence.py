import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from typing import Dict, List, Optional

class PDPExplainer:
    def __init__(self, model, X_test: pd.DataFrame, missing_columns: List[str]):
        """
        Args:
            model: Trained model.
            X_test: Test dataset.
            missing_columns: List of columns which were subjected to imputation.
        """
        self.model = model
        self.X_test = X_test
        self.missing_columns = missing_columns
        self.pdp_results: Dict[str, pd.DataFrame] = {}

    def compute_pdp(self):
        """Compute partial dependence for all missing/imputed columns."""
        for feature in self.missing_columns:
            pdp = partial_dependence(self.model, self.X_test, features=[feature], kind='average')
            values = pdp['grid_values'][0]
            avg_predictions = pdp['average'][0]
            self.pdp_results[feature] = pd.DataFrame({
                'feature_value': values,
                'average_prediction': avg_predictions
            })
        return self.pdp_results

    def plot_single_pdp(self, feature: str) -> plt.Figure:
        """Plot PDP curve for a single feature."""
        if feature not in self.pdp_results:
            raise ValueError(f"PDP for feature '{feature}' not computed yet.")

        df = self.pdp_results[feature]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['feature_value'], df['average_prediction'], label=feature, marker='o')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Average Prediction')
        ax.set_title(f'Partial Dependence for {feature}')
        ax.legend()
        plt.tight_layout()
        return fig

    def compare_pdp(self, other_pdp: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compare PDPs using l2 norm for curves.

        Args:
            other_pdp: PDP result dict from another model (imputed).

        Returns:
            Dict[str, float]: MBD score per feature.
        """
        mbd_scores = {}
        for feature in self.missing_columns:
            df1 = self.pdp_results[feature]
            df2 = other_pdp[feature]
            merged = pd.merge(df1, df2, on="feature_value", suffixes=('_original', '_other'))
            mbd_score = self._curve_l2_norm(merged['average_prediction_original'], merged['average_prediction_other'])
            mbd_scores[feature] = mbd_score
        return mbd_scores

    def plot_multiple_pdps(self, pdp_dicts: List[Dict[str, pd.DataFrame]], labels: List[str], feature: str) -> plt.Figure:
        """Plot multiple PDP curves for a given feature."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for pdp_result, label in zip(pdp_dicts, labels):
            df = pdp_result[feature]
            ax.plot(df['feature_value'], df['average_prediction'], label=label, marker='o')

        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Average Prediction')
        ax.set_title(f'Partial Dependence Comparison: {feature}')
        ax.legend()
        plt.tight_layout()
        return fig

    @staticmethod
    def _curve_l2_norm(y1: pd.Series, y2: pd.Series) -> float:
        """Compute a simple distance for two curves (here: 1 - normalized L2 distance)."""
        norm_diff = ((y1 - y2) ** 2).sum() ** 0.5
        norm_base = (y1 ** 2).sum() ** 0.5
        if norm_base == 0:
            return 0.0
        return 1 - (norm_diff / norm_base)
