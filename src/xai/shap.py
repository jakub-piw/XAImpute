import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost

class SHAPExplainer:
    def __init__(self, model, X_test: pd.DataFrame, task_type: str):
        """
        Args:
            model: Trained model.
            X_test (pd.DataFrame): Test set to calculate SHAP values.
            task_type (str): 'classification' or 'regression'.
        """
        self.model = model
        self.X_test = X_test
        self.task_type = task_type
        self.explainer = self._select_explainer()
        self.shap_values: Optional[np.ndarray] = None

    def _select_explainer(self):
        """Dynamically selects the appropriate SHAP explainer."""
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        is_tree_model = isinstance(self.model, (
            RandomForestClassifier,
            RandomForestRegressor,
            xgboost.XGBClassifier,
            xgboost.XGBRegressor
        ))
        is_knn_model = isinstance(self.model, (KNeighborsClassifier, KNeighborsRegressor))

        if is_tree_model:
            print("[INFO] Using TreeExplainer.")
            if self.task_type == "classification":
                return shap.TreeExplainer(self.model, data=self.X_test, feature_perturbation="interventional")
            else:
                return shap.TreeExplainer(self.model, data=self.X_test)

        elif is_knn_model:
            print("[INFO] Using PermutationExplainer for KNN with nperm=10.")
            return shap.PermutationExplainer(self.model.predict, self.X_test, nperm=5)

        else:
            print("[INFO] Using model-agnostic Explainer.")
            if self.task_type == "classification":
                return shap.Explainer(lambda x: self.model.predict_proba(x)[:, 1], self.X_test)
            else:
                return shap.Explainer(self.model.predict, self.X_test)

    def calculate_shap_values(self) -> np.ndarray:
        """Calculates SHAP values for the test set."""
        print(type(self.explainer))
        if self.explainer.__class__.__name__ == 'TreeExplainer':
            self.shap_values = self.explainer(self.X_test, check_additivity=False)
            if self.shap_values.values.ndim == 3:
                self.shap_values.values = self.shap_values.values[..., 1]
        else:
            self.shap_values = self.explainer(self.X_test)
        return self.shap_values.values

    def plot_feature_importance(self, max_display: int = 20) -> plt.Figure:
        """Plots a bar chart for feature importance based on SHAP values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(self.shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        return fig

    def plot_beeswarm(self, max_display: int = 20) -> plt.Figure:
        """Creates a beeswarm plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        return fig

    def plot_waterfall(self, idx: int) -> plt.Figure:
        """Creates a waterfall plot for a specific observation."""
        shap.plots.waterfall(self.shap_values[idx], show=False)
        fig = plt.gcf()  
        plt.tight_layout()
        return fig

    def evaluate_shap_similarity(
        self,
        other_shap_values: np.ndarray,
        metrics: List[str] = ["mae", "rmse"]
    ) -> Dict[str, Any]:
        """
        Compares SHAP values between two models.

        Args:
            other_shap_values (np.ndarray): SHAP values from another model (same shape as self.shap_values).
            metrics (List[str]): List of metrics to compute: 'mae', 'rmse'.

        Returns:
            Dict[str, Any]: Dictionary with comparison scores for each feature and overall.
        """
        results = {}

        shap_ori = self.shap_values.values if hasattr(self.shap_values, "values") else self.shap_values
        shap_imp = other_shap_values.values if hasattr(other_shap_values, "values") else self.shap_values

        if shap_ori.shape != shap_imp.shape:
            raise ValueError("SHAP arrays must have the same shape!")

        for metric in metrics:
            feature_scores = {}
            for i, feature_name in enumerate(self.X_test.columns):
                if metric == "mae":
                    score = mean_absolute_error(shap_ori[:, i], shap_imp[:, i])
                elif metric == "rmse":
                    score = root_mean_squared_error(shap_ori[:, i], shap_imp[:, i])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                feature_scores[feature_name] = score

            if metric == "mae":
                overall = mean_absolute_error(shap_ori.flatten(), shap_imp.flatten())
            elif metric == "rmse":
                overall = root_mean_squared_error(shap_ori.flatten(), shap_imp.flatten())

            results[metric] = {
                "per_feature": feature_scores,
                "overall": overall
            }

        return results
