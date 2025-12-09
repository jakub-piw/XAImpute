from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
import numpy as np
import pandas as pd

class ModelEvaluator:
    def __init__(self, task_type: str):
        self.task_type = task_type

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
        """Evaluate model performance using appropriate metrics."""
        results = {}
        
        if self.task_type == "classification":
            # Classification metrics
            results["accuracy"] = accuracy_score(y_true, y_pred)
            results["precision"] = precision_score(y_true, y_pred)
            results["recall"] = recall_score(y_true, y_pred)
            results["auc"] = roc_auc_score(y_true, y_prob)
        
        elif self.task_type == "regression":
            # Regression metrics
            results["rmse"] = root_mean_squared_error(y_true, y_pred)
            results["mae"] = mean_absolute_error(y_true, y_pred)
            results["mape"] = mean_absolute_percentage_error(y_true, y_pred)  
            
        return results

class ImputerEvaluator:
    def __init__(self):
        pass

    def evaluate(self, X_original: pd.DataFrame, X_imputed: pd.DataFrame) -> dict:
        """Evaluate imputation quality by comparing original vs imputed data."""
        rmse = root_mean_squared_error(X_original, X_imputed)
        mae = mean_absolute_error(X_original, X_imputed)
        mape = mean_absolute_percentage_error(X_original, X_imputed) 
        return {"rmse": rmse, "mae": mae, "mape": mape}

class PredictionSimilarityEvaluator:
    def __init__(self, task_type: str):
        self.task_type = task_type

    def evaluate(self, y_pred_original: np.ndarray, y_pred_imputed: np.ndarray) -> dict:
        """Evaluate the similarity of predictions from original vs imputed models."""
        results = {}
        results["mse"] = root_mean_squared_error(y_pred_original, y_pred_imputed)
        results["mae"] = mean_absolute_error(y_pred_original, y_pred_imputed)
            
        return results
