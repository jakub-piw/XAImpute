from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Union
from sklearn.model_selection import GridSearchCV
from typing import Dict, Self

class ModelConfig(BaseModel):
    model_name: str
    task_type: str  # 'classification' or 'regression'
    random_state: int = 42

class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self.get_model(config.task_type, config.model_name, config.random_state)
        self.best_params_ = None 

    def get_model(self, task_type: str, model_name: str, random_state: int, custom_params: dict = None) -> BaseEstimator:
        """Return the corresponding model based on task type and model name with optional hyperparameters."""
        model = None
        if task_type == "classification":
            if model_name == "logistic_regression":
                model = LogisticRegression(random_state=random_state, max_iter=1000)
            elif model_name == "random_forest":
                model = RandomForestClassifier(random_state=random_state)
            elif model_name == "knn":
                model = KNeighborsClassifier()
            elif model_name == "xgboost":
                model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            elif model_name == "qda":
                model = QuadraticDiscriminantAnalysis()
            else:
                raise ValueError(f"Unknown classification model: {model_name}")

        elif task_type == "regression":
            if model_name == "linear_regression":
                model = LinearRegression()
            elif model_name == "random_forest":
                model = RandomForestRegressor(random_state=random_state)
            elif model_name == "knn":
                model = KNeighborsRegressor()
            elif model_name == "xgboost":
                model = XGBRegressor(random_state=random_state)
            else:
                raise ValueError(f"Unknown regression model: {model_name}")

        else:
            raise ValueError("Unknown task type")

        if custom_params:
            model.set_params(**custom_params)

        return model

    def get_param_grid(self) -> Dict[str, list]:
        """Return default parameter grid based only on model_name."""
        if self.config.model_name == "logistic_regression":
            return {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        elif self.config.model_name == "random_forest":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            }
        elif self.config.model_name == "knn":
            return {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "p": [1, 2]  # Minkowski distance: 1 - Manhattan, 2 - Euclidean
            }
        elif self.config.model_name == "xgboost":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        elif self.config.model_name == "qda":
            return {
                "reg_param": [0.0, 0.01, 0.1, 0.5]
            }
        elif self.config.model_name == "linear_regression":
            return {}  # No hyperparameters to tune
        else:
            raise ValueError(f"No param grid for model: {self.config.model_name}")


    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:
        """
        Automatically perform hyperparameter tuning with GridSearchCV.
        """
        param_grid = self.get_param_grid()

        if not param_grid:
            print(f"No hyperparameters to tune for model {self.config.model_name}. Skipping tuning.")
            return self

        search = GridSearchCV(
            self.model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring="neg_mean_squared_error" if self.config.task_type == "regression" else "accuracy"
        )

        search.fit(X_train, y_train)

        self.best_params_ = search.best_params_

        self.model = self.get_model(
            task_type=self.config.task_type,
            model_name=self.config.model_name,
            random_state=self.config.random_state,
            custom_params=self.best_params_
        )

        return self

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelTrainer":
        """Fit the model on training data."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Returns probability predictions in classification."""
        if self.config.task_type == 'classification':
            return self.model.predict_proba(X_test)[:, 1]
