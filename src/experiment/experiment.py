from typing import List
import pandas as pd
from ..data.dataset_loader import DatasetLoader
from ..models.modeltrainer import ModelTrainer
from ..imputation.imputers import BaseImputer
from ..models.evaluators import ModelEvaluator, ImputerEvaluator, PredictionSimilarityEvaluator
from ..xai.shap import SHAPExplainer
from ..xai.permutation_importance import PermutationImportanceExplainer
from ..xai.partial_dependence import PDPExplainer
from .experiment_logger import ExperimentLogger

class ExperimentRun:
    
    def __init__(
            self, 
            dataset_loader: DatasetLoader, 
            model_trainer: ModelTrainer,
            imputers: List[BaseImputer],
            experiment_logger: ExperimentLogger,
            random_state: int = 42,
            ):
        self.dataset_loader = dataset_loader
        self.model_trainer = model_trainer
        self.imputers = imputers
        self.experiment_logger = experiment_logger
        self.random_state = random_state

    def run(self):
        self.model_trainer.tune_hyperparameters(
            self.dataset_loader.X_train,
            self.dataset_loader.y_train)
        
        self.model_trainer.train(
            self.dataset_loader.X_train,
            self.dataset_loader.y_train
        )

        model_setup = {}
        model_setup['model'] = self.model_trainer.model
        model_setup['params'] = self.model_trainer.best_params_
        print('Model setup:', model_setup)
        self.experiment_logger.save_pickle(model_setup, 'model_setup.pkl')
        
        y_pred_ori = (self.model_trainer.predict_proba(self.dataset_loader.X_test) 
                      if self.model_trainer.config.task_type == 'classification' 
                      else self.model_trainer.predict(self.dataset_loader.X_test))
        eval_ori = self._evaluate_original_model()

        for imputer in self.imputers:
            imputer_name = imputer.__class__.__name__
            print(imputer_name)
            if imputer_name == 'SoftImpuer':
                X_train_imp = imputer.transform(self.dataset_loader.X_train_missing)
                _, X_test_imp = imputer.fit_transform_split(self.dataset_loader.X_train_missing, self.dataset_loader.X_test_missing)
            else:
                X_train_imp = imputer.fit_transform(self.dataset_loader.X_train_missing)
                X_test_imp = imputer.transform(self.dataset_loader.X_test_missing)

            self.model_trainer.train(X_train_imp, self.dataset_loader.y_train)
            _ = self._evaluate_imputed_model(eval_ori, X_test_imp, y_pred_ori, imputer_name)


    def _evaluate_original_model(self):
        model = self.model_trainer.model
        task = self.model_trainer.config.task_type
        evaluation_results = {}
        columns = list(
            self.dataset_loader.injector._get_continuous_columns(
                self.dataset_loader.X_train
                ))

        model_eval = ModelEvaluator(task)
        y_pred = self.model_trainer.predict(self.dataset_loader.X_test)
        if task == 'classification':
            y_pred_proba = self.model_trainer.predict_proba(self.dataset_loader.X_test)
        else: 
            y_pred_proba = None
        evaluation_results['model_evaluation'] = model_eval.evaluate(self.dataset_loader.y_test, y_pred, y_pred_proba)

        shap_exp = SHAPExplainer(model, self.dataset_loader.X_test, task)
        shap_exp.calculate_shap_values()
        evaluation_results['shap'] = shap_exp.shap_values

        fig_shap_fi = shap_exp.plot_feature_importance()
        self.experiment_logger.save_plot(fig_shap_fi, 'shap_fi_original.png')
        fig_shap_beeswarm = shap_exp.plot_beeswarm()
        self.experiment_logger.save_plot(fig_shap_beeswarm, 'shap_beeswarm_original.png')

        perm_exp = PermutationImportanceExplainer(
            model, self.dataset_loader.X_test, 
            self.dataset_loader.y_test, task)
        perm_exp.compute_importance()
        evaluation_results['pfi'] = perm_exp.importance_df

        fig_pfi = perm_exp.plot_importance()
        self.experiment_logger.save_plot(fig_pfi, 'pfi_original.png')

        pdp_exp = PDPExplainer(model, self.dataset_loader.X_test, columns)
        pdp_exp.compute_pdp()
        evaluation_results['pdp'] = pdp_exp.pdp_results

        self.experiment_logger.save_pickle(evaluation_results, 'evaluation_original.pkl')

        return evaluation_results
    
    def _evaluate_imputed_model(
            self, evaluation_original: dict,
            X_test_imputed: pd.DataFrame,
            y_pred_ori: pd.Series, 
            imputer_name: str):
        model = self.model_trainer.model
        task = self.model_trainer.config.task_type
        evaluation_results = {}
        columns = list(
            self.dataset_loader.injector._get_continuous_columns(
                self.dataset_loader.X_train
                ))

        model_eval = ModelEvaluator(task)
        y_pred = self.model_trainer.predict(self.dataset_loader.X_test)
        if task == 'classification':
            y_pred_proba = self.model_trainer.predict_proba(self.dataset_loader.X_test)
        else: 
            y_pred_proba = None
        evaluation_results['model_evaluation'] = model_eval.evaluate(self.dataset_loader.y_test, y_pred, y_pred_proba)

        imputer_eval = ImputerEvaluator()
        evaluation_results['imputer_evaluation'] = imputer_eval.evaluate(self.dataset_loader.X_test, X_test_imputed)

        prediction_eval = PredictionSimilarityEvaluator(task)
        y_pred_imputed = y_pred_proba if task == 'classification' else y_pred
        evaluation_results['pred_similarity_evaluation'] = prediction_eval.evaluate(y_pred_ori, y_pred_imputed)


        shap_exp = SHAPExplainer(model, X_test_imputed, task)
        shap_exp.calculate_shap_values()
        evaluation_results['shap'] = shap_exp.shap_values

        evaluation_results['shap_compared'] = shap_exp.evaluate_shap_similarity(evaluation_original['shap'])

        fig_shap_fi = shap_exp.plot_feature_importance()
        self.experiment_logger.save_plot(fig_shap_fi, f'shap_fi_{imputer_name}.png')
        fig_shap_beeswarm = shap_exp.plot_beeswarm()
        self.experiment_logger.save_plot(fig_shap_beeswarm, f'shap_beeswarm_{imputer_name}.png')

        perm_exp = PermutationImportanceExplainer(
            model, self.dataset_loader.X_test, 
            self.dataset_loader.y_test, task)
        perm_exp.compute_importance()
        evaluation_results['pfi'] = perm_exp.importance_df

        pfi_diff_df = perm_exp.compare_importances(evaluation_original['pfi'])
        evaluation_results['pfi_compared'] = pfi_diff_df

        fig_pfi = perm_exp.plot_importance()
        self.experiment_logger.save_plot(fig_pfi, f'pfi_{imputer_name}.png')

        fig_pfi_comp = perm_exp.plot_difference(pfi_diff_df)
        self.experiment_logger.save_plot(fig_pfi_comp, f'pfi_compared_{imputer_name}.png')

        pdp_exp = PDPExplainer(model, self.dataset_loader.X_test, columns)
        pdp_exp.compute_pdp()
        evaluation_results['pdp'] = pdp_exp.pdp_results

        evaluation_results['pdp_compared'] = pdp_exp.compare_pdp(evaluation_original['pdp'])

        self.experiment_logger.save_pickle(evaluation_results, f'evaluation_{imputer_name}.pkl')

        return evaluation_results