from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import math
from typing import List, Optional, Dict, Any


def _load_pickle(path: Path) -> Optional[dict]:
    """
    Loads pickle file from the given path.
    """
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _extract_pfi_list(pfi_obj: Any) -> Optional[List[str]]:
    """
    Helper function to parse permutation feature importance (PFI) object
    and return a list of feature names ordered by importance.
    """
    if pfi_obj is None:
        return None
    try:
        if hasattr(pfi_obj, 'columns') and 'Feature' in pfi_obj.columns:
            df = pfi_obj.copy()
            if 'Importance' in df.columns:
                df = df.sort_values('Importance', ascending=False)
            return list(df['Feature'].astype(str).tolist())
        if isinstance(pfi_obj, dict):
            return [k for k, _ in sorted(pfi_obj.items(), key=lambda kv: -kv[1])]
        return list(pfi_obj)
    except Exception:
        return None


def collect_experiment_results(base_dir: str = 'results_all') -> pd.DataFrame:
    """Aggregate experimental results from nested result folders into a DataFrame.

    The function expects folder structure:
      base_dir/results_{seed}/{dataset}/{missing_pct}/{model}/evaluation_*.pkl

    For each evaluation_*.pkl it extracts relevant fields and returns one row per
    evaluation file (including evaluation_original.pkl). Columns include:
      - seed, dataset, missing_pct, missing_fraction, model, imputer
      - model_rmse, model_mae
      - imputer_rmse, imputer_mae (None for original)
      - pred_rmse, pred_mae (None for original)
      - shap_rmse_overall (if available)
      - pfi_features (list ordered by importance)
      - pdp_compared (dict of feature->float if available)

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    # For comparison: store original results for each (seed, dataset, missing_pct, model)
    originals = {}
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for seed_dir in sorted(base.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed_name = seed_dir.name
        seed = None
        if seed_name.startswith('results_'):
            try:
                seed = int(seed_name.split('_', 1)[1])
            except Exception:
                seed = seed_name
        else:
            seed = seed_name

        for dataset_dir in sorted(seed_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            for missing_dir in sorted(dataset_dir.iterdir()):
                if not missing_dir.is_dir():
                    continue
                try:
                    missing_pct = int(missing_dir.name)
                except Exception:
                    missing_pct = missing_dir.name

                missing_frac = None
                try:
                    missing_frac = int(missing_pct) / 100.0
                except Exception:
                    pass

                for model_dir in sorted(missing_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name

                    # Process evaluation_original.pkl first
                    original_pkl = model_dir / 'evaluation_original.pkl'
                    pkls = list(sorted(model_dir.glob('*.pkl')))
                    if original_pkl in pkls:
                        pkls.remove(original_pkl)
                        pkls = [original_pkl] + pkls

                    for pkl in pkls:
                        if pkl.name == 'model_setup.pkl':
                            continue

                        data = _load_pickle(pkl)
                        if data is None:
                            continue

                        if pkl.name == 'evaluation_original.pkl' or 'original' in pkl.name:
                            imputer = 'original'
                        else:
                            imputer = pkl.name.replace('evaluation_', '').replace('.pkl', '')

                        model_eval = data.get('model_evaluation', {}) or {}

                        accuracy = model_eval.get('accuracy')
                        precision = model_eval.get('precision')
                        recall = model_eval.get('recall')
                        auc = model_eval.get('auc')

                        if any(v is not None for v in (accuracy, precision, recall, auc)):
                            task = 'classification'
                            model_rmse = None
                            model_mae = None
                        else:
                            task = 'regression'
                            model_rmse = model_eval.get('rmse')
                            model_mae = model_eval.get('mae')
                            accuracy = None
                            precision = None
                            recall = None
                            auc = None

                        imputer_eval = data.get('imputer_evaluation', {}) or {}
                        imputer_rmse = imputer_eval.get('rmse')
                        imputer_mae = imputer_eval.get('mae')

                        pred_sim = data.get('pred_similarity_evaluation', {}) or {}
                        pred_rmse = pred_sim.get('mse')
                        pred_mae = pred_sim.get('mae')

                        shap_compared = data.get('shap_compared') or {}
                        shap_rmse_overall = None
                        try:
                            shap_rmse_overall = shap_compared.get('rmse', {}).get('overall')
                        except Exception:
                            shap_rmse_overall = None

                        pfi_obj = data.get('pfi')
                        pfi_features = _extract_pfi_list(pfi_obj)

                        pdp_compared = data.get('pdp_compared')
                        pdp_aggregated = np.mean(list(pdp_compared.values())) if pdp_compared else None

                        # Store original results for comparison
                        key = (seed, dataset, missing_pct, model)
                        if imputer == 'original':
                            originals[key] = {
                                'model_rmse': model_rmse,
                                'accuracy': accuracy,
                                'auc': auc
                            }

                        # Compute deltas for imputers
                        delta_rmse = None
                        delta_accuracy = None
                        delta_auc = None
                        if imputer != 'original' and key in originals:
                            orig = originals[key]
                            if task == 'regression' and orig['model_rmse'] is not None and model_rmse is not None:
                                delta_rmse = model_rmse - orig['model_rmse']
                            if task == 'classification':
                                if orig['accuracy'] is not None and accuracy is not None:
                                    delta_accuracy = orig['accuracy'] - accuracy
                                if orig['auc'] is not None and auc is not None:
                                    delta_auc = orig['auc'] - auc

                        row = {
                            'seed': seed,
                            'dataset': dataset,
                            'missing_pct': missing_pct,
                            'missing_frac': missing_frac,
                            'model': model,
                            'task': task,
                            'imputer': imputer,
                            'model_rmse': model_rmse,
                            'model_mae': model_mae,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'auc': auc,
                            'imputer_rmse': imputer_rmse,
                            'imputer_mae': imputer_mae,
                            'pred_rmse': pred_rmse,
                            'pred_mae': pred_mae,
                            'shap_rmse_overall': shap_rmse_overall,
                            'pfi_features': pfi_features,
                            'pdp_compared': pdp_compared,
                            'pdp_aggregated': pdp_aggregated,
                            'delta_rmse': delta_rmse,
                            'delta_accuracy': delta_accuracy,
                            'delta_auc': delta_auc,
                            'result_path': str(pkl)
                        }

                        rows.append(row)

    df = pd.DataFrame(rows)
    return df


__all__ = ['collect_experiment_results']
