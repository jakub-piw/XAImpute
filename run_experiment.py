from src.data.dataset_loader import DatasetLoader, DatasetConfig
from src.imputation.imputers import (
    RandomImputer, MeanImputer, MICEImputer, 
    kNNImputer, SoftImputer, kNNImputerConfig, MICEImputerConfig,
    SoftImputerConfig, RandomImputerConfig, kNNImputerConfig
)
from src.models.modeltrainer import ModelTrainer, ModelConfig

from src.experiment.experiment import ExperimentRun
from src.experiment.experiment_logger import ExperimentLogger

import argparse
from pathlib import Path

BASE_COLAB_PATH = '/content/drive/MyDrive/XAImpute/'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment with dataset and options")
    parser.add_argument("--dataset_path", "-d", type=Path, required=True,
                        help="Path to the dataset CSV file")
    parser.add_argument("--target_column", "-c", type=str, required=True,
                        help="name od the target column in the dataset")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["classification", "regression"],
                        help="task type: classification or regression")
    parser.add_argument("--random_state", "-s", type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument("--missing_rate", "-m", type=float, nargs='+', default=[0.1, 0.25, 0.5],
                        help="One or more missing rates between 0 and 1 (e.g. 0.1 or 0.1 0.25 0.5)")
    return parser.parse_args()


def main(args):
    args = parse_args()
    if isinstance(args['missing_rate'], list):
        miss_rates = args['missing_rate']
    else:
        miss_rates = [args['missing_rate']]
    for r in miss_rates:
        if r < 0 or r > 1:
            raise ValueError(f"missing_rate values must be between 0 and 1. Got {r}")


    if args['task'] == 'regression':
        model_names = ['linear_regression', "random_forest", "knn", "xgboost"]
    else:
        model_names = ['logistic_regression', "random_forest", "knn", "xgboost"]


    random_config = RandomImputerConfig(random_state=args['random_state'])
    random_imputer = RandomImputer(random_config)

    mean_imputer = MeanImputer()

    mice_config = MICEImputerConfig(random_state=args['random_state'])
    mice_imputer = MICEImputer(mice_config)

    knn_config = kNNImputerConfig()
    knn_imputer = kNNImputer(knn_config)

    soft_config = SoftImputerConfig()
    soft_imputer = SoftImputer(soft_config)

    imputers = [random_imputer, mean_imputer, mice_imputer, knn_imputer, soft_imputer]

    dataset_conf = DatasetConfig(
        filepath=f'datasets/{args["dataset"]}',
        target_column=args['target_column'],
        test_size='0.25',
        random_state=args['random_state']
        )
    diabetes_loader = DatasetLoader(dataset_conf)
    dataset_name = args['dataset'].split('.')[0]

    print('Dataset loader got')
    for miss_rate in miss_rates:
        diabetes_loader.run(proportion=miss_rate)
        for model_name in model_names:
            print(miss_rate, model_name)
            modelconfig = ModelConfig(model_name=model_name, task_type=args['task'], random_state=args['random_state'])
            model_trainer = ModelTrainer(modelconfig)
            exp_logger = ExperimentLogger(base_dir=f'results_{args["random_state"]}', dataset=dataset_name, missing_rate=str(int(miss_rate*100)), model=model_name)
            experiment = ExperimentRun(diabetes_loader, model_trainer, imputers, exp_logger, args['random_state'])
            experiment.run()

if __name__ == "__main__":
    main()