import openml
import pandas as pd
from ucimlrepo import fetch_ucirepo
import kagglehub

def download_openml_csv(dataset_id, filename, class_map=None, class_col='class'):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, _, _, _ = dataset.get_data()
    df = X.copy()
    if class_map:
        df[class_col] = df[class_col].map(class_map)
    df.to_csv(filename, index=False)

def download_ucirepo_csv(repo_id, filename):
    ds = fetch_ucirepo(id=repo_id)
    X = ds.data.features
    y = ds.data.targets
    if not isinstance(X, pd.DataFrame):
        import numpy as np
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    if isinstance(y, pd.DataFrame):
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    elif isinstance(y, pd.Series):
        target_name = y.name if y.name else "target"
        df = X.reset_index(drop=True).copy()
        df[target_name] = y.reset_index(drop=True)
    else:
        import numpy as np
        y = pd.Series(y)
        target_name = y.name if y.name else "target"
        df = X.reset_index(drop=True).copy()
        df[target_name] = y.reset_index(drop=True)
    df.to_csv(filename, index=False)

def download_kagglehub_csv(kaggle_id, file_path, drop_cols, out_csv):
    path = kagglehub.dataset_download(kaggle_id)
    df = pd.read_csv(file_path.format(path=path))
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    df.to_csv(out_csv, index=False)


# Diabetes
download_openml_csv(
    dataset_id=37,
    filename="datasets/diabetes.csv",
    class_map={'tested_positive': 1, 'tested_negative': 0},
    class_col='class'
)

# CPU
download_openml_csv(
    dataset_id=562,
    filename="datasets/cpu.csv"
)

# Phoneme
download_openml_csv(
    dataset_id=1489,
    filename="datasets/phoneme.csv",
    class_map={'1': 0, '2': 1},
    class_col='Class'
)

# Concrete
download_ucirepo_csv(
    repo_id=165,
    filename="datasets/concrete.csv"
)

# Loan
download_kagglehub_csv(
    kaggle_id="itssuru/loan-data",
    file_path="C:/Users/jakub/.cache/kagglehub/datasets/itssuru/loan-data/versions/1/loan_data.csv",
    drop_cols=['purpose'],
    out_csv="datasets/loan.csv"
)

# Gym Exercises
download_kagglehub_csv(
    kaggle_id="valakhorasani/gym-members-exercise-dataset",
    file_path="{path}/gym_members_exercise_tracking.csv",
    drop_cols=['Gender', 'Workout_Type'],
    out_csv="datasets/gym_excercises.csv"
)

# Housing
def download_housing():
    path = kagglehub.dataset_download("fratzcan/usa-house-prices")
    df = pd.read_csv(f'{path}/USA Housing Dataset.csv')
    df['price'] = df['price'].round(0).astype('Int64')
    df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1, inplace=True)
    df.to_csv("datasets/housing.csv", index=False)
download_housing()

# Sensors
download_kagglehub_csv(
    kaggle_id="ziya07/iot-integrated-predictive-maintenance-dataset",
    file_path="{path}/predictive_maintenance_dataset.csv",
    drop_cols=['timestamp', 'machine_id'],
    out_csv="datasets/sensors.csv"
)