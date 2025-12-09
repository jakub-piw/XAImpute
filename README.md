# XAImpute

A repository for running experiments testing the impact of missing values and different imputation methods on the performance of ML models and XAI methods.

## Usage

- **`run_experiment.py`**  
  Runs all planned experiments.

- **`datasets/downloading.py`**  
  Downloads all datasets used in these experiments.

## Directory Structure

The `src` directory contains all needed modules:

- **`data`**  
  Dataset loaders and missing values injectors.

- **`imputation`**  
  Imputation methods.

- **`models`**  
  Modeling pipeline: training, hyperparameter tuning, and evaluation of models and imputers.

- **`xai`**  
  Wrappers for XAI methods, evaluation through visualization and quantification.

- **`experiment`**  
  Wraps all modules and creates a full multifaceted pipeline from loading data to saving final results.

- **`analysis`**  
  Tools to aggregate, analyze, and visualize final experiment results.