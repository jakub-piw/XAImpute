import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class ExperimentLogger:
    def __init__(self, base_dir: str, dataset: str, missing_rate: float, model: str):
        self.path = os.path.join(base_dir, dataset, str(missing_rate), model)
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, "plots"), exist_ok=True)

    def save_json(self, data: dict, filename: str):
        with open(os.path.join(self.path, filename), "w") as f:
            json.dump(data, f, indent=4)

    def save_csv(self, df: pd.DataFrame, filename: str):
        df.to_csv(os.path.join(self.path, filename), index=False)

    def save_plot(self, fig: plt.Figure, filename: str):
        fig.savefig(os.path.join(self.path, "plots", filename), bbox_inches='tight')
        plt.close(fig)

    def save_pickle(self, data: dict, filename: str):
        print('Saving pickle to', os.path.join(self.path, filename))
        with open(os.path.join(self.path, filename), "wb") as f:
            pickle.dump(data, f)   

    def save_dict(self, data: dict, filename: str):
        with open(os.path.join(self.path, filename), "w") as f:
            json.dump(data, f, indent=4)

