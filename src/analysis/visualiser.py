import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ResultsVisualiser:
            
    """
    Visualisation class for experiment results (heatmaps, boxplots, etc).
    Unified layout and color palette (blue/purple).
    """

    def __init__(self):
        self.palette = sns.color_palette(["#396CAA", "#AD3B37", "#0E7427", "#7C2EBB", "#727272", "#73BFD6", "#F59C96FF", "#A7F063", "#FFC000"])
        sns.set(style="whitegrid")

    def plot_feature_rank_barplot_faceted_pfi(
        self,
        df,
        dataset,
        pfi_col='pfi_features',
        model_col='model',
        imputer_col='imputer',
        missing_frac_col='missing_frac',
        seed_col='seed',
        groupby_col='imputer',  # or 'missing_frac'
        top_n=5,
        figsize=(14, 10),
    ):
        """
        For a given dataset, plot faceted barplots of mean feature ranks for top N features from the original model,
        grouped by either imputer or missing_frac. Facets are arranged in a 2x2 grid.

        Args:
            df (pd.DataFrame): DataFrame with columns for pfi_features (list), model, imputer, missing_frac, seed, and dataset.
            dataset (str): Dataset to filter.
            groupby_col (str): Either 'imputer' or 'missing_frac' for bar grouping.
            top_n (int): Number of top features to select from original model.
            figsize (tuple): Figure size.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        df = df[df['dataset'] == dataset].copy()
        models = sorted(df[model_col].unique())
        group_values = sorted(df[groupby_col].unique())

        n_facets = len(models)
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
        axes = axes.flatten()

        for ax_idx, model in enumerate(models):
            ax = axes[ax_idx]
            orig_mask = (df[model_col] == model) & (df[imputer_col] == 'original')
            orig_runs = df[orig_mask]
            rank_rows = []
            for _, row in orig_runs.iterrows():
                for rank, feature in enumerate(row[pfi_col]):
                    rank_rows.append({'feature': feature, 'rank': rank, 'seed': row[seed_col]})
            orig_ranks_df = pd.DataFrame(rank_rows)

            mean_ranks = orig_ranks_df.groupby('feature')['rank'].mean().sort_values()
            top_features = mean_ranks.head(top_n).index.tolist()

            agg_rows = []
            for group_val in group_values:
                sub = df[(df[model_col] == model) & (df[groupby_col] == group_val) & (df[imputer_col] != 'original' if groupby_col == 'imputer' else True)]
                for feature in top_features:
                    ranks = []
                    for _, row in sub.iterrows():
                        if feature in row[pfi_col]:
                            ranks.append(row[pfi_col].index(feature))
                    if ranks:
                        agg_rows.append({
                            groupby_col: group_val,
                            'feature': feature,
                            'mean_rank': np.mean(ranks),
                            'std_rank': np.std(ranks)
                        })

            agg_df = pd.DataFrame(agg_rows)

            bar_width = 0.8 / len(group_values) if len(group_values) > 0 else 0.8
            x = np.arange(len(top_features))
            palette = sns.color_palette("viridis", len(group_values))
            for i, group_val in enumerate(group_values):
                sub = agg_df[agg_df[groupby_col] == group_val]
                means = [sub[sub['feature'] == f]['mean_rank'].values[0] if f in sub['feature'].values else np.nan for f in top_features]
                stds = [sub[sub['feature'] == f]['std_rank'].values[0] if f in sub['feature'].values else 0 for f in top_features]
                ax.bar(
                    x + i * bar_width,
                    means,
                    bar_width,
                    yerr=stds,
                    label=f"{groupby_col}={group_val}",
                    color=palette[i],
                    alpha=0.8,
                )

            ax.set_xticks(x + bar_width * (len(group_values) - 1) / 2)
            ax.set_xticklabels(top_features, rotation=45)
            ax.set_title(f"Model: {model}")
            ax.set_ylabel("Mean Rank")
            ax.legend(fontsize=8, loc='best')

        for ax in axes[n_facets:]:
            ax.axis('off')

        fig.suptitle(f"Feature Ranks for Top {top_n} Features ({dataset})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x: str, y: str, hue: str = None, size: str = None, style: str = None, figsize=(8,6), alpha=0.7):
        """
        Plot a scatter plot for any two metrics, with optional hue, size, and style.

        Parameters:
        - df: pandas DataFrame with results
        - x: column for X axis
        - y: column for Y axis
        - hue: column for color grouping (e.g. 'imputer', 'task')
        - size: column for point size (e.g. 'missing_frac')
        - style: column for marker style (e.g. 'model')
        - figsize: figure size
        - alpha: point transparency
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, style=style, palette=self.palette, alpha=alpha)
        plt.title(f'Scatter: {y} vs {x}' + (f' by {hue}' if hue else ''), fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_facet_grid(self, df: pd.DataFrame, x: str, y: str, plot_type: str = 'box', hue: str = None, figsize=(12,8)):
        """
        Plot a FacetGrid with panels for each dataset. Each panel can show a boxplot, scatter, or heatmap.

        Parameters:
        - df: pandas DataFrame with results
        - x: column for X axis (grouping)
        - y: metric column to plot
        - plot_type: 'box', 'scatter', or 'heatmap'
        - hue: column for color grouping
        - figsize: figure size
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        datasets = df['dataset'].unique()
        n = len(datasets)
        ncols = int((n + 1) // 2)
        nrows = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            data = df[df['dataset'] == dataset]
            if plot_type == 'box':
                sns.boxplot(data=data, x=x, y=y, hue=hue, palette=self.palette, ax=ax, fill=True)
            elif plot_type == 'scatter':
                sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=self.palette, ax=ax)
            elif plot_type == 'heatmap':
                if x in data.columns and y in data.columns:
                    pivot = data.pivot_table(index=y, columns=x, values=hue if hue else y, aggfunc='mean')
                    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt='.2f', cbar=False)
            else:
                raise ValueError("plot_type must be 'box', 'scatter', or 'heatmap'")
            ax.set_title(f"Dataset: {dataset}", fontsize=12)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, df: pd.DataFrame, x: str, y: str, metric: str, aggfunc='mean', cmap='Blues', annot=True, figsize=(8,6)):
        """
        Plot a heatmap for selected dimensions and metric.

        Parameters:
        - df: pandas DataFrame with results
        - x: column name for X axis (e.g. 'missing_pct')
        - y: column name for Y axis (e.g. 'dataset')
        - metric: metric column to aggregate (e.g. 'imputer_rmse', 'delta_rmse', 'shap_rmse_overall')
        - aggfunc: aggregation function ('mean', 'median', etc)
        - cmap: colormap
        - annot: show values on heatmap
        - figsize: figure size
        """
        pivot = df.pivot_table(index=y, columns=x, values=metric, aggfunc=aggfunc)
        plt.figure(figsize=figsize)
        sns.heatmap(pivot, annot=annot, fmt='.2f', cmap=cmap, linewidths=0.5)
        plt.title(f'Heatmap: {metric} by {y} and {x}', fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_boxplot(self, df: pd.DataFrame, x: str, y: str, hue: str = None, metric: str = None, figsize=(8,6), show_points=False):
        """
        Plot a boxplot for a selected metric grouped by a dimension, with optional hue.

        Parameters:
        - df: pandas DataFrame with results
        - x: column name for X axis (grouping, e.g. 'imputer', 'missing_pct')
        - y: metric column to plot (e.g. 'imputer_rmse', 'delta_rmse', 'shap_rmse_overall')
        - hue: column for color grouping (e.g. 'task', 'model')
        - figsize: figure size
        - show_points: overlay individual data points (swarmplot)
        """
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df, x=x, y=y, hue=hue, palette=self.palette, showfliers=True)
        if show_points:
            sns.swarmplot(data=df, x=x, y=y, hue=hue, dodge=True, color="#868e96", alpha=0.5, size=3)
        plt.title(f'Boxplot: {y} grouped by {x}' + (f' and {hue}' if hue else ''), fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left') if hue else None
        plt.tight_layout()
        plt.show()

    def plot_multiple_pdp_curves(self, pdp_dicts, feature, labels=None, figsize=(10, 6)):
        """
        Plot multiple PDP curves for a given feature from different models/imputers.

        Parameters:
        - pdp_dicts: list of dicts, each dict is {feature: pd.DataFrame}
        - feature: feature name to plot
        - labels: list of legend labels (e.g. imputer/model names)
        - figsize: figure size
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        if labels is None:
            labels = [f'Model {i+1}' for i in range(len(pdp_dicts))]
        colors = self.palette if len(pdp_dicts) <= len(self.palette) else None
        for idx, (pdp, label) in enumerate(zip(pdp_dicts, labels)):
            df = pdp[feature]
            color = colors[idx] if colors is not None else None
            plt.plot(df['feature_value'], df['average_prediction'], marker='o', label=label, color=color)
        plt.xlabel('Feature Value')
        plt.ylabel('Average Prediction')
        plt.title(f'Partial Dependence: {feature}')
        plt.legend()
        plt.tight_layout()
        plt.show()