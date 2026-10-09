"""Анализ результатов Random Search и Bayesian Optimization."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


RANDOM_SEARCH_FILE = 'random_search_history.json'
BAYESIAN_FILE = 'optimization_history_kappa2.json'
BAYESIAN_FILE = 'optimization_histor_EI.json'
PLOTS_DIR = 'optimization_plots'


def load_history(filename):
    """Загружает историю экспериментов из JSON."""
    with open(filename, 'r') as f:
        history = json.load(f)

    return history


def history_to_dataframe(history):
    """Преобразует историю trials в DataFrame."""
    rows = []
    for item in history:
        row = {'trial': item['trial'], 'accuracy': item['accuracy']}
        for key, value in item['params'].items():
            row[key] = value
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_parameters(df):
    """Преобразует гиперпараметры в числовой вид (sgd — 0, adam — 1, adamw — 2)."""
    data = df.copy()

    if 'optimizer' in data.columns:
        optimizer_mapping = {'sgd': 0, 'adam': 1, 'adamw': 2}
        data['optimizer'] = data['optimizer'].map(optimizer_mapping)

    parameters = [
        column
        for column in data.columns
        if column not in ['trial', 'accuracy']
    ]

    return data[parameters]


def save_plot(filename):
    """Сохраняет текущий график в папку с результатами."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    filepath = os.path.join(PLOTS_DIR, filename)

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"График сохранен: {filepath}")


def plot_objective(random_df, bayesian_df):
    """Строит значение целевой функции для каждого trial."""
    plt.figure(figsize=(10, 6))

    if random_df is not None:
        plt.plot(
            random_df['trial'],
            random_df['accuracy'],
            marker='o',
            label='Random Search'
        )

        random_best = np.maximum.accumulate(random_df['accuracy'])

        plt.plot(
            random_df['trial'],
            random_best,
            linestyle='--',
            linewidth=2,
            label='Random Search best'
        )

    if bayesian_df is not None:
        plt.plot(
            bayesian_df['trial'],
            bayesian_df['accuracy'],
            marker='o',
            label='Bayesian Optimization'
        )

        bayesian_best = np.maximum.accumulate(bayesian_df['accuracy'])

        plt.plot(
            bayesian_df['trial'],
            bayesian_best,
            linestyle='--',
            linewidth=2,
            label='Bayesian Optimization best'
        )

    plt.xlabel('Номер trial')
    plt.ylabel('Validation accuracy')
    plt.title('Значение целевой функции в зависимости от номера trial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_plot('01_objective.png')

    plt.show()


def plot_parameter_space(random_df, bayesian_df):
    """Строит PCA-проекцию гиперпараметров отдельно для двух методов."""
    datasets = []

    if random_df is not None:
        random_parameters = prepare_parameters(random_df)
        random_parameters['method'] = 'Random Search'
        random_parameters['accuracy'] = random_df['accuracy'].values
        datasets.append(random_parameters)

    if bayesian_df is not None:
        bayesian_parameters = prepare_parameters(bayesian_df)
        bayesian_parameters['method'] = 'Bayesian Optimization'
        bayesian_parameters['accuracy'] = bayesian_df['accuracy'].values
        datasets.append(bayesian_parameters)

    data = pd.concat(datasets, ignore_index=True)

    parameter_names = [
        column
        for column in data.columns
        if column not in ['method', 'accuracy']
    ]

    X = data[parameter_names].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    data['PC1'] = X_pca[:, 0]
    data['PC2'] = X_pca[:, 1]

    min_accuracy = data['accuracy'].min()
    max_accuracy = data['accuracy'].max()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharex=True,
        sharey=True
    )

    scatter = None

    # Random Search
    ax = axes[0]

    if random_df is not None:
        random_points = data[data['method'] == 'Random Search']

        scatter = ax.scatter(
            random_points['PC1'],
            random_points['PC2'],
            c=random_points['accuracy'],
            cmap='viridis',
            vmin=min_accuracy,
            vmax=max_accuracy,
            s=100,
            marker='o',
            edgecolors='black'
        )

        # Номер trial около каждой точки
        for index, (_, point) in enumerate(random_points.iterrows(), start=1):
            ax.annotate(
                str(index),
                (point['PC1'], point['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        ax.set_title('Random Search')

    else:
        ax.set_title('Random Search - нет данных')

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )
    ax.grid(True, alpha=0.3)

    # Bayesian Optimization
    ax = axes[1]

    if bayesian_df is not None:
        bayesian_points = data[data['method'] == 'Bayesian Optimization']

        scatter = ax.scatter(
            bayesian_points['PC1'],
            bayesian_points['PC2'],
            c=bayesian_points['accuracy'],
            cmap='viridis',
            vmin=min_accuracy,
            vmax=max_accuracy,
            s=100,
            marker='o',
            edgecolors='black'
        )

        # Номер trial около каждой точки
        for index, (_, point) in enumerate(bayesian_points.iterrows(), start=1):
            ax.annotate(
                str(index),
                (point['PC1'], point['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        ax.set_title('Bayesian Optimization')

    else:
        ax.set_title('Bayesian Optimization - нет данных')

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Пространство протестированных гиперпараметров',
        fontsize=14
    )

    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=axes,
            fraction=0.025,
            pad=0.04
        )
        colorbar.set_label('Validation accuracy')

    plt.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.1,
        top=0.88,
        wspace=0.15
    )

    save_plot('02_parameter_space.png')

    plt.show()


def calculate_feature_importance(df):
    """Оценивает важность гиперпараметров с помощью Random Forest."""
    X = prepare_parameters(df)
    y = df['accuracy'].values

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=5
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        'parameter': X.columns,
        'importance': model.feature_importances_
    })

    importance = importance.sort_values(
        'importance',
        ascending=False
    )

    return importance


def plot_feature_importance(random_df, bayesian_df):
    """Строит важность гиперпараметров."""
    if random_df is not None:
        random_importance = calculate_feature_importance(random_df)

        print('\nВажность гиперпараметров - Random Search:')
        print(random_importance.to_string(index=False))

        plt.figure(figsize=(10, 7))

        plt.barh(
            random_importance['parameter'],
            random_importance['importance']
        )

        plt.xlabel('Важность')
        plt.ylabel('Гиперпараметр')
        plt.title('Важность гиперпараметров - Random Search')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        save_plot('03_feature_importance_random_search.png')

        plt.show()

    if bayesian_df is not None:
        bayesian_importance = calculate_feature_importance(bayesian_df)

        print('\nВажность гиперпараметров - Bayesian Optimization:')
        print(bayesian_importance.to_string(index=False))

        plt.figure(figsize=(10, 7))

        plt.barh(
            bayesian_importance['parameter'],
            bayesian_importance['importance']
        )

        plt.xlabel('Важность')
        plt.ylabel('Гиперпараметр')
        plt.title('Важность гиперпараметров - Bayesian Optimization')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        save_plot('04_feature_importance_bayesian_optimization.png')

        plt.show()


def print_results(random_df, bayesian_df):
    """Выводит лучшие результаты."""
    if random_df is not None:
        best_random = random_df.loc[random_df['accuracy'].idxmax()]

        print('\nRandom Search:')
        print(f"Лучший accuracy: {best_random['accuracy']:.4f}")
        print(f"Trial: {int(best_random['trial'])}")
        print('Гиперпараметры:')

        for key, value in best_random.items():
            if key not in ['trial', 'accuracy']:
                print(f"  {key}: {value}")

    if bayesian_df is not None:
        best_bayesian = bayesian_df.loc[
            bayesian_df['accuracy'].idxmax()
        ]

        print('\nBayesian Optimization:')
        print(f"Лучший accuracy: {best_bayesian['accuracy']:.4f}")
        print(f"Trial: {int(best_bayesian['trial'])}")
        print('Гиперпараметры:')

        for key, value in best_bayesian.items():
            if key not in ['trial', 'accuracy']:
                print(f"  {key}: {value}")


def main():
    print('Анализ результатов оптимизации')

    os.makedirs(PLOTS_DIR, exist_ok=True)

    random_df = None
    bayesian_df = None

    # Загружаем Random Search
    try:
        random_history = load_history(RANDOM_SEARCH_FILE)
        random_df = history_to_dataframe(random_history)

        print(f"Random Search: {len(random_df)} trials")

    except FileNotFoundError:
        print(f"Файл {RANDOM_SEARCH_FILE} не найден")

    # Загружаем Bayesian Optimization
    try:
        bayesian_history = load_history(BAYESIAN_FILE)
        bayesian_df = history_to_dataframe(bayesian_history)

        print(f"Bayesian Optimization: {len(bayesian_df)} trials")

    except FileNotFoundError:
        print(f"Файл {BAYESIAN_FILE} не найден")

    if random_df is None and bayesian_df is None:
        print('Нет данных для анализа')
        return

    # Значение целевой функции
    plot_objective(random_df, bayesian_df)

    # Пространство гиперпараметров
    plot_parameter_space(random_df, bayesian_df)

    # Важность гиперпараметров
    plot_feature_importance(random_df, bayesian_df)

    # Лучшие результаты
    print_results(random_df, bayesian_df)

    print(f'\nВсе графики сохранены в папку: {PLOTS_DIR}')


if __name__ == '__main__':
    main()