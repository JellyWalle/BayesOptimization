"""Сравнение TPESampler и RandomSampler в Optuna для настройки ResNet."""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.importance import get_param_importances

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


DATA_DIR = './dataset/indoorCVPR_09/Images'
N_TRIALS = 20
RANDOM_SEED = 47
TRIAL_EPOCHS = 3

CLASSES = os.listdir(DATA_DIR)
NUM_CLASSES = len(CLASSES)


class ResNetWithDropout(nn.Module):
    """ResNet18 с настраиваемым dropout."""

    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.network = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


def suggest_parameters(trial):
    """Выбирает гиперпараметры; lr и weight_decay — в логарифмической шкале."""
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_int('batch_size', 16, 128),
        'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw']),
        'momentum': trial.suggest_float('momentum', 0.8, 0.99),
        'beta1': trial.suggest_float('beta1', 0.8, 0.99),
        'beta2': trial.suggest_float('beta2', 0.9, 0.999),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'brightness': trial.suggest_float('brightness', 0.0, 0.5),
        'contrast': trial.suggest_float('contrast', 0.0, 0.5),
        'saturation': trial.suggest_float('saturation', 0.0, 0.5),
        'hue': trial.suggest_float('hue', 0.0, 0.2),
        'affine_degrees': trial.suggest_float('affine_degrees', 0.0, 30.0),
        'affine_translate_x': trial.suggest_float('affine_translate_x', 0.0, 0.2),
        'affine_translate_y': trial.suggest_float('affine_translate_y', 0.0, 0.2),
        'affine_scale': trial.suggest_float('affine_scale', 0.8, 1.2),
        'random_erase_p': trial.suggest_float('random_erase_p', 0.0, 0.5)
    }
    return params


def objective_function(trial):
    """Обучает ResNet с параметрами Optuna trial, возвращает validation accuracy."""
    params = suggest_parameters(trial)
    batch_size = int(params['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Аугментации
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=params['brightness'],
            contrast=params['contrast'],
            saturation=params['saturation'],
            hue=params['hue']
        ),
        transforms.RandomAffine(
            degrees=params['affine_degrees'],
            translate=(params['affine_translate_x'], params['affine_translate_y']),
            scale=(params['affine_scale'], params['affine_scale'])
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=params['random_erase_p'])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Dataset
    full_dataset = ImageFolder(DATA_DIR, transform=train_transform)
    torch.manual_seed(42)
    train_ds, val_ds, _ = random_split(full_dataset, [13000, 2000, 620])
    # Отдельный dataset сохраняет тренировочные аугментации.
    val_dataset = ImageFolder(DATA_DIR, transform=val_transform)
    val_ds = Subset(val_dataset, val_ds.indices)

    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=2, pin_memory=True)

    # Модель
    model = ResNetWithDropout(dropout_rate=params['dropout_rate']).to(device)

    # Optimizer
    if params['optimizer'] == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=params['lr'],
                              momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=params['lr'],
                               betas=(params['beta1'], params['beta2']),
                               weight_decay=params['weight_decay'])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                betas=(params['beta1'], params['beta2']),
                                weight_decay=params['weight_decay'])

    # Обучение
    model.train()

    for epoch in range(TRIAL_EPOCHS):
        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, label_smoothing=params['label_smoothing'])
            loss.backward()
            opt.step()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    return val_accuracy


def run_optimization(method, study_name, seed, n_trials):
    """Запускает TPESampler (bayesian) или RandomSampler (random)."""
    if method == 'bayesian':
        sampler = TPESampler(seed=seed, multivariate=True)
    elif method == 'random':
        sampler = RandomSampler(seed=seed)
    else:
        raise ValueError("method должен быть 'bayesian' или 'random'")

    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name)

    if method == 'bayesian':
        print("БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ")
        print("Optuna TPESampler")
    else:
        print("СЛУЧАЙНЫЙ ПОИСК")
        print("Optuna RandomSampler")

    print("=" * 70)
    study.optimize(objective_function, n_trials=n_trials)
    return study


def get_history(study):
    """Преобразует историю Optuna Study в DataFrame."""
    rows = []
    best_value = -np.inf

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        value = trial.value
        best_value = max(best_value, value)
        row = {'trial': trial.number + 1, 'accuracy': value, 'best_accuracy': best_value}

        for key, value in trial.params.items():
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def plot_objective_history(bayesian_history, random_history):
    """Строит validation accuracy в зависимости от номера trial."""
    plt.figure(figsize=(14, 7))
    plt.plot(bayesian_history['trial'], bayesian_history['accuracy'], 'o--', linewidth=2,
             label='Bayesian Optimization')
    plt.plot(random_history['trial'], random_history['accuracy'], 'o--', linewidth=2, label='Random Search')
    plt.plot(bayesian_history['trial'], bayesian_history['best_accuracy'], linewidth=3, label='BO best-so-far')
    plt.plot(random_history['trial'], random_history['best_accuracy'], linewidth=3, label='Random best-so-far')
    plt.xlabel('Номер шага / trial', fontsize=14)
    plt.ylabel('Validation accuracy', fontsize=14)
    plt.title('Сравнение Bayesian Optimization и Random Search', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('01_objective_comparison.png', dpi=150)
    plt.close()
    print('Сохранён: 01_objective_comparison.png')


def prepare_features(history):
    """Преобразует гиперпараметры в числовую матрицу с one-hot кодированием optimizer."""
    data = history.copy()
    data = pd.get_dummies(data, columns=['optimizer'], dtype=float)
    feature_columns = [col for col in data.columns if col not in ['trial', 'accuracy', 'best_accuracy', 'method']]
    X = data[feature_columns].astype(float)
    return X


def plot_search_space(bayesian_history, random_history):
    """Строит PCA-проекцию конфигураций; цвет показывает validation accuracy."""
    bayesian_history = bayesian_history.copy()
    random_history = random_history.copy()
    bayesian_history['method'] = 'Bayesian Optimization'
    random_history['method'] = 'Random Search'
    history = pd.concat([bayesian_history, random_history], ignore_index=True)

    X = prepare_features(history)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    history['PC1'] = X_pca[:, 0]
    history['PC2'] = X_pca[:, 1]

    min_accuracy = history['accuracy'].min()
    max_accuracy = history['accuracy'].max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    bayesian_points = history[history['method'] == 'Bayesian Optimization']
    random_points = history[history['method'] == 'Random Search']

    scatter1 = axes[0].scatter(bayesian_points['PC1'], bayesian_points['PC2'],
                               c=bayesian_points['accuracy'], s=100, alpha=0.85, cmap='viridis',
                               vmin=min_accuracy, vmax=max_accuracy, edgecolors='k', linewidths=0.5)
    for _, point in bayesian_points.iterrows():
        axes[0].annotate(
            str(int(point['trial'])),
            (point['PC1'], point['PC2']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    axes[0].set_title('Bayesian Optimization', fontsize=16)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=14)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    scatter2 = axes[1].scatter(random_points['PC1'], random_points['PC2'],
                               c=random_points['accuracy'], s=100, alpha=0.85, cmap='viridis',
                               vmin=min_accuracy, vmax=max_accuracy, edgecolors='k', linewidths=0.5)
    for _, point in random_points.iterrows():
        axes[1].annotate(
            str(int(point['trial'])),
            (point['PC1'], point['PC2']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    axes[1].set_title('Random Search', fontsize=16)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=14)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        'Пространство протестированных гиперпараметров',
        fontsize=16
    )
    cbar = fig.colorbar(
        scatter1,
        ax=axes,
        fraction=0.025,
        pad=0.04
    )
    cbar.set_label('Validation accuracy', fontsize=12)
    plt.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.1,
        top=0.88,
        wspace=0.15
    )

    plt.savefig(
        '02_hyperparameter_space_pca.png',
        dpi=150,
        bbox_inches='tight'
    )

    plt.close()

    print('Сохранён: 02_hyperparameter_space_pca.png')


def plot_parameter_importance(study, method_name):
    """Оценивает важность гиперпараметров средствами Optuna."""
    importances = get_param_importances(study)
    names = list(importances.keys())
    values = list(importances.values())

    # Сортируем для горизонтального графика
    order = np.argsort(values)
    names = np.array(names)[order]
    values = np.array(values)[order]
    plt.figure(figsize=(11, 8))
    plt.barh(names, values)
    plt.xlabel('Относительная важность', fontsize=14)
    plt.ylabel('Гиперпараметр', fontsize=14)
    plt.title(f'Важность гиперпараметров — {method_name}', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    filename = '03_parameter_importance_' + method_name.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Сохранён: {filename}')
    return importances


def save_history(history, filename):
    """Сохраняет историю Optuna trials."""
    history.to_json(filename, orient='records', indent=2)
    print(f'Сохранён: {filename}')


def print_results(bayesian_study, random_study):
    print("Результаты сравнения Bayesian Optimization и Random Search:")
    bo_best = bayesian_study.best_trial
    random_best = random_study.best_trial

    print("\nBayesian Optimization:")
    print(f"  Best accuracy: {bo_best.value:.4f}")
    print(f"  Trial: {bo_best.number + 1}")

    print("\nRandom Search:")
    print(f"  Best accuracy: {random_best.value:.4f}")
    print(f"  Trial: {random_best.number + 1}")

    print("\nРазница:")
    difference = bo_best.value - random_best.value
    print(f"  BO - Random = {difference:.4f}")

    print("\nЛучшие параметры BO:")

    for key, value in bo_best.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\nЛучшие параметры Random Search:")

    for key, value in random_best.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

def load_study_from_json(filename, study_name):
    """Восстанавливает Optuna Study из сохранённой истории JSON."""

    with open(filename, 'r') as f:
        history = json.load(f)

    distributions = {
        'lr': optuna.distributions.FloatDistribution(
            1e-5,
            1e-2,
            log=True
        ),
        'weight_decay': optuna.distributions.FloatDistribution(
            1e-6,
            1e-2,
            log=True
        ),
        'batch_size': optuna.distributions.IntDistribution(
            16,
            128
        ),
        'optimizer': optuna.distributions.CategoricalDistribution(
            ['sgd', 'adam', 'adamw']
        ),
        'momentum': optuna.distributions.FloatDistribution(
            0.8,
            0.99
        ),
        'beta1': optuna.distributions.FloatDistribution(
            0.8,
            0.99
        ),
        'beta2': optuna.distributions.FloatDistribution(
            0.9,
            0.999
        ),
        'dropout_rate': optuna.distributions.FloatDistribution(
            0.0,
            0.5
        ),
        'label_smoothing': optuna.distributions.FloatDistribution(
            0.0,
            0.2
        ),
        'brightness': optuna.distributions.FloatDistribution(
            0.0,
            0.5
        ),
        'contrast': optuna.distributions.FloatDistribution(
            0.0,
            0.5
        ),
        'saturation': optuna.distributions.FloatDistribution(
            0.0,
            0.5
        ),
        'hue': optuna.distributions.FloatDistribution(
            0.0,
            0.2
        ),
        'affine_degrees': optuna.distributions.FloatDistribution(
            0.0,
            30.0
        ),
        'affine_translate_x': optuna.distributions.FloatDistribution(
            0.0,
            0.2
        ),
        'affine_translate_y': optuna.distributions.FloatDistribution(
            0.0,
            0.2
        ),
        'affine_scale': optuna.distributions.FloatDistribution(
            0.8,
            1.2
        ),
        'random_erase_p': optuna.distributions.FloatDistribution(
            0.0,
            0.5
        )
    }

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name
    )

    for item in history:
        params = {
            key: value
            for key, value in item.items()
            if key not in [
                'trial',
                'accuracy',
                'best_accuracy'
            ]
        }

        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=item['accuracy']
        )

        study.add_trial(trial)

    return study

def main():
    print("Сравнение TPESampler и RandomSampler в Optuna для настройки ResNet")
    print("ResNet + Optuna")
    print(f"Количество гиперпараметров: 18")
    print(f"Количество trials: {N_TRIALS}")
    print(f"Random seed: {RANDOM_SEED}")

    # Bayesian Optimization
    #bayesian_study = run_optimization(
    #    method='bayesian',
    #    study_name='resnet_bayesian',
    #    seed=RANDOM_SEED,
    #    n_trials=N_TRIALS
    #)
#
    ## Random Search
    #random_study = run_optimization(
    #    method='random',
    #    study_name='resnet_random',
    #    seed=RANDOM_SEED,
    #    n_trials=N_TRIALS
    #)
    bayesian_study = load_study_from_json('optuna_bayesian_history.json', 'resnet_bayesian')
    random_study = load_study_from_json('optuna_random_history.json', 'resnet_random')

    # Истории
    bayesian_history = get_history(bayesian_study)
    random_history = get_history(random_study)
    #save_history(bayesian_history, 'optuna_bayesian_history.json')
    #save_history(random_history, 'optuna_random_history.json')

    # График целевой функции
    plot_objective_history(bayesian_history, random_history)

    # Пространство параметров
    plot_search_space(bayesian_history, random_history)

    # Важность гиперпараметров
    bo_importances = plot_parameter_importance(bayesian_study, 'Bayesian Optimization')
    random_importances = plot_parameter_importance(random_study, 'Random Search')

    # Сохраняем важность
    with open('parameter_importance.json', 'w') as f:
        json.dump({'bayesian': bo_importances, 'random': random_importances}, f, indent=2)

    print('Сохранён: parameter_importance.json')

    # Итоги
    #print_results(bayesian_study, random_study)
    print("\n" + "=" * 70)
    print("Все эксперименты завершены.")
    print("=" * 70)


if __name__ == '__main__':
    main()
