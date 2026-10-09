"""
Случайный поиск для ResNet (18 гиперпараметров)
Сохраняет историю в random_search_history.json
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

from metrics import accuracy

DATA_DIR = './dataset/indoorCVPR_09/Images'
CLASSES = os.listdir(DATA_DIR)
NUM_CLASSES = len(CLASSES)


class ResNetWithDropout(nn.Module):
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

SEARCH_SPACE = {
    'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
    'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
    'batch_size': {'type': 'int', 'low': 16, 'high': 128},
    'optimizer': {'type': 'categorical', 'choices': ['sgd', 'adam', 'adamw']},
    'momentum': {'type': 'float', 'low': 0.8, 'high': 0.99},
    'beta1': {'type': 'float', 'low': 0.8, 'high': 0.99},
    'beta2': {'type': 'float', 'low': 0.9, 'high': 0.999},
    'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
    'label_smoothing': {'type': 'float', 'low': 0.0, 'high': 0.2},
    'brightness': {'type': 'float', 'low': 0.0, 'high': 0.5},
    'contrast': {'type': 'float', 'low': 0.0, 'high': 0.5},
    'saturation': {'type': 'float', 'low': 0.0, 'high': 0.5},
    'hue': {'type': 'float', 'low': 0.0, 'high': 0.2},
    'affine_degrees': {'type': 'float', 'low': 0.0, 'high': 30.0},
    'affine_translate_x': {'type': 'float', 'low': 0.0, 'high': 0.2},
    'affine_translate_y': {'type': 'float', 'low': 0.0, 'high': 0.2},
    'affine_scale': {'type': 'float', 'low': 0.8, 'high': 1.2},
    'random_erase_p': {'type': 'float', 'low': 0.0, 'high': 0.5},
}


def sample_random_config(search_space, rng):
    """Генерирует случайную конфигурацию из пространства поиска."""
    config = {}
    for name, space in search_space.items():
        if space['type'] == 'float':
            if space.get('log'):
                val = np.exp(rng.uniform(np.log(space['low']), np.log(space['high'])))
            else:
                val = rng.uniform(space['low'], space['high'])
            config[name] = val
        elif space['type'] == 'int':
            config[name] = rng.randint(space['low'], space['high'] + 1)
        elif space['type'] == 'categorical':
            config[name] = rng.choice(space['choices'])
    return config


def objective_function(config):
    """Обучает ResNet с заданными гиперпараметрами, возвращает val_accuracy."""
    batch_size = int(config['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=config['brightness'],
            contrast=config['contrast'],
            saturation=config['saturation'],
            hue=config['hue']
        ),
        transforms.RandomAffine(
            degrees=config['affine_degrees'],
            translate=(config['affine_translate_x'], config['affine_translate_y']),
            scale=(config['affine_scale'], config['affine_scale'])
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=config['random_erase_p'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(DATA_DIR, transform=train_transform)
    torch.manual_seed(42)
    train_ds, val_ds, _ = random_split(full_dataset, [13000, 2000, 620])
    # Отдельный dataset сохраняет тренировочные аугментации.
    val_dataset = ImageFolder(DATA_DIR, transform=val_transform)
    val_ds = Subset(val_dataset, val_ds.indices)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=2, pin_memory=True)

    model = ResNetWithDropout(dropout_rate=config['dropout_rate']).to(device)

    opt_name = config['optimizer']
    if opt_name == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=config['lr'],
                              momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif opt_name == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'],
                               betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])

    trial_epochs = 3
    model.train()
    for _ in range(trial_epochs):
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, label_smoothing=config['label_smoothing'])
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main():
    n_trials = 20
    rng = np.random.RandomState(49)

    print("Запуск случайного поиска гиперпараметров для ResNet")
    print(f"Количество гиперпараметров: {len(SEARCH_SPACE)}")
    print(f"Количество trials: {n_trials}")

    history = []
    best_accuracy = 0.0

    for i in range(n_trials):
        print(f"\n Trial {i+1}/{n_trials}")
        config = sample_random_config(SEARCH_SPACE, rng)

        # Выводим конфигурацию
        for k, v in config.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        # Вычисляем accuracy на валидации
        val_acc = objective_function(config)
        print(f" Val Accuracy: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc

        history.append({
            'trial': i + 1,
            'accuracy': val_acc,
            'best_accuracy_so_far': best_accuracy,
            'params': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                       for k, v in config.items()}
        })

    # Сохраняем историю
    with open('random_search_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Лучшая validation accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()