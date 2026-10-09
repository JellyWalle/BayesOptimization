"""
Запуск BO для ResNet с 18 гиперпараметрами.
"""
import os
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from optimize import BayesianOptimization
from parameters import FloatParameter
from train import ResNet, ImageClassificationBase
from metrics import accuracy

DATA_DIR = './dataset/indoorCVPR_09/Images'
CLASSES = os.listdir(DATA_DIR)
NUM_CLASSES = len(CLASSES)

class ResNetWithDropout(ImageClassificationBase):
    """ResNet с dropout."""
    
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        import torchvision.models as models
        from torchvision.models.resnet import ResNet18_Weights
        import torch.nn as nn
        
        self.network = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.network.fc.in_features
        
        # Добавляем dropout
        self.network.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
    
    def forward(self, xb):
        return self.network(xb)


def objective_function(
    lr, weight_decay, batch_size, optimizer,
    momentum, beta1, beta2,
    dropout_rate, label_smoothing,
    brightness, contrast, saturation, hue,
    affine_degrees, affine_translate_x, affine_translate_y, affine_scale,
    random_erase_p
):
    batch_size = int(batch_size)
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        transforms.RandomAffine(
            degrees=affine_degrees,
            translate=(affine_translate_x, affine_translate_y),
            scale=(affine_scale, affine_scale)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=random_erase_p)
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
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size * 2, 
        num_workers=2, 
        pin_memory=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetWithDropout(dropout_rate=dropout_rate)
    model.device = device
    model = model.to(device)
    
    if optimizer == 'sgd':
        opt = torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    elif optimizer == 'adam':
        opt = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay
        )
    else:  # adamw
        opt = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay
        )
    
    trial_epochs = 3
    model.train()
    
    for epoch in range(trial_epochs):
        train_losses = []
        for batch in train_dl:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            opt.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(
                outputs, 
                labels, 
                label_smoothing=label_smoothing
            )
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
    
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_dl:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = val_correct / val_total
    
    return val_accuracy


pbounds = {
    'lr': FloatParameter('lr', 1e-5, 1e-2, log=True),
    'weight_decay': FloatParameter('weight_decay', 1e-6, 1e-2, log=True),
    'batch_size': (4, 64, int),
    'optimizer': ['sgd', 'adam', 'adamw'],
    'momentum': (0.8, 0.99),
    'beta1': (0.8, 0.99),
    'beta2': (0.9, 0.999),
    'dropout_rate': (0.0, 0.5),
    'label_smoothing': (0.0, 0.2),
    'brightness': (0.0, 0.5),
    'contrast': (0.0, 0.5),
    'saturation': (0.0, 0.5),
    'hue': (0.0, 0.2),
    'affine_degrees': (0.0, 30.0),
    'affine_translate_x': (0.0, 0.2),
    'affine_translate_y': (0.0, 0.2),
    'affine_scale': (0.8, 1.2),
    'random_erase_p': (0.0, 0.5),
}

def main():
    print("Запуск байесовской оптимизации для ResNet")
    print(f"Количество гиперпараметров: {len(pbounds)}")

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=49,
        verbose=2
    )

    optimizer.maximize(init_points=5, n_iter=15)
    
    print(f"\nЛучшая validation accuracy: {optimizer.max['target']:.4f}")
    print("\nЛучшие гиперпараметры:")
    for key, value in optimizer.max['params'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    import json
    history = []
    for i, res in enumerate(optimizer.res):
        history.append({
            'trial': i + 1,
            'accuracy': res['target'],
            'params': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) 
                      for k, v in res['params'].items()}
        })
    
    with open('optimization_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Всего trials: {len(optimizer.res)}")

if __name__ == '__main__':
    main()