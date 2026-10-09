import os
import torch
import torchvision
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet18_Weights

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from metrics import *

import matplotlib.pyplot as plt

DATA_DIR  = './dataset/indoorCVPR_09/Images'
CLASSES = os.listdir(DATA_DIR)

class ImageClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
    
    def training_step(self, batch):
        images, labels = batch 
        images = images.to(self.device)       # Move images to device
        labels = labels.to(self.device)       # Move labels to device
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.to(self.device)       # Move images to device
        labels = labels.to(self.device)       # Move labels to device
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def main(batch_size:int = 25, device:str = 'cuda', num_epochs:int = 10, opt_func = torch.optim.Adam, lr:float = 1e-6 ):

    random_seed = 42
    torch.manual_seed(random_seed)

    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    device = torch.device(device)

    transformations = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
        ])
    
    dataset = ImageFolder(DATA_DIR, transform = transformations)

    train_ds, val_ds, test_ds = random_split(dataset, [13000, 2000, 620])
    print("Тренировочный класс: ", len(train_ds), "\n",
          "Валидационный класс: ", len(val_ds), "\n",
          "Тестовый класс:  ",  len(test_ds) 
    )

    batch_size = 25

    train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)

    model = ResNet()
    model.device = device           # Set device for data transfer
    model = model.to(device)        # Move model to device
    #print(evaluate(model, val_dl))
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_accuracies(history)
    plot_losses(history)
    print(evaluate(model, val_dl))



if __name__ == '__main__':
    main()