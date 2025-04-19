import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import MyDataset
from model import build_model
from logger import Logger


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

DEVICE = ('mps' if torch.backends.mps.is_available() else 'cpu')


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, total_samples, total_correct = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        output = model(data)
        loss = criterion(output, target - 1)  # class index should start from 0 for `CrossEntropyLoss`
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        total_correct += (predicted + 1 == target).sum().item()
    
    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / total_samples
    return train_loss, train_acc


def val(model, val_loader, criterion):
    model.eval()
    total_loss, total_samples, total_correct = 0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            output = model(data)
            total_loss += criterion(output, target - 1).item()
            
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted + 1 == target).sum().item()
    
    val_loss = total_loss / len(val_loader)
    val_acc = total_correct / total_samples
    return val_loss, val_acc


def worker(data_path='./data/train', model_path='./output/trained_cnn.pth', 
           batch_size=32, epochs=50, lr=0.001, patience=10, dropout_rate=0.1, num_classes=15):    
    LOG = Logger('log/train.log', level='info')
    LOG.logger.info('%s train %s' % ('-' * 10, '-' * 10))
    
    dataset = MyDataset(path=data_path, is_train=True)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.15, stratify=dataset.label, random_state=42)
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(MyDataset(path=data_path, is_train=False), val_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = build_model(dropout_rate=dropout_rate, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_train_acc, best_val_acc, cnt = 0, 0, 0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = val(model, val_loader, criterion)
        scheduler.step()

        LOG.logger.info('epoch %d\ttrain_loss %.4f\ttrain_acc %.4f\tval_loss %.4f\tval_acc %.4f' % (epoch, train_loss, train_acc, val_loss, val_acc))
        if val_acc >= best_val_acc:
            best_train_acc = train_acc
            best_val_acc = val_acc
            cnt = 0
            torch.save(model.state_dict(), model_path)
        else:
            cnt += 1
            if cnt >= patience:
                break

    LOG.logger.info('work finish, train accuracy %.04f val accuracy %.04f for model %s' % (best_train_acc, best_val_acc, model_path))
    return best_train_acc
