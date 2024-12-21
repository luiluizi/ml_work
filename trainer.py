import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, criterion, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    def update_lr(self, new_lr):
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.lr = new_lr

        
    def train_epoch(self, train_loader, pseudo_loader=None):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 训练伪标签数据
        if pseudo_loader:
            for batch in tqdm(pseudo_loader, desc='Pseudo Training', leave=False, ncols=85):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss = loss.sum()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            return total_loss / len(pseudo_loader)

        # 训练有标签数据
        for batch in tqdm(train_loader, desc='Training', leave=False, ncols=85):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f'train acc:{correct / total:.4f}')
            
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        return correct / total
    
    def generate_pseudo_labels(self, unlabeled_loader, threshold=0.9):
        self.model.eval()
        pseudo_texts = []
        pseudo_labels = []
        
        with torch.no_grad():
            for x in tqdm(unlabeled_loader, desc='Gen Pseudo Label', leave=False, ncols=85):
                x = x.to(self.device)
                output = self.model(x)
                probs = F.softmax(output, dim=1)
                max_probs, pred = probs.max(dim=1)
                
                # 只选择高置信度的样本
                confident_idx = max_probs >= threshold
                if confident_idx.sum() > 0:
                    pseudo_texts.extend(x[confident_idx].cpu().numpy())
                    pseudo_labels.extend(pred[confident_idx].cpu().numpy())
                    
        return pseudo_texts, pseudo_labels
    
