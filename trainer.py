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
        self.temperature = 0
    
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
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                loss = self.criterion(output, y)
                loss = loss.sum()
                total_loss += loss.item()
        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}", end=' ')
        print(f"Validation loss: {total_loss / len(val_loader):.4f}")
        return val_acc
        return total_loss / len(val_loader)

    def test(self, test_loader, output_file):
        self.model.eval()
        with torch.no_grad():
            with open(output_file, 'w') as f:
                f.write("id,label\n")
                for i, x in enumerate(test_loader):
                    x = x.to(self.device)
                    output = self.model(x)
                    pred = output.argmax(dim=1)
                    for j in range(pred.size(0)):
                        f.write(f"{i * test_loader.batch_size + j},{pred[j].item()}\n")
        print(f"Results saved to {output_file}")
    
    def generate_pseudo_labels(self, unlabeled_loader, max_samples_per_class, threshold=0.9):
        self.model.eval()
        pseudo_texts = []
        pseudo_labels = []
        class_counts = {0: 0, 1: 0} 

        with torch.no_grad():
            for x in tqdm(unlabeled_loader, desc='Gen Pseudo Label', leave=False, ncols=85):
                x = x.to(self.device)
                output = self.model(x)
                probs = F.softmax(output, dim=1)
                max_probs, pred = probs.max(dim=1)
                
                # 只选择高置信度的样本
                confident_idx = max_probs >= threshold
                if confident_idx.sum() > 0:
                    confident_samples = x[confident_idx].cpu().numpy()
                    confident_preds = pred[confident_idx].cpu().numpy()

                    # 针对每个类别控制样本数量
                    for sample, label in zip(confident_samples, confident_preds):
                        if class_counts[label] < max_samples_per_class:
                            pseudo_texts.append(sample)
                            pseudo_labels.append(label)
                            class_counts[label] += 1
                    pseudo_texts.extend(x[confident_idx].cpu().numpy())
                    pseudo_labels.extend(pred[confident_idx].cpu().numpy())

                    if all(count >= max_samples_per_class for count in class_counts.values()):
                        break
                    
        return pseudo_texts, pseudo_labels
