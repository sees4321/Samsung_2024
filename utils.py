import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as opt

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OPT_DICT = {'Adam':opt.Adam,
            'AdamW':opt.AdamW,
            'SGD':opt.SGD}

def ManualSeed(seed:int,deterministic=False):
    # random seed 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic: # True면 cudnn seed 고정 (정확한 재현 필요한거 아니면 제외)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def split_subjects(test_subject, num_subjects=32, train_size=28):
    subjects = [i for i in range(num_subjects) if i != test_subject]
    random.shuffle(subjects)

    train_subjects = [False]*(num_subjects)
    val_subjects = [False]*(num_subjects)
    for i, v in enumerate(subjects):
        if i < train_size:
            train_subjects[v] = True
        else:
            val_subjects[v] = True

    return train_subjects, val_subjects

def expand_dim_(data:np.ndarray):
    return np.expand_dims(data,2)

class CustomDataSet(Dataset):
    # x_tensor: data
    # y_tensor: label
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

def DoTrain_bin(model:nn.Module, 
                train_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                inception=False,
                **kwargs):
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    tr_acc = np.zeros(num_epoch)
    tr_loss = np.zeros(num_epoch)
    correct = 0
    total = 0
    model.train()
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            if inception: out = out.logits
            pred = torch.squeeze(sigmoid(out))
            loss = criterion(pred, y.float())
            loss.backward()
            optimizer.step()

            predicted = (pred > 0.5).int()
            total += y.size(0)
            correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        tr_loss[epoch] = round(trn_loss/len(train_loader), 4)
        tr_acc[epoch] = round(100 * correct / total, 4)
    return tr_acc, tr_loss

def DoTest_bin(model:nn.Module, tst_loader:DataLoader):
    sigmoid = nn.Sigmoid()
    total = 0
    correct = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, y in tst_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            pred = sigmoid(torch.squeeze(pred.data))
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets