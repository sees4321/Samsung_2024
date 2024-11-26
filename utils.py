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

class EarlyStopping:
    def __init__(self, model, patience=3, delta=0.0, mode='min', verbose=False):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        self.model = model
        self.epoch = 0

    def __call__(self, score, epoch):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                # 모델 저장
                torch.save(self.model.state_dict(), f'best_model.pth')
                self.epoch = epoch
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                # 모델 저장
                torch.save(self.model.state_dict(), f'best_model.pth')
                self.epoch = epoch
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False

def DoTrain_bin(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                **kwargs):
    criterion = nn.BCELoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    tr_acc, tr_loss = [], []
    vl_acc, vl_loss = [], []
    tr_correct, tr_total = 0, 0
    vl_correct, vl_total = 0, 0
    early_stopped = False
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        model.train()
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = torch.squeeze(model(x))
            loss = criterion(pred, y.float())
            loss.backward()
            optimizer.step()

            predicted = (pred > 0.5).int()
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))

        if early_stop:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                # if epoch % 10 == 0:
                for i, data in enumerate(val_loader, 0):
                    x, y = data
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    pred = torch.squeeze(model(x))
                    predicted = (pred > 0.5).int()
                    vl_total += y.size(0)
                    vl_correct += (predicted == y).sum().item()
                    loss = criterion(pred, y.float())
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc =round(100 * vl_correct / vl_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    early_stop(val_loss, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def DoTest_bin(model:nn.Module, tst_loader:DataLoader):
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
            pred = torch.squeeze(pred)
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

