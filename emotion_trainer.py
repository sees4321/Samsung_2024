import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

from models.autoencoder_kl import AutoencoderKL
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DEVICE, EarlyStopping, JukeboxLoss

OPT_DICT = {'Adam':opt.Adam,
            'AdamW':opt.AdamW,
            'SGD':opt.SGD}

def train_bin_cls(model:nn.Module, 
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
            if pred.ndim == 0: pred = pred.unsqueeze(0)
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
                    if pred.ndim == 0: pred = pred.unsqueeze(0)
                    loss = criterion(pred, y.float())
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc =round(100 * vl_correct / vl_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    if early_stop.mode == 'min':
                        early_stop(val_loss, epoch)
                    else:
                        early_stop(val_acc, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def test_bin_cls(model:nn.Module, tst_loader:DataLoader):
    total = 0
    correct = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, y in tst_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = torch.squeeze(model(x))
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def train_cls(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                **kwargs):
    criterion = nn.CrossEntropyLoss()
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
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(pred, 1).int()
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
                    predicted = torch.argmax(pred, 1).int()
                    vl_total += y.size(0)
                    vl_correct += (predicted == y).sum().item()
                    loss = criterion(pred, y)
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc = round(100 * vl_correct / vl_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    if early_stop.mode == 'min':
                        early_stop(val_loss, epoch)
                    else:
                        early_stop(val_acc, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def test_cls(model:nn.Module, tst_loader:DataLoader):
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
            predicted = torch.argmax(pred, 1).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def train_ae(model:nn.Module,
             train_loader:DataLoader,
             val_loader:DataLoader,
             num_epoch:int, 
             optimizer_name:str,
             learning_rate:str, 
             early_stop:EarlyStopping = None,
             min_epoch:int = 0,
             criterion_mode:int = 0,
             w_kl:float = 1e-9,
             w_sp:float = 1e-4):

    criterion = nn.L1Loss() if criterion_mode else nn.MSELoss()
    loss_spec = JukeboxLoss(1)
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    early_stopped = False
    tr_loss = []
    vl_loss = []
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        model.train()
        for i, data in enumerate(train_loader, 0):
            x, _ = data
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, sigma = model(x)
            loss_recon = criterion(recon.float(), x.float())
            loss_kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1])
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            loss_g = loss_recon + loss_kl * w_kl + loss_spec(recon.float(),x.float()) * w_sp
            loss_g.backward()
            optimizer.step()
        tr_loss.append(loss_g.item())

        if early_stop:
            with torch.no_grad():
                model.eval()
                for i, data in enumerate(val_loader, 0):
                    x, _ = data
                    x = x.to(DEVICE)
                    recon, mu, sigma = model(x)
                    loss_recon = criterion(recon.float(), x.float())
                    loss_kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1])
                    loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
                    loss_g = loss_recon + loss_kl * w_kl + loss_spec(recon.float(),x.float()) * w_sp
                    loss_g.backward()
                    optimizer.step()
                val_loss = loss_g.item()
                vl_loss.append(val_loss)

                if epoch > min_epoch: 
                    early_stop(val_loss, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_loss, vl_loss


# Target distribution 정의
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# IDEC 클래스 정의
class ImprovedDeepEmbeddedClustering:
    def __init__(self, 
                 embed_dim:int,
                 in_chan:int,
                 hid_chan:int, 
                 z_chan:int,
                 n_clusters:int, 
                 optimizer_name:str,
                 learning_rate:str, 
                 criterion_mode:int = 0,
                 w_kl:float = 1e-9,
                 w_spectral:float = 1e-4,
                 ):

        self.ae = AutoencoderKL(embed_dim, in_chan, hid_chan, z_chan, 8).to(DEVICE)
        self.n_clusters = n_clusters
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, embed_dim))
        torch.nn.init.xavier_uniform_(self.cluster_layer)
        self.optimizer = OPT_DICT[optimizer_name](self.ae.parameters(), lr=float(learning_rate))
        self.criterion = nn.L1Loss() if criterion_mode else nn.MSELoss()
        self.loss_spec = JukeboxLoss(1)
        self.w_kl = w_kl
        self.w_sp = w_spectral
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

    def pretrain(self, data_loader:DataLoader, epochs=50):
        print("Pretraining...")
        for epoch in range(epochs):
            for batch in data_loader:
                x, _, _ = batch
                x = x.to(DEVICE)
                recon, mu, sigma = self.ae(x)
                # recon, mu = self.ae(x)
                
                loss_recon = self.criterion(recon.float(), x.float())
                loss_kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1])
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
                loss_g = loss_recon + loss_kl * self.w_kl + self.loss_spec(recon.float(),x.float()) * self.w_sp
                
                self.optimizer.zero_grad()
                loss_g.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_g.item():.4f}")

    def fit(self, data_loader:DataLoader, epochs=50):
        print("Training...")
        # Initialize cluster centers with KMeans
        data = []
        with torch.no_grad():
            for batch in data_loader:
                x, _, _ = batch
                x = x.to(DEVICE)
                mu, sigma = self.ae.encode(x)
                z = self.ae.sampling(mu, sigma)
                data.append(z)
        data = torch.cat(data).cpu().numpy()
        y_pred = self.kmeans.fit_predict(np.squeeze(data))
        self.cluster_layer.data = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float).to(DEVICE)

        # Train with clustering loss
        for epoch in range(epochs):
            for batch in data_loader:
                x, _, _ = batch
                x = x.to(DEVICE)
                recon, mu, sigma = self.ae(x)

                # clustering loss
                z = self.ae.sampling(mu, sigma)
                # Compute soft assignments
                q = 1.0 / (1.0 + torch.sum((z - self.cluster_layer) ** 2, dim=2))
                q = (q.T / q.sum(1)).T
                p = target_distribution(q.cpu().detach().numpy())
                p = torch.tensor(p, dtype=torch.float32).to(DEVICE)
                loss_cl = torch.sum(p * torch.log(p / q))
                # loss_cl = nn.functional.kl_div(q.log(), torch.tensor(p, dtype=torch.float).to(DEVICE), reduction='batchmean')
                
                # reconstruction loss
                loss_recon = self.criterion(recon.float(), x.float())
                # loss_kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1])
                # loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
                # loss_g = loss_recon + loss_kl * self.w_kl + self.loss_spec(recon.float(),x.float()) * self.w_sp

                loss = loss_cl + 0.1*loss_recon

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
