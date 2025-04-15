import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DEVICE, EarlyStopping, JukeboxLoss

OPT_DICT = {'Adam':opt.Adam,
            'AdamW':opt.AdamW,
            'SGD':opt.SGD}

def train_ae(model:nn.Module,
             train_loader:DataLoader,
             val_loader:DataLoader,
             num_epoch:int, 
             optimizer_name:str,
             learning_rate:str, 
             early_stop:EarlyStopping = None,
             min_epoch:int = 0,
             criterion_mode:int = 0,
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
        total_loss = 0
        for batch, mask in train_loader:
            batch = batch.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()
            recon, _ = model(batch)
            loss_recon = criterion(recon*mask, batch*mask)
            loss_g = loss_recon # + loss_spec(recon*mask, batch*mask) * w_sp
            loss_g.backward()
            optimizer.step()
            total_loss += loss_g.item()
            # tr_loss.append(loss_g.item())
        
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {total_loss/len(train_loader):.6f}")


        # if early_stop:
        #     with torch.no_grad():
        #         model.eval()
        #         for i, data in enumerate(val_loader, 0):
        #             x, _ = data
        #             x = x.to(DEVICE)
        #             recon, _ = model(x)
        #             loss_recon = criterion(recon.float(), x.float())
        #             loss_g = loss_recon + loss_spec(recon.float(), x.float()) * w_sp
        #             loss_g.backward()
        #             optimizer.step()
        #         val_loss = loss_g.item()
        #         vl_loss.append(val_loss)

        #         if epoch > min_epoch: 
        #             early_stop(val_loss, epoch)
        #         if early_stop.early_stop:
        #             early_stopped = True
        #             break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_loss, vl_loss

def train_aekl(model:nn.Module,
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
        total_loss = 0
        for batch, mask in train_loader:
            batch = batch.to(DEVICE)
            mask = mask.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, sigma = model(batch)
            loss_recon = criterion(recon * mask, batch * mask)
            loss_kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1])
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            loss_g = loss_recon + loss_kl * w_kl + loss_spec(recon * mask, batch * mask) * w_sp
            loss_g.backward()
            optimizer.step()
            total_loss += loss_g.item()
        # tr_loss.append(loss_g.item())
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {total_loss/len(train_loader):.6f}")

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

def train_aekl2(model:nn.Module,
                train_loader:DataLoader,
                num_epoch:int, 
                optimizer_name:str,
                learning_rate:str, 
                criterion_mode:int = 0,
                w_kl:float = 1e-4,
                w_sp:float = 1e-9):

    criterion = nn.L1Loss() if criterion_mode else nn.MSELoss()
    loss_spec = JukeboxLoss(1)
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    early_stopped = False
    tr_loss = []
    vl_loss = []
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            recon, mu, sigma = model(batch)
            loss_recon = criterion(recon, batch)
            loss_kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1])
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            loss_g = loss_recon + loss_kl * w_kl + loss_spec(recon, batch) * w_sp
            loss_g.backward()
            optimizer.step()
            total_loss += loss_g.item()
        # tr_loss.append(loss_g.item())
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {total_loss/len(train_loader):.6f}")
        
    torch.save(model.state_dict(), f'best_model.pth')
    return tr_loss, vl_loss