import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from transfer_trainer import *
from transfer_modules import Transfer_DataModule
from models.autoencoder_transformer import TransformerAutoencoder
from models.autoencoder_kl import AutoencoderKL
from torchmetrics.classification import BinaryConfusionMatrix
from utils import *

ManualSeed(222)

def train(chan_mode):
    fs = 125
    learning_rate = 1e-3
    num_batch = 16
    num_epochs = 1000
    min_epochs = 25
    num_chan = [8,3,3,2]
    time = datetime.datetime.now().strftime('%m%d_%H%M')
    
    data_module = Transfer_DataModule(stress=True,
                                    emotion=False,
                                    nback=True,
                                    d2=True,
                                    chan_mode=chan_mode,
                                    batch_size=num_batch)
    data_loader = data_module.dataloader

    # model = TransformerAutoencoder(input_dim=num_chan[chan_mode], 
    #                                embed_dim=100, 
    #                                patch_size=25, 
    #                                num_heads=4,
    #                                num_layers=3).to(DEVICE)
    
    model = AutoencoderKL(16, 2, 32, 4, 8).to(DEVICE)

    tr_loss = []
    vl_loss = []
    # train_loss, _ = train_ae(model, 
    #                         train_loader=data_loader, 
    #                         val_loader=None,
    #                         num_epoch=num_epochs, 
    #                         optimizer_name='Adam',
    #                         learning_rate=str(learning_rate),
    #                         early_stop=None,
    #                         min_epoch=min_epochs,
    #                         criterion_mode=0)
    train_aekl(model, data_loader, None, num_epochs, 'Adam', str(learning_rate), criterion_mode=0)

    # tr_loss.append(train_loss)


if __name__ == "__main__":
    # for i in range(3):
    #     train(i+1)
    train(3)
