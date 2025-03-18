import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from transfer_trainer import *
from emotion_modules import Emotion_DataModule
from emotion_trainer import train_bin_cls, test_bin_cls
from models.autoencoder_transformer import TransformerAutoencoder, AutoencoderClassifier
from models.autoencoder_kl import AutoencoderKL, AutoencoderClassifierKL
from torchmetrics.classification import BinaryConfusionMatrix
from utils import *

seed = 2222
ManualSeed(seed)

def train(typ, chan_mode):
    # h = np.array([6, 4, 4, 4, 6, 6, 4, 3, 0, 0, 6, 8, 6, 4, 3, 4, 2, 2, 4, 4, 1, 2, 4, 3, 3, 2, 6, 1, 5, 3, 3, 2], int)
    # h = h > 1
    # num_subj = sum(h)
    num_subj = 32
    learning_rate = 1e-3
    num_batch = 32
    num_epochs = 50
    min_epochs = 15
    num_chan = [8,3,3,2]
    time = datetime.datetime.now().strftime('%m%d_%H%M')

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    preds = np.zeros((num_subj,60)) # model predictions
    targets = np.zeros((num_subj,60)) # labels
    
    for subj in range(num_subj):
        emotion_dataset = Emotion_DataModule('D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion',
                                            label_mode=1, 
                                            label_type=typ, 
                                            test_subj=subj, 
                                            sample_half=True,
                                            channel_mode=chan_mode,
                                            window_len=60,
                                            overlap_len=0,
                                            num_val=2,
                                            batch_size=num_batch,
                                            transform=None,
                                            subj_selection=None)
        test_loader = emotion_dataset.test_loader
        val_loader = emotion_dataset.val_loader
        train_loader = emotion_dataset.train_loader

        # model_ae = TransformerAutoencoder(input_dim=num_chan[chan_mode], 
        #                                 embed_dim=500, 
        #                                 patch_size=25, 
        #                                 num_heads=4,
        #                                 num_layers=3).to(DEVICE)
        model_ae = AutoencoderKL(16, 2, 32, 4, 8).to(DEVICE)
        model_ae.load_state_dict(torch.load('aekl500.pth'))
        # model = AutoencoderClassifier(model_ae, 7500, 500, 25, 1).to(DEVICE)
        model = AutoencoderClassifierKL(model_ae, 7500, 16, 4, 1).to(DEVICE)
        for name, param in model.named_parameters():
            if name[:2] == 'ae': param.requires_grad = False

        es = EarlyStopping(model, patience=10, mode='min')
        train_acc, train_loss, val_acc, val_loss = train_bin_cls(model, 
                                                                train_loader=train_loader, 
                                                                val_loader=val_loader,
                                                                num_epoch=num_epochs, 
                                                                optimizer_name='Adam',
                                                                learning_rate=str(learning_rate),
                                                                early_stop=es,
                                                                min_epoch=min_epochs)
        tr_acc.append(train_acc)
        tr_loss.append(train_loss)
        vl_acc.append(val_acc)
        vl_loss.append(val_loss)

        model.load_state_dict(torch.load('best_model.pth'))
        test_acc, preds, targets = test_bin_cls(model, tst_loader=test_loader)
        ts_acc.append(test_acc)
        bcm = BinaryConfusionMatrix()
        cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
        # cf = bcm(torch.from_numpy(np.argmax(preds,1)), torch.from_numpy(targets))
        ts_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
        ts_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))
        # ts_acc.append(val_acc[-1])
        # ts_acc[subj], preds[subj], targets[subj] = DoTest_bin(model, tst_loader=test_loader)
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[-1]:.2f} %, training loss: {train_loss[-1]:.3f}')
        print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    # print(f'avg Acc: {np.mean(ts_acc)} %')
    print(f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}, sen: {np.mean(ts_sen)*100:.2f}, spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')


if __name__ == "__main__":
    # for i in range(3):
    #     train(i+1)
    train(0,3)
    train(1,3)
