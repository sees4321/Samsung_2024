import datetime
import matplotlib.pyplot as plt
import numpy as np

from emotion_trainer import train_bin_cls, test_bin_cls, train_cls, test_cls
from models.eegnet import EEGNet
from models.shallowfbcspnet import ShallowFBCSPNet
from models.hirenet import HiRENet, make_input
from models.deep4net import Deep4Net
from models.MTCA_CapsNet import MTCA_CapsNet
from nback_modules import NBack_DataModule
from torchmetrics.classification import BinaryConfusionMatrix
from utils import *

seed = 2222
ManualSeed(seed)

def train(chan_mode):
    fs = 125
    window_len = 6
    overlap_len = 0
    learning_rate = 7e-4 #7e4
    num_batch = 32
    num_epochs = 100
    min_epochs = 25
    time = datetime.datetime.now().strftime('%m%d_%H%M')

    dataset = NBack_DataModule("D:\One_한양대학교\private object minsu\coding\data\samsung_2024\\nback_segmented_v3",
                               chan_mode=chan_mode,
                               fs=fs,
                               window_len=window_len,
                               overlap_len=overlap_len,
                               rejection=True,
                               num_val=3,
                               batch_size=num_batch,
                               transform=None
                               )

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    preds = np.zeros((len(dataset),60)) # model predictions
    targets = np.zeros((len(dataset),60)) # labels
    
    for subj, data_loaders in enumerate(dataset):
        train_loader, val_loader, test_loader = data_loaders

        # model = Deep4Net(dataset.data_shape, 1).to(DEVICE)
        # model = ShallowFBCSPNet(dataset.data_shape, 125, 1, 'mean').to(DEVICE)
        # model = EEGNet(dataset.data_shape, fs, 1).to(DEVICE)
        # model = HiRENet(n_chan, 16, 1).to(DEVICE)
        model = MTCA_CapsNet(dataset.data_shape[0], dataset.data_shape[1], 8, 9, 1).to(DEVICE)
        
        es = EarlyStopping(model, patience=20, mode='min')
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
        print(f'[{subj:0>2}] acc: {test_acc:.2f} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    print(f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}, sen: {np.mean(ts_sen)*100:.2f}, spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')
    # SaveResults_mat(f'eegnet_{time}',ts_acc,preds,targets,tr_acc,tr_loss,num_batch,num_epochs,learning_rate)


if __name__ == "__main__":
    for i in range(3):
        train(i+1)