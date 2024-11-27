import datetime
import matplotlib.pyplot as plt
import numpy as np

from emotion_trainer import train_bin_cls, test_bin_cls 
from models.eegnet import EEGNet
from models.shallowfbcspnet import ShallowFBCSPNet
from models.hirenet import HiRENet, make_input
from models.MTCA_CapsNet import MTCA_CapsNet
from modules import Emotion_DataModule
from utils import *

seed = 2222
ManualSeed(seed)


def main(typ, chan, n_chan):
    learning_rate = 5e-4
    num_batch = 32
    num_epochs = 100
    num_subj = 32
    time = datetime.datetime.now().strftime('%m%d_%H%M')

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    preds = np.zeros((num_subj,60)) # model predictions
    targets = np.zeros((num_subj,60)) # labels
    for subj in range(num_subj):                                      
        emotion_dataset = Emotion_DataModule('D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion',
                                            label_mode=0, 
                                            label_type=typ, 
                                            test_subj=subj, 
                                            sample_half=True,
                                            channel_mode=chan,
                                            window_len=60,
                                            overlap_len=0,
                                            num_train=28,
                                            batch_size=num_batch,
                                            transform=make_input)
        test_loader = emotion_dataset.test_loader
        val_loader = emotion_dataset.val_loader
        train_loader = emotion_dataset.train_loader

        # model = ShallowFBCSPNet([3,125*60], 125).to(DEVICE)
        # model = EEGNet([n_chan,125*60], 125, 1).to(DEVICE)
        model = HiRENet(n_chan, 16).to(DEVICE)
        # model = MTCA_CapsNet(2, 7500).to(DEVICE)
        
        es = EarlyStopping(model, patience=5, mode='min')
        train_acc, train_loss, val_acc, val_loss = train_bin_cls(model, 
                                                                train_loader=train_loader, 
                                                                val_loader=val_loader,
                                                                num_epoch=num_epochs, 
                                                                optimizer_name='Adam',
                                                                learning_rate=str(learning_rate),
                                                                early_stop=es,
                                                                min_epoch=30)
        tr_acc.append(train_acc)
        tr_loss.append(train_loss)
        vl_acc.append(val_acc)
        vl_loss.append(val_loss)

        model.load_state_dict(torch.load('best_model.pth'))
        test_acc, _, _ = test_bin_cls(model, tst_loader=test_loader)
        ts_acc.append(test_acc)
        # ts_acc[subj], preds[subj], targets[subj] = DoTest_bin(model, tst_loader=test_loader)
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[-1]:.2f} %, training loss: {train_loss[-1]:.3f}')
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')
        # %%
        # plt.figure(figsize=(15,5))
        # plt.subplot(121)
        # plt.plot(train_acc)
        # plt.plot(val_acc)
        # plt.subplot(122)
        # plt.plot(train_loss)
        # plt.plot(val_loss)
        # plt.show()

    print(f'avg Acc: {np.mean(ts_acc)} %')
    print('end')

# main(0,3,2)

for typ in range(2):
    for chan in range(3):
        n_chan = 2 if chan == 2 else 3
        main(typ, chan+1, n_chan)
# SaveResults_mat(f'eegnet_{time}',ts_acc,preds,targets,tr_acc,tr_loss,num_batch,num_epochs,learning_rate)