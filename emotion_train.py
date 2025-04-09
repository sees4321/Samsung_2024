import datetime
import matplotlib.pyplot as plt
import numpy as np

from emotion_trainer import train_bin_cls, test_bin_cls, train_cls, test_cls
from models.tsception import TSception, channel_selection_
from models.eegnet import EEGNet
from models.deep4net import Deep4Net
from models.shallowfbcspnet import ShallowFBCSPNet
from models.syncnet2 import SyncNet2, SyncNet3
from models.hirenet import HiRENet, make_input, HiRENet_2, make_input_2
from models.MTCA_CapsNet import MTCA_CapsNet
from emotion_modules import Emotion_DataModule, Emotion_DataModule_temp
from torchmetrics.classification import BinaryConfusionMatrix
from utils import *

def main(typ, chan, n_chan):
    ManualSeed(0)
    # h = np.array([6, 4, 4, 4, 6, 6, 4, 3, 0, 0, 6, 8, 6, 4, 3, 4, 2, 2, 4, 4, 1, 2, 4, 3, 3, 2, 6, 1, 5, 3, 3, 2], int)
    # h = h > 1
    # num_subj = sum(h)
    num_subj = 32
    learning_rate = 5e-4
    num_batch = 16
    num_epochs = 30
    min_epochs = num_epochs
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
                                            channel_mode=chan,
                                            window_len=60,
                                            overlap_len=0,
                                            num_val=0,
                                            batch_size=num_batch,
                                            transform=None,
                                            subj_selection=None)
        test_loader = emotion_dataset.test_loader
        val_loader = emotion_dataset.val_loader
        train_loader = emotion_dataset.train_loader

        # model = TSception(6).to(DEVICE)
        # model = Deep4Net([n_chan, 125*15],1,'mean').to(DEVICE)
        # model = ShallowFBCSPNet([n_chan, 125*60], 125).to(DEVICE)
        # model = EEGNet([n_chan, 125*60], 125, 1).to(DEVICE)
        # model = HiRENet(n_chan, 16, 1).to(DEVICE)
        # model = MTCA_CapsNet(2, 7500).to(DEVICE)
        # model = HiRENet_2([n_chan, 1250], 32).to(DEVICE)
        model = SyncNet3(emotion_dataset.data_shape, 
                        data_mode=1,
                        num_segments=12,
                        embed_dim=256,
                        num_heads=4,
                        num_layers=2,
                        use_lstm=False,
                        num_groups=4,
                        actv_mode="elu",
                        pool_mode="mean", 
                        num_classes=1).to(DEVICE)
        
        # es = EarlyStopping(model, patience=10, mode='min')
        train_acc, train_loss, val_acc, val_loss = train_bin_cls(model, 
                                                                train_loader=train_loader, 
                                                                val_loader=val_loader,
                                                                num_epoch=num_epochs, 
                                                                optimizer_name='Adam',
                                                                learning_rate=str(learning_rate),
                                                                early_stop=None,
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
        print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[-1]:.2f} %, training loss: {train_loss[-1]:.3f}')
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    # print(f'avg Acc: {np.mean(ts_acc)} %')
    print(f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}, sen: {np.mean(ts_sen)*100:.2f}, spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')

#type chan n_chan
#a0 v1 / 0full, 123 / 8 3 3 2
for v in [[1,3],[2,3],[3,2]]:
    main(0,v[0],v[1])
    main(1,v[0],v[1])
    print()

# main(0,0,8)
# main(1,0,8)

# main(0,1,3)
# main(1,1,3)

# for typ in range(2):
#     for chan in range(3):
#         n_chan = 2 if chan == 2 else 3
#         main(typ, chan+1, n_chan)
# SaveResults_mat(f'eegnet_{time}',ts_acc,preds,targets,tr_acc,tr_loss,num_batch,num_epochs,learning_rate)