import numpy as np
import datetime

from models.eegnet import EEGNet
from models.hirenet import *
from modules import Emotion_DataModule
from utils import *

seed = 2222
ManualSeed(seed)


learning_rate = 1e-4
num_batch = 32
num_epochs = 50
num_subj = 32
time = datetime.datetime.now().strftime('%m%d_%H%M')

tr_acc = np.zeros((num_subj,num_epochs))
tr_loss = np.zeros((num_subj,num_epochs))
ts_acc = np.zeros(num_subj)
preds = np.zeros((num_subj,60)) # model predictions
targets = np.zeros((num_subj,60)) # labels
for subj in range(num_subj):                                      
    emotion_dataset = Emotion_DataModule('D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion',
                                         label_mode=0, 
                                         test_subj=subj, 
                                         sample_half=True,
                                         channel_mode=3,
                                         window_len=60,
                                         overlap_len=0,
                                         num_train=28,
                                         batch_size=num_batch,
                                         transform=make_input)
    test_loader = emotion_dataset.test_loader
    val_loader = emotion_dataset.val_loader
    train_loader = emotion_dataset.train_loader

    # model = EEGNet([2,7500], 125, 1).to(DEVICE)
    model = HiRENet(2,32).to(DEVICE)


    tr_acc[subj], tr_loss[subj] = DoTrain_bin(model, 
                                              train_loader=train_loader, 
                                              num_epoch=num_epochs, 
                                              optimizer_name='Adam',
                                              learning_rate=str(learning_rate))
    # ts_acc[subj], preds[subj], targets[subj] = DoTest_bin(model, tst_loader=test_loader)
    ts_acc[subj], _, _ = DoTest_bin(model, tst_loader=test_loader)
    print(f'[{subj:0>2}] Acc: {ts_acc[subj]} %, training Acc: {tr_acc[subj,-1]} %')
print(f'avg Acc: {np.mean(ts_acc)} %')
print('end')
# SaveResults_mat(f'eegnet_{time}',ts_acc,preds,targets,tr_acc,tr_loss,num_batch,num_epochs,learning_rate)