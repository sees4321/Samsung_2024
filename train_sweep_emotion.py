import numpy as np
import datetime

from models.eegnet import EEGNet
from models.hirenet import *
from modules import Emotion_DataModule
from utils import *

seed = 2222
ManualSeed(seed)

def evaluation(h1,h2,h3):
    learning_rate = h1
    num_batch = h2
    num_epochs = h3
    num_subj = 32

    tr_acc = np.zeros((num_subj,num_epochs))
    ts_acc = np.zeros(num_subj)

    for subj in range(num_subj):                                      
        emotion_dataset = Emotion_DataModule('D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion',
                                            label_mode=0, 
                                            test_subj=subj, 
                                            sample_half=True,
                                            channel_mode=2,
                                            window_len=60,
                                            overlap_len=0,
                                            num_train=28,
                                            batch_size=num_batch,
                                            transform=make_input)
        test_loader = emotion_dataset.test_loader
        val_loader = emotion_dataset.val_loader
        train_loader = emotion_dataset.train_loader

        # model = EEGNet([2,7500], 125, 1).to(DEVICE)
        model = HiRENet(3,32).to(DEVICE)

        tr_acc[subj], _ = DoTrain_bin(model, 
                                                train_loader=train_loader, 
                                                num_epoch=num_epochs, 
                                                optimizer_name='Adam',
                                                learning_rate=str(learning_rate))
        ts_acc[subj], _, _ = DoTest_bin(model, tst_loader=test_loader)
    return np.mean(ts_acc)


h1 = [1e-4, 1e-3, 5e-4]
h2 = [16, 32, 64]
h3 = [50, 100]

result = []
for i in h1:
    result.append(evaluation(i,16,50))
h1 = h1[np.argmax(result)]
result = []
for i in h2:
    result.append(evaluation(h1,i,50))
h2 = h2[np.argmax(result)]
result = []
for i in h3:
    result.append(evaluation(h1,h2,50))
h3 = h3[np.argmax(result)]

print(h1,h2,h3)
print(result)