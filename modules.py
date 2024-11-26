import os
import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from utils import *

class Emotion_DataModule():
    r'''
    Create emotion dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        label_mode (int): 0 - using emotion self rating, 1 - using target emotion.
        label_type (int): 0 - valence binary classification, 1 - valence binray classification.
        test_subj (int): subject number for test. 
        sample_half (bool): True - use 60 sec length time samples from the end, False - use full 120 sec length time samples. (default: True)
        channel_mode (int): 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right). (default: 0)
        window_len (int): window length in seconds for epoching. (default: 60)
        overlap_len (int): overlap length in seconds for epoching. (default: 0)
        num_train (int): number of subjects for training. (default: 28)
    '''
    def __init__(self, 
                 path:str, # 'D:/One_한양대학교/private object minsu/coding/data/samsung_2024/emotion'
                 label_mode:int,
                 label_type:int,
                 test_subj:int,
                 sample_half:bool = True,
                 channel_mode:int = 0,
                 window_len:int = 60,
                 overlap_len:int = 0,
                 num_train:int = 28,
                 batch_size:int = 16,
                 transform = None
                 ):
        super().__init__()

        # load original data + label
        self.data = np.load(os.path.join(path, 'emotion_data.npy'))/1000 # (32, 9, 8, 15000) 
        if label_mode == 0:
            self.label = np.load(os.path.join(path, 'emotion_label.npy')) # (32, 9, 2)
            self.label = np.array(self.label[:, 1:, label_type] > 2, int) # (32, 8)
        else:
            self.label = np.load(os.path.join(path, 'emotion_label2.npy'))[:, 1:] # (32, 9)
            if label_type == 0: # arousal
                self.label[self.label == 1.0] = 1
                self.label[self.label == 2.0] = 0
                self.label[self.label == 3.0] = 1
                self.label[self.label == 4.0] = 0
            else: # valence
                self.label[self.label == 1.0] = 1
                self.label[self.label == 2.0] = 1
                self.label[self.label == 3.0] = 0
                self.label[self.label == 4.0] = 0


        # channel selection & sampling
        fs = 125
        channel_selection = [[0,8], [0,3], [3,6], [6,8]]
        self.data = self.data[:, 1:, 
                              channel_selection[channel_mode][0]:channel_selection[channel_mode][1],
                              self.data.shape[3]//2 if sample_half else 0:] # (32, 8, 8, samples)

        # epoching
        if window_len < 60:
            new_dat = []
            for i in range(0, self.data.shape[3] - window_len*fs, (window_len - overlap_len)*fs):
                new_dat.append(self.data[:,:,:,i:i+window_len*fs])
            self.data = np.stack(new_dat, 1)
            self.data = np.swapaxes(self.data, 0, 2)
            self.data = np.concatenate(self.data)
            self.data = np.swapaxes(self.data, 0, 1)
            self.label = np.repeat(self.label, len(new_dat), 1)
        
        if transform:
            self.data = transform(self.data)

        # train-val-test split
        data_torch = torch.from_numpy(self.data[test_subj]).float()
        label_torch = torch.from_numpy(self.label[test_subj]).long()
        self.test_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=False)

        train_subjects, val_subjects = split_subjects(test_subj, 32, num_train)

        data_torch = torch.from_numpy(np.concatenate(self.data[train_subjects])).float()
        label_torch = torch.from_numpy(np.concatenate(self.label[train_subjects])).long()
        self.train_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=False)

        data_torch = torch.from_numpy(np.concatenate(self.data[val_subjects])).float()
        label_torch = torch.from_numpy(np.concatenate(self.label[val_subjects])).long()
        self.val_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=True)
        

def __main__():
    emotion_dataset = Emotion_DataModule('D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion',
                                        label_mode=0, 
                                        label_type=0, 
                                        test_subj=0, 
                                        channel_mode=1, 
                                        window_len=2, 
                                        overlap_len=1,
                                        batch_size=16)

    test_loader = emotion_dataset.test_loader
    val_loader = emotion_dataset.val_loader
    train_loader = emotion_dataset.train_loader
