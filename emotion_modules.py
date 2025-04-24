import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from utils import *

class Emotion_DataModule():
    r'''
    Create emotion dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        label_mode (int): 0 - using emotion self rating, 1 - using target emotion.
        label_type (int): 0 - arousal binary classification, 1 - valence binray classification.
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
                 num_val:int = 2,
                 batch_size:int = 16,
                 transform = None,
                 subj_selection = None
                 ):
        super().__init__()

        self.data_npz = np.load(os.path.join(path, 'emotion.npz'))
        # load original data + label
        # self.data = np.load(os.path.join(path, 'emotion_data.npy'))/1000 # (32, 9, 8, 15000)
        self.data = self.data_npz['eeg'] 
        if label_mode == 0:
            # self.label = np.load(os.path.join(path, 'emotion_label.npy')) # (32, 9, 2)
            self.label = self.data_npz['label_rating']
            self.label = np.array(self.label[:, 1:, label_type] > 2, int) # (32, 8)
        elif label_mode == 2: # 3 class
            # self.label = np.load(os.path.join(path, 'emotion_label.npy')) # (32, 9, 2)
            self.label = self.data_npz['label_rating']
            self.label = np.array(self.label[:, 1:, label_type] > 2, int) + np.array(self.label[:, 1:, label_type] > 3, int)
        else:
            # self.label = np.load(os.path.join(path, 'emotion_label2.npy'))[:, 1:] # (32, 9)
            self.label = self.data_npz['label_target'][:,1:]
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
        
        if subj_selection is not None:
            self.data = self.data[subj_selection]
            self.label = self.label[subj_selection]

        # channel selection & sampling
        fs = 125
        channel_selection = [[0,8], [0,3], [3,6], [6,8]]
        self.data = self.data[:, :, 
                              channel_selection[channel_mode][0]:channel_selection[channel_mode][1],
                              self.data.shape[3]//2 if sample_half else 0:] # (32, 8, 8, samples)

        # epoching
        if window_len < 60:
            new_dat = []
            for i in range(0, self.data.shape[3] - window_len*fs + 1, window_len*fs):
                new_dat.append(self.data[:,:,:,i:i+window_len*fs])
            self.data = np.stack(new_dat, 1)
            self.data = np.swapaxes(self.data, 0, 2)
            self.data = np.concatenate(self.data)
            self.data = np.swapaxes(self.data, 0, 1)
            self.label = np.repeat(self.label, len(new_dat), 1)
        
        if transform:
            self.data = transform(self.data)
        self.data_shape = [self.data.shape[-2], self.data.shape[-1]]

        # train-val-test split
        data_torch = torch.from_numpy(self.data[test_subj]).float()
        label_torch = torch.from_numpy(self.label[test_subj]).long()
        self.test_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=False)

        train_subjects, val_subjects = split_subjects(test_subj, self.data.shape[0], num_val)

        data_torch = torch.from_numpy(np.concatenate(self.data[train_subjects])).float()
        label_torch = torch.from_numpy(np.concatenate(self.label[train_subjects])).long()
        self.train_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=False)
        
        if sum(val_subjects) > 0:
            data_torch = torch.from_numpy(np.concatenate(self.data[val_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate(self.label[val_subjects])).long()
            self.val_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=True)
        else:
            self.val_loader = None
        
class Emotion_DataModule_temp():
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

        self.label = np.load(os.path.join(path, 'emotion_label.npy'))[:,1:,label_type] # (32, 8)


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
        dat_num = self.data[test_subj]
        lab_num = self.label[test_subj]
        dat_num = dat_num[lab_num != 3]
        lab_num = lab_num[lab_num != 3]
        lab_num = np.array(lab_num > 3, int)
        data_torch = torch.from_numpy(dat_num).float()
        label_torch = torch.from_numpy(lab_num).long()
        self.test_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=False)

        train_subjects, val_subjects = split_subjects(test_subj, 32, num_train)

        dat_num = self.data[train_subjects]
        lab_num = self.label[train_subjects]
        dat_num = dat_num[lab_num != 3]
        lab_num = lab_num[lab_num != 3]
        lab_num = np.array(lab_num > 3, int)
        data_torch = torch.from_numpy(dat_num).float()
        label_torch = torch.from_numpy(lab_num).long()
        self.train_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=False)

        dat_num = self.data[val_subjects]
        lab_num = self.label[val_subjects]
        dat_num = dat_num[lab_num != 3]
        lab_num = lab_num[lab_num != 3]
        lab_num = np.array(lab_num > 3, int)
        data_torch = torch.from_numpy(dat_num).float()
        label_torch = torch.from_numpy(lab_num).long()
        self.val_loader = DataLoader(CustomDataSet(data_torch, label_torch), batch_size, shuffle=True)

class Emotion_DataModule_Unsupervised():
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
                 # label_mode:int,
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
        # if label_mode == 0:
        lab = np.load(os.path.join(path, 'emotion_label.npy'))[:, 1:] # (32, 8, 2)
        self.label = np.zeros((32,8), int)
        self.label[(lab[:, :, 0] < 3) & (lab[:, :, 1] < 3)] = 0
        self.label[(lab[:, :, 0] < 3) & (lab[:, :, 1] >= 3)] = 1
        self.label[(lab[:, :, 0] >= 3) & (lab[:, :, 1] < 3)] = 2
        self.label[(lab[:, :, 0] >= 3) & (lab[:, :, 1] >= 3)] = 3
        # else:
        self.label2 = np.load(os.path.join(path, 'emotion_label2.npy'))[:, 1:] # (32, 8)
        self.label2 -= 1


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
            self.label2 = np.repeat(self.label2, len(new_dat), 1)
        
        if transform:
            self.data = transform(self.data)

        # train-val-test split
        data_torch = torch.from_numpy(self.data[test_subj]).float()
        label_torch = torch.from_numpy(self.label[test_subj]).long()
        label_torch2 = torch.from_numpy(self.label2[test_subj]).long()
        self.test_loader = DataLoader(CustomDataSet2(data_torch, label_torch, label_torch2), batch_size, shuffle=False)

        train_subjects, val_subjects = split_subjects(test_subj, 32, num_train)

        data_torch = torch.from_numpy(np.concatenate(self.data[train_subjects])).float()
        label_torch = torch.from_numpy(np.concatenate(self.label[train_subjects])).long()
        label_torch2 = torch.from_numpy(np.concatenate(self.label2[train_subjects])).long()
        self.train_loader = DataLoader(CustomDataSet2(data_torch, label_torch, label_torch2), batch_size, shuffle=False)

        data_torch = torch.from_numpy(np.concatenate(self.data[val_subjects])).float()
        label_torch = torch.from_numpy(np.concatenate(self.label[val_subjects])).long()
        label_torch2 = torch.from_numpy(np.concatenate(self.label2[val_subjects])).long()
        self.val_loader = DataLoader(CustomDataSet2(data_torch, label_torch, label_torch2), batch_size, shuffle=True)

if __name__ ==  "__main__":
    emotion_dataset = Emotion_DataModule('D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion',
                                        label_mode=0, 
                                        label_type=0, 
                                        test_subj=0, 
                                        channel_mode=1, 
                                        window_len=10, 
                                        batch_size=16)

    test_loader = emotion_dataset.test_loader
    val_loader = emotion_dataset.val_loader
    train_loader = emotion_dataset.train_loader
