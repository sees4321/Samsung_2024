import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from utils import *

class NBack_DataModule():
    r'''
    Create nback dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        chan_mode (int): 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right).
        fs (int): sampling frequency. (default: 125)
        epoch_len (int): window length in seconds for epoching. (default: 5)
        overlap_len (int): overlap length in seconds for epoching. (default: 0)
        rejection (bool): perform 100 uV rejection (default: True)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 path:str, # 'D:/One_한양대학교/private object minsu/coding/data/samsung_2024/emotion'
                 chan_mode:int,
                 fs:int = 125,
                 epoch_len:int = 5,
                 overlap_len:int = 0,
                 rejection:bool = True,
                 num_val:int = 3,
                 batch_size:int = 16,
                 transform = None,
                 ):
        super().__init__()

        chan_selection = [[0,8],[0,3],[3,6],[6,8]]
        nback = [0,2]
        epoch_len *= fs
        overlap_len *= fs
        self.batch_size = batch_size
        self.num_val = num_val
        self.test_idx = 0
        
        self.data = {}
        self.label = {}
        for i in range(len(nback)):
            path_folder = os.path.join(path,f"{nback[i]}_back")
            datalist = os.listdir(path_folder)
            data_subj = []
            iteration = 0
            subj = 0
            for path_dat in datalist:
                iteration += 1
                # data.append(np.load(os.path.join(path_folder, path_dat)))
                dat = np.load(os.path.join(path_folder, path_dat))[chan_selection[chan_mode][0]:chan_selection[chan_mode][1]]
                assert dat.shape[-1] >= epoch_len, 'there is data that is shorter than designated epoch length'

                for t in range(dat.shape[1] - epoch_len, 0, -(epoch_len-overlap_len)):
                    temp = dat[:,t:t+epoch_len]
                    if rejection:
                        if np.max(abs(temp)) <= 100:
                            data_subj.append(temp)
                    else:
                        data_subj.append(temp)

                if iteration == 3:
                    iteration = 0
                    if data_subj != []:
                        temp = np.stack(data_subj,0)
                        if transform:
                            temp = transform(temp)
                        if subj in self.data.keys():
                            self.data[subj] = np.concatenate([self.data[subj], temp])
                            self.label[subj] = np.concatenate([self.label[subj], np.ones((len(data_subj),)) * i])
                        else:
                            self.data[subj] = temp
                            self.label[subj] = np.ones((len(data_subj),)) * i
                        data_subj = []
                    subj += 1
        
        self.data_shape = list(self.data[0].shape[-2:])
        self.subjects = list(self.data.keys())
    
    def __len__(self):
        return len(self.subjects)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.test_idx < len(self.data):
            data_torch = torch.from_numpy(self.data[self.subjects[self.test_idx]]).float()
            label_torch = torch.from_numpy(self.label[self.subjects[self.test_idx]]).long()
            test_loader = DataLoader(CustomDataSet(data_torch, label_torch), self.batch_size, shuffle=False)

            # train_subjects, val_subjects = split_subjects(self.test_subj, len(self.data), self.num_val)
            train_subjects, val_subjects = self.train_val_split()
            data_torch = torch.from_numpy(np.concatenate([self.data[i] for i in train_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in train_subjects])).long()
            train_loader = DataLoader(CustomDataSet(data_torch, label_torch), self.batch_size, shuffle=False)

            data_torch = torch.from_numpy(np.concatenate([self.data[i] for i in val_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in val_subjects])).long()
            val_loader = DataLoader(CustomDataSet(data_torch, label_torch), self.batch_size, shuffle=True)

            self.test_idx += 1
            return train_loader, val_loader, test_loader
        else:
            raise StopIteration
        
    def train_val_split(self):
        subj = [i for i in self.subjects if i != self.subjects[self.test_idx]]
        random.shuffle(subj)
        return subj[self.num_val:], subj[:self.num_val]


if __name__ == "__main__":
    from utils import expand_dim_
    dataset = NBack_DataModule("D:\One_한양대학교\private object minsu\coding\data\samsung_2024\\nback_segmented_v3",
                               chan_mode=1,
                               fs=125,
                               epoch_len=5,
                               overlap_len=0,
                               rejection=True,
                               num_val=2,
                               batch_size=16,
                               transform=expand_dim_,)
    print(dataset.data_shape)
    for train_loader, val_loader, test_loader in dataset:
        print(len(train_loader), len(val_loader), len(test_loader))
