import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from utils import *

# 가변 길이 데이터를 포함하는 Dataset 구현
class EEGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # 리스트 형태로 [Tensor(T1), Tensor(T2), ...]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]  # 개별 데이터 반환

# 배치 내에서 가장 긴 데이터에 맞춰 패딩 적용
def collate_fn(batch):
    batch_sizes = [x.shape[-1] for x in batch]  # 각 샘플의 길이 저장
    max_len = max(batch_sizes)  # 가장 긴 데이터 길이에 맞춤
    t = 4
    if max_len % t != 0:
        max_len = max_len // t * t + t
    
    padded_batch = []
    masks = []
    for x in batch:
        pad_len = max_len - x.shape[-1]
        if pad_len > 0:
            padded_x = F.pad(x, (0, pad_len), "constant", 0)  # 0으로 패딩
            mask = torch.cat([torch.ones_like(x), torch.zeros((x.shape[-2], pad_len))], dim=-1)  # 마스크 생성
        else:
            padded_x = x
            mask = torch.ones_like(x)
        
        padded_batch.append(padded_x)
        masks.append(mask)
    
    return torch.stack(padded_batch), torch.stack(masks)  # (batch, channels, max_len), (batch, channels, max_len)

class Transfer_DataModule():
    r'''
    Create nback dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        chan_mode (int): 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right).
        fs (int): sampling frequency. (default: 125)
        window_len (int): window length in seconds for epoching. (default: 5)
        overlap_len (int): overlap length in seconds for epoching. (default: 0)
        rejection (bool): perform 100 uV rejection (default: True)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 stress:bool,
                 emotion:bool,
                 nback:bool,
                 d2:bool,
                 chan_mode:int,
                 batch_size:int = 16,
                 ):
        super().__init__()

        chan_selection = [[0,8],[0,3],[3,6],[6,8]]
        self.batch_size = batch_size
        self.data = []

        # stress
        if stress:
            path = 'D:\One_한양대학교\private object minsu\coding\data\samsung_2024\Preprocessed_Stress'
            for i in ['Relax', 'Stress']:
                path_folder = os.path.join(path, i)
                datalist = os.listdir(path_folder)
                for path_dat in datalist:
                    dat = torch.from_numpy(np.load(os.path.join(path_folder, path_dat))[chan_selection[chan_mode][0]:chan_selection[chan_mode][1]]).float()
                    self.data.append(dat)

        # emotion
        if emotion:
            path = 'D:\One_한양대학교\private object minsu\coding\data\samsung_2024\emotion'
            dat = np.load(path + '/emotion_data.npy')
            dat = np.concatenate(dat)
            for i in [0, 7500]:
                self.data += list(torch.from_numpy(dat[:,chan_selection[chan_mode][0]:chan_selection[chan_mode][1],i:i+7500]).float())

        # nback
        if nback:
            path = 'D:\One_한양대학교\private object minsu\coding\data\samsung_2024\\nback_segmented_v3'
            for i in [0,2,3]:
                path_folder = os.path.join(path, f"{i}_back")
                datalist = os.listdir(path_folder)
                for path_dat in datalist:
                    dat = torch.from_numpy(np.load(os.path.join(path_folder, path_dat))[chan_selection[chan_mode][0]:chan_selection[chan_mode][1]]).float()
                    self.data.append(dat)
        
        # d2
        if d2:
            path = 'D:\One_한양대학교\private object minsu\coding\data\samsung_2024\d2'
            for i in range(32):
                path_folder = os.path.join(path, f'S{i}')
                datalist = os.listdir(path_folder)
                for path_dat in datalist:
                    if path_dat[0] == 'c': 
                        continue
                    dat = torch.from_numpy(np.load(os.path.join(path_folder, path_dat))[chan_selection[chan_mode][0]:chan_selection[chan_mode][1]]).float()
                    self.data.append(dat)
        
        self.data = EEGDataset(self.data)
        self.dataloader = DataLoader(self.data, batch_size=batch_size, collate_fn=collate_fn)


class Transfer_DataModule_2():
    r'''
    Create nback dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        stress (bool): load stress data (bool)
        emotion (bool): load emotion data (bool)
        d2 (bool): load d2 data (bool)
        chan_mode (int): 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right).
        fs (int): sampling frequency. (default: 125)
        window_len (int): window length in seconds for epoching. (default: 5)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 path:str,# ='D:\One_한양대학교\private object minsu\coding\data\samsung_2024'
                 stress:bool,
                 emotion:bool,
                 d2:bool,
                 chan_mode:int,
                 fs:int=125,
                 window_len:int = 5,
                 batch_size:int = 16,
                 ):
        super().__init__()

        fs = 125
        chan_selection = [[0,8],[0,3],[3,6],[6,8]]
        self.window_len = fs * window_len
        self.chan = chan_selection[chan_mode]
        self.num_chan = self.chan[1] - self.chan[0]
        self.batch_size = batch_size
        self.data = np.empty((0, self.num_chan, self.window_len))

        # stress
        if stress:
            dat = np.load(path+'/mist/mist.npz')
            self.data = np.vstack((self.data, self.window_slicing(np.concatenate(dat['eeg'])[:,self.chan[0]:self.chan[1]]))) # (31, 2, 8, 7500)
            self.data = np.vstack((self.data, self.window_slicing(dat['eeg_med'])[:,self.chan[0]:self.chan[1]])) # (31, 8, 41250)
            self.data = np.vstack((self.data, self.window_slicing(np.concatenate(dat['eeg_task'])[:,self.chan[0]:self.chan[1]]))) # (31, 2, 8, 18750)
            self.data = np.vstack((self.data, self.window_slicing(dat['eeg_resting'])[:,self.chan[0]:self.chan[1]])) # (31, 8, 26250)

        # emotion
        if emotion:
            dat = np.load(path+'/emotion/emotion.npz')
            self.data = np.vstack((self.data, self.window_slicing(np.concatenate(dat['eeg'])[:,self.chan[0]:self.chan[1]]))) # (31, 8, 8, 15000)
            self.data = np.vstack((self.data, self.window_slicing(np.concatenate(dat['eeg_washoff'])[:,self.chan[0]:self.chan[1]]))) # (31, 8, 8, 3750)
            self.data = np.vstack((self.data, self.window_slicing(np.concatenate(dat['eeg_resting'])[:,self.chan[0]:self.chan[1]]))) # (31, 2, 8, 7500)

        
        # d2
        if d2:
            dat = np.load(f'{path}/d2_2/d2_{window_len}.npy')
            self.data = np.vstack((self.data, dat[:,self.chan[0]:self.chan[1]]))
        
        self.data = torch.from_numpy(self.data).float()
        self.dataloader = DataLoader(EEGDataset(self.data), batch_size=batch_size, shuffle=False)
    
    def window_slicing(self, arr):
        shape = arr.shape
        n_windows = int(shape[-1] // self.window_len)

        out = np.stack([arr[:,:,i*self.window_len:(i+1)*self.window_len] for i in range(n_windows)], 1)
        return np.concatenate(out)

if __name__ == "__main__":
    # dataset = Transfer_DataModule(stress=True,
    #                               emotion=True,
    #                               nback=True,
    #                               d2=True,
    #                               chan_mode=1,
    #                               batch_size=16)
    dataset = Transfer_DataModule_2('D:\One_한양대학교\private object minsu\coding\data\samsung_2024',
                                    stress=True,
                                    emotion=True,
                                    d2=True,
                                    chan_mode=1,
                                    window_len=5,
                                    batch_size=16)
    print(dataset.data.shape)