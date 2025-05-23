import torch
import torch.nn as nn
import numpy as np

from scipy.signal import hilbert

class resBlock(nn.Module):
    def __init__(self, inn, hid, k):
        super(resBlock,self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(inn,inn,1),
            nn.Conv2d(inn,inn,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(inn),
            nn.ELU(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(inn,hid,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(hid),
            nn.ELU(),
            nn.Conv2d(hid,inn,1),
        )
        self.relu = nn.ELU()

    def forward(self,x):
        out = self.conv(x)
        out = self.layer(out) + out
        out = self.relu(out)
        return out
    
class HiRENet(nn.Module):
    def __init__(self, num_chan, conv_chan, num_classes=1, withhil = True):
        super(HiRENet, self).__init__()

        self.withhil = withhil
        self.num_chan = num_chan
        self.conv_chan = conv_chan
        self.num_cls = num_classes

        self.layerx = resBlock(self.num_chan, self.conv_chan, 13)
        if self.withhil:
            self.layery = resBlock(self.num_chan, self.conv_chan, 13)
            self.num_chan = self.num_chan*2

        self.layer4 = nn.Sequential( 
                nn.Conv2d(self.num_chan, self.conv_chan, kernel_size=(1, 13), padding=(0, 13//2)),
                nn.BatchNorm2d(self.conv_chan, momentum=.1, affine=True),
                nn.ELU(),
                nn.Conv2d(self.conv_chan, self.conv_chan*2, kernel_size=(30, 1)),
                nn.BatchNorm2d(self.conv_chan*2, momentum=.1, affine=True),
                nn.ELU(),
            )    
        self.avgdrp = nn.Sequential(
            nn.AvgPool2d((1,13),(1,2)),
            nn.Dropout2d(0.5)
        )
        
        self.fc_module = nn.Sequential(
            nn.Conv2d(self.conv_chan*2,self.num_cls,(1,119)),
            nn.Sigmoid() if self.num_cls == 1 else nn.LogSoftmax(),
        )

    def forward(self, x):
        out = self.layerx(x[:,:,:30,:])
        if self.withhil:
            outy = self.layery(x[:,:,30:,:])
            out = torch.cat((out,outy),dim=1)
        out = self.layer4(out)
        out = self.avgdrp(out)
        out = self.fc_module(out)
        return torch.squeeze(out)

class resBlock_2(nn.Module):
    def __init__(self, inn, hid, k):
        super(resBlock_2,self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(inn,inn,1),
            nn.Conv2d(1,inn,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(inn),
            nn.ELU(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(inn,hid,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(hid),
            nn.ELU(),
            nn.Conv2d(hid,inn,1),
        )
        self.relu = nn.ELU()

    def forward(self,x):
        out = self.conv(x)
        out = self.layer(out) + out
        out = self.relu(out)
        return out

class HiRENet_2(nn.Module):
    def __init__(self, input_size, hid, num_classes=1, withhil = True):
        super(HiRENet_2, self).__init__()

        self.num_chan = input_size[0]
        self.sample_len = input_size[1]
        self.withhil = withhil
        self.num_cls = num_classes

        self.layerx = resBlock_2(hid, hid*2, 13)
        if self.withhil:
            self.layery = resBlock_2(hid, hid*2, 13)

        self.layer4 = nn.Sequential( 
                nn.Conv2d(hid*2, hid*4, kernel_size=(1, 13), padding=(0, 13//2)),
                nn.BatchNorm2d(hid*4, momentum=.1, affine=True),
                nn.ELU(),
                nn.Conv2d(hid*4, hid*4, kernel_size=(self.num_chan, 1)),
                nn.BatchNorm2d(hid*4, momentum=.1, affine=True),
                nn.ELU(),
            )    
        self.avgdrp = nn.Sequential(
            nn.AvgPool2d((1,13),(1,2)),
            nn.Dropout2d(0.5)
        )
        conv_len = self.sample_len // 2 - 6
        self.fc_module = nn.Sequential(
            nn.Conv2d(hid*4, self.num_cls, (1,conv_len)),
            nn.Sigmoid() if self.num_cls == 1 else nn.LogSoftmax(),
        )

    def forward(self, x): # B x 8 * 250
        out = self.layerx(x[:,:,:self.num_chan,:])
        if self.withhil:
            outy = self.layery(x[:,:,self.num_chan:,:])
            out = torch.cat((out,outy),dim=1)
        out = self.layer4(out)
        out = self.avgdrp(out)
        out = self.fc_module(out)
        return torch.squeeze(out)

def make_input(data:np.ndarray): # 32 * 8 * 8 * 7500 => 32 * 8 * 8 * 30 * 250
    ### data shape: 32 * 7680 (32 electrode channels * 60 s samples of 128 Hz)
    num_axis = len(data.shape)
    assert num_axis in [3,4], 'num_axis must be 4 or 3'
    dat2 = np.imag(hilbert(data))
    if num_axis == 4:
        a,b,c,d = data.shape
        dat1 = data.reshape(a,b,c,30,d//30)
        dat2 = dat2.reshape(a,b,c,30,d//30)
    elif num_axis == 3:
        b,c,d = data.shape
        dat1 = data.reshape(b,c,30,d//30)
        dat2 = dat2.reshape(b,c,30,d//30)
    return np.concatenate((dat1, dat2), axis=num_axis-1)

def make_input_2(data:np.ndarray): # sub * trial * elec * reduced_samlple_len => 32 * 8 * 8 * 30 * 250
    ### data shape: 32 * 7680 (32 electrode channels * 60 s samples of 128 Hz)
    dat2 = np.imag(hilbert(data))
    out = np.concatenate((data, dat2), axis=-2)
    return np.expand_dims(out,-3)

if __name__ == "__main__":
    model = HiRENet_2([2, 250], 32)
    a = torch.randn((16, 1, 4, 250)) # (B, chan, elec*2, sample_len)
    out = model(a)
    print(out.shape)