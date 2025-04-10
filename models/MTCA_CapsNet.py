import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel_size), stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)
        # print(outputs.shape)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        # print(outputs.shape)
        return squash(outputs)

class MTCA_CapsNet(nn.Module):
    def __init__(self,
                 elecrode_chan: int = 32,
                 time_samples: int = 7680,
                 num_caps = 8,
                 kernel_caps = 9,
                 num_classes = 1):
        # input_size: 32 x 7680
        super(MTCA_CapsNet, self).__init__()
        
        self.max_pool = nn.MaxPool1d(time_samples)
        self.avg_pool = nn.AvgPool1d(time_samples)
        self.chan_linear = nn.Linear(elecrode_chan,elecrode_chan)
        self.sigmoid = nn.Sigmoid()

        self.conv_first = nn.Sequential(
            nn.Conv2d(1,16,(1,kernel_caps),(1,2)),
            nn.ELU(),
            nn.AvgPool2d((1,2),(1,2))
        )

        self.primarycaps = PrimaryCapsule(16,16,num_caps,kernel_caps)
        # 3923968 256
        # 
        in_num_cap = ((time_samples - kernel_caps + 1) // 4 - kernel_caps + 1) * 2 * elecrode_chan
        self.digitcaps = DenseCapsule(in_num_cap,num_caps,16,16)

        self.fc = nn.Sequential(
            nn.Conv1d(16,num_classes,16),
            nn.Sigmoid()
            # nn.LogSoftmax(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_size: 32 x 7680
        mx = self.max_pool(x)
        av = self.avg_pool(x)
        chan_attn = self.chan_linear(mx.squeeze(-1)) + self.chan_linear(av.squeeze(-1))
        chan_attn = self.sigmoid(chan_attn)
        x = x * chan_attn.unsqueeze(2)

        # print(x.shape)
        x = self.conv_first(x.unsqueeze(1))
        # print(x.shape)
        x = self.primarycaps(x) # 16 * 7664
        # print(x.shape)
        x = self.digitcaps(x)
        # print(x.shape)
        x = self.fc(x)
        return torch.squeeze(x)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# a = torch.randn((16,32,7680)).to(device)
# encoder = MTCA_CapsNet().to(device)
# print(encoder(a).shape)