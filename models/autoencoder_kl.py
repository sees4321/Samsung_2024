import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, pad=False):
        super().__init__()
        # self.conv = nn.Conv1d(
        #     in_channels, in_channels, kernel_size=3, stride=1, padding=1
        # )
        self.pad = pad
        self.conv = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
        )

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # print(f'Before: {x.shape}')
        x = self.conv(x)
        if self.pad:
            x = F.pad(x, (1,1), mode='replicate')
        # print(f'After: {x.shape}')
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, padding=0, stride=2
        )

    def forward(self, x):
        pad = (0, 1)
        # x = F.pad(x, pad, mode="constant", value=0)
        x = F.pad(x, pad, mode="replicate")
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.reshape(b, c, h )
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h )  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h )
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h)

        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        z_channels: int,
        ch_mult: Tuple[int],
        num_res_blocks: int,
        resolution: Tuple[int],
        attn_resolutions: Tuple[int],
        ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convolution
        blocks.append(
            nn.Conv1d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        )

        # residual and downsampling blocks,
        # with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = n_channels * in_ch_mult[i]
            block_out_ch = n_channels * ch_mult[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if max(curr_res) in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = tuple(ti // 2 for ti in curr_res)

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv1d(block_in_ch, z_channels, kernel_size=3, stride=1, padding=1)
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        z_channels: int,
        out_channels: int,
        ch_mult: Tuple[int],
        num_res_blocks: int,
        resolution: Tuple[int],
        attn_resolutions: Tuple[int],
        ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)

        block_in_ch = n_channels * ch_mult[-1]
        curr_res = tuple(ti // 2 ** (self.num_resolutions - 1) for ti in resolution)

        blocks = []
        # initial conv
        blocks.append(
            nn.Conv1d(z_channels, block_in_ch, kernel_size=3, stride=1, padding=1)
        )

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = n_channels * ch_mult[i]

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if max(curr_res) in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                if i == 1:
                    blocks.append(Upsample(block_in_ch, True))
                else:
                    blocks.append(Upsample(block_in_ch))
                curr_res = tuple(ti * 2 for ti in curr_res)

        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv1d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class AutoencoderKL(nn.Module):
    def __init__(self, 
                 embed_dim:int, 
                 in_channels:int,
                 hid_channels:int,
                 z_channels:int,
                 resolution:int,
                 ch_mult=(1,2,4)
                 ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            n_channels=hid_channels,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=2,
            resolution=(resolution,),
            attn_resolutions=()
        )
        self.decoder = Decoder(
            n_channels=hid_channels,
            z_channels=z_channels,
            out_channels=in_channels,
            ch_mult=ch_mult,
            num_res_blocks=2,
            resolution=(resolution,),
            attn_resolutions=()
        )
        self.quant_conv_mu = torch.nn.Conv1d(z_channels, embed_dim, 1)
        self.quant_conv_log_sigma = torch.nn.Conv1d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def sampling(self, z_mu, z_sigma):
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x):
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, get_ldm_inputs=False):
        if get_ldm_inputs:
            return self.get_ldm_inputs(x)
        else:
            # z_mu, _ = self.encode(x)
            # reconstruction = self.decode(z_mu)
            # return reconstruction, z_mu
            z_mu, z_sigma = self.encode(x)
            z = self.sampling(z_mu, z_sigma)
            reconstruction = self.decode(z_mu)
            return reconstruction, z_mu, z_sigma

    def get_ldm_inputs(self, img):
        z_mu, z_sigma = self.encode(img)
        z = self.sampling(z_mu, z_sigma)
        return z

    def reconstruct_ldm_outputs(self, z):
        x_hat = self.decode(z)
        return x_hat

class AutoencoderClassifierKL(nn.Module):
    def __init__(self, autoencoder, input_len, emb_dim, patch_size, n_classes=1):
        super().__init__()
        self.ae_model = autoencoder

        self.block_cls = nn.Sequential(
            nn.Linear(input_len//patch_size*emb_dim, 500),
            nn.ELU(),
            nn.Linear(500, n_classes),
            nn.Sigmoid() if n_classes == 1 else nn.LogSoftmax()
        )

    def forward(self, x:torch.Tensor):
        _, x, _ = self.ae_model(x)
        x = x.flatten(1)
        x = self.block_cls(x)
        return x

class AutoencoderClassifierKL2(nn.Module):
    def __init__(self, autoencoder, input_len, emb_dim, patch_size, n_classes=1):
        super().__init__()
        self.ae_model = autoencoder

        self.block_cls = nn.Sequential(
            nn.Linear(256*emb_dim//2, 1024),
            # nn.ELU(),
            nn.Linear(1024, n_classes),
            nn.Sigmoid() if n_classes == 1 else nn.LogSoftmax()
        )
        self.emb = nn.Linear(input_len//patch_size, 256)
        self.lstm = nn.LSTM(emb_dim, emb_dim//2, 2, batch_first=True)

    def forward(self, x:torch.Tensor):
        _, x, _ = self.ae_model(x)
        x = self.emb(x)
        x, (h_n, c_n) = self.lstm(x.transpose(1,2))
        x = x.flatten(1)
        x = self.block_cls(x)
        return x

class AutoencoderClassifierKL3(nn.Module):
    def __init__(self, autoencoder, input_len, emb_dim, patch_size, n_classes=1):
        super().__init__()
        self.ae_model = autoencoder

        self.block1 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 13),
            nn.GroupNorm(4, emb_dim),
            nn.ELU(),
            nn.MaxPool1d(2,2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 13),
            nn.GroupNorm(4, emb_dim),
            nn.ELU(),
            nn.MaxPool1d(2,2)
        )

        self.dim_feat = ((input_len//patch_size - 12)//2 - 12)//2
        self.block_cls = nn.Sequential(
            nn.Linear(self.dim_feat*emb_dim, 128),
            # nn.ELU(),
            nn.Linear(128, n_classes),
            nn.Sigmoid() if n_classes == 1 else nn.LogSoftmax()
        )

    def forward(self, x:torch.Tensor):
        _, x, _ = self.ae_model(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        x = self.block_cls(x)
        return x

if __name__ == "__main__":
    model = AutoencoderKL(128, 2, 32, 16, 8)
    x = torch.randn((64,2,625))
    out = model(x)
    print(x.shape,out[0].shape)