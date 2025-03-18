import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(input_dim * patch_size, embed_dim)
    
    def forward(self, x):
        B, C, T = x.shape  # (batch, channels, time)
        num_patches = T // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]  # Trim to fit patches
        x = x.view(B, C, num_patches, self.patch_size)  # (B, C, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, num_patches, -1)  # (B, num_patches, C * patch_size)
        x = self.projection(x)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, output_dim, patch_size):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.reconstruction = nn.Linear(embed_dim, output_dim * patch_size)
        self.patch_size = patch_size
    
    def forward(self, x, memory):
        x = self.decoder(x, memory)  # (B, num_patches, embed_dim)
        x = self.reconstruction(x)  # (B, num_patches, output_dim * patch_size)
        return x

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, num_heads, num_layers):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_dim, embed_dim, patch_size)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers, input_dim, patch_size)
    
    def forward(self, x):
        B, C, T = x.shape        
        x_embed = self.patch_embed(x)  # (B, num_patches, embed_dim)
        latent = self.encoder(x_embed)  # Encode
        recon = self.decoder(x_embed, latent)  # Decode
        num_patches = recon.shape[1]
        recon = recon.view(B, C, -1)  # Flatten
        return recon, latent

class AutoencoderClassifier(nn.Module):
    def __init__(self, autoencoder, input_len, emb_dim, patch_size, n_classes=1):
        super().__init__()
        self.ae_model = autoencoder

        self.block_cls = nn.Sequential(
            nn.Linear(input_len//patch_size*emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, n_classes),
            nn.Sigmoid() if n_classes == 1 else nn.LogSoftmax()
        )

    def forward(self, x:torch.Tensor):
        _, x = self.ae_model(x)
        x = x.flatten(1)
        x = self.block_cls(x)
        return x

if __name__ == "__main__":
    # 예제 입력 처리
    batch_size = 4
    channels = 1
    time_steps = 256  # 가변 길이 가능
    input_dim = channels  # EEG 채널 수
    embed_dim = 64
    patch_size = 16
    num_heads = 4
    num_layers = 3

    model = TransformerAutoencoder(input_dim, embed_dim, patch_size, num_heads, num_layers)
    data = torch.randn(batch_size, channels, time_steps)  # (B, C, T)
    output = model(data)
    print(output.shape)  # (B, T)

    data = torch.randn(batch_size, channels, time_steps//2)  # (B, C, T)
    output = model(data)
    print(output.shape)  # (B, T)