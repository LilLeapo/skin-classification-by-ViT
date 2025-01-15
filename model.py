import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as models

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Use ResNet18 as the CNN backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer and pooling
        self.features = nn.Sequential(*list(resnet.children())[:-3])
        # Add a 1x1 convolution to adjust the channel dimension
        self.channel_adj = nn.Conv2d(256, output_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.channel_adj(x)
        return x

class ViT(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        num_classes=1, 
        dim=512, 
        depth=6, 
        heads=8, 
        mlp_dim=1024, 
        channels=3, 
        dim_head=64, 
        dropout=0., 
        emb_dropout=0.
    ):
        super().__init__()
        
        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(dim // 2)  # Use half dimensions for CNN features
        
        # Calculate the size of feature maps after CNN
        self.feature_size = image_size // patch_size  # Should be 14 for 224/16
        num_patches = self.feature_size ** 2
        
        # Patch embedding for remaining channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * channels, dim // 2),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # CNN feature extraction
        cnn_features = self.cnn(img)  # [B, dim//2, 14, 14]
        cnn_features = rearrange(cnn_features, 'b c h w -> b (h w) c')  # [B, 196, dim//2]
        
        # Standard ViT patch embedding
        vit_features = self.to_patch_embedding(img)  # [B, 196, dim//2]
        
        # Concatenate CNN and ViT features
        x = torch.cat([cnn_features, vit_features], dim=-1)  # [B, 196, dim]
        
        # Add position embeddings and cls token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transform
        x = self.transformer(x)

        # Classification
        return self.mlp_head(x[:, 0]) 