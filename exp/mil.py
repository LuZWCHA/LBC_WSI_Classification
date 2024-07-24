import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # breakpoint()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        t = time.time()
        torch.save(attn, f"/nasdata/private/zwlu/Now/ai_trainer/outputs/mil/dt1_feat/attn_res_new_sampler/{t}.pth")        
        print( f"/nasdata/private/zwlu/Now/ai_trainer/outputs/mil/dt1_feat/attn_res_new_sampler/{t}.pth")
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MILClassifier(nn.Module):
    """MIL classifier"""

    def __init__(self, in_dim=512, num_blocks=1, fc_dim=1024, num_classes=3):
        super().__init__()

        attn_blocks = []
        for _ in range(num_blocks):
            attn_blocks.append(
                Block(in_dim, num_heads=8, drop=0.3, attn_drop=0.3)
            )
        self.attn_blocks = nn.Sequential(*attn_blocks)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))

        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, fc_dim)
        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(self, x): # (B, N, C)
        # bre akpoint()
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, N + 1, C)
        x = self.norm(self.attn_blocks(x))[:, 0]  # (B, C)
        x = F.relu(self.fc(self.norm(x)))
        y = self.classifier(x)
        return y, x