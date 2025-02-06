import torch
import torch.nn as nn
import torch.nn.functional as F
import functools 
import numpy as np

#-----------------------Residual Block-----------------------------------------------#
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x
    
#-----------------------Convmixer-----------------------------------------------#
from einops import reduce, rearrange
class PCA(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dw = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding="same", groups=dim)
    self.prob = nn.Softmax(dim=1)

  def forward(self,x):
    c = reduce(x, 'b c h w -> b c', 'mean')
    x = self.dw(x)
    c_ = reduce(x, 'b c h w -> b c', 'mean')
    raise_ch = self.prob(c_ - c)
    att_score = torch.sigmoid(c_ + c_*raise_ch)
    return torch.einsum('bchw, bc -> bchw', x, att_score)

class PSA(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.pw = nn.Conv2d(dim, dim, kernel_size=1)
    self.prob = nn.Softmax2d()

  def forward(self,x):
    s = reduce(x, 'b c w h -> b w h', 'mean')
    xp = self.pw(x)
    s_ = reduce(xp, 'b c w h -> b w h', 'mean')
    raise_sp = self.prob(s_ - s)
    att_score = torch.sigmoid(s_ + s_*raise_sp)
    return torch.einsum('bchw, bwh -> bchw', x, att_score)
class ConvMixer(nn.Module):
    def __init__(self, in_channels=1, dim=64, depth=1, kernel_size=64, patch_size=4):
        super(ConvMixer, self).__init__()
        
        # Initial Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        # PCA Layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        PCA(dim),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )
                ),
                # PSA 
                
                PSA(dim),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )
            for _ in range(depth)
        ])
        
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x
    
# -------------------Feature Extraction--------------------------------------------------#    
class LKA(nn.Module):
    def __init__(self, dim, kernel_size, dilated_rate=3):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size, padding='same', groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding='same', groups=dim, dilation=dilated_rate)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)

        return u*attn

class my_norm(nn.Module):
    def __init__(self, shape=4096):
        super().__init__()
        self.shape = shape
        self.norm = nn.LayerNorm(shape)
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
        x = self.norm(x)
        x = x.view(B,C,H,W)
        return x

class MultiScaleExtractor(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # self.head_pw = nn.Conv2d(dim, dim, 1)
        self.tail_pw = nn.Conv2d(dim, dim, 1)

        self.LKA3 = LKA(dim, kernel_size=3)
        self.LKA5 = LKA(dim, kernel_size=5)
        self.LKA7 = LKA(dim, kernel_size=7)
        self.norm3 = nn.BatchNorm2d(dim)
        self.norm5 = nn.BatchNorm2d(dim)
        self.norm7 = nn.BatchNorm2d(dim)

        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.conv_cn = nn.Conv2d(dim, dim, 3, groups=dim,padding=1)
        self.norm_last = nn.BatchNorm2d(dim)
    def forward(self, x):
        x_copy = x.clone()
        # x = self.head_pw(x)

        x3 = self.LKA3(x) + x
        x3 = self.norm3(x3)
        x5 = self.LKA5(x) + x
        x5 = self.norm5(x5)
        x7 = self.LKA7(x) + x
        x7 = self.norm7(x7)

        x = F.gelu(x3 + x5 + x7)
        x = self.tail_pw(x) + x_copy

        x = self.pointwise(x)
        x = self.conv_cn(x)
        x = F.gelu(self.norm_last(x))
        return x

def Feature_Extractor(dim=64, patch_size=4, depth=2):
    return nn.Sequential(
        nn.Conv2d(1, dim//2, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(1024),
        nn.GELU(),
        nn.Conv2d(dim//2, dim, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(256),
        nn.GELU(),
        *[MultiScaleExtractor(dim=dim) for _ in range(depth)]
    )


class SB(nn.Module):
    def __init__(self, features, G=64, d = 64):

        super(SB, self).__init__()
        self.features = features
        self.ln1 = nn.LayerNorm(features)
        self.ln2 = nn.LayerNorm(features)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=True),
                                nn.BatchNorm2d(d).eval(),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2):
        batch_size = f1.shape[0]

        # f1 = self.act(self.dw1(f1))
        # f2 = self.act(self.dw2(f2))

        f1 = f1.permute(0,2,3,1)
        f2 = f2.permute(0,2,3,1)
        f1 = self.ln1(f1)
        f2 = self.ln2(f2)
        f1 = f1.permute(0,3,1,2)
        f2 = f2.permute(0,3,1,2)

        feats = torch.cat((f1,f2), dim=1)
        feats = feats.view(batch_size, 2, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, 2, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats*attention_vectors, dim=1)

        return feats_V, attention_vectors
