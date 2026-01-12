# agents/tiny_tor_cnn.py  – *exact* arch used for training
import torch
import torch.nn as nn, torch.nn.functional as F

class ConvBN(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k//2)
        self.bn   = nn.BatchNorm2d(cout)
    def forward(self,x): return F.gelu(self.bn(self.conv(x)))

class TinyTorCNN(nn.Module):
    def __init__(self, cin=14, base=32, depth=(2,2,3,3)):
        super().__init__()
        layers=[]; C=cin
        for i,n in enumerate(depth):
            for _ in range(n):
                layers.append(ConvBN(C, base<<i)); C=base<<i
            layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers[:-1])
        self.head    = nn.Sequential(nn.Conv2d(C,64,1), nn.GELU(),
                                     nn.Conv2d(64,1,1))
    def forward(self,x): return torch.sigmoid(self.head(self.encoder(x)))
