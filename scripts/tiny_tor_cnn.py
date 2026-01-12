import torch, torch.nn as nn, torch.nn.functional as F

class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=k//2),
            nn.BatchNorm2d(c_out),
            nn.GELU())
    def forward(self,x): return self.block(x)

class TinyTorCNN(nn.Module):
    """
    Input 14‑chan (6 vars×2 tilts + rangeFold + r + 1/r).
    Output 1×H×W likelihood map.
    """
    def __init__(self, cin=14, base=32, depth=(2,2,3,3)):
        super().__init__()
        layers, c = [], cin
        for i, n_blk in enumerate(depth):
            for _ in range(n_blk):
                layers.append(ConvBN(c, base<<i))
                c = base<<i
            layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers[:-1])  # drop last pool
        self.head = nn.Sequential(
            nn.Conv2d(c, 64, 1), nn.GELU(),
            nn.Conv2d(64, 1, 1))        # logits
    def forward(self, x):
        x = self.encoder(x)
        return torch.sigmoid(self.head(x))          # 1×h×w