import lightning as L, torch, numpy as np, xarray as xr, glob, os, json
from torch.utils.data import DataLoader, Dataset
from tiny_tor_cnn import TinyTorCNN

NORM = json.load(open("radar_norm.json"))
VARS = ["DBZ","VEL","KDP","RHOHV","ZDR","WIDTH"]

class TorNetDS(Dataset):
    def __init__(self, files, norm):
        self.files, self.norm = files, norm
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        ds = xr.open_dataset(self.files[idx])
        X = []
        for var in VARS:
            arr = ds[var][ -1, ... ].transpose('sweep','azimuth','range').values
            arr = np.nan_to_num(arr, nan=0.0)
            X.append(arr / self.norm[var])
        mask = ds.range_folded_mask[-1].transpose('sweep','azimuth','range').values
        X.append(mask/1.0)
        # radial coordinate helpers
        r = np.linspace(0,1,240)[None,None,:].repeat(2,0).repeat(120,1)
        X.append(r);        X.append(1.0/(r+1e-3))
        X = np.stack(X).astype('float32')
        y = ds.frame_labels[-1].item()      # 1/0
        ds.close()
        return torch.tensor(X), torch.tensor(y, dtype=torch.float32)

def make_loader(split):
    root = os.environ['TORNET_ROOT']
    files = glob.glob(f"{root}/{split}/20??/*.nc")
    return DataLoader(TorNetDS(files, NORM), batch_size=8,
                      shuffle=(split=='train'), num_workers=4, pin_memory=True)

class LitTiny(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = TinyTorCNN()
        self.loss = torch.nn.BCELoss()
    def forward(self,x): return self.net(x)
    def step(self,batch,stage):
        x,y = batch
        p = self(x).amax(dim=[2,3])         # global max‑pool
        l = self.loss(p.squeeze(), y)
        self.log(f"{stage}_loss", l)
        return l
    def training_step(self,batch,idx):  return self.step(batch,"tr")
    def validation_step(self,batch,idx):return self.step(batch,"val")
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-2)

if __name__=="__main__":
    tr, val = make_loader("train"), make_loader("test")   # TorNet already split
    model = LitTiny()
    trainer = L.Trainer(max_epochs=10, accelerator="auto",
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        default_root_dir="./tor_train")
    trainer.fit(model, tr, val)
    model.net.eval().cpu()
    model.net.save_pretrained("./tiny_tor_cnn.pt")  # custom util below
    
import torch, io, json
def save_pretrained(net, path):
    torch.save(net.state_dict(), path)
TinyTorCNN.save_pretrained = staticmethod(save_pretrained)

def from_pretrained(path):
    model = TinyTorCNN(); model.load_state_dict(torch.load(path,map_location='cpu'))
    model.eval()
    return model
TinyTorCNN.from_pretrained = staticmethod(from_pretrained)

