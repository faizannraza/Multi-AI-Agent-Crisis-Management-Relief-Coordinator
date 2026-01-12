import xarray as xr, numpy as np, glob, json, os, random

root = os.environ['TORNET_ROOT']
files = glob.glob(f"{root}/train/20??/*.nc")
sample = random.sample(files, 2500)

# collect 99th-percentiles per variable
dct = {v: [] for v in ["DBZ","VEL","KDP","RHOHV","ZDR","WIDTH"]}
for f in sample:
    ds = xr.open_dataset(f)
    for v in dct:
        arr = ds[v].values
        arr = arr[arr > -999]  # drop fill
        if arr.size:
            dct[v].append(np.percentile(arr,99))
    ds.close()

norm = {k: float(np.median(v)) for k,v in dct.items()}
json.dump(norm, open("radar_norm.json","w"), indent=2)
print("Saved radar_norm.json:", norm)