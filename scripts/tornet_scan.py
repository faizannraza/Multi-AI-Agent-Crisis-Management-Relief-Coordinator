#!/usr/bin/env python3
"""
tornet_scan.py
--------------

Quickly scan a TorNet‐Mini (or TorNet) folder, list every *.nc file that
*TinyTorCNN* can consume, and – if a label CSV is present – show whether
the volume is tornadic (1) or non-tornadic (0).

────────────────────────── HOW TO USE ──────────────────────────
1. Edit the two paths below:
      RADAR_DIR  –  directory with the .nc volumes
      LABELS_CSV –  *optional* explicit path to the CSV
                    (leave as None to auto-detect)
2. Run:
      python tornet_scan.py
3. Copy any file shown under “good_files” for your /event call.
"""

# ───────────────────────────────────────────────────────────────
# 0)  USER SETTINGS  ────  >>>>>>  CHANGE THESE TWO  <<<<<<
# ───────────────────────────────────────────────────────────────
from pathlib import Path

RADAR_DIR  = Path("/Users/muhammadfaizanraza/Desktop/PSGVSpring2/tornet_2013/test/2013")   # <── change me
LABELS_CSV = Path("/Users/muhammadfaizanraza/Desktop/PSGVSpring2/tornet_2013/catalog.csv")
# ───────────────────────────────────────────────────────────────


# 1)  Imports
import xarray as xr
import pandas as pd
import warnings

# variables that TinyTorCNN expects (order doesn’t matter)
_EXPECTED_VARS = [
    "DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask"
]


# 2)  Helper: does this file have the right variables / shapes?
def is_usable_nc(nc_path: Path) -> bool:
    try:
        with xr.open_dataset(nc_path) as ds:
            if not all(v in ds.variables for v in _EXPECTED_VARS):
                return False
            # very cheap shape sanity-check (120×240×14 after stacking)
            t, az, rng, swp = ds[_EXPECTED_VARS[0]].shape
            return (t, az, rng, swp) == (1, 360, 240, 2) or rng == 240
    except Exception:
        return False
    return True


# 3)  Load labels if a CSV exists (optional)
def load_labels(folder: Path, explicit_csv: Path | None):
    if explicit_csv and explicit_csv.exists():
        df = pd.read_csv(explicit_csv)
    else:
        # auto-detect common filenames
        for name in ("labels_test.csv", "catalog.csv"):
            p = folder / name
            if p.exists():
                df = pd.read_csv(p)
                break
        else:
            return {}
    col_file = "file_name" if "file_name" in df.columns else "filename"
    col_lbl  = "tornado"  if "tornado"  in df.columns else "tor"
    return dict(zip(df[col_file], df[col_lbl]))


# 4)  Main scan
def main():
    if not RADAR_DIR.is_dir():
        raise SystemExit(f"❌  RADAR_DIR does not exist → {RADAR_DIR}")

    labels = load_labels(RADAR_DIR, LABELS_CSV)
    if labels:
        print(f"✓ loaded {len(labels):,} labels from CSV")
    else:
        print("ℹ️  no label CSV found – proceeding label-free")

    good, bad = [], []
    for nc in sorted(RADAR_DIR.rglob("*.nc")):
        (good if is_usable_nc(nc) else bad).append(nc)

    print(f"\n✔ usable volumes : {len(good):,}")
    print(f"✖ unsuitable     : {len(bad):,}\n")

    print("First 15 good files:")
    for nc in good[:15]:
        lbl = labels.get(nc.name)
        tag = f"tornadic={lbl}" if lbl is not None else ""
        print(f"  {nc.name:<30} {tag}")

    # write the list to disk for convenience
    out = RADAR_DIR / "good_files.txt"
    out.write_text("\n".join(str(p) for p in good))
    print(f"\n📝  Full list saved to {out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()