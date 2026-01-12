# ──────────── Run once on your laptop ────────────
conda create -n tornet-gpu python=3.10 -y
conda activate tornet-gpu

# Core libs
pip install -q "torch>=2.1" lightning xarray netCDF4 h5py tqdm
pip install -q zenodo_get fastapi uvicorn pydantic scikit-learn transformers peft accelerate sentencepiece datasets

# (Your CUDA toolkit’s driver + torch wheel handle GPU automatically)