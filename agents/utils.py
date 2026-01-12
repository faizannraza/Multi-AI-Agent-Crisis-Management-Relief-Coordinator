# agents/utils.py
from pathlib import Path
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# ─────────────────────────────────  tweet helpers
def _load_lora(folder: Path, num_labels: int):
    base = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    cfg   = LoraConfig.from_pretrained(folder)
    model = get_peft_model(base, cfg)
    # explicitly load the safetensors adapter if present
    adapter_file = "adapter_model.safetensors" if (folder/"adapter_model.safetensors").exists() else None
    model.load_adapter(
        folder,
        adapter_name="default",
        file_name=adapter_file
    )
    model.eval()
    return model


def load_tweet_bin():
    return _load_lora(Path("models/tweet_bin"),   2)


def load_tweet_multi():
    return _load_lora(Path("models/tweet_multi"), 4)


# ─────────────────────────────────  radar helper
_CKPT_PATH  = Path("models/radar/tiny_tor_cnn1.pt")
_NORM_PATH  = Path("models/radar/radar_norm.json")


def _rename_legacy_keys(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map legacy keys like 'encoder.0.weight' → 'encoder.0.conv.weight'
    and 'encoder.2.bias'  → 'encoder.1.conv.bias' (etc.).

    This covers the old checkpoint format shipped with the course‐material.
    """
    remapped = {}
    for k, v in ckpt.items():
        if not k.startswith("encoder."):
            remapped[k] = v
            continue

        # old keys look like:  encoder.{idx}.{param}
        parts = k.split(".")             # ['encoder', '0', 'weight']
        idx   = int(parts[1])
        param = parts[2]                 # weight / bias

        # every *even* block (0,2,4,…) is Conv, odd block is BN
        block   = idx // 2               # 0,1,2,3,…
        is_conv = idx % 2 == 0

        if is_conv:
            new_key = f"encoder.{block}.conv.{param}"
        else:
            new_key = f"encoder.{block}.bn.{param}"

        remapped[new_key] = v

    return remapped


def load_radar():
    """
    Return (trained_model, normalisation_dict).

    • Converts *legacy* checkpoints to the current TinyTorCNN layer names  
    • Drops any weights whose shapes don’t match the model  
    • Always puts the model into `.eval()` mode.
    """
    from .tiny_tor_cnn import TinyTorCNN

    if not _CKPT_PATH.exists():
        raise FileNotFoundError(
            f"[load_radar] Trained weights not found → {_CKPT_PATH}"
        )

    net  = TinyTorCNN()
    ckpt = torch.load(_CKPT_PATH, map_location="cpu")

    # --- automatic legacy-key remap -----------------------------------------
    if any(k.startswith("encoder.") and ".conv." not in k for k in ckpt):
        ckpt = _rename_legacy_keys(ckpt)
    # ------------------------------------------------------------------------

    # filter out any keys whose tensor shape doesn’t match net.state_dict()
    own_state = net.state_dict()
    filtered_ckpt = {
        k: v for k, v in ckpt.items()
        if k in own_state and v.shape == own_state[k].shape
    }

    # now load whatever matches; mismatched layers are skipped
    net.load_state_dict(filtered_ckpt, strict=False)
    net.eval()  # deterministic inference

    # fallback normalisation
    if _NORM_PATH.exists():
        norm = json.load(_NORM_PATH.open())
    else:
        norm = {k: 1.0 for k in ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]}

    return net, norm
