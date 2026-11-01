"""Download pretrained ResNet-18 (torchvision ImageNet weights) from the Hugging Face hub."""
import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

REPO = "timm/resnet18.tv_in1k"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/resnet18_tv_in1k.safetensors")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    p = hf_hub_download(REPO, "model.safetensors")
    shutil.copy(p, a.out)
    print(f"downloaded {REPO} -> {a.out}")
