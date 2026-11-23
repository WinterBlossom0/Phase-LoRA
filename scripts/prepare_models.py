"""Download the base models into models/, pinned to the revisions every run used, and check their weights.

  python scripts/prepare_models.py                     # every model
  python scripts/prepare_models.py lfm3-1.7b-base      # chosen models

A model whose weights already match their sha256 is not downloaded again.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

from huggingface_hub import snapshot_download

from phase_lora import MODEL_DIR

# local name -> (Hub repository, commit, sha256 of model.safetensors); from the download records of the runs' models
MODELS = {
    "qwen3-0.6b-base": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
    "qwen3-1.7b-base": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                        "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"),
    "lfm3-1.7b-base": ("LiquidAI/LFM3-1.7B-Base", "7bca5038823c0d0f14efe06439109354a10e014c",
                       "61a3f541d65b03a2180d08964bf0d55e349463a2706032814d455ae046bc8b59"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    names = sys.argv[1:] or list(MODELS)
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        sys.exit(f"unknown models {unknown}; known: {sorted(MODELS)}")
    for name in names:
        repository, revision, expected = MODELS[name]
        weights = MODEL_DIR / name / "model.safetensors"
        if weights.exists() and sha256(weights) == expected:
            print(f"{name}: present, weights match")
            continue
        snapshot_download(repository, revision=revision, local_dir=MODEL_DIR / name)
        if sha256(weights) != expected:
            sys.exit(f"{name}: weights do not match sha256 {expected}")
        print(f"{name}: downloaded {repository}@{revision[:12]}, weights match")


if __name__ == "__main__":
    main()
