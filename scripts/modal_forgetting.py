"""Run scripts/forgetting.py on one Modal H100 for every main run.

  modal volume create --version=2 phase-lora-forgetting                   # once
  modal volume put phase-lora-forgetting <staging>/runs /runs              # the main runs' JSON and weights
  modal run scripts/modal_forgetting.py::smoke                             # environment and swap check, ~5 min
  modal run --detach scripts/modal_forgetting.py::main                     # every main run, ~15 min
  modal volume get phase-lora-forgetting /runs/forgetting runs/

The image installs the exact library versions of uv.lock (exported at image build time). The base models are
downloaded inside the container by scripts/prepare_models.py at their pinned revisions and checked by sha256.
Results are written to the volume one file per run, so a preempted run restarts and skips finished runs.
"""

import os
import subprocess
from pathlib import Path

import modal

REMOTE = "/root/phase-lora"
VOLUME_RUNS = "/vol/runs"
LOCAL_REPO = Path(__file__).resolve().parents[1] if modal.is_local() else Path(REMOTE)

if modal.is_local():
    LOCKED = subprocess.run(["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project"],
                            cwd=LOCAL_REPO, capture_output=True, text=True, check=True).stdout
    REQUIREMENTS = LOCAL_REPO / "verify_runs" / "modal_requirements.txt"     # verify_runs/ is git-ignored scratch
    REQUIREMENTS.parent.mkdir(exist_ok=True)
    REQUIREMENTS.write_text(LOCKED)
else:
    REQUIREMENTS = None

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements(str(REQUIREMENTS), extra_index_url="https://download.pytorch.org/whl/cu130")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/root/hf", "PHASE_LORA_RUNS_DIR": VOLUME_RUNS})
    .add_local_dir(LOCAL_REPO / "src", f"{REMOTE}/src")
    .add_local_dir(LOCAL_REPO / "scripts", f"{REMOTE}/scripts")
    .add_local_file(LOCAL_REPO / "code_versions.json", f"{REMOTE}/code_versions.json")
    .add_local_file(LOCAL_REPO / "runs" / "score_all.txt", f"{REMOTE}/score_all.txt")
)
app = modal.App("phase-lora-forgetting", image=image)
volume = modal.Volume.from_name("phase-lora-forgetting", version=2)


def run_list(limit=None):
    """The main runs (runs/score_all.txt, as run_sweep --phase list writes it), as paths on the volume."""
    names = [Path(line).name for line in Path(f"{REMOTE}/score_all.txt").read_text().split()]
    return [f"{VOLUME_RUNS}/{name}" for name in names][:limit]


def prepare():
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout, flush=True)
    subprocess.run(["python", "scripts/prepare_models.py"], cwd=REMOTE, check=True)


@app.function(gpu="H100!", timeout=1800, cpu=8.0, memory=65536, volumes={"/vol": volume})
def smoke():
    """Versions, model checksums, the swap-versus-fresh check on two runs of each kind for one model and task,
    and the full measurement of two runs; everything a paid run executes, at small size."""
    import torch
    import transformers
    import peft
    print("torch", torch.__version__, torch.cuda.get_device_name(0), "transformers", transformers.__version__,
          "peft", peft.__version__, flush=True)
    prepare()
    runs = run_list()
    check_runs = [path for path in runs if "_svamp_qwen3-0.6b-base_" in path and ("_s1_" in path or "_s2_" in path)]
    subprocess.run(["python", "scripts/forgetting.py", "--check", "--check-windows", "4", "--batch-size", "2",
                    "--runs", *check_runs], cwd=REMOTE, check=True)
    # smoke results go to the container's disk, so they never mix with the real results on the volume
    subprocess.run(["python", "scripts/forgetting.py", "--batch-size", "8", "--runs", *runs[:2]], cwd=REMOTE,
                   check=True, env={**os.environ, "PHASE_LORA_FORGETTING_OUT": "/root/smoke_out"})


@app.function(gpu="H100!", timeout=2 * 3600, cpu=8.0, memory=65536, volumes={"/vol": volume})
def measure():
    prepare()
    process = subprocess.run(["python", "scripts/forgetting.py", "--batch-size", "8", "--runs", *run_list()],
                             cwd=REMOTE)
    volume.commit()
    if process.returncode:
        raise SystemExit(f"forgetting.py exited {process.returncode}")


@app.local_entrypoint()
def main():
    measure.remote()
