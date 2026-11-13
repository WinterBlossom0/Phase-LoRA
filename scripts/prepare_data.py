"""Download the MAFAND en-tsn and en-lug, SVAMP and ARC-Challenge splits into data/, pinned.

  python scripts/prepare_data.py            # download, then check against data_manifest.json
  python scripts/prepare_data.py --check    # only check the files in data/ against data_manifest.json

Each dataset lands as data/<name>/{train,dev,test}.jsonl with its source rows
unchanged, plus data/MANIFEST.json (source, revision, row counts, sha256), so a
result can always be traced to the exact bytes it trained and scored on.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

from datasets import load_dataset

from phase_lora import DATA_DIR

EXPECTED = Path(__file__).resolve().parents[1] / "data_manifest.json"   # the bytes every run used; tracked

SVAMP_REV = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"
ARC_REV = "210d026faf9955653af8916fad021475a3f00453"
LAFAND_REV = "c1a423500753a28b2fae8f5d5e1634de768cd89b"
# masakhane/mafand on the Hub is a loading script, which datasets>=4 refuses to run;
# the script itself reads these JSON-lines files, so read them directly.
LAFAND_URL = f"https://raw.githubusercontent.com/masakhane-io/lafand-mt/{LAFAND_REV}/data/json_files/"
# SVAMP ships train/test only. Hold out the last train rows for early stopping, so
# the reported test score is never also the score that picked the checkpoint.
SVAMP_DEV_ROWS = 100


def write_split(name, split, rows, manifest, source, revision):
    path = DATA_DIR / name / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest[f"{name}/{split}"] = {"source": source, "revision": revision, "rows": len(rows),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(f"  {name}/{split}: {len(rows)} rows")


def fetch_mafand(manifest, code):
    """The published en-<code> train, dev and test splits of MAFAND-MT, unchanged."""
    for split in ("train", "dev", "test"):
        url = f"{LAFAND_URL}en-{code}/{split}.json"
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode("utf-8").split("\n")   # not splitlines(): see _local_rows
        rows = [json.loads(line) for line in lines if line.strip()]
        write_split(f"mafand_{code}", split, rows, manifest, url, LAFAND_REV)


def fetch_svamp(manifest):
    train = load_dataset("ChilleD/SVAMP", split="train", revision=SVAMP_REV).to_list()
    cut = len(train) - SVAMP_DEV_ROWS
    write_split("svamp", "train", train[:cut], manifest, "ChilleD/SVAMP train", SVAMP_REV)
    write_split("svamp", "dev", train[cut:], manifest, f"ChilleD/SVAMP train, last {SVAMP_DEV_ROWS} rows", SVAMP_REV)
    test = load_dataset("ChilleD/SVAMP", split="test", revision=SVAMP_REV).to_list()
    write_split("svamp", "test", test, manifest, "ChilleD/SVAMP test", SVAMP_REV)


def fetch_arc(manifest):
    for source_split, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=source_split, revision=ARC_REV).to_list()
        write_split("arc", split, rows, manifest, f"allenai/ai2_arc ARC-Challenge {source_split}", ARC_REV)


def files_match():
    """Every file named in data_manifest.json exists in data/ with the recorded sha256."""
    for key, entry in json.loads(EXPECTED.read_text()).items():
        path = DATA_DIR / f"{key}.jsonl"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            print(f"{path} is missing or differs from {EXPECTED.name}")
            return False
    return True


def main():
    if sys.argv[1:] == ["--check"]:
        sys.exit(0 if files_match() else 1)
    manifest = {}
    for code in ("tsn", "lug"):
        fetch_mafand(manifest, code)
    fetch_svamp(manifest)
    fetch_arc(manifest)
    (DATA_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {DATA_DIR / 'MANIFEST.json'}")
    if manifest != json.loads(EXPECTED.read_text()):
        sys.exit(f"the downloaded data differ from the data of the paper's runs ({EXPECTED.name})")
    print(f"data match {EXPECTED.name}")


if __name__ == "__main__":
    main()
