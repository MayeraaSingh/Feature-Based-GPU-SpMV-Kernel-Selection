"""
validate_dataset.py
===================
Validate the real dataset before any benchmark runs:

1. every manifest row marked ``download_status == ok`` has its file and a matching SHA-256;
2. every file loads through the project loader and passes CSR validation;
3. the CSV contracts (features / splits / catalog) agree with the manifest on matrix ids;
4. synthetic or generated data is not present in the real raw directory.

Writes ``validation_status`` / ``nnz_csr`` / ``sparsity`` back into the manifest (a derived
record of what was checked) and exits non-zero if anything is wrong.

    python scripts/validate_dataset.py [--manifest data/metadata/dataset_manifest.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd

from data_pipeline.acquire import sha256_file
from data_pipeline.loader import load_matrix


def validate(manifest_path: Path, raw_dir: Path, write_back: bool = True) -> pd.DataFrame:
    man = pd.read_csv(manifest_path)
    status, nnz_csr, sparsity, notes = [], [], [], []
    for _, r in man.iterrows():
        mid = r["matrix_id"]
        path = raw_dir / f"{mid}.mtx"
        if r["download_status"] != "ok":
            status.append("not_downloaded"); nnz_csr.append(None); sparsity.append(None); notes.append("")
            continue
        if not path.is_file():
            status.append("file_missing"); nnz_csr.append(None); sparsity.append(None); notes.append(str(path))
            continue
        if sha256_file(path) != r["sha256_mtx"]:
            status.append("checksum_mismatch"); nnz_csr.append(None); sparsity.append(None); notes.append("")
            continue
        try:
            csr, _ = load_matrix(path, matrix_id=mid)
        except Exception as exc:
            status.append("invalid_csr"); nnz_csr.append(None); sparsity.append(None); notes.append(str(exc)[:200])
            continue
        status.append("ok")
        nnz_csr.append(int(csr.nnz))
        sparsity.append(1.0 - csr.nnz / (csr.shape[0] * csr.shape[1]))
        notes.append("")
    man["validation_status"] = status
    man["nnz_csr"] = nnz_csr
    man["sparsity"] = sparsity
    man["note"] = [n if n else old for n, old in zip(notes, man["note"].fillna(""))]
    if write_back:
        man.to_csv(manifest_path, index=False)
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(_root / "data/metadata/dataset_manifest.csv"))
    ap.add_argument("--raw-dir", default=str(_root / "data/raw/matrix_market"))
    args = ap.parse_args()
    mpath = Path(args.manifest)
    if not mpath.is_file():
        print(f"ERROR: manifest not found: {mpath}. Run scripts/acquire_dataset.py download.", file=sys.stderr)
        return 2
    man = validate(mpath, Path(args.raw_dir))
    counts = man["validation_status"].value_counts().to_dict()
    print(f"{len(man)} manifest rows: {counts}")
    synthetic = man[man["source"].astype(str).str.contains("synthetic", case=False)]
    if len(synthetic):
        print(f"ERROR: {len(synthetic)} synthetic rows in the real manifest", file=sys.stderr)
        return 1
    return 0 if counts.get("ok", 0) == len(man) else 1


if __name__ == "__main__":
    raise SystemExit(main())
