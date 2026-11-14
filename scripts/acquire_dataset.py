"""
acquire_dataset.py
==================
Fetch real matrices from NIST Matrix Market with full provenance.

    python scripts/acquire_dataset.py crawl                     # build data/metadata/mm_candidates.csv
    python scripts/acquire_dataset.py download --min-nnz 20000  # download selected matrices
    python scripts/acquire_dataset.py verify                    # re-check SHA-256 of local files

The candidate table and manifest are small and tracked in Git; the .mtx files
are not (see .gitignore).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd

from data_pipeline import acquire as acq

META = _root / "data" / "metadata"
CANDIDATES = META / "mm_candidates.csv"
MANIFEST = META / "dataset_manifest.csv"
RAW = _root / "data" / "raw" / "matrix_market"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("crawl")
    sub.add_parser("refresh-missing")
    d = sub.add_parser("download")
    d.add_argument("--min-nnz", type=int, default=20000)
    d.add_argument("--max-nnz", type=int, default=50_000_000)
    d.add_argument("--raw-dir", default=str(RAW))
    sub.add_parser("verify")
    args = ap.parse_args()

    if args.cmd == "crawl":
        df = acq.build_candidate_table()
        META.mkdir(parents=True, exist_ok=True)
        df.to_csv(CANDIDATES, index=False)
        print(f"wrote {CANDIDATES} ({len(df)} candidate pages)")
        return 0

    if args.cmd == "refresh-missing":
        df = acq.refresh_missing(pd.read_csv(CANDIDATES))
        df.to_csv(CANDIDATES, index=False)
        print(f"remaining unparsed: {int(df['nnz_full'].isna().sum())}")
        return 0

    if not CANDIDATES.is_file():
        print(f"ERROR: {CANDIDATES} missing; run 'crawl' first", file=sys.stderr)
        return 2

    if args.cmd == "download":
        cand = acq.select_candidates(pd.read_csv(CANDIDATES), args.min_nnz, args.max_nnz)
        cand.to_csv(META / "mm_selection.csv", index=False)
        sel = cand[cand["selected"]]
        print(f"{len(sel)} of {len(cand)} candidates selected (min_nnz={args.min_nnz})")
        recs = [acq.download_matrix(r, Path(args.raw_dir)) for _, r in sel.iterrows()]
        man = pd.DataFrame(recs, columns=acq.MANIFEST_COLUMNS)
        man.to_csv(MANIFEST, index=False)
        print(man["download_status"].value_counts().to_string())
        return 0 if (man["download_status"] == "ok").all() else 1

    if args.cmd == "verify":
        man = pd.read_csv(MANIFEST)
        res = acq.verify_manifest_checksums(man, RAW)
        bad = res[~res["ok"]]
        print(f"verified {len(res)} files; {len(bad)} problem(s)")
        if len(bad):
            print(bad.to_string(index=False))
        return 1 if len(bad) else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
