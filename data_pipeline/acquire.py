"""
acquire.py
==========
Reproducible acquisition of real sparse matrices from the NIST Matrix Market.

Why this source: the SuiteSparse collection returned HTTP 403 for file
downloads when this project started; NIST Matrix Market is publicly reachable.
Every downloaded matrix gets a provenance row (source URL, discipline,
acquisition date, SHA-256) in ``data/metadata/dataset_manifest.csv``.

Licence: the Matrix Market site publishes no explicit licence text that we
could locate (its disclaimer URL returns 404). Raw matrices are therefore NOT
committed to Git; they are fetched on demand by ``scripts/acquire_dataset.py``
and verified against checksums recorded in the manifest.

Matrix *groups* (used for leakage-safe splitting) are derived from the Matrix
Market set name: trailing digits and an ``_old`` suffix are stripped so that
sets that are obviously versions or siblings of each other (``drivcav`` and
``drivcav_old``, ``bcsstruc1..5``) share a group. This is a heuristic and is
documented in docs/dataset_provenance.md.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd

logger = logging.getLogger(__name__)

BASE = "https://math.nist.gov"
INDEX_URL = BASE + "/MatrixMarket/matrices.html"
USER_AGENT = "spmv-research-acquire/1.0"
SOURCE_NAME = "NIST Matrix Market"
LICENSE_NOTE = (
    "No explicit licence text found on the Matrix Market site; NIST public "
    "service. Raw files are not redistributed by this repository."
)

CANDIDATE_COLUMNS = [
    "matrix_id", "name", "collection", "set", "group", "page_url", "file_url",
    "rows", "cols", "entries_stored", "nnz_full", "field", "symmetry",
    "discipline", "source_note", "file_bytes",
]
MANIFEST_COLUMNS = [
    "matrix_id", "name", "source", "source_url", "license", "collection", "set",
    "matrix_group", "application_domain", "rows", "cols", "nnz_stored", "nnz_csr",
    "sparsity", "format", "field", "symmetry", "acquired_date", "sha256_gz",
    "sha256_mtx", "download_status", "validation_status", "note",
]


def http_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_from_set(set_name: str) -> str:
    """Heuristic family name for a Matrix Market set (see module docstring)."""
    s = set_name.lower()
    s = re.sub(r"_old$", "", s)
    s = re.sub(r"\d+$", "", s)
    return s or set_name.lower()


def _clean_note(txt: str) -> str:
    """Drop the site footer that the page-text regexes sometimes swallow."""
    return txt.split(" The Matrix Market is a service")[0].strip()[:200]


def _text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?</script>|<!--.*?-->", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html)).strip()


def parse_index(html: str) -> List[str]:
    """Absolute URLs of all per-matrix pages listed on the 'by name' page."""
    links = re.findall(r'href="([^"]*/data/[^"]*\.html)"', html, flags=re.I)
    return sorted({urljoin(INDEX_URL, l) for l in links})



_SPARSKIT_FIELD = {"R": "real", "C": "complex", "P": "pattern", "I": "integer"}
_SPARSKIT_SYM = {"S": "symmetric", "U": "unsymmetric", "H": "hermitian", "Z": "skew-symmetric", "R": "rectangular"}


def _int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _parse_alt_stats(text: str):
    """Parse the two non-default statistics blocks found on some Matrix Market pages.

    * SPARSKIT block: ``Key = X , Type = RSA`` ... ``Dimension N = ..``  (square);
      symmetric matrices store one triangle, so the full nonzero count is
      ``2 * strict_lower + diagonal``.
    * MMSTATS block:  ``Matrix type: matrix coordinate complex general``,
      ``Dimension M/N``, ``Number of matrix entries``.
    Returns (rows, cols, stored, nnz_full, field, symmetry); unknown -> None.
    """
    t = re.search(r"Type = ([RCPIQ])([SUHZR])A", text)
    if t:
        n = _int(r"Dimension N = (\d+)", text)
        stored = _int(r"Number of nonzero elements = (\d+)", text)
        lower = _int(r"strict lower part = (\d+)", text)
        diag = _int(r"main diagonal = (\d+)", text)
        sym = _SPARSKIT_SYM.get(t.group(2))
        full = stored
        if sym == "symmetric" and lower is not None and diag is not None:
            full = 2 * lower + diag
        return n, n, stored, full, _SPARSKIT_FIELD.get(t.group(1)), sym
    t = re.search(r"Matrix type: matrix coordinate (\w+) (\w+)", text)
    if t:
        m = _int(r"Dimension M = (\d+)", text)
        n = _int(r"Dimension N = (\d+)", text)
        entries = _int(r"Number of matrix entries = (\d+)", text)
        return m, n, entries, entries, t.group(1).lower(), t.group(2).lower()
    return None, None, None, None, None, None


def parse_matrix_page(html: str, page_url: str) -> Dict[str, object]:
    """Extract provenance and size metadata from a matrix page. Missing -> None."""
    text = _text(html)
    m = re.search(r"/data/([^/]+)/([^/]+)/([^/]+)\.html$", page_url)
    if not m:
        raise ValueError(f"unexpected matrix page URL: {page_url}")
    coll, set_name, name = m.group(1), m.group(2), m.group(3)

    size = re.search(r"(\d+) x (\d+), (\d+) entries", text)
    typ = re.search(r"\b(real|complex|integer|pattern)\s+(symmetric|unsymmetric|hermitian|skew-symmetric|rectangular)", text, flags=re.I)
    full = re.search(r"A-A' (\d+)", text)
    disc = re.search(r"Discipline:\s*(.*?)\s*(?:Accession:|Set Information|$)", text)
    src = re.search(r"Source:\s*(.*?)\s*Discipline:", text)
    gz = re.search(r'href="([^"]+\.mtx\.gz)"', html, flags=re.I)
    nbytes = re.search(r"\.mtx\.gz\s*\((\d+) bytes\)", text)

    rows = int(size.group(1)) if size else None
    cols = int(size.group(2)) if size else None
    stored = int(size.group(3)) if size else None
    # "Nonzeros total" row: the first integer following the column header run.
    tot = re.search(r"total diagonal below diagonal above diagonal A-A' (\d+)", text)
    nnz_full = int(tot.group(1)) if tot else stored
    field = typ.group(1).lower() if typ else None
    symmetry = typ.group(2).lower() if typ else None
    if rows is None:  # alternative statistics blocks (SPARSKIT / MMSTATS), see below
        rows, cols, stored, nnz_full, field, symmetry = _parse_alt_stats(text)
    return {
        "matrix_id": name,
        "name": name,
        "collection": coll,
        "set": set_name,
        "group": group_from_set(set_name),
        "page_url": page_url,
        "file_url": urljoin(page_url, gz.group(1)) if gz else None,
        "rows": rows,
        "cols": cols,
        "entries_stored": stored,
        "nnz_full": nnz_full,
        "field": field,
        "symmetry": symmetry,
        "discipline": _clean_note(disc.group(1)) if disc else None,
        "source_note": _clean_note(src.group(1)) if src else None,
        "file_bytes": int(nbytes.group(1)) if nbytes else None,
    }


def build_candidate_table(delay_s: float = 0.15, limit: Optional[int] = None) -> pd.DataFrame:
    """Crawl the index and every matrix page politely. ~500 small requests."""
    index_html = http_get(INDEX_URL).decode("latin-1")
    pages = parse_index(index_html)
    if limit:
        pages = pages[:limit]
    rows = []
    for i, url in enumerate(pages):
        try:
            rows.append(parse_matrix_page(http_get(url).decode("latin-1"), url))
        except Exception as exc:  # recorded, not hidden
            logger.warning("page failed %s: %s", url, exc)
            rows.append({"matrix_id": url.rsplit("/", 1)[-1][:-5], "page_url": url, "source_note": f"PAGE_ERROR: {exc}"})
        time.sleep(delay_s)
    df = pd.DataFrame(rows)
    for c in CANDIDATE_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_COLUMNS].sort_values("matrix_id").reset_index(drop=True)


def select_candidates(
    cand: pd.DataFrame,
    min_nnz: int,
    max_nnz: int,
    allowed_fields: Iterable[str] = ("real", "integer", "pattern"),
) -> pd.DataFrame:
    """Deterministic, rule-based selection. Adds a ``selected`` / ``reason`` column.

    Rules (applied in order, first failure is the recorded reason):
      * page parsed and file URL present
      * numeric field type in ``allowed_fields`` (complex is excluded: the
        benchmark is real float32)
      * nnz_full within [min_nnz, max_nnz]
    The rules look only at size and type, never at kernel timings.
    """
    out = cand.copy()
    reasons = []
    for _, r in out.iterrows():
        if r["file_url"] is None or pd.isna(r["file_url"]) or pd.isna(r["nnz_full"]):
            reasons.append("page_unparsed")
        elif r["field"] not in set(allowed_fields):
            reasons.append(f"field={r['field']}")
        elif r["nnz_full"] < min_nnz:
            reasons.append(f"nnz<{min_nnz}")
        elif r["nnz_full"] > max_nnz:
            reasons.append(f"nnz>{max_nnz}")
        else:
            reasons.append("selected")
    out["reason"] = reasons
    out["selected"] = out["reason"] == "selected"
    return out


def download_matrix(row: pd.Series, raw_dir: Path, delay_s: float = 0.15) -> Dict[str, object]:
    """Download one matrix, verify gzip integrity, write ``<id>.mtx``, return manifest row."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rec: Dict[str, object] = {
        "matrix_id": row["matrix_id"], "name": row["name"], "source": SOURCE_NAME,
        "source_url": row["file_url"], "license": LICENSE_NOTE,
        "collection": row["collection"], "set": row["set"], "matrix_group": row["group"],
        "application_domain": row["discipline"], "rows": row["rows"], "cols": row["cols"],
        "nnz_stored": row["entries_stored"], "nnz_csr": None, "sparsity": None,
        "format": "MatrixMarket coordinate", "field": row["field"], "symmetry": row["symmetry"],
        "acquired_date": dt.date.today().isoformat(), "sha256_gz": None, "sha256_mtx": None,
        "download_status": "pending", "validation_status": "not_run", "note": "",
    }
    target = raw_dir / f"{row['matrix_id']}.mtx"
    try:
        blob = http_get(row["file_url"])
        mtx = gzip.decompress(blob)
        target.write_bytes(mtx)
        rec["sha256_gz"] = sha256_bytes(blob)
        rec["sha256_mtx"] = sha256_bytes(mtx)
        rec["download_status"] = "ok"
    except Exception as exc:
        rec["download_status"] = "failed"
        rec["note"] = str(exc)[:300]
    time.sleep(delay_s)
    return rec


def verify_manifest_checksums(manifest: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Re-hash every downloaded file. Returns rows with ``ok`` True/False and a reason."""
    results = []
    for _, r in manifest.iterrows():
        if r["download_status"] != "ok":
            continue
        p = raw_dir / f"{r['matrix_id']}.mtx"
        if not p.is_file():
            results.append({"matrix_id": r["matrix_id"], "ok": False, "reason": "file_missing"})
            continue
        actual = sha256_file(p)
        ok = actual == r["sha256_mtx"]
        results.append({"matrix_id": r["matrix_id"], "ok": ok, "reason": "" if ok else "checksum_mismatch"})
    return pd.DataFrame(results, columns=["matrix_id", "ok", "reason"])


def refresh_missing(cand: pd.DataFrame, delay_s: float = 0.15) -> pd.DataFrame:
    """Re-fetch only the rows whose size could not be parsed (e.g. after a parser fix)."""
    out = cand.copy()
    for i, r in out[out["nnz_full"].isna()].iterrows():
        try:
            rec = parse_matrix_page(http_get(r["page_url"]).decode("latin-1"), r["page_url"])
            for k, v in rec.items():
                out.at[i, k] = v
        except Exception as exc:
            logger.warning("refresh failed %s: %s", r["page_url"], exc)
        time.sleep(delay_s)
    return out
