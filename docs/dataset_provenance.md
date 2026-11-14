# Dataset Provenance

All matrices are **real** sparse matrices from the **NIST Matrix Market** (<https://math.nist.gov/MatrixMarket/>).
No synthetic matrix is part of the real experiment (the seven synthetic pilot matrices in `data/metadata/*.csv`
are used only for software tests and demos and are never mixed with the real data).

## Why this source
The SuiteSparse collection returned HTTP 403 for matrix-file downloads when the project started (and again when
re-checked), so it is not used. The Matrix Market pages and files were reachable and each matrix has a page with
size, type and source information.

## Provenance records
* `data/metadata/mm_candidates.csv` - all 498 matrix pages crawled by `scripts/acquire_dataset.py crawl`
  (name, collection, set, group, page URL, file URL, size, type, discipline, source note).
* `data/metadata/mm_selection.csv` - the same table with the selection decision and reason for every candidate.
* `data/metadata/dataset_manifest.csv` - one row per downloaded matrix: id, name, source, **source URL**, licence note,
  collection, set, group, application domain, rows, cols, stored and CSR nonzeros, sparsity, format, field, symmetry,
  **acquisition date, SHA-256 of the .gz and of the .mtx**, download status and validation status.
* Raw `.mtx` files live in `data/raw/matrix_market/` (git-ignored). `python scripts/acquire_dataset.py verify`
  re-hashes them; `python scripts/validate_dataset.py` also loads and CSR-validates every file.

## Licence
The Matrix Market site states no explicit licence that we could locate (its disclaimer URL returns 404); it is a
NIST public service. The repository therefore does **not** redistribute matrix files; it records URLs and checksums and
fetches files on demand. Individual matrices originate from their listed contributors (the `source_note` column);
cite the original sources and the Matrix Market when publishing.

## Selection rules (decided before any timing existed)
1. Numeric field `real`, `integer` or `pattern` (complex matrices are excluded: the benchmark is real float32).
   Pattern matrices have no stored values; Matrix Market convention (value 1) is used. Their structure, which is what
   matters for kernel behaviour, is real, their values are not. 112 candidates are pattern matrices.
2. Expanded nonzeros >= 20,000 (smaller matrices run in about a microsecond and are dominated by timer and launch
   floors). Result: 157 matrices downloaded, 0 download failures.
3. Validation (`data_pipeline/validator.py`): explicit zeros are removed (e.g. `memplus` 126,150 -> 99,147), so
   `nnz_csr` can be smaller than the file's count and a matrix can fall below the 20,000 threshold afterwards (it stays).
4. **Exact duplicate sparsity patterns are dropped** (hash of `indptr` + `indices`; the smallest `matrix_id` is kept).
   Matrices that differ only in values have identical structural features and CSR behaviour, so keeping them would
   inflate the sample without adding structure. 35 were dropped (157 -> 122), 31 of them in the `drivcav` family; every drop is
   logged in `data/metadata/real/redundancy_log.csv` with the kept representative.
5. No matrix was removed for being inconvenient for the GPU or for its timings.

Structural diversity of the 122 matrices: 441 to 90,449 rows (median about 3,600), 20 thousand to 4.4 million nonzeros,
mean row length 4 to 173, maximum row length up to 4,230. Empty rows are rare (6 of 122 matrices have any) (the sources are mostly finite-element, fluid,
structural and economic models; there are no web graphs or power-law networks), which limits conclusions about highly
skewed matrices.

## Groups and split
* Group = Matrix Market set name with trailing digits and an `_old` suffix removed (`bcsstruc1..5` -> `bcsstruc`,
  `drivcav_old` -> `drivcav`). This is a heuristic for "related matrices"; related groups can still be split across
  train/val/test (a threat to validity). 23 groups (de-duplication did not change the count); three groups hold most matrices
  (`fidap` 40, `bcsstruc` 23, `drivcav` 15).
* Split (`data_pipeline/splits.py`): whole groups are assigned to a split by size-aware greedy packing, seed
  **20261115**, ratios 70/15/15. The earlier group split ignored group size (audit finding F1). Adding a group-count term
  was tried (weights 0.25, 0.5, 1.0) and rejected because it shrank validation and test to 5-12 matrices; the term is
  off (weight 0). Only group sizes and structure were used to make this choice, never timings.
* Frozen split: **train 86 matrices (8 groups: fidap, bcsstruc, drivcav, sherman, psmigr, astroph, dwave, econaus),
  validation 18 (8 groups: econiea, tokamak, mhd, lshape, crystal, dwt, hamm, saylor), test 18 (7 groups: cylshell, gemat,
  lns, airfoil, bcspwr, mvmrwk, platz)**. `data/metadata/real/splits.csv` is committed; do not regenerate it after
  looking at test results.

Consequences: train is dominated by three families; the test set has 7 groups (4 of them single matrices), so group-level
confidence intervals are wide and per-group results are anecdotal.
