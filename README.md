# GAR-DSVDD — Blobs-Only Benchmark

This repository contains a cleaned, GitHub-ready version of the code for **Graph-Attention Regularized Deep SVDD (GAR-DSVDD)** and baselines,
restricted to the **blobs** toy dataset. All other datasets (including *wiper*) and any references to "Ours" have been removed.

## What's Included
- `src/experiment.py` — experiment driver (adapted from `paper1_v20.py`) with support for blobs only.
- `src/gar_dsvdd.py` — GAR-DSVDD model implementation.
- `src/paper1_gar_dsvdd_v3.py` — helper utilities (cleaned).
- `scripts/run.py` — parameterized runner with **best parameters** baked in (from your original code). It runs blobs only.
- `results/` — output folder for CSVs/plots.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT license.

## Quickstart

```bash
# 1) create & activate a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# Windows: python -m venv .venv && .venv\Scripts\activate

# 2) install requirements
pip install -r requirements.txt

# 3) run the best-params experiment on blobs only
python scripts/run.py
```

> **Note:** If you previously had flags related to the *wiper* dataset, they are removed. Only `blobs` is supported here.

## Repository Layout
```
gar-dsvdd-blobs-only/
├─ src/
│  ├─ experiment.py
│  ├─ gar_dsvdd.py
│  └─ paper1_gar_dsvdd_v3.py
├─ scripts/
│  └─ run.py
├─ results/
├─ requirements.txt
└─ LICENSE
```

## Citation
If you use GAR-DSVDD in academic work, please cite your manuscript appropriately.


### Selecting exact seeds
Run a specific seed (or list of seeds) by setting the `SEED_LIST` environment variable or passing `--seed_list`:

```bash
# Single seed
SEED_LIST=7 python scripts/run.py

# Multiple seeds
SEED_LIST=7,0,5 python scripts/run.py

# Or pass through to the experiment directly
python scripts/run.py -- --seed_list 7,0,5
```
