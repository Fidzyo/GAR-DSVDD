# GAR-DSVDD — Blobs-Only Benchmark

This repository contains a cleaned, GitHub-ready version of the code for **Graph-Attention Regularized Deep SVDD (GAR-DSVDD)** and baselines.

## What's Included
- `src/experiment.py` — experiment driver.
- `src/gar_dsvdd.py` — GAR-DSVDD model implementation.
- `src/experiment_utils.py` — helper utilities.
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

## Repository Layout
```
gar-dsvdd-blobs-only/
├─ src/
│  ├─ experiment.py
│  ├─ gar_dsvdd.py
│  └─ experiment_utils.py
├─ scripts/
│  └─ run.py
├─ results/
├─ requirements.txt
└─ LICENSE
```

## Citation
If you use GAR-DSVDD in academic work, please cite your manuscript appropriately.



```bash
# Single seed
SEED_LIST=7 python scripts/run.py

# Multiple seeds
SEED_LIST=7,0,5 python scripts/run.py

# Or pass through to the experiment directly
python scripts/run.py -- --seed_list 7,0,5
```
