# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 19:51:20 2025

@author: Taha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 14:58:06 2025

@author: Taha
"""

"""
grid_runner_wiper (removed).py — robust launcher for paper1_v20.py + GAR-DSVDD on toy/WIPER datasets.
Runs a full grid and APPENDS all per-run all_runs.csv files into one master CSV.

Usage:
- Put single values in a 1-element list, e.g. latent = [16]
- Put multiple values to sweep, e.g. k = [3, 5, 15]
- Every parameter in PARAMS is passed to the underlying paper script explicitly.
"""

import itertools, subprocess, sys, shlex, os, glob, csv, datetime, signal

# ---------- Lock working directory & paths ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH  = os.path.join(PROJECT_ROOT, '../src/experiment.py')
GAR_PATH     = os.path.join(PROJECT_ROOT, "paper1_gar_dsvdd_v3.py")   # GAR-DSVDD experiment script
OUTDIR_ABS   = os.path.join(PROJECT_ROOT, "plots")
GLOBAL_CSV   = os.path.join(OUTDIR_ABS, "all_runs_long.csv")
os.environ.setdefault("MPLBACKEND", "Agg")

# ======================================================================================
# IMPORTANT: Every entry below is a LIST (vector). Use length-1 lists for "fixed" args.
# ======================================================================================

# ===========================
# General / data parameters
# ===========================
datasets            = ["blobs"]   # dataset name(s); e.g., "blobs", "moons", "rings", "tricls", or "wiper (removed)"
n_norm              = [1000]      # number of normal points to sample (toy datasets)
n_anom              = [500]       # number of anomaly points to sample (toy datasets)
rho                 = [0.03]      # label noise ratio for labeled normals in split builder
epsilon             = [0.20]      # anomaly contamination ratio in unlabeled pool
val_frac            = [0.20]      # fraction of data used for validation split
test_frac           = [0.20]      # fraction of data used for test split
labeled_anom_frac   = [0.00]      # fraction of labeled anomalies (if any)
seeds               = [1]         # number of random seeds (run count per dataset)
wiper (removed)_use_sessions  = [1]         # for WIPER only: use session-aware splitting (1=yes, 0=no)

# =========================================
# Method selection (which models to run)
# =========================================
method = ["gar_dsvdd"]  # choose from: 'gar_dsvdd','gar_dsvdd','deepsvdd','deepsad','ocsvm','svdd_qp','s3svdd_qp','all'

# =========================================
# Shared representation / training (deep)
# =========================================
latent          = [16]    # encoder latent width (used by deep models)
epochs_warm     = [5]     # (OURS baseline only) warm-up epochs
epochs_main     = [120]   # main training epochs
lr              = [5e-4]  # learning rate for deep optimizers
wd              = [1e-5]  # weight decay (L2) for deep optimizers

# ===============================
# OURS (baseline) graph controls
# ===============================
graph           = ["attention"]  # OURS-only: graph type ('attention' or 'gaussian')
graph_space     = ["latent"]     # OURS-only: build graph in 'latent' or 'input' space
k               = [9]            # OURS-only: k for k-NN graph
alpha           = [0.01]         # OURS-only: unlabeled center-loss weight
mu              = [19500]        # OURS-only: score-Laplacian weight
heads           = [8]            # OURS-only: attention heads
dk              = [32]           # OURS-only: attention head width
refresh_every   = [10]           # OURS-only: rebuild latent kNN index every T epochs (0/None disables)
ema_beta        = [0.9]          # OURS-only: EMA smoothing for attention weights

# Graph artifacts (OURS & GAR)
save_graph      = [1]            # save graph artifacts (1=yes, 0=no)
graph_dir       = ["graphs"]     # sub-dir to write graph artifacts

# ===============================
# Threshold selection strategy
# ===============================
tau_strategy_ours      = ["train_quantile"]  # OURS: 'train_quantile' | 'fpr' | 'val_quantile' | 'youden' | 'zero'
tau_strategy_deepsvdd  = ["train_quantile"]  # DeepSVDD: threshold strategy
tau_strategy_deepsad   = ["train_quantile"]  # DeepSAD: threshold strategy
tau_strategy_ocsvm     = ["zero"]            # OCSVM: decision function is signed; often 'zero'
tau_strategy_svdd_qp   = ["zero"]            # SVDD-QP: typically 'zero'
tau_strategy_s3svdd_qp = ["zero"]            # S3SVDD-QP: typically 'zero'

# Quantiles (used when a strategy is a *quantile)
quantile_ours       = [0.95]
quantile_deepsvdd   = [0.95]
quantile_deepsad    = [0.95]
quantile_svdd_qp    = [0.95]
quantile_s3svdd_qp  = [0.95]

# Target FPRs (used when strategy == 'fpr')
target_fpr_ours       = [0.01]
target_fpr_deepsvdd   = [0.01]
target_fpr_deepsad    = [0.01]
target_fpr_ocsvm      = [0.01]
target_fpr_svdd_qp    = [0.01]
target_fpr_s3svdd_qp  = [0.01]

# ===============================
# Plotting / reporting
# ===============================
plot_sets      = ["train,unlabeled,test"]  # which groups to overlay in plot panels
plot_auto_unl  = [1]                       # auto-include unlabeled in plots if available (1=yes)
plot_boundary  = [1]                       # draw decision boundary (1=yes)
select_metric  = ["pAUC@1FPR"]             # console “best-by-method” metric

# ===============================
# DeepSAD controls (author-style)
# ===============================
sad_use_unlabeled = [0]      # include unlabeled data in DeepSAD loss
sad_use_lab_anom  = [0]      # include labeled anomalies in DeepSAD loss
sad_margin        = [1.0]    # DeepSAD margin hyperparameter
sad_eta           = [1.0]    # DeepSAD unlabeled weight
sad_lr            = [1e-3]   # DeepSAD learning rate
sad_wd            = [1e-5]   # DeepSAD weight decay
sad_batch_size    = [None]   # DeepSAD batch size (None: default in script)

# ===============================
# DeepSVDD controls (author-style)
# ===============================
deepsvdd_objective        = ["oneclass"]  # 'oneclass' | 'soft'
deepsvdd_nu               = [0.01]        # soft-boundary nu if objective='soft'
deepsvdd_pretrain_epochs  = [0]           # pretrain epochs (if baseline supports)
deepsvdd_batch_size       = [None]        # batch size (None: default)
deepsvdd_lr               = [5e-4]        # learning rate
deepsvdd_wd               = [1e-5]        # weight decay
deepsvdd_train_on         = ["lab_only"]  # 'lab_only' | 'lab+unl' (if supported)

# ===============================
# OCSVM / QP kernels
# ===============================
nu         = [0.01]    # OCSVM nu (or SVDD-QP nu)
gamma      = ["auto"]  # OCSVM RBF kernel gamma ('auto' or numeric)
rbf_gamma  = [1]       # SVDD-QP/S3SVDD-QP RBF gamma

# ===============================
# S3SVDD_QP specific (QP params)
# ===============================
s3_k           = [9]          # S3SVDD-QP: k for graph regularizer
s3_mu          = [10]         # S3SVDD-QP: Laplacian weight
s3_rbf_gamma   = [1]          # S3SVDD-QP: kernel gamma
s3_nu          = [0.1]        # S3SVDD-QP: nu
laplacian_type = ["sym"]      # S3SVDD-QP: 'sym' or 'rw' laplacian

# ===============================
# GAR-DSVDD specifics (method-scoped)
# ===============================
gar_alpha         = [0.9]            # GAR: unlabeled center weight (alpha)
gar_k             = [5]            # GAR: k-NN size
gar_graph         = ["attention"]  # GAR: 'attention' (with base blend) or 'gaussian'
gar_graph_space   = ["latent"]     # GAR: build kNN in 'latent' or 'input' space
gar_heads         = [8]            # GAR: multi-head attention count
gar_dk            = [32]           # GAR: per-head projection width
gar_refresh_every = [5]            # GAR: rebuild kNN every T epochs (0/None disables)
gar_ema_beta      = [0.3]          # GAR: EMA coefficient for attention smoothing

# Graph / attention specifics (GAR)
gar_base_kernel   = ["gaussian"]   # GAR: base edge affinity ('gaussian' or 'constant')
gar_attn_tau      = [1]          # GAR: attention softmax temperature
gar_attn_gamma    = [1]          # GAR: score-damping γ in attention logits
gar_attn_mu       = [1]          # GAR: blend [0=base only, 1=attention only]

# Objective & thresholding (GAR)
gar_beta          = [0.1]            # GAR: weak centering on ||c||^2
gar_lambda_u      = [1300]          # GAR: Laplacian smoothness weight 0:no graph
gar_tau_strategy  = ["train_quantile"]  # GAR: 'train_quantile' | 'fpr' | 'val_quantile' | 'youden' | 'zero'
gar_quantile      = [0.95]         # GAR: quantile for threshold if strategy uses quantiles
gar_target_fpr    = [0.1]          # GAR: target FPR if strategy == 'fpr'

# ===============================
# Output / CSV
# ===============================
outdir          = [OUTDIR_ABS]  # base output directory for run_* folders
grid_append_csv = [GLOBAL_CSV]  # path to append all per-run all_runs.csv rows

# ======================================================================================
# Build a single PARAMS dict mapping flag -> list-of-values (vectors). Keep ALL flags.
# ======================================================================================
PARAMS = {
    # Data & splits
    "--datasets": datasets,
    "--n_norm": n_norm,
    "--n_anom": n_anom,
    "--rho": rho,
    "--epsilon": epsilon,
    "--val_frac": val_frac,
    "--test_frac": test_frac,
    "--labeled_anom_frac": labeled_anom_frac,
    "--seeds": seeds,
    "--wiper (removed)_use_sessions": wiper (removed)_use_sessions,

    # Methods to run
    "--method": method,

    # Shared deep training (used by paper1_v20.py; GAR also uses latent/lr/wd/epochs_main)
    "--latent": latent,
    "--epochs_warm": epochs_warm,
    "--epochs_main": epochs_main,
    "--lr": lr,
    "--wd": wd,

    # OURS (baseline) graph controls — only used when method == "gar_dsvdd"
    "--graph": graph,
    "--graph_space": graph_space,
    "--k": k,
    "--alpha": alpha,
    "--mu": mu,
    "--heads": heads,
    "--dk": dk,
    "--refresh_every": refresh_every,
    "--ema_beta": ema_beta,

    # Graph artifacts (OURS & GAR)
    "--save_graph": save_graph,
    "--graph_dir": graph_dir,

    # Per-method τ strategies
    "--tau_strategy_ours":      tau_strategy_ours,
    "--tau_strategy_deepsvdd":  tau_strategy_deepsvdd,
    "--tau_strategy_deepsad":   tau_strategy_deepsad,
    "--tau_strategy_ocsvm":     tau_strategy_ocsvm,
    "--tau_strategy_svdd_qp":   tau_strategy_svdd_qp,
    "--tau_strategy_s3svdd_qp": tau_strategy_s3svdd_qp,

    # Per-method quantiles
    "--quantile_ours":      quantile_ours,
    "--quantile_deepsvdd":  quantile_deepsvdd,
    "--quantile_deepsad":   quantile_deepsad,
    "--quantile_svdd_qp":   quantile_svdd_qp,
    "--quantile_s3svdd_qp": quantile_s3svdd_qp,

    # Per-method target FPRs
    "--target_fpr_ours":       target_fpr_ours,
    "--target_fpr_deepsvdd":   target_fpr_deepsvdd,
    "--target_fpr_deepsad":    target_fpr_deepsad,
    "--target_fpr_ocsvm":      target_fpr_ocsvm,
    "--target_fpr_svdd_qp":    target_fpr_svdd_qp,
    "--target_fpr_s3svdd_qp":  target_fpr_s3svdd_qp,

    # =======================
    # GAR-DSVDD (method-scoped flags)
    # =======================
    "--gar_alpha":          gar_alpha,
    "--gar_k":              gar_k,
    "--gar_graph":          gar_graph,
    "--gar_graph_space":    gar_graph_space,
    "--gar_heads":          gar_heads,
    "--gar_dk":             gar_dk,
    "--gar_refresh_every":  gar_refresh_every,
    "--gar_ema_beta":       gar_ema_beta,

    "--gar_base_kernel":    gar_base_kernel,
    "--gar_attn_tau":       gar_attn_tau,
    "--gar_attn_gamma":     gar_attn_gamma,
    "--gar_attn_mu":        gar_attn_mu,

    "--gar_beta":           gar_beta,
    "--gar_lambda_u":       gar_lambda_u,

    "--gar_tau_strategy":   gar_tau_strategy,
    "--gar_quantile":       gar_quantile,
    "--gar_target_fpr":     gar_target_fpr,

    # Plotting / reporting (shared behavior across scripts)
    "--plot_sets":     plot_sets,
    "--plot_auto_unl": plot_auto_unl,
    "--plot_boundary": plot_boundary,
    "--select_metric": select_metric,

    # Output / CSV
    "--outdir":          outdir,
    "--grid_append_csv": grid_append_csv,
}

# =========================
# Method-aware grid pruning
# =========================
def _collapse_to_singletons(d, keys_to_keep):
    """Return ONLY the keys in keys_to_keep; drop everything else."""
    pruned = {}
    for k in keys_to_keep:
        if k in d:
            v = d[k]
            pruned[k] = v if (isinstance(v, list) and len(v) > 0) else v
    return pruned

# Flags that are generally useful regardless of method
COMMON = {
    "--datasets","--n_norm","--n_anom",
    "--rho","--epsilon","--val_frac","--test_frac","--labeled_anom_frac",
    "--seeds","--wiper (removed)_use_sessions",
    "--plot_sets","--plot_auto_unl","--plot_boundary","--select_metric",
    "--outdir","--grid_append_csv","--method"
}

METHOD_KEYS = {
    "ocsvm": COMMON | {
        "--nu","--gamma",
        "--tau_strategy_ocsvm","--target_fpr_ocsvm"
    },
    "gar_dsvdd": {
        # Data & shared deep
        "--datasets","--n_norm","--n_anom","--rho","--epsilon","--val_frac","--test_frac",
        "--labeled_anom_frac","--seeds","--wiper (removed)_use_sessions",
        "--latent","--epochs_main","--lr","--wd",
        # GAR-specific flags (scoped)
        "--gar_alpha","--gar_k","--gar_graph","--gar_graph_space",
        "--gar_heads","--gar_dk","--gar_refresh_every","--gar_ema_beta",
        "--gar_base_kernel","--gar_attn_tau","--gar_attn_gamma","--gar_attn_mu",
        "--gar_beta","--gar_lambda_u",
        "--gar_tau_strategy","--gar_quantile","--gar_target_fpr",
        # NEW: allow GAR to save graph artifacts too
        "--save_graph","--graph_dir",
        # Output & plotting
        "--outdir","--grid_append_csv",
        "--plot_sets","--plot_auto_unl","--plot_boundary","--select_metric",
    },
    "gar_dsvdd": COMMON | {
        "--latent","--epochs_warm","--epochs_main","--lr","--wd",
        "--graph","--graph_space","--k","--alpha","--mu","--heads","--dk",
        "--refresh_every","--ema_beta","--save_graph","--graph_dir",
        "--tau_strategy_ours","--quantile_ours","--target_fpr_ours"
    },
    "deepsvdd": COMMON | {
        "--latent","--epochs_main",
        "--deepsvdd_objective","--deepsvdd_nu",
        "--deepsvdd_pretrain_epochs","--deepsvdd_batch_size",
        "--deepsvdd_lr","--deepsvdd_wd","--deepsvdd_train_on",
        "--tau_strategy_deepsvdd","--quantile_deepsvdd","--target_fpr_deepsvdd"
    },
    "deepsad": COMMON | {
        "--latent","--epochs_main",
        "--sad_use_unlabeled","--sad_use_lab_anom",
        "--sad_margin","--sad_eta","--sad_lr","--sad_wd","--sad_batch_size",
        "--tau_strategy_deepsad","--quantile_deepsad","--target_fpr_deepsad"
    },
    "svdd_qp": COMMON | {
        "--nu","--rbf_gamma",
        "--tau_strategy_svdd_qp","--quantile_svdd_qp","--target_fpr_svdd_qp"
    },
    "s3svdd_qp": COMMON | {
        "--laplacian_type",
        "--s3_k","--s3_mu","--s3_rbf_gamma","--s3_nu",
        "--tau_strategy_s3svdd_qp","--quantile_s3svdd_qp","--target_fpr_s3svdd_qp"
    },
    "all": set(PARAMS.keys())
}

# =========================================
# Helpers for aggregation
# =========================================
def _list_run_dirs():
    return set(glob.glob(os.path.join(OUTDIR_ABS, "run_*")))

def _find_new_run_dirs(before):
    after = _list_run_dirs()
    return sorted(list(after - before))

def _append_csv_rows(src_csv, writer, wrote_header_flag):
    with open(src_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if (not wrote_header_flag[0]) and header:
            writer.writerow(header)
            wrote_header_flag[0] = True
        for row in r:
            writer.writerow(row)

def _stringify(v):
    """Coerce CLI values to strings, preserving 'None' as an omitted token."""
    if v is None:
        return "None"
    return str(v)

# =========================================
# Main launch + aggregate  (per-method grids)
# =========================================
def main():
    os.makedirs(OUTDIR_ABS, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_csv = os.path.join(OUTDIR_ABS, f"grid_all_runs_{ts}.csv")
    aggregated_dirs = set()  # prevent duplicate appends if a run dir reappears

    print("[runner] PROJECT_ROOT:", PROJECT_ROOT)
    print("[runner] SCRIPT_PATH :", SCRIPT_PATH)
    print("[runner] GAR_PATH   :", GAR_PATH)
    print("[runner] OUTDIR_ABS  :", OUTDIR_ABS)
    print("[runner] MASTER CSV  :", master_csv)

    wrote_header = [False]
    seen_signatures = set()

    # Expand methods: if 'all' is present, run each method separately to avoid cross-multiplication
    requested = [str(m).lower() for m in PARAMS["--method"]]
    if "all" in requested:
        methods_to_run = ["gar_dsvdd", "gar_dsvdd", "deepsvdd", "deepsad", "ocsvm", "svdd_qp", "s3svdd_qp"]
    else:
        methods_to_run = requested

    try:
        with open(master_csv, "w", encoding="utf-8", newline="") as master_fp:
            master_wr = csv.writer(master_fp)

            for m_lower in methods_to_run:
                # 1) Start from method-aware singleton collapse
                keep = METHOD_KEYS.get(m_lower, set(PARAMS.keys()))
                subparams = _collapse_to_singletons(PARAMS, keep)
                # 2) Force the method flag itself (baseline script only).
                #    GAR-DSVDD script does NOT accept --method.
                if m_lower != "gar_dsvdd":
                    subparams["--method"] = [m_lower]

                # 3) Defensive de-dup of overlapping knobs
                if m_lower == "s3svdd_qp":
                    if "--s3_rbf_gamma" in subparams and "--rbf_gamma" in subparams:
                        del subparams["--rbf_gamma"]
                    if "--s3_nu" in subparams and "--nu" in subparams:
                        del subparams["--nu"]
                elif m_lower == "svdd_qp":
                    for k_drop in ("--s3_k", "--s3_mu", "--s3_rbf_gamma", "--s3_nu"):
                        if k_drop in subparams:
                            del subparams[k_drop]
                elif m_lower == "ocsvm":
                    # Ensure no attention extras leak into OCSVM (paranoia)
                    for k_drop in ("--gar_attn_gamma","--gar_attn_tau","--gar_attn_mu","--gar_base_kernel"):
                        if k_drop in subparams:
                            del subparams[k_drop]

                # Build cartesian product for THIS method only
                keys = list(subparams.keys())
                vals = [subparams[k] for k in keys]

                # Visible combo count
                combo_count = 1
                for vlist in vals:
                    combo_count *= max(1, len(vlist))
                print(f"\n[runner] Method '{m_lower}': planned combinations = {combo_count}")

                for combo in itertools.product(*vals):
                    combo_args, desc_bits = [], []
                    for k_, v_ in zip(keys, combo):
                        # Omit None values entirely
                        if v_ is None or (isinstance(v_, str) and v_.lower() == "none"):
                            continue
                        v_str = "1" if isinstance(v_, bool) and v_ else ("0" if isinstance(v_, bool) else str(v_))
                        combo_args += [k_, v_str]
                        desc_bits.append(f"{k_.lstrip('-')}={v_str}")

                    signature = (m_lower, tuple(combo_args))
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)

                    combo_desc = ", ".join(desc_bits)
                    print("\n>>> Running combo:", combo_desc)

                    script_for_method = GAR_PATH if m_lower == "gar_dsvdd" else SCRIPT_PATH
                    cmd = [sys.executable, script_for_method]
    
    # Forward seed_list if provided
    if SEED_LIST:
        cmd.extend(['--seed_list', SEED_LIST]) + combo_args
                    print(">>> Command:", " ".join(shlex.quote(x) for x in cmd))

                    before = _list_run_dirs()
                    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)

                    if result.stdout:
                        print(result.stdout, end="")
                    if result.returncode != 0:
                        print(f"----- STDERR from {os.path.basename(script_for_method)} -----")
                        if result.stderr:
                            print(result.stderr, end="")
                        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

                    # Aggregate newly created run_* dirs once
                    new_dirs = _find_new_run_dirs(before)
                    for run_dir in new_dirs:
                        per_csv = os.path.join(run_dir, "all_runs.csv")
                        if per_csv and os.path.isfile(per_csv):
                            if run_dir not in aggregated_dirs:
                                _append_csv_rows(per_csv, master_wr, wrote_header)
                                aggregated_dirs.add(run_dir)
                                print(">>> Aggregated:", per_csv)
                        else:
                            print(">>> (skip) No all_runs.csv in:", run_dir)

        print("[runner] Master grid CSV:", master_csv)

    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user. Partial results saved:", master_csv)
        raise
    except Exception as e:
        print(f"[runner] ERROR: {e}\nPartial results saved: {master_csv}")
        raise


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except Exception:
        pass
    main()
