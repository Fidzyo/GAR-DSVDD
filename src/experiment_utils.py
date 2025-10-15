from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:18:06 2025

@author: Taha
"""

# -*- coding: utf-8 -*-
"""
GAR-DSVDD (GAR-DSVDD-style)

Key training pieces (this version):
- Score f(x) = ||z - c||^2 - m   with m = softplus(eta) (learnable), c fixed to labeled-normal mean.
- Loss
    L = mean(d2_lab) + alpha * mean(d2_unl)                       # center loss on (lab + α·unl)
        + lambda_u * mean_i Σ_j \hat{w}_{ij} (d2_i - d2_j)^2      # smooth raw distances d2 (not f)
        + beta * ||c||^2                                          # c is fixed; term kept for parity/logging
  where d2 = ||z - c||^2 and \hat{w} rows are normalized.

- W is a convex blend of base kNN affinity and score-aware attention:
    e_ij = (q_i^T k_j)/sqrt(dk) - attn_gamma * (ReLU(f_i) + ReLU(f_j))
    W_attn = softmax(e_ij / attn_tau)   (rowwise over k-NN)
    W = (1 - attn_mu) * W_base + attn_mu * W_attn
  Attention uses f only to *downweight* high-scoring nodes when building the graph,
  but the training smoother operates on d2.

- Laplacian smoother uses ROW-NORMALIZED weights to avoid density bias:
    smooth = mean_i Σ_j \hat{w}_{ij} (d2_i - d2_j)^2

Notes:
- c is NOT optimized (DeepSVDD style).
- m is used for scoring and attention gating; it does not appear in the loss anymore.
"""

import os, sys, math, json, argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

# ===== import shared pipeline pieces from paper1_v20 to guarantee parity =====
try:
    from paper1_v20 import (
        dev, set_seed, ensure_dir, args_to_dict, make_run_id, format_table,
        _coerce_to_2d_features, Split, make_dataset,
        build_splits_by_session, build_splits,
        partial_auc_at_fpr, compute_confusion, eval_metrics,
        _pretty_method_title, plot_panel, _save_graph_artifacts,
        save_and_show, aggregate_seed_tables, print_seed_table,
        write_master_csv, append_global_long_csv, report_best_by_method,
    )
except Exception:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def set_seed(s:int=42):
        np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    def ensure_dir(p:str): os.makedirs(p, exist_ok=True)
    def args_to_dict(ns): return vars(ns).copy()
    import hashlib
    def make_run_id(args):
        blob = json.dumps(args_to_dict(args), sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:10]
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    def partial_auc_at_fpr(y_true,scores,max_fpr=0.01):
        fpr,tpr,_=roc_curve(y_true,scores,pos_label=1)
        idx=np.searchsorted(fpr,max_fpr,'right')
        fpr_seg=np.r_[fpr[:idx],[max_fpr]]
        tpr_seg=np.r_[tpr[:idx],[np.interp(max_fpr,fpr,tpr)]]
        area=np.trapz(tpr_seg,fpr_seg)
        return float(area/max_fpr) if max_fpr>0 else float('nan')
    def compute_confusion(y_true,scores,tau):
        y_pred=(scores>tau).astype(int)
        TP=int(np.sum((y_true==1)&(y_pred==1)))
        FP=int(np.sum((y_true==0)&(y_pred==1)))
        TN=int(np.sum((y_true==0)&(y_pred==0)))
        FN=int(np.sum((y_true==1)&(y_pred==0)))
        return dict(TP=TP,FP=FP,TN=TN,FN=FN)
    def eval_metrics(y_true,scores,tau)->Dict[str,float]:
        auroc=roc_auc_score(y_true,scores)
        auprc=average_precision_score(y_true,scores)
        pauc=partial_auc_at_fpr(y_true,scores,0.01)
        cm=compute_confusion(y_true,scores,tau)
        TP,FP,TN,FN=cm['TP'],cm['FP'],cm['TN'],cm['FN']
        acc=(TP+TN)/max(1,TP+FP+TN+FN)
        prec=TP/max(1,TP+FP)
        rec=TP/max(1,TP+FN)
        f1=2*prec*rec/max(1e-12,(prec+rec)) if (prec+rec)>0 else 0.0
        tnr=TN/max(1,TN+FP); bacc=0.5*(rec+tnr)
        return {"AUROC":auroc,"AUPRC":auprc,"pAUC@1FPR":pauc,"Accuracy":acc,
                "F1":f1,"DetectionRate_TPR":rec,"BalancedAcc":bacc,
                "TP":TP,"FP":FP,"TN":TN,"FN":FN}
    def plot_panel(*a, **k): pass
    def _save_graph_artifacts(*a, **k): pass
    def save_and_show(fig, path:str): fig.savefig(path, dpi=800, bbox_inches="tight"); plt.close(fig)
    def aggregate_seed_tables(all_results): return {}
    def print_seed_table(*a, **k): pass
    def write_master_csv(outdir, rows): pass
    def append_global_long_csv(path, run_id, rows): pass
    def report_best_by_method(*a, **k): pass

# ============================
# GAR-DSVDD model
# ============================

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=(64,64), out_dim=16):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=True), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, out_dim, bias=False)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GAR_DSVDD(nn.Module):
    """
    Score f(x) = ||z - c||^2 - m, with m = softplus(eta).
    Loss:
      L = mean(d2_lab) + alpha * mean(d2_unl)
          + lambda_u * mean_i Σ_j \hat{w}_{ij} (d2_i - d2_j)^2
          + beta * ||c||^2

    Graph:
      - kNN neighbors in chosen space ('input' or 'latent').
      - Base affinity W_base: Gaussian kernel or constant.
      - Score-aware attention (multi-head):
          e_ij = (q_i^T k_j)/sqrt(dk) - attn_gamma*(ReLU(f_i)+ReLU(f_j))
          a_ij = softmax(e_ij / attn_tau)  (rowwise over neighbors)
      - Blend: W = (1-attn_mu)*W_base + attn_mu*a      (then row-normalized for the smoothness term)
      - Optional EMA smoothing over attention weights (ema_beta in [0,1]).
    """
    def __init__(self, in_dim:int, out_dim:int,
                 lr=5e-4, wd=1e-5, epochs:int=120,
                 k:int=9, graph:str='attention', graph_space:str='latent',
                 heads:int=8, dk:int=32, refresh_every: Optional[int]=10, ema_beta: float = 0.9,
                 beta: float = 1e-3, lambda_u: float = 0.1,
                 base_kernel: str = 'gaussian', attn_tau: float = 1.0,
                 attn_gamma: float = 0.5, attn_mu: float = 0.7):
        super().__init__()
        self.enc = MLP(in_dim, (64,64), out_dim).to(dev)
        self.c = None  # fixed DeepSVDD center: mean of labeled normals

        self.lr, self.wd, self.epochs = float(lr), float(wd), int(epochs)
        self.k = int(k)
        self.graph = str(graph)                 # 'attention' | 'gaussian'
        self.graph_space = str(graph_space)     # 'input' | 'latent'
        self.refresh_every = None if (refresh_every is None or int(refresh_every) <= 0) else int(refresh_every)

        # Loss + graph knobs
        self.beta = float(beta)
        self.lambda_u = float(lambda_u)
        self.base_kernel = str(base_kernel)     # 'gaussian' | 'constant'
        self.attn_tau = float(attn_tau)         # temperature
        self.attn_gamma = float(attn_gamma)     # score penalty weight
        self.attn_mu = float(attn_mu)           # base/attention blend
        self.ema_beta = float(ema_beta)         # EMA smoothing for attention

        # attention params
        self.heads, self.dk = int(heads), int(dk)
        if self.graph == 'attention':
            self.WQ = nn.ModuleList([nn.Linear(out_dim, self.dk, bias=False) for _ in range(self.heads)]).to(dev)
            self.WK = nn.ModuleList([nn.Linear(out_dim, self.dk, bias=False) for _ in range(self.heads)]).to(dev)

        # learnable margin param  (m = softplus(eta))
        self.eta = nn.Parameter(torch.tensor(0.0, device=dev))
        self.softplus = nn.Softplus()

        # runtime caches
        self.nn_idx: Optional[np.ndarray] = None
        self.W_base: Optional[np.ndarray] = None
        self.train_cache: Optional[np.ndarray] = None
        self.n_lab = 0
        self._attn_w_ema: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _init_center(self, X_lab_t: torch.Tensor):
        Z = self.enc(X_lab_t).detach()
        self.c = Z.mean(0)

    @torch.no_grad()
    def _embed(self, X_np: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(X_np.astype(np.float32)).to(dev)
        return self.enc(Xt).detach().cpu().numpy()

    def _build_knn(self, X_space: np.ndarray):
        N = X_space.shape[0]
        k_eff = min(max(1, self.k), max(1, N-1))
        if k_eff != self.k: self.k = k_eff
        nn = NearestNeighbors(n_neighbors=self.k+1, metric='euclidean').fit(X_space)
        d, idx = nn.kneighbors(X_space, return_distance=True)
        self.nn_idx = idx[:, 1:]          # drop self
        d = d[:, 1:]
        if self.base_kernel == 'gaussian':
            sigma = max(1e-6, np.median(d))
            self.W_base = np.exp(-(d**2)/(2*sigma**2))
        else:
            self.W_base = np.ones_like(d, dtype=np.float32)

    def _maybe_refresh(self, epoch: int):
        if (self.graph_space == 'latent') and (self.refresh_every is not None) and ((epoch+1) % self.refresh_every == 0):
            X_space = self._embed(self.train_cache)
            self._build_knn(X_space)

    def _attention_weights(self, Z_all: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Score-aware, temperature-scaled, row-normalized attention with optional EMA smoothing."""
        idx = torch.from_numpy(self.nn_idx.astype(np.int64)).to(Z_all.device)  # (N, k)
        Z_nei = Z_all[idx]                                                     # (N, k, d)

        f_pos_i = torch.relu(f).unsqueeze(1)           # (N,1)
        f_pos_j = torch.relu(f[idx])                   # (N,k)
        damp = self.attn_gamma * (f_pos_i + f_pos_j)   # (N,k)

        scale = 1.0 / math.sqrt(self.dk)
        acc = None
        for h in range(self.heads):
            q = self.WQ[h](Z_all)                      # (N, dk)
            k = self.WK[h](Z_nei)                      # (N, k, dk)
            logits = (q.unsqueeze(1) * k).sum(-1) * scale
            logits = (logits - damp) / max(1e-6, self.attn_tau)
            w = torch.softmax(logits, dim=1)           # (N, k), rows sum to 1
            acc = w if acc is None else acc + w

        W_attn = acc / self.heads                      # (N, k)

        # EMA smoothing on the weights (stabilizes training)
        if self._attn_w_ema is None:
            self._attn_w_ema = W_attn.detach()
        else:
            self._attn_w_ema = self.ema_beta * self._attn_w_ema + (1 - self.ema_beta) * W_attn.detach()
        return self._attn_w_ema.clamp_min(0.0)

    def export_graph(self):
        if self.nn_idx is None:
            space = self.train_cache if self.graph_space == 'input' else self._embed(self.train_cache)
            self._build_knn(space)
        with torch.no_grad():
            X_all_t = torch.from_numpy(self.train_cache.astype(np.float32)).to(dev)
            Z = self.enc(X_all_t)
            d2 = ((Z - self.c)**2).sum(1)
            m  = self.softplus(self.eta)
            f  = d2 - m
            if self.graph == 'attention':
                W_attn = self._attention_weights(Z, f)
                W = (1 - self.attn_mu) * torch.from_numpy(self.W_base).to(W_attn) + self.attn_mu * W_attn
                W = W.detach().cpu().numpy()
            else:
                W = self.W_base
        coords = self.train_cache[:, :2] if self.graph_space == 'input' else self._embed(self.train_cache)[:, :2]
        return self.nn_idx, W, coords, ('attention' if self.graph == 'attention' else 'gaussian')

    def fit(self, X_lab_norm: np.ndarray, X_unl: np.ndarray, alpha: float = 0.3):
        """
        GAR-DSVDD-style training:
          L = mean(d2_lab) + alpha * mean(d2_unl) + lambda_u * Σ_ij \hat{w}_{ij} (d2_i - d2_j)^2 + beta ||c||^2
        c is fixed to the labeled-normal mean. Graph W is attention-based with score-aware damping,
        but the smoother itself operates on d2 (not f).
        """
        X_all = np.vstack([X_lab_norm, X_unl]).astype(np.float32)
        self.train_cache = X_all
        self.n_lab = X_lab_norm.shape[0]
        alpha = float(alpha)

        X_lab_t = torch.from_numpy(X_lab_norm.astype(np.float32)).to(dev)
        X_all_t = torch.from_numpy(X_all).to(dev)

        # Fixed center
        self._init_center(X_lab_t)

        # Optimizer
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # Initial kNN space
        space = X_all if self.graph_space == 'input' else self._embed(X_all)
        self._build_knn(space)

        for ep in range(self.epochs):
            self.train(); opt.zero_grad()

            Z  = self.enc(X_all_t)                           # (N, d)
            d2 = ((Z - self.c)**2).sum(1)                    # (N,)

            # center + α-pull (GAR-DSVDD)
            d2_lab = d2[:self.n_lab]
            d2_unl = d2[self.n_lab:]
            L_center = d2_lab.mean() + (alpha * d2_unl.mean() if d2_unl.numel() else 0.0)

            # attention weights use f only for damping; training smooths d2
            with torch.no_grad():
                m = self.softplus(self.eta)
                f = d2 - m
            if self.graph == 'attention':
                W_attn = self._attention_weights(Z, f)                 # (N, k)
                W_base = torch.from_numpy(self.W_base).to(W_attn)      # (N, k)
                W = (1.0 - self.attn_mu) * W_base + self.attn_mu * W_attn
            else:
                W = torch.from_numpy(self.W_base).to(d2)               # (N, k)

            # Row-normalize for the smoother
            Wn = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)

            idx     = torch.from_numpy(self.nn_idx.astype(np.int64)).to(d2.device)  # (N, k)
            d2_nei  = d2[idx]                                                       # (N, k)
            smooth  = (Wn * (d2.unsqueeze(1) - d2_nei)**2).sum(1).mean()

            loss = L_center + self.lambda_u * smooth + self.beta * (self.c**2).sum()
            loss.backward(); opt.step()

            self.eval()
            self._maybe_refresh(ep)

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
        Z  = self.enc(Xt)
        d2 = ((Z - self.c)**2).sum(1)
        m  = self.softplus(self.eta)
        f  = d2 - m
        return f.detach().cpu().numpy()

# =========================
# τ selection (GAR section)
# =========================

def pick_tau_gar(strategy: str,
                 s_train_norm: np.ndarray,
                 s_val_norm: Optional[np.ndarray],
                 s_val_all: Optional[np.ndarray],
                 y_val_all: Optional[np.ndarray],
                 quantile: float,
                 target_fpr: float) -> float:
    from sklearn.metrics import roc_curve
    if strategy == 'zero': return 0.0
    if strategy in ('train_quantile','quantile'): return float(np.quantile(s_train_norm, quantile))
    if strategy == 'val_quantile':
        assert s_val_norm is not None, "val_quantile needs validation normals"
        return float(np.quantile(s_val_norm, quantile))
    if strategy == 'fpr':
        assert (s_val_all is not None) and (y_val_all is not None), "fpr needs full validation set"
        fpr,tpr,thr = roc_curve(y_val_all, s_val_all, pos_label=1)
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return float(thr[idx])
    if strategy == 'youden':
        assert (s_val_all is not None) and (y_val_all is not None), "youden needs full validation set"
        fpr,tpr,thr = roc_curve(y_val_all, s_val_all, pos_label=1)
        J = tpr - fpr
        return float(thr[int(np.argmax(J))])
    raise ValueError(f"Unknown tau strategy: {strategy}")

# =========================
# One full run (per dataset/seed) — matches paper1_v20
# =========================

def run_once(args, dataset: str, seed: int, run_dir: str, master_rows: List[Dict[str, Any]]):
    set_seed(seed)

    # ----- load / split -----
    if dataset == 'wiper':
        Xp, yp, sp = "wiper_clip_X.npy", "wiper_clip_y.npy", "wiper_clip_sessions.npy"
        assert os.path.isfile(Xp) and os.path.isfile(yp), "Missing wiper_clip_X/y.npy"
        X_raw = np.load(Xp, allow_pickle=True); y = np.load(yp).astype(int)
        X = _coerce_to_2d_features(X_raw)
        if args.wiper_use_sessions:
            assert os.path.isfile(sp), "Missing wiper_clip_sessions.npy"
            sessions = np.load(sp)
            split = build_splits_by_session(
                X, y, sessions,
                rho=args.rho, epsilon=args.epsilon, seed=seed,
                val_frac=args.val_frac, test_frac=args.test_frac,
                labeled_anom_frac=args.labeled_anom_frac
            )
        else:
            split = build_splits(
                X, y, rho=args.rho, epsilon=args.epsilon, seed=seed,
                val_frac=args.val_frac, test_frac=args.test_frac,
                labeled_anom_frac=args.labeled_anom_frac
            )
    else:
        X, y = make_dataset(dataset, n_norm=args.n_norm, n_anom=args.n_anom, seed=seed)
        split = build_splits(
            X, y, rho=args.rho, epsilon=args.epsilon, seed=seed,
            val_frac=args.val_frac, test_frac=args.test_frac,
            labeled_anom_frac=args.labeled_anom_frac
        )

    # ----- build & train GAR-DSVDD -----
    in_dim = int(split.X_lab_norm.shape[1])
    gar = GAR_DSVDD(
        in_dim=in_dim, out_dim=args.latent,
        lr=args.lr, wd=args.wd, epochs=args.epochs_main,
        k=args.k, graph=args.graph, graph_space=args.graph_space,
        heads=args.heads, dk=args.dk, refresh_every=args.refresh_every, ema_beta=args.ema_beta,
        beta=args.beta, lambda_u=args.lambda_u,
        base_kernel=args.base_kernel, attn_tau=args.attn_tau,
        attn_gamma=args.attn_gamma, attn_mu=args.attn_mu
    ).to(dev)

    # α is now USED by the loss
    gar.fit(split.X_lab_norm, split.X_unl, alpha=args.alpha)

    # ----- τ -----
    s_train = gar.score(split.X_lab_norm)
    s_val_n = gar.score(split.X_val[split.y_val == 0]) if args.tau_strategy_gar in ('val_quantile',) else None
    s_val   = gar.score(split.X_val) if args.tau_strategy_gar in ('fpr','youden','val_quantile') else None
    tau = pick_tau_gar(
        strategy=args.tau_strategy_gar,
        s_train_norm=s_train, s_val_norm=s_val_n, s_val_all=s_val, y_val_all=(split.y_val if s_val is not None else None),
        quantile=args.quantile_gar, target_fpr=args.target_fpr_gar
    )

    # ----- test metrics -----
    test_scores = gar.score(split.X_test)
    metrics = eval_metrics(split.y_test, test_scores, tau)

    # ----- plots (only if 2-D) -----
    X_all = np.vstack([split.X_lab_norm, split.X_unl, split.X_val, split.X_test])
    do_plots = (X_all.shape[1] == 2)
    if do_plots:
        fig_train, ax_train = plt.subplots(1, 1, figsize=(5.0, 4.2))
        ax_train.scatter(split.X_lab_norm[:,0], split.X_lab_norm[:,1], s=28, c='#2ca02c', marker='o',
                         label='Labeled Normal', edgecolors='k', linewidths=0.4)
        if split.X_lab_anom.size:
            ax_train.scatter(split.X_lab_anom[:,0], split.X_lab_anom[:,1], s=36, c='#d62728', marker='*',
                             label='Labeled Anomaly', edgecolors='k', linewidths=0.5)
        if split.X_unl.size:
            unl_n = split.X_unl[split.y_unl_true==0]; unl_a = split.X_unl[split.y_unl_true==1]
            if len(unl_n): ax_train.scatter(unl_n[:,0], unl_n[:,1], s=18, facecolors='none', edgecolors='#1f77b4', marker='o', label='Unlabeled (true normal)')
            if len(unl_a): ax_train.scatter(unl_a[:,0], unl_a[:,1], s=18, facecolors='none', edgecolors='#ff7f0e', marker='o', label='Unlabeled (true anomaly)')
        ax_train.set_title("Training data", fontsize=14)
        ax_train.set_xlabel(r"$x_1$"); ax_train.set_ylabel(r"$x_2$")
        ax_train.legend(loc='upper right', fontsize=7, framealpha=0.8)
        fig_train.tight_layout()
        out_train = os.path.join(run_dir, f"train_{dataset}_seed{seed}.png")
        save_and_show(fig_train, out_train)

        fig_score, ax_s = plt.subplots(1, 1, figsize=(5.0, 4.2))
        plot_sets = list(args.plot_sets)
        if args.plot_auto_unl and ('unlabeled' not in plot_sets): plot_sets.append('unlabeled')
        plot_panel(ax_s, gar, X_all, split, tau,
                   title="GAR-DSVDD — anomaly score",
                   plot_sets=plot_sets,
                   show_boundary=bool(args.plot_boundary),
                   mode='score')
        fig_score.tight_layout()
        out_score = os.path.join(run_dir, f"grid_{dataset}_seed{seed}_SCORE.png")
        save_and_show(fig_score, out_score)

        fig_quant, ax_q = plt.subplots(1, 1, figsize=(5.0, 4.2))
        ref_scores = s_train if args.tau_strategy_gar in ('train_quantile','zero','quantile') else gar.score(split.X_val)
        plot_panel(ax_q, gar, X_all, split, tau,
                   title="GAR-DSVDD — anomaly score quantile",
                   plot_sets=plot_sets,
                   show_boundary=bool(args.plot_boundary),
                   mode='quantile', ref_scores=ref_scores)
        fig_quant.tight_layout()
        out_quant = os.path.join(run_dir, f"grid_{dataset}_seed{seed}_QUANT_SYNC.png")
        save_and_show(fig_quant, out_quant)

        if int(args.save_graph):
            gdir = os.path.join(run_dir, args.graph_dir); ensure_dir(gdir)
            _save_graph_artifacts(gar, gdir, f'{dataset}_seed{seed}', top_m=2)

        print(f"[plot] saved {out_score}")
        print(f"[plot] saved {out_quant}")
        print(f"[plot] saved {out_train}")
    else:
        print("[plot] skipped (feature dim != 2)")

    # ----- per-seed CSV -----
    per_csv = os.path.join(run_dir, f"results_{dataset}_seed{seed}.csv")
    if not os.path.isfile(per_csv):
        with open(per_csv, "w", encoding="utf-8") as f:
            f.write("dataset,seed,method,tau,AUROC,AUPRC,pAUC@1FPR,Accuracy,F1,DetectionRate_TPR,"
                    "BalancedAcc,TP,FP,TN,FN,params_json\n")
    with open(per_csv, "a", encoding="utf-8") as f:
        f.write(f"{dataset},{seed},GAR-DSVDD,{tau:.6f},{metrics['AUROC']:.6f},{metrics['AUPRC']:.6f},"
                f"{metrics['pAUC@1FPR']:.6f},{metrics['Accuracy']:.6f},{metrics['F1']:.6f},"
                f"{metrics['DetectionRate_TPR']:.6f},{metrics['BalancedAcc']:.6f},"
                f"{metrics['TP']},{metrics['FP']},{metrics['TN']},{metrics['FN']},"
                f"{json.dumps(args_to_dict(args), sort_keys=True)}\n")

    row = {"dataset": dataset, "seed": seed, "method": "GAR-DSVDD", "tau": float(tau)}
    row.update(metrics); row["params"] = args_to_dict(args)
    master_rows.append(row)
    return {"GAR-DSVDD": metrics}

# =========================
# CLI / main — mirrors paper1_v20
# =========================

def parse_args():
    p = argparse.ArgumentParser()

    # ----- data / splits -----
    p.add_argument('--datasets', type=str, default='moons,blobs,rings,tricls')
    p.add_argument('--n_norm', type=int, default=1200)
    p.add_argument('--n_anom', type=int, default=400)
    p.add_argument('--rho', type=float, default=0.1)
    p.add_argument('--epsilon', type=float, default=0.05)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--test_frac', type=float, default=0.2)
    p.add_argument('--labeled_anom_frac', type=float, default=0.0)
    
    # accepted for runner parity
    p.add_argument('--method', type=str, default='gar_dsvdd')

    # ----- shared / training -----
    p.add_argument('--latent', type=int, default=16)
    p.add_argument('--epochs_main', type=int, default=120)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--wd', type=float, default=1e-5)

    # ===== GAR-only (accept generic and gar_ aliases) =====
    p.add_argument('--alpha', '--gar_alpha', dest='alpha', type=float, default=0.3,
                   help="Weight on unlabeled center pull (mean(d^2) over unlabeled).")
    p.add_argument('--k', '--gar_k', dest='k', type=int, default=9,
                   help="k for k-NN graph.")

    p.add_argument('--graph', '--gar_graph', dest='graph',
                   type=str, choices=['attention','gaussian'], default='attention',
                   help="Graph type: 'attention' (with base blend) or 'gaussian' (base only).")
    p.add_argument('--graph_space', '--gar_graph_space', dest='graph_space',
                   type=str, choices=['input','latent'], default='latent',
                   help="Build the kNN graph in 'input' or 'latent' space.")

    p.add_argument('--heads', '--gar_heads', dest='heads', type=int, default=8,
                   help="Number of attention heads.")
    p.add_argument('--dk', '--gar_dk', dest='dk', type=int, default=32,
                   help="Per-head projection width.")

    p.add_argument('--refresh_every', '--gar_refresh_every', dest='refresh_every',
                   type=int, default=10,
                   help="Rebuild latent kNN every T epochs (0/None disables).")
    p.add_argument('--ema_beta', '--gar_ema_beta', dest='ema_beta',
                   type=float, default=0.9,
                   help="EMA smoothing for attention weights (0=no smoothing, 1=freeze).")

    p.add_argument('--base_kernel', '--gar_base_kernel', dest='base_kernel',
                   type=str, choices=['gaussian','constant'], default='gaussian',
                   help="Base affinity on kNN edges.")
    p.add_argument('--attn_tau', '--gar_attn_tau', dest='attn_tau',
                   type=float, default=1.0,
                   help="Attention softmax temperature.")
    p.add_argument('--attn_gamma', '--gar_attn_gamma', dest='attn_gamma',
                   type=float, default=0.5,
                   help="Score-penalty γ inside attention logits.")
    p.add_argument('--attn_mu', '--gar_attn_mu', dest='attn_mu',
                   type=float, default=0.7,
                   help="Blend: 0=base only, 1=attention only.")

    p.add_argument('--beta', '--gar_beta', dest='beta', type=float, default=1e-3,
                   help="Weak centering on ||c||^2 (c is fixed; term kept for parity).")
    p.add_argument('--lambda_u', '--gar_lambda_u', dest='lambda_u',
                   type=float, default=0.1,
                   help="Graph smoothness weight on d2 (not f).")

    # τ selection (accept legacy 'quantile' alias)
    p.add_argument('--tau_strategy_gar', '--gar_tau_strategy', dest='tau_strategy_gar',
                   type=str,
                   choices=['train_quantile','val_quantile','fpr','youden','zero','quantile'],
                   default='train_quantile',
                   help="Thresholding strategy. ('quantile' is alias for 'train_quantile').")
    p.add_argument('--quantile_gar', '--gar_quantile', dest='quantile_gar',
                   type=float, default=0.95,
                   help="Quantile q when using a *quantile strategy.")
    p.add_argument('--target_fpr_gar', '--gar_target_fpr', dest='target_fpr_gar',
                   type=float, default=0.01,
                   help="Target FPR when using strategy 'fpr'.")

    # ----- plotting / reporting parity -----
    p.add_argument('--plot_sets', type=str, default='train,unlabeled,test')
    p.add_argument('--plot_auto_unl', type=int, default=1)
    p.add_argument('--plot_boundary', type=int, default=1)
    p.add_argument('--save_graph', type=int, default=0)
    p.add_argument('--graph_dir', type=str, default='graphs')

    p.add_argument('--select_metric', type=str, default='AUROC',
                   choices=['AUROC','AUPRC','pAUC@1FPR','Accuracy','F1','DetectionRate_TPR','BalancedAcc'])

    # ----- run control / output -----
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--outdir', type=str, default='plots')
    p.add_argument('--grid_append_csv', type=str, default='')

    # Keep ipykernel behavior identical to before
    args = p.parse_args(args=[]) if 'ipykernel' in sys.modules else p.parse_args()

    # Normalize legacy alias: 'quantile' → 'train_quantile'
    if getattr(args, 'tau_strategy_gar', None) == 'quantile':
        args.tau_strategy_gar = 'train_quantile'

    # Normalize refresh_every: <=0 → None
    if (args.refresh_every is None) or (isinstance(args.refresh_every, int) and args.refresh_every <= 0):
        args.refresh_every = None

    return args


def main():
    args = parse_args()

    args.plot_sets = [s.strip() for s in args.plot_sets.split(',') if s.strip()]
    if (args.refresh_every is None) or (isinstance(args.refresh_every, int) and args.refresh_every <= 0):
        args.refresh_every = None

    print("Device:", dev)
    print("Args:", args)

    dataset_list = [s.strip() for s in args.datasets.split(',') if s.strip()]
    run_id = make_run_id(args)
    param_dir = os.path.join(args.outdir, f"run_{run_id}")
    ensure_dir(param_dir)

    with open(os.path.join(param_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(args_to_dict(args), f, indent=2, sort_keys=True)

    master_rows = []
    agg_all_datasets = {}

    for ds in dataset_list:
        seed_results = []
        for s in range(args.seeds):
            run_dir_ds_seed = os.path.join(param_dir, f"{ds}_seed{s}")
            ensure_dir(run_dir_ds_seed)
            res = run_once(args, ds, seed=s, run_dir=run_dir_ds_seed, master_rows=master_rows)
            print_seed_table(res, seed=s, dataset=ds)
            seed_results.append(res)

        agg = aggregate_seed_tables(seed_results)
        agg_all_datasets[ds] = agg

    write_master_csv(param_dir, master_rows)
    report_best_by_method(param_dir, master_rows, args.select_metric)

    with open(os.path.join(param_dir, "aggregated.json"), "w", encoding="utf-8") as f:
        json.dump(agg_all_datasets, f, indent=2, sort_keys=True)

    if isinstance(args.grid_append_csv, str) and args.grid_append_csv.strip():
        try:
            append_global_long_csv(args.grid_append_csv, run_id, master_rows)
        except Exception as e:
            print(f"[warn] failed to append global CSV ({args.grid_append_csv}): {e}")

if __name__ == "__main__":
    main()
