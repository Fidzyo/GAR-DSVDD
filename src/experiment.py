
# ONLY_BLOBS_GUARD
if isinstance(dataset, str) and dataset != 'blobs':
    raise ValueError('This repository supports only the blobs dataset.')

from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Experiment Suite — Blobs-only, semi-supervised anomaly detectionnchmarks (v14, blobs-only, partial labels, global CSV append)

Key behavior:
• GAR-DSVDD(attention/gaussian graph): semi-supervised on (labeled normals ∪ unlabeled).
• DeepSVDD / OCSVM / SVDD / S3SVDD: train ONLY on labeled normals (independent of ε).
• DeepSVDD flag deepsvdd_train_on is accepted but forced to 'lab_only' (console warning if not).
• 
Tau strategy:
• Each method has its own τ strategy and params.
  - GAR-DSVDD, deepsvdd,   - ocsvm: τ = zero   (because score = -decision_function; native boundary at 0)
• Other options: val_quantile | fpr | youden | zero

wiper (removed) dataset: (removed)
• Loads per-clip .npy files: wiper (removed)_clip_X/y/sessions.npy
• X may be (N, D), (N, T, F), or object array of variable-length (Ti, F).
• Session-aware splits when --wiper (removed)_use_sessions=1 (default).

Outputs:
• Per-seed CSVs, per-run master CSV with wide columns.
• Optional global LONG CSV appender:
    --grid_append_csv /abs/path/all_runs_long.csv
"""

import os, sys, math, argparse, json, hashlib
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve)
from sklearn.svm import OneClassSVM

import torch
import torch.nn as nn
import torch.optim as optim

# ========= Utils & IO =========

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return vars(args).copy()

def make_run_id(args: argparse.Namespace) -> str:
    keys = [
        # core / GAR-DSVDD
        "method","graph","graph_space","alpha","mu","k","heads","dk","refresh_every","ema_beta",
        "latent","epochs_warm","epochs_main","lr","wd",

        # DeepSVDD (author-style)
        "deepsvdd_objective","deepsvdd_nu","deepsvdd_pretrain_epochs",
        "deepsvdd_batch_size","deepsvdd_lr","deepsvdd_wd","deepsvdd_train_on",

        #         "sad_use_unlabeled","sad_use_lab_anom","sad_margin","sad_eta","sad_lr","sad_wd","sad_batch_size",

        # OCSVM / QP
        "nu","gamma","rbf_gamma","laplacian_type",

        # tau
        "tau_strategy_GAR-DSVDD","tau_strategy_deepsvdd","tau_strategy_        "tau_strategy_ocsvm","tau_strategy_svdd","tau_strategy_s3svdd",
        "quantile_GAR-DSVDD","quantile_deepsvdd","quantile_        "target_fpr_GAR-DSVDD","target_fpr_deepsvdd","target_fpr_        "target_fpr_ocsvm","target_fpr_svdd","target_fpr_s3svdd",

        # io / data
        "save_graph","graph_dir","wiper (removed)_use_sessions","datasets","rho","epsilon","val_frac","test_frac","labeled_anom_frac"
    ]
    blob = "|".join(f"{k}={getattr(args,k,None)}" for k in keys if hasattr(args,k))
    return hashlib.md5(blob.encode()).hexdigest()[:10]

def format_table(rows: List[List[Any]], headers: List[str], floatfmt: str=".4f") -> str:
    cols = len(headers)
    widths = [len(h) for h in headers]
    fmt_rows = []
    for r in rows:
        fr=[]
        for i,v in enumerate(r):
            s = f"{v:{floatfmt}}" if isinstance(v,float) else str(v)
            widths[i] = max(widths[i], len(s)); fr.append(s)
        fmt_rows.append(fr)
    sep=" | "
    head = sep.join(h.ljust(widths[i]) for i,h in enumerate(headers))
    dash = "-+-".join("-"*w for w in widths)
    body = [sep.join(fmt_rows[i][j].ljust(widths[j]) for j in range(cols)) for i in range(len(fmt_rows))]
    return "\n".join([head,dash]+body)

# --------- Feature coercion (handles whole-recording MFCCs) ---------

def _coerce_to_2d_features(X):
    """
    Accepts:
      - 2D array: (N, D) -> returned as float32
      - 3D array: (N, T, F) -> [mean, std] pooled across time -> (N, 2F)
      - object array / list of (Ti, F): per-clip temporal [mean, std] -> (N, 2F)
    """
    # list / object array path
    if isinstance(X, list) or (isinstance(X, np.ndarray) and X.dtype == object):
        X_list = list(X)
        feats = []
        for Xi in X_list:
            Xi = np.asarray(Xi, dtype=np.float32)
            if Xi.ndim == 1:
                feats.append(Xi[np.newaxis, :])  # rare
            elif Xi.ndim == 2:
                m = Xi.mean(axis=0)
                s = Xi.std(axis=0)
                feats.append(np.concatenate([m, s], axis=0)[np.newaxis, :])
            else:
                raise ValueError(f"Per-recording feature has ndim={Xi.ndim}, expected 1 or 2.")
        return np.concatenate(feats, axis=0).astype(np.float32)

    # ndarray path
    X = np.asarray(X)
    if X.ndim == 2:
        return X.astype(np.float32)
    if X.ndim == 3:
        m = X.mean(axis=1)
        s = X.std(axis=1)
        return np.concatenate([m, s], axis=1).astype(np.float32)
    raise ValueError(f"X.ndim={X.ndim} not supported; expected 2 or 3 or object-array.")

# ========= Data & Splits =========

@dataclass
class Split:
    X_lab_norm: np.ndarray; y_lab_norm: np.ndarray
    X_lab_anom: np.ndarray; y_lab_anom: np.ndarray
    X_unl: np.ndarray;     y_unl_true: np.ndarray
    X_val: np.ndarray;     y_val: np.ndarray
    X_test: np.ndarray;    y_test: np.ndarray

def make_dataset(name: str, n_norm=1000, n_anom=200, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    if name == 'moons':
        X, yb = make_moons(n_samples=n_norm+n_anom, noise=0.15, random_state=seed)
        y = yb.astype(int)
    elif name == 'rings':
        n1, n2 = n_norm, n_anom
        r1 = 1.0 + 0.05*rng.randn(n1); t1 = 2*np.pi*rng.rand(n1)
        r2 = 2.0 + 0.05*rng.randn(n2); t2 = 2*np.pi*rng.rand(n2)
        x1 = np.stack([r1*np.cos(t1), r1*np.sin(t1)],1)
        x2 = np.stack([r2*np.cos(t2), r2*np.sin(t2)],1)
        X = np.vstack([x1,x2]); y = np.r_[np.zeros(n1,int), np.ones(n2,int)]
        X += 0.02*rng.randn(*X.shape)
    elif name == 'blobs':
        X_all,y_all = make_blobs(n_samples=n_norm+n_anom,
                                 centers=[[0,0],[3.0,0.0]],
                                 cluster_std=[0.8,0.8], random_state=seed)
        X,y = X_all.astype(np.float32), y_all.astype(int)
    elif name == 'tricls':
        n1 = n_norm//2; n2 = n_norm-n1; na=n_anom
        left  = rng.randn(n1,2)*[0.4,0.4] + [-2.0, 0.0]
        right = rng.randn(n2,2)*[0.4,0.4] + [ 2.0, 0.0]
        anom  = rng.randn(na,2)*[0.35,0.35]+[0.0,1.0]
        X = np.vstack([left,right,anom]); y = np.r_[np.zeros(n1+n2,int), np.ones(na,int)]
    elif name == 'wiper (removed)':
    raise ValueError('wiper (removed) dataset removed in this repo; only "blobs" is supported.')
    else:
        raise ValueError("Unknown dataset")
    return X.astype(np.float32), y.astype(int)

def build_splits_by_session(X, y, sessions, rho, epsilon, seed=0, val_frac=0.2, test_frac=0.2, labeled_anom_frac=0.0) -> Split:
    rng = np.random.RandomState(seed)
    sess_ids = np.unique(sessions)
    rng.shuffle(sess_ids)

    n_test = int(round(len(sess_ids) * test_frac))
    n_val  = int(round(len(sess_ids) * val_frac))
    test_s = set(sess_ids[:n_test])
    val_s  = set(sess_ids[n_test:n_test+n_val])
    train_s= set(sess_ids[n_test+n_val:])

    train_mask = np.isin(sessions, list(train_s))
    val_mask   = np.isin(sessions, list(val_s))
    test_mask  = np.isin(sessions, list(test_s))

    Xtr, ytr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    rng2 = np.random.RandomState(seed)
    idx_n = np.where(ytr==0)[0]; idx_a = np.where(ytr==1)[0]
    rng2.shuffle(idx_n); rng2.shuffle(idx_a)

    n_lab = max(1, int(len(idx_n)*rho))
    lab_norm_idx = idx_n[:n_lab]
    X_lab_norm = Xtr[lab_norm_idx]; y_lab_norm = np.zeros(n_lab, int)

    n_la = max(0, int(len(idx_a)*labeled_anom_frac))
    lab_anom_idx = idx_a[:n_la]
    X_lab_anom = Xtr[lab_anom_idx]; y_lab_anom = np.ones(n_la, int)

    rem_norm_idx = idx_n[n_lab:]
    idx_a_unl = idx_a[n_la:]
    m = int(round((epsilon*len(rem_norm_idx))/max(1e-8,(1.0-epsilon))))
    m = min(m, len(idx_a_unl))
    unl_idx = np.r_[rem_norm_idx, idx_a_unl[:m]]
    X_unl = Xtr[unl_idx]; y_unl_true = ytr[unl_idx]

    return Split(X_lab_norm,y_lab_norm,X_lab_anom,y_lab_anom,
                 X_unl,y_unl_true,X_val,y_val,X_test,y_test)

def build_splits(X,y, rho, epsilon, seed=0, val_frac=0.2, test_frac=0.2, labeled_anom_frac=0.0) -> Split:
    rng = np.random.RandomState(seed)
    idx_n = np.where(y==0)[0]; idx_a = np.where(y==1)[0]
    rng.shuffle(idx_n); rng.shuffle(idx_a)
    ntn=int(len(idx_n)*test_frac); nta=int(len(idx_a)*test_frac)
    test_idx=np.r_[idx_n[:ntn], idx_a[:nta]]
    idx_n=idx_n[ntn:]; idx_a=idx_a[nta:]
    X_test,y_test=X[test_idx], y[test_idx]
    nvn=int(len(idx_n)*val_frac); nva=int(len(idx_a)*val_frac)
    val_idx=np.r_[idx_n[:nvn], idx_a[:nva]]
    idx_n=idx_n[nvn:]; idx_a=idx_a[nva:]
    X_val,y_val=X[val_idx], y[val_idx]
    n_lab=max(1,int(len(idx_n)*rho))
    lab_norm_idx=idx_n[:n_lab]
    X_lab_norm=X[lab_norm_idx]; y_lab_norm=np.zeros(n_lab,int)
    n_la=max(0,int(len(idx_a)*labeled_anom_frac))
    lab_anom_idx=idx_a[:n_la]
    X_lab_anom=X[lab_anom_idx]; y_lab_anom=np.ones(n_la,int)
    rem_norm_idx=idx_n[n_lab:]
    idx_a_unl=idx_a[n_la:]
    m=int(round((epsilon*len(rem_norm_idx))/max(1e-8,(1.0-epsilon))))
    m=min(m,len(idx_a_unl))
    unl_idx=np.r_[rem_norm_idx, idx_a_unl[:m]]
    X_unl=X[unl_idx]; y_unl_true=y[unl_idx]
    return Split(X_lab_norm,y_lab_norm,X_lab_anom,y_lab_anom,X_unl,y_unl_true,X_val,y_val,X_test,y_test)

# ========= Metrics & thresholding =========

def partial_auc_at_fpr(y_true,scores,max_fpr=0.01)->float:
    fpr,tpr,_=roc_curve(y_true,scores,pos_label=1)
    idx=np.searchsorted(fpr,max_fpr,'right')
    fpr_seg=np.r_[fpr[:idx], [max_fpr]]
    tpr_seg=np.r_[tpr[:idx], [np.interp(max_fpr,fpr,tpr)]]
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
    return {
        'GAR-DSVDD': 'GAR-DSVDD',
        'DeepSVDD': 'deepsvdd',
        'OCSVM': 'ocsvm',
        'SVDD': 'svdd',
        'S3SVDD': 's3svdd',
    }.get(name, 'GAR-DSVDD')

# --- Per-method tau selection helpers ---

def _pick_tau(strategy: str,
              scores_train_norm: np.ndarray,
              scores_val_norm: Optional[np.ndarray],
              scores_val_all: Optional[np.ndarray],
              y_val_all: Optional[np.ndarray],
              quantile: float,
              target_fpr: float) -> float:
    if strategy == "zero":
        return 0.0
    if strategy == "train_quantile":
        return float(np.quantile(scores_train_norm, quantile))
    if strategy == "val_quantile":
        assert scores_val_norm is not None, "val_quantile needs validation normals"
        return float(np.quantile(scores_val_norm, quantile))
    if strategy == "fpr":
        assert (scores_val_all is not None) and (y_val_all is not None), "fpr needs full validation set"
        fpr,tpr,thr = roc_curve(y_val_all, scores_val_all, pos_label=1)
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return float(thr[idx])
    if strategy == "youden":
        assert (scores_val_all is not None) and (y_val_all is not None), "youden needs full validation set"
        fpr,tpr,thr = roc_curve(y_val_all, scores_val_all, pos_label=1)
        J = tpr - fpr
        return float(thr[int(np.argmax(J))])
    raise ValueError(f"Unknown tau strategy: {strategy}")

def _method_key(name: str) -> str:
    return {
        'GAR-DSVDD': 'GAR-DSVDD',
        'DeepSVDD': 'deepsvdd',
        'OCSVM': 'ocsvm',
        'SVDD': 'svdd',
        'S3SVDD': 's3svdd',
    }.get(name, 'GAR-DSVDD').get(name, "GAR-DSVDD")

def get_tau_cfg_for_method(args, method_name: str):
    key = _method_key(method_name)
    strat = getattr(args, f"tau_strategy_{key}")
    q = getattr(args, f"quantile_{key}", None)
    fpr = getattr(args, f"target_fpr_{key}", None)
    return strat, q, fpr

# ========= Models =========

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=(64,64), out_dim=16):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=True), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, out_dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepSVDD:
    """
    Deep SVDD with optional soft-boundary objective.

    objective: 'oneclass' (hard) or 'soft' (soft-boundary with nu)
    nu: only used when objective == 'soft'
    """
    warned = False

    def __init__(self,
                 in_dim: int = 2,
                 hidden: Tuple[int, ...] = (64, 64),
                 out_dim: int = 16,
                 objective: str = 'oneclass',
                 nu: float = 0.1,
                 lr: float = 5e-4,
                 wd: float = 1e-5,
                 epochs: int = 120,
                 pretrain_epochs: int = 0,
                 batch_size: Optional[int] = None,
                 train_on: str = 'lab_only'):
        self.enc = MLP(in_dim, hidden, out_dim).to(dev)
        self.objective = str(objective)
        self.nu = float(nu)
        self.lr = float(lr)
        self.wd = float(wd)
        self.epochs = int(epochs)
        self.pretrain_epochs = int(pretrain_epochs)
        self.batch_size = batch_size

        self.c: Optional[torch.Tensor] = None  # center in latent space

        if train_on != 'lab_only' and not DeepSVDD.warned:
            print("[warn] DeepSVDD is forced to 'lab_only' in this pipeline; ignoring --deepsvdd_train_on")
            DeepSVDD.warned = True
        self.train_on = 'lab_only'

        # Soft-boundary radius parameter: softplus keeps R^2 >= 0
        self.R2_param = torch.nn.Parameter(torch.tensor(0.0, device=dev))
        self.softplus = torch.nn.Softplus(beta=10.0)

    @torch.no_grad()
    def _init_center(self, X_lab_t: torch.Tensor) -> None:
        Z = self.enc(X_lab_t).detach()
        self.c = Z.mean(0)

        # Initialize R^2 near a high quantile if using soft-boundary
        if self.objective == 'soft':
            d2 = ((Z - self.c) ** 2).sum(1)
            q = torch.quantile(d2, q=min(0.99, max(0.80, 1.0 - self.nu + 1e-3)))
            R2_init = float(torch.clamp(q, min=0.0).item())
            # inverse-softplus init (approx) to place softplus(R2_param) ~ R2_init
            with torch.no_grad():
                self.R2_param.copy_(torch.log(torch.exp(torch.tensor(R2_init, device=dev)) - 1.0 + 1e-6))

    def _R2(self) -> torch.Tensor:
        return self.softplus(self.R2_param)

    def _pretrain_autoencoder(self, X_lab_t: torch.Tensor) -> None:
        # Stub: add AE pretraining here if desired when self.pretrain_epochs > 0
        if self.pretrain_epochs and self.pretrain_epochs > 0:
            pass

    def fit(self, X_lab_norm: np.ndarray, X_unl_unused: np.ndarray) -> None:
        X_lab = X_lab_norm.astype(np.float32)
        Xt_lab = torch.from_numpy(X_lab).to(dev)

        params = list(self.enc.parameters())
        if self.objective == 'soft':
            params += [self.R2_param]
        opt = optim.AdamW(params, lr=self.lr, weight_decay=self.wd)

        self._pretrain_autoencoder(Xt_lab)
        self._init_center(Xt_lab)

        for _ in range(self.epochs):
            self.enc.train()
            opt.zero_grad()

            Z = self.enc(Xt_lab)
            d2 = ((Z - self.c) ** 2).sum(1)

            if self.objective == 'soft':
                R2 = self._R2()
                loss = R2 + (1.0 / max(self.nu, 1e-6)) * torch.relu(d2 - R2).mean()
            else:
                loss = d2.mean()

            loss.backward()
            opt.step()

        self.enc.eval()

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
        Z = self.enc(Xt)
        d2 = ((Z - self.c) ** 2).sum(1)
        return d2.detach().cpu().numpy()

    @torch.no_grad()
    def get_radius2(self) -> Optional[float]:
        """Return learned R^2 if using the soft-boundary objective; else None."""
        if self.objective == 'soft':
            return float(self._R2().detach().cpu().item())
        return None


class GARTrainer:
    def __init__(self,in_dim=2,hidden=(64,64),out_dim=16, lr=5e-4, wd=1e-5,
                 epochs_warm=5, epochs_main=120, k=15, alpha=0.3, mu=0.5,
                 knn_metric='euclidean', graph_type='attention', graph_space='input',
                 heads=2, dk=32, refresh_every: Optional[int]=None, ema_beta=0.9):
        self.enc=MLP(in_dim,hidden,out_dim).to(dev)
        self.c=None; self.lr=lr; self.wd=wd
        self.epochs_warm=epochs_warm; self.epochs_main=epochs_main
        self.k=int(k); self.alpha=float(alpha); self.mu=float(mu)
        self.knn_metric=knn_metric
        self.graph_type=graph_type
        self.graph_space=graph_space
        if refresh_every is None or (isinstance(refresh_every,int) and refresh_every<=0):
            self.refresh_every = None
        else:
            self.refresh_every = int(refresh_every)
        self.ema_beta=float(ema_beta)
        self.nn_idx=None
        self.nn_w=None
        self.split_n_lab=0
        self.train_cache=None
        self.heads=int(heads); self.dk=int(dk)
        if self.graph_type=='attention':
            self.WQ=nn.ModuleList([nn.Linear(out_dim,self.dk,bias=False) for _ in range(self.heads)]).to(dev)
            self.WK=nn.ModuleList([nn.Linear(out_dim,self.dk,bias=False) for _ in range(self.heads)]).to(dev)
            self.attn_w=None

    @torch.no_grad()
    def _init_center(self,X_lab: torch.Tensor):
        Z=self.enc(X_lab).detach(); self.c=Z.mean(0)

    @torch.no_grad()
    def _embed_all(self,X_np):
        Xt=torch.from_numpy(X_np.astype(np.float32)).to(dev)
        return self.enc(Xt).detach().cpu().numpy()

    def _build_knn_from_space(self, X_space: np.ndarray):
        # Clamp k to valid range to avoid errors on tiny datasets
        k_eff = min(max(1, self.k), max(1, X_space.shape[0]-1))
        if k_eff != self.k:
            self.k = k_eff
        nn = NearestNeighbors(n_neighbors=self.k+1, metric=self.knn_metric).fit(X_space)
        d,idx = nn.kneighbors(X_space, return_distance=True)
        self.nn_idx=idx[:,1:]
        d=d[:,1:]
        if self.graph_type=='gaussian':
            sigma=max(1e-6, np.median(d))
            self.nn_w=np.exp(-(d**2)/(2*sigma**2))
        else:
            self.nn_w=None

    def _maybe_refresh(self, epoch: int, Z_latent_now: np.ndarray):
        if (self.graph_space=='latent') and (self.refresh_every is not None):
            if (epoch+1) % self.refresh_every == 0:
                self._build_knn_from_space(Z_latent_now)

    def _compute_attention(self,Z_all: torch.Tensor)->torch.Tensor:
        idx=torch.from_numpy(self.nn_idx.astype(np.int64)).to(dev)
        Z_nei=Z_all[idx]
        acc=None; scale=1.0/math.sqrt(self.dk)
        for h in range(self.heads):
            q=self.WQ[h](Z_all)
            k=self.WK[h](Z_nei)
            s=(q.unsqueeze(1)*k).sum(-1)*scale
            w=torch.softmax(s,dim=1)
            acc = w if acc is None else acc+w
        W = acc/self.heads
        self.attn_w = W.detach() if self.attn_w is None else self.ema_beta*self.attn_w + (1-self.ema_beta)*W.detach()
        return self.attn_w

    def export_graph(self):
        kind = self.graph_type
        idx = self.nn_idx
        if kind == 'gaussian':
            W = torch.from_numpy(self.nn_w).float()
        else:
            if self.attn_w is None:
                with torch.no_grad():
                    X_all_t = torch.from_numpy(self.train_cache).to(dev)
                    Z = self.enc(X_all_t)
                    _ = self._compute_attention(Z)
            W = self.attn_w.detach().cpu()
        if self.graph_space == 'input':
            coords = self.train_cache[:, :2]
        else:
            with torch.no_grad():
                coords = self._embed_all(self.train_cache)[:, :2]
        return idx, W.cpu().numpy(), coords, kind

    def fit(self,X_lab_norm,X_unl):
        X_train=np.vstack([X_lab_norm,X_unl]).astype(np.float32)
        self.train_cache=X_train; self.split_n_lab=len(X_lab_norm)
        X_lab_t=torch.from_numpy(X_lab_norm).to(dev)
        X_all_t=torch.from_numpy(X_train).to(dev)
        opt=optim.AdamW(self.enc.parameters(), lr=self.lr, weight_decay=self.wd)
        self._init_center(X_lab_t)

        for _ in range(self.epochs_warm):
            self.enc.train(); opt.zero_grad()
            Z=self.enc(X_all_t)
            d2=((Z-self.c)**2).sum(1)
            loss=d2[:self.split_n_lab].mean()+self.alpha*d2[self.split_n_lab:].mean()
            loss.backward(); opt.step()
        self.enc.eval()

        with torch.no_grad():
            if self.graph_space == 'latent':
                Z0 = self._embed_all(self.train_cache)
                self._build_knn_from_space(Z0)
            else:
                self._build_knn_from_space(self.train_cache)

        for ep in range(self.epochs_main):
            self.enc.train(); opt.zero_grad()
            Z=self.enc(X_all_t); d2=((Z-self.c)**2).sum(1)
            d2_lab=d2[:self.split_n_lab]; d2_unl=d2[self.split_n_lab:]

            if self.graph_type=='gaussian':
                W=self.nn_w
                rows=np.repeat(np.arange(W.shape[0]),W.shape[1])
                cols=self.nn_idx.reshape(-1)
                w=torch.from_numpy(W.reshape(-1)).to(dev)
                f=d2
                f_rows=f[torch.from_numpy(rows).to(dev)]
                f_cols=f[torch.from_numpy(cols).to(dev)]
                smooth=(w*(f_rows-f_cols)**2).mean()
            else:
                W=self._compute_attention(Z)
                idx=torch.from_numpy(self.nn_idx.astype(np.int64)).to(dev)
                f=d2.unsqueeze(1); f_nei=d2[idx]
                smooth=(W*(f-f_nei)**2).mean()

            loss=d2_lab.mean()+self.alpha*d2_unl.mean()+self.mu*smooth
            loss.backward(); opt.step()

            with torch.no_grad():
                if self.graph_space=='latent' and self.refresh_every is not None:
                    Z_now = self.enc(X_all_t).detach().cpu().numpy()
                    self._maybe_refresh(ep, Z_now)

        self.enc.eval()

    @torch.no_grad()
    def score(self,X):
        Xt=torch.from_numpy(X.astype(np.float32)).to(dev)
        Z=self.enc(Xt); d2=((Z-self.c)**2).sum(1)
        return d2.detach().cpu().numpy()

class OCSVM:
    def __init__(self, nu=0.1, gamma='scale'):
        self.nu=nu; self.gamma=gamma
        self.clf=OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
        self.scaler=StandardScaler()
    def fit(self, X_lab_norm, X_unl_unused):
        Xs=self.scaler.fit_transform(X_lab_norm.astype(np.float32))
        self.clf.fit(Xs)
    def score(self,X):
        Xs=self.scaler.transform(X.astype(np.float32))
        return (-self.clf.decision_function(Xs).ravel())

def _rbf_kernel(X,Y=None,gamma=1.0):
    Y=X if Y is None else Y
    Xn=(X**2).sum(1,keepdims=True); Yn=(Y**2).sum(1,keepdims=True)
    K=Xn+Yn.T-2.0*np.dot(X,Y.T)
    return np.exp(-gamma*np.clip(K,0,None))

def _knn_graph_L(X: np.ndarray, k: int = 15, laplacian_type: str = 'sym') -> np.ndarray:
    """
    Build a kNN graph Laplacian on X.

    Parameters
    ----------
    X : (N, D) array
        Input features (use training normals for S3SVDD).
    k : int, default=15
        #neighbors per node (undirected). Clamped to [1, N-1].
        Pass from CLI (e.g., --s3_k) or fall back to --k.
    laplacian_type : {'sym', 'unnorm'}, default='sym'
        'sym'   -> normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        'unnorm'-> unnormalized Laplacian: L = D - W

    Returns
    -------
    L : (N, N) array
        Graph Laplacian as float64.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N < 2:
        # Degenerate case: no edges, zero Laplacian
        return np.zeros((N, N), dtype=np.float64)

    # Clamp k to a valid range
    k_eff = int(max(1, min(int(k), N - 1)))

    # kNN on X; add 1 for self then drop it
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean')
    nn.fit(X)
    _, idx = nn.kneighbors(X, return_distance=True)

    # Symmetric (undirected) binary adjacency
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in idx[i, 1:]:        # skip self
            W[i, j] = 1.0
            W[j, i] = 1.0

    # Degree and Laplacian
    d = W.sum(axis=1)
    if laplacian_type == 'unnorm':
        L = np.diag(d) - W
    else:  # 'sym' normalized
        with np.errstate(divide='ignore'):
            dmh = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        Dmh = np.diag(dmh)
        L = np.eye(N, dtype=np.float64) - (Dmh @ W @ Dmh)

    return L


class SVDD:
    def __init__(self, nu=0.1, gamma=1.0):
        self.nu=float(nu); self.gamma=float(gamma)
        self.Xtr=None; self.alpha=None; self.const_term=None
    def fit(self,X_lab_norm,X_unl_unused):
        try:
            from cvxopt import matrix, solvers
            solvers.options['show_progress']=False
        except Exception as e:
            raise RuntimeError("cvxopt required for QP solvers.") from e
        X=X_lab_norm.astype(np.float64)
        self.Xtr=X; N=X.shape[0]
        K=_rbf_kernel(X,gamma=self.gamma)
        C=1.0/(self.nu*max(1,N))
        Q=matrix(K); p=matrix(-np.diag(K))
        A=matrix(np.ones((1,N))); b=matrix(np.ones(1))
        G=matrix(np.vstack([-np.eye(N), np.eye(N)]))
        h=matrix(np.hstack([np.zeros(N), C*np.ones(N)]))
        sol=solvers.qp(Q,p,G,h,A,b)
        a=np.array(sol['x']).ravel()
        self.alpha=a; self.const_term=float(a@K@a)
        # ---------- ADD: compute hypersphere radius R^2 ----------
        # Support vectors with 0 < alpha_i < C
        tol = 1e-8
        sv_mask = (a > tol) & (a < C - tol)
        if np.any(sv_mask):
            i = np.where(sv_mask)[0][0]
            # RBF: K(x_i, x_i) = 1.0
            self.R2 = float(1.0 - 2.0 * (K[:, i] @ a) + self.const_term)
        else:
            # Fallback: numeric radius from training scores at (1 - nu)-quantile
            train_scores = (1.0 - 2.0*(K @ a) + self.const_term)
            self.R2 = float(np.quantile(train_scores, 1.0 - self.nu))
        self.R2 = max(self.R2, 0.0)
        
    def score(self,X):
        X=X.astype(np.float64)
        kx=_rbf_kernel(X,self.Xtr,gamma=self.gamma)
        return (1.0 - 2.0*(kx@self.alpha) + self.const_term)

class S3SVDD(SVDD):
    def __init__(self, nu=0.1, gamma=1.0, mu=0.5, k=15, laplacian_type: str = 'sym'):
        super().__init__(nu, gamma)
        self.mu = float(mu)
        self.k = int(k)
        self.laplacian_type = str(laplacian_type)

    def fit(self, X_lab_norm, X_unl):
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        # labeled part (QP lives here)
        X = X_lab_norm.astype(np.float64)
        self.Xtr = X
        N = X.shape[0]

        # kernel on labeled
        K = _rbf_kernel(X, gamma=self.gamma)

        # --- NEW: build graph on labeled + unlabeled, then slice LL block ---
        if X_unl is not None and X_unl.size:
            X_all = np.vstack([X_lab_norm, X_unl]).astype(np.float64)
            L_all = _knn_graph_L(X_all, k=self.k, laplacian_type=self.laplacian_type)
            L_ll = L_all[:N, :N]           # labeled–labeled block
        else:
            L_ll = _knn_graph_L(X, k=self.k, laplacian_type=self.laplacian_type)

        # graph-regularized kernel matrix over labeled normals
        Kreg = K + self.mu * (K @ L_ll @ K)

        # standard SVDD QP on labeled normals
        C = 1.0 / (self.nu * max(1, N))
        Q = matrix(Kreg); p = matrix(-np.diag(K))
        A = matrix(np.ones((1, N))); b = matrix(np.ones(1))
        G = matrix(np.vstack([-np.eye(N), np.eye(N)]))
        h = matrix(np.hstack([np.zeros(N), C * np.ones(N)]))
        sol = solvers.qp(Q, p, G, h, A, b)

        a = np.array(sol['x']).ravel()
        self.alpha = a
        self.const_term = float(a @ K @ a)

        # radius R^2 from support vectors (fallback to quantile if none)
        tol = 1e-8
        sv_mask = (a > tol) & (a < C - tol)
        if np.any(sv_mask):
            i = np.where(sv_mask)[0][0]
            self.R2 = float(1.0 - 2.0 * (K[:, i] @ a) + self.const_term)
        else:
            train_scores = (1.0 - 2.0 * (K @ a) + self.const_term)
            self.R2 = float(np.quantile(train_scores, 1.0 - self.nu))
        self.R2 = max(self.R2, 0.0)



class     """
    Author-style toggles:
      - use_unlabeled: include unlabeled pool
      - use_lab_anom: if labeled anomalies exist and this is 1, push them away
      - margin: hinge margin for anomalies
      - eta: weight for the anomaly term (authors use η)
      - independent lr/wd/epochs/batch_size
    """
    def __init__(self,in_dim=2,hidden=(64,64),out_dim=16,
                 lr=1e-3, wd=1e-5, epochs=120,
                 margin=1.0, eta=1.0,
                 use_unlabeled=True, use_lab_anom=True,
                 batch_size: Optional[int] = None):
        self.enc=MLP(in_dim,hidden,out_dim).to(dev); self.c=None
        self.lr=lr; self.wd=wd; self.epochs=epochs
        self.margin=float(margin); self.eta=float(eta)
        self.use_unlabeled=bool(use_unlabeled)
        self.use_lab_anom=bool(use_lab_anom)
        self.batch_size = batch_size
    @torch.no_grad()
    def _init_center(self,X_lab_norm_t):
        Z=self.enc(X_lab_norm_t).detach(); self.c=Z.mean(0)
    def fit(self, X_lab_norm, X_unl, X_lab_anom: Optional[np.ndarray] = None):
        X_lab=X_lab_norm.astype(np.float32)
        Xt_lab=torch.from_numpy(X_lab).to(dev)
        Xt_unl=torch.from_numpy(X_unl.astype(np.float32)).to(dev) if self.use_unlabeled and X_unl.size else None
        Xt_an = torch.from_numpy(X_lab_anom.astype(np.float32)).to(dev) if (self.use_lab_anom and X_lab_anom is not None and X_lab_anom.size) else None
        opt=optim.AdamW(self.enc.parameters(), lr=self.lr, weight_decay=self.wd)
        self._init_center(Xt_lab)
        for _ in range(self.epochs):
            self.enc.train(); opt.zero_grad()
            Z_lab=self.enc(Xt_lab); d2_lab=((Z_lab-self.c)**2).sum(1)
            loss = d2_lab.mean()
            if Xt_unl is not None:
                Z_unl=self.enc(Xt_unl); d2_unl=((Z_unl-self.c)**2).sum(1)
                loss = loss + d2_unl.mean()
            if Xt_an is not None:
                Z_an = self.enc(Xt_an); d2_an = ((Z_an - self.c)**2).sum(1)
                loss = loss + self.eta * torch.relu(self.margin - d2_an).mean()
            loss.backward(); opt.step()
        self.enc.eval()
    @torch.no_grad()
    def score(self,X):
        Xt=torch.from_numpy(X.astype(np.float32)).to(dev)
        Z=self.enc(Xt); d2=((Z-self.c)**2).sum(1)
        return d2.detach().cpu().numpy()

# ========= Plotting (2-D only) =========

def _pretty_method_title(name: str) -> str:
    """Normalize method names for plot titles."""
    mapping = {
        'DeepSVDD':    'DeepSVDD',
        'OCSVM':       'OCSVM',
        'SVDD':     'SVDD',
        'S3SVDD':   'S3SVDD',
        '    }
    return mapping.get(name, name)


def plot_panel(ax, model, X_all, split: Split, tau: float, title: str,
               grid_step: float = 0.05,
               plot_sets: List[str] = ['train','test'],
               show_boundary: bool = True,
               mode: str = 'score',
               ref_scores: Optional[np.ndarray] = None):
    x_min, x_max = X_all[:, 0].min() - 0.6, X_all[:, 0].max() + 0.6
    y_min, y_max = X_all[:, 1].min() - 0.6, X_all[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    XY = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    Z_raw = model.score(XY).reshape(xx.shape)

    if mode == 'quantile' and ref_scores is not None and len(ref_scores) > 0:
        ref_sorted = np.sort(ref_scores.astype(np.float64))
        Z_color = (np.searchsorted(ref_sorted, Z_raw.ravel(), side='right') /
                   max(1, ref_sorted.size)).reshape(Z_raw.shape)
        cbar_label = 'Anomaly score quantile'
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    elif mode == 'quantile':
        Zf = Z_raw.ravel()
        ranks = np.argsort(np.argsort(Zf, kind='mergesort'))
        denom = max(1, Zf.size - 1)
        Z_color = (ranks / denom).reshape(Z_raw.shape)
        cbar_label = 'Anomaly score quantile (grid, 0–1)'
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        Z_color = Z_raw
        # ← change “s(x)” to “f(x)”
        cbar_label = r'Anomaly score $f(x)$'
        ticks = None

    cf = ax.contourf(xx, yy, Z_color, levels=40, alpha=0.85, cmap='plasma')

    # --- highlight normal region {f(x) <= tau} with soft fill + dashed boundary ---
    if show_boundary:
        try:
        # light fill *inside* the normal region
            ax.contourf(xx, yy, Z_raw, levels=[np.nanmin(Z_raw), float(tau)], colors=['#7ac37a'], alpha=0.18)
        except Exception:
            pass

        # dashed boundary at f(x) = tau
        ax.contour(xx, yy, Z_raw,
                   levels=[float(tau)],
                   colors='white', linestyles='--', linewidths=2.0)


    if 'train' in plot_sets and split.X_lab_norm.size > 0:
        ax.scatter(split.X_lab_norm[:, 0], split.X_lab_norm[:, 1],
                   s=28, c='#2ca02c', marker='o', label='Labeled Normal',
                   edgecolors='k', linewidths=0.4)

    if 'lab_anom' in plot_sets and split.X_lab_anom.size > 0:
        ax.scatter(split.X_lab_anom[:, 0], split.X_lab_anom[:, 1],
                   s=36, c='#d62728', marker='*', label='Labeled Anomaly',
                   edgecolors='k', linewidths=0.5)

    if 'unlabeled' in plot_sets and split.X_unl.size > 0:
        unl_n = split.X_unl[split.y_unl_true == 0]
        unl_a = split.X_unl[split.y_unl_true == 1]
        if len(unl_n):
            ax.scatter(unl_n[:, 0], unl_n[:, 1],
                       s=18, facecolors='none', edgecolors='#1f77b4',
                       marker='o', label='Unlabeled (true normal)')
        if len(unl_a):
            ax.scatter(unl_a[:, 0], unl_a[:, 1],
                       s=18, facecolors='none', edgecolors='#ff7f0e',
                       marker='o', label='Unlabeled (true anomaly)')

    if 'val' in plot_sets and split.X_val.size > 0:
        val_n = split.X_val[split.y_val == 0]
        val_a = split.X_val[split.y_val == 1]
        if len(val_n):
            ax.scatter(val_n[:, 0], val_n[:, 1],
                       s=26, c='#17becf', marker='^', label='Val Normal',
                       edgecolors='k', linewidths='0.3')
        if len(val_a):
            ax.scatter(val_a[:, 0], val_a[:, 1],
                       s=26, c='#e377c2', marker='^', label='Val Anomaly',
                       edgecolors='k', linewidths='0.3')

    if 'test' in plot_sets and split.X_test.size > 0:
        te_n = split.X_test[split.y_test == 0]
        te_a = split.X_test[split.y_test == 1]
        if len(te_n):
            ax.scatter(te_n[:, 0], te_n[:, 1],
                       s=26, c='#7f7f7f', marker='s', label='Test Normal',
                       edgecolors='k', linewidths=0.3)
        if len(te_a):
            ax.scatter(te_a[:, 0], te_a[:, 1],
                       s=26, c='#d62728', marker='s', label='Test Anomaly',
                       edgecolors='k', linewidths=0.3)

    # ← add axis labels everywhere
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=1)

    cbar = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)
    if ticks is not None:
        cbar.set_ticks(ticks)

    return cf

def _save_graph_artifacts(model: OurPaper1, out_dir: str, tag: str, top_m: int = 2):
    ensure_dir(out_dir)
    idx, W, coords, kind = model.export_graph()
    np.save(os.path.join(out_dir, f'{tag}_{kind}_idx.npy'), idx)
    np.save(os.path.join(out_dir, f'{tag}_{kind}_W.npy'), W)
    np.save(os.path.join(out_dir, f'{tag}_{kind}_coords.npy'), coords)
    csv_path = os.path.join(out_dir, f'{tag}_{kind}_edges.csv')
    with open(csv_path, 'w', encoding='utf-8') as gf:
        gf.write('src,dst,weight\n')
        for i in range(idx.shape[0]):
            order = np.argsort(-W[i])[:min(top_m, W.shape[1])]
            for r in order:
                gf.write(f'{i},{idx[i, r]},{W[i, r]:.6f}\n')
    fig_g, ax_g = plt.subplots(figsize=(5,5))
    ax_g.scatter(coords[:,0], coords[:,1], s=10, c='k', alpha=0.5)
    for i in range(idx.shape[0]):
        order = np.argsort(-W[i])[:min(top_m, W.shape[1])]
        for r in order:
            j = idx[i, r]
            x = [coords[i,0], coords[j,0]]
            y = [coords[i,1], coords[j,1]]
            lw = 0.5 + 2.0 * (W[i, r] / (W.max() + 1e-8))
            ax_g.plot(x, y, '-', alpha=0.35, linewidth=lw)
    ax_g.set_title(f'{kind} graph')
    fig_g.tight_layout()
    fig_g.savefig(os.path.join(out_dir, f'{tag}_{kind}_graph.png'), dpi=400, bbox_inches='tight')
    plt.close(fig_g)

def save_and_show(fig, path: str):
    fig.savefig(path, dpi=800, bbox_inches="tight")
    plt.show(block=False); plt.pause(0.001); plt.close(fig)

# ========= Training & evaluation loops =========

def build_models(args, in_dim: int) -> Dict[str, Any]:
    models = {}

    # ---- GAR-DSVDD(graph-regularized center) ----
    if args.method in ('GAR-DSVDD', 'all'):
        models['GAR-DSVDD'] = GARTrainer(
            in_dim=in_dim, hidden=(64, 64), out_dim=args.latent,
            lr=args.lr, wd=args.wd,
            epochs_warm=args.epochs_warm, epochs_main=args.epochs_main,
            k=int(args.k), alpha=float(args.alpha), mu=float(args.mu),
            graph_type=args.graph, graph_space=args.graph_space,
            heads=int(args.heads), dk=int(args.dk),
            refresh_every=(None if (args.refresh_every is None or int(args.refresh_every) <= 0)
                           else int(args.refresh_every)),
            ema_beta=float(args.ema_beta),
        )

    # ---- DeepSVDD (author-style controls) ----
    if args.method in ('deepsvdd', 'all'):
        models['DeepSVDD'] = DeepSVDD(
            in_dim=in_dim, hidden=(64, 64), out_dim=args.latent,
            objective=args.deepsvdd_objective,         # 'oneclass' | 'soft'
            nu=float(args.deepsvdd_nu),                # only used if objective='soft'
            lr=float(args.deepsvdd_lr),
            wd=float(args.deepsvdd_wd),
            epochs=int(args.epochs_main),
            pretrain_epochs=int(args.deepsvdd_pretrain_epochs),
            batch_size=(None if str(args.deepsvdd_batch_size) in ('None', '', '0')
                        else int(args.deepsvdd_batch_size)),
            train_on=args.deepsvdd_train_on,           # pipeline may still force lab_only
        )

    # ---- OCSVM ----
    if args.method in ('ocsvm', 'all'):
        models['OCSVM'] = OCSVM(nu=float(args.nu), gamma=args.gamma)

    # ---- SVDD / S3SVDD (require cvxopt) ----
    if args.method in ('svdd','s3svdd','all'):
        cvx_ok = True
        try:
            import cvxopt  # noqa: F401
        except Exception:
            cvx_ok = False
            print("[warn] cvxopt not found; skipping SVDD / S3SVDD.")

        if cvx_ok:
            if args.method in ('svdd', 'all'):
                models['SVDD'] = SVDD(
                    nu=float(args.nu),
                    gamma=float(args.rbf_gamma),
                )

            if args.method in ('s3svdd', 'all'):
                # Allow S3-specific overrides; fall back to shared args when absent
                k3  = (args.s3_k if hasattr(args, 's3_k') and args.s3_k is not None else args.k)
                mu3 = (args.s3_mu if hasattr(args, 's3_mu') and args.s3_mu is not None else args.mu)
                g3  = (args.s3_rbf_gamma if hasattr(args, 's3_rbf_gamma') and args.s3_rbf_gamma is not None else args.rbf_gamma)
                nu3 = (args.s3_nu if hasattr(args, 's3_nu') and args.s3_nu is not None else args.nu)

                models['S3SVDD'] = S3SVDD(
                    nu=float(nu3),
                    gamma=float(g3),
                    mu=float(mu3),
                    k=int(k3),
                    laplacian_type=getattr(args, 'laplacian_type', 'sym'),
                )

    # ----     if args.method in ('        models['            in_dim=in_dim, hidden=(64, 64), out_dim=args.latent,
            lr=float(args.sad_lr), wd=float(args.sad_wd), epochs=int(args.epochs_main),
            margin=float(args.sad_margin),
            eta=float(getattr(args, 'sad_eta', 1.0)),
            use_unlabeled=bool(int(args.sad_use_unlabeled)),
            use_lab_anom=bool(int(getattr(args, 'sad_use_lab_anom', 0))),
            batch_size=(None if str(getattr(args, 'sad_batch_size', None)) in ('None', '', '0')
                        else int(args.sad_batch_size)),
        )

    return models


def fit_and_calibrate(args, split: Split, model, method_name: str) -> float:
    if isinstance(model, OurPaper1):
        model.fit(split.X_lab_norm, split.X_unl)
    elif isinstance(model,         X_an = split.X_lab_anom if (bool(args.sad_use_lab_anom) and split.X_lab_anom.size) else None
        model.fit(
            split.X_lab_norm,
            split.X_unl if bool(args.sad_use_unlabeled) else np.empty((0, split.X_lab_norm.shape[1]), np.float32),
            X_lab_anom=X_an
        )
    else:
        model.fit(split.X_lab_norm, np.empty((0, split.X_lab_norm.shape[1]), np.float32))

    # DeepSVDD (soft-boundary): threshold is learned radius R^2
    if isinstance(model, DeepSVDD) and getattr(model, "objective", "oneclass") == "soft":
        return float(model._R2().detach().cpu().item())

    # SVDD / S3SVDD: if you store the sphere radius during fit(), use it
    if isinstance(model, (SVDD, S3SVDD)) and getattr(model, "R2", None) is not None:
        return float(model.R2)

    # Otherwise fall back to the configured strategy (quantile / fpr / youden / zero)
    strat, q, fpr = get_tau_cfg_for_method(args, method_name)
    s_train_norm = model.score(split.X_lab_norm)
    s_val_norm = model.score(split.X_val[split.y_val == 0]) if ('val' in strat or strat in ('fpr', 'youden')) else None
    s_val_all  = model.score(split.X_val) if strat in ('fpr', 'youden', 'val_quantile') else None

    if q is None: q = 0.99
    if fpr is None: fpr = 0.05

    tau = _pick_tau(
        strategy=strat,
        scores_train_norm=s_train_norm,
        scores_val_norm=s_val_norm,
        scores_val_all=s_val_all,
        y_val_all=(split.y_val if strat in ('fpr', 'youden') else None),
        quantile=q,
        target_fpr=fpr
    )
    return tau

def _print_data_summary(dataset: str, seed: int, split: Split):
    n_lab = split.X_lab_norm.shape[0]
    n_unl = split.X_unl.shape[0]
    n_unl_norm = int(np.sum(split.y_unl_true==0))
    n_unl_anom = int(np.sum(split.y_unl_true==1))
    eps_hat = n_unl_anom / max(1, n_unl)
    print(f"\n[split] {dataset} seed={seed} | labeled_normals={n_lab} | unlabeled={n_unl} "
          f"(norm={n_unl_norm}, anom={n_unl_anom}, eps_hat={eps_hat:.3f})")
    print(f"[shapes] X_lab_norm={split.X_lab_norm.shape}, X_unl={split.X_unl.shape}, "
          f"X_val={split.X_val.shape}, X_test={split.X_test.shape}, in_dim={split.X_lab_norm.shape[1]}")

def run_once(args, dataset: str, seed: int, run_dir: str,
             master_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    set_seed(seed)

    # --- helper: pretty display names for figure titles
    def _pretty_method_title(name: str) -> str:
        mapping = {
            'DeepSVDD':   'DeepSVDD',
            'OCSVM':      'OCSVM',
            'SVDD':    'SVDD',
            'S3SVDD':  'S3SVDD',
            '        }
        return mapping.get(name, name)

    
        elif strat in ('val_quantile', 'fpr', 'youden'):
            ref_scores = model.score(split.X_val)
        else:
            ref_scores = model.score(split.X_lab_norm)

        if do_plots:
            pretty = _pretty_method_title(name)

            # SCORE subplot
            ax_s = axs_score[j]
            plot_panel(ax_s, model, X_all, split, tau,
                       title=f"{pretty} — anomaly score",
                       plot_sets=plot_sets_local,
                       show_boundary=bool(args.plot_boundary),
                       mode='score')
            ax_s.set_xlabel(r"$x_1$")
            ax_s.set_ylabel(r"$x_2$")
            ax_s.set_title(f"{pretty} — anomaly score", fontsize=14, pad=2)

            # QUANTILE subplot
            ax_q = axs_quant[j]
            plot_panel(ax_q, model, X_all, split, tau,
                       title=f"{pretty} — anomaly score quantile",
                       plot_sets=plot_sets_local,
                       show_boundary=bool(args.plot_boundary),
                       mode='quantile',
                       ref_scores=ref_scores)
            ax_q.set_xlabel(r"$x_1$")
            ax_q.set_ylabel(r"$x_2$")
            ax_q.set_title(f"{pretty} — anomaly score quantile", fontsize=14, pad=2)

        params_json = json.dumps(args_to_dict(args), sort_keys=True)
        with open(per_csv, "a", encoding="utf-8") as f:
            f.write(f"{dataset},{seed},{name},{tau:.6f},{metrics['AUROC']:.6f},{metrics['AUPRC']:.6f},"
                    f"{metrics['pAUC@1FPR']:.6f},{metrics['Accuracy']:.6f},{metrics['F1']:.6f},"
                    f"{metrics['DetectionRate_TPR']:.6f},{metrics['BalancedAcc']:.6f},"
                    f"{metrics['TP']},{metrics['FP']},{metrics['TN']},{metrics['FN']},{params_json}\n")

        row = dict(params=args_to_dict(args))
        row.update(dict(dataset=dataset, seed=seed, method=name, tau=tau))
        row.update(metrics)
        master_rows.append(row)

    if do_plots:
        fig_score.tight_layout()
        out_score = os.path.join(run_dir, f"grid_{dataset}_seed{seed}_SCORE.png")
        save_and_show(fig_score, out_score)

        fig_quant.tight_layout()
        out_quant = os.path.join(run_dir, f"grid_{dataset}_seed{seed}_QUANT_SYNC.png")
        save_and_show(fig_quant, out_quant)

        print(f"[plot] saved {out_score}")
        print(f"[plot] saved {out_quant}")
        print(f"[plot] saved {out_train}")
    else:
        print("[plot] skipped (feature dim != 2)")

    return results

def aggregate_seed_tables(all_results: List[Dict[str, Dict[str,float]]]) -> Dict[str, Dict[str,float]]:
    methods=sorted({m for r in all_results for m in r.keys()})
    keys=sorted([k for k in all_results[0][methods[0]].keys()])
    out={}
    for m in methods:
        agg={}
        for k in keys:
            vals=[r[m][k] for r in all_results]
            agg[k+'_mean']=float(np.mean(vals)); agg[k+'_std']=float(np.std(vals))
        out[m]=agg
    return out

def print_seed_table(res: Dict[str, Dict[str,float]], seed: int, dataset: str):
    headers=["Method","AUROC","AUPRC","pAUC@1FPR","Accuracy","F1","DetectionRate_TPR","BalancedAcc","TP","FP","TN","FN"]
    rows=[]
    for m,d in sorted(res.items()):
        rows.append([m,d["AUROC"],d["AUPRC"],d["pAUC@1FPR"],d["Accuracy"],d["F1"],d["DetectionRate_TPR"],d["BalancedAcc"],int(d["TP"]),int(d["FP"]),int(d["TN"]),int(d["FN"])])
    print(f"\nSeed {seed} — {dataset}:\n"+format_table(rows,headers))

def write_master_csv(outdir: str, master_rows: List[Dict[str,Any]]):
    master_path=os.path.join(outdir,"all_runs.csv")
    param_keys=sorted({k for r in master_rows for k in r['params'].keys()})
    metric_keys=["AUROC","AUPRC","pAUC@1FPR","Accuracy","F1","DetectionRate_TPR","BalancedAcc","TP","FP","TN","FN","tau"]
    with open(master_path,"w",encoding="utf-8") as f:
        header=["dataset","seed","method"] + metric_keys + param_keys
        f.write(",".join(header)+"\n")
        for r in master_rows:
            base=[r["dataset"], str(r["seed"]), r["method"]]
            vals=[f"{r[k]:.6f}" if isinstance(r[k],float) else str(r[k]) for k in metric_keys]
            pvals=[str(r["params"].get(k,"")) for k in param_keys]
            f.write(",".join(base+vals+pvals)+"\n")
    print(f"[csv] wrote master CSV: {master_path}")

def append_global_long_csv(path: str, run_id: str, master_rows: List[Dict[str,Any]]):
    """
    Append-friendly 'long' CSV with stable columns across invocations.
    Includes params_json rather than expanding per-arg columns.
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    header = ["run_id","dataset","seed","method",
              "tau","AUROC","AUPRC","pAUC@1FPR","Accuracy","F1","DetectionRate_TPR","BalancedAcc",
              "TP","FP","TN","FN","params_json"]
    file_exists = os.path.isfile(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(header) + "\n")
        for r in master_rows:
            params_json = json.dumps(r.get("params",{}), sort_keys=True)
            row = [
                run_id,
                r["dataset"],
                str(r["seed"]),
                r["method"],
                f"{r.get('tau', float('nan')):.6f}",
                f"{r.get('AUROC', float('nan')):.6f}",
                f"{r.get('AUPRC', float('nan')):.6f}",
                f"{r.get('pAUC@1FPR', float('nan')):.6f}",
                f"{r.get('Accuracy', float('nan')):.6f}",
                f"{r.get('F1', float('nan')):.6f}",
                f"{r.get('DetectionRate_TPR', float('nan')):.6f}",
                f"{r.get('BalancedAcc', float('nan')):.6f}",
                str(r.get('TP',"")),
                str(r.get('FP',"")),
                str(r.get('TN',"")),
                str(r.get('FN',"")),
                params_json.replace("\n"," ").replace("\r"," ")
            ]
            f.write(",".join(row) + "\n")
    print(f"[csv] appended global LONG CSV: {path}")

def report_best_by_method(outdir: str, master_rows: List[Dict[str,Any]], select_metric: str):
    by_method={}
    for r in master_rows:
        m=r["method"]; score=r.get(select_metric, None)
        if score is None: continue
        if (m not in by_method) or (score > by_method[m]["score"]):
            by_method[m]={"score":score, "row":r}
    print(f"\n=== Best runs by method (metric={select_metric}) ===")
    headers=["Method","Dataset","Seed","Score","Params (subset)"]
    rows=[]
    for m,info in sorted(by_method.items()):
        r=info["row"]
        keep=[
            "graph","graph_space","alpha","mu","k","heads","dk","latent",
            "epochs_warm","epochs_main","lr","wd",
            "deepsvdd_objective","deepsvdd_nu","deepsvdd_lr","deepsvdd_wd","deepsvdd_pretrain_epochs",
            "sad_use_unlabeled","sad_use_lab_anom","sad_margin","sad_eta","sad_lr","sad_wd",
            "nu","rbf_gamma","laplacian_type",
            "tau_strategy_GAR-DSVDD","tau_strategy_deepsvdd","tau_strategy_            "tau_strategy_svdd","tau_strategy_s3svdd","refresh_every","deepsvdd_train_on","save_graph","graph_dir"
        ]
        psub={k:r["params"].get(k) for k in keep if k in r["params"]}
        rows.append([m, r["dataset"], r["seed"], round(info["score"],6), json.dumps(psub, sort_keys=True)])
    print(format_table(rows, headers))

# ========= CLI / Main =========

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets', type=str, default='blobs')
    p.add_argument('--n_norm', type=int, default=1200)
    p.add_argument('--n_anom', type=int, default=400)
    p.add_argument('--rho', type=float, default=0.1)
    p.add_argument('--epsilon', type=float, default=0.05)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--test_frac', type=float, default=0.2)
    p.add_argument('--labeled_anom_frac', type=float, default=0.0)

    p.add_argument('--method', type=str, default='all', choices=['gar_dsvdd','deepsvdd','ocsvm','svdd','s3svdd','all'])

    # Shared representation
    p.add_argument('--latent', type=int, default=16)
    p.add_argument('--epochs_warm', type=int, default=5,
                   help="Warm-up epochs (GAR-DSVDD): center loss on (lab + alpha*unl), no graph.")
    p.add_argument('--epochs_main', type=int, default=120)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--wd', type=float, default=1e-5)

    # Graph (GAR-DSVDD)
    p.add_argument('--alpha', type=float, default=0.3)
    p.add_argument('--mu', type=float, default=0.5)
    p.add_argument('--k', type=int, default=15)
    p.add_argument('--graph', type=str, default='attention', )
    p.add_argument('--graph_space', type=str, default='input', ,
                   help="Space to build kNN neighbor indices. 'input'=fixed; 'latent'=can refresh.")
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--dk', type=int, default=32)
    p.add_argument('--refresh_every', type=int, default=None,
                   help="Positive int = refresh cadence (latent space only). Omit or <=0 = disabled.")
    p.add_argument('--ema_beta', type=float, default=0.9)

    # DeepSVDD (author-style)
    p.add_argument('--deepsvdd_objective', type=str, default='soft',
                   ,
                   help="DeepSVDD objective: 'oneclass' (hard boundary) or 'soft' (soft-boundary with nu).")
    p.add_argument('--deepsvdd_nu', type=float, default=0.1,
                   help="Soft-boundary nu (fraction of outliers allowed). Used only if objective='soft'.")
    p.add_argument('--deepsvdd_pretrain_epochs', type=int, default=0,
                   help="If >0, run a (stub) pretrain phase (decoder not implemented here).")
    p.add_argument('--deepsvdd_batch_size', type=int, default=None,
                   help="Batch size (unused in this full-batch toy impl; kept for parity).")
    p.add_argument('--deepsvdd_lr', type=float, default=5e-4)
    p.add_argument('--deepsvdd_wd', type=float, default=1e-5)
    p.add_argument('--deepsvdd_train_on', type=str, default='lab_only',
                   ,
                   help='Accepted for CLI parity; internally still lab_only.')

    # OCSVM/QP
    p.add_argument('--nu', type=float, default=0.1)
    p.add_argument('--gamma', type=str, default='scale',      help="OCSVM gamma ('scale'|'auto' or numeric).")
    p.add_argument('--rbf_gamma', type=float, default=1.0,    help="RBF gamma for QP-based SVDD variants.")
    p.add_argument('--laplacian_type', type=str, default='sym', ,
                   help="Graph Laplacian type for S3SVDD.")

    #     p.add_argument('--sad_use_unlabeled', type=int, default=1)
    p.add_argument('--sad_use_lab_anom', type=int, default=0,
                   help="If 1 and labeled anomalies exist, push them outside a margin.")
    p.add_argument('--sad_lr', type=float, default=1e-3)
    p.add_argument('--sad_wd', type=float, default=1e-5)
    p.add_argument('--sad_margin', type=float, default=1.0,
                   help="Hinge margin for labeled anomalies.")
    p.add_argument('--sad_eta', type=float, default=1.0,
                   help="Weight for anomaly hinge term.")
    p.add_argument('--sad_batch_size', type=int, default=None,
                   help="Batch size (unused in this full-batch toy impl; kept for parity).")
    # --- S3SVDD (method-specific overrides; each falls back to shared flags) ---
    p.add_argument('--s3_k', type=int, default=None,
                   help="S3SVDD: k for KNN graph; if None, falls back to --k.")
    p.add_argument('--s3_mu', type=float, default=None,
                   help="S3SVDD: smoothness weight mu; if None, falls back to --mu.")
    p.add_argument('--s3_rbf_gamma', type=float, default=None,
                   help="S3SVDD: RBF kernel gamma; if None, falls back to --rbf_gamma.")
    p.add_argument('--s3_nu', type=float, default=None,
                   help="S3SVDD: nu (outlier frac); if None, falls back to --nu.")


    # Per-method τ strategy & params
    p.add_argument('--tau_strategy_GAR-DSVDD', type=str, default='train_quantile',
                   )
    p.add_argument('--tau_strategy_deepsvdd', type=str, default='train_quantile',
                   )
    p.add_argument('--tau_strategy_                   )
    p.add_argument('--tau_strategy_ocsvm', type=str, default='zero',
                   )
    p.add_argument('--tau_strategy_svdd', type=str, default='train_quantile',
                   )
    p.add_argument('--tau_strategy_s3svdd', type=str, default='train_quantile',
                   )

    p.add_argument('--quantile_GAR-DSVDD', type=float, default=0.99)
    p.add_argument('--quantile_deepsvdd', type=float, default=0.99)
    p.add_argument('--quantile_    p.add_argument('--quantile_svdd', type=float, default=0.99)
    p.add_argument('--quantile_s3svdd', type=float, default=0.99)

    p.add_argument('--target_fpr_GAR-DSVDD', type=float, default=0.05)
    p.add_argument('--target_fpr_deepsvdd', type=float, default=0.05)
    p.add_argument('--target_fpr_    p.add_argument('--target_fpr_ocsvm', type=float, default=0.05)
    p.add_argument('--target_fpr_svdd', type=float, default=0.05)
    p.add_argument('--target_fpr_s3svdd', type=float, default=0.05)

    # Graph saving / visualization
    p.add_argument('--plot_sets', type=str, default='train,test',
               help="Comma-separated subsets to plot: train,lab_anom,unlabeled,val,test")
    p.add_argument('--plot_auto_unl', type=int, default=1,
               help="If 1, auto-include unlabeled in plots for methods that use unlabeled.")
    p.add_argument('--plot_boundary', type=int, default=1,
               help="If 0, hide decision boundary (plot only anomaly score field).")

    p.add_argument('--save_graph', type=int, default=0,
                   help='1 to save graph matrices/edges/PNG after training (GAR-DSVDD/gaussian).')
    p.add_argument('--graph_dir', type=str, default='graphs',
                   help='Subdirectory to write graph files when --save_graph=1.')

    p.add_argument('--select_metric', type=str, default='AUROC',
                   )

    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--seed_list', type=str, default='', help='Comma-separated seeds to run (e.g., 7,0,5). If set, overrides --seeds.')
    p.add_argument('--outdir', type=str, default='plots')

    # global long CSV appender (across many runs / grid search)
    p.add_argument('--grid_append_csv', type=str, default='',
                   help="Optional absolute path to a global LONG CSV to append each run's rows.")

    return p.parse_args(args=[]) if 'ipykernel' in sys.modules else p.parse_args()

def main():
    args=parse_args()
    args.plot_sets = [s.strip() for s in args.plot_sets.split(',') if s.strip()]
    if (args.refresh_every is None) or (isinstance(args.refresh_every,int) and args.refresh_every<=0):
        args.refresh_every = None

    print("Device:", dev); print("Args:", args)
    dataset_list = ['blobs']
    run_id=make_run_id(args)
    param_dir=os.path.join(args.outdir, f"run_{run_id}")
    ensure_dir(param_dir)

    with open(os.path.join(param_dir,"params.json"),"w",encoding="utf-8") as f:
        json.dump(args_to_dict(args), f, indent=2, sort_keys=True)

    master_rows=[]
    agg_all_datasets = {}

    for ds in dataset_list:
        seed_results=[]
        seed_seq = [int(s.strip()) for s in args.seed_list.split(',') if s.strip()] if str(getattr(args,'seed_list','')).strip() else list(range(args.seeds))
        for s in seed_seq:
            run_dir_ds_seed=os.path.join(param_dir, f"{ds}_seed{s}")
            ensure_dir(run_dir_ds_seed)
            res = run_once(args, ds, seed=s, run_dir=run_dir_ds_seed, master_rows=master_rows)
            print_seed_table(res, seed=s, dataset=ds)
            seed_results.append(res)
        agg = aggregate_seed_tables(seed_results)
        agg_all_datasets[ds]=agg

    write_master_csv(param_dir, master_rows)
    report_best_by_method(param_dir, master_rows, args.select_metric)

    with open(os.path.join(param_dir,"aggregated.json"),"w",encoding="utf-8") as f:
        json.dump(agg_all_datasets, f, indent=2, sort_keys=True)

    # ---- Append to global LONG CSV if requested
    if isinstance(args.grid_append_csv, str) and args.grid_append_csv.strip():
        try:
            append_global_long_csv(args.grid_append_csv, run_id, master_rows)
        except Exception as e:
            print(f"[warn] failed to append global CSV ({args.grid_append_csv}): {e}")

if __name__=="__main__":
    main()
