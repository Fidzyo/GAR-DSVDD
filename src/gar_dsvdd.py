
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_margin(eta: torch.Tensor) -> torch.Tensor:
    return F.softplus(eta)


def squared_distances_from_z(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """d2 = ||z - c||^2 (vector over batch)."""
    return torch.sum((z - c) ** 2, dim=1)


def score_from_z(z: torch.Tensor, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """f = d2 - m (vector over batch)."""
    d2 = squared_distances_from_z(z, c)
    return d2 - m


def laplacian_quadratic_from_edges(signal: torch.Tensor,
                                   edges: torch.Tensor,
                                   w: torch.Tensor) -> torch.Tensor:
    """
    0.5 * sum_{(i,j) in E} w_ij * (signal_i - signal_j)^2
    Normalized by number of edge weights to be scale-stable across graphs.
    """
    i, j = edges[0], edges[1]
    diff = signal[i] - signal[j]
    val = 0.5 * torch.sum(w * (diff * diff))
    return val / (w.numel() + 1e-12)


def laplacian_quadratic_from_sparseL(signal: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    signal^T L signal for a sparse Laplacian L (COO).
    Normalized by nnz for scale stability.
    """
    y = torch.sparse.mm(L, signal.unsqueeze(1)).squeeze(1)
    val = torch.dot(signal, y)
    nnz = L._nnz()
    return val / (nnz + 1e-12)


class GARDSVDD(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        beta: float = 0.0,
        lambda_u: float = 1.0,
        alpha: float = 0.0,                  # NEW: α-pull on unlabeled d2
        init_center_from: str = "labeled",   # 'labeled' | 'all' | 'zeros'
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.beta = float(beta)
        self.lambda_u = float(lambda_u)
        self.alpha = float(alpha)
        self.register_buffer("c", None)      # c will be set as a BUFFER (fixed)
        self.eta = nn.Parameter(torch.tensor(0.0))  # m = softplus(eta)
        self._init_center_from = init_center_from
        self.device = device or torch.device("cpu")
        self.to(self.device)

    @torch.no_grad()
    def initialize_center(self, x: torch.Tensor, x_lab_mask: Optional[torch.Tensor] = None) -> None:
        """
        Compute center c from labeled subset (default) or all/zeros and store as a BUFFER (fixed).
        """
        self.encoder.eval()
        z = self.encoder(x.to(self.device))
        if self._init_center_from == "zeros":
            c0 = torch.zeros(z.shape[1], device=self.device)
        else:
            if x_lab_mask is not None and self._init_center_from == "labeled":
                z_use = z[x_lab_mask]
                if z_use.numel() == 0:
                    z_use = z
            elif self._init_center_from == "all":
                z_use = z
            else:
                z_use = z
            c0 = z_use.mean(dim=0)
        # store as fixed buffer (no gradients)
        self.c = c0.detach().clone().to(self.device)  # type: ignore[assignment]
        self.register_buffer("c", self.c)             # ensure it's a buffer
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def svdd_hinge_on_labeled(self, z_lab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Soft-boundary hinge on labeled normals only:
            L_svdd = mean(relu(f_lab)) + beta * ||c||^2
        Returns: (L_svdd, m, f_lab)
        """
        m = soft_margin(self.eta)
        f_lab = score_from_z(z_lab, self.c, m)
        svdd = F.relu(f_lab).mean() + self.beta * torch.sum(self.c * self.c)
        return svdd, m, f_lab

    def graph_loss_from_edges(self,
                              d2_all: torch.Tensor,
                              edges: torch.Tensor,
                              w: torch.Tensor) -> torch.Tensor:
        """Graph smoothness on d2 over edge list."""
        return laplacian_quadratic_from_edges(d2_all, edges, w)

    def graph_loss_from_sparseL(self,
                                d2_all: torch.Tensor,
                                L_sparse: torch.Tensor) -> torch.Tensor:
        """Graph smoothness on d2 with sparse Laplacian."""
        return laplacian_quadratic_from_sparseL(d2_all, L_sparse)

    def total_loss(
        self,
        x_lab: torch.Tensor,
        x_all: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        L_sparse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        x_all is expected to be concatenated as [x_lab; x_unl] so that |x_lab| is the split point.
        """
        n_lab = x_lab.shape[0]

        z_lab = self.forward(x_lab.to(self.device))
        z_all = self.forward(x_all.to(self.device))

        # --- labeled hinge (soft-boundary) ---
        L_svdd, m, f_lab = self.svdd_hinge_on_labeled(z_lab)

        # --- α-pull on unlabeled (raw d2, no margin) ---
        d2_all = squared_distances_from_z(z_all, self.c)
        L_alpha = torch.tensor(0.0, device=self.device)
        if d2_all.shape[0] > n_lab and self.alpha > 0.0:
            d2_unl = d2_all[n_lab:]
            if d2_unl.numel() > 0:
                L_alpha = self.alpha * d2_unl.mean()

        # --- graph smoothness on d2 (latent) ---
        if L_sparse is not None:
            L_graph = self.graph_loss_from_sparseL(d2_all, L_sparse)
        else:
            assert edges is not None and w is not None, "Provide (edges, w) or L_sparse for the graph term."
            L_graph = self.graph_loss_from_edges(d2_all, edges, w)

        total = L_svdd + L_alpha + self.lambda_u * L_graph

        stats = {
            "loss": total.detach(),
            "L_svdd": L_svdd.detach(),
            "L_alpha": L_alpha.detach(),
            "L_graph(d2)": L_graph.detach(),
            "margin_m": m.detach(),
            "mean_f_lab": f_lab.mean().detach(),
            "pos_rate_lab": (f_lab > 0).float().mean().detach(),
            "mean_d2_lab": d2_all[:n_lab].mean().detach(),
            "mean_d2_unl": (d2_all[n_lab:].mean().detach()
                            if d2_all.shape[0] > n_lab and d2_all[n_lab:].numel() > 0
                            else torch.tensor(0.0, device=self.device)),
        }
        return total, stats

    @torch.no_grad()
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns anomaly score f(x) = ||z-c||^2 - m (used by the rest of the pipeline).
        """
        self.eval()
        z = self.forward(x.to(self.device))
        m = soft_margin(self.eta)
        return score_from_z(z, self.c, m)

    # -------- AdamW param groups (decay only on encoder weights) --------
    def adamw_param_groups(self, weight_decay: float) -> list:
        decay, no_decay = [], []
        for name, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            # Heuristic: matrices (ndim>=2) get weight decay; biases/norm params do not.
            if p.ndim >= 2 and ("norm" not in name.lower()) and ("bn" not in name.lower()):
                decay.append(p)
            else:
                no_decay.append(p)
        # c is a buffer now; do NOT add it to param groups.
        no_decay.append(self.eta)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
