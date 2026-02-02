from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class WindowSpec:
    start: int  # inclusive
    end: int    # exclusive


def make_windows(B_eff: int, L: int, stride: int) -> List[WindowSpec]:
    windows: List[WindowSpec] = []
    b = 0
    while b < B_eff:
        windows.append(WindowSpec(start=b, end=min(B_eff, b + L)))
        b += stride
    return windows


def make_position_split(max_len: int, fit_frac: float = 0.5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(max_len, generator=g)
    n_fit = int(round(max_len * fit_frac))
    fit_pos = torch.zeros(max_len, dtype=torch.bool)
    fit_pos[perm[:n_fit]] = True
    return fit_pos


class WindowDirAccumulator:
    """
    Accumulates competitor directions per window across batches.
    Uses reservoir sampling to cap memory per window.
    """

    def __init__(self, windows: List[WindowSpec], D: int, cap_per_window: int = 20000, seed: int = 0):
        self.windows = windows
        self.D = D
        self.cap = cap_per_window
        self.seed = seed

        self._buf: List[List[torch.Tensor]] = [[] for _ in range(len(windows))]
        self._seen: List[int] = [0 for _ in range(len(windows))]
        self._g = torch.Generator()
        self._g.manual_seed(seed)

    @torch.no_grad()
    def add(self, j: int, a: torch.Tensor) -> None:
        # a: (D,) float32 cpu
        self._seen[j] += 1
        seen = self._seen[j]

        if len(self._buf[j]) < self.cap:
            self._buf[j].append(a)
            return

        # Reservoir sampling replace with probability cap/seen
        r = int(torch.randint(low=0, high=seen, size=(1,), generator=self._g).item())
        if r < self.cap:
            self._buf[j][r] = a

    @torch.no_grad()
    def stack(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for j in range(len(self.windows)):
            if len(self._buf[j]) == 0:
                out.append(torch.empty(0, self.D, dtype=torch.float32))
            else:
                out.append(torch.stack(self._buf[j], dim=0).contiguous())  # (M_j,D)
        return out


@torch.no_grad()
def collect_and_accumulate_directions(
    acc: WindowDirAccumulator,
    topk_ids: torch.Tensor,          # (B,N,T,topk) cpu
    y_star: torch.Tensor,            # (N,T) cpu
    G_start: torch.Tensor,           # (B_eff,N,T) cpu bool
    attention_mask: torch.Tensor,    # (N,T) cpu
    fit_pos_mask: torch.Tensor,      # (T,) cpu bool
    W: torch.Tensor,                 # (V,D) cpu
    K: int = 8,
):
    B, N, T, topk = topk_ids.shape
    B_eff = G_start.shape[0]

    valid = attention_mask.to(torch.bool)
    pos_ok = fit_pos_mask[:T].to(torch.bool).unsqueeze(0).expand(N, T)

    Wc = W.float()

    for j, wspec in enumerate(acc.windows):
        for b in range(wspec.start, min(wspec.end, B_eff)):
            mask = G_start[b] & valid & pos_ok
            if not mask.any():
                continue

            idx_nt = mask.nonzero(as_tuple=False)
            for nt in idx_nt:
                n = int(nt[0].item())
                t = int(nt[1].item())
                ys = int(y_star[n, t].item())
                cand = topk_ids[b, n, t]

                picked = 0
                for y in cand.tolist():
                    if y == ys:
                        continue
                    a = (Wc[ys] - Wc[y]).contiguous()  # (D,)
                    acc.add(j, a)
                    picked += 1
                    if picked >= K:
                        break


@torch.no_grad()
def fit_subspaces_from_accumulator(
    acc: WindowDirAccumulator,
    k: int,
) -> List[Optional[torch.Tensor]]:
    # Returns U_j list, each (D,k) cpu float32 or None
    dirs = acc.stack()

    U_list: List[Optional[torch.Tensor]] = []
    for A in dirs:
        if A.numel() == 0:
            U_list.append(None)
            continue
        M, D = A.shape
        if M < k:
            U_list.append(None)
            continue

        Aj = A.t().contiguous()  # (D,M)
        U, S, Vh = torch.linalg.svd(Aj, full_matrices=False)
        U_list.append(U[:, :k].contiguous())
    return U_list


@torch.no_grad()
def grassmann_drift(Ua: torch.Tensor, Ub: torch.Tensor) -> torch.Tensor:
    Pa = Ua @ Ua.t()
    Pb = Ub @ Ub.t()
    return torch.linalg.matrix_norm(Pa - Pb, ord=2)


@torch.no_grad()
def drift_series(U_list: List[Optional[torch.Tensor]]) -> torch.Tensor:
    J = len(U_list)
    out = torch.full((max(J - 1, 0),), float("nan"), dtype=torch.float32)
    for j in range(J - 1):
        Ua = U_list[j]
        Ub = U_list[j + 1]
        if Ua is None or Ub is None:
            continue
        out[j] = grassmann_drift(Ua, Ub).float()
    return out
