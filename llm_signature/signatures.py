# signatures.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class FlowOut:
    # Motion side
    step_mag: torch.Tensor      # (B-1, N, T) float32
    step_mag_c: torch.Tensor    # (B-1, N, T) float32
    turn_angle: torch.Tensor    # (B-1, N, T) float32
    token_drift: torch.Tensor   # (N, T) float32
    drift_win: torch.Tensor     # (J-1,) float32

    # Attribution side
    r_eta: torch.Tensor         # (B-1, N, T) float32
    R_attn: torch.Tensor        # (N, T) float32
    R_mlp: torch.Tensor         # (N, T) float32
    attn_mag: torch.Tensor      # (B-1, N, T) float32
    mlp_mag: torch.Tensor       # (B-1, N, T) float32
    comp_mag: torch.Tensor      # (B-1, N, T) float32


def make_windows(num_blocks: int, L: int, stride: int) -> List[Tuple[int, int]]:
    """
    English comment: Windows cover block indices [0, num_blocks-1] with inclusive ends.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if L > num_blocks:
        L = num_blocks

    last_start = max(0, num_blocks - L)
    starts = list(range(0, last_start + 1, stride))
    if len(starts) == 0 or starts[-1] != last_start:
        starts.append(last_start)

    windows: List[Tuple[int, int]] = []
    for s in starts:
        windows.append((s, s + L - 1))
    return windows


def assign_window_ids(num_blocks: int, windows: List[Tuple[int, int]]) -> torch.Tensor:
    """
    English comment: Pick the latest-started window that has started at block b.
    This gives a unique j(b) for every block.
    """
    starts = [w[0] for w in windows]
    J = len(starts)
    j_of_b = torch.empty((num_blocks,), dtype=torch.long)
    j = 0
    for b in range(num_blocks):
        while (j + 1 < J) and (starts[j + 1] <= b):
            j += 1
        j_of_b[b] = j
    return j_of_b


def _transport_between(Uj: torch.Tensor, Ujp1: torch.Tensor) -> torch.Tensor:
    """
    English comment: R = P Q^T from compact SVD of U_{j+1}^T U_j.
    """
    # (k,k)
    M = Ujp1.transpose(0, 1) @ Uj
    # torch.linalg.svd returns U, S, Vh
    P, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = P @ Vh
    return R


def compute_transports(U_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    English comment: R_list[j] maps coordinates from window j to window j+1.
    """
    R_list: List[torch.Tensor] = []
    for j in range(len(U_list) - 1):
        R_list.append(_transport_between(U_list[j], U_list[j + 1]))
    return R_list


def drift_series_from_subspaces(U_list: List[torch.Tensor]) -> torch.Tensor:
    """
    English comment: d_G(Uj, Uj+1) using principal angles from SVD of U_{j+1}^T U_j.
    For orthonormal bases, ||U U^T - V V^T||_2 = sin(theta_max) with cos(theta_i) = sigma_i.
    """
    if len(U_list) <= 1:
        return torch.zeros((0,), dtype=torch.float32)

    vals: List[float] = []
    for j in range(len(U_list) - 1):
        M = U_list[j + 1].transpose(0, 1) @ U_list[j]  # (k,k)
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
        sigma_min = float(S.min().clamp(0.0, 1.0).item())
        dG = (1.0 - sigma_min * sigma_min) ** 0.5
        vals.append(dG)
    return torch.tensor(vals, dtype=torch.float32)


def _proj_to_k(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    English comment: Project (N,T,D) onto (D,k) to get (N,T,k).
    """
    return torch.einsum("ntd,dk->ntk", x, U)


def _apply_R(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    English comment: Apply (k,k) to (N,T,k).
    """
    return torch.einsum("ntk,kl->ntl", x, R)


def _safe_unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    English comment: Normalize with eps.
    """
    n = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / (n + eps)


def _angle(u: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    """
    English comment: Angle in radians between two unit-like vectors.
    """
    dot = (u * v).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(dot)


def _masked_nanmedian(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    English comment: Median over entries where mask==True.
    x and mask share the same shape except along dim.
    """
    x2 = x.clone()
    x2[~mask] = torch.nan
    med = torch.nanmedian(x2, dim=dim).values
    med = torch.nan_to_num(med, nan=0.0)
    return med


def _jvp_norm(kind: str, gamma: torch.Tensor, eps: float, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    English comment: JVP for LayerNorm or RMSNorm (no beta).
    x, v have shape (N,T,D). gamma has shape (D,).
    """
    D = x.shape[-1]
    g = gamma.view(1, 1, D)

    if kind == "rmsnorm":
        # y = x / rms(x), rms = sqrt(mean(x^2) + eps)
        ms = (x * x).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + eps)
        inv = 1.0 / rms
        # dy = v/rms - x * mean(x*v) / rms^3
        xv = (x * v).mean(dim=-1, keepdim=True)
        dy = v * inv - x * (xv * (inv ** 3))
        return dy * g

    # LayerNorm
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    sig = torch.sqrt(var + eps)

    vmu = v.mean(dim=-1, keepdim=True)
    vc = v - vmu
    # J_LN v = vc/sig - xc * mean(xc*vc) / sig^3
    xcv = (xc * vc).mean(dim=-1, keepdim=True)
    dy = vc / sig - xc * (xcv / (sig ** 3))
    return dy * g


def compute_flow_signatures(
    trace_h_out: torch.Tensor,
    trace_o: torch.Tensor,
    trace_m: torch.Tensor,
    U_list: List[torch.Tensor],
    windows: List[Tuple[int, int]],
    attention_mask: torch.Tensor,
    norm_cfg_by_depth: Optional[List[Optional[Dict[str, Any]]]] = None,
    beta_by_depth: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    drift_anchor: str = "end",
    simpson: bool = True,
    device: str = 'cuda:0'
) -> FlowOut:
    if trace_h_out.ndim != 4:
        raise ValueError("trace_h_out must have shape (B,N,T,D)")
    B, N, T, D = trace_h_out.shape
    if B < 2:
        raise ValueError("Need B>=2 to form depth steps")

    attn_mask = attention_mask.to(device=device, dtype=torch.bool)
    if attn_mask.shape != (N, T):
        raise ValueError("attention_mask must have shape (N,T)")

    # Everything to the same device
    U_list = [u.to(device=device, dtype=torch.float32) for u in U_list]
    h = trace_h_out.to(device=device, dtype=torch.float32)
    o = trace_o.to(device=device, dtype=torch.float32)
    m = trace_m.to(device=device, dtype=torch.float32)

    if beta_by_depth is None:
        beta = torch.zeros((B, D), dtype=torch.float32, device=device)
    else:
        if beta_by_depth.shape != (B, D):
            raise ValueError("beta_by_depth must have shape (B,D)")
        beta = beta_by_depth.to(device=device, dtype=torch.float32)

    tilde_h = h - beta.view(B, 1, 1, D)

    j_of_b = assign_window_ids(B, windows).to(device=device)
    R_list = compute_transports(U_list)  # already on device
    drift_win = drift_series_from_subspaces(U_list).to(device=device)

    k = U_list[0].shape[1]
    p = torch.zeros((B, N, T, k), dtype=torch.float32, device=device)

    for j, _ in enumerate(windows):
        bs = (j_of_b == j).nonzero(as_tuple=False).view(-1)
        if bs.numel() == 0:
            continue
        Uj = U_list[j]
        p[bs] = torch.einsum("bntd,dk->bntk", tilde_h[bs], Uj)

    Bm1 = B - 1
    # These MUST be on device
    step_mag   = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)
    step_mag_c = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)
    turn_angle = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)

    attn_mag = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)
    mlp_mag  = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)
    comp_mag = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)
    r_eta    = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)

    ratio_attn = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)
    ratio_mlp  = torch.zeros((Bm1, N, T), dtype=torch.float32, device=device)

    def _get_cfg(depth_idx: int) -> Optional[Dict[str, Any]]:
        if norm_cfg_by_depth is None:
            return None
        if depth_idx < 0 or depth_idx >= len(norm_cfg_by_depth):
            return None
        return norm_cfg_by_depth[depth_idx]

    if simpson:
        alphas = [0.0, 0.5, 1.0]
        weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]
    else:
        alphas = [1.0]
        weights = [1.0]

    for b in range(Bm1):
        jb = int(j_of_b[b].item())
        jbp1 = int(j_of_b[b + 1].item())
        U_tgt = U_list[jbp1]  # already device float32

        # R_b MUST be on device
        if jbp1 == jb:
            Rb = torch.eye(k, dtype=torch.float32, device=device)
        else:
            Rb = R_list[jb].to(device=device, dtype=torch.float32)

        p_b = p[b]
        p_bp1 = p[b + 1]
        delta_p = p_bp1 - _apply_R(p_b, Rb)

        step_mag[b] = torch.linalg.norm(delta_p, dim=-1)

        mask_ntk = attn_mask.unsqueeze(-1).expand(N, T, k)
        med = _masked_nanmedian(delta_p, mask_ntk, dim=1)
        delta_pc = delta_p - med.unsqueeze(1)
        step_mag_c[b] = torch.linalg.norm(delta_pc, dim=-1)

        u_b = _safe_unit(p_b, eps)
        u_bp1 = _safe_unit(p_bp1, eps)
        turn_angle[b] = _angle(u_bp1, _apply_R(u_b, Rb), eps)

        o_b = o[b]
        m_b = m[b]
        dp_o = _proj_to_k(o_b, U_tgt)
        dp_m = _proj_to_k(m_b, U_tgt)
        attn_mag[b] = torch.linalg.norm(dp_o, dim=-1)
        mlp_mag[b] = torch.linalg.norm(dp_m, dim=-1)

        cfg = _get_cfg(b + 1)
        if cfg is None:
            dh_attn = o_b
            dh_mlp = m_b
        else:
            kind = str(cfg["kind"])
            gamma = cfg["weight"].to(device=device, dtype=torch.float32)
            eps_norm = float(cfg["eps"])

            h_b_raw = h[b]
            inj = o_b + m_b

            dh_attn = torch.zeros_like(o_b)
            dh_mlp = torch.zeros_like(m_b)
            for a, wgt in zip(alphas, weights):
                x = h_b_raw + float(a) * inj
                dh_attn = dh_attn + float(wgt) * _jvp_norm(kind, gamma, eps_norm, x, o_b)
                dh_mlp = dh_mlp + float(wgt) * _jvp_norm(kind, gamma, eps_norm, x, m_b)

        dp_attn = _proj_to_k(dh_attn, U_tgt)
        dp_mlp  = _proj_to_k(dh_mlp, U_tgt)
        dp_comp = dp_attn + dp_mlp

        comp_mag[b] = torch.linalg.norm(dp_comp, dim=-1)

        if cfg is None:
            eta = torch.zeros_like(dp_comp)
        else:
            kind = str(cfg["kind"])
            gamma = cfg["weight"].to(device=device, dtype=torch.float32)
            eps_norm = float(cfg["eps"])
            h_b_raw = h[b]
            inj = o_b + m_b
            x1 = h_b_raw + inj
            dh_end = _jvp_norm(kind, gamma, eps_norm, x1, inj)
            dp_end = _proj_to_k(dh_end, U_tgt)
            eta = dp_comp - dp_end

        r_eta[b] = torch.linalg.norm(eta, dim=-1) / (torch.linalg.norm(dp_comp, dim=-1) + eps)

        u_tgt = u_bp1

        def _perp_norm(v: torch.Tensor) -> torch.Tensor:
            dot = (u_tgt * v).sum(dim=-1, keepdim=True)
            v_perp = v - dot * u_tgt
            return torch.linalg.norm(v_perp, dim=-1)

        denom = _perp_norm(dp_comp) + eps
        ratio_attn[b] = _perp_norm(dp_attn) / denom
        ratio_mlp[b]  = _perp_norm(dp_mlp) / denom

    mask_bnt = attn_mask.unsqueeze(0).expand(Bm1, N, T)
    ratio_attn2 = ratio_attn.clone()
    ratio_mlp2  = ratio_mlp.clone()
    ratio_attn2[~mask_bnt] = torch.nan
    ratio_mlp2[~mask_bnt]  = torch.nan

    R_attn = torch.nanmedian(ratio_attn2, dim=0).values
    R_mlp  = torch.nanmedian(ratio_mlp2, dim=0).values
    R_attn = torch.nan_to_num(R_attn, nan=0.0)
    R_mlp  = torch.nan_to_num(R_mlp, nan=0.0)

    token_drift = torch.zeros((N, T), dtype=torch.float32, device=device)
    J = len(U_list)
    for j in range(J - 1):
        b0, b1 = windows[j]
        b_star = b1 if drift_anchor == "end" else b0
        b_star = int(max(0, min(B - 1, b_star)))

        Uj = U_list[j]
        Ujp1 = U_list[j + 1]
        x = tilde_h[b_star]

        pj = torch.einsum("ntd,dk->ntk", x, Uj)
        pj = torch.einsum("ntk,dk->ntd", pj, Uj)
        pjp1 = torch.einsum("ntd,dk->ntk", x, Ujp1)
        pjp1 = torch.einsum("ntk,dk->ntd", pjp1, Ujp1)

        diff = pjp1 - pj
        num = torch.linalg.norm(diff, dim=-1)
        den = torch.linalg.norm(x, dim=-1) + eps
        token_drift = token_drift + (num / den)

    token_drift = token_drift * attn_mask.to(dtype=torch.float32)

    return FlowOut(
        step_mag=step_mag,
        step_mag_c=step_mag_c,
        turn_angle=turn_angle,
        token_drift=token_drift,
        drift_win=drift_win,
        r_eta=r_eta,
        R_attn=R_attn,
        R_mlp=R_mlp,
        attn_mag=attn_mag,
        mlp_mag=mlp_mag,
        comp_mag=comp_mag,
    )