# validators/utils.py
import torch
import random
import os
import argparse
from typing import List, Dict, Any, Optional
import wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _quantiles(x: torch.Tensor, qs: List[float]) -> List[float]:
    x = x.detach().to(torch.float32).flatten()
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    return [float(torch.quantile(x, q).item()) for q in qs]


def _pearsonr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.detach().to(torch.float32).flatten()
    y = y.detach().to(torch.float32).flatten()
    n = int(min(x.numel(), y.numel()))
    if n <= 1:
        return float("nan")
    x = x[:n]
    y = y[:n]
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.std(unbiased=False) * y.std(unbiased=False) + eps)
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((x * y).mean().item() / float(denom.item()))


def _brier(prob: torch.Tensor, y01: torch.Tensor) -> float:
    prob = prob.detach().to(torch.float32).flatten()
    y01 = y01.detach().to(torch.float32).flatten()
    if prob.numel() == 0:
        return float("nan")
    return float(((prob - y01) ** 2).mean().item())


def _rate_near(prob: torch.Tensor, lo: float = 0.01, hi: float = 0.99) -> Dict[str, float]:
    prob = prob.detach().to(torch.float32).flatten()
    if prob.numel() == 0:
        return {"near0": float("nan"), "near1": float("nan")}
    return {
        "near0": float((prob <= lo).to(torch.float32).mean().item()),
        "near1": float((prob >= hi).to(torch.float32).mean().item()),
    }


def _safe_auroc_from_score(score: torch.Tensor, labels: torch.Tensor) -> float:
    score = score.detach().to(torch.float32).flatten()
    labels = labels.detach().to(torch.long).flatten()
    return binary_auroc(score, labels)


def _short_path(p: str, keep: int = 2) -> str:
    parts = str(p).split(os.sep)
    if len(parts) <= keep:
        return str(p)
    return os.sep.join(parts[-keep:])


def _safe_rate(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.to(torch.float32).mean().item())


def get_evt_mask(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    English comment:
    New flow collate already provides evt_valid (M,L).
    We keep this function to centralize masking decisions.
    """
    evt_valid = batch.get("evt_valid", None)
    if evt_valid is None:
        raise RuntimeError("batch must contain evt_valid (M,L)")
    evt_mask = evt_valid.to(device=device, dtype=torch.bool)
    return evt_mask


def masked_max_pool(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return masked.max(dim=1).values


def masked_logsumexp_pool(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return torch.logsumexp(masked, dim=1)


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    English comment:
    AUROC for binary labels in {0,1}. Returns NaN if only one class exists.
    """
    scores = scores.detach().flatten().to(torch.float32)
    labels = labels.detach().flatten().to(torch.long)

    ok = (labels == 0) | (labels == 1)
    scores = scores[ok]
    labels = labels[ok]

    n = int(labels.numel())
    if n == 0:
        return float("nan")

    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_scores, order = torch.sort(scores, descending=False)
    sorted_labels = labels[order]

    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    avg_ranks = (starts + 1 + ends).to(torch.float32) / 2.0

    ranks_sorted = torch.repeat_interleave(avg_ranks, counts)
    sum_ranks_pos = ranks_sorted[sorted_labels == 1].sum()

    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (float(n_pos) * float(n_neg))
    return float(auc.item())


def init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    if wandb is None:
        print("[warn] wandb not installed. continuing without wandb.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=args.wandb_run_name if args.wandb_run_name else None,
        tags=args.wandb_tags if args.wandb_tags else None,
        notes=args.wandb_notes if args.wandb_notes else None,
        config=vars(args),
    )
    return run


def _batch_debug_print(tag: str, batch: Dict[str, Any], evt_mask: torch.Tensor, sample_logit: torch.Tensor) -> None:
    evt_x = batch["evt_x"]
    labels = batch["labels"]

    M, L, F = evt_x.shape
    valid_rate = _safe_rate(evt_mask.to(torch.float32))
    any_rate = _safe_rate(evt_mask.any(dim=1).to(torch.float32))

    labels_cpu = labels.detach().to("cpu")
    n0 = int((labels_cpu == 0).sum().item())
    n1 = int((labels_cpu == 1).sum().item())
    not01 = int(((labels_cpu != 0) & (labels_cpu != 1)).sum().item())

    logit = sample_logit.detach()
    has_nan = torch.isnan(logit).any().item()
    has_inf = torch.isinf(logit).any().item()

    with torch.no_grad():
        valid_cnt = evt_mask.sum(dim=1).to(torch.float32)  # (M,)
        qv = _quantiles(valid_cnt, [0.0, 0.5, 0.9, 1.0])

        prob = torch.sigmoid(sample_logit)
        qp = _quantiles(prob, [0.01, 0.1, 0.5, 0.9, 0.99])
        ql = _quantiles(sample_logit, [0.01, 0.1, 0.5, 0.9, 0.99])

        ok = ((labels == 0) | (labels == 1)) & evt_mask.any(dim=1)
        y01 = labels.to(torch.float32)
        brier = _brier(prob[ok], y01[ok]) if int(ok.sum().item()) > 0 else float("nan")
        near = _rate_near(prob[ok]) if int(ok.sum().item()) > 0 else {"near0": float("nan"), "near1": float("nan")}

        len_T = batch.get("len_T", None)
        if len_T is not None:
            len_T = len_T.to(sample_logit.device).to(torch.float32)
            corr_len = _pearsonr(prob[ok], len_T[ok]) if int(ok.sum().item()) > 1 else float("nan")
        else:
            corr_len = float("nan")

        corr_valid = _pearsonr(prob[ok], valid_cnt[ok]) if int(ok.sum().item()) > 1 else float("nan")

    print(
        f"[{tag}] M={M} L={L} F={F} "
        f"valid_rate={valid_rate:.3f} any_rate={any_rate:.3f} "
        f"valid_cnt(q0={qv[0]:.0f}, q50={qv[1]:.0f}, q90={qv[2]:.0f}, q100={qv[3]:.0f}) "
        f"labels(n0={n0}, n1={n1}, not01={not01}) "
        f"logit(q01={ql[0]:.3f}, q10={ql[1]:.3f}, q50={ql[2]:.3f}, q90={ql[3]:.3f}, q99={ql[4]:.3f}, nan={int(has_nan)}, inf={int(has_inf)}) "
        f"prob(q01={qp[0]:.3f}, q10={qp[1]:.3f}, q50={qp[2]:.3f}, q90={qp[3]:.3f}, q99={qp[4]:.3f}) "
        f"brier={brier:.4f} near0={near['near0']:.3f} near1={near['near1']:.3f} "
        f"corr(prob,lenT)={corr_len:.3f} corr(prob,validCnt)={corr_valid:.3f}"
    )