# refine/find_culprit_events.py
from __future__ import annotations

from typing import Any, Dict, List
import torch


def _masked_argmax_1d(x: torch.Tensor, valid: torch.Tensor) -> int:
    """
    x: (L,) float
    valid: (L,) bool
    Returns argmax index over valid positions.
    """
    if x.ndim != 1 or valid.ndim != 1:
        raise RuntimeError("Expected 1D tensors for masked argmax")

    if int(x.shape[0]) != int(valid.shape[0]):
        raise RuntimeError(f"Shape mismatch: x={tuple(x.shape)} valid={tuple(valid.shape)}")

    if not bool(valid.any().item()):
        return -1

    x2 = x.clone()
    x2[~valid] = -float("inf")
    return int(torch.argmax(x2).item())


@torch.no_grad()
def find_culprit_event_per_sample(
    kept_samples: List[Dict[str, Any]],
    require_pred1: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each sample dict in kept_samples:
      - culprit j = masked argmax over logits_evt with evt_valid
      - (b,t) from evt_b, evt_t at j
    Returns one record per sample, no sorting, no top_k slicing.
    """

    culprits: List[Dict[str, Any]] = []

    for si, s in enumerate(kept_samples):
        logits_evt = s.get("logits_evt", None)     # (L,)
        evt_valid = s.get("evt_valid", None)       # (L,)
        evt_b = s.get("evt_b", None)               # (L,)
        evt_t = s.get("evt_t", None)               # (L,)
        prob = s.get("validator_prob", None)       # scalar or (1,)
        pred = s.get("validator_pred", None)       # scalar or (1,)

        if require_pred1:
            if isinstance(pred, torch.Tensor):
                pred_val = int(pred.item()) if pred.ndim == 0 else int(pred.view(-1)[0].item())
            elif pred is None:
                raise RuntimeError("sample has no validator_pred but require_pred1=True")
            else:
                pred_val = int(pred)
            if pred_val != 1:
                continue

        if not (isinstance(logits_evt, torch.Tensor) and logits_evt.ndim == 1):
            raise RuntimeError("sample['logits_evt'] must be a 1D tensor (L,)")
        if not (isinstance(evt_valid, torch.Tensor) and evt_valid.ndim == 1):
            raise RuntimeError("sample['evt_valid'] must be a 1D tensor (L,)")
        if not (isinstance(evt_b, torch.Tensor) and evt_b.ndim == 1):
            raise RuntimeError("sample['evt_b'] must be a 1D tensor (L,)")
        if not (isinstance(evt_t, torch.Tensor) and evt_t.ndim == 1):
            raise RuntimeError("sample['evt_t'] must be a 1D tensor (L,)")

        L = int(logits_evt.shape[0])
        if int(evt_valid.shape[0]) != L:
            raise RuntimeError(f"evt_valid length mismatch: L={L} evt_valid={int(evt_valid.shape[0])}")
        if int(evt_b.numel()) != L or int(evt_t.numel()) != L:
            raise RuntimeError(f"evt_b/evt_t length mismatch: L={L} evt_b={int(evt_b.numel())} evt_t={int(evt_t.numel())}")

        j = _masked_argmax_1d(logits_evt, evt_valid)
        hit = (j >= 0) and bool(evt_valid[j].item())

        b_val = int(evt_b[j].item()) if j >= 0 else -1
        t_val = int(evt_t[j].item()) if j >= 0 else -1

        prob_val = float("nan")
        if isinstance(prob, torch.Tensor):
            prob_val = float(prob.item()) if prob.ndim == 0 else float(prob.view(-1)[0].item())
        elif prob is not None:
            prob_val = float(prob)

        label = s.get("label", None)
        if label is None:
            y = s.get("labels", None)
            if isinstance(y, torch.Tensor):
                label = int(y.item()) if y.ndim == 0 else int(y.view(-1)[0].item())

        rec: Dict[str, Any] = {
            "sample_index": int(si),
            "prob": float(prob_val),
            "j": int(j),
            "b": int(b_val),
            "t": int(t_val),
            "hit": bool(hit),
            "label": label,
            "path": s.get("path", None),
            "prompt": s.get("prompt", None),
            "generated_text": s.get("generated_text", None),
        }

        input_ids = s.get("input_ids", None)
        if isinstance(input_ids, torch.Tensor):
            rec["input_ids"] = input_ids

        answer_mask = s.get("answer_mask", None)
        if isinstance(answer_mask, torch.Tensor):
            rec["answer_mask"] = answer_mask

        culprits.append(rec)

    return culprits
