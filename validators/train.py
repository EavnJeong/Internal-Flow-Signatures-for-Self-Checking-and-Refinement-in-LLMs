# validators/train.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from validators.utils import (
    masked_max_pool,
    masked_logsumexp_pool,
    _safe_auroc_from_score,
)

def run_epoch(
    loader,
    model: nn.Module,
    device: torch.device,
    pool: str,
    bce: nn.Module,
    amp: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
    debug_every: int = 0,
    tag: str = "train",
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    skipped_batches = 0
    grad_norm_val = 0.0

    # Per class counters
    correct_pos = 0.0
    correct_neg = 0.0
    count_pos = 0
    count_neg = 0

    probs_cpu: List[torch.Tensor] = []
    y_cpu: List[torch.Tensor] = []

    for it, batch in enumerate(loader):
        evt_x = batch["evt_x"].to(device=device, dtype=torch.float32)          # (M,L,F)
        evt_valid = batch["evt_valid"].to(device=device, dtype=torch.bool)    # (M,L)
        y = batch["labels"].to(device=device, dtype=torch.float32)            # (M,)

        keep = (y == 0) | (y == 1)
        if not bool(keep.any().item()):
            skipped_batches += 1
            continue

        evt_x = evt_x[keep]
        evt_valid = evt_valid[keep]
        y = y[keep]

        M = int(y.shape[0])
        if M == 0:
            skipped_batches += 1
            continue

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            out = model(evt_x=evt_x, evt_valid=evt_valid)
            logits_evt = out["logits"]  # (M,L)

            if pool == "max":
                logits_bag = masked_max_pool(logits_evt, evt_valid)  # (M,)
            elif pool == "lse":
                logits_bag = masked_logsumexp_pool(logits_evt, evt_valid)  # (M,)
            else:
                raise ValueError(f"Unknown pool={pool}")

            loss = bce(logits_bag, y)

        if train_mode:
            if scaler is not None and (amp and device.type == "cuda"):
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm_val = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)).item())
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    grad_norm_val = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)).item())
                optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits_bag)
            pred = (probs >= 0.5).to(torch.long)
            y_long = y.to(torch.long)

            correct = (pred == y_long).to(torch.float32).sum().item()

            total_loss += float(loss.item()) * M
            total_correct += float(correct)
            total_count += int(M)

            pos_mask = (y_long == 1)
            neg_mask = (y_long == 0)

            if bool(pos_mask.any().item()):
                correct_pos += float((pred[pos_mask] == 1).to(torch.float32).sum().item())
                count_pos += int(pos_mask.sum().item())

            if bool(neg_mask.any().item()):
                correct_neg += float((pred[neg_mask] == 0).to(torch.float32).sum().item())
                count_neg += int(neg_mask.sum().item())

            probs_cpu.append(probs.detach().to("cpu"))
            y_cpu.append(y_long.detach().to("cpu"))

        if debug_every and debug_every > 0 and (it % int(debug_every) == 0):
            with torch.no_grad():
                pos_rate = float((y_long == 1).to(torch.float32).mean().item())
                valid_rate = float(evt_valid.to(torch.float32).mean().item())
                any_nan_prob = bool(torch.isnan(probs).any().item())
                print(
                    f"[{tag}] it={it} M={M} pos_rate={pos_rate:.3f} valid_rate={valid_rate:.3f} "
                    f"loss={float(loss.item()):.4f} nan_prob={any_nan_prob}"
                )

    if total_count <= 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "acc_pos": float("nan"),
            "acc_neg": float("nan"),
            "auroc": float("nan"),
            "n": 0.0,
            "n_pos": 0.0,
            "n_neg": 0.0,
            "skipped_batches": float(skipped_batches),
            "grad_norm": float(grad_norm_val),
        }

    probs_all = torch.cat(probs_cpu, dim=0)
    y_all = torch.cat(y_cpu, dim=0)

    acc = total_correct / float(total_count)
    acc_pos = (correct_pos / float(count_pos)) if count_pos > 0 else float("nan")
    acc_neg = (correct_neg / float(count_neg)) if count_neg > 0 else float("nan")

    auroc = _safe_auroc_from_score(probs_all, y_all)

    return {
        "loss": total_loss / float(total_count),
        "acc": float(acc),
        "acc_halluci": float(acc_pos),
        "acc_normal": float(acc_neg),
        "auroc": float(auroc),
        "skipped_batches": float(skipped_batches),
        "grad_norm": float(grad_norm_val),
        "acc_sum": float(acc_pos + acc_neg)
    }
