# refine/utils.py
from typing import Any, Dict, List, Optional
import copy
import torch
from validators.utils import masked_logsumexp_pool, masked_max_pool
from validators.model import FlowGRUValidator


def _build_model_from_ckpt(ckpt: Dict[str, Any], feat_dim: int, device: torch.device) -> FlowGRUValidator:
    ckpt_args = ckpt.get("args", {}) or {}
    model = FlowGRUValidator(
        feat_dim=feat_dim,
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        embed_dim=int(ckpt_args.get("embed_dim", 128)),
        enc_hidden_dim=int(ckpt_args.get("enc_hidden_dim", 256)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        gru_layers=int(ckpt_args.get("gru_layers", 1)),
        enc_layers=int(ckpt_args.get("enc_layers", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def _maybe_to_cpu(v: Any, store_on_cpu: bool) -> Any:
    if not store_on_cpu:
        return v
    if isinstance(v, torch.Tensor):
        return v.detach().to("cpu")
    return v


def _subset_batch_by_samples(batch: Dict[str, Any], idx: List[int]) -> Dict[str, Any]:
    """
    Keep only per-sample fields whose first dimension equals M.
    Do not touch shared fields like evt_b, evt_t (length L).
    """
    if len(idx) == 0:
        return {}

    y = batch.get("labels", None)
    if not isinstance(y, torch.Tensor) or y.ndim == 0:
        raise RuntimeError("batch['labels'] must be a 1D tensor to infer M")
    M = int(y.shape[0])

    idx_t = torch.tensor(idx, dtype=torch.long)

    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.ndim >= 1 and int(v.shape[0]) == M:
                out[k] = v.index_select(0, idx_t.to(v.device))
            else:
                out[k] = v
            continue

        if isinstance(v, list):
            if len(v) == M:
                out[k] = [v[i] for i in idx]
            else:
                out[k] = v
            continue

        if isinstance(v, tuple):
            if len(v) == M:
                out[k] = tuple(v[i] for i in idx)
            else:
                out[k] = v
            continue

        out[k] = v

    return out


def _split_batch_dict_into_samples(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a filtered batch dict into independent sample dicts.
    Any field with leading dimension M is sliced per sample.
    Shared fields are copied to avoid accidental mutation sharing.
    """
    y = batch.get("labels", None)
    if not isinstance(y, torch.Tensor) or y.ndim != 1:
        raise RuntimeError("batch['labels'] must be a 1D tensor for splitting")
    M = int(y.shape[0])

    samples: List[Dict[str, Any]] = []
    for i in range(M):
        s: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and int(v.shape[0]) == M:
                s[k] = v[i]
                continue
            if isinstance(v, list) and len(v) == M:
                s[k] = v[i]
                continue
            if isinstance(v, tuple) and len(v) == M:
                s[k] = v[i]
                continue

            if isinstance(v, dict):
                s[k] = copy.deepcopy(v)
            else:
                s[k] = v
        samples.append(s)

    return samples


@torch.no_grad()
def collect_pred1_samples(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    pool: str,
    amp: bool,
    threshold: float,
    max_keep_samples: Optional[int],
    store_on_cpu: bool,
    debug_every: int,
) -> List[Dict[str, Any]]:
    """
    Return independent per-sample dicts, only for validator_pred == 1.
    """
    kept_samples: List[Dict[str, Any]] = []
    n_kept = 0

    for it, batch in enumerate(loader):
        evt_x = batch["evt_x"].to(device=device, dtype=torch.float32)       # (M,L,F)
        evt_valid = batch["evt_valid"].to(device=device, dtype=torch.bool) # (M,L)
        y = batch["labels"].to(device=device, dtype=torch.float32)         # (M,)

        keep01 = (y == 0) | (y == 1)
        if not bool(keep01.any().item()):
            continue

        evt_x2 = evt_x[keep01]
        evt_valid2 = evt_valid[keep01]
        y2 = y[keep01]
        M2 = int(y2.shape[0])
        if M2 == 0:
            continue

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            out = model(evt_x=evt_x2, evt_valid=evt_valid2)
            logits_evt = out["logits"]  # (M2,L)

            if pool == "max":
                logits_bag = masked_max_pool(logits_evt, evt_valid2)  # (M2,)
            elif pool == "lse":
                logits_bag = masked_logsumexp_pool(logits_evt, evt_valid2)  # (M2,)
            else:
                raise ValueError(f"Unknown pool={pool}")

            probs = torch.sigmoid(logits_bag)  # (M2,)

        pred1 = probs >= float(threshold)
        if not bool(pred1.any().item()):
            if debug_every and debug_every > 0 and (it % int(debug_every) == 0):
                print(f"[collect] it={it} kept_samples={n_kept}")
            continue

        idx_local = pred1.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        idx_keep01 = keep01.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        idx_orig = [idx_keep01[j] for j in idx_local]

        out_batch = _subset_batch_by_samples(batch, idx_orig)

        probs_sel = probs.detach().cpu()[idx_local] if store_on_cpu else probs[idx_local]
        out_batch["validator_prob"] = probs_sel
        out_batch["validator_pred"] = torch.ones(len(idx_orig), dtype=torch.long)

        logits_evt_sel = logits_evt.detach().cpu()[idx_local] if store_on_cpu else logits_evt[idx_local]
        logits_bag_sel = logits_bag.detach().cpu()[idx_local] if store_on_cpu else logits_bag[idx_local]
        out_batch["logits_evt"] = logits_evt_sel
        out_batch["logits_bag"] = logits_bag_sel

        for k in list(out_batch.keys()):
            out_batch[k] = _maybe_to_cpu(out_batch[k], store_on_cpu)

        samples = _split_batch_dict_into_samples(out_batch)
        for s in samples:
            kept_samples.append(s)
            n_kept += 1
            if max_keep_samples is not None and n_kept >= int(max_keep_samples):
                return kept_samples

        if debug_every and debug_every > 0 and (it % int(debug_every) == 0):
            print(f"[collect] it={it} kept_samples={n_kept}")

    return kept_samples


def add_culprit_per_sample(
    samples: List[Dict[str, Any]],
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Attach culprit info without duplicating samples.
    Store top-k candidates inside each sample dict.
    """
    out: List[Dict[str, Any]] = []
    for s in samples:
        logits_evt = s["logits_evt"]      # (L,)
        evt_valid = s["evt_valid"]        # (L,)
        evt_t = s.get("evt_t", None)      # (L,) shared in original, kept here
        evt_b = s.get("evt_b", None)      # (L,) shared in original, kept here

        if isinstance(logits_evt, torch.Tensor) and logits_evt.ndim == 1:
            le = logits_evt
        else:
            raise RuntimeError("sample['logits_evt'] must be 1D per sample")

        if not (isinstance(evt_valid, torch.Tensor) and evt_valid.ndim == 1):
            raise RuntimeError("sample['evt_valid'] must be 1D per sample")

        valid_idx = torch.nonzero(evt_valid, as_tuple=False).view(-1)
        if valid_idx.numel() == 0:
            s["culprit_candidates"] = []
            out.append(s)
            continue

        scores = le.index_select(0, valid_idx)
        k = min(int(top_k), int(scores.numel()))
        topv, topj = torch.topk(scores, k=k, largest=True, sorted=True)
        top_l = valid_idx.index_select(0, topj)

        candidates = []
        for rank in range(k):
            l = int(top_l[rank].item())
            cand = {
                "l": l,
                "logit": float(le[l].item()),
            }
            if isinstance(evt_t, torch.Tensor) and evt_t.ndim == 1:
                cand["t"] = int(evt_t[l].item())
            if isinstance(evt_b, torch.Tensor) and evt_b.ndim == 1:
                cand["b"] = int(evt_b[l].item())
            candidates.append(cand)

        s["culprit_candidates"] = candidates
        if len(candidates) > 0:
            s["t"] = candidates[0].get("t", None)
            s["b"] = candidates[0].get("b", None)

        out.append(s)

    return out


def _safe_rate(labels: torch.Tensor, v: int) -> float:
    if labels.numel() == 0:
        return 0.0
    return (100.0 * (labels == v).sum().item() / float(labels.numel()))


def _find_sample_index(c0: Dict[str, Any]) -> Optional[int]:
    for k in ["sample_idx", "idx", "global_idx", "example_idx", "item_idx"]:
        if k in c0 and c0[k] is not None:
            try:
                return int(c0[k])
            except Exception:
                pass
    return None


def _build_judge_example(c0: Dict[str, Any], task: str) -> Dict[str, Any]:
    ctx = (
        c0.get("context")
        or c0.get("knowledge")
        or c0.get("document")
        or ""
    )

    if str(task).lower().strip() in ["dialogue", "dialog"]:
        q = (
            c0.get("dialogue_history")
            or c0.get("history")
            or c0.get("user_query")
            or c0.get("query")
            or c0.get("question")
            or ""
        )
        return {"dialogue_history": str(q), "context": str(ctx)}
    else:
        q = (
            c0.get("query")
            or c0.get("question")
            or c0.get("user_query")
            or ""
        )
        return {"query": str(q), "context": str(ctx)}
