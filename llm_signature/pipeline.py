# llm_signature/pipeline.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_max_len_global(model, tokenizer) -> int:
    max_len = 4096
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        try:
            max_len = int(model.config.max_position_embeddings)
        except Exception:
            pass
    elif hasattr(tokenizer, "model_max_length"):
        try:
            max_len = int(tokenizer.model_max_length)
        except Exception:
            pass
    return int(min(max_len, 65536))


def _norm_kind(mod: nn.Module) -> str:
    name = mod.__class__.__name__.lower()
    if "rms" in name:
        return "rmsnorm"
    if "layernorm" in name or "layer_norm" in name:
        return "layernorm"
    return "layernorm"


def _norm_eps(mod: nn.Module) -> float:
    for k in ["eps", "variance_epsilon", "epsilon"]:
        if hasattr(mod, k):
            try:
                return float(getattr(mod, k))
            except Exception:
                pass
    return 1e-5


def get_boundary_norm_cfg_by_depth(model: nn.Module, adapter) -> List[Optional[Dict[str, Any]]]:
    """
    English comment:
      Index b points to the boundary norm at depth b.
      cfg[0] is a placeholder.
    """
    blocks = adapter.iter_blocks(model)
    cfg: List[Optional[Dict[str, Any]]] = [None]
    for blk in blocks:
        norm = None
        for attr in ["input_layernorm", "ln_1", "norm1", "layernorm1"]:
            if hasattr(blk, attr):
                norm = getattr(blk, attr)
                break
        if norm is None or not hasattr(norm, "weight"):
            cfg.append(None)
            continue
        kind = _norm_kind(norm)
        w = norm.weight.detach().to("cpu").float()
        eps = _norm_eps(norm)
        cfg.append({"kind": kind, "weight": w, "eps": eps})
    return cfg


def build_prompt_prefixes(batch: Any, task: str) -> List[str]:
    from llm_signature.prompt import build_prompts_from_halueval

    return build_prompts_from_halueval(batch, task=task)


@torch.inference_mode()
def generate_answers(
    model,
    tokenizer,
    prompt_prefixes: List[str],
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], Dict[str, Any]]:
    from llm_signature.prompt import encode_prompts

    input_ids, attention_mask = encode_prompts(
        prompt_prefixes,
        tokenizer,
        device=device,
        max_length=max_length,
    )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    pref_lens = attention_mask.sum(dim=1).to(torch.long)  # (N,)
    answers: List[str] = []
    for i in range(out_ids.shape[0]):
        start = int(pref_lens[i].item())
        new_ids = out_ids[i, start:]
        txt = tokenizer.decode(new_ids, skip_special_tokens=True)
        answers.append(txt.strip())

    gen_meta = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    return answers, gen_meta


def filter_by_labels(
    batch: Any,
    prompt_prefixes: List[str],
    answers: List[str],
    labels: torch.Tensor,
    judge_meta: List[Dict[str, Any]],
    require_label: bool,
) -> Tuple[Any, List[str], List[str], torch.Tensor, List[Dict[str, Any]]]:
    if not require_label:
        return batch, prompt_prefixes, answers, labels, judge_meta

    keep = (labels >= 0)
    if int(keep.sum().item()) == 0:
        return batch, [], [], labels[:0], []

    keep_idx = keep.nonzero(as_tuple=False).view(-1).tolist()
    prompt_prefixes = [prompt_prefixes[i] for i in keep_idx]
    answers = [answers[i] for i in keep_idx]
    judge_meta = [judge_meta[i] for i in keep_idx]
    labels = labels[keep]

    if isinstance(batch, dict):
        new_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, list) and len(v) >= (max(keep_idx) + 1):
                new_batch[k] = [v[i] for i in keep_idx]
            else:
                new_batch[k] = v
        batch = new_batch

    return batch, prompt_prefixes, answers, labels, judge_meta


def make_teacher_forcing_inputs(
    tokenizer,
    prompt_prefixes: List[str],
    answers: List[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from llm_signature.prompt import encode_prompts

    full_texts: List[str] = []
    for p, a in zip(prompt_prefixes, answers):
        a2 = "" if a is None else str(a).strip()
        full_texts.append(p + (" " + a2 if a2 else ""))

    input_ids, attention_mask = encode_prompts(
        full_texts,
        tokenizer,
        device=device,
        max_length=max_length,
    )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    pref_tok = tokenizer(
        prompt_prefixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    pref_attn = pref_tok.get("attention_mask", None)
    if pref_attn is None:
        pref_attn = torch.ones_like(pref_tok["input_ids"], dtype=torch.long)
    pref_lens = pref_attn.sum(dim=1).to(torch.long).to(device)

    N, T = input_ids.shape
    idx = torch.arange(T, device=device).view(1, T).expand(N, T)
    answer_mask = (idx >= pref_lens.view(N, 1)) & attention_mask.to(torch.bool)

    return input_ids, attention_mask, answer_mask


def build_mask_for_signatures(
    tf_attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    mask_answer_only: bool,
    min_mask_tokens: int,
) -> torch.Tensor:
    tf_cpu = tf_attention_mask.detach().to("cpu")
    if tf_cpu.dtype != torch.bool:
        tf_cpu = tf_cpu.to(torch.bool)

    ans_cpu = answer_mask.detach().to("cpu")
    if ans_cpu.dtype != torch.bool:
        ans_cpu = ans_cpu.to(torch.bool)

    mask = ans_cpu if mask_answer_only else tf_cpu
    if int(mask.sum().item()) < int(min_mask_tokens):
        mask = tf_cpu
    return mask


def _fit_subspace_from_competitors(
    topk_ids: torch.Tensor,        # (B,N,T,topk)
    attention_mask: torch.Tensor,  # (N,T) bool
    W: torch.Tensor,               # (V,D) float32
    b0: int,
    b1: int,
    K: int,
    k: int,
    cap: int,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    B, N, T, topk = topk_ids.shape
    if topk < K + 1:
        raise ValueError("topk_ids last dim must be >= K+1")
    if b0 < 0 or b1 >= B or b0 > b1:
        raise ValueError("invalid window range")

    am = attention_mask.to(torch.bool)  # (N,T)

    yhat = topk_ids[b0 : b1 + 1, :, :, 0]           # (L,N,T)
    ycomp = topk_ids[b0 : b1 + 1, :, :, 1 : K + 1]  # (L,N,T,K)

    yhat2 = yhat.unsqueeze(-1).expand_as(ycomp)
    vm = am.unsqueeze(0).unsqueeze(-1).expand_as(ycomp)

    yhat_flat = yhat2[vm].reshape(-1)
    ycomp_flat = ycomp[vm].reshape(-1)

    M = int(yhat_flat.numel())
    if M == 0:
        Q, _ = torch.linalg.qr(torch.randn((W.shape[1], k), generator=g), mode="reduced")
        return Q.to(torch.float32)

    if M > cap:
        idx = torch.randperm(M, generator=g)[:cap]
        yhat_flat = yhat_flat[idx]
        ycomp_flat = ycomp_flat[idx]

    dirs = W[yhat_flat] - W[ycomp_flat]  # (M,D)
    _, _, Vh = torch.linalg.svd(dirs, full_matrices=False)

    r = min(int(k), int(Vh.shape[0]))
    Uj = Vh[:r].transpose(0, 1).contiguous()  # (D,r)

    if r < k:
        Q, _ = torch.linalg.qr(torch.randn((W.shape[1], k), generator=g), mode="reduced")
        Q = Q.to(torch.float32)
        Uj2 = torch.cat([Uj, Q[:, r:]], dim=1)
        Uj2, _ = torch.linalg.qr(Uj2, mode="reduced")
        return Uj2[:, :k].to(torch.float32)

    Uj, _ = torch.linalg.qr(Uj, mode="reduced")
    return Uj[:, :k].to(torch.float32)


def build_U_list(
    topk_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    W: torch.Tensor,
    windows: List[Tuple[int, int]],
    K: int,
    k: int,
    cap_per_window: int,
    seed: int,
) -> List[torch.Tensor]:
    U_list: List[torch.Tensor] = []
    for j, (b0, b1) in enumerate(windows):
        Uj = _fit_subspace_from_competitors(
            topk_ids=topk_ids,
            attention_mask=attention_mask.to(torch.bool),
            W=W,
            b0=int(b0),
            b1=int(b1),
            K=int(K),
            k=int(k),
            cap=int(cap_per_window),
            seed=int(seed) + 1000 * int(j),
        )
        U_list.append(Uj)
    return U_list


def build_extra_payload(
    batch: Any,
    prompt_prefixes: List[str],
    answers: List[str],
    labels: torch.Tensor,
    judge_meta: List[Dict[str, Any]],
    gen_meta: Dict[str, Any],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    meta = batch.get("meta", None) if isinstance(batch, dict) else None
    task = batch.get("task", None) if isinstance(batch, dict) else None
    return {
        "meta": meta,
        "task": task,
        "prompt": prompt_prefixes,
        "generated_text": answers,
        "judge_meta": judge_meta,
        "gen_cfg": gen_meta,
        "run_args": args,
        "labels": labels.detach().to("cpu"),
    }


def save_flow(
    out_dir: str,
    step: int,
    flow,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: Optional[torch.Tensor],
    windows: List[Tuple[int, int]],
    labels: torch.Tensor,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "step_mag": flow.step_mag.to(torch.float32),
        "step_mag_c": flow.step_mag_c.to(torch.float32),
        "turn_angle": flow.turn_angle.to(torch.float32),
        "token_drift": flow.token_drift.to(torch.float32),
        "drift_win": flow.drift_win.to(torch.float32),
        "r_eta": flow.r_eta.to(torch.float32),
        "R_attn": flow.R_attn.to(torch.float32),
        "R_mlp": flow.R_mlp.to(torch.float32),
        "attn_mag": flow.attn_mag.to(torch.float32),
        "mlp_mag": flow.mlp_mag.to(torch.float32),
        "comp_mag": flow.comp_mag.to(torch.float32),
        "input_ids": input_ids.detach().to("cpu"),
        "attention_mask": attention_mask.detach().to("cpu"),
        "answer_mask": answer_mask.detach().to("cpu") if isinstance(answer_mask, torch.Tensor) else None,
        "windows": windows,
        "labels": labels.detach().to("cpu"),
    }

    if isinstance(extra, dict):
        for k, v in extra.items():
            if k in payload:
                payload[f"extra_{k}"] = v
            else:
                payload[k] = v

    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"flow_{int(step):06d}.pt")
    torch.save(payload, path)
    return path
