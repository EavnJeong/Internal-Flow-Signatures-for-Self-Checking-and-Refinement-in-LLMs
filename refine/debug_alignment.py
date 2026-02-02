# refine/debug_alignment.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def debug_culprit_alignment(
    culprit: Dict[str, Any],
    tokenizer=None,
    window: int = 8,
) -> None:
    """
    Checks whether culprit t lies in answer_mask region and prints nearby tokens.
    """
    t = int(culprit.get("t", -1))
    b = int(culprit.get("b", -1))
    prob = float(culprit.get("prob", float("nan")))
    path = culprit.get("path", None)

    input_ids = culprit.get("input_ids", None)      # (T_max,)
    answer_mask = culprit.get("answer_mask", None)  # (T_max,) bool

    print(f"[culprit] prob={prob:.4f} b={b} t={t} path={path}")

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        print("[warn] input_ids missing or not 1D")
        return
    if not isinstance(answer_mask, torch.Tensor) or answer_mask.ndim != 1:
        print("[warn] answer_mask missing or not 1D")
        return
    if t < 0 or t >= int(input_ids.numel()):
        print("[warn] t out of range for input_ids")
        return

    in_answer = bool(answer_mask[t].item())
    print(f"[align] answer_mask[t]={int(in_answer)}")

    lo = max(0, t - int(window))
    hi = min(int(input_ids.numel()), t + int(window) + 1)

    ids_slice = input_ids[lo:hi].tolist()
    am_slice = answer_mask[lo:hi].to(torch.long).tolist()

    if tokenizer is None:
        print(f"[tokens] ids[{lo}:{hi}] = {ids_slice}")
        print(f"[tokens] answer_mask[{lo}:{hi}] = {am_slice}")
        return

    toks = tokenizer.convert_ids_to_tokens(ids_slice)
    marked: List[str] = []
    for k, tk in enumerate(toks):
        pos = lo + k
        tag = "A" if am_slice[k] == 1 else "P"
        focus = "<" if pos == t else " "
        marked.append(f"{focus}{tag}:{tk}")

    print("[tokens] " + " ".join(marked))
