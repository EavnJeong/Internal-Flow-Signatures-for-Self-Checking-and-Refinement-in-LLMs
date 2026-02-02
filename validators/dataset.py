# validators/dataset.py
import os
import json
import glob
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np


def _list_flow_files(flow_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(flow_dir, "flow_*.pt")))


def _stable_shuffle(items: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def _split_8_2(files: List[str]) -> Tuple[List[str], List[str]]:
    n = len(files)
    n_tr = int(0.8 * n)
    return files[:n_tr], files[n_tr:]


def _maybe_load_split(cache_path: str) -> Optional[Dict[str, Any]]:
    if not cache_path:
        return None
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_split(cache_path: str, payload: Dict[str, Any]) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)


def _as_float_cpu(x: Any) -> Optional[torch.Tensor]:
    if x is None or (not torch.is_tensor(x)):
        return None
    return x.detach().to("cpu", dtype=torch.float32)


def _as_long_cpu(x: Any) -> Optional[torch.Tensor]:
    if x is None or (not torch.is_tensor(x)):
        return None
    return x.detach().to("cpu", dtype=torch.long)


def _pad_depth(x: torch.Tensor, target_B: int) -> torch.Tensor:
    # x: (B0, N, T)
    B0, N, T = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if B0 == target_B:
        return x
    if B0 == target_B - 1:
        pad = torch.zeros((1, N, T), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)
    raise RuntimeError(f"Unexpected depth size: got {B0}, expected {target_B} or {target_B-1}")


def _broadcast_token_to_depth(x_tok: torch.Tensor, B: int) -> torch.Tensor:
    # x_tok: (N, T) -> (B, N, T)
    return x_tok.unsqueeze(0).expand(B, -1, -1).contiguous()


def _infer_labels(payload: Dict[str, Any]) -> torch.Tensor:
    # Returns labels as (N,) long. If missing, fill -1.
    y = payload.get("labels", None)
    if y is None:
        attn = _as_long_cpu(payload.get("attention_mask", None))
        if attn is None:
            raise RuntimeError("Missing attention_mask, cannot infer N for labels")
        N = int(attn.shape[0])
        return torch.full((N,), -1, dtype=torch.long)

    if torch.is_tensor(y):
        y = y.detach().to("cpu")
        if y.ndim == 0:
            attn = _as_long_cpu(payload.get("attention_mask", None))
            if attn is None:
                raise RuntimeError("Missing attention_mask, cannot broadcast scalar label")
            N = int(attn.shape[0])
            return torch.full((N,), int(y.item()), dtype=torch.long)
        return y.to(torch.long)

    if isinstance(y, list):
        return torch.tensor([int(v) for v in y], dtype=torch.long)

    try:
        return torch.tensor([int(y)], dtype=torch.long)
    except Exception:
        return torch.tensor([-1], dtype=torch.long)


def _count_binary_labels_in_file(path: str) -> Tuple[int, int, int]:
    """
    English comment: Counts labels in one flow file.
    Returns: (n0, n1, n_other)
    """
    payload = torch.load(path, map_location="cpu")
    y = _infer_labels(payload)  # (N,)
    y = y.to(torch.long).view(-1)

    n0 = int((y == 0).sum().item())
    n1 = int((y == 1).sum().item())
    n_other = int((~((y == 0) | (y == 1))).sum().item())
    return n0, n1, n_other


def _make_balanced_file_sampler(
    train_files: List[str],
    seed: int,
    drop_non_binary_labels: bool,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """
    English comment: Builds a file-level sampler that oversamples files rich in the global minority class.
    """
    rng = np.random.RandomState(int(seed))

    counts = []
    total0, total1, total_other = 0, 0, 0
    for p in train_files:
        n0, n1, no = _count_binary_labels_in_file(p)
        if drop_non_binary_labels:
            # Keep only 0/1 counts for global stats
            pass
        total0 += n0
        total1 += n1
        total_other += no
        counts.append((n0, n1, no))

    # Decide minority class from global counts (binary only)
    # If one side is zero, sampler cannot fix it; we still return uniform-ish weights.
    if total0 == 0 and total1 == 0:
        minority = 1
    else:
        minority = 0 if total0 < total1 else 1

    weights = []
    eps = 1e-3
    base = 0.05  # ensures every file still has non-zero probability
    for (n0, n1, _) in counts:
        n_bin = n0 + n1
        if n_bin <= 0:
            w = base
        else:
            n_min = n0 if minority == 0 else n1
            ratio = float(n_min) / float(n_bin + eps)
            w = base + ratio
        weights.append(w)

    weights_t = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_t,
        num_samples=len(train_files),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    info = {
        "train_files": len(train_files),
        "global_n0": total0,
        "global_n1": total1,
        "global_n_other": total_other,
        "minority_class": minority,
        "mean_weight": float(np.mean(weights)),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
    }
    return sampler, info


class FlowFileDataset(Dataset):
    """
    One item corresponds to one flow_*.pt file.
    The file internally contains N samples.

    __getitem__ returns:
      x_grid: (N, B_eff, T, F)
      attention_mask: (N, T) bool
      labels: (N,) long
      len_T: (N,) long
      len_B: (N,) long
      meta/task: original lists if present
      path: str
    """
    def __init__(
        self,
        files: List[str],
        feature_names: List[str],
        cache_size: int = 2,
        drop_non_binary_labels: bool = False,
        debug_print_once: bool = False,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.feature_names = list(feature_names)
        self.cache_size = int(cache_size)
        self.drop_non_binary_labels = bool(drop_non_binary_labels)
        self.debug_print_once = bool(debug_print_once)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_keys: List[str] = []
        self._did_print = False

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: str) -> Dict[str, Any]:
        if path in self._cache:
            return self._cache[path]
        payload = torch.load(path, map_location="cpu")
        self._cache[path] = payload
        self._cache_keys.append(path)
        if len(self._cache_keys) > self.cache_size:
            old = self._cache_keys.pop(0)
            if old in self._cache:
                del self._cache[old]
        return payload

    def _build_x_grid(self, payload: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step_mag = _as_float_cpu(payload.get("step_mag", None))
        if step_mag is None or step_mag.ndim != 3:
            raise RuntimeError("payload.step_mag must exist with shape (B_eff, N, T)")

        B_eff = int(step_mag.shape[0])
        N = int(step_mag.shape[1])
        T = int(step_mag.shape[2])

        attn = _as_long_cpu(payload.get("attention_mask", None))
        if attn is None or attn.ndim != 2:
            raise RuntimeError("payload.attention_mask must exist with shape (N, T)")
        attn_bool = attn.to(torch.bool)

        labels = _infer_labels(payload)
        if labels.numel() != N:
            # If labels came as scalar or weird, fix to length N
            if labels.numel() == 1:
                labels = labels.expand(N).contiguous()
            else:
                raise RuntimeError(f"labels length mismatch: got {labels.numel()}, expected {N}")

        def get_depth_feature(name: str) -> torch.Tensor:
            x = _as_float_cpu(payload.get(name, None))
            if x is None:
                return torch.zeros((B_eff, N, T), dtype=torch.float32)
            if x.ndim != 3:
                raise RuntimeError(f"Unsupported shape for {name}: {tuple(x.shape)}")
            # (B?, N, T)
            return _pad_depth(x, B_eff)

        def get_token_feature(name: str) -> torch.Tensor:
            x = _as_float_cpu(payload.get(name, None))
            if x is None:
                return torch.zeros((B_eff, N, T), dtype=torch.float32)
            if x.ndim != 2:
                raise RuntimeError(f"Unsupported shape for {name}: {tuple(x.shape)}")
            # (N, T) -> (B, N, T)
            return _broadcast_token_to_depth(x, B_eff)

        feat_cols: List[torch.Tensor] = []
        for nm in self.feature_names:
            if nm in ["token_drift", "R_attn", "R_mlp"]:
                col = get_token_feature(nm)
            else:
                col = get_depth_feature(nm)
            feat_cols.append(col)

        # (F, B, N, T) -> (N, B, T, F)
        x_stack = torch.stack(feat_cols, dim=0).permute(2, 1, 3, 0).contiguous()

        # Zero out padded tokens
        x_stack = x_stack * attn_bool.view(N, 1, T, 1).to(torch.float32)

        if self.drop_non_binary_labels:
            keep = (labels == 0) | (labels == 1)
            x_stack = x_stack[keep]
            attn_bool = attn_bool[keep]
            labels = labels[keep]

        if self.debug_print_once and (not self._did_print):
            self._did_print = True
            print(f"[ds] file contains N={int(x_stack.shape[0])} samples, B_eff={B_eff}, T={T}, F={int(x_stack.shape[-1])}")
            print(f"[ds] attn_mask any_rate={float(attn_bool.any(dim=1).float().mean().item()):.3f} "
                  f"token_valid_rate={float(attn_bool.float().mean().item()):.3f}")
            uniq = torch.unique(labels)
            print(f"[ds] label uniq={uniq.tolist()}")

        return x_stack, attn_bool, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        payload = self._load(path)

        x_grid, attn_bool, labels = self._build_x_grid(payload)  # (N, B, T, F), (N, T), (N,)

        N = int(x_grid.shape[0])
        B_eff = int(x_grid.shape[1])
        T = int(x_grid.shape[2])

        len_T = attn_bool.sum(dim=1).to(torch.long)                 # (N,)
        len_B = torch.full((N,), B_eff, dtype=torch.long)           # (N,)

        meta = payload.get("meta", None)
        task = payload.get("task", None)

        input_ids = payload.get("input_ids", None)        # expected (N, T) long tensor or None
        answer_mask = payload.get("answer_mask", None)    # expected (N, T) long/bool tensor or None
        prompt = payload.get("prompt", None)              # expected list[str] length N or str
        generated_text = payload.get("generated_text", None)  # expected list[str] length N or str

        return {
            "x_grid": x_grid,
            "attention_mask": attn_bool,
            "labels": labels,
            "len_T": len_T,
            "len_B": len_B,
            "meta": meta,
            "task": task,
            "path": path,
            "N_file": N,

            "prompt": prompt,
            "generated_text": generated_text,
            "input_ids": input_ids,
            "answer_mask": answer_mask,
        }

def flow_file_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # English comments only

    if len(batch) == 0:
        raise RuntimeError("Empty batch")

    # Infer feature dim and max(B,T) across files
    F0 = int(batch[0]["x_grid"].shape[-1])
    B_max = 0
    T_max = 0
    for item in batch:
        x = item["x_grid"]  # (N, B, T, F)
        if int(x.shape[-1]) != F0:
            raise RuntimeError("Feature dim mismatch across files")
        B_max = max(B_max, int(x.shape[1]))
        T_max = max(T_max, int(x.shape[2]))

    # Buffers for padded tensors
    xs: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    len_Ts: List[torch.Tensor] = []
    len_Bs: List[torch.Tensor] = []

    # Per-sample metadata
    meta_all: List[Any] = []
    task_all: List[Any] = []
    path_all: List[str] = []

    prompt_all: List[Any] = []
    gen_text_all: List[Any] = []

    input_ids_all: List[torch.Tensor] = []
    answer_mask_all: List[torch.Tensor] = []

    pad_id = 0

    for item in batch:
        x = item["x_grid"]                 # (N, B, T, F)
        m_tok = item["attention_mask"]     # (N, T) bool
        y = item["labels"]                 # (N,)
        lt = item["len_T"]                 # (N,)
        lb = item["len_B"]                 # (N,) (usually filled with B_eff)

        N_i = int(x.shape[0])
        B_i = int(x.shape[1])
        T_i = int(x.shape[2])
        F_i = int(x.shape[3])

        if F_i != F0:
            raise RuntimeError("Feature dim mismatch across files")

        # Pad x to (N_i, B_max, T_max, F0)
        x_pad = torch.zeros((N_i, B_max, T_max, F0), dtype=torch.float32)
        x_pad[:, :B_i, :T_i, :] = x.to(torch.float32)

        # Build valid mask over (B,T) from token mask
        # m_tok is (N_i, T_i). We expand to (N_i, B_i, T_i) and pad to (N_i, B_max, T_max).
        valid_pad = torch.zeros((N_i, B_max, T_max), dtype=torch.bool)
        m_bool = m_tok.to(torch.bool)
        valid_pad[:, :B_i, :T_i] = m_bool[:, None, :T_i].expand(N_i, B_i, T_i)

        xs.append(x_pad)
        valids.append(valid_pad)
        labels.append(y.to(torch.long))
        len_Ts.append(lt.to(torch.long))

        if torch.is_tensor(lb) and lb.numel() == N_i:
            len_Bs.append(lb.to(torch.long))
        else:
            len_Bs.append(torch.full((N_i,), B_i, dtype=torch.long))

        # Meta/task/path: expand to per-sample lists
        meta = item.get("meta", None)
        task = item.get("task", None)
        path = item.get("path", "")

        if isinstance(meta, list) and len(meta) == N_i:
            meta_all.extend(meta)
        else:
            meta_all.extend([meta] * N_i)

        if isinstance(task, list) and len(task) == N_i:
            task_all.extend(task)
        else:
            task_all.extend([task] * N_i)

        path_all.extend([path] * N_i)

        # Prompt / generated_text
        p = item.get("prompt", None)
        if isinstance(p, list) and len(p) == N_i:
            prompt_all.extend(p)
        else:
            prompt_all.extend([p] * N_i)

        g = item.get("generated_text", None)
        if isinstance(g, list) and len(g) == N_i:
            gen_text_all.extend(g)
        else:
            gen_text_all.extend([g] * N_i)

        # input_ids: pad to (N_i, T_max)
        ids = item.get("input_ids", None)
        if torch.is_tensor(ids):
            ids = ids.detach().to("cpu")
            if ids.ndim == 1:
                ids = ids.view(1, -1).expand(N_i, -1).contiguous()
            ids_pad = torch.full((N_i, T_max), pad_id, dtype=torch.long)
            tt = min(int(ids.shape[1]), int(T_max))
            ids_pad[:, :tt] = ids[:, :tt].to(torch.long)
        else:
            ids_pad = torch.full((N_i, T_max), pad_id, dtype=torch.long)
        input_ids_all.append(ids_pad)

        # answer_mask: pad to (N_i, T_max) bool
        am = item.get("answer_mask", None)
        if torch.is_tensor(am):
            am = am.detach().to("cpu")
            if am.ndim == 1:
                am = am.view(1, -1).expand(N_i, -1).contiguous()
            am_pad = torch.zeros((N_i, T_max), dtype=torch.bool)
            tt = min(int(am.shape[1]), int(T_max))
            am_pad[:, :tt] = am[:, :tt].to(torch.bool)
        else:
            am_pad = torch.zeros((N_i, T_max), dtype=torch.bool)
        answer_mask_all.append(am_pad)

    # Concatenate across files into a single sample batch
    x_all = torch.cat(xs, dim=0)            # (M, B_max, T_max, F0)
    v_all = torch.cat(valids, dim=0)        # (M, B_max, T_max)
    y_all = torch.cat(labels, dim=0)        # (M,)
    len_T_all = torch.cat(len_Ts, dim=0)    # (M,)
    len_B_all = torch.cat(len_Bs, dim=0)    # (M,)

    M = int(x_all.shape[0])
    L = int(B_max * T_max)

    # Flatten (B,T) -> L
    evt_x = x_all.reshape(M, L, F0)         # (M, L, F)
    evt_valid = v_all.reshape(M, L)         # (M, L)

    # Event indices (shared across batch)
    evt_b = torch.repeat_interleave(torch.arange(B_max, dtype=torch.long), T_max)  # (L,)
    evt_t = torch.arange(T_max, dtype=torch.long).repeat(B_max)                    # (L,)

    # Concatenate raw token fields
    input_ids_cat = torch.cat(input_ids_all, dim=0)        # (M, T_max)
    answer_mask_cat = torch.cat(answer_mask_all, dim=0)    # (M, T_max)

    return {
        "evt_x": evt_x,
        "evt_valid": evt_valid,
        "len_T": len_T_all,
        "len_B": len_B_all,
        "evt_t": evt_t,
        "evt_b": evt_b,
        "labels": y_all,
        "meta": meta_all,
        "task": task_all,
        "path": path_all,
        "M": M,
        "B_max": B_max,
        "T_max": T_max,
        "prompt": prompt_all,
        "generated_text": gen_text_all,
        "input_ids": input_ids_cat,
        "answer_mask": answer_mask_cat,
    }


def build_dataloader(args) -> Tuple[DataLoader, DataLoader]:
    """
    Here args.batch_size means number of files per batch.
    Total sample batch M equals (files_per_batch) times (N per file), if N is fixed.
    """
    flow_dir = os.path.join(args.flow_root, args.dataset, args.task, args.model)
    files = _list_flow_files(flow_dir)
    if len(files) == 0:
        raise RuntimeError(f"No flow files found in: {flow_dir}")

    if getattr(args, "max_files", None) is not None:
        files = files[: int(args.max_files)]

    cache_path = getattr(args, "split_cache", "") or ""
    split_payload = _maybe_load_split(cache_path)

    if split_payload is not None:
        tr_files = split_payload.get("train_files", [])
        te_files = split_payload.get("test_files", [])
        if len(tr_files) + len(te_files) == len(files) and set(tr_files + te_files) == set(files):
            train_files, test_files = tr_files, te_files
        else:
            split_payload = None

    if split_payload is None:
        use_files = files
        if getattr(args, "shuffle_files", False):
            use_files = _stable_shuffle(files, seed=int(args.seed))
        train_files, test_files = _split_8_2(use_files)
        _save_split(
            cache_path,
            {
                "flow_dir": flow_dir,
                "seed": int(args.seed),
                "shuffle_files": bool(getattr(args, "shuffle_files", False)),
                "train_files": train_files,
                "test_files": test_files,
            },
        )

    feature_names = [
        "step_mag",
        "step_mag_c",
        "turn_angle",
        "attn_mag",
        "mlp_mag",
        "comp_mag",
        "r_eta",
        "token_drift",
        "R_attn",
        "R_mlp",
    ]

    drop_non_binary = bool(getattr(args, "drop_non_binary_labels", False))
    train_ds = FlowFileDataset(
        files=train_files,
        feature_names=feature_names,
        cache_size=2,
        drop_non_binary_labels=drop_non_binary,
        debug_print_once=True,
    )
    test_ds = FlowFileDataset(
        files=test_files,
        feature_names=feature_names,
        cache_size=2,
        drop_non_binary_labels=drop_non_binary,
        debug_print_once=False,
    )

    use_balanced_sampler = bool(getattr(args, "balance_files", False))
    train_sampler = None
    if use_balanced_sampler:
        train_sampler, info = _make_balanced_file_sampler(
            train_files=train_files,
            seed=int(args.seed),
            drop_non_binary_labels=drop_non_binary,
        )
        print(f"[balance] file_sampler on, info={info}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=False if train_sampler is not None else bool(getattr(args, "shuffle_in_loader", False)),
        sampler=train_sampler,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", False)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
        collate_fn=flow_file_collate,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", False)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
        collate_fn=flow_file_collate,
        drop_last=False,
    )

    print(f"[split] n_files train={len(train_files)} test={len(test_files)} flow_dir={flow_dir}")
    print(f"[loader] files_per_batch={int(args.batch_size)} (batch_size counts files)")

    return train_loader, test_loader
