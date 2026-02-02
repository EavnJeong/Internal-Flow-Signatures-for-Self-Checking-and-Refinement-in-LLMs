# analyze_validator_stats.py
# English comments only

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt

from validators.dataset import build_dataloader
from validators.model import FlowGRUValidator
from validators.utils import masked_max_pool, masked_logsumexp_pool


FEATURE_NAMES = [
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

FEATURE_GROUPS = {
    "motion": ["step_mag", "step_mag_c", "turn_angle"],
    "attention": ["attn_mag", "R_attn"],
    "mlp": ["mlp_mag", "R_mlp"],
    "competitor": ["comp_mag"],
    "drift": ["token_drift", "r_eta"],
}

GROUP_ORDER = ["motion", "attention", "mlp", "competitor", "drift"]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _feature_index_map(feature_names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(feature_names)}


def _group_to_indices(groups: Dict[str, List[str]], feat_map: Dict[str, int]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for g, names in groups.items():
        idxs: List[int] = []
        for n in names:
            if n not in feat_map:
                raise RuntimeError(f"Unknown feature name in group {g}: {n}")
            idxs.append(int(feat_map[n]))
        out[g] = idxs
    return out


def _load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt or "args" not in ckpt:
        raise RuntimeError("Checkpoint must contain keys: model, args")
    return ckpt


def _build_model_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> Tuple[FlowGRUValidator, Dict[str, Any]]:
    a = ckpt["args"]
    feat_dim = int(a.get("feat_dim", 10))

    model = FlowGRUValidator(
        feat_dim=feat_dim,
        hidden_dim=int(a.get("hidden_dim", 256)),
        embed_dim=int(a.get("embed_dim", 128)),
        enc_hidden_dim=int(a.get("enc_hidden_dim", 256)),
        dropout=float(a.get("dropout", 0.1)),
        gru_layers=int(a.get("gru_layers", 1)),
        enc_layers=int(a.get("enc_layers", 2)),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, a


def _pool_logits(logits_evt: torch.Tensor, evt_valid: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "max":
        return masked_max_pool(logits_evt, evt_valid)
    if pool == "lse":
        return masked_logsumexp_pool(logits_evt, evt_valid)
    raise ValueError(f"Unknown pool: {pool}")


def _infer_logits_evt(
    model: FlowGRUValidator,
    evt_x: torch.Tensor,
    evt_valid: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        out = model(evt_x=evt_x, evt_valid=evt_valid)
        return out["logits"].detach()


def _predict_from_logits_evt(
    logits_evt: torch.Tensor,
    evt_valid: torch.Tensor,
    pool: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_bag = _pool_logits(logits_evt, evt_valid, pool=pool)
    prob = torch.sigmoid(logits_bag)
    pred = (prob >= 0.5).to(torch.long)
    return logits_bag.detach(), prob.detach(), pred.detach()


def _conf_mode(y_true: torch.Tensor, y_pred: torch.Tensor) -> List[str]:
    yt = y_true.to(torch.long)
    yp = y_pred.to(torch.long)
    out: List[str] = []
    for i in range(int(yt.numel())):
        t = int(yt[i].item())
        p = int(yp[i].item())
        if t == 1 and p == 1:
            out.append("tp")
        elif t == 0 and p == 1:
            out.append("fp")
        elif t == 0 and p == 0:
            out.append("tn")
        elif t == 1 and p == 0:
            out.append("fn")
        else:
            out.append("other")
    return out


def _max_pool_argmax(
    logits_evt: torch.Tensor,
    evt_valid: torch.Tensor,
) -> torch.Tensor:
    neg_inf = torch.finfo(logits_evt.dtype).min
    masked = torch.where(evt_valid, logits_evt, torch.full_like(logits_evt, neg_inf))
    return masked.argmax(dim=1)


def _gradxinput_attribution(
    model: FlowGRUValidator,
    evt_x: torch.Tensor,
    evt_valid: torch.Tensor,
    pool: str,
) -> torch.Tensor:
    prev_model_training = model.training
    prev_gru_training = model.gru.training
    prev_gru_dropout = float(model.gru.dropout)

    model.eval()
    model.gru.train(True)
    model.gru.dropout = 0.0

    try:
        x = evt_x.detach().clone().requires_grad_(True)
        out = model(evt_x=x, evt_valid=evt_valid)
        logits_evt = out["logits"]
        logits_bag = _pool_logits(logits_evt, evt_valid, pool=pool)

        target = logits_bag.sum()
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        target.backward()

        grad = x.grad.detach()
        attr = (grad * x.detach())
        attr = attr * evt_valid.to(attr.dtype).unsqueeze(-1)
        return attr.detach()

    finally:
        model.gru.dropout = prev_gru_dropout
        model.gru.train(prev_gru_training)
        model.train(prev_model_training)


def _scatter_depth_scores(
    pos_score: torch.Tensor,   # (M,L)
    evt_b: torch.Tensor,       # (L,)
    b_max: int,
) -> torch.Tensor:
    device = pos_score.device
    M, L = pos_score.shape
    idx = evt_b.to(device=device, dtype=torch.long).view(1, L).expand(M, L)
    out = torch.zeros((M, int(b_max)), device=device, dtype=pos_score.dtype)
    out.scatter_add_(1, idx, pos_score)
    return out


def _scatter_token_scores(
    pos_score: torch.Tensor,   # (M,L)
    evt_t: torch.Tensor,       # (L,)
    t_max: int,
) -> torch.Tensor:
    device = pos_score.device
    M, L = pos_score.shape
    idx = evt_t.to(device=device, dtype=torch.long).view(1, L).expand(M, L)
    out = torch.zeros((M, int(t_max)), device=device, dtype=pos_score.dtype)
    out.scatter_add_(1, idx, pos_score)
    return out


def _safe_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = x.sum(dim=1, keepdim=True) + float(eps)
    return x / denom


# Add below helpers near other helpers

def _occlusion_group_delta_logits_bag(
    model: FlowGRUValidator,
    evt_x: torch.Tensor,            # (M,L,F)
    evt_valid: torch.Tensor,        # (M,L)
    pool: str,
    group_indices: Dict[str, List[int]],
    group_order: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # English comment:
    # Return base logits_bag: (M,) and occlusion delta per group: (M,G)
    with torch.no_grad():
        out = model(evt_x=evt_x, evt_valid=evt_valid)
        logits_evt = out["logits"]
        base_logits_bag = _pool_logits(logits_evt, evt_valid, pool=pool)

    M = int(evt_x.shape[0])
    G = int(len(group_order))
    deltas = torch.zeros((M, G), device=evt_x.device, dtype=torch.float32)

    for gi, g in enumerate(group_order):
        idxs = group_indices[g]
        x2 = evt_x.clone()
        x2[:, :, idxs] = 0.0
        with torch.no_grad():
            out2 = model(evt_x=x2, evt_valid=evt_valid)
            logits_bag2 = _pool_logits(out2["logits"], evt_valid, pool=pool)
        deltas[:, gi] = (base_logits_bag - logits_bag2).to(torch.float32)

    return base_logits_bag.detach(), deltas.detach()


def _merge_modes_weighted(
    report: Dict[str, Any],
    modes: List[str],
    key: str,
    group_names: List[str],
) -> Dict[str, float]:
    # English comment:
    # Weighted merge across modes using sample counts
    total = 0
    acc = {g: 0.0 for g in group_names}
    for m in modes:
        n = int(report["count"].get(m, 0))
        if n <= 0:
            continue
        total += n
        vals = report.get(key, {}).get(m, {})
        for g in group_names:
            acc[g] += float(vals.get(g, 0.0)) * float(n)
    if total <= 0:
        return {g: float("nan") for g in group_names}
    return {g: float(acc[g] / float(total)) for g in group_names}


def _merge_hist_weighted(
    report: Dict[str, Any],
    modes: List[str],
    key: str,
) -> List[float]:
    # English comment:
    # Sum histograms across modes
    out: List[float] = []
    for m in modes:
        h = report.get(key, {}).get(m, [])
        if len(h) > len(out):
            out = out + [0.0] * (len(h) - len(out))
        for i in range(len(h)):
            out[i] += float(h[i])
    return out


def _js_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    # English comment:
    # Jensen Shannon divergence in nats
    import math
    B = max(len(p), len(q))
    pp = [0.0] * B
    qq = [0.0] * B
    for i in range(B):
        pp[i] = float(p[i]) if i < len(p) else 0.0
        qq[i] = float(q[i]) if i < len(q) else 0.0
    sp = sum(pp) + eps
    sq = sum(qq) + eps
    pp = [x / sp for x in pp]
    qq = [x / sq for x in qq]
    mm = [0.5 * (pp[i] + qq[i]) for i in range(B)]

    def _kl(a: List[float], b: List[float]) -> float:
        s = 0.0
        for i in range(B):
            ai = a[i] + eps
            bi = b[i] + eps
            s += ai * math.log(ai / bi)
        return s

    return 0.5 * _kl(pp, mm) + 0.5 * _kl(qq, mm)


def _plot_bar_signed(out_path: str, title: str, vals: Dict[str, float], group_names: List[str]) -> None:
    xs = list(range(len(group_names)))
    ys = [float(vals.get(g, 0.0)) for g in group_names]
    ymax = max([abs(y) for y in ys] + [1e-6])

    plt.figure()
    plt.bar(xs, ys)
    plt.xticks(xs, group_names, rotation=30, ha="right")
    plt.ylim(-1.1 * ymax, 1.1 * ymax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_heatmap_gb(out_path: str, title: str, mat_gb: torch.Tensor, group_names: List[str]) -> None:
    # English comment:
    # mat_gb: (G,B)
    arr = mat_gb.detach().cpu().numpy()
    plt.figure()
    plt.imshow(arr, aspect="auto")
    plt.yticks(list(range(len(group_names))), group_names)
    plt.xlabel("depth b")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# Replace your RunningStats class with this version

class RunningStats:
    def __init__(self, group_names: List[str]) -> None:
        self.group_names = list(group_names)
        self.G = int(len(self.group_names))

        self.count: Dict[str, int] = {}

        self.group_mass_sum: Dict[str, torch.Tensor] = {}
        self.group_mass_sumsq: Dict[str, torch.Tensor] = {}
        self.top1_count: Dict[str, Dict[str, int]] = {}

        self.occl_delta_sum: Dict[str, torch.Tensor] = {}
        self.occl_delta_sumsq: Dict[str, torch.Tensor] = {}
        self.occl_top1_count: Dict[str, Dict[str, int]] = {}

        self.depth_shape_sum: Dict[str, torch.Tensor] = {}
        self.depth_count: Dict[str, int] = {}
        self.max_b_seen = 0

        self.hot_depth_count: Dict[str, torch.Tensor] = {}
        self.hot_token_count: Dict[str, torch.Tensor] = {}
        self.max_t_seen = 0

    def _ensure_mode(self, mode: str) -> None:
        if mode in self.count:
            return
        self.count[mode] = 0

        self.group_mass_sum[mode] = torch.zeros((self.G,), dtype=torch.float64)
        self.group_mass_sumsq[mode] = torch.zeros((self.G,), dtype=torch.float64)
        self.top1_count[mode] = {g: 0 for g in self.group_names}

        self.occl_delta_sum[mode] = torch.zeros((self.G,), dtype=torch.float64)
        self.occl_delta_sumsq[mode] = torch.zeros((self.G,), dtype=torch.float64)
        self.occl_top1_count[mode] = {g: 0 for g in self.group_names}

        self.depth_shape_sum[mode] = torch.zeros((self.G, int(self.max_b_seen)), dtype=torch.float64)
        self.depth_count[mode] = 0
        self.hot_depth_count[mode] = torch.zeros((int(self.max_b_seen),), dtype=torch.float64)
        self.hot_token_count[mode] = torch.zeros((int(self.max_t_seen),), dtype=torch.float64)

    def _ensure_depth_size(self, b_max: int) -> None:
        target = max(int(b_max), int(self.max_b_seen))
        if target <= self.max_b_seen:
            return
        self.max_b_seen = target

        for m in list(self.depth_shape_sum.keys()):
            old = self.depth_shape_sum[m]
            G = int(old.shape[0])
            old_b = int(old.shape[1])
            if old_b < target:
                pad = torch.zeros((G, target - old_b), dtype=torch.float64)
                self.depth_shape_sum[m] = torch.cat([old, pad], dim=1)

            old_hot = self.hot_depth_count[m]
            old_hot_b = int(old_hot.shape[0])
            if old_hot_b < target:
                pad2 = torch.zeros((target - old_hot_b,), dtype=torch.float64)
                self.hot_depth_count[m] = torch.cat([old_hot, pad2], dim=0)

    def _ensure_token_size(self, t_max: int) -> None:
        target = max(int(t_max), int(self.max_t_seen))
        if target <= self.max_t_seen:
            return
        self.max_t_seen = target

        for m in list(self.hot_token_count.keys()):
            old = self.hot_token_count[m]
            old_t = int(old.shape[0])
            if old_t < target:
                pad = torch.zeros((target - old_t,), dtype=torch.float64)
                self.hot_token_count[m] = torch.cat([old, pad], dim=0)

    def add_batch(
        self,
        mode_list: List[str],
        group_mass_norm: torch.Tensor,     # (M,G) float32
        depth_shape: torch.Tensor,         # (M,G,B) float32
        hot_b: torch.Tensor,               # (M,) long
        hot_t: torch.Tensor,               # (M,) long
        occl_delta: torch.Tensor,          # (M,G) float32
        b_max: int,
        t_max: int,
    ) -> None:
        M = int(group_mass_norm.shape[0])
        if M <= 0:
            return

        unique_modes = sorted(set(mode_list))
        for m in unique_modes:
            self._ensure_mode(m)

        self._ensure_depth_size(b_max)
        self._ensure_token_size(t_max)

        gm64 = group_mass_norm.detach().to(torch.float64)
        ds64 = depth_shape.detach().to(torch.float64)
        od64 = occl_delta.detach().to(torch.float64)

        top1 = torch.argmax(gm64, dim=1)
        top1_occl = torch.argmax(od64.abs(), dim=1)

        for i in range(M):
            m = mode_list[i]
            self.count[m] += 1

            self.group_mass_sum[m] += gm64[i]
            self.group_mass_sumsq[m] += (gm64[i] * gm64[i])

            gi = int(top1[i].item())
            self.top1_count[m][self.group_names[gi]] += 1

            self.occl_delta_sum[m] += od64[i]
            self.occl_delta_sumsq[m] += (od64[i] * od64[i])

            gj = int(top1_occl[i].item())
            self.occl_top1_count[m][self.group_names[gj]] += 1

            if int(self.depth_shape_sum[m].shape[1]) < int(b_max):
                self._ensure_depth_size(b_max)

            self.depth_shape_sum[m][:, :b_max] += ds64[i, :, :b_max]
            self.depth_count[m] += 1

            bb = int(hot_b[i].item())
            tt = int(hot_t[i].item())
            if 0 <= bb < self.max_b_seen:
                self.hot_depth_count[m][bb] += 1.0
            if 0 <= tt < self.max_t_seen:
                self.hot_token_count[m][tt] += 1.0

    def to_report(self) -> Dict[str, Any]:
        rep: Dict[str, Any] = {}
        rep["count"] = dict(self.count)

        rep["top1_fraction"] = {}
        rep["group_mass_mean"] = {}
        rep["group_mass_std"] = {}

        rep["occlusion_delta_mean"] = {}
        rep["occlusion_delta_std"] = {}
        rep["occlusion_top1_fraction"] = {}

        for m in self.count.keys():
            n = float(self.count[m]) if self.count[m] > 0 else 0.0

            rep["top1_fraction"][m] = {}
            for g in self.group_names:
                rep["top1_fraction"][m][g] = (float(self.top1_count[m][g]) / n) if n > 0 else float("nan")

            gm_mean = (self.group_mass_sum[m] / n) if n > 0 else self.group_mass_sum[m] * float("nan")
            gm_var = (self.group_mass_sumsq[m] / n) - (gm_mean * gm_mean) if n > 0 else gm_mean * float("nan")
            gm_std = torch.sqrt(torch.clamp(gm_var, min=0.0))

            rep["group_mass_mean"][m] = {g: float(gm_mean[i].item()) for i, g in enumerate(self.group_names)}
            rep["group_mass_std"][m] = {g: float(gm_std[i].item()) for i, g in enumerate(self.group_names)}

            od_mean = (self.occl_delta_sum[m] / n) if n > 0 else self.occl_delta_sum[m] * float("nan")
            od_var = (self.occl_delta_sumsq[m] / n) - (od_mean * od_mean) if n > 0 else od_mean * float("nan")
            od_std = torch.sqrt(torch.clamp(od_var, min=0.0))

            rep["occlusion_delta_mean"][m] = {g: float(od_mean[i].item()) for i, g in enumerate(self.group_names)}
            rep["occlusion_delta_std"][m] = {g: float(od_std[i].item()) for i, g in enumerate(self.group_names)}

            rep["occlusion_top1_fraction"][m] = {}
            for g in self.group_names:
                rep["occlusion_top1_fraction"][m][g] = (float(self.occl_top1_count[m][g]) / n) if n > 0 else float("nan")

        rep["depth_shape_mean"] = {}
        for m in self.count.keys():
            n = float(self.depth_count[m]) if self.depth_count[m] > 0 else 0.0
            mat = self.depth_shape_sum[m] / n if n > 0 else self.depth_shape_sum[m] * float("nan")
            rep["depth_shape_mean"][m] = {g: [float(v) for v in mat[i].tolist()] for i, g in enumerate(self.group_names)}

        rep["hotspot_depth_hist"] = {}
        rep["hotspot_token_hist"] = {}
        for m in self.count.keys():
            rep["hotspot_depth_hist"][m] = [float(v) for v in self.hot_depth_count[m].tolist()]
            rep["hotspot_token_hist"][m] = [float(v) for v in self.hot_token_count[m].tolist()]

        rep["group_order"] = list(self.group_names)
        rep["max_b_seen"] = int(self.max_b_seen)
        rep["max_t_seen"] = int(self.max_t_seen)
        return rep


def _plot_bar_fraction(out_path: str, title: str, frac: Dict[str, float], group_names: List[str]) -> None:
    xs = list(range(len(group_names)))
    ys = [float(frac.get(g, 0.0)) for g in group_names]

    plt.figure()
    plt.bar(xs, ys)
    plt.xticks(xs, group_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_bar_mass(out_path: str, title: str, mass: Dict[str, float], group_names: List[str]) -> None:
    xs = list(range(len(group_names)))
    ys = [float(mass.get(g, 0.0)) for g in group_names]

    plt.figure()
    plt.bar(xs, ys)
    plt.xticks(xs, group_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_depth_curve_compare(
    out_path: str,
    title: str,
    y1: List[float],
    y0: List[float],
) -> None:
    B = int(max(len(y1), len(y0)))
    x = list(range(B))
    y1p = y1 + [0.0] * (B - len(y1))
    y0p = y0 + [0.0] * (B - len(y0))

    plt.figure()
    plt.plot(x, y1p, label="y=1")
    plt.plot(x, y0p, label="y=0")
    plt.xlabel("depth b")
    plt.ylabel("mean normalized mass")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_hist(out_path: str, title: str, hist: List[float]) -> None:
    xs = list(range(len(hist)))
    plt.figure()
    plt.bar(xs, hist)
    plt.xlabel("index")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool", type=str, default="", choices=["", "max", "lse"])

    parser.add_argument("--out_dir", type=str, default="validator_report")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])

    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--max_batch_samples", type=int, default=512)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--flow_root", type=str, default="flow")
    parser.add_argument("--dataset", type=str, default="halueval")
    parser.add_argument("--task", type=str, default="dialogue")
    parser.add_argument("--model", type=str, default="qwen25")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--shuffle_in_loader", action="store_true")

    parser.add_argument("--do_occlusion", action="store_true")
    parser.add_argument("--no_heatmap", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    _ensure_dir(args.out_dir)

    ckpt = _load_ckpt(args.ckpt_path, device=device)
    model, a = _build_model_from_ckpt(ckpt, device=device)

    pool = args.pool if args.pool else str(a.get("pool", "max"))
    if pool not in ["max", "lse"]:
        pool = "max"

    train_loader, test_loader = build_dataloader(args)
    loader = test_loader if args.split == "test" else train_loader

    feat_map = _feature_index_map(FEATURE_NAMES)
    group_indices = _group_to_indices(FEATURE_GROUPS, feat_map)

    group_names = list(GROUP_ORDER)
    G = int(len(group_names))
    stats = RunningStats(group_names=group_names)

    total_samples_used = 0
    batches_used = 0

    for bi, batch in enumerate(loader):
        if bi >= int(args.max_batches):
            break
        if total_samples_used >= int(args.max_samples):
            break

        y_true_all = batch["labels"].to(torch.long)
        keep01 = (y_true_all == 0) | (y_true_all == 1)
        if not bool(keep01.any().item()):
            continue

        evt_x = batch["evt_x"][keep01].to(device=device, dtype=torch.float32)
        evt_valid = batch["evt_valid"][keep01].to(device=device, dtype=torch.bool)
        y_true = y_true_all[keep01].to(torch.long)

        M = int(y_true.shape[0])
        if M <= 0:
            continue

        if M > int(args.max_batch_samples):
            perm = torch.randperm(M, device=y_true.device)[: int(args.max_batch_samples)]
            evt_x = evt_x[perm]
            evt_valid = evt_valid[perm]
            y_true = y_true[perm]
            M = int(y_true.shape[0])

        b_max = int(batch["B_max"])
        t_max = int(batch["T_max"])
        evt_b = batch["evt_b"].to(device=device, dtype=torch.long)
        evt_t = batch["evt_t"].to(device=device, dtype=torch.long)

        logits_evt = _infer_logits_evt(model, evt_x, evt_valid)
        logits_bag, prob, y_pred = _predict_from_logits_evt(logits_evt, evt_valid, pool=pool)

        hot_pos = _max_pool_argmax(logits_evt, evt_valid)
        hot_b = evt_b[hot_pos].detach().to("cpu")
        hot_t = evt_t[hot_pos].detach().to("cpu")

        attr = _gradxinput_attribution(model, evt_x, evt_valid, pool=pool)
        abs_attr = attr.abs()
        valid_f = evt_valid.to(torch.float32)

        group_mass_raw = torch.zeros((M, G), device=device, dtype=torch.float32)
        depth_shape = torch.zeros((M, G, int(b_max)), device=device, dtype=torch.float32)

        for gi, g in enumerate(group_names):
            idxs = group_indices[g]
            pos_score = abs_attr[:, :, idxs].sum(dim=2) * valid_f
            group_mass_raw[:, gi] = pos_score.sum(dim=1)
            depth_g = _scatter_depth_scores(pos_score, evt_b, b_max=b_max)
            depth_shape[:, gi, :] = _safe_normalize_rows(depth_g)

        group_mass_norm = _safe_normalize_rows(group_mass_raw)

        mode_list = _conf_mode(y_true.to("cpu"), y_pred.to("cpu"))

        if bool(args.do_occlusion):
            _, occl_delta = _occlusion_group_delta_logits_bag(
                model=model,
                evt_x=evt_x,
                evt_valid=evt_valid,
                pool=pool,
                group_indices=group_indices,
                group_order=group_names,
            )
            occl_delta_cpu = occl_delta.to("cpu")
        else:
            occl_delta_cpu = torch.zeros((M, G), dtype=torch.float32)

        stats.add_batch(
            mode_list=mode_list,
            group_mass_norm=group_mass_norm.to("cpu"),
            depth_shape=depth_shape.to("cpu"),
            hot_b=hot_b,
            hot_t=hot_t,
            occl_delta=occl_delta_cpu,
            b_max=b_max,
            t_max=t_max,
        )

        total_samples_used += int(M)
        batches_used = bi + 1

    report = stats.to_report()
    report["pool"] = pool
    report["ckpt_path"] = args.ckpt_path
    report["split"] = args.split
    report["task"] = args.task
    report["model"] = args.model
    report["max_batches_used"] = int(batches_used)
    report["total_samples_used"] = int(total_samples_used)

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    y1_modes = ["tp", "fn"]
    y0_modes = ["tn", "fp"]

    frac_y1 = _merge_modes_weighted(report, y1_modes, "top1_fraction", group_names)
    frac_y0 = _merge_modes_weighted(report, y0_modes, "top1_fraction", group_names)

    mass_y1 = _merge_modes_weighted(report, y1_modes, "group_mass_mean", group_names)
    mass_y0 = _merge_modes_weighted(report, y0_modes, "group_mass_mean", group_names)

    _plot_bar_fraction(
        out_path=os.path.join(args.out_dir, "top1_fraction_y1.png"),
        title="Top1 group fraction for y=1",
        frac=frac_y1,
        group_names=group_names,
    )
    _plot_bar_fraction(
        out_path=os.path.join(args.out_dir, "top1_fraction_y0.png"),
        title="Top1 group fraction for y=0",
        frac=frac_y0,
        group_names=group_names,
    )
    _plot_bar_mass(
        out_path=os.path.join(args.out_dir, "mass_mean_y1.png"),
        title="Mean normalized mass for y=1",
        mass=mass_y1,
        group_names=group_names,
    )
    _plot_bar_mass(
        out_path=os.path.join(args.out_dir, "mass_mean_y0.png"),
        title="Mean normalized mass for y=0",
        mass=mass_y0,
        group_names=group_names,
    )

    if "occlusion_delta_mean" in report:
        occl_y1 = _merge_modes_weighted(report, y1_modes, "occlusion_delta_mean", group_names)
        occl_y0 = _merge_modes_weighted(report, y0_modes, "occlusion_delta_mean", group_names)

        occl_top1_y1 = _merge_modes_weighted(report, y1_modes, "occlusion_top1_fraction", group_names)
        occl_top1_y0 = _merge_modes_weighted(report, y0_modes, "occlusion_top1_fraction", group_names)

        _plot_bar_signed(
            out_path=os.path.join(args.out_dir, "occlusion_delta_mean_y1.png"),
            title="Occlusion delta mean for y=1",
            vals=occl_y1,
            group_names=group_names,
        )
        _plot_bar_signed(
            out_path=os.path.join(args.out_dir, "occlusion_delta_mean_y0.png"),
            title="Occlusion delta mean for y=0",
            vals=occl_y0,
            group_names=group_names,
        )
        _plot_bar_fraction(
            out_path=os.path.join(args.out_dir, "occlusion_top1_fraction_y1.png"),
            title="Occlusion top1 fraction for y=1",
            frac=occl_top1_y1,
            group_names=group_names,
        )
        _plot_bar_fraction(
            out_path=os.path.join(args.out_dir, "occlusion_top1_fraction_y0.png"),
            title="Occlusion top1 fraction for y=0",
            frac=occl_top1_y0,
            group_names=group_names,
        )

    B = int(report["max_b_seen"])
    for g in group_names:
        y1_curve = [0.0] * B
        y0_curve = [0.0] * B
        total1 = 0
        total0 = 0

        for m in y1_modes:
            n = int(report["count"].get(m, 0))
            if n <= 0:
                continue
            total1 += n
            curve = report["depth_shape_mean"].get(m, {}).get(g, [])
            for i in range(min(len(curve), B)):
                y1_curve[i] += float(curve[i]) * float(n)

        for m in y0_modes:
            n = int(report["count"].get(m, 0))
            if n <= 0:
                continue
            total0 += n
            curve = report["depth_shape_mean"].get(m, {}).get(g, [])
            for i in range(min(len(curve), B)):
                y0_curve[i] += float(curve[i]) * float(n)

        if total1 > 0:
            y1_curve = [v / float(total1) for v in y1_curve]
        if total0 > 0:
            y0_curve = [v / float(total0) for v in y0_curve]

        _plot_depth_curve_compare(
            out_path=os.path.join(args.out_dir, f"depth_curve_{g}.png"),
            title=f"Depth shape compare for group {g}",
            y1=y1_curve,
            y0=y0_curve,
        )

    hot_depth_y1 = _merge_hist_weighted(report, y1_modes, "hotspot_depth_hist")
    hot_depth_y0 = _merge_hist_weighted(report, y0_modes, "hotspot_depth_hist")

    _plot_hist(
        out_path=os.path.join(args.out_dir, "hotspot_depth_hist_y1.png"),
        title="Hotspot depth histogram y=1",
        hist=hot_depth_y1,
    )
    _plot_hist(
        out_path=os.path.join(args.out_dir, "hotspot_depth_hist_y0.png"),
        title="Hotspot depth histogram y=0",
        hist=hot_depth_y0,
    )

    js_depth = _js_divergence(hot_depth_y1, hot_depth_y0)

    if not bool(args.no_heatmap):
        mat_y1 = torch.zeros((G, B), dtype=torch.float32)
        mat_y0 = torch.zeros((G, B), dtype=torch.float32)

        for gi, g in enumerate(group_names):
            y1_curve = [0.0] * B
            y0_curve = [0.0] * B
            total1 = 0
            total0 = 0

            for m in y1_modes:
                n = int(report["count"].get(m, 0))
                if n <= 0:
                    continue
                total1 += n
                curve = report["depth_shape_mean"].get(m, {}).get(g, [])
                for i in range(min(len(curve), B)):
                    y1_curve[i] += float(curve[i]) * float(n)

            for m in y0_modes:
                n = int(report["count"].get(m, 0))
                if n <= 0:
                    continue
                total0 += n
                curve = report["depth_shape_mean"].get(m, {}).get(g, [])
                for i in range(min(len(curve), B)):
                    y0_curve[i] += float(curve[i]) * float(n)

            if total1 > 0:
                y1_curve = [v / float(total1) for v in y1_curve]
            if total0 > 0:
                y0_curve = [v / float(total0) for v in y0_curve]

            mat_y1[gi] = torch.tensor(y1_curve, dtype=torch.float32)
            mat_y0[gi] = torch.tensor(y0_curve, dtype=torch.float32)

        _plot_heatmap_gb(
            out_path=os.path.join(args.out_dir, "depth_shape_heatmap_y1.png"),
            title="Depth shape heatmap y=1",
            mat_gb=mat_y1,
            group_names=group_names,
        )
        _plot_heatmap_gb(
            out_path=os.path.join(args.out_dir, "depth_shape_heatmap_y0.png"),
            title="Depth shape heatmap y=0",
            mat_gb=mat_y0,
            group_names=group_names,
        )
        _plot_heatmap_gb(
            out_path=os.path.join(args.out_dir, "depth_shape_heatmap_diff.png"),
            title="Depth shape heatmap y1 minus y0",
            mat_gb=(mat_y1 - mat_y0),
            group_names=group_names,
        )

    c_tp = int(report["count"].get("tp", 0))
    c_fp = int(report["count"].get("fp", 0))
    c_tn = int(report["count"].get("tn", 0))
    c_fn = int(report["count"].get("fn", 0))

    total = c_tp + c_fp + c_tn + c_fn
    prec = float(c_tp) / float(c_tp + c_fp + 1e-12)
    rec = float(c_tp) / float(c_tp + c_fn + 1e-12)

    print(f"[done] wrote report to {report_path}")
    print(f"[done] figures saved under {args.out_dir}")
    print("")
    print("Summary")
    print(f"total {total} tp {c_tp} fp {c_fp} tn {c_tn} fn {c_fn}")
    print(f"precision_y1 {prec:.4f} recall_y1 {rec:.4f}")
    print(f"js_hotspot_depth_y1_y0 {js_depth:.6f}")
    print("group_mass_mean_y1")
    for g in group_names:
        print(f"{g} {mass_y1.get(g, float('nan')):.6f}")
    print("group_mass_mean_y0")
    for g in group_names:
        print(f"{g} {mass_y0.get(g, float('nan')):.6f}")

    if "occlusion_delta_mean" in report:
        occl_y1 = _merge_modes_weighted(report, y1_modes, "occlusion_delta_mean", group_names)
        occl_y0 = _merge_modes_weighted(report, y0_modes, "occlusion_delta_mean", group_names)
        print("occlusion_delta_mean_y1")
        for g in group_names:
            print(f"{g} {occl_y1.get(g, float('nan')):.6f}")
        print("occlusion_delta_mean_y0")
        for g in group_names:
            print(f"{g} {occl_y0.get(g, float('nan')):.6f}")


if __name__ == "__main__":
    main()
