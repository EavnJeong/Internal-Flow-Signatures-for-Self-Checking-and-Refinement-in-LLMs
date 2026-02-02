# refine/intervene_generate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import StoppingCriteriaList

from refine.stopping import StopOnNewline


def _get_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    raise RuntimeError("Could not find transformer layers on this model.")


def _get_lm_head_weight(model) -> torch.Tensor:
    emb = None
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
    if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor):
        return emb.weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and isinstance(model.lm_head.weight, torch.Tensor):
        return model.lm_head.weight
    raise RuntimeError("Could not find lm head weight on this model.")


def _unpack_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _repack_output(h, rest):
    if rest is None:
        return h
    return (h, *rest)


def _procrustes(U_out: torch.Tensor, U_in: torch.Tensor) -> torch.Tensor:
    # English comment: U_out, U_in are (D, k) with orthonormal columns.
    # English comment: Returns R (k, k) mapping in-coordinates into out-coordinates.
    M = U_out.transpose(0, 1) @ U_in  # (k, k)
    P, _, Qh = torch.linalg.svd(M, full_matrices=False)
    return P @ Qh


def _fallback_basis(D: int, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k2 = min(int(k), int(D))
    U = torch.zeros((D, k2), device=device, dtype=dtype)
    U[:k2, :k2] = torch.eye(k2, device=device, dtype=dtype)
    return U


def _basis_from_competitors(
    W: torch.Tensor,
    y_hat: int,
    comp_ids: torch.Tensor,
    k_dim: int,
) -> torch.Tensor:
    # English comment: W is (V, D). Each row is a token readout direction.
    w_top = W[y_hat]  # (D,)
    Wc = W.index_select(0, comp_ids)  # (K, D)
    D_mat = (w_top.unsqueeze(0) - Wc).to(dtype=torch.float32)  # (K, D)

    # English comment: Row-normalize for stability.
    D_mat = D_mat / (D_mat.norm(dim=1, keepdim=True) + 1e-12)

    # English comment: SVD on (K, D), take top-k right singular vectors.
    try:
        _, _, Vh = torch.linalg.svd(D_mat, full_matrices=False)
        V = Vh.transpose(0, 1)  # (D, min(K, D))
        k2 = min(int(k_dim), int(V.shape[1]))
        return V[:, :k2].to(dtype=W.dtype)
    except Exception:
        return _fallback_basis(D=int(W.shape[1]), k=int(k_dim), device=W.device, dtype=W.dtype)


def _topk_competitors_from_hidden(
    W: torch.Tensor,
    h: torch.Tensor,
    comp_k: int,
) -> Tuple[int, torch.Tensor]:
    # English comment: h is (B, D), typically B=1.
    logits = h.to(dtype=W.dtype) @ W.transpose(0, 1)  # (B, V)
    _, idx = torch.topk(logits, k=int(comp_k) + 1, dim=-1)
    y_hat = int(idx[0, 0].item())
    comp_ids = idx[0, 1:].to(dtype=torch.long)
    return y_hat, comp_ids


@dataclass
class InterventionConfig:
    layer_idx: int

    # English comment: Calibration window on the prefix.
    ref_span: int = 64

    # English comment: Upper clamp band around ref_step. Only caps large steps.
    max_ratio: float = 1.15

    # English comment: Lower clamp disabled by default to avoid amplifying small steps.
    # English comment: Set a positive value only if you intentionally want to increase step size.
    min_ratio: Optional[float] = None

    enabled: bool = True

    # English comment: Competitor-basis settings.
    k_dim: int = 16
    comp_k: int = 32

    # English comment: Apply to many generation steps. None means no limit.
    max_steps: Optional[int] = None

    # English comment: Freeze competitor set and basis after calibration to keep ref_step comparable.
    freeze_basis_after_calibration: bool = True

    # English comment: If freeze_basis_after_calibration=False, you may still refresh competitors.
    refresh_competitors_each_step: bool = False

    # English comment: Run one calibration forward on the prefix so S>1 appears at least once.
    calibrate_on_prefix: bool = True

    # English comment: Do not modify outputs during calibration pass.
    apply_on_prefix: bool = False


class HiddenStepClampHook:
    """
    Clamp the k-space depth step at a chosen transformer block.

    English comment:
    - Calibration stage: runs on a full prefix (S>1) to estimate ref_step and optionally freeze basis.
    - Step stage: runs on cached decoding (often S=1) and caps dp norm when it exceeds the upper band.
    """
    def __init__(self, cfg: InterventionConfig, model):
        self.cfg = cfg
        self.W = _get_lm_head_weight(model)  # (V, D)

        self.ref_step: Optional[torch.Tensor] = None  # (B,)

        self.last_info: Dict[str, Any] = {}
        self.step_idx: int = 0

        # English comment: Cached competitor set.
        self._cached_yhat: Optional[int] = None
        self._cached_comp_ids: Optional[torch.Tensor] = None

        # English comment: Cached basis and transport.
        self._cached_U: Optional[torch.Tensor] = None  # (D, k)
        self._cached_R: Optional[torch.Tensor] = None  # (k, k)

        # English comment: When True, competitor set and basis stay fixed after calibration.
        self._basis_frozen: bool = False

    def _maybe_refresh_competitors(self, h_for_logits: torch.Tensor, force: bool) -> Tuple[int, torch.Tensor]:
        if self._basis_frozen:
            if self._cached_yhat is None or self._cached_comp_ids is None:
                # English comment: Should not happen, but keep it safe.
                yhat, comp_ids = _topk_competitors_from_hidden(self.W, h_for_logits, comp_k=int(self.cfg.comp_k))
                self._cached_yhat = int(yhat)
                self._cached_comp_ids = comp_ids.detach().to(device="cpu")
                self._cached_U = None
                self._cached_R = None
            yhat2 = int(self._cached_yhat)
            comp2 = self._cached_comp_ids.to(device=self.W.device)
            return yhat2, comp2

        if force or self._cached_yhat is None or self._cached_comp_ids is None:
            yhat, comp_ids = _topk_competitors_from_hidden(self.W, h_for_logits, comp_k=int(self.cfg.comp_k))
            self._cached_yhat = int(yhat)
            self._cached_comp_ids = comp_ids.detach().to(device="cpu")
            # English comment: Invalidate basis cache when competitors change.
            self._cached_U = None
            self._cached_R = None

        yhat2 = int(self._cached_yhat)
        comp2 = self._cached_comp_ids.to(device=self.W.device)
        return yhat2, comp2

    def _get_basis_and_transport(
        self,
        yhat: int,
        comp_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_U is None or self._cached_R is None:
            U = _basis_from_competitors(self.W, y_hat=yhat, comp_ids=comp_ids, k_dim=int(self.cfg.k_dim))  # (D, k)
            # English comment: Keep R close to identity by using the same basis on both sides.
            R = _procrustes(U.to(torch.float32), U.to(torch.float32)).to(dtype=U.dtype)  # (k, k)
            self._cached_U = U.detach().to(device="cpu")
            self._cached_R = R.detach().to(device="cpu")

        U2 = self._cached_U.to(device=device, dtype=dtype)
        R2 = self._cached_R.to(device=device, dtype=dtype)
        return U2, R2

    def __call__(self, module, inputs, output):
        if not self.cfg.enabled:
            return output

        h_out, rest = _unpack_output(output)
        if not isinstance(h_out, torch.Tensor) or h_out.ndim != 3:
            return output

        if not isinstance(inputs, (tuple, list)) or len(inputs) == 0:
            return output
        h_in = inputs[0]
        if not isinstance(h_in, torch.Tensor) or h_in.ndim != 3:
            return output

        if h_in.shape != h_out.shape:
            return output

        B, S, D = h_out.shape
        if B <= 0 or S <= 0:
            return output

        # English comment: In cached decoding, many models expose S=1 at each layer.
        pos = -1 if S > 1 else 0
        h_in_last = h_in[:, pos, :]    # (B, D)
        h_out_last = h_out[:, pos, :]  # (B, D)

        with torch.no_grad():
            stage = "prefix" if S > 1 else "step"

            # English comment: Stop applying after cfg.max_steps step calls.
            if stage == "step":
                idx = int(self.step_idx)
                self.step_idx += 1
                if self.cfg.max_steps is not None and idx >= int(self.cfg.max_steps):
                    self.last_info = {"S": int(S), "stage": stage, "mode": "skip_max_steps", "step_idx": int(idx)}
                    return output
            else:
                idx = -1

            # English comment: Competitor refresh policy.
            if stage == "prefix":
                force_refresh = True
            else:
                force_refresh = bool(self.cfg.refresh_competitors_each_step)

            yhat, comp_ids = self._maybe_refresh_competitors(h_out_last, force=force_refresh)
            U, R = self._get_basis_and_transport(yhat, comp_ids, device=h_out.device, dtype=h_out.dtype)  # (D, k), (k, k)

            # English comment: Coordinates and transported delta at the last position.
            p_in = h_in_last @ U  # (B, k)
            p_out = h_out_last @ U  # (B, k)
            Rp_in = p_in @ R.transpose(0, 1)  # (B, k)
            dp = p_out - Rp_in  # (B, k)
            dp_norm = dp.norm(dim=-1)  # (B,)

            if stage == "prefix":
                span = min(int(self.cfg.ref_span), S - 1)
                if span <= 0:
                    self.last_info = {"S": int(S), "stage": stage, "mode": "no_span"}
                    return output

                lo = S - 1 - span
                hi = S - 1

                h_in_ref = h_in[:, lo:hi, :]    # (B, span, D)
                h_out_ref = h_out[:, lo:hi, :]  # (B, span, D)

                p_in_ref = torch.einsum("bsd,dk->bsk", h_in_ref, U)
                p_out_ref = torch.einsum("bsd,dk->bsk", h_out_ref, U)
                Rp_in_ref = torch.einsum("bsk,kl->bsl", p_in_ref, R.transpose(0, 1))
                dp_ref = p_out_ref - Rp_in_ref
                ref_step = dp_ref.norm(dim=-1).median(dim=1).values  # (B,)
                self.ref_step = ref_step.detach()

                if bool(self.cfg.freeze_basis_after_calibration):
                    self._basis_frozen = True

                self.last_info = {
                    "S": int(S),
                    "stage": stage,
                    "mode": "calibrated",
                    "ref_step_mean": float(ref_step.mean().item()),
                    "k_dim": int(U.shape[1]),
                    "comp_k": int(comp_ids.numel()),
                    "basis_frozen": bool(self._basis_frozen),
                }

                # English comment: Default behavior keeps prefix unchanged.
                if not bool(self.cfg.apply_on_prefix):
                    return output

            # English comment: Step stage needs ref_step.
            if self.ref_step is None:
                self.last_info = {"S": int(S), "stage": stage, "mode": "step_no_ref", "step_idx": int(idx)}
                return output

            ref_step2 = self.ref_step.to(device=dp.device, dtype=dp.dtype)

            # English comment: Only cap large steps. No upscaling unless min_ratio is explicitly set.
            max_step = ref_step2 * float(self.cfg.max_ratio)
            over = dp_norm > max_step

            if self.cfg.min_ratio is not None and float(self.cfg.min_ratio) > 0.0:
                min_step = ref_step2 * float(self.cfg.min_ratio)
                under = dp_norm < min_step
                do_change = over | under
                target = torch.where(over, max_step, dp_norm)
                target = torch.where(under, min_step, target)
            else:
                do_change = over
                target = torch.where(over, max_step, dp_norm)

            if not bool(do_change.any().item()):
                self.last_info = {
                    "S": int(S),
                    "stage": stage,
                    "mode": "no_change",
                    "step_idx": int(idx),
                    "k_dim": int(U.shape[1]),
                    "comp_k": int(comp_ids.numel()),
                    "dp": float(dp_norm.mean().item()),
                    "ref": float(ref_step2.mean().item()),
                }
                return output

            scale = (target / (dp_norm + 1e-12)).view(B, 1)
            dp2 = dp * scale
            p_out2 = dp2 + Rp_in

            # English comment: Replace only the U component at the last position.
            proj_cur = p_out @ U.transpose(0, 1)   # (B, D)
            proj_new = p_out2 @ U.transpose(0, 1)  # (B, D)
            resid = h_out_last - proj_cur
            h_out_last2 = resid + proj_new

            h2 = h_out.clone()
            h2[:, pos, :] = h_out_last2

            self.last_info = {
                "S": int(S),
                "stage": stage,
                "mode": "cap_step",
                "step_idx": int(idx),
                "changed": True,
                "k_dim": int(U.shape[1]),
                "comp_k": int(comp_ids.numel()),
                "dp_before": float(dp_norm.mean().item()),
                "dp_after": float(dp2.norm(dim=-1).mean().item()),
                "ref": float(ref_step2.mean().item()),
                "basis_frozen": bool(self._basis_frozen),
            }

            return _repack_output(h2, rest)


@torch.no_grad()
def rollback_and_regenerate_with_hidden_intervention(
    model,
    tokenizer,
    input_ids_1d: torch.Tensor,
    rollback_t: int,
    gen_max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    cfg: Optional[InterventionConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device

    if not isinstance(input_ids_1d, torch.Tensor) or input_ids_1d.ndim != 1:
        raise RuntimeError("input_ids_1d must be a 1D torch.Tensor")

    L = int(input_ids_1d.numel())
    rollback_t = int(rollback_t)
    rollback_t = max(1, min(rollback_t, L))

    prefix = input_ids_1d[:rollback_t].to(device=device, dtype=torch.long).unsqueeze(0)
    start_len = int(prefix.shape[1])

    nl = tokenizer.encode("\n", add_special_tokens=False)
    stopping = StoppingCriteriaList([StopOnNewline(nl, start_len)]) if len(nl) > 0 else None

    hook_handle = None
    hook_obj: Optional[HiddenStepClampHook] = None

    if cfg is not None and cfg.enabled:
        layers = _get_layers(model)
        if cfg.layer_idx < 0 or cfg.layer_idx >= len(layers):
            raise RuntimeError(f"layer_idx out of range: {cfg.layer_idx} not in [0, {len(layers) - 1}]")

        hook_obj = HiddenStepClampHook(cfg, model=model)
        hook_handle = layers[cfg.layer_idx].register_forward_hook(hook_obj)

        # English comment: Calibration pass to populate ref_step with an S>1 forward.
        if bool(cfg.calibrate_on_prefix):
            try:
                _ = model(prefix, use_cache=False)
            except TypeError:
                _ = model(prefix)

    try:
        out_ids = model.generate(
            input_ids=prefix,
            do_sample=True if temperature > 0 else False,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(gen_max_new_tokens),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            stopping_criteria=stopping,
        )

        out_ids_1d = out_ids[0].detach().to("cpu")
        text = tokenizer.decode(out_ids_1d, skip_special_tokens=True)

        return {
            "rollback_t": int(rollback_t),
            "out_ids": out_ids_1d,
            "text": text,
            "hook_last_info": (hook_obj.last_info if hook_obj is not None else {}),
        }
    finally:
        if hook_handle is not None:
            hook_handle.remove()
