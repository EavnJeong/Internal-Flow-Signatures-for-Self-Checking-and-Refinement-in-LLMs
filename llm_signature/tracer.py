from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class Trace:
    h_in: torch.Tensor     # (B, N, T, D) input to each block
    h_out: torch.Tensor    # (B, N, T, D) output of each block
    o: torch.Tensor        # (B, N, T, D) attention output per block (pre residual add)
    m: torch.Tensor        # (B, N, T, D) mlp output per block (pre residual add)
    attention_mask: Optional[torch.Tensor] = None


def _first_tensor(x: Any) -> torch.Tensor:
    # English comment: HF modules may return tuples, this picks the first Tensor payload.
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return x[0]
    raise TypeError(f"Unsupported module output type: {type(x)}")


class ModelAdapter:
    """
    English comment: Adapter isolates model-family specific module paths.
    """

    def iter_blocks(self, model: nn.Module) -> List[nn.Module]:
        raise NotImplementedError

    def get_attn(self, block: nn.Module) -> nn.Module:
        raise NotImplementedError

    def get_mlp(self, block: nn.Module) -> nn.Module:
        raise NotImplementedError


class LlamaLikeAdapter(ModelAdapter):
    """
    English comment: Many decoder-only HF models expose blocks at model.model.layers.
    Each block typically has self_attn and mlp.
    """

    def iter_blocks(self, model: nn.Module) -> List[nn.Module]:
        inner = getattr(model, "model", model)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise AttributeError("Could not find model.model.layers or model.layers.")
        return list(layers)

    def get_layers(self, model: nn.Module) -> List[nn.Module]:
        # English comment: Alias used by helper utilities.
        return self.iter_blocks(model)

    def get_attn(self, block: nn.Module) -> nn.Module:
        attn = getattr(block, "self_attn", None)
        if attn is None:
            raise AttributeError("Block has no self_attn.")
        return attn

    def get_mlp(self, block: nn.Module) -> nn.Module:
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            raise AttributeError("Block has no mlp.")
        return mlp


class TraceCollector:
    def __init__(
        self,
        model: nn.Module,
        adapter: ModelAdapter,
        store_device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.store_device = torch.device(store_device)
        self.dtype = dtype

    def _save(self, t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if self.dtype is not None:
            t = t.to(self.dtype)
        return t.to(self.store_device)

    @torch.no_grad()
    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **forward_kwargs: Any,
    ) -> Trace:
        blocks = self.adapter.iter_blocks(self.model)
        B = len(blocks)

        h_in_list: List[Optional[torch.Tensor]] = [None] * B
        h_out_list: List[Optional[torch.Tensor]] = [None] * B
        o_list: List[Optional[torch.Tensor]] = [None] * B
        m_list: List[Optional[torch.Tensor]] = [None] * B

        handles: List[Any] = []

        for b, block in enumerate(blocks):
            attn = self.adapter.get_attn(block)
            mlp = self.adapter.get_mlp(block)

            def make_block_pre_hook(idx: int) -> Callable:
                def _pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
                    x = inputs[0]
                    if not isinstance(x, torch.Tensor):
                        raise TypeError("Block input is not a Tensor.")
                    h_in_list[idx] = self._save(x)
                return _pre_hook

            def make_block_hook(idx: int) -> Callable:
                def _hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
                    y = _first_tensor(output)
                    h_out_list[idx] = self._save(y)
                return _hook

            def make_attn_hook(idx: int) -> Callable:
                def _hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
                    y = _first_tensor(output)
                    o_list[idx] = self._save(y)
                return _hook

            def make_mlp_hook(idx: int) -> Callable:
                def _hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
                    y = _first_tensor(output)
                    m_list[idx] = self._save(y)
                return _hook

            handles.append(block.register_forward_pre_hook(make_block_pre_hook(b)))
            handles.append(block.register_forward_hook(make_block_hook(b)))
            handles.append(attn.register_forward_hook(make_attn_hook(b)))
            handles.append(mlp.register_forward_hook(make_mlp_hook(b)))

        try:
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                **forward_kwargs,
            )
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        for name, lst in [("h_in", h_in_list), ("h_out", h_out_list), ("o", o_list), ("m", m_list)]:
            if any(x is None for x in lst):
                missing = [i for i, x in enumerate(lst) if x is None]
                raise RuntimeError(f"Missing captures for {name} at blocks: {missing}")

        h_in = torch.stack([x for x in h_in_list if x is not None], dim=0)
        h_out = torch.stack([x for x in h_out_list if x is not None], dim=0)
        o = torch.stack([x for x in o_list if x is not None], dim=0)
        m = torch.stack([x for x in m_list if x is not None], dim=0)

        if h_in.shape != h_out.shape or o.shape != h_out.shape or m.shape != h_out.shape:
            raise RuntimeError(
                f"Shape mismatch: h_in {tuple(h_in.shape)}, h_out {tuple(h_out.shape)}, "
                f"o {tuple(o.shape)}, m {tuple(m.shape)}"
            )

        am = attention_mask.detach().to(self.store_device) if attention_mask is not None else None
        return Trace(h_in=h_in, h_out=h_out, o=o, m=m, attention_mask=am)
