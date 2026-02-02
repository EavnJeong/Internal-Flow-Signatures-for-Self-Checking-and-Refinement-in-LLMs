# # validators/model.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, Optional, List, Tuple

# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import PackedSequence


# @dataclass
# class FlowLayout:
#     """
#     English comment:
#     Current evt_x feature order, F=10:
#       0 step_mag
#       1 step_mag_c
#       2 turn_angle
#       3 attn_mag
#       4 mlp_mag
#       5 comp_mag
#       6 r_eta
#       7 token_drift
#       8 R_attn
#       9 R_mlp
#     """
#     feat_dim: int = 10

#     def continuous_indices(self) -> List[int]:
#         return list(range(self.feat_dim))


# class MLPEncoder(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         hidden_dim: int,
#         dropout: float,
#         num_layers: int = 2,
#     ):
#         super().__init__()
#         layers: List[nn.Module] = []
#         d = int(in_dim)
#         for _ in range(max(1, int(num_layers) - 1)):
#             layers.append(nn.Linear(d, int(hidden_dim)))
#             layers.append(nn.GELU())
#             layers.append(nn.Dropout(float(dropout)))
#             d = int(hidden_dim)
#         layers.append(nn.Linear(d, int(out_dim)))
#         self.net = nn.Sequential(*layers)
#         self.ln = nn.LayerNorm(int(out_dim))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.net(x)
#         x = self.ln(x)
#         return x


# class FlowGRUValidator(nn.Module):
#     """
#     English comment:
#     Validator for evt_x shaped (M, L, F) with F=10 flow signatures.
#     Uses packed GRU to ignore padded steps.
#     Includes a safe fallback for very long sequences that can exceed cuDNN RNN limits (~65535).
#     """

#     def __init__(
#         self,
#         feat_dim: int = 10,
#         hidden_dim: int = 256,
#         embed_dim: int = 128,
#         enc_hidden_dim: int = 256,
#         dropout: float = 0.1,
#         gru_layers: int = 1,
#         enc_layers: int = 2,
#         layout: Optional[FlowLayout] = None,
#         max_rnn_steps: int = 60000,
#         debug: bool = False,
#     ):
#         super().__init__()
#         self.layout = layout if layout is not None else FlowLayout(feat_dim=int(feat_dim))
#         self.feat_dim = int(self.layout.feat_dim)

#         self.hidden_dim = int(hidden_dim)
#         self.embed_dim = int(embed_dim)
#         self.enc_hidden_dim = int(enc_hidden_dim)
#         self.dropout = float(dropout)
#         self.gru_layers = int(gru_layers)
#         self.enc_layers = int(enc_layers)

#         self.cont_idx = self.layout.continuous_indices()
#         self.cont_dim = len(self.cont_idx)

#         self.cont_ln = nn.LayerNorm(self.cont_dim)

#         self.encoder = MLPEncoder(
#             in_dim=self.cont_dim,
#             out_dim=self.embed_dim,
#             hidden_dim=self.enc_hidden_dim,
#             dropout=self.dropout,
#             num_layers=self.enc_layers,
#         )

#         self.gru = nn.GRU(
#             input_size=self.embed_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=self.gru_layers,
#             dropout=(self.dropout if self.gru_layers > 1 else 0.0),
#             batch_first=True,
#         )

#         self.head = nn.Linear(self.hidden_dim, 1)

#         # Learnable default initial hidden per layer (broadcast across batch)
#         self.h0 = nn.Parameter(torch.zeros(self.gru_layers, self.hidden_dim))

#         # Debug counter (buffer so it moves with .to(device))
#         self.register_buffer("_dbg_step", torch.zeros((), dtype=torch.long))

#         # Chunk length to avoid cuDNN RNN sequence length limit
#         self.max_rnn_steps = int(max_rnn_steps)
#         if self.max_rnn_steps <= 0:
#             raise ValueError("max_rnn_steps must be positive")

#         self.debug = bool(debug)

#     def _build_h0(
#         self,
#         M: int,
#         device: torch.device,
#         dtype: torch.dtype,
#         h_init: Optional[torch.Tensor],
#     ) -> torch.Tensor:
#         # Output shape: (layers, M, H)
#         if h_init is None:
#             h0 = (
#                 self.h0.to(device=device, dtype=dtype)
#                 .unsqueeze(1)
#                 .expand(self.gru_layers, M, self.hidden_dim)
#                 .contiguous()
#             )
#             return h0

#         h_init = h_init.to(device=device, dtype=dtype)

#         if h_init.dim() == 2:
#             # (M, H) provided for the last layer only
#             if h_init.shape != (M, self.hidden_dim):
#                 raise ValueError(
#                     f"h_init shape mismatch. expected {(M, self.hidden_dim)}, got {tuple(h_init.shape)}"
#                 )
#             h0 = (
#                 self.h0.to(device=device, dtype=dtype)
#                 .unsqueeze(1)
#                 .expand(self.gru_layers, M, self.hidden_dim)
#                 .contiguous()
#             )
#             h0[-1] = h_init
#             return h0

#         if h_init.dim() == 3:
#             if h_init.shape != (self.gru_layers, M, self.hidden_dim):
#                 raise ValueError(
#                     f"h_init shape mismatch. expected {(self.gru_layers, M, self.hidden_dim)}, got {tuple(h_init.shape)}"
#                 )
#             return h_init.contiguous()

#         raise ValueError(f"h_init must be 2D or 3D, got {tuple(h_init.shape)}")

#     def _pack_contiguous(
#         self,
#         emb: torch.Tensor,      # (M, T, E)
#         lengths: torch.Tensor,  # (M,)
#     ) -> PackedSequence:
#         # Ensure contiguous input to avoid cuDNN issues
#         emb = emb.contiguous()
#         lengths_cpu = lengths.detach().to("cpu")

#         packed = nn.utils.rnn.pack_padded_sequence(
#             emb, lengths_cpu, batch_first=True, enforce_sorted=False
#         )

#         # Ensure packed.data contiguous
#         if not packed.data.is_contiguous():
#             packed = PackedSequence(
#                 packed.data.contiguous(),
#                 packed.batch_sizes,
#                 packed.sorted_indices,
#                 packed.unsorted_indices,
#             )
#         return packed

#     def _gru_packed_single(
#         self,
#         emb: torch.Tensor,          # (M, L, E)
#         lengths: torch.Tensor,      # (M,)
#         h0: torch.Tensor,           # (layers, M, H)
#         total_length: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         packed = self._pack_contiguous(emb, lengths)

#         if self.debug:
#             self._dbg_step += 1
#             step = int(self._dbg_step.item())
#             print(
#                 f"[DBG model pre-GRU] step={step} "
#                 f"packed.data.shape={tuple(packed.data.shape)} "
#                 f"packed.contig={packed.data.is_contiguous()} "
#                 f"batch_sizes.shape={tuple(packed.batch_sizes.shape)} "
#                 f"batch_sizes[:5]={packed.batch_sizes[:5].tolist()}"
#             )

#         h_packed, h_last = self.gru(packed, h0)
#         h_seq, _ = nn.utils.rnn.pad_packed_sequence(
#             h_packed, batch_first=True, total_length=total_length
#         )  # (M, L, H)
#         return h_seq, h_last

#     def _gru_packed_chunked(
#         self,
#         emb: torch.Tensor,          # (M, L, E)
#         lengths: torch.Tensor,      # (M,)
#         h0: torch.Tensor,           # (layers, M, H)
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         English comment:
#         Chunk the time dimension to bypass cuDNN RNN sequence length limits.
#         We process each chunk with packed sequences over the active subset only,
#         and carry hidden states across chunks.
#         """
#         M, L, _ = emb.shape
#         H = self.hidden_dim
#         device = emb.device
#         dtype = emb.dtype

#         chunk = int(self.max_rnn_steps)
#         h_all = h0  # (layers, M, H)
#         outs: List[torch.Tensor] = []

#         for start in range(0, L, chunk):
#             end = min(L, start + chunk)
#             T = end - start

#             emb_chunk = emb[:, start:end, :].contiguous()

#             # Valid steps remaining in this chunk for each sequence
#             len_chunk = (lengths - start).clamp(min=0, max=T).to(torch.long)
#             active = len_chunk > 0

#             if not bool(active.any()):
#                 outs.append(torch.zeros((M, T, H), device=device, dtype=dtype))
#                 continue

#             emb_act = emb_chunk[active]
#             len_act = len_chunk[active]
#             h0_act = h_all[:, active, :].contiguous()

#             packed = self._pack_contiguous(emb_act, len_act)

#             if self.debug:
#                 self._dbg_step += 1
#                 step = int(self._dbg_step.item())
#                 print(
#                     f"[DBG model pre-GRU/chunk] step={step} chunk=[{start},{end}) "
#                     f"packed.data.shape={tuple(packed.data.shape)} "
#                     f"packed.contig={packed.data.is_contiguous()} "
#                     f"batch_sizes.shape={tuple(packed.batch_sizes.shape)} "
#                     f"batch_sizes[:5]={packed.batch_sizes[:5].tolist()}"
#                 )

#             packed_out, h_last_act = self.gru(packed, h0_act)

#             seq_act, _ = nn.utils.rnn.pad_packed_sequence(
#                 packed_out, batch_first=True, total_length=T
#             )  # (activeM, T, H)

#             seq_full = torch.zeros((M, T, H), device=device, dtype=dtype)
#             seq_full[active] = seq_act
#             outs.append(seq_full)

#             # Update hidden states for active sequences
#             h_next = h_all.clone()
#             h_next[:, active, :] = h_last_act
#             h_all = h_next

#         h_seq = torch.cat(outs, dim=1)  # (M, L, H)
#         h_last = h_all                  # (layers, M, H)
#         return h_seq, h_last

#     def forward(
#         self,
#         evt_x: torch.Tensor,                       # (M, L, F)
#         evt_valid: Optional[torch.Tensor] = None,  # (M, L) bool
#         h_init: Optional[torch.Tensor] = None,     # (M, H) or (layers, M, H)
#     ) -> Dict[str, torch.Tensor]:
#         if evt_x.dim() != 3:
#             raise ValueError(f"evt_x must be (M,L,F), got {tuple(evt_x.shape)}")

#         M, L, Fdim = evt_x.shape
#         if Fdim != self.feat_dim:
#             raise ValueError(f"feat_dim mismatch. expected {self.feat_dim}, got {Fdim}")

#         device = evt_x.device
#         dtype = evt_x.dtype

#         if evt_valid is None:
#             evt_valid = torch.ones((M, L), dtype=torch.bool, device=device)
#         else:
#             evt_valid = evt_valid.to(device=device, dtype=torch.bool)

#         # Lengths for packing
#         lengths = evt_valid.sum(dim=1).to(torch.long)
#         lengths = torch.clamp(lengths, min=1)

#         # Mask inputs
#         x = evt_x.to(dtype=dtype)
#         x = x * evt_valid.to(dtype=dtype).unsqueeze(-1)

#         cont = x[:, :, self.cont_idx]
#         cont = self.cont_ln(cont)

#         flat = cont.reshape(M * L, self.cont_dim)
#         emb = self.encoder(flat).view(M, L, self.embed_dim)

#         # Zero out padded embeddings
#         emb = emb * evt_valid.to(dtype=dtype).unsqueeze(-1)

#         # Build initial hidden state
#         h0 = self._build_h0(M=M, device=device, dtype=dtype, h_init=h_init)

#         # cuDNN RNN can error out when max length exceeds its supported limit.
#         # Use chunked GRU in that case.
#         max_len = int(lengths.max().item())
#         if max_len > 65535:
#             h_seq, h_last = self._gru_packed_chunked(emb, lengths, h0)
#         else:
#             h_seq, h_last = self._gru_packed_single(emb, lengths, h0, total_length=L)

#         logits = self.head(h_seq).squeeze(-1)  # (M, L)

#         return {
#             "logits": logits,
#             "h_last": h_last[-1],  # (M, H)
#         }


# validators/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn


@dataclass
class FlowLayout:
    """
    English comment:
    Current evt_x feature order, F=10:
      0 step_mag
      1 step_mag_c
      2 turn_angle
      3 attn_mag
      4 mlp_mag
      5 comp_mag
      6 r_eta
      7 token_drift
      8 R_attn
      9 R_mlp
    """
    feat_dim: int = 10

    def continuous_indices(self) -> List[int]:
        return list(range(self.feat_dim))


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for _ in range(max(1, int(num_layers) - 1)):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
            d = int(hidden_dim)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.ln(x)
        return x


class FlowGRUValidator(nn.Module):
    """
    English comment:
    Validator for evt_x shaped (M,L,F) with F=10 flow signatures.
    No phase logic, no reset embedding.
    """

    def __init__(
        self,
        feat_dim: int = 10,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        enc_hidden_dim: int = 256,
        dropout: float = 0.1,
        gru_layers: int = 1,
        enc_layers: int = 2,
        layout: Optional[FlowLayout] = None,
    ):
        super().__init__()
        self.layout = layout if layout is not None else FlowLayout(feat_dim=int(feat_dim))
        self.feat_dim = int(self.layout.feat_dim)

        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.enc_hidden_dim = int(enc_hidden_dim)
        self.dropout = float(dropout)
        self.gru_layers = int(gru_layers)
        self.enc_layers = int(enc_layers)

        self.cont_idx = self.layout.continuous_indices()
        self.cont_dim = len(self.cont_idx)

        self.cont_ln = nn.LayerNorm(self.cont_dim)

        self.encoder = MLPEncoder(
            in_dim=self.cont_dim,
            out_dim=self.embed_dim,
            hidden_dim=self.enc_hidden_dim,
            dropout=self.dropout,
            num_layers=self.enc_layers,
        )

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            dropout=(self.dropout if self.gru_layers > 1 else 0.0),
            batch_first=True,
        )

        self.head = nn.Linear(self.hidden_dim, 1)

        self.h0 = nn.Parameter(torch.zeros(self.gru_layers, self.hidden_dim))

    def forward(
        self,
        evt_x: torch.Tensor,                       # (M,L,F)
        evt_valid: Optional[torch.Tensor] = None,  # (M,L) bool
        h_init: Optional[torch.Tensor] = None,     # (M,H) or (layers,M,H)
    ) -> Dict[str, torch.Tensor]:
        if evt_x.dim() != 3:
            raise ValueError(f"evt_x must be (M,L,F), got {tuple(evt_x.shape)}")

        M, L, Fdim = evt_x.shape
        if Fdim != self.feat_dim:
            raise ValueError(f"feat_dim mismatch. expected {self.feat_dim}, got {Fdim}")

        device = evt_x.device
        dtype = evt_x.dtype

        if evt_valid is None:
            evt_valid = torch.ones((M, L), dtype=torch.bool, device=device)
        else:
            evt_valid = evt_valid.to(device=device, dtype=torch.bool)

        # lengths for packing
        lengths = evt_valid.sum(dim=1).to(torch.long)
        lengths = torch.clamp(lengths, min=1)  # avoid empty sequences

        x = evt_x.to(dtype=dtype)
        x = x * evt_valid.to(dtype=dtype).unsqueeze(-1)

        cont = x[:, :, self.cont_idx]
        cont = self.cont_ln(cont)

        flat = cont.reshape(M * L, self.cont_dim)
        emb = self.encoder(flat).view(M, L, self.embed_dim)

        # force padded embeddings to zero after encoder
        emb = emb * evt_valid.to(dtype=dtype).unsqueeze(-1)

        # build h0
        if h_init is None:
            h0 = self.h0.to(device=device, dtype=dtype).unsqueeze(1).expand(self.gru_layers, M, self.hidden_dim).contiguous()
        else:
            h_init = h_init.to(device=device, dtype=dtype)
            if h_init.dim() == 2:
                h0 = self.h0.to(device=device, dtype=dtype).unsqueeze(1).expand(self.gru_layers, M, self.hidden_dim).contiguous()
                h0[-1] = h_init
            elif h_init.dim() == 3:
                if h_init.shape != (self.gru_layers, M, self.hidden_dim):
                    raise ValueError(f"h_init shape mismatch. expected {(self.gru_layers, M, self.hidden_dim)}, got {tuple(h_init.shape)}")
                h0 = h_init
            else:
                raise ValueError(f"h_init must be 2D or 3D, got {tuple(h_init.shape)}")

        # pack so GRU does not update on padded time steps
        lengths_cpu = lengths.detach().to("cpu")
        emb_packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        h_packed, h_last = self.gru(emb_packed, h0)

        h_seq, _ = nn.utils.rnn.pad_packed_sequence(
            h_packed, batch_first=True, total_length=L
        )  # (M,L,H)

        logits = self.head(h_seq).squeeze(-1)  # (M,L)

        return {
            "logits": logits,
            "h_last": h_last[-1],              # (M,H)
        }
