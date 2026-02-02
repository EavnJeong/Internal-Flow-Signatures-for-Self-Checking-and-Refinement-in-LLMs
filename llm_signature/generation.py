# generation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GenerationBatch:
    # Prefix tensors from tokenization (padded)
    prefix_input_ids: torch.Tensor        # (N, Tp)
    prefix_attention_mask: torch.Tensor   # (N, Tp) long

    # Generated tokens per sample (trimmed, unpadded)
    gen_output_ids: List[torch.Tensor]    # list of (Li,) long on CPU
    generated_text: List[str]             # list of decoded generated text (new tokens only)

    # Teacher forcing tensors (prefix tokens without pads + generated tokens), padded
    full_input_ids: torch.Tensor          # (N, Tf)
    full_attention_mask: torch.Tensor     # (N, Tf) long
    answer_mask: torch.Tensor             # (N, Tf) bool

    # Metadata useful for debugging
    prefix_lens: torch.Tensor             # (N,) long, number of non-pad prefix tokens
    full_lens: torch.Tensor               # (N,) long, total non-pad tokens in full sequence

    gen_cfg: Dict[str, Any]


def _ensure_attention_mask(tok: Dict[str, torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
    am = tok.get("attention_mask", None)
    if am is None:
        am = torch.ones_like(input_ids, dtype=torch.long)
    return am.to(torch.long)


def tokenize_prefixes(
    tokenizer,
    prompt_prefixes: List[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English comment:
      Tokenizes prefix texts into padded tensors.
      Works with left or right padding.
    """
    tok = tokenizer(
        prompt_prefixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = _ensure_attention_mask(tok, tok["input_ids"]).to(device)
    return input_ids, attention_mask


def _trim_generated_part(
    full_sequences: torch.Tensor,
    prefix_padded_len: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> List[torch.Tensor]:
    """
    English comment:
      Extracts generated portion from sequences[:, prefix_padded_len:].
      Trims at first eos_token_id if present.
      Removes trailing pad_token_id.
      Returns list of 1D CPU long tensors.
    """
    N = int(full_sequences.shape[0])
    gen_part = full_sequences[:, prefix_padded_len:]  # (N, Lnew_padded)
    out: List[torch.Tensor] = []

    for i in range(N):
        ids = gen_part[i].detach().to("cpu").to(torch.long)

        # Cut at eos if provided
        if eos_token_id is not None:
            eos_pos = (ids == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                cut = int(eos_pos[0].item())
                ids = ids[:cut]

        # Drop trailing pad ids
        if pad_token_id is not None and ids.numel() > 0:
            while ids.numel() > 0 and int(ids[-1].item()) == int(pad_token_id):
                ids = ids[:-1]

        out.append(ids.contiguous())

    return out


def _build_full_teacher_forcing(
    prefix_input_ids: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    gen_output_ids: List[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comment:
      Creates full_input_ids by concatenating:
        prefix_tokens_without_pad + gen_output_ids[i]
      Then pads across batch.
      Returns:
        full_input_ids (N,Tf), full_attention_mask (N,Tf), answer_mask (N,Tf),
        prefix_lens (N,), full_lens (N,)
    """
    N = int(prefix_input_ids.shape[0])
    if len(gen_output_ids) != N:
        raise ValueError("gen_output_ids length mismatch")

    prefix_lens = prefix_attention_mask.to(torch.long).sum(dim=1).detach().to("cpu")  # (N,)
    full_list: List[torch.Tensor] = []
    full_lens_list: List[int] = []

    for i in range(N):
        am = prefix_attention_mask[i].to(torch.bool)
        pref_tokens = prefix_input_ids[i][am].detach().to("cpu").to(torch.long)  # (Lp,)
        gen_tokens = gen_output_ids[i].detach().to("cpu").to(torch.long)         # (Lg,)
        full_i = torch.cat([pref_tokens, gen_tokens], dim=0) if gen_tokens.numel() > 0 else pref_tokens
        full_list.append(full_i)
        full_lens_list.append(int(full_i.numel()))

    Tf = max(full_lens_list) if len(full_lens_list) > 0 else 0
    if Tf == 0:
        full_input_ids = torch.empty((N, 0), dtype=torch.long, device=device)
        full_attention_mask = torch.empty((N, 0), dtype=torch.long, device=device)
        answer_mask = torch.empty((N, 0), dtype=torch.bool, device=device)
        return (
            full_input_ids,
            full_attention_mask,
            answer_mask,
            prefix_lens.to(device),
            torch.tensor(full_lens_list, dtype=torch.long, device=device),
        )

    full_input_ids = torch.full((N, Tf), int(pad_token_id), dtype=torch.long, device=device)
    full_attention_mask = torch.zeros((N, Tf), dtype=torch.long, device=device)
    answer_mask = torch.zeros((N, Tf), dtype=torch.bool, device=device)

    full_lens = torch.tensor(full_lens_list, dtype=torch.long, device=device)

    for i in range(N):
        ids = full_list[i].to(device)
        L = int(ids.numel())
        full_input_ids[i, :L] = ids
        full_attention_mask[i, :L] = 1

        Lp = int(prefix_lens[i].item())
        if Lp < 0:
            Lp = 0
        if Lp > L:
            Lp = L
        if L > Lp:
            answer_mask[i, Lp:L] = True

    return full_input_ids, full_attention_mask, answer_mask, prefix_lens.to(device), full_lens


@torch.inference_mode()
def generate_and_build_teacher_forcing(
    model,
    tokenizer,
    prompt_prefixes: List[str],
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> GenerationBatch:
    """
    English comment:
      1) Tokenize prefixes in batch
      2) Generate sequences
      3) Extract generated tokens from the generated sequences (token space)
      4) Build full teacher forcing input by token concatenation
      5) Decode generated tokens only for judge usage
    """
    model.eval()

    prefix_input_ids, prefix_attention_mask = tokenize_prefixes(
        tokenizer=tokenizer,
        prompt_prefixes=prompt_prefixes,
        device=device,
        max_length=max_length,
    )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        if eos_token_id is None:
            raise ValueError("pad_token_id and eos_token_id are both None")
        pad_token_id = int(eos_token_id)

    gen_cfg: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "pad_token_id": int(pad_token_id),
    }
    if eos_token_id is not None:
        gen_cfg["eos_token_id"] = int(eos_token_id)

    # model.generate returns sequences that include the prefix padded length
    prefix_padded_len = int(prefix_input_ids.shape[1])

    sequences = model.generate(
        input_ids=prefix_input_ids,
        attention_mask=prefix_attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=int(pad_token_id),
        eos_token_id=(int(eos_token_id) if eos_token_id is not None else None),
    )

    gen_output_ids = _trim_generated_part(
        full_sequences=sequences,
        prefix_padded_len=prefix_padded_len,
        eos_token_id=(int(eos_token_id) if eos_token_id is not None else None),
        pad_token_id=int(pad_token_id),
    )

    # Decode only newly generated tokens, used for judge input
    generated_text: List[str] = []
    for ids in gen_output_ids:
        if ids.numel() == 0:
            generated_text.append("")
            continue
        txt = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        generated_text.append(str(txt).strip())

    full_input_ids, full_attention_mask, answer_mask, prefix_lens, full_lens = _build_full_teacher_forcing(
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        gen_output_ids=gen_output_ids,
        pad_token_id=int(pad_token_id),
        device=device,
    )

    return GenerationBatch(
        prefix_input_ids=prefix_input_ids.detach(),
        prefix_attention_mask=prefix_attention_mask.detach(),
        gen_output_ids=gen_output_ids,
        generated_text=generated_text,
        full_input_ids=full_input_ids.detach(),
        full_attention_mask=full_attention_mask.detach(),
        answer_mask=answer_mask.detach(),
        prefix_lens=prefix_lens.detach(),
        full_lens=full_lens.detach(),
        gen_cfg=gen_cfg,
    )
