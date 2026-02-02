from typing import Any, Dict, List, Tuple, Union
import torch


def _as_list_of_dict(batch: Any) -> List[Dict[str, Any]]:
    if isinstance(batch, list):
        if len(batch) == 0:
            return []
        if isinstance(batch[0], dict):
            return batch
        raise TypeError("Batch is a list but not list[dict].")

    if isinstance(batch, dict):
        # dict of lists -> list of dicts
        keys = list(batch.keys())
        n = None
        for k in keys:
            v = batch[k]
            if isinstance(v, list):
                n = len(v)
                break
        if n is None:
            # single example dict
            return [batch]
        out = []
        for i in range(n):
            ex = {k: (batch[k][i] if isinstance(batch[k], list) else batch[k]) for k in keys}
            out.append(ex)
        return out

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def build_prompts_from_halueval(batch: Any, task: str) -> List[str]:
    examples = _as_list_of_dict(batch)

    prompts: List[str] = []
    for ex in examples:
        q = ex.get("question") or ex.get("query") or ex.get("input") or ex.get("prompt")
        ctx = ex.get("context") or ex.get("passage") or ex.get("article") or ""
        choices = ex.get("choices") or ex.get("options") or None

        if task == "qa":
            p = ""
            if ctx:
                p += f"Context:\n{ctx}\n\n"
            p += f"Question:\n{q}\n\nAnswer:"
        elif task == "summarization":
            p = f"Summarize the following text:\n{ctx or q}\n\nSummary:"
        elif task == "dialogue":
            p = f"{q}\n\nAssistant:"
        else:  # general
            p = f"{q}\n\nAnswer:"

        if choices:
            if isinstance(choices, list):
                p = p + "\n\nChoices:\n" + "\n".join([f"- {c}" for c in choices])
            else:
                p = p + f"\n\nChoices:\n{choices}"

        prompts.append(p)

    return prompts


def encode_prompts(
    prompts: List[str],
    tokenizer,
    device: torch.device,
    max_length: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
