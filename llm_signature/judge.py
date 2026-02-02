# llm_signature/judge.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import Literal


def _as_examples(batch: Any) -> List[Dict[str, Any]]:
    """
    English comment: Normalize batch into list[dict].
    """
    if isinstance(batch, list):
        if len(batch) == 0:
            return []
        if isinstance(batch[0], dict):
            return batch
        raise TypeError("batch is list but not list[dict]")

    if isinstance(batch, dict):
        keys = list(batch.keys())
        n: Optional[int] = None
        for k in keys:
            if isinstance(batch[k], list):
                n = len(batch[k])
                break
        if n is None:
            return [batch]
        out: List[Dict[str, Any]] = []
        for i in range(n):
            ex: Dict[str, Any] = {}
            for k in keys:
                v = batch[k]
                ex[k] = v[i] if isinstance(v, list) else v
            out.append(ex)
        return out

    raise TypeError(f"unsupported batch type: {type(batch)}")


class JudgeItem(BaseModel):
    label: Literal[0, 1, -1] = Field(
        description="1 hallucination, 0 not hallucination, -1 unknown"
    )
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class JudgeBatch(BaseModel):
    results: List[JudgeItem]


def _chunked(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [xs]
    return [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]


def judge_labels_openai(
    batch: Any,
    task: str,
    prompt_prefixes: List[str],
    answers: List[str],
    model: str = "gpt-5-mini",
    api_key_env: str = "OPENAI_API_KEY",
    timeout_s: float = 60.0,
    max_items_per_call: int = 16,
    max_retries: int = 3,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    English comment:
      Returns:
        labels: (N,) long with values in {0,1,-1}
        metas: list of dicts with reason, confidence, and raw fields
    """
    if len(prompt_prefixes) != len(answers):
        raise ValueError("prompt_prefixes and answers length mismatch")

    api_key = os.environ.get(api_key_env, None)
    if not api_key:
        raise RuntimeError(f"Missing API key in env: {api_key_env}")

    client = OpenAI(api_key=api_key)

    examples = _as_examples(batch)
    if len(examples) not in (0, len(prompt_prefixes)):
        # Fallback: still judge using prompt and answer only
        examples = [{} for _ in range(len(prompt_prefixes))]

    items: List[Dict[str, Any]] = []
    for ex, p, a in zip(examples, prompt_prefixes, answers):
        ctx = ex.get("context") or ex.get("knowledge") or ex.get("document") or ""
        q = ex.get("query") or ex.get("question") or ex.get("dialogue_history") or ex.get("user_query") or ""
        items.append(
            {
                "task": str(task),
                "query": "" if q is None else str(q),
                "context": "" if ctx is None else str(ctx),
                "prompt": "" if p is None else str(p),
                "answer": "" if a is None else str(a),
            }
        )

    labels_out = torch.full((len(items),), -1, dtype=torch.long)
    metas_out: List[Dict[str, Any]] = [{"judge": "openai", "raw": None} for _ in range(len(items))]

    chunks = _chunked(list(range(len(items))), max_items_per_call)

    system_text = (
        "You are a strict hallucination judge.\n"
        "Decide whether the answer contains hallucination.\n"
        "Use the provided query and context when available.\n"
        "Return label 1 if the answer makes factual claims not supported by the context or obviously incorrect.\n"
        "Return label 0 if the answer is supported or appropriately uncertain.\n"
        "Return label -1 if you cannot judge.\n"
        "Keep reason short."
    )

    for chunk_ids in chunks:
        chunk_items = [items[i] for i in chunk_ids]

        user_text = {
            "items": chunk_items,
            "output_spec": {
                "results": [
                    {"label": "1|0|-1", "reason": "short string", "confidence": "0..1"}
                ]
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                # Structured Outputs parsing helper in the OpenAI SDK.
                # response.output_parsed is a Pydantic object. :contentReference[oaicite:1]{index=1}
                response = client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": str(user_text)},
                    ],
                    text_format=JudgeBatch,
                    timeout=timeout_s,
                )
                parsed: JudgeBatch = response.output_parsed
                if not isinstance(parsed, JudgeBatch) or len(parsed.results) != len(chunk_items):
                    raise RuntimeError("invalid judge output length")

                for local_j, item_id in enumerate(chunk_ids):
                    r = parsed.results[local_j]
                    labels_out[item_id] = int(r.label)
                    metas_out[item_id] = {
                        "judge": "openai",
                        "label": int(r.label),
                        "reason": r.reason,
                        "confidence": float(r.confidence),
                        "raw": r.model_dump(),
                    }
                last_err = None
                break

            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                else:
                    for item_id in chunk_ids:
                        metas_out[item_id] = {
                            "judge": "openai",
                            "label": -1,
                            "reason": "",
                            "confidence": 0.0,
                            "raw": {"error": repr(last_err)},
                        }

    return labels_out, metas_out
