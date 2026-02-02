# refine/stopping.py
from __future__ import annotations

from typing import List
import torch
from transformers import StoppingCriteria


class StopOnNewline(StoppingCriteria):
    """
    Stops generation when a newline token appears in the generated suffix.
    Works best when tokenizer encodes '\n' as a single token for the model.
    """
    def __init__(self, newline_ids: List[int], start_len: int):
        super().__init__()
        self.newline_ids = set(int(x) for x in newline_ids)
        self.start_len = int(start_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: (B, cur_len)
        cur_len = int(input_ids.shape[1])
        if cur_len <= self.start_len:
            return False
        last_id = int(input_ids[0, -1].item())
        return last_id in self.newline_ids