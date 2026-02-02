import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import torch
from torch.utils.data import Dataset


Task = Literal["qa", "dialogue", "summarization", "general"]
Mode = Literal["sample_one", "expand"]


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return [x for x in obj["data"] if isinstance(x, dict)]
            # Rare case: dict of id -> example
            vals = list(obj.values())
            if all(isinstance(v, dict) for v in vals):
                return vals  # type: ignore[return-value]
        raise ValueError("Unsupported JSON structure")
    except json.JSONDecodeError:
        pass

    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            data.append(json.loads(s))
    return [x for x in data if isinstance(x, dict)]


def _label_from_yesno(x: Any) -> Optional[bool]:
    # True means hallucination present
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"yes", "true", "1"}:
            return True
        if v in {"no", "false", "0"}:
            return False
    return None


@dataclass
class HaluEvalItem:
    task: Task
    context: str
    query: str
    output: str
    label: Optional[bool]  # True hallucination, False non-hallucination
    meta: Dict[str, Any]


class HaluEvalDataset(Dataset):
    """
    Category-specific HaluEval dataset.

    Expected fields (per official repo README):
      - qa_data.json: knowledge, question, right_answer, hallucinated_answer
      - dialogue_data.json: knowledge, dialogue_history, right_response, hallucinated_response
      - summarization_data.json: document, right_summary, hallucinated_summary
      - general_data.json: user_query, chatgpt_response, hallucination_label
    """

    def __init__(
        self,
        data_path: str,
        task: Task,
        mode: Mode = "sample_one",
        seed: int = 0,
        limit: int = 0,
        keep_raw: bool = False,
        require_label: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.task = task
        self.mode = mode
        self.seed = int(seed)
        self.keep_raw = bool(keep_raw)
        self.require_label = bool(require_label)

        raw = _read_json_or_jsonl(data_path)
        if limit and limit > 0:
            raw = raw[:limit]

        self.items: List[HaluEvalItem] = []
        if task == "general":
            self._build_general(raw)
        elif task == "qa":
            self._build_qa(raw)
        elif task == "dialogue":
            self._build_dialogue(raw)
        elif task == "summarization":
            self._build_summarization(raw)
        else:
            raise ValueError(f"Unknown task: {task}")

        if self.require_label:
            self.items = [it for it in self.items if it.label is not None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        it = self.items[i]
        return {
            "task": it.task,
            "context": it.context,
            "query": it.query,
            "output": it.output,
            "label": it.label,
            "meta": it.meta,
        }

    def _meta(self, idx: int, ex: Dict[str, Any], chosen: str) -> Dict[str, Any]:
        m: Dict[str, Any] = {"index": idx, "chosen": chosen}
        if self.keep_raw:
            m["raw"] = ex
        return m

    def _build_general(self, raw: List[Dict[str, Any]]) -> None:
        # user_query, chatgpt_response, hallucination_label
        for idx, ex in enumerate(raw):
            q = str(ex.get("user_query", "")).strip()
            out = str(ex.get("chatgpt_response", "")).strip()
            lb = _label_from_yesno(ex.get("hallucination"))
            if not out:
                continue
            self.items.append(
                HaluEvalItem(
                    task="general",
                    context="",
                    query=q,
                    output=out,
                    label=lb,
                    meta=self._meta(idx, ex, "direct"),
                )
            )

    def _build_qa(self, raw: List[Dict[str, Any]]) -> None:
        # knowledge, question, right_answer, hallucinated_answer
        for idx, ex in enumerate(raw):
            knowledge = str(ex.get("knowledge", "")).strip()
            question = str(ex.get("question", "")).strip()
            right = str(ex.get("right_answer", "")).strip()
            hall = str(ex.get("hallucinated_answer", "")).strip()
            if not right or not hall:
                continue

            if self.mode == "expand":
                self.items.append(
                    HaluEvalItem(
                        task="qa",
                        context=knowledge,
                        query=question,
                        output=right,
                        label=False,
                        meta=self._meta(idx, ex, "right_answer"),
                    )
                )
                self.items.append(
                    HaluEvalItem(
                        task="qa",
                        context=knowledge,
                        query=question,
                        output=hall,
                        label=True,
                        meta=self._meta(idx, ex, "hallucinated_answer"),
                    )
                )
            else:
                rng = random.Random(self.seed * 1_000_003 + idx)
                choose_hall = rng.random() < 0.5
                self.items.append(
                    HaluEvalItem(
                        task="qa",
                        context=knowledge,
                        query=question,
                        output=(hall if choose_hall else right),
                        label=(True if choose_hall else False),
                        meta=self._meta(idx, ex, "hallucinated_answer" if choose_hall else "right_answer"),
                    )
                )

    def _build_dialogue(self, raw: List[Dict[str, Any]]) -> None:
        # knowledge, dialogue_history, right_response, hallucinated_response
        for idx, ex in enumerate(raw):
            knowledge = str(ex.get("knowledge", "")).strip()
            history = str(ex.get("dialogue_history", "")).strip()
            right = str(ex.get("right_response", "")).strip()
            hall = str(ex.get("hallucinated_response", "")).strip()
            if not right or not hall:
                continue

            # Keep consistent interface: context carries knowledge, query carries dialogue history
            if self.mode == "expand":
                self.items.append(
                    HaluEvalItem(
                        task="dialogue",
                        context=knowledge,
                        query=history,
                        output=right,
                        label=False,
                        meta=self._meta(idx, ex, "right_response"),
                    )
                )
                self.items.append(
                    HaluEvalItem(
                        task="dialogue",
                        context=knowledge,
                        query=history,
                        output=hall,
                        label=True,
                        meta=self._meta(idx, ex, "hallucinated_response"),
                    )
                )
            else:
                rng = random.Random(self.seed * 1_000_003 + idx)
                choose_hall = rng.random() < 0.5
                self.items.append(
                    HaluEvalItem(
                        task="dialogue",
                        context=knowledge,
                        query=history,
                        output=(hall if choose_hall else right),
                        label=(True if choose_hall else False),
                        meta=self._meta(idx, ex, "hallucinated_response" if choose_hall else "right_response"),
                    )
                )

    def _build_summarization(self, raw: List[Dict[str, Any]]) -> None:
        # document, right_summary, hallucinated_summary
        for idx, ex in enumerate(raw):
            doc = str(ex.get("document", "")).strip()
            right = str(ex.get("right_summary", "")).strip()
            hall = str(ex.get("hallucinated_summary", "")).strip()
            if not right or not hall:
                continue

            # Keep consistent interface: context carries document, query can be a fixed instruction
            instruction = "Summarize the document."
            if self.mode == "expand":
                self.items.append(
                    HaluEvalItem(
                        task="summarization",
                        context=doc,
                        query=instruction,
                        output=right,
                        label=False,
                        meta=self._meta(idx, ex, "right_summary"),
                    )
                )
                self.items.append(
                    HaluEvalItem(
                        task="summarization",
                        context=doc,
                        query=instruction,
                        output=hall,
                        label=True,
                        meta=self._meta(idx, ex, "hallucinated_summary"),
                    )
                )
            else:
                rng = random.Random(self.seed * 1_000_003 + idx)
                choose_hall = rng.random() < 0.5
                self.items.append(
                    HaluEvalItem(
                        task="summarization",
                        context=doc,
                        query=instruction,
                        output=(hall if choose_hall else right),
                        label=(True if choose_hall else False),
                        meta=self._meta(idx, ex, "hallucinated_summary" if choose_hall else "right_summary"),
                    )
                )


def collate_halueval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = [b["task"] for b in batch]
    contexts = [b["context"] for b in batch]
    queries = [b["query"] for b in batch]
    outputs = [b["output"] for b in batch]
    metas = [b.get("meta", {}) for b in batch]
    labels_list = [b.get("label", None) for b in batch]
    labels = torch.tensor(
        [(-1 if lb is None else (1 if lb else 0)) for lb in labels_list],
        dtype=torch.long,
    )
    return {
        "task": tasks,
        "context": contexts,
        "query": queries,
        "output": outputs,
        "label": labels,
        "meta": metas,
    }