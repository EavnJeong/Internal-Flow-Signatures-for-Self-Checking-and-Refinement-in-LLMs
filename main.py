# main.py
import argparse
import os
import json
from typing import List, Dict, Any

import torch

from llm_signature.models import load_model
from llm_signature.judge import judge_labels_openai

from validators.dataset import build_dataloader
from validators.utils import set_seed

from refine.utils import _build_model_from_ckpt, collect_pred1_samples
from refine.find_culprit_events import find_culprit_event_per_sample
# from refine.debug_alignment import debug_culprit_alignment
from refine.intervene_generate import (
    InterventionConfig,
    rollback_and_regenerate_with_hidden_intervention,
)
from refine.utils import _find_sample_index, _build_judge_example, _safe_rate


def _counts_from_labels(labels: torch.Tensor) -> Dict[int, int]:
    return {
        -1: int((labels == -1).sum().item()),
        0: int((labels == 0).sum().item()),
        1: int((labels == 1).sum().item()),
    }


def _apply_transition_counts(counts: Dict[int, int], trans: Dict[str, int]) -> Dict[int, int]:
    out = dict(counts)
    for k, v in trans.items():
        a_str, b_str = k.split("->")
        a = int(a_str)
        b = int(b_str)
        v = int(v)
        out[a] = out.get(a, 0) - v
        out[b] = out.get(b, 0) + v
    return out


def _rates_from_counts(counts: Dict[int, int]) -> Dict[str, float]:
    n = counts.get(-1, 0) + counts.get(0, 0) + counts.get(1, 0)
    if n <= 0:
        return {"hallucination_rate": 0.0, "correction_rate": 0.0, "unidentified_rate": 0.0}
    return {
        "hallucination_rate": counts.get(1, 0) / n,
        "correction_rate": counts.get(0, 0) / n,
        "unidentified_rate": counts.get(-1, 0) / n,
    }


def extract_answer_only(text: str) -> str:
    if not text:
        return ""
    markers = ["\n\nAnswer:", "\nAnswer:"]
    best = None
    for m in markers:
        idx = text.rfind(m)
        if idx != -1:
            cand = text[idx + len(m):].strip()
            if cand:
                best = cand
                break
    return best if best is not None else text.strip()


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _collect_dataset_labels(test_loader) -> torch.Tensor:
    # English comment: Collect dataset-provided labels into a flat (N,) long tensor.
    labels_all: List[torch.Tensor] = []
    for batch in test_loader:
        lbs = batch.get("labels", None)
        if lbs is None:
            continue

        if isinstance(lbs, list):
            for x in lbs:
                if isinstance(x, torch.Tensor):
                    labels_all.append(x)
                else:
                    labels_all.append(torch.tensor(x))
            continue

        if isinstance(lbs, torch.Tensor):
            if lbs.ndim == 0:
                labels_all.append(lbs)
            else:
                for i in range(int(lbs.shape[0])):
                    labels_all.append(lbs[i])
            continue

        raise TypeError(f"unsupported labels type: {type(lbs)}")

    if len(labels_all) == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.stack(labels_all, dim=0).long()


def _flush_judge_buffers(
    *,
    args: argparse.Namespace,
    pending_records: List[Dict[str, Any]],
    pending_examples: List[Dict[str, Any]],
    pending_prompts: List[str],
    pending_answers_pregen: List[str],
    pending_answers_regen: List[str],
    refined_labels_all_t: torch.Tensor,
    transition_counts_all: Dict[str, int],
    transition_counts_nonzero: Dict[str, int],
    changed_items_nonzero: List[Dict[str, Any]],
) -> Dict[str, int]:
    if len(pending_examples) == 0:
        return {"n_updated_full_nonzero": 0, "n_nonzero": 0}

    print("[dbg] start judge chunk. n_items=", len(pending_examples), flush=True)
    labels_pregen, metas_pregen = judge_labels_openai(
        batch=pending_examples,
        task=str(args.task),
        prompt_prefixes=pending_prompts,
        answers=pending_answers_pregen,
        model=getattr(args, "judge_model", "gpt-4o-mini"),
        api_key_env=getattr(args, "judge_api_key_env", "OPENAI_API_KEY"),
        timeout_s=float(getattr(args, "judge_timeout_s", 60.0)),
        max_items_per_call=int(getattr(args, "judge_max_items_per_call", 16)),
        max_retries=int(getattr(args, "judge_max_retries", 3)),
    )

    labels_regen, metas_regen = judge_labels_openai(
        batch=pending_examples,
        task=str(args.task),
        prompt_prefixes=pending_prompts,
        answers=pending_answers_regen,
        model=getattr(args, "judge_model", "gpt-4o-mini"),
        api_key_env=getattr(args, "judge_api_key_env", "OPENAI_API_KEY"),
        timeout_s=float(getattr(args, "judge_timeout_s", 60.0)),
        max_items_per_call=int(getattr(args, "judge_max_items_per_call", 16)),
        max_retries=int(getattr(args, "judge_max_retries", 3)),
    )
    print("[dbg] judge chunk done. n_labels=", int(labels_pregen.numel()), flush=True)

    if int(labels_pregen.numel()) != len(pending_records) or int(labels_regen.numel()) != len(pending_records):
        raise RuntimeError("judge output size mismatch with pending records")

    n_updated_full_nonzero = 0
    n_nonzero = 0

    for i in range(len(pending_records)):
        rec = pending_records[i]
        c0 = rec["c0"]
        t0 = int(rec["t0"])
        b0 = int(rec["b0"])

        pregen_label = int(labels_pregen[i].item())
        regen_label = int(labels_regen[i].item())

        key_all = f"{pregen_label}->{regen_label}"
        transition_counts_all[key_all] = transition_counts_all.get(key_all, 0) + 1

        if pregen_label == 1:
            n_nonzero += 1
            key_nz = f"{pregen_label}->{regen_label}"
            transition_counts_nonzero[key_nz] = transition_counts_nonzero.get(key_nz, 0) + 1

            si = _find_sample_index(c0)
            if si is not None and 0 <= si < refined_labels_all_t.numel():
                refined_labels_all_t[si] = regen_label
                n_updated_full_nonzero += 1

            if pregen_label != regen_label:
                changed_items_nonzero.append(
                    {
                        "t": t0,
                        "b": b0,
                        "from": pregen_label,
                        "to": regen_label,
                        "pregen_reason": metas_pregen[i].get("reason", ""),
                        "pregen_conf": metas_pregen[i].get("confidence", 0.0),
                        "regen_reason": metas_regen[i].get("reason", ""),
                        "regen_conf": metas_regen[i].get("confidence", 0.0),
                    }
                )

        print("[judge] pregen_label=", pregen_label, "regen_label=", regen_label)
        print("--------------------------------------------")

    pending_records.clear()
    pending_examples.clear()
    pending_prompts.clear()
    pending_answers_pregen.clear()
    pending_answers_regen.clear()

    return {"n_updated_full_nonzero": n_updated_full_nonzero, "n_nonzero": n_nonzero}


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    llm, tokenizer = load_model(args.model, device_map=args.device)
    _, test_loader = build_dataloader(args)

    first = next(iter(test_loader))
    feat_dim = int(first["evt_x"].shape[-1])

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    validator = _build_model_from_ckpt(ckpt=ckpt, feat_dim=feat_dim, device=device)

    kept_samples = collect_pred1_samples(
        model=validator,
        loader=test_loader,
        device=device,
        pool=args.pool,
        amp=bool(args.amp and device.type == "cuda"),
        threshold=float(args.threshold),
        max_keep_samples=args.max_keep_batches,
        store_on_cpu=bool(args.store_on_cpu),
        debug_every=int(args.debug_every),
    )

    culprit_samples = find_culprit_event_per_sample(kept_samples)

    draft_labels_all_t = _collect_dataset_labels(test_loader)
    draft_hallucination_rate = _safe_rate(draft_labels_all_t, 1)
    draft_correction_rate = _safe_rate(draft_labels_all_t, 0)
    draft_unidentified_rate = _safe_rate(draft_labels_all_t, -1)

    refined_labels_all_t = draft_labels_all_t.clone()

    regen_records: List[Dict[str, Any]] = []

    transition_counts_all: Dict[str, int] = {}
    transition_counts_nonzero: Dict[str, int] = {}

    changed_items_nonzero: List[Dict[str, Any]] = []

    pending_records: List[Dict[str, Any]] = []
    pending_examples: List[Dict[str, Any]] = []
    pending_prompts: List[str] = []
    pending_answers_pregen: List[str] = []
    pending_answers_regen: List[str] = []

    n_updated_full_nonzero_total = 0
    n_nonzero_total = 0

    flush_at = int(getattr(args, "judge_max_items_per_call", 16))
    if flush_at <= 0:
        flush_at = 16

    for step_idx, c0 in enumerate(culprit_samples):
        t0 = int(c0["t"])
        b0 = int(c0["b"])
        input_ids_1d = c0["input_ids"]
        previous_answer = c0.get("generated_text", "")

        if isinstance(input_ids_1d, torch.Tensor) and input_ids_1d.ndim == 2:
            input_ids_1d = input_ids_1d[0]
        if not (isinstance(input_ids_1d, torch.Tensor) and input_ids_1d.ndim == 1):
            raise RuntimeError("culprit sample has no valid input_ids")

        is_flow = (args.mode == "flow")
        cfg = InterventionConfig(
            layer_idx=b0,
            ref_span=64,
            max_ratio=1.05,
            k_dim=16,
            comp_k=32,
            enabled=is_flow,
        )

        res = rollback_and_regenerate_with_hidden_intervention(
            model=llm,
            tokenizer=tokenizer,
            input_ids_1d=input_ids_1d,
            rollback_t=t0,
            gen_max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            cfg=cfg,
        )

        regen_text_raw = res.get("text", "")
        prompt_text = tokenizer.decode(input_ids_1d, skip_special_tokens=False)
        ex = _build_judge_example(c0, task=args.task)

        regen_text = extract_answer_only(str(regen_text_raw))

        regen_records.append(
            {
                "c0": c0,
                "t0": t0,
                "b0": b0,
                "step": int(step_idx),
                "previous_answer": str(previous_answer),
                "regen_text": str(regen_text),
                "prompt_text": str(prompt_text),
                "ex": ex,
                "rollback_t": res.get("rollback_t", t0),
                "hook_info": res.get("hook_last_info", None),
            }
        )

        pending_records.append({"c0": c0, "t0": t0, "b0": b0, "step": int(step_idx)})
        pending_examples.append(ex)
        pending_prompts.append(str(prompt_text))
        pending_answers_pregen.append(str(previous_answer))
        pending_answers_regen.append(str(regen_text))

        print("===========================================")
        print("[prompt] = ", c0["prompt"])
        print("--------------------------------------------")
        print("[regen] text=\n", regen_text)
        print("--------------------------------------------")
        print("[pregen] previous answer=\n", previous_answer)
        print("--------------------------------------------")
        print("label (dataset):", c0.get("label", None))
        print("===========================================")

        if args.save_jsonl:
            _append_jsonl(
                args.save_jsonl,
                {
                    "stage": "regen",
                    "step": int(step_idx),
                    "pregen": str(previous_answer),
                    "regen": str(regen_text),
                    "label_dataset": c0.get("label", None),
                },
            )

        if len(pending_examples) >= flush_at:
            out = _flush_judge_buffers(
                args=args,
                pending_records=pending_records,
                pending_examples=pending_examples,
                pending_prompts=pending_prompts,
                pending_answers_pregen=pending_answers_pregen,
                pending_answers_regen=pending_answers_regen,
                refined_labels_all_t=refined_labels_all_t,
                transition_counts_all=transition_counts_all,
                transition_counts_nonzero=transition_counts_nonzero,
                changed_items_nonzero=changed_items_nonzero,
            )
            n_updated_full_nonzero_total += int(out["n_updated_full_nonzero"])
            n_nonzero_total += int(out["n_nonzero"])

    if len(pending_examples) > 0:
        out = _flush_judge_buffers(
            args=args,
            pending_records=pending_records,
            pending_examples=pending_examples,
            pending_prompts=pending_prompts,
            pending_answers_pregen=pending_answers_pregen,
            pending_answers_regen=pending_answers_regen,
            refined_labels_all_t=refined_labels_all_t,
            transition_counts_all=transition_counts_all,
            transition_counts_nonzero=transition_counts_nonzero,
            changed_items_nonzero=changed_items_nonzero,
        )
        n_updated_full_nonzero_total += int(out["n_updated_full_nonzero"])
        n_nonzero_total += int(out["n_nonzero"])

    print("[stats] transitions_all:", dict(sorted(transition_counts_all.items(), key=lambda x: x[0])))
    print("[stats] transitions_nonzero:", dict(sorted(transition_counts_nonzero.items(), key=lambda x: x[0])))

    if draft_labels_all_t.numel() > 0:
        counts_before = _counts_from_labels(draft_labels_all_t)
        rates_before = _rates_from_counts(counts_before)

        counts_after_nz = _apply_transition_counts(counts_before, transition_counts_nonzero)
        rates_after_nz = _rates_from_counts(counts_after_nz)

        counts_after_all = _apply_transition_counts(counts_before, transition_counts_all)
        rates_after_all = _rates_from_counts(counts_after_all)

        print(
            f"[stats] full_before(draft): "
            f"counts={{-1:{counts_before[-1]}, 0:{counts_before[0]}, 1:{counts_before[1]}}}, "
            f"hallucination_rate={rates_before['hallucination_rate']:.4f}, "
            f"correction_rate={rates_before['correction_rate']:.4f}, "
            f"unidentified_rate={rates_before['unidentified_rate']:.4f}"
        )

        print(
            f"[stats] full_after(nonzero_applied, count_shift): "
            f"applied_n={sum(transition_counts_nonzero.values())}, "
            f"counts={{-1:{counts_after_nz.get(-1,0)}, 0:{counts_after_nz.get(0,0)}, 1:{counts_after_nz.get(1,0)}}}, "
            f"hallucination_rate={rates_after_nz['hallucination_rate']:.4f}, "
            f"correction_rate={rates_after_nz['correction_rate']:.4f}, "
            f"unidentified_rate={rates_after_nz['unidentified_rate']:.4f}"
        )

        print(
            f"[stats] full_after(all_applied, count_shift): "
            f"applied_n={sum(transition_counts_all.values())}, "
            f"counts={{-1:{counts_after_all.get(-1,0)}, 0:{counts_after_all.get(0,0)}, 1:{counts_after_all.get(1,0)}}}, "
            f"hallucination_rate={rates_after_all['hallucination_rate']:.4f}, "
            f"correction_rate={rates_after_all['correction_rate']:.4f}, "
            f"unidentified_rate={rates_after_all['unidentified_rate']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)

    # Data
    parser.add_argument("--flow_root", type=str, default="flow")
    parser.add_argument("--dataset", type=str, default="halueval")
    parser.add_argument("--task", type=str, default="dialogue")
    parser.add_argument("--model", type=str, default="qwen25")

    # Loader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--shuffle_in_loader", action="store_true")
    parser.add_argument("--strict_keys", action="store_true")

    # Infer
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pool", type=str, default="max", choices=["max", "lse"])
    parser.add_argument("--threshold", type=float, default=0.5)

    # Control
    parser.add_argument("--max_keep_batches", type=int, default=None)
    parser.add_argument("--store_on_cpu", action="store_true", default=True)
    parser.add_argument("--debug_every", type=int, default=0)
    parser.add_argument("--print_k", type=int, default=10)

    # Judge (OpenAI)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--judge_timeout_s", type=float, default=60.0)
    parser.add_argument("--judge_max_items_per_call", type=int, default=16)
    parser.add_argument("--judge_max_retries", type=int, default=3)

    parser.add_argument("--save_jsonl", type=str, default="logs/pregen_regen.jsonl")
    parser.add_argument("--mode", type=str, default="flow", choices=["flow", "rollback"])

    args = parser.parse_args()
    set_seed(int(args.seed))
    main(args)
