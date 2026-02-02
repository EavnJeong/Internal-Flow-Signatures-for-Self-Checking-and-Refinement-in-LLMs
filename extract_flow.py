# extract_flow.py
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch
from tqdm import tqdm

from llm_signature.models import load_model
from dataset.getter import load_dataset
from llm_signature.tracer import LlamaLikeAdapter, TraceCollector
from llm_signature.utils import get_W, depth_topk_from_trace, get_beta_by_depth
from llm_signature.signatures import make_windows, compute_flow_signatures, drift_series_from_subspaces
from llm_signature.judge import judge_labels_openai as judge_labels

from llm_signature.pipeline import (
    ensure_dir,
    infer_max_len_global,
    get_boundary_norm_cfg_by_depth,
    build_prompt_prefixes,
    generate_answers,
    filter_by_labels,
    make_teacher_forcing_inputs,
    build_mask_for_signatures,
    build_U_list,
    save_flow,
    build_extra_payload,
)


def main(args: argparse.Namespace) -> None:
    with open("config/data.json", "r") as f:
        data_config = json.load(f)

    model, tokenizer = load_model(args.model, device_map=args.device_map)
    adapter = LlamaLikeAdapter()

    collector = TraceCollector(
        model=model,
        adapter=adapter,
        store_device="cuda:0",
        dtype=torch.float16,
    )

    W = get_W(model).detach().to("cuda:0").float()
    norm_cfg_by_depth = get_boundary_norm_cfg_by_depth(model, adapter)
    max_len_global = infer_max_len_global(model, tokenizer)

    args.out_dir = os.path.join(f"flow/{args.dataset}/{args.task}/{args.model}/")
    ensure_dir(args.out_dir)

    dataloader = load_dataset(args, data_config[args.dataset])

    beta_by_depth: Optional[torch.Tensor] = None

    for step, batch in enumerate(tqdm(dataloader, desc="Extract flow")):
        flow_path = os.path.join(args.out_dir, f"flow_{step:06d}.pt")
        if os.path.exists(flow_path):
            print(f"[batch {step}] exists -> skip ({flow_path})")
            continue

        t0 = time.time()

        # 1) Prompts
        prompt_prefixes = build_prompt_prefixes(batch=batch, task=args.task)

        # 2) Generate
        answers, gen_meta = generate_answers(
            model=model,
            tokenizer=tokenizer,
            prompt_prefixes=prompt_prefixes,
            device=model.device,
            max_length=max_len_global,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        t1 = time.time()

        # 3) Judge labels
        labels, judge_meta = judge_labels(
            batch=batch,
            task=args.task,
            prompt_prefixes=prompt_prefixes,
            answers=answers,
            api_key_env=args.judge_api_key_env,
            timeout_s=args.judge_timeout_s,
        )
        t2 = time.time()

        # 4) Filter (optional)
        batch, prompt_prefixes, answers, labels, judge_meta = filter_by_labels(
            batch=batch,
            prompt_prefixes=prompt_prefixes,
            answers=answers,
            labels=labels,
            judge_meta=judge_meta,
            require_label=args.require_judge_label,
        )
        if len(prompt_prefixes) == 0:
            continue

        # 5) Teacher forcing input + answer_mask
        tf_input_ids, tf_attention_mask, answer_mask = make_teacher_forcing_inputs(
            tokenizer=tokenizer,
            prompt_prefixes=prompt_prefixes,
            answers=answers,
            device=model.device,
            max_length=max_len_global,
        )
        t3 = time.time()

        # 6) Trace
        trace = collector.collect(input_ids=tf_input_ids, attention_mask=tf_attention_mask)
        t4 = time.time()

        for need in ["h_out", "o", "m"]:
            if not hasattr(trace, need) or getattr(trace, need) is None:
                raise RuntimeError(f"TraceCollector must capture trace.{need}")

        B, N, T, D = trace.h_out.shape

        if beta_by_depth is None:
            beta_by_depth = get_beta_by_depth(
                model=model,
                adapter=adapter,
                B=int(B),
                D=int(D),
                device="cuda:0",
                dtype=torch.float32,
            )

        # 7) Mask for signatures + U fitting
        mask_for_sig = build_mask_for_signatures(
            tf_attention_mask=tf_attention_mask,
            answer_mask=answer_mask,
            mask_answer_only=args.mask_answer_only,
            min_mask_tokens=args.min_mask_tokens,
        )

        # 8) Topk
        if args.topk < args.K + 1:
            raise ValueError("--topk must be >= K+1")
        topk_ids, _ = depth_topk_from_trace(trace.h_out, W, topk=args.topk, device=args.logit_device)
        topk_ids = topk_ids.detach().to("cuda:0").long()
        t5 = time.time()

        # 9) Windows + subspaces + drift
        windows = make_windows(num_blocks=int(B), L=args.L, stride=args.stride)
        U_list = build_U_list(
            topk_ids=topk_ids,
            attention_mask=mask_for_sig,
            W=W,
            windows=windows,
            K=args.K,
            k=args.k,
            cap_per_window=args.max_dirs_per_window,
            seed=args.seed,
        )
        drift_win = drift_series_from_subspaces(U_list)
        t6 = time.time()

        # 10) Signatures
        flow = compute_flow_signatures(
            trace_h_out=trace.h_out,
            trace_o=trace.o,
            trace_m=trace.m,
            U_list=U_list,
            windows=windows,
            attention_mask=mask_for_sig,
            norm_cfg_by_depth=norm_cfg_by_depth,
            beta_by_depth=beta_by_depth,
            eps=args.eps,
            drift_anchor=args.drift_anchor,
            simpson=(not args.no_simpson),
            device="cuda:0",
        )
        flow.drift_win.copy_(drift_win)
        t7 = time.time()

        extra = build_extra_payload(
            batch=batch,
            prompt_prefixes=prompt_prefixes,
            answers=answers,
            labels=labels,
            judge_meta=judge_meta,
            gen_meta=gen_meta,
            args=vars(args),
        )

        # 11) Save
        save_flow(
            out_dir=args.out_dir,
            step=step,
            flow=flow,
            input_ids=tf_input_ids,
            attention_mask=tf_attention_mask,
            answer_mask=answer_mask,
            windows=windows,
            labels=labels,
            extra=extra,
        )

        print(
            f"[batch {step}] B={int(B)} N={int(N)} T={int(T)} D={int(D)} windows={len(windows)} "
            f"time(prompt+gen)={t1-t0:.3f}s time(judge)={t2-t1:.3f}s time(tf)={t3-t2:.3f}s "
            f"time(trace)={t4-t3:.3f}s time(topk)={t5-t4:.3f}s time(U)={t6-t5:.3f}s time(sig)={t7-t6:.3f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model and data
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--model", type=str, default="qwen25",
                        choices=["qwen25", "gemma", "phi", "llama3", "mistral"])
    parser.add_argument("--task", type=str, default="qa")
    parser.add_argument("--dataset", type=str, default="halueval", choices=["halueval"])
    parser.add_argument("--batch_size", type=int, default=4)

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # judge
    parser.add_argument("--judge", type=str, default="none", choices=["none", "rest"])
    parser.add_argument("--judge_endpoint", type=str, default="")
    parser.add_argument("--judge_timeout_s", type=float, default=60.0)
    parser.add_argument("--require_judge_label", action="store_true")
    parser.add_argument("--judge_api_key_env", type=str, default="OPENAI_API_KEY")

    # competitor and subspace
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_dirs_per_window", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)

    # masks
    parser.add_argument("--mask_answer_only", action="store_true")
    parser.add_argument("--min_mask_tokens", type=int, default=16)

    # attribution and numerics
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--drift_anchor", type=str, default="end", choices=["end", "start"])
    parser.add_argument("--no_simpson", action="store_true")

    # run control
    parser.add_argument("--logit_device", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.judge == "rest" and not args.judge_endpoint:
        raise ValueError("--judge rest requires --judge_endpoint")

    if not args.mask_answer_only:
        args.mask_answer_only = True

    main(args)
