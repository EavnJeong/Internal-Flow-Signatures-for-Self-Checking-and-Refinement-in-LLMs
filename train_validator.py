# train_validator.py
import random
import os
import argparse
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from validators.dataset import build_dataloader
from validators.model import FlowGRUValidator
from validators.utils import set_seed, init_wandb
from validators.train import run_epoch


def estimate_pos_weight(train_loader) -> float:
    n_pos = 0
    n_neg = 0
    for batch in train_loader:
        y = batch["labels"].view(-1)
        n_pos += int((y == 1).sum().item())
        n_neg += int((y == 0).sum().item())
    if n_pos <= 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def _save_ckpt(
    path: str,
    epoch: int,
    model: nn.Module,
    opt: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    args: argparse.Namespace,
    best_auc: float,
) -> Dict[str, Any]:
    ckpt: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_auc": float(best_auc),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    else:
        ckpt["scaler"] = None
    torch.save(ckpt, path)
    return ckpt


def _load_ckpt_into(
    ckpt: Dict[str, Any],
    model: nn.Module,
    opt: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> None:
    model.load_state_dict(ckpt["model"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler", None) is not None:
        scaler.load_state_dict(ckpt["scaler"])


def main(args: argparse.Namespace) -> None:
    run = init_wandb(args)

    train_loader, test_loader = build_dataloader(args)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    first = next(iter(train_loader))
    feat_dim = int(first["evt_x"].shape[-1])

    model = FlowGRUValidator(
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        enc_hidden_dim=args.enc_hidden_dim,
        dropout=args.dropout,
        gru_layers=args.gru_layers,
        enc_layers=args.enc_layers,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp_scaler = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_scaler)

    pos_w = estimate_pos_weight(train_loader)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device), reduction="mean")
    print(f"[init] pos_weight={pos_w:.3f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] device={device} feat_dim={feat_dim} n_params={n_params}")

    if run is not None:
        try:
            run.config.update({"feat_dim": feat_dim, "n_params": n_params, "device": str(device)}, allow_val_change=True)
        except Exception:
            pass

    patience = int(getattr(args, "patience", 3))
    best_auc = float("-inf")
    best_ckpt: Optional[Dict[str, Any]] = None

    prev_te_auc = float("nan")
    drop_streak = 0

    for epoch in range(int(args.epochs)):
        tr = run_epoch(
            loader=train_loader,
            model=model,
            device=device,
            pool=args.pool,
            bce=bce,
            amp=args.amp,
            optimizer=opt,
            scaler=scaler,
            grad_clip=args.grad_clip,
            debug_every=args.debug_every,
            tag="train",
        )

        te = run_epoch(
            loader=test_loader,
            model=model,
            device=device,
            pool=args.pool,
            bce=bce,
            amp=args.amp,
            optimizer=None,
            scaler=None,
            grad_clip=0.0,
            debug_every=args.debug_every_test,
            tag="test",
        )

        te_auc = float(te.get("auroc", float("nan")))

        print(
            f"[epoch={epoch}] "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} auroc={tr['auroc']:.4f} skipB={int(tr['skipped_batches'])} gradN={tr['grad_norm']:.3f} | "
            f"test loss={te['loss']:.4f} acc={te['acc']:.4f} auroc={te_auc:.4f} skipB={int(te['skipped_batches'])} "
            f"drop_streak={drop_streak}/{patience} best_auc={best_auc:.4f}"
        )

        improved = (te_auc == te_auc) and (te_auc > best_auc)
        if improved:
            best_auc = te_auc
            drop_streak = 0
            if args.ckpt_path:
                best_ckpt = _save_ckpt(
                    path=args.ckpt_path,
                    epoch=epoch,
                    model=model,
                    opt=opt,
                    scaler=scaler if use_amp_scaler else None,
                    args=args,
                    best_auc=best_auc,
                )
                if run is not None and args.wandb_save_ckpt:
                    try:
                        wandb.save(args.ckpt_path)
                    except Exception:
                        pass

        if (prev_te_auc == prev_te_auc) and (te_auc == te_auc):
            if te_auc < prev_te_auc:
                drop_streak += 1
            else:
                drop_streak = 0
        else:
            drop_streak += 1

        if best_ckpt is not None and drop_streak >= patience:
            _load_ckpt_into(
                best_ckpt,
                model=model,
                opt=opt,
                scaler=scaler if use_amp_scaler else None,
            )
            if args.ckpt_path:
                torch.save(best_ckpt, args.ckpt_path)
            print(f"[rollback] restored best_ckpt with best_auc={float(best_ckpt['best_auc']):.4f}")
            drop_streak = 0
            prev_te_auc = float(best_ckpt["best_auc"])
        else:
            prev_te_auc = te_auc

        if run is not None:
            try:
                log_payload: Dict[str, float] = {"epoch": float(epoch), "lr": float(opt.param_groups[0]["lr"])}
                log_payload["best/test_auroc"] = float(best_auc) if (best_auc == best_auc) else float("nan")
                log_payload["test/auroc_current"] = float(te_auc) if (te_auc == te_auc) else float("nan")
                log_payload["ctrl/drop_streak"] = float(drop_streak)
                log_payload["ctrl/patience"] = float(patience)

                for prefix, out in [("train", tr), ("test", te)]:
                    for k, v in out.items():
                        if isinstance(v, (int, float)) and (v == v):
                            log_payload[f"{prefix}/{k}"] = float(v)

                wandb.log(log_payload, step=epoch)
            except Exception:
                pass

    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--shuffle_in_loader", action="store_true")
    parser.add_argument("--strict_keys", action="store_true")

    # Train
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pool", type=str, default="max", choices=["max", "lse"])

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--enc_hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--enc_layers", type=int, default=2)

    # Debug
    parser.add_argument("--debug_every", type=int, default=1)
    parser.add_argument("--debug_every_test", type=int, default=1)

    # Ckpt
    parser.add_argument("--wandb_save_ckpt", action="store_true")

    # Control
    parser.add_argument("--patience", type=int, default=3)

    # WandB
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="validator")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", type=str, default="")

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_seed(args.seed)
    args.ckpt_path = f"validator_ckpt_{args.wandb_run_name}.pt"
    main(args)
