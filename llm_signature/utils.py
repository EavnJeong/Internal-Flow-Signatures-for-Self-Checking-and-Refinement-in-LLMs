import torch


@torch.no_grad()
def get_W(model):
    out = model.get_output_embeddings()
    if out is None:
        raise RuntimeError("No output embeddings found.")
    return out.weight  # (V, D)


@torch.no_grad()
def depth_topk_from_trace(trace_h_out, W, topk=32, device="cuda"):
    B, N, T, D = trace_h_out.shape

    # Use the same dtype as h for safe matmul
    # Also move to compute device once
    W = W.to(device)

    topk_ids = []
    topk_probs = []
    for b in range(B):
        h = trace_h_out[b].to(device)            # (N,T,D), dtype maybe float16
        Wb = W.to(dtype=h.dtype)                 # match dtype
        logits = torch.matmul(h, Wb.t())         # (N,T,V)
        probs = torch.softmax(logits, dim=-1)
        p, idx = torch.topk(probs, k=topk, dim=-1)
        topk_ids.append(idx.cpu())
        topk_probs.append(p.cpu())

    return torch.stack(topk_ids, 0), torch.stack(topk_probs, 0)


@torch.no_grad()
def final_y_star_from_trace(trace_h_out, W, device="cuda"):
    # trace_h_out: (B,N,T,D) on cpu ok
    hB = trace_h_out[-1].to(device)          # (N,T,D)
    W = W.to(device).to(dtype=hB.dtype)      # (V,D)
    logits = torch.matmul(hB, W.t())         # (N,T,V)
    y_star = torch.argmax(logits, dim=-1)    # (N,T)
    return y_star.cpu()


@torch.no_grad()
def get_beta_by_depth(model, adapter, B: int, D: int, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    betas = torch.zeros(B, D, device=device, dtype=dtype)

    layers = adapter.get_layers(model)  # should return a list-like of transformer blocks
    # If adapter returns a different count, we use min alignment.
    L = min(len(layers), B)

    for b in range(L):
        layer = layers[b]

        # Try common norm attributes. Most RMSNorm have no bias, so these will be None.
        cand = []
        for name in ["post_attention_layernorm", "input_layernorm", "final_layernorm", "ln", "norm"]:
            if hasattr(layer, name):
                cand.append(getattr(layer, name))

        beta = None
        for norm in cand:
            if norm is None:
                continue
            if hasattr(norm, "bias") and norm.bias is not None:
                beta = norm.bias.detach()
                break

        if beta is not None:
            betas[b, : beta.numel()] = beta.to(device=device, dtype=dtype)

    return betas
