import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, dtype=torch.float32, device_map="auto"):
    if model_name == 'qwen25':
        name = "Qwen/Qwen2.5-7B-Instruct"
    elif model_name == 'gemma':
        name = "google/gemma-2-9b-it"
    elif model_name == 'phi':
        name = "microsoft/Phi-3.5-mini-instruct"
    elif model_name == 'llama3':
        name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == 'mistral':
        name = "mistralai/Mistral-7B-Instruct-v0.2"
    else:
        raise ValueError(f"Model {model_name} not supported.")

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if model_name == 'llama3' or model_name == 'qwen25':
        tok.padding_side = "left"
        
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name, 
        dtype=dtype,
        device_map=device_map
    )
    model.eval()
    return model, tok