"""
Model utilities for loading LLMs and extracting hidden state representations.
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    dtype: str = "float16",
    device: str = "cuda",
):
    """Load a HuggingFace model and tokenizer."""
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def get_layer_count(model) -> int:
    """Get the number of transformer layers in the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError(f"Cannot determine layer count for model type: {type(model)}")


def get_layer_module(model, layer_idx: int):
    """Get a specific transformer layer module."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Cannot access layer {layer_idx} for model type: {type(model)}")


@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    layers: Optional[list[int]] = None,
    token_position: str = "last",
    batch_size: int = 16,
    max_seq_length: int = 512,
) -> dict[int, torch.Tensor]:
    """
    Extract hidden states from specified layers for a list of texts.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: list of input texts
        layers: list of layer indices (None = all layers)
        token_position: "last" (last non-pad token), "mean" (mean pool), "first"
        batch_size: batch size for inference
        max_seq_length: max tokens per input

    Returns:
        dict mapping layer_idx -> tensor of shape (num_texts, hidden_dim)
    """
    num_layers = get_layer_count(model)
    if layers is None:
        layers = list(range(num_layers))

    all_hidden = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting representations"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

        attention_mask = inputs["attention_mask"]  # (batch, seq_len)

        for layer in layers:
            hs = hidden_states[layer]  # (batch, seq_len, hidden_dim)

            if token_position == "last":
                # Get last non-padding token
                seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
                reps = hs[torch.arange(hs.size(0)), seq_lengths]  # (batch, hidden_dim)
            elif token_position == "mean":
                # Mean pooling over non-padding tokens
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
                reps = (hs * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden_dim)
            elif token_position == "first":
                reps = hs[:, 0, :]  # (batch, hidden_dim)
            else:
                raise ValueError(f"Unknown token_position: {token_position}")

            all_hidden[layer].append(reps.cpu())

    # Concatenate all batches
    return {layer: torch.cat(tensors, dim=0) for layer, tensors in all_hidden.items()}


def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flush():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
