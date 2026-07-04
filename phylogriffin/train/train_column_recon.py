"""
PhyloGriffin v3 -- Train-1: Masked Column Reconstruction.
Self-supervised pre-training of the Griffin column processor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import math
from typing import Optional

from ..config import PhyloGriffinConfig


def train_column_reconstruction(
    model: "ColumnProcessor",
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> "ColumnProcessor":
    model = model.to(device)
    model.train()

    alphabet_size = config.alphabet_size
    mask_token_idx = alphabet_size

    mask_head = nn.Linear(config.griffin.d_model, alphabet_size).to(device)

    params = list(model.parameters()) + list(mask_head.parameters())
    optimizer = AdamW(params, lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=config.training.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.training.max_steps - config.training.warmup_steps
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[config.training.warmup_steps])

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=config.training.max_steps, desc="Train-1")

    while global_step < config.training.max_steps:
        for batch in dataloader:
            if global_step >= config.training.max_steps:
                break

            msa = batch["msa"].to(device)
            mask = batch["mask"].to(device)

            B, N, L = msa.shape

            masked_msa = msa.clone()
            mask_positions = torch.rand(B, N, L, device=device) < 0.15
            mask_positions = mask_positions & mask

            replace_with_mask = torch.rand(B, N, L, device=device) < 0.8
            replace_with_random = (torch.rand(B, N, L, device=device) >= 0.8) & (torch.rand(B, N, L, device=device) < 0.9)

            for b in range(B):
                mask_pos = mask_positions[b]
                masked_msa[b][mask_pos & replace_with_mask[b]] = mask_token_idx
                n_random = (mask_pos & replace_with_random[b]).sum().item()
                if n_random > 0:
                    random_tokens = torch.randint(0, alphabet_size, (n_random,), device=device)
                    masked_msa[b][mask_pos & replace_with_random[b]] = random_tokens

            seq_emb, _ = model(masked_msa, mask)

            x = model.token_embed(masked_msa)
            mask_copy = mask

            if model.titans is not None:
                model.titans.reset_state()

            for layer_idx, layer in enumerate(model.griffin_layers):
                if layer.is_recurrent:
                    x_temporal = model.rg_lru[layer.rg_lru_idx](x, mask_copy)
                else:
                    x_temporal = model.local_attn[layer.attn_idx](x, mask_copy)

                x = model.rmsnorms[layer_idx](x + x_temporal)

                if model.titans is not None:
                    for t in range(L):
                        col = x[:, :, t, :]
                        col_mask = mask_copy[:, :, t]
                        x[:, :, t, :] = model.titans(col, col_mask)

                x = model.rmsnorms_mlp[layer_idx](x + model.mlps[layer_idx](x))

            logits = torch.zeros(B, N, L, alphabet_size, device=device)

            for b in range(B):
                logits_b = mask_head(x[b])
                logits[b, :, :logits_b.shape[1]] = logits_b

            loss = F.cross_entropy(
                logits[mask_positions].view(-1, alphabet_size),
                msa[mask_positions].view(-1),
                ignore_index=config.pad_idx,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            pbar.update(1)
            global_step += 1

    pbar.close()
    return model
