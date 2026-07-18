"""
PhyloGriffin v3 -- Train-1: Masked Column Reconstruction.
Self-supervised pre-training of the Griffin column processor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PhyloGriffinConfig

if TYPE_CHECKING:
    from ..model.column_processor import ColumnProcessor


def train_column_reconstruction(
    model: ColumnProcessor,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> ColumnProcessor:
    model = model.to(device)
    model.train()

    alphabet_size = config.alphabet_size
    mask_token_idx = alphabet_size

    mask_head = nn.Linear(config.griffin.d_model, alphabet_size).to(device)

    params = list(model.parameters()) + list(mask_head.parameters())
    optimizer = AdamW(
        params, lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=config.training.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.training.max_steps - config.training.warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.training.warmup_steps],
    )

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
            replace_with_random = (torch.rand(B, N, L, device=device) >= 0.8) & (
                torch.rand(B, N, L, device=device) < 0.9
            )

            for b in range(B):
                mask_pos = mask_positions[b]
                masked_msa[b][mask_pos & replace_with_mask[b]] = mask_token_idx
                n_random = (mask_pos & replace_with_random[b]).sum().item()
                if n_random > 0:
                    random_tokens = torch.randint(0, alphabet_size, (n_random,), device=device)
                    masked_msa[b][mask_pos & replace_with_random[b]] = random_tokens

            hidden = model.forward_hidden(masked_msa, mask)

            logits = mask_head(hidden).view(B, N, L, alphabet_size)

            loss = F.cross_entropy(
                logits[mask_positions].view(-1, alphabet_size),
                msa[mask_positions].view(-1),
                ignore_index=config.pad_idx,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_postfix(
                {"loss": f"{running_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            )
            pbar.update(1)
            global_step += 1

    pbar.close()
    return model
