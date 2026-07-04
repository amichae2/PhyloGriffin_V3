"""
PhyloGriffin v3 -- Train-4: Diffusion Denoiser.
Train the denoising diffusion model for tree generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional

from ..config import PhyloGriffinConfig


def train_diffusion(
    column_processor: nn.Module,
    diffusion: nn.Module,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> nn.Module:
    column_processor = column_processor.to(device)
    column_processor.eval()
    for p in column_processor.parameters():
        p.requires_grad = False

    diffusion = diffusion.to(device)
    diffusion.train()

    optimizer = AdamW(diffusion.parameters(), lr=2e-4,
                      weight_decay=config.training.weight_decay)
    max_steps = 50000
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Train-4")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            sub_msa = batch["sub_msa"].to(device)
            sub_mask = batch["sub_mask"].to(device)
            true_tree = batch["true_tree"]

            if isinstance(true_tree, list):
                true_tree = true_tree[0]

            M = sub_msa.shape[0]

            if sub_msa.dim() == 2:
                sub_msa = sub_msa.unsqueeze(0)
                sub_mask = sub_mask.unsqueeze(0)

            with torch.no_grad():
                seq_emb, _ = column_processor(sub_msa, sub_mask)

            seq_emb = seq_emb.squeeze(0) if seq_emb.dim() == 3 else seq_emb

            from ..tree_utils import newick_to_splits, get_leaf_order

            n_splits_max = config.diffusion.n_splits_max
            splits = newick_to_splits(true_tree, M)

            S_0 = torch.zeros(M, n_splits_max, device=device)
            b_0 = torch.zeros(n_splits_max, device=device)
            p_0 = torch.zeros(M, device=device)

            for j, (mask, blen) in enumerate(splits[:n_splits_max]):
                S_0[:, j] = torch.from_numpy(mask.astype(float) * 2 - 1).float().to(device)
                b_0[j] = min(max(blen, config.diffusion.branch_length_min),
                            config.diffusion.branch_length_max)

            leaf_names = get_leaf_order(true_tree)
            for i in range(M):
                p_0[i] = 0.1

            t = torch.randint(1, diffusion.n_steps + 1, (1,), device=device)
            alpha_bar_t = diffusion.alphas_cumprod[t]

            eps_S = torch.randn_like(S_0)
            eps_b = torch.randn_like(b_0)
            eps_p = torch.randn_like(p_0)

            S_t = torch.sqrt(alpha_bar_t) * S_0 + torch.sqrt(1 - alpha_bar_t) * eps_S
            b_t = torch.sqrt(alpha_bar_t) * b_0 + torch.sqrt(1 - alpha_bar_t) * eps_b
            p_t = torch.sqrt(alpha_bar_t) * p_0 + torch.sqrt(1 - alpha_bar_t) * eps_p

            t_emb = diffusion._time_embedding(t).squeeze(0)

            hat_eps_S, hat_eps_b, hat_eps_p = diffusion.denoiser(
                S_t, b_t, p_t, t_emb, seq_emb
            )

            loss_S = F.mse_loss(hat_eps_S[:, :len(splits)], eps_S[:, :len(splits)])
            loss_b = F.mse_loss(hat_eps_b[:len(splits)], eps_b[:len(splits)])
            loss_p = F.mse_loss(hat_eps_p, eps_p)

            loss = loss_S + loss_b + loss_p

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), config.training.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})
            pbar.update(1)
            global_step += 1

    pbar.close()
    return diffusion
