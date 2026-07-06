"""
PhyloGriffin v3 -- Train-2: Contrastive Phylogenetic Embedding.
Fine-tune the column processor + Titans memory for phylogenetic distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from typing import Optional

from ..config import PhyloGriffinConfig


def train_contrastive(
    model: "ColumnProcessor",
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> "ColumnProcessor":
    model = model.to(device)
    model.train()

    temperature = 0.1
    triplet_margin = 0.5

    n_freeze = min(6, len(model.layers))
    for i, layer in enumerate(model.layers):
        requires_grad = i >= n_freeze
        for p in layer.parameters():
            p.requires_grad = requires_grad

    if model.titans is not None:
        for p in model.titans.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=1e-4, weight_decay=config.training.weight_decay)

    warmup_steps = 500
    max_steps = 50000
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Train-2")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            msa = batch["msa"].to(device)
            mask = batch["mask"].to(device)
            pairwise_dist = batch["pairwise_distances"].to(device)

            B, batch_N, batch_L = msa.shape

            seq_emb, _ = model(msa, mask)

            total_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                n = (mask[b].sum(dim=1) > 0).sum().item()
                emb = seq_emb[b, :n]
                dist = pairwise_dist[b, :n, :n]

                emb_norm = F.normalize(emb, dim=-1)

                valid_mask = dist > 0

                if valid_mask.sum() < 4:
                    continue

                flat_dist = dist[valid_mask]
                if flat_dist.numel() == 0:
                    continue

                p25 = torch.quantile(flat_dist, 0.25)
                p75 = torch.quantile(flat_dist, 0.75)

                pos_mask = (dist <= p25) & (dist > 0)
                neg_mask = dist >= p75

                pos_indices = torch.where(pos_mask)
                neg_indices = torch.where(neg_mask)

                if len(pos_indices[0]) == 0:
                    continue

                n_pos = min(len(pos_indices[0]), 500)

                sim_matrix = torch.mm(emb_norm, emb_norm.t()) / temperature
                exp_sim = torch.exp(sim_matrix)
                sum_exp = exp_sim.sum(dim=1)

                pi = pos_indices[0][:n_pos]
                pj = pos_indices[1][:n_pos]

                pos_sims = sim_matrix[pi, pj]
                pos_exp_sums = sum_exp[pi]
                contrast_loss = -(pos_sims - torch.log(pos_exp_sums + 1e-8)).mean()

                triplet_loss = torch.tensor(0.0, device=device)
                if len(neg_indices[0]) > 0:
                    n_triplets = min(100, len(pos_indices[0]), len(neg_indices[0]))
                    pi_t = pos_indices[0][:n_triplets]
                    pj_t = pos_indices[1][:n_triplets]
                    ni_t = neg_indices[0][torch.arange(n_triplets) % len(neg_indices[0])]
                    nj_t = neg_indices[1][torch.arange(n_triplets) % len(neg_indices[0])]

                    d_pos = torch.norm(emb[pi_t] - emb[pj_t], p=2, dim=-1)
                    d_neg = torch.norm(emb[ni_t] - emb[nj_t], p=2, dim=-1)
                    triplet_loss = F.relu(triplet_margin + d_pos - d_neg).mean()

                total_loss += contrast_loss + 0.5 * triplet_loss

            total_loss = total_loss / B

            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * total_loss.item()
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            pbar.update(1)
            global_step += 1

    pbar.close()

    for p in model.parameters():
        p.requires_grad = True

    return model
