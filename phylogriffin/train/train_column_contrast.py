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

    for layer in model.griffin_layers[:6]:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in model.griffin_layers[6:]:
        for p in layer.parameters():
            p.requires_grad = True

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

            total_loss = 0.0
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

                n_pos = len(pos_indices[0])
                pos_pairs = list(zip(pos_indices[0][:min(n_pos, 500)].tolist(),
                                      pos_indices[1][:min(n_pos, 500)].tolist()))

                sim_matrix = torch.mm(emb_norm, emb_norm.t()) / temperature

                contrast_loss = 0.0
                for i, j in pos_pairs:
                    exp_sim = torch.exp(sim_matrix[i])
                    contrast_loss += -torch.log(
                        torch.exp(sim_matrix[i, j]) / (exp_sim.sum() + 1e-8)
                    )
                if len(pos_pairs) > 0:
                    contrast_loss = contrast_loss / len(pos_pairs)

                triplet_loss = 0.0
                if len(neg_indices[0]) > 0 and len(pos_indices[0]) > 0:
                    n_triplets = min(100, len(pos_indices[0]), len(neg_indices[0]))
                    for _ in range(n_triplets):
                        pi = pos_indices[0][_]
                        pj = pos_indices[1][_]
                        ni = neg_indices[0][_ % len(neg_indices[0])]
                        nj = neg_indices[1][_ % len(neg_indices[0])]

                        d_pos = torch.norm(emb[pi] - emb[pj], p=2)
                        d_neg = torch.norm(emb[ni] - emb[nj], p=2)
                        triplet_loss += F.relu(triplet_margin + d_pos - d_neg)
                    triplet_loss = triplet_loss / n_triplets

                total_loss += contrast_loss + 0.5 * triplet_loss

            total_loss = total_loss / B

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
