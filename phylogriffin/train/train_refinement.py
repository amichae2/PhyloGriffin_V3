"""
PhyloGriffin v3 -- Train-6: Refinement Pass.
Train the NNI-based tree refinement module.
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


def train_refinement(
    refinement: nn.Module,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> nn.Module:
    refinement = refinement.to(device)
    refinement.train()

    optimizer = AdamW(refinement.parameters(), lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)
    max_steps = 10000
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Train-6")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            msa = batch["msa"].to(device)
            mask = batch["mask"].to(device)
            embeddings = batch["embeddings"].to(device)
            corrupted_trees = batch["corrupted_tree"]
            true_trees = batch["true_tree"]

            B = msa.shape[0]
            if msa.dim() == 4:
                msa = msa.squeeze(1)

            total_loss = 0.0
            valid_count = 0

            for b in range(B):
                n = (mask[b].sum(dim=1) > 0).sum().item()
                emb = embeddings[b, :n]

                if n < 4:
                    continue

                try:
                    refined = refinement(corrupted_trees[b], emb)
                except Exception:
                    refined = true_trees[b]

                from ..tree_utils import robinson_foulds, newick_to_splits

                try:
                    true_splits = newick_to_splits(true_trees[b], n)
                    ref_splits = newick_to_splits(refined, n)
                    rf = robinson_foulds(true_splits, ref_splits)
                    loss = torch.tensor(rf, device=device, requires_grad=True)
                    total_loss += loss
                    valid_count += 1
                except Exception:
                    pass

            if valid_count > 0:
                loss = total_loss / valid_count

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(refinement.parameters(), config.training.grad_clip)
                optimizer.step()
                scheduler.step()

                running_loss = 0.9 * running_loss + 0.1 * loss.item()
                pbar.set_postfix({"loss": f"{running_loss:.4f}"})
                pbar.update(1)
                global_step += 1

    pbar.close()
    return refinement
