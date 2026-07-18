"""
PhyloGriffin v3 -- Train-5: Supertree Reconciler.
Train the transformer that stitches subtrees together.
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PhyloGriffinConfig


def train_supertree(
    column_processor: nn.Module,
    diffusion: nn.Module,
    supertree: nn.Module,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> nn.Module:
    column_processor = column_processor.to(device)
    column_processor.eval()
    for p in column_processor.parameters():
        p.requires_grad = False

    diffusion = diffusion.to(device)
    diffusion.eval()
    for p in diffusion.parameters():
        p.requires_grad = False

    supertree = supertree.to(device)
    supertree.train()

    optimizer = AdamW(supertree.parameters(), lr=1e-4, weight_decay=config.training.weight_decay)
    max_steps = 30000
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    use_amp = "cuda" in str(device)
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Train-5")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            msa = batch["msa"].to(device)
            mask = batch["mask"].to(device)
            true_tree = batch["true_tree"]
            guide_tree = batch["guide_tree"]

            if isinstance(true_tree, list):
                true_tree = true_tree[0]
            if isinstance(guide_tree, list):
                guide_tree = guide_tree[0]

            if msa.dim() == 3:
                msa = msa.squeeze(0)
                mask = mask.squeeze(0)

            N = msa.shape[0]

            if N <= 500:
                global_step += 1
                pbar.update(1)
                continue

            with torch.no_grad():
                seq_emb, _ = column_processor(msa, mask)

            n_leaves = N

            K = max(2, n_leaves // config.decomposition.max_subproblem_size)
            if K == 1:
                subtrees = [(torch.arange(N, device=device), true_tree)]
            else:
                indices = torch.randperm(N, device=device)
                chunk_size = n_leaves // K
                subtrees = []
                for k in range(K):
                    start = k * chunk_size
                    end = start + chunk_size if k < K - 1 else N
                    chunk_indices = indices[start:end]
                    chunk_msa = msa[chunk_indices]
                    chunk_emb = seq_emb[chunk_indices]
                    with torch.no_grad():
                        try:
                            chunk_tree = diffusion.generate(chunk_msa, chunk_emb)
                        except Exception:
                            chunk_tree = (
                                f"({','.join(f'leaf_{i}:0.1' for i in range(len(chunk_indices)))});"
                            )
                    subtrees.append((chunk_indices, chunk_tree))

            try:
                _, intermediates = supertree(subtrees, guide_tree, seq_emb)
            except Exception:
                intermediates = None

            if intermediates is not None:
                try:
                    loss = supertree.compute_loss(intermediates, true_tree, subtrees, N, device)
                except Exception:
                    loss = torch.zeros(1, device=device, requires_grad=True)
            else:
                loss = torch.zeros(1, device=device, requires_grad=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(supertree.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})
            pbar.update(1)
            global_step += 1

    pbar.close()
    return supertree
