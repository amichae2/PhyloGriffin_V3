"""
PhyloGriffin v3 -- Train-3: Graph Predictor.
Train the edge predictor for phylogenetic adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PhyloGriffinConfig


def train_graph_predictor(
    column_processor: nn.Module,
    graph_predictor: nn.Module,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> nn.Module:
    column_processor = column_processor.to(device)
    column_processor.eval()
    for p in column_processor.parameters():
        p.requires_grad = False

    graph_predictor = graph_predictor.to(device)
    graph_predictor.train()

    optimizer = AdamW(
        graph_predictor.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    max_steps = 20000
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    use_amp = "cuda" in str(device)
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Train-3")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            msa = batch["msa"].to(device)
            mask = batch["mask"].to(device)
            tree_newick_list = batch["tree_newick"]

            B, batch_N, batch_L = msa.shape

            with torch.no_grad():
                seq_emb, _ = column_processor(msa, mask)

            total_loss = 0.0
            for b in range(B):
                n = (mask[b].sum(dim=1) > 0).sum().item()
                emb = seq_emb[b, :n]

                from ..tree_utils import (
                    get_leaf_order,
                    parse_newick,
                    patristic_distances,
                )

                parse_newick(tree_newick_list[b])
                get_leaf_order(tree_newick_list[b])

                if n < 3:
                    continue

                dist = patristic_distances(tree_newick_list[b], n)
                dist_tensor = torch.from_numpy(dist).to(device)

                positive_pairs = []
                negative_pairs = []

                dist_tensor.diagonal(offset=0).fill_(float("inf"))
                flat = dist_tensor.flatten()
                sorted_vals, sorted_idx = torch.sort(flat)

                n_pairs_needed = min(100, n * (n - 1) // 2)
                pos_candidates = sorted_idx[:n_pairs_needed]
                neg_candidates = sorted_idx[-n_pairs_needed // 10 :]

                for idx in pos_candidates[:100].tolist():
                    i = idx // n
                    j = idx % n
                    if i != j and dist_tensor[i, j] > 0:
                        positive_pairs.append((i, j))

                for idx in neg_candidates[:500].tolist():
                    i = idx // n
                    j = idx % n
                    if i != j:
                        negative_pairs.append((i, j))

                if not positive_pairs or not negative_pairs:
                    continue

                all_i = []
                all_j = []
                all_labels = []

                for i, j in positive_pairs:
                    all_i.append(i)
                    all_j.append(j)
                    all_labels.append(1.0)

                for i, j in negative_pairs:
                    all_i.append(i)
                    all_j.append(j)
                    all_labels.append(0.0)

                emb_i = emb[torch.tensor(all_i, device=device)]
                emb_j = emb[torch.tensor(all_j, device=device)]
                labels = torch.tensor(all_labels, device=device)

                probs = graph_predictor.forward_batch(emb_i, emb_j)

                pos_count = labels.sum()
                neg_count = len(labels) - pos_count
                pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)], device=device)

                loss = F.binary_cross_entropy(probs, labels, pos_weight=pos_weight)
                total_loss += loss

            if isinstance(total_loss, torch.Tensor) and total_loss > 0:
                total_loss = total_loss / B

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    graph_predictor.parameters(), config.training.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss = 0.9 * running_loss + 0.1 * total_loss.item()
                pbar.set_postfix({"loss": f"{running_loss:.4f}"})
                pbar.update(1)
                global_step += 1

    pbar.close()
    return graph_predictor
