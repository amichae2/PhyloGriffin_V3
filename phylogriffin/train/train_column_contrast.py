"""
PhyloGriffin v3 -- Train-2: Contrastive Phylogenetic Embedding.
Fine-tune the column processor + Titans memory for phylogenetic distance.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PhyloGriffinConfig

if TYPE_CHECKING:
    from ..model.column_processor import ColumnProcessor

logger = logging.getLogger("phylogriffin.train.contrast")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

WARMUP_LOG_STEPS = 5
TIMING_LOG_INTERVAL = 50
GPU_LOG_INTERVAL = 100


def train_contrastive(
    model: ColumnProcessor,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> ColumnProcessor:
    train_start_time = time.time()

    logger.info("=" * 80)
    logger.info("STAGE 2: Contrastive Phylogenetic Embedding Training")
    logger.info("=" * 80)

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

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_trainable
    logger.info(
        f"Model parameters: {n_total:,} total, {n_trainable:,} trainable, {n_frozen:,} frozen"
    )
    logger.info(
        f"Trainable layers: {sum(1 for p in model.parameters() if p.requires_grad)} param tensors"
    )
    logger.info(f"Temperature: {temperature}, Triplet margin: {triplet_margin}")
    logger.info(f"n_freeze: {n_freeze} / {len(model.layers)} layers frozen")
    logger.info(f"Device: {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=1e-4, weight_decay=config.training.weight_decay)

    warmup_steps = 500
    max_steps = 50000
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    if hasattr(dataloader, "num_workers"):
        logger.info(f"DataLoader workers: {dataloader.num_workers}")
    if hasattr(dataloader, "batch_size"):
        logger.info(f"DataLoader batch_size: {dataloader.batch_size}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    logger.info("=" * 80)

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Train-2")

    step_timings: list[float] = []
    timing_data_times: list[float] = []
    timing_forward_times: list[float] = []
    timing_backward_times: list[float] = []
    timing_loss_times: list[float] = []

    t_first_fetch = time.time()

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            t_data_start = time.time()

            msa = batch["msa"].to(device)
            mask = batch["mask"].to(device)
            pairwise_dist = batch["pairwise_distances"].to(device)

            data_time = time.time() - t_data_start

            if global_step < WARMUP_LOG_STEPS:
                logger.info(
                    f"  Batch {global_step} received after "
                    f"{time.time() - t_first_fetch:.3f}s from first fetch"
                )
                t_first_fetch = time.time()
                logger.info(f"  Step {global_step}: data loading took {data_time:.3f}s")
                logger.info(f"    msa shape: {msa.shape}, mask shape: {mask.shape}")
                logger.info(f"    pairwise_dist shape: {pairwise_dist.shape}")
                logger.info(f"    msa on device: {msa.device}")

            if global_step == 0:
                logger.info("Dataset info (first batch):")
                logger.info(f"  msa shape: {msa.shape}")
                logger.info(f"  msa dtype: {msa.dtype}")
                logger.info(f"  mask shape: {mask.shape}")
                logger.info(f"  pairwise_distances shape: {pairwise_dist.shape}")
                logger.info(f"  N sequences per batch: {msa.shape[1]}")
                logger.info(f"  L sites per sequence: {msa.shape[2]}")

            B, batch_N, batch_L = msa.shape

            t_forward_start = time.time()
            seq_emb, _ = model(msa, mask)
            forward_time = time.time() - t_forward_start

            if global_step < WARMUP_LOG_STEPS:
                logger.info(f"  Step {global_step}: forward pass took {forward_time:.3f}s")
                logger.info(f"    seq_emb shape: {seq_emb.shape}")

            t_loss_start = time.time()

            total_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                if global_step < WARMUP_LOG_STEPS:
                    t_b_start = time.time()

                n = (mask[b].sum(dim=1) > 0).sum().item()
                emb = seq_emb[b, :n]
                dist = pairwise_dist[b, :n, :n]

                emb_norm = F.normalize(emb, dim=-1)

                valid_mask = dist > 0

                if valid_mask.sum() < 4:
                    if global_step < 20:
                        logger.debug(
                            f"Step {global_step}, batch {b}: skipped batch "
                            f"(only {valid_mask.sum().item()} valid pairs)"
                        )
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

                if global_step < WARMUP_LOG_STEPS:
                    t_b_end = time.time()
                    logger.info(
                        f"    batch element {b}: n={n}, "
                        f"contrast_loss={contrast_loss.item():.4f}, "
                        f"triplet_loss={triplet_loss.item():.4f}, "
                        f"time={t_b_end - t_b_start:.3f}s"
                    )

            total_loss = total_loss / B

            loss_time = time.time() - t_loss_start

            if global_step < WARMUP_LOG_STEPS:
                logger.info(f"  Step {global_step}: loss computation took {loss_time:.3f}s")
                logger.info(f"    total_loss: {total_loss.item():.4f}")

            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() == 0.0:
                reason = (
                    "NaN"
                    if torch.isnan(total_loss)
                    else "Inf"
                    if torch.isinf(total_loss)
                    else "zero"
                )
                logger.warning(f"Step {global_step}: skipping due to {reason} loss")
                optimizer.zero_grad()
                continue

            t_backward_start = time.time()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip)
            optimizer.step()
            scheduler.step()
            backward_time = time.time() - t_backward_start

            if global_step < WARMUP_LOG_STEPS:
                logger.info(f"  Step {global_step}: backward took {backward_time:.3f}s")
                step_time = data_time + forward_time + loss_time + backward_time
                step_timings.append(step_time)
                logger.info(f"  Step {global_step}: total step time = {step_time:.3f}s")
                logger.info(
                    f"    breakdown: data={data_time:.3f}s, "
                    f"forward={forward_time:.3f}s, "
                    f"loss={loss_time:.3f}s, "
                    f"backward={backward_time:.3f}s"
                )

            timing_data_times.append(data_time)
            timing_forward_times.append(forward_time)
            timing_backward_times.append(backward_time)
            timing_loss_times.append(loss_time)

            if global_step == WARMUP_LOG_STEPS - 1:
                avg_step = sum(step_timings) / len(step_timings)
                avg_data = sum(timing_data_times[-WARMUP_LOG_STEPS:]) / WARMUP_LOG_STEPS
                logger.info(f"  Warmup complete. Average step time: {avg_step:.3f}s")
                logger.info(
                    f"  Estimated time for {max_steps} steps: "
                    f"{avg_step * max_steps / 3600:.1f} hours"
                )
                if avg_data > 0.5 * avg_step:
                    pct = avg_data / avg_step * 100
                    logger.warning(
                        f"  WARNING: Data loading is {pct:.0f}% of step time. "
                        "Consider increasing num_workers or pre-caching data."
                    )

            running_loss = 0.9 * running_loss + 0.1 * total_loss.item()
            pbar.set_postfix(
                {"loss": f"{running_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            )
            pbar.update(1)
            global_step += 1

            if 0 < global_step % TIMING_LOG_INTERVAL == 0:
                n_timing = min(TIMING_LOG_INTERVAL, len(timing_data_times))
                recent_data = timing_data_times[-n_timing:]
                recent_fwd = timing_forward_times[-n_timing:]
                recent_bwd = timing_backward_times[-n_timing:]
                recent_loss_t = timing_loss_times[-n_timing:]
                avg_data_t = sum(recent_data) / n_timing
                avg_fwd_t = sum(recent_fwd) / n_timing
                avg_bwd_t = sum(recent_bwd) / n_timing
                avg_loss_t = sum(recent_loss_t) / n_timing
                avg_total = avg_data_t + avg_fwd_t + avg_bwd_t + avg_loss_t
                eta_h = avg_total * (max_steps - global_step) / 3600
                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"avg_total={avg_total:.3f}s "
                    f"(data={avg_data_t:.3f}s [{avg_data_t / avg_total * 100:.0f}%], "
                    f"fwd={avg_fwd_t:.3f}s [{avg_fwd_t / avg_total * 100:.0f}%], "
                    f"loss={avg_loss_t:.3f}s [{avg_loss_t / avg_total * 100:.0f}%], "
                    f"bwd={avg_bwd_t:.3f}s [{avg_bwd_t / avg_total * 100:.0f}%]) | "
                    f"loss={running_loss:.4f} | "
                    f"ETA={eta_h:.1f}h"
                )

            if global_step % GPU_LOG_INTERVAL == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"  GPU memory: allocated={allocated:.2f}GB, "
                    f"reserved={reserved:.2f}GB, "
                    f"peak={max_allocated:.2f}GB"
                )

    pbar.close()

    total_train_time = time.time() - train_start_time
    logger.info("=" * 80)
    logger.info(f"Stage 2 training complete in {total_train_time / 3600:.2f} hours")
    logger.info(f"Final loss: {running_loss:.4f}")
    logger.info(f"Steps completed: {global_step}/{max_steps}")
    if timing_data_times:
        avg_data_pct = sum(timing_data_times) / len(timing_data_times)
        avg_total_s = avg_data_pct + (
            sum(timing_forward_times) / len(timing_forward_times) if timing_forward_times else 0
        )
        if avg_total_s > 0:
            data_frac = sum(timing_data_times) / len(timing_data_times) / avg_total_s * 100
            logger.info(f"Average data loading time per step: {data_frac:.1f}% of total step time")
            if data_frac > 50:
                logger.warning("Data loading was the primary bottleneck. Recommendations:")
                logger.warning("  1. Increase num_workers in the DataLoader (try 4-8)")
                logger.warning("  2. Pre-compute and cache patristic_distances to disk")
                logger.warning("  3. Reduce n_leaves_range to generate smaller trees")
                logger.warning("  4. Use a persistent_workers=True in the DataLoader")
    logger.info("=" * 80)

    for p in model.parameters():
        p.requires_grad = True

    return model
