"""
PhyloGriffin v3 -- Stage D: Per-Subproblem Diffusion Tree Generator.
Generates a phylogenetic tree via denoising diffusion modelled on continuous split representation.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PhyloGriffinConfig
from ..tree_utils import splits_to_newick


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight / rms


def cosine_beta_schedule(n_steps, s=0.008):
    steps = n_steps + 1
    x = np.linspace(0, n_steps, steps)
    alphas_cumprod = np.cos(((x / n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(torch.tensor(betas, dtype=torch.float32), max=0.999)


def linear_beta_schedule(n_steps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps)


class DenoiserGNN(nn.Module):
    def __init__(self, config: PhyloGriffinConfig, d_time: int):
        super().__init__()
        self.d_model = config.griffin.d_model
        self.d_hidden = config.diffusion.denoiser_hidden
        self.d_time = d_time
        self.n_layers = config.diffusion.denoiser_layers

        self.leaf_init_proj = nn.Linear(self.d_model + 1 + self.d_time, self.d_hidden)
        self.split_init_proj = nn.Linear(1 + self.d_hidden + self.d_time, self.d_hidden)

        self.msg_l2s = nn.ModuleList(
            [nn.Linear(2 * self.d_hidden + 1, self.d_hidden) for _ in range(self.n_layers)]
        )
        self.msg_s2l = nn.ModuleList(
            [nn.Linear(2 * self.d_hidden + 1, self.d_hidden) for _ in range(self.n_layers)]
        )

        self.norm_split_1 = nn.ModuleList([RMSNorm(self.d_hidden) for _ in range(self.n_layers)])
        self.norm_split_2 = nn.ModuleList([RMSNorm(self.d_hidden) for _ in range(self.n_layers)])
        self.norm_leaf_1 = nn.ModuleList([RMSNorm(self.d_hidden) for _ in range(self.n_layers)])
        self.norm_leaf_2 = nn.ModuleList([RMSNorm(self.d_hidden) for _ in range(self.n_layers)])

        self.mlp_split = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_hidden, self.d_hidden * 4),
                    nn.ReLU(),
                    nn.Linear(self.d_hidden * 4, self.d_hidden),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.mlp_leaf = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_hidden, self.d_hidden * 4),
                    nn.ReLU(),
                    nn.Linear(self.d_hidden * 4, self.d_hidden),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.eps_splits_net = nn.Sequential(
            nn.Linear(2 * self.d_hidden + 1, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, 1),
        )

        self.eps_branch_net = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, 1),
        )

        self.eps_pendant_net = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, 1),
        )

    def forward(self, splits, branch_lengths, pendant_lengths, t_emb, seq_embeddings):
        M = seq_embeddings.shape[0]
        n_splits_max = splits.shape[1]

        t_leaf = t_emb.unsqueeze(0).expand(M, -1)
        leaf_feat = torch.cat(
            [
                seq_embeddings,
                pendant_lengths.unsqueeze(-1),
                t_leaf,
            ],
            dim=-1,
        )
        leaf_feat = self.leaf_init_proj(leaf_feat)

        weights = F.softmax(splits.abs(), dim=0)
        leaf_aggregate = torch.matmul(weights.T, leaf_feat)

        t_split = t_emb.unsqueeze(0).expand(n_splits_max, -1)
        split_feat = torch.cat(
            [
                branch_lengths.unsqueeze(-1),
                leaf_aggregate,
                t_split,
            ],
            dim=-1,
        )
        split_feat = self.split_init_proj(split_feat)

        for layer_idx in range(self.n_layers):
            split_msg = self._leaf_to_split_message(leaf_feat, split_feat, splits, layer_idx)
            split_feat = self.norm_split_1[layer_idx](split_feat + split_msg)
            split_feat = self.norm_split_2[layer_idx](
                split_feat + self.mlp_split[layer_idx](split_feat)
            )

            leaf_msg = self._split_to_leaf_message(leaf_feat, split_feat, splits, layer_idx)
            leaf_feat = self.norm_leaf_1[layer_idx](leaf_feat + leaf_msg)
            leaf_feat = self.norm_leaf_2[layer_idx](leaf_feat + self.mlp_leaf[layer_idx](leaf_feat))

        eps_splits = self._compute_eps_splits(leaf_feat, split_feat, splits)
        eps_branch = self.eps_branch_net(split_feat).squeeze(-1)
        eps_pendant = self.eps_pendant_net(leaf_feat).squeeze(-1)

        return eps_splits, eps_branch, eps_pendant

    def _leaf_to_split_message(self, leaf_feat, split_feat, splits, layer_idx):
        M, d_hidden = leaf_feat.shape
        S = split_feat.shape[0]
        split_msg = torch.zeros(S, d_hidden, device=leaf_feat.device)

        chunk_size = 128
        for j_start in range(0, S, chunk_size):
            j_end = min(j_start + chunk_size, S)
            chunk_S = j_end - j_start

            leaf_exp = leaf_feat.unsqueeze(1).expand(M, chunk_S, d_hidden)
            split_exp = split_feat[j_start:j_end].unsqueeze(0).expand(M, -1, d_hidden)
            splits_chunk = splits[:, j_start:j_end]

            pair_feat = torch.cat(
                [
                    leaf_exp,
                    split_exp,
                    splits_chunk.unsqueeze(-1),
                ],
                dim=-1,
            )

            msg = self.msg_l2s[layer_idx](pair_feat)

            weights = splits_chunk.abs()
            denom = weights.sum(dim=0).unsqueeze(-1).clamp(min=1e-8)
            split_msg[j_start:j_end] = (weights.unsqueeze(-1) * msg).sum(dim=0) / denom

        return split_msg

    def _split_to_leaf_message(self, leaf_feat, split_feat, splits, layer_idx):
        M, d_hidden = leaf_feat.shape
        S = split_feat.shape[0]
        leaf_msg = torch.zeros(M, d_hidden, device=leaf_feat.device)

        chunk_size = 128
        for j_start in range(0, S, chunk_size):
            j_end = min(j_start + chunk_size, S)
            chunk_S = j_end - j_start

            leaf_exp = leaf_feat.unsqueeze(1).expand(M, chunk_S, d_hidden)
            split_exp = split_feat[j_start:j_end].unsqueeze(0).expand(M, -1, d_hidden)
            splits_chunk = splits[:, j_start:j_end]

            pair_feat = torch.cat(
                [
                    leaf_exp,
                    split_exp,
                    splits_chunk.unsqueeze(-1),
                ],
                dim=-1,
            )

            msg = self.msg_s2l[layer_idx](pair_feat)

            weights = splits_chunk.abs()
            denom = weights.sum(dim=1).unsqueeze(-1).clamp(min=1e-8)
            leaf_msg = leaf_msg + (weights.unsqueeze(-1) * msg).sum(dim=1) / denom

        return leaf_msg

    def _compute_eps_splits(self, leaf_feat, split_feat, splits):
        M, d_hidden = leaf_feat.shape
        S = split_feat.shape[0]
        eps_splits = torch.zeros(M, S, device=leaf_feat.device)

        chunk_size = 128
        for j_start in range(0, S, chunk_size):
            j_end = min(j_start + chunk_size, S)
            chunk_S = j_end - j_start

            leaf_exp = leaf_feat.unsqueeze(1).expand(M, chunk_S, d_hidden)
            split_exp = split_feat[j_start:j_end].unsqueeze(0).expand(M, -1, d_hidden)
            splits_chunk = splits[:, j_start:j_end]

            pair_feat = torch.cat(
                [
                    leaf_exp,
                    split_exp,
                    splits_chunk.unsqueeze(-1),
                ],
                dim=-1,
            )

            out = self.eps_splits_net(pair_feat)
            eps_splits[:, j_start:j_end] = out.squeeze(-1)

        return eps_splits


class DiffusionTreeGenerator(nn.Module):
    def __init__(self, config: PhyloGriffinConfig):
        super().__init__()
        self.config = config
        self.n_steps = config.diffusion.n_diffusion_steps
        self.n_splits_max = config.diffusion.n_splits_max
        self.d_time = config.diffusion.d_time

        self.denoiser = DenoiserGNN(config, self.d_time)

        if config.diffusion.noise_schedule == "cosine":
            betas = cosine_beta_schedule(self.n_steps)
        elif config.diffusion.noise_schedule == "linear":
            betas = linear_beta_schedule(self.n_steps)
        else:
            raise ValueError(f"Unknown noise schedule: {config.diffusion.noise_schedule}")

        betas = torch.cat([torch.zeros(1), betas], dim=0)
        alphas = torch.cat([torch.ones(1), 1.0 - betas[1:]], dim=0)
        alphas_cumprod = torch.cat([torch.ones(1), torch.cumprod(alphas[1:], dim=0)], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _time_embedding(self, t):
        half = self.d_time // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        freqs = freqs.to(t.device)
        args = t.float() * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb.squeeze(0)

    def forward(
        self,
        sub_msa,
        sub_embeddings,
        true_splits=None,
        true_branch_lengths=None,
        true_pendant_lengths=None,
        t=None,
    ):
        M = sub_embeddings.shape[0]
        device = sub_embeddings.device

        if t is None:
            t = torch.randint(1, self.n_steps + 1, (1,), device=device)

        if true_splits is not None:
            S_0 = true_splits
            b_0 = true_branch_lengths
            p_0 = true_pendant_lengths
        else:
            S_0 = torch.randn(M, self.n_splits_max, device=device) * 0.1
            b_0 = torch.randn(self.n_splits_max, device=device) * 0.1
            p_0 = torch.randn(M, device=device) * 0.1

        S_t, b_t, p_t, eps_S, eps_b, eps_p = self._add_noise(S_0, b_0, p_0, t)

        t_emb = self._time_embedding(t)

        hat_eps_S, hat_eps_b, hat_eps_p = self.denoiser(S_t, b_t, p_t, t_emb, sub_embeddings)

        return {
            "eps_S": eps_S,
            "eps_b": eps_b,
            "eps_p": eps_p,
            "hat_eps_S": hat_eps_S,
            "hat_eps_b": hat_eps_b,
            "hat_eps_p": hat_eps_p,
        }

    def _add_noise(self, S_0, b_0, p_0, t):
        alpha_bar_t = self.alphas_cumprod[t]
        eps_S = torch.randn_like(S_0)
        eps_b = torch.randn_like(b_0)
        eps_p = torch.randn_like(p_0)

        sqrt_alpha = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)

        S_t = sqrt_alpha * S_0 + sqrt_one_minus_alpha * eps_S
        b_t = sqrt_alpha * b_0 + sqrt_one_minus_alpha * eps_b
        p_t = sqrt_alpha * p_0 + sqrt_one_minus_alpha * eps_p

        return S_t, b_t, p_t, eps_S, eps_b, eps_p

    def _denoise(
        self, noisy_splits, noisy_branch_lengths, noisy_pendant_lengths, t, sub_embeddings
    ):
        t_emb = self._time_embedding(t)
        return self.denoiser(
            noisy_splits, noisy_branch_lengths, noisy_pendant_lengths, t_emb, sub_embeddings
        )

    @torch.no_grad()
    def generate(self, sub_msa, sub_embeddings, leaf_names=None):
        M = sub_embeddings.shape[0]
        device = sub_embeddings.device
        n_splits_max = self.n_splits_max

        if leaf_names is None:
            leaf_names = [str(i) for i in range(M)]

        splits = torch.randn(M, n_splits_max, device=device)
        branch_lengths = torch.randn(n_splits_max, device=device)
        pendant_lengths = torch.randn(M, device=device)

        for t in reversed(range(1, self.n_steps + 1)):
            t_tensor = torch.tensor([t], device=device)
            t_emb = self._time_embedding(t_tensor)

            eps_s, eps_b, eps_p = self.denoiser(
                splits, branch_lengths, pendant_lengths, t_emb, sub_embeddings
            )

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 1:
                noise_s = torch.randn_like(splits)
                noise_b = torch.randn_like(branch_lengths)
                noise_p = torch.randn_like(pendant_lengths)
            else:
                noise_s = torch.zeros_like(splits)
                noise_b = torch.zeros_like(branch_lengths)
                noise_p = torch.zeros_like(pendant_lengths)

            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

            splits = (1.0 / sqrt_alpha) * (
                splits - (beta_t / sqrt_one_minus_alpha_bar) * eps_s
            ) + torch.sqrt(beta_t) * noise_s

            branch_lengths = (1.0 / sqrt_alpha) * (
                branch_lengths - (beta_t / sqrt_one_minus_alpha_bar) * eps_b
            ) + torch.sqrt(beta_t) * noise_b

            pendant_lengths = (1.0 / sqrt_alpha) * (
                pendant_lengths - (beta_t / sqrt_one_minus_alpha_bar) * eps_p
            ) + torch.sqrt(beta_t) * noise_p

        newick = self._discretize(splits, branch_lengths, pendant_lengths, leaf_names)
        if newick is None:
            newick = self._simple_nj(sub_embeddings, leaf_names)
        return newick

    def _discretize(self, splits_continuous, branch_lengths, pendant_lengths, leaf_names):
        M = splits_continuous.shape[0]
        threshold = 0.05

        splits_np = splits_continuous.detach().cpu().numpy()
        branch_np = branch_lengths.detach().cpu().numpy()

        bipartitions = []
        n_splits = splits_np.shape[1]

        for j in range(n_splits):
            left_set = set(np.where(splits_np[:, j] > threshold)[0].tolist())
            right_set = set(np.where(splits_np[:, j] < -threshold)[0].tolist())

            if len(left_set) >= 2 and len(right_set) >= 2:
                mask = np.zeros(M, dtype=bool)
                mask[list(left_set)] = True
                bipartitions.append((mask, abs(float(branch_np[j]))))
            elif len(left_set) >= 2 and len(right_set) < 2:
                mask = np.zeros(M, dtype=bool)
                mask[list(left_set)] = True
                bipartitions.append((mask, abs(float(branch_np[j]))))
            elif len(right_set) >= 2:
                mask = np.zeros(M, dtype=bool)
                mask[list(right_set)] = True
                bipartitions.append((mask, abs(float(branch_np[j]))))

        bipartitions.sort(key=lambda x: x[1], reverse=True)

        accepted = []
        max_splits = M - 3

        for mask, bl in bipartitions:
            if len(accepted) >= max_splits:
                break

            s1 = set(np.where(mask)[0].tolist())
            compatible = True
            for acc_mask, _ in accepted:
                s2 = set(np.where(acc_mask)[0].tolist())
                if not (s1.issubset(s2) or s2.issubset(s1) or len(s1.intersection(s2)) == 0):
                    compatible = False
                    break

            if compatible:
                accepted.append((mask, bl))

        if len(accepted) < max_splits and len(accepted) < M - 3:
            return None

        if len(accepted) > 0:
            try:
                return splits_to_newick(accepted, leaf_names)
            except ValueError:
                pass

        return None

    def _simple_nj(self, embeddings, leaf_names):
        M = embeddings.shape[0]

        raw_dist = torch.cdist(embeddings, embeddings, p=2).cpu().numpy()

        max_nodes = 2 * M
        D = np.full((max_nodes, max_nodes), np.inf)
        D[:M, :M] = raw_dist
        for i in range(M):
            D[i, i] = 0.0

        subtrees = {i: leaf_names[i] for i in range(M)}
        next_idx = M
        active = list(range(M))
        n_active = M

        while n_active > 3:
            n = n_active
            r = np.array([sum(D[active[i], active[j]] for j in range(n)) for i in range(n)])

            Q = np.full((n, n), np.inf)
            for i in range(n):
                for j in range(i + 1, n):
                    q = (n - 2) * D[active[i], active[j]] - r[i] - r[j]
                    Q[i, j] = q
                    Q[j, i] = q

            i, j = np.unravel_index(np.argmin(Q), Q.shape)
            actual_i, actual_j = active[i], active[j]

            div = 2.0 * max(n - 2, 1)
            d_iu = max(0.5 * D[actual_i, actual_j] + (r[i] - r[j]) / div, 0.001)
            d_ju = max(D[actual_i, actual_j] - d_iu, 0.001)

            subtrees[next_idx] = (
                f"({subtrees[actual_i]}:{d_iu:.6f},{subtrees[actual_j]}:{d_ju:.6f})"
            )

            for k in active:
                if k != actual_i and k != actual_j:
                    D[next_idx, k] = 0.5 * (D[actual_i, k] + D[actual_j, k] - D[actual_i, actual_j])
                    D[k, next_idx] = D[next_idx, k]
            D[next_idx, next_idx] = 0.0

            active = [k for k in active if k != actual_i and k != actual_j] + [next_idx]
            n_active -= 1
            next_idx += 1

        if n_active == 3:
            a, b, c = active
            newick = f"({subtrees[a]}:0.001,{subtrees[b]}:0.001,{subtrees[c]}:0.001);"
        elif n_active == 2:
            a, b = active
            newick = f"({subtrees[a]}:0.001,{subtrees[b]}:0.001);"
        else:
            newick = f"({subtrees[active[0]]});"

        return newick
