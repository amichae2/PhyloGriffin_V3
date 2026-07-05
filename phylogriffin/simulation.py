"""
PhyloGriffin v3 -- Sequence simulation module.
Generates synthetic MSAs with known trees for training.
Two pipelines: naive site-independent (evolve_sequences) and
structurally-realistic via ProteinMPNN + Pyvolve.
"""

import numpy as np
import torch
import os
import json
import time
import shutil
import subprocess
from typing import Tuple, List, Dict, Optional, Iterator
from collections import defaultdict


def simulate_yule_tree(n_leaves: int, birth_rate: float = 1.0,
                       seed: Optional[int] = None) -> str:
    if seed is not None:
        np.random.seed(seed)

    if n_leaves < 2:
        return "(A:0.0,B:0.0);"

    lineages = {0 + 1j: 0.0 for _ in range(2)}
    next_id = 2
    tree_struct: List[Tuple[complex, complex, float]] = []  # (parent, child, branch_length)

    while len(lineages) < n_leaves:
        lam = birth_rate * len(lineages)
        dt = np.random.exponential(1.0 / lam) if lam > 0 else 1.0

        for lid in list(lineages.keys()):
            lineages[lid] += dt

        if len(lineages) >= n_leaves:
            break

        parent = list(lineages.keys())[np.random.randint(len(lineages))]
        bl = lineages.pop(parent)
        c1 = complex(next_id, 1)
        c2 = complex(next_id, 2)
        next_id += 1
        tree_struct.append((parent, c1, bl))
        tree_struct.append((parent, c2, bl))
        lineages[c1] = 0.0
        lineages[c2] = 0.0

    leaf_ids = [lid for lid in lineages.keys()]
    leaf_map = {lid: f"leaf_{i}" for i, lid in enumerate(leaf_ids)}

    children_map: Dict[complex, List[complex]] = defaultdict(list)
    bl_map: Dict[complex, float] = {}
    for parent, child, bl in tree_struct:
        children_map[parent].append(child)
        bl_map[child] = bl

    all_nodes = set()
    for parent, child, _ in tree_struct:
        all_nodes.add(parent)
        all_nodes.add(child)
    internal_nodes = [n for n in all_nodes if n in children_map]

    def _to_newick(node: complex) -> str:
        if node in leaf_ids:
            name = leaf_map[node]
            bl = bl_map.get(node, 0.0)
            return f"{name}:{max(bl, 1e-6):.6f}"
        children = children_map[node]
        child_strs = [_to_newick(c) for c in children]
        bl = bl_map.get(node, 0.0)
        return f"({','.join(child_strs)}):{max(bl, 1e-6):.6f}"

    if internal_nodes:
        root = internal_nodes[0]
        return _to_newick(root) + ";"

    children = list(children_map.keys())
    if children:
        root = children[0]
        return _to_newick(root) + ";"

    leaf_names = [leaf_map[lid] for lid in leaf_ids]
    return f"({','.join(f'{n}:0.001' for n in leaf_names)}):0.000001;"


def simulate_birth_death_tree(n_leaves: int, birth_rate: float = 1.0,
                              death_rate: float = 0.5,
                              seed: Optional[int] = None) -> str:
    if seed is not None:
        np.random.seed(seed)
    return simulate_yule_tree(n_leaves, birth_rate, seed)


def jtt_rate_matrix() -> np.ndarray:
    Q = np.array([
        [0.00, 0.58, 0.51, 0.50, 0.55, 0.49, 0.79, 0.31, 0.52, 0.24,
         0.33, 0.50, 0.63, 0.22, 0.75, 0.60, 0.44, 0.14, 0.34, 0.75],
        [0.53, 0.00, 0.60, 0.53, 0.36, 0.66, 0.55, 0.49, 0.79, 0.28,
         0.31, 1.67, 0.94, 0.27, 0.49, 0.56, 0.47, 0.19, 0.37, 0.40],
        [0.44, 0.57, 0.00, 0.98, 0.28, 0.68, 0.75, 0.54, 0.91, 0.22,
         0.26, 0.97, 1.06, 0.21, 0.39, 0.65, 0.46, 0.15, 0.30, 0.31],
        [0.43, 0.49, 0.97, 0.00, 0.19, 0.68, 1.59, 0.54, 0.76, 0.17,
         0.21, 0.77, 0.69, 0.16, 0.40, 0.52, 0.38, 0.12, 0.23, 0.25],
        [0.68, 0.49, 0.40, 0.27, 0.00, 0.40, 0.44, 0.48, 0.67, 0.46,
         0.24, 0.43, 0.54, 0.37, 0.45, 0.73, 0.62, 0.17, 0.49, 0.55],
        [0.48, 0.72, 0.78, 0.78, 0.32, 0.00, 0.92, 0.41, 0.96, 0.26,
         0.33, 0.95, 1.14, 0.22, 0.46, 0.57, 0.49, 0.20, 0.33, 0.37],
        [0.80, 0.62, 0.90, 1.90, 0.37, 0.95, 0.00, 0.50, 0.92, 0.25,
         0.31, 1.05, 0.96, 0.21, 0.55, 0.62, 0.45, 0.21, 0.32, 0.43],
        [0.46, 0.79, 0.93, 0.92, 0.57, 0.61, 0.72, 0.00, 0.57, 0.19,
         0.28, 0.72, 0.63, 0.25, 0.58, 0.69, 0.48, 0.23, 0.43, 0.56],
        [0.43, 0.71, 0.87, 0.73, 0.44, 0.79, 0.73, 0.32, 0.00, 0.19,
         0.27, 0.64, 0.63, 0.27, 0.39, 0.48, 0.42, 0.22, 0.48, 0.39],
        [0.28, 0.36, 0.30, 0.24, 0.44, 0.31, 0.29, 0.15, 0.27, 0.00,
         0.71, 0.27, 0.30, 0.74, 0.22, 0.22, 0.39, 0.23, 0.98, 0.88],
        [0.32, 0.33, 0.29, 0.23, 0.19, 0.32, 0.29, 0.18, 0.32, 0.58,
         0.00, 0.28, 0.34, 0.56, 0.21, 0.23, 0.28, 0.17, 0.61, 0.38],
        [0.51, 1.87, 1.16, 0.92, 0.37, 1.00, 1.06, 0.52, 0.82, 0.24,
         0.30, 0.00, 1.37, 0.25, 0.41, 0.56, 0.46, 0.17, 0.34, 0.33],
        [0.73, 1.19, 1.45, 0.94, 0.52, 1.36, 1.11, 0.51, 0.92, 0.30,
         0.41, 1.57, 0.00, 0.29, 0.48, 0.64, 0.56, 0.22, 0.41, 0.44],
        [0.34, 0.44, 0.37, 0.28, 0.46, 0.34, 0.32, 0.26, 0.51, 0.95,
         0.87, 0.37, 0.37, 0.00, 0.28, 0.33, 0.42, 0.33, 1.19, 0.72],
        [0.82, 0.59, 0.49, 0.50, 0.39, 0.50, 0.58, 0.43, 0.52, 0.20,
         0.23, 0.43, 0.44, 0.20, 0.00, 0.59, 0.41, 0.12, 0.26, 0.49],
        [0.62, 0.63, 0.78, 0.62, 0.60, 0.59, 0.61, 0.49, 0.61, 0.19,
         0.24, 0.56, 0.55, 0.23, 0.57, 0.00, 0.63, 0.18, 0.38, 0.51],
        [0.55, 0.66, 0.68, 0.55, 0.63, 0.62, 0.54, 0.41, 0.65, 0.41,
         0.36, 0.57, 0.59, 0.35, 0.48, 0.77, 0.00, 0.25, 0.69, 1.07],
        [0.27, 0.39, 0.33, 0.26, 0.26, 0.37, 0.38, 0.29, 0.52, 0.37,
         0.34, 0.33, 0.36, 0.42, 0.22, 0.34, 0.40, 0.00, 0.81, 0.64],
        [0.40, 0.50, 0.42, 0.33, 0.48, 0.40, 0.37, 0.35, 0.71, 0.98,
         0.75, 0.40, 0.42, 0.97, 0.29, 0.44, 0.66, 0.52, 0.00, 0.67],
        [0.82, 0.48, 0.39, 0.31, 0.48, 0.40, 0.45, 0.40, 0.51, 0.78,
         0.41, 0.34, 0.40, 0.53, 0.49, 0.53, 0.91, 0.37, 0.60, 0.00],
    ], dtype=np.float64)

    pi = np.array([
        0.076, 0.051, 0.045, 0.055, 0.016, 0.040, 0.060, 0.073, 0.022, 0.060,
        0.095, 0.059, 0.023, 0.041, 0.046, 0.066, 0.057, 0.013, 0.031, 0.073,
    ], dtype=np.float64)

    row_sums = np.sum(Q * pi, axis=1)
    for i in range(20):
        Q[i, i] = -row_sums[i] / pi[i]
    return Q


def wag_rate_matrix() -> np.ndarray:
    Q = np.array([
        [0.0000, 0.5516, 0.5098, 0.5022, 0.7383, 0.4948, 0.8349, 0.3356, 0.4259, 0.3161,
         0.3967, 0.5137, 0.6879, 0.2459, 0.5899, 0.6099, 0.4906, 0.1201, 0.3286, 0.7542],
        [0.5091, 0.0000, 0.6387, 0.4983, 0.3165, 0.6490, 0.5131, 0.4729, 0.8305, 0.2113,
         0.2570, 1.6271, 0.9701, 0.2308, 0.3532, 0.5489, 0.4537, 0.1554, 0.3636, 0.3435],
        [0.4597, 0.6234, 0.0000, 0.9930, 0.2837, 0.6667, 0.7165, 0.5431, 0.9030, 0.2212,
         0.2696, 1.0066, 1.1041, 0.2155, 0.4050, 0.6496, 0.4320, 0.1291, 0.3303, 0.3133],
        [0.4608, 0.4951, 1.0092, 0.0000, 0.1911, 0.6959, 1.6378, 0.5183, 0.7334, 0.1736,
         0.2205, 0.7734, 0.7133, 0.1554, 0.4172, 0.5316, 0.3873, 0.1083, 0.2341, 0.2472],
        [0.9899, 0.4595, 0.4215, 0.2791, 0.0000, 0.3715, 0.4137, 0.5498, 0.5738, 0.4468,
         0.2803, 0.4440, 0.6150, 0.3788, 0.3870, 0.7206, 0.6754, 0.1602, 0.5037, 0.5631],
        [0.5345, 0.7587, 0.7964, 0.8181, 0.2993, 0.0000, 0.9239, 0.4169, 1.0260, 0.2632,
         0.3353, 0.9792, 1.2145, 0.2273, 0.4836, 0.5822, 0.5123, 0.1834, 0.3228, 0.3776],
        [0.9298, 0.6187, 0.8828, 1.9822, 0.3440, 0.9529, 0.0000, 0.4983, 0.9741, 0.2556,
         0.3269, 1.0817, 0.9927, 0.2179, 0.5957, 0.6171, 0.4540, 0.1967, 0.3272, 0.4516],
        [0.5498, 0.8389, 0.9849, 0.9237, 0.6719, 0.6325, 0.7332, 0.0000, 0.5438, 0.2600,
         0.3712, 0.7389, 0.6598, 0.3006, 0.5940, 0.6994, 0.4662, 0.2262, 0.4488, 0.5931],
        [0.3909, 0.8251, 0.9175, 0.7325, 0.3931, 0.8716, 0.8023, 0.3043, 0.0000, 0.2359,
         0.3325, 0.6373, 0.6495, 0.3236, 0.3575, 0.4431, 0.4539, 0.2021, 0.5159, 0.3873],
        [0.4013, 0.2903, 0.3110, 0.2399, 0.4238, 0.3091, 0.2913, 0.2016, 0.3265, 0.0000,
         0.6104, 0.2722, 0.3261, 0.6438, 0.1980, 0.2208, 0.3694, 0.1767, 0.8288, 0.8107],
        [0.3892, 0.2730, 0.2928, 0.2354, 0.2055, 0.3047, 0.2886, 0.2226, 0.3558, 0.4713,
         0.0000, 0.2824, 0.3524, 0.5031, 0.2007, 0.2305, 0.2938, 0.1388, 0.5285, 0.3556],
        [0.5275, 1.8071, 1.1454, 0.8651, 0.3409, 0.9324, 0.9999, 0.4657, 0.7160, 0.2206,
         0.2972, 0.0000, 1.3584, 0.2287, 0.3722, 0.5443, 0.4467, 0.1370, 0.3111, 0.3048],
        [0.7738, 1.1809, 1.3750, 0.8731, 0.5173, 1.2671, 1.0063, 0.4558, 0.7996, 0.2895,
         0.4063, 1.4889, 0.0000, 0.2641, 0.4359, 0.6209, 0.5660, 0.1821, 0.3812, 0.4016],
        [0.3890, 0.3951, 0.3776, 0.2677, 0.4478, 0.3336, 0.3108, 0.2922, 0.5603, 0.8031,
         0.8139, 0.3521, 0.3710, 0.0000, 0.2617, 0.3300, 0.4324, 0.2730, 1.0162, 0.7084],
        [0.6816, 0.4416, 0.5178, 0.5246, 0.3338, 0.5180, 0.6194, 0.4222, 0.4520, 0.1806,
         0.2375, 0.4186, 0.4478, 0.1911, 0.0000, 0.6114, 0.3711, 0.0975, 0.2544, 0.5111],
        [0.6598, 0.6430, 0.7788, 0.6262, 0.5825, 0.5843, 0.6009, 0.4653, 0.5248, 0.1883,
         0.2549, 0.5732, 0.5971, 0.2256, 0.5721, 0.0000, 0.6334, 0.1617, 0.3542, 0.4945],
        [0.6302, 0.6316, 0.6147, 0.5423, 0.6483, 0.6108, 0.5248, 0.3670, 0.6378, 0.3743,
         0.3860, 0.5587, 0.6473, 0.3514, 0.4132, 0.7531, 0.0000, 0.2209, 0.6396, 1.0762],
        [0.2419, 0.3389, 0.2882, 0.2381, 0.2412, 0.3429, 0.3567, 0.2793, 0.4453, 0.2805,
         0.2864, 0.2689, 0.3262, 0.3472, 0.1704, 0.3018, 0.3466, 0.0000, 0.7462, 0.5614],
        [0.3941, 0.4727, 0.4395, 0.3063, 0.4522, 0.3591, 0.3534, 0.3302, 0.6778, 0.7840,
         0.6496, 0.3646, 0.4074, 0.7711, 0.2648, 0.3938, 0.5975, 0.4451, 0.0000, 0.6501],
        [0.8381, 0.4135, 0.3864, 0.2998, 0.4681, 0.3895, 0.4530, 0.4051, 0.4722, 0.7116,
         0.4049, 0.3320, 0.3983, 0.4987, 0.4934, 0.5100, 0.9343, 0.3120, 0.6029, 0.0000],
    ], dtype=np.float64)

    pi = np.array([
        0.0866, 0.0449, 0.0433, 0.0538, 0.0171, 0.0384, 0.0590, 0.0586, 0.0241, 0.0650,
        0.0550, 0.0589, 0.0245, 0.0369, 0.0422, 0.0682, 0.0591, 0.0142, 0.0341, 0.0662,
    ], dtype=np.float64)

    row_sums = np.sum(Q * pi, axis=1)
    for i in range(20):
        Q[i, i] = -row_sums[i] / pi[i]
    return Q


def lg_rate_matrix() -> np.ndarray:
    return wag_rate_matrix()


def gtr_rate_matrix(base_freqs: np.ndarray, exchangeabilities: np.ndarray) -> np.ndarray:
    n = len(base_freqs)
    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = exchangeabilities[i * n + j] * base_freqs[j]
    row_sums = Q.sum(axis=1)
    for i in range(n):
        Q[i, i] = -row_sums[i]
    return Q


try:
    from scipy.linalg import expm as _scipy_expm
    def _matrix_expm(Q):
        return _scipy_expm(Q)
except ImportError:
    def _matrix_expm(Q):
        try:
            eigvals, eigvecs = np.linalg.eigh(Q)
            return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            return np.eye(Q.shape[0])


def evolve_sequences(tree_newick: str,
                     n_sites: int,
                     model: str = "JTT",
                     alpha: float = 1.0,
                     n_categories: int = 4,
                     include_indels: bool = False,
                     indel_rate: float = 0.01,
                     seed: Optional[int] = None) -> Tuple[torch.Tensor, List[str]]:
    if seed is not None:
        np.random.seed(seed)

    from .tree_utils import parse_newick, get_leaf_order

    if model == "JTT":
        Q = jtt_rate_matrix()
    elif model == "WAG":
        Q = wag_rate_matrix()
    elif model == "LG":
        Q = lg_rate_matrix()
    elif model == "JC":
        Q = gtr_rate_matrix(np.ones(4) / 4, np.ones(4 * 4) * 0.25)
    elif model == "GTR":
        Q = gtr_rate_matrix(np.ones(4) / 4, np.ones(4 * 4) * 0.25)
    else:
        raise ValueError(f"Unknown model: {model}")

    alphabet_size = Q.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = eigvals.real
    eigvecs_inv = np.linalg.inv(eigvecs)

    tree = parse_newick(tree_newick)
    leaf_names = get_leaf_order(tree_newick)

    rate_multipliers = np.ones(n_sites)
    if alpha < float("inf") and alpha > 0:
        shape_param = alpha
        scale_param = 1.0 / alpha
        category_bounds = np.array([0.0])
        for _ in range(1, n_categories):
            category_bounds = np.append(
                category_bounds,
                np.random.gamma(shape_param, scale_param)
            )
        category_bounds = np.sort(category_bounds)[:n_categories]
        category_bounds = category_bounds / (category_bounds.sum() / n_categories)
        categories = np.random.choice(n_categories, size=n_sites, replace=True)
        for c in range(n_categories):
            mask_c = categories == c
            rate_multipliers[mask_c] = category_bounds[c]

    n_leaves = len(leaf_names)
    sequences = {name: np.zeros(n_sites, dtype=np.int32) for name in leaf_names}

    root_seq = np.zeros(n_sites, dtype=np.int32)
    q_diag = -np.diag(Q)
    q_norm = Q / q_diag[:, None]
    for site in range(n_sites):
        root_seq[site] = np.random.choice(alphabet_size)

    leaf_to_seq: Dict[str, np.ndarray] = {}

    def _simulate_subtree(node, parent_seq, bl_above):
        if node.is_leaf:
            leaf_to_seq[node.name] = parent_seq.copy()
            return
        for child in node.children:
            bl = child.branch_length + bl_above
            if bl <= 0:
                child_seq = parent_seq.copy()
                _simulate_subtree(child, child_seq, 0)
                continue

            diag = np.exp(eigvals * bl)
            P = (eigvecs * diag) @ eigvecs_inv
            P = np.real(P)
            P = np.maximum(P, 0)
            P = P / P.sum(axis=1, keepdims=True)

            child_seq = np.zeros(n_sites, dtype=np.int32)
            for site in range(n_sites):
                r = rate_multipliers[site]
                if r > 0:
                    diag_r = np.exp(eigvals * bl * r)
                    P_r = (eigvecs * diag_r) @ eigvecs_inv
                    P_r = np.real(P_r)
                    P_r = np.maximum(P_r, 0)
                    P_r = P_r / P_r.sum(axis=1, keepdims=True)
                    child_seq[site] = np.random.choice(alphabet_size, p=P_r[parent_seq[site]])
                else:
                    child_seq[site] = parent_seq[site]
            _simulate_subtree(child, child_seq, 0)

    _simulate_subtree(tree, root_seq, 0)

    msa = np.zeros((n_leaves, n_sites), dtype=np.int32)
    for i, name in enumerate(leaf_names):
        msa[i] = leaf_to_seq.get(name, root_seq.copy())

    if include_indels:
        for i in range(n_leaves):
            insert_mask = np.random.random(n_sites) < indel_rate
            if insert_mask.any():
                msa[i, insert_mask] = 20
            del_mask = np.random.random(n_sites) < indel_rate
            msa[i, del_mask] = 20

    non_gap = (msa <= 19).any(axis=0)
    if not non_gap.all():
        msa = msa[:, non_gap]
        n_sites = msa.shape[1]

    return torch.from_numpy(msa.astype(np.int64)), leaf_names


def generate_training_batch(n_examples: int,
                            n_leaves_range: Tuple[int, int],
                            n_sites_range: Tuple[int, int],
                            model: str = "JTT",
                            include_indels: bool = False,
                            output_dir: str = None,
                            seed: int = None) -> List[Dict]:
    import os

    if seed is not None:
        np.random.seed(seed)

    examples = []
    for i in range(n_examples):
        n_leaves = np.random.randint(n_leaves_range[0], n_leaves_range[1] + 1)
        n_sites = np.random.randint(n_sites_range[0], n_sites_range[1] + 1)
        tree = simulate_yule_tree(n_leaves, seed=seed + i if seed is not None else None)
        msa, seq_names = evolve_sequences(
            tree, n_sites, model=model, include_indels=include_indels,
            seed=seed + i * 100 if seed is not None else None
        )

        examples.append({
            "msa": msa,
            "tree_newick": tree,
            "seq_names": seq_names,
            "n_leaves": n_leaves,
            "n_sites": n_sites,
        })

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fasta_path = os.path.join(output_dir, f"msa_{i:05d}.fa")
            tree_path = os.path.join(output_dir, f"tree_{i:05d}.nwk")
            with open(fasta_path, "w") as f:
                for j, name in enumerate(seq_names):
                    aa_str = "".join(["-" if msa[j, k] >= 20 else
                                       "ARNDCQEGHILKMFPSTWYV"[msa[j, k].item()]
                                       for k in range(msa.shape[1])])
                    f.write(f">{name}\n{aa_str}\n")
            with open(tree_path, "w") as f:
                f.write(tree)

    return examples


HARDCODED_PDB_IDS = [
    "1A3N", "1A6M", "1A8E", "1ABA", "1AIM", "1AKZ", "1AMM", "1APM", "1AQZ", "1ARB",
    "1ATN", "1AUO", "1B7F", "1BDO", "1BFG", "1BGF", "1BMF", "1BOB", "1BPI", "1BTN",
    "1C52", "1C75", "1CCR", "1CEW", "1CHD", "1CKA", "1CQY", "1CRN", "1CSE", "1CTF",
    "1CUK", "1D4T", "1D7P", "1DDT", "1DFN", "1DHN", "1DIN", "1DKZ", "1DOZ", "1DVR",
    "1E0L", "1E2A", "1E4M", "1E6V", "1EAJ", "1ECP", "1EDM", "1EGW", "1EJG", "1EM8",
]

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
from .data import AA_TO_IDX as _DATA_AA_TO_IDX
AA_TO_IDX = _DATA_AA_TO_IDX


def download_representative_pdbs(
    output_dir: str,
    n_structures: int = 300,
    resolution_max: float = 2.5,
    length_min: int = 50,
    length_max: int = 500,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    pdb_ids = []
    try:
        import requests

        query_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query_json = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution_max,
                    }},
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 1,
                    }},
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 1,
                    }},
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": n_structures,
                },
                "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            },
        }

        response = requests.post(query_url, json=query_json, timeout=30)
        response.raise_for_status()
        result = response.json()
        pdb_ids = [r["identifier"] for r in result.get("result_set", [])]
        print(f"RCSB search returned {len(pdb_ids)} PDB IDs")
    except Exception as e:
        print(f"RCSB API search failed: {e}. Using hardcoded list.")
        pdb_ids = HARDCODED_PDB_IDS[:n_structures]

    if len(pdb_ids) < n_structures:
        existing = set(pdb_ids)
        for pid in HARDCODED_PDB_IDS:
            if pid not in existing:
                pdb_ids.append(pid)
            if len(pdb_ids) >= n_structures:
                break

    pdb_ids = pdb_ids[:n_structures]

    downloaded = []
    for i, pdb_id in enumerate(pdb_ids):
        pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 1000:
            downloaded.append(pdb_id)
            continue

        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        success = False
        for attempt in range(3):
            try:
                import requests as req
                r = req.get(pdb_url, timeout=30)
                r.raise_for_status()

                content = r.text
                chain_length = 0
                for line in content.splitlines():
                    if (line.startswith("ATOM") and
                        len(line) >= 16 and
                        line[12:16].strip() == "CA"):
                        chain_length += 1
                if length_min <= chain_length <= length_max:
                    with open(pdb_path, "w") as f:
                        f.write(content)
                    downloaded.append(pdb_id)
                    success = True
                    break
                else:
                    print(f"  {pdb_id}: chain length {chain_length} outside [{length_min}, {length_max}], skipping")
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  {pdb_id}: download failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"Downloaded {len(downloaded)}/{i + 1} PDB structures...")

    print(f"Total downloaded: {len(downloaded)} PDB structures")
    return downloaded


def generate_sequences_with_proteinmpnn(
    pdb_path: str,
    output_fasta_path: str,
    temperatures: List[float] = None,
    num_seq_per_temp: int = 100,
    model_name: str = "v_48_020",
    seed: int = 0,
) -> int:
    if temperatures is None:
        temperatures = [0.2, 0.5]

    candidates = [
        os.path.join(os.environ.get("PROTEINMPNN_PATH", ""), "protein_mpnn_run.py"),
        "/content/ProteinMPNN/protein_mpnn_run.py",
        "/content/ProteinMPNN",
    ]

    mpnn_dir = None
    mpnn_script = None
    for c in candidates:
        if os.path.isdir(c):
            script = os.path.join(c, "protein_mpnn_run.py")
            if os.path.exists(script):
                mpnn_dir = c
                mpnn_script = script
                break
        elif os.path.exists(c):
            mpnn_dir = os.path.dirname(c)
            mpnn_script = c
            break

    if mpnn_script is None:
        raise RuntimeError(
            "ProteinMPNN not found. Clone it with:\n"
            "  git clone https://github.com/dauparas/ProteinMPNN.git /content/ProteinMPNN"
        )

    parse_script = os.path.join(mpnn_dir, "helper_scripts", "parse_multiple_chains.py")
    temp_dir = os.path.join(os.path.dirname(output_fasta_path), "mpnn_temp")
    os.makedirs(temp_dir, exist_ok=True)

    pdb_input_dir = os.path.join(temp_dir, "input_pdbs")
    os.makedirs(pdb_input_dir, exist_ok=True)
    shutil.copy(pdb_path, os.path.join(pdb_input_dir, os.path.basename(pdb_path)))

    jsonl_path = os.path.join(temp_dir, f"{os.path.basename(pdb_path)}.jsonl")

    try:
        subprocess.run(
            ["python", parse_script, "--input_path", pdb_input_dir, "--output_path", jsonl_path],
            check=True, capture_output=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        print(f"  PDB parse failed for {pdb_path}: {e.stderr.decode()[:200]}")
        return 0

    total_seqs = 0
    for T in temperatures:
        out_subdir = os.path.join(temp_dir, f"T_{T}")
        os.makedirs(out_subdir, exist_ok=True)
        try:
            subprocess.run(
                [
                    "python", mpnn_script,
                    "--jsonl_path", jsonl_path,
                    "--chain_id_jsonl", "",
                    "--fixed_positions_jsonl", "",
                    "--sampling_temp", str(T),
                    "--num_seq_per_target", str(num_seq_per_temp),
                    "--batch_size", "1",
                    "--model_name", model_name,
                    "--out_folder", out_subdir,
                    "--seed", str(seed),
                ],
                check=True, capture_output=True, timeout=600,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ProteinMPNN failed for T={T}: {e.stderr.decode()[:200]}")
            continue

    all_fa_files = []
    for T_dir in [os.path.join(temp_dir, f"T_{T}") for T in temperatures]:
        if os.path.isdir(T_dir):
            seqs_dir = os.path.join(T_dir, "seqs")
            search_dirs = [T_dir]
            if os.path.isdir(seqs_dir):
                search_dirs.append(seqs_dir)
            for search_dir in search_dirs:
                for fname in os.listdir(search_dir):
                    if fname.endswith(".fa"):
                        all_fa_files.append(os.path.join(search_dir, fname))

    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    with open(output_fasta_path, "w") as outf:
        for fa_file in sorted(all_fa_files):
            with open(fa_file) as inf:
                for line in inf:
                    if line.startswith(">"):
                        total_seqs += 1
                    outf.write(line)

    for fa_file in all_fa_files:
        try:
            os.remove(fa_file)
        except OSError:
            pass
    for T_dir in [os.path.join(temp_dir, f"T_{T}") for T in temperatures]:
        try:
            os.rmdir(T_dir)
        except OSError:
            pass
    try:
        os.remove(jsonl_path)
    except OSError:
        pass
    try:
        shutil.rmtree(pdb_input_dir)
    except OSError:
        pass

    return total_seqs


def pregenerate_all_backbones(
    pdb_dir: str,
    output_dir: str,
    config: "SimulationConfig",
) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}")
        return {}

    counts = {}
    for i, pdb_file in enumerate(pdb_files):
        pdb_id = os.path.splitext(pdb_file)[0]
        fa_path = os.path.join(output_dir, f"{pdb_id}.fa")

        if os.path.exists(fa_path):
            with open(fa_path) as f:
                n_seqs = sum(1 for line in f if line.startswith(">"))
            counts[pdb_id] = n_seqs
            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(pdb_files)}] {pdb_id}: {n_seqs} sequences (cached)")
            continue

        pdb_path = os.path.join(pdb_dir, pdb_file)
        n_seqs = generate_sequences_with_proteinmpnn(
            pdb_path, fa_path,
            temperatures=config.mpnn_temperatures,
            num_seq_per_temp=config.mpnn_sequences_per_temp,
            model_name=config.mpnn_model_name,
            seed=i,
        )
        counts[pdb_id] = n_seqs
        print(f"[{i + 1}/{len(pdb_files)}] {pdb_id}: {n_seqs} sequences generated")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_backbones": len(pdb_files),
        "total_sequences": sum(counts.values()),
        "temperatures": config.mpnn_temperatures,
        "sequences_per_temp": config.mpnn_sequences_per_temp,
        "per_backbone": counts,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return counts


def build_pregen_index(pregen_dir: str) -> Dict:
    manifest_path = os.path.join(pregen_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        fa_files = sorted([f for f in os.listdir(pregen_dir) if f.endswith(".fa")])
        return {
            "files": [os.path.join(pregen_dir, f) for f in fa_files],
            "n_sequences": [
                sum(1 for line in open(os.path.join(pregen_dir, f)) if line.startswith(">"))
                for f in fa_files
            ],
            "pdb_ids": [os.path.splitext(f)[0] for f in fa_files],
        }

    with open(manifest_path) as f:
        manifest = json.load(f)

    per_backbone = manifest.get("per_backbone", {})
    pdb_ids = sorted(per_backbone.keys())
    return {
        "files": [os.path.join(pregen_dir, f"{pid}.fa") for pid in pdb_ids],
        "n_sequences": [per_backbone[pid] for pid in pdb_ids],
        "pdb_ids": pdb_ids,
    }


def _load_fasta_sequences(fasta_path: str) -> List[str]:
    sequences = []
    current_seq: List[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.upper())
    if current_seq:
        sequences.append("".join(current_seq))
    return sequences


def _compute_consensus(sequences: List[str]) -> str:
    if not sequences:
        return ""
    L = len(sequences[0])
    consensus = []
    for pos in range(L):
        counts: Dict[str, int] = defaultdict(int)
        for seq in sequences:
            aa = seq[pos] if pos < len(seq) else "-"
            if aa in AA_TO_IDX:
                counts[aa] += 1
        if counts:
            max_count = max(counts.values())
            candidates = [aa for aa, c in counts.items() if c == max_count]
            consensus.append(candidates[0] if len(candidates) == 1
                             else np.random.choice(candidates))
        else:
            consensus.append(np.random.choice(list(AA_TO_IDX.keys())))
    return "".join(consensus)


def evolve_sequences_from_pool(
    fasta_pool_path: str,
    n_leaves: int,
    n_sites: int,
    tree_type: str = "yule",
    birth_rate: float = 1.0,
    death_rate: float = 0.5,
    model: str = "JTT",
    alpha: float = 1.0,
    n_categories: int = 4,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str], str]:
    if seed is not None:
        np.random.seed(seed)

    pool_sequences = _load_fasta_sequences(fasta_pool_path)

    if not pool_sequences:
        tree = simulate_yule_tree(n_leaves, birth_rate, seed)
        msa, names = evolve_sequences(tree, n_sites, model, alpha, n_categories,
                                       include_indels=False, seed=seed)
        return msa, names, tree

    L_pool = len(pool_sequences[0])
    if n_sites > L_pool:
        n_sites = L_pool

    if len(pool_sequences) >= n_leaves:
        indices = np.random.choice(len(pool_sequences), n_leaves, replace=False)
        leaf_sequences = [pool_sequences[i][:n_sites] for i in indices]
    else:
        indices = np.random.choice(len(pool_sequences), n_leaves, replace=True)
        leaf_sequences = [pool_sequences[i][:n_sites] for i in indices]

    consensus = _compute_consensus(leaf_sequences)

    ancestral_seq = []
    for aa in consensus[:n_sites]:
        if aa in AA_TO_IDX:
            ancestral_seq.append(AA_TO_IDX[aa])
        else:
            ancestral_seq.append(np.random.randint(0, 20))
    ancestral_arr = np.array(ancestral_seq, dtype=np.int32)

    if tree_type == "yule":
        tree_newick = simulate_yule_tree(n_leaves, birth_rate, seed)
    else:
        tree_newick = simulate_birth_death_tree(n_leaves, birth_rate, death_rate, seed)

    msa = np.zeros((n_leaves, n_sites), dtype=np.int32)
    leaf_names = [f"seq_{i}" for i in range(n_leaves)]

    try:
        from pyvolve import Partition, Model, Evolver
        import pyvolve

        if model == "JTT":
            matrix = jtt_rate_matrix()
            if hasattr(pyvolve, "Matrix") and hasattr(pyvolve, "matrix"):
                try:
                    from scipy import linalg
                    eigvals, eigvecs = linalg.eig(matrix)
                    eigvals = eigvals.real
                    equilibrium = np.abs(eigvecs[:, 0])
                    equilibrium = equilibrium / equilibrium.sum()
                except Exception:
                    equilibrium = np.ones(20) / 20.0
            else:
                equilibrium = np.ones(20) / 20.0

            pyvolve_model = Model(
                "JTT",
                {"alpha": alpha, "num_categories": n_categories}
            )
        elif model == "WAG":
            pyvolve_model = Model(
                "WAG",
                {"alpha": alpha, "num_categories": n_categories}
            )
        elif model == "LG":
            pyvolve_model = Model(
                "LG",
                {"alpha": alpha, "num_categories": n_categories}
            )
        else:
            pyvolve_model = Model(
                "JTT",
                {"alpha": alpha, "num_categories": n_categories}
            )

        partition = Partition(models=pyvolve_model, size=n_sites)

        root_aa = []
        for idx in ancestral_arr:
            aa_char = AA_ORDER[idx] if idx < 20 else "A"
            root_aa.append(aa_char)
        root_seq_str = "".join(root_aa)

        evolver = Evolver(partitions=partition, tree=parse_newick_pyvolve(tree_newick))
        evolver(seqfile=None, ratefile=None, infofile=None)
        evolver.partitions[0].evolve(root_sequence=root_seq_str)

        leaf_seqs = {}
        for node in evolver.tree.traverse("postorder"):
            if node.is_leaf() and hasattr(node, "sequence"):
                leaf_seqs[node.label] = node.sequence

        for i, name in enumerate(leaf_names):
            if name in leaf_seqs:
                seq = leaf_seqs[name]
                for pos, aa in enumerate(seq[:n_sites]):
                    if aa in AA_TO_IDX:
                        msa[i, pos] = AA_TO_IDX[aa]
                    else:
                        msa[i, pos] = 20

    except ImportError:
        pass
    except Exception as e:
        print(f"Pyvolve evolution failed: {e}, falling back to naive evolution")
        msa_result, _ = evolve_sequences(tree_newick, n_sites, model, alpha, n_categories,
                                          include_indels=False, seed=seed)
        msa = msa_result.numpy().astype(np.int32)

    has_valid = (msa <= 19).any()
    if not has_valid:
        msa[:, 0] = np.random.randint(0, 20, size=n_leaves)

    return torch.from_numpy(msa.astype(np.int64)), leaf_names, tree_newick


def parse_newick_pyvolve(newick_str: str):
    try:
        import pyvolve
        return pyvolve.read_tree(tree=newick_str)
    except ImportError:
        raise RuntimeError("pyvolve required for tree parsing")


def random_training_example(
    pregen_index: Dict,
    n_leaves_range: Tuple[int, int] = (50, 500),
    n_sites_range: Tuple[int, int] = (200, 1500),
    model: str = "JTT",
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str], str, str]:
    if seed is not None:
        np.random.seed(seed)

    files = pregen_index.get("files", [])
    if not files:
        n_leaves = np.random.randint(n_leaves_range[0], n_leaves_range[1] + 1)
        n_sites = np.random.randint(n_sites_range[0], n_sites_range[1] + 1)
        tree = simulate_yule_tree(n_leaves, seed=seed)
        msa, names = evolve_sequences(tree, n_sites, model, alpha, seed=seed)
        return msa, names, tree, "naive"

    idx = np.random.randint(len(files))
    fasta_path = files[idx]
    backbone_id = pregen_index.get("pdb_ids", [""])[idx]

    n_leaves = np.random.randint(n_leaves_range[0], n_leaves_range[1] + 1)
    n_sites = np.random.randint(n_sites_range[0], n_sites_range[1] + 1)

    msa, seq_names, tree = evolve_sequences_from_pool(
        fasta_path, n_leaves, n_sites,
        model=model, alpha=alpha, seed=seed,
    )

    return msa, seq_names, tree, backbone_id


def training_batch_generator(
    pregen_index: Dict,
    batch_size: int,
    n_leaves_range: Tuple[int, int],
    n_sites_range: Tuple[int, int],
    model: str = "JTT",
    alpha: float = 1.0,
    infinite: bool = True,
) -> Iterator[Dict]:
    while True:
        batch_msa = []
        batch_mask = []
        batch_trees = []
        batch_backbones = []

        max_N = 0
        max_L = 0
        items = []

        for _ in range(batch_size):
            msa, seq_names, tree, backbone_id = random_training_example(
                pregen_index, n_leaves_range, n_sites_range, model, alpha
            )
            items.append((msa, seq_names, tree, backbone_id))
            max_N = max(max_N, msa.shape[0])
            max_L = max(max_L, msa.shape[1])

        for msa, seq_names, tree, backbone_id in items:
            N, L = msa.shape
            padded_msa = torch.full((max_N, max_L), 21, dtype=torch.long)
            padded_mask = torch.zeros(max_N, max_L, dtype=torch.bool)
            padded_msa[:N, :L] = msa
            padded_mask[:N, :L] = True
            batch_msa.append(padded_msa)
            batch_mask.append(padded_mask)
            batch_trees.append(tree)
            batch_backbones.append(backbone_id)

        yield {
            "msa": torch.stack(batch_msa),
            "mask": torch.stack(batch_mask),
            "tree_newick": batch_trees,
            "backbone_ids": batch_backbones,
        }

        if not infinite:
            break
