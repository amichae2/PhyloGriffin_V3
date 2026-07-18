"""
PhyloGriffin v3 -- Sequence simulation module.
Generates synthetic MSAs with known trees for training.
Two pipelines: naive site-independent (evolve_sequences) and
structurally-realistic via ProteinMPNN + Pyvolve.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

from .data import AA_TO_IDX as _DATA_AA_TO_IDX

AA_TO_IDX = _DATA_AA_TO_IDX

if TYPE_CHECKING:
    from .config import SimulationConfig


def simulate_yule_tree(n_leaves: int, birth_rate: float = 1.0, seed: int | None = None) -> str:
    if seed is not None:
        np.random.seed(seed)

    if n_leaves < 2:
        return "(A:0.0,B:0.0);"

    lineages = {complex(0, i + 1): 0.0 for i in range(2)}
    next_id = 2
    tree_struct: list[tuple[complex, complex, float]] = []  # (parent, child, branch_length)

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

    children_map: dict[complex, list[complex]] = defaultdict(list)
    bl_map: dict[complex, float] = {}
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
        root_candidates = [n for n in children_map if n not in bl_map]
        if len(root_candidates) == 1:
            return _to_newick(root_candidates[0]) + ";"
        else:
            child_strs = [_to_newick(rc) for rc in root_candidates]
            return f"({','.join(child_strs)}):0.000001;"

    children = list(children_map.keys())
    if children:
        root = children[0]
        return _to_newick(root) + ";"

    leaf_names = [leaf_map[lid] for lid in leaf_ids]
    return f"({','.join(f'{n}:0.001' for n in leaf_names)}):0.000001;"


def simulate_birth_death_tree(
    n_leaves: int, birth_rate: float = 1.0, death_rate: float = 0.5, seed: int | None = None
) -> str:
    if seed is not None:
        np.random.seed(seed)

    if n_leaves < 2:
        return "(A:0.0,B:0.0);"

    if death_rate >= birth_rate:
        return simulate_yule_tree(n_leaves, birth_rate, seed if seed is not None else None)

    total_rate = birth_rate + death_rate
    birth_prob = birth_rate / total_rate

    max_attempts = 100
    for _ in range(max_attempts):
        lineages = {0 + 0j: 0.0}
        next_id = 1
        tree_struct: list[tuple[complex, complex, float]] = []

        while len(lineages) < n_leaves:
            n_extant = len(lineages)
            lam = total_rate * n_extant
            dt = np.random.exponential(1.0 / lam) if lam > 0 else 1.0

            for lid in list(lineages.keys()):
                lineages[lid] += dt

            if len(lineages) >= n_leaves:
                break

            parent = list(lineages.keys())[np.random.randint(n_extant)]
            bl = lineages.pop(parent)

            if np.random.random() < birth_prob:
                c1 = complex(next_id, 1)
                c2 = complex(next_id, 2)
                next_id += 1
                tree_struct.append((parent, c1, bl))
                tree_struct.append((parent, c2, bl))
                lineages[c1] = 0.0
                lineages[c2] = 0.0

            if len(lineages) == 0:
                break

        if len(lineages) >= n_leaves:
            break
    else:
        return simulate_yule_tree(n_leaves, birth_rate, seed if seed is not None else None)

    leaf_ids = set(lineages.keys())
    leaf_map = {
        lid: f"leaf_{i}" for i, lid in enumerate(sorted(leaf_ids, key=lambda x: (x.real, x.imag)))
    }

    children_map: dict[complex, list[complex]] = defaultdict(list)
    bl_map: dict[complex, float] = {}
    for parent, child, bl in tree_struct:
        children_map[parent].append(child)
        bl_map[child] = bl

    def _is_alive(node: complex) -> bool:
        if node in leaf_ids:
            return True
        return any(_is_alive(c) for c in children_map.get(node, []))

    alive_cache: dict[complex, bool] = {}
    for parent in children_map:
        alive_cache[parent] = _is_alive(parent)

    def _to_newick(node: complex, extra_bl: float = 0.0) -> str:
        if node in leaf_ids:
            name = leaf_map[node]
            bl = bl_map.get(node, 0.0) + extra_bl
            return f"{name}:{max(bl, 1e-6):.6f}"
        children = children_map.get(node, [])
        alive = [c for c in children if alive_cache.get(c, c in leaf_ids)]
        if len(alive) == 0:
            return f"dead_node:{max(extra_bl, 1e-6):.6f}"
        if len(alive) == 1:
            return _to_newick(alive[0], extra_bl + bl_map.get(node, 0.0))
        child_strs = [_to_newick(c) for c in alive]
        bl = bl_map.get(node, 0.0) + extra_bl
        return f"({','.join(child_strs)}):{max(bl, 1e-6):.6f}"

    root = 0 + 0j
    if root in children_map:
        alive_children = [c for c in children_map[root] if alive_cache.get(c, c in leaf_ids)]
        if alive_children:
            if len(alive_children) == 1:
                return _to_newick(alive_children[0]) + ";"
            child_strs = [_to_newick(c) for c in alive_children]
            return f"({','.join(child_strs)}):0.000001;"

    leaf_names = [leaf_map[lid] for lid in sorted(leaf_ids, key=lambda x: (x.real, x.imag))]
    return f"({','.join(f'{n}:0.001' for n in leaf_names)}):0.000001;"


def _build_rate_matrix(exchangeabilities: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    Q = exchangeabilities * freqs
    np.fill_diagonal(Q, 0.0)
    row_sums = Q.sum(axis=1)
    np.fill_diagonal(Q, -row_sums)
    mu = -(Q.diagonal() * freqs).sum()
    Q /= mu
    return Q


def jtt_rate_matrix() -> np.ndarray:
    from pyvolve.empirical_matrices import jtt_freqs, jtt_matrix

    return _build_rate_matrix(
        np.array(jtt_matrix, dtype=np.float64), np.array(jtt_freqs, dtype=np.float64)
    )


def wag_rate_matrix() -> np.ndarray:
    from pyvolve.empirical_matrices import wag_freqs, wag_matrix

    return _build_rate_matrix(
        np.array(wag_matrix, dtype=np.float64), np.array(wag_freqs, dtype=np.float64)
    )


def lg_rate_matrix() -> np.ndarray:
    from pyvolve.empirical_matrices import lg_freqs, lg_matrix

    return _build_rate_matrix(
        np.array(lg_matrix, dtype=np.float64), np.array(lg_freqs, dtype=np.float64)
    )


def jc_rate_matrix() -> np.ndarray:
    Q = np.full((4, 4), 1.0 / 3.0, dtype=np.float64)
    np.fill_diagonal(Q, -1.0)
    return Q


def gtr_rate_matrix(base_freqs: np.ndarray, exchange_rates: np.ndarray) -> np.ndarray:
    n = len(base_freqs)
    Q = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = exchange_rates[idx] * base_freqs[j]
            Q[j, i] = exchange_rates[idx] * base_freqs[i]
            idx += 1
    for i in range(n):
        Q[i, i] = -Q[i, :].sum()
    mu = -np.sum(base_freqs * np.diag(Q))
    if mu > 0:
        Q /= mu
    return Q


def _discrete_gamma_rates(alpha: float, n_categories: int) -> np.ndarray:
    from scipy import stats

    if alpha >= 1e6 or n_categories <= 1:
        return np.ones(n_categories, dtype=np.float64)

    shape = alpha
    scale = 1.0 / alpha

    bin_probs = np.linspace(0, 1, n_categories + 1)
    boundaries = stats.gamma.ppf(bin_probs, a=shape, scale=scale)

    rates = np.zeros(n_categories, dtype=np.float64)
    for i in range(n_categories):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi - lo < 1e-12:
            rates[i] = lo
        else:
            rates[i] = stats.gamma.expect(
                lambda x: x, args=(shape,), scale=scale, lb=lo, ub=hi, conditional=True
            )

    rates = rates / rates.mean()
    return rates


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


def evolve_sequences(
    tree_newick: str,
    n_sites: int,
    model: str = "JTT",
    alpha: float = 1.0,
    n_categories: int = 4,
    include_indels: bool = False,
    indel_rate: float = 0.01,
    seed: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    if seed is not None:
        np.random.seed(seed)

    from .tree_utils import get_leaf_order, parse_newick

    if model == "JTT":
        Q = jtt_rate_matrix()
    elif model == "WAG":
        Q = wag_rate_matrix()
    elif model == "LG":
        Q = lg_rate_matrix()
    elif model == "JC":
        Q = jc_rate_matrix()
    elif model == "GTR":
        Q = jc_rate_matrix()
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
        rate_categories = _discrete_gamma_rates(alpha, n_categories)
        categories = np.random.choice(n_categories, size=n_sites, replace=True)
        rate_multipliers = rate_categories[categories]

    n_leaves = len(leaf_names)

    root_seq = np.zeros(n_sites, dtype=np.int32)
    for site in range(n_sites):
        root_seq[site] = np.random.choice(alphabet_size)

    leaf_to_seq: dict[str, np.ndarray] = {}

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


def generate_training_batch(
    n_examples: int,
    n_leaves_range: tuple[int, int],
    n_sites_range: tuple[int, int],
    model: str = "JTT",
    include_indels: bool = False,
    output_dir: str = None,
    seed: int = None,
) -> list[dict]:
    import os

    if seed is not None:
        np.random.seed(seed)

    examples = []
    for i in range(n_examples):
        n_leaves = np.random.randint(n_leaves_range[0], n_leaves_range[1] + 1)
        n_sites = np.random.randint(n_sites_range[0], n_sites_range[1] + 1)
        tree = simulate_yule_tree(n_leaves, seed=seed + i if seed is not None else None)
        msa, seq_names = evolve_sequences(
            tree,
            n_sites,
            model=model,
            include_indels=include_indels,
            seed=seed + i * 100 if seed is not None else None,
        )

        examples.append(
            {
                "msa": msa,
                "tree_newick": tree,
                "seq_names": seq_names,
                "n_leaves": n_leaves,
                "n_sites": n_sites,
            }
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fasta_path = os.path.join(output_dir, f"msa_{i:05d}.fa")
            tree_path = os.path.join(output_dir, f"tree_{i:05d}.nwk")
            with open(fasta_path, "w") as f:
                for j, name in enumerate(seq_names):
                    aa_str = "".join(
                        [
                            "-" if msa[j, k] >= 20 else "ARNDCQEGHILKMFPSTWYV"[msa[j, k].item()]
                            for k in range(msa.shape[1])
                        ]
                    )
                    f.write(f">{name}\n{aa_str}\n")
            with open(tree_path, "w") as f:
                f.write(tree)

    return examples


HARDCODED_PDB_IDS = [
    "1A3N",
    "1A6M",
    "1A8E",
    "1ABA",
    "1AIM",
    "1AKZ",
    "1AMM",
    "1APM",
    "1AQZ",
    "1ARB",
    "1ATN",
    "1AUO",
    "1B7F",
    "1BDO",
    "1BFG",
    "1BGF",
    "1BMF",
    "1BOB",
    "1BPI",
    "1BTN",
    "1C52",
    "1C75",
    "1CCR",
    "1CEW",
    "1CHD",
    "1CKA",
    "1CQY",
    "1CRN",
    "1CSE",
    "1CTF",
    "1CUK",
    "1D4T",
    "1D7P",
    "1DDT",
    "1DFN",
    "1DHN",
    "1DIN",
    "1DKZ",
    "1DOZ",
    "1DVR",
    "1E0L",
    "1E2A",
    "1E4M",
    "1E6V",
    "1EAJ",
    "1ECP",
    "1EDM",
    "1EGW",
    "1EJG",
    "1EM8",
]

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


AA_TO_IDX = _DATA_AA_TO_IDX

AA_TO_IDX = _DATA_AA_TO_IDX


def download_representative_pdbs(
    output_dir: str,
    n_structures: int = 300,
    resolution_max: float = 2.5,
    length_min: int = 50,
    length_max: int = 500,
) -> list[str]:
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
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": resolution_max,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                            "operator": "equals",
                            "value": 1,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "greater_or_equal",
                            "value": 1,
                        },
                    },
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
        for attempt in range(3):
            try:
                import requests as req

                r = req.get(pdb_url, timeout=30)
                r.raise_for_status()

                content = r.text
                chain_length = 0
                for line in content.splitlines():
                    if line.startswith("ATOM") and len(line) >= 16 and line[12:16].strip() == "CA":
                        chain_length += 1
                if length_min <= chain_length <= length_max:
                    with open(pdb_path, "w") as f:
                        f.write(content)
                    downloaded.append(pdb_id)
                    break
                else:
                    print(
                        f"  {pdb_id}: chain length {chain_length} outside [{length_min}, {length_max}], skipping"
                    )
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2**attempt)
                else:
                    print(f"  {pdb_id}: download failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"Downloaded {len(downloaded)}/{i + 1} PDB structures...")

    print(f"Total downloaded: {len(downloaded)} PDB structures")
    return downloaded


def generate_sequences_with_proteinmpnn(
    pdb_path: str,
    output_fasta_path: str,
    temperatures: list[float] = None,
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
            check=True,
            capture_output=True,
            timeout=120,
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
                    "python",
                    mpnn_script,
                    "--jsonl_path",
                    jsonl_path,
                    "--chain_id_jsonl",
                    "",
                    "--fixed_positions_jsonl",
                    "",
                    "--sampling_temp",
                    str(T),
                    "--num_seq_per_target",
                    str(num_seq_per_temp),
                    "--batch_size",
                    "1",
                    "--model_name",
                    model_name,
                    "--out_folder",
                    out_subdir,
                    "--seed",
                    str(seed),
                ],
                check=True,
                capture_output=True,
                timeout=600,
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

    os.path.splitext(os.path.basename(pdb_path))[0]
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
    config: SimulationConfig,
) -> dict[str, int]:
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
            pdb_path,
            fa_path,
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


def build_pregen_index(pregen_dir: str) -> dict:
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


def _load_fasta_sequences(fasta_path: str) -> list[str]:
    sequences = []
    current_seq: list[str] = []
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


def _compute_consensus(sequences: list[str]) -> str:
    if not sequences:
        return ""
    L = len(sequences[0])
    consensus = []
    for pos in range(L):
        counts: dict[str, int] = defaultdict(int)
        for seq in sequences:
            aa = seq[pos] if pos < len(seq) else "-"
            if aa in AA_TO_IDX:
                counts[aa] += 1
        if counts:
            max_count = max(counts.values())
            candidates = [aa for aa, c in counts.items() if c == max_count]
            consensus.append(
                candidates[0] if len(candidates) == 1 else np.random.choice(candidates)
            )
        else:
            consensus.append(np.random.choice(list(AA_TO_IDX.keys())))
    return "".join(consensus)


def _evolve_from_root_vectorized(
    tree_newick: str,
    root_seq: np.ndarray,
    model: str = "JTT",
    alpha: float = 1.0,
    n_categories: int = 4,
    seed: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    if seed is not None:
        np.random.seed(seed)

    from .tree_utils import get_leaf_order, parse_newick

    if model == "JTT":
        Q = jtt_rate_matrix()
    elif model == "WAG":
        Q = wag_rate_matrix()
    elif model == "LG":
        Q = lg_rate_matrix()
    else:
        Q = jtt_rate_matrix()

    n_sites = len(root_seq)
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = eigvals.real
    eigvecs_inv = np.linalg.inv(eigvecs)

    rate_multipliers = np.ones(n_sites)
    if alpha < float("inf") and alpha > 0:
        rate_categories = _discrete_gamma_rates(alpha, n_categories)
        categories = np.random.choice(n_categories, size=n_sites, replace=True)
        rate_multipliers = rate_categories[categories]

    leaf_names = get_leaf_order(tree_newick)
    n_leaves = len(leaf_names)
    tree = parse_newick(tree_newick)

    unique_rates = np.unique(rate_multipliers)
    leaf_to_seq = {}

    def _simulate_subtree(node, parent_seq):
        if node.is_leaf:
            leaf_to_seq[node.name] = parent_seq.copy()
            return
        for child in node.children:
            bl = child.branch_length
            if bl <= 0:
                _simulate_subtree(child, parent_seq)
                continue

            child_seq = np.zeros(n_sites, dtype=np.int32)
            for r in unique_rates:
                mask = rate_multipliers == r
                if not mask.any():
                    continue
                diag = np.exp(eigvals * bl * r)
                P = (eigvecs * diag) @ eigvecs_inv
                P = np.real(P)
                P = np.maximum(P, 0)
                P = P / P.sum(axis=1, keepdims=True)

                parent_states = parent_seq[mask]
                probs = P[parent_states]
                cumsum = np.cumsum(probs, axis=1)
                rand = np.random.random(len(parent_states))
                child_states = (rand[:, None] < cumsum).argmax(axis=1)
                child_seq[mask] = child_states

            _simulate_subtree(child, child_seq)

    _simulate_subtree(tree, root_seq)

    msa = np.zeros((n_leaves, n_sites), dtype=np.int32)
    for i, name in enumerate(leaf_names):
        msa[i] = leaf_to_seq.get(name, root_seq.copy())

    return msa, leaf_names


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
    seed: int | None = None,
) -> tuple[torch.Tensor, list[str], str]:
    if seed is not None:
        np.random.seed(seed)

    pool_sequences = _load_fasta_sequences(fasta_pool_path)

    if not pool_sequences:
        tree = simulate_yule_tree(n_leaves, birth_rate, seed)
        msa, names = evolve_sequences(
            tree, n_sites, model, alpha, n_categories, include_indels=False, seed=seed
        )
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
        msa_np, leaf_names = _evolve_from_root_vectorized(
            tree_newick, ancestral_arr, model, alpha, n_categories, seed
        )
        msa = msa_np.astype(np.int64)
    except Exception as e:
        print(f"Vectorized evolution failed: {e}, falling back to naive evolution")
        msa_result, _ = evolve_sequences(
            tree_newick, n_sites, model, alpha, n_categories, include_indels=False, seed=seed
        )
        msa = msa_result.numpy().astype(np.int32)
        leaf_names = [f"seq_{i}" for i in range(n_leaves)]

    has_valid = (msa <= 19).any()
    if not has_valid:
        msa[:, 0] = np.random.randint(0, 20, size=n_leaves)

    return torch.from_numpy(msa.astype(np.int64)), leaf_names, tree_newick


def random_training_example(
    pregen_index: dict,
    n_leaves_range: tuple[int, int] = (50, 500),
    n_sites_range: tuple[int, int] = (200, 1500),
    model: str = "JTT",
    alpha: float = 1.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, list[str], str, str]:
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
        fasta_path,
        n_leaves,
        n_sites,
        model=model,
        alpha=alpha,
        seed=seed,
    )

    return msa, seq_names, tree, backbone_id


def training_batch_generator(
    pregen_index: dict,
    batch_size: int,
    n_leaves_range: tuple[int, int],
    n_sites_range: tuple[int, int],
    model: str = "JTT",
    alpha: float = 1.0,
    infinite: bool = True,
) -> Iterator[dict]:
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

        for msa, _seq_names, tree, backbone_id in items:
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
