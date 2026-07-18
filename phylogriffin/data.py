"""
PhyloGriffin v3 -- Data loading and dataset classes.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

AA_TO_IDX: dict[str, int] = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}

IDX_TO_AA: dict[int, str] = {v: k for k, v in AA_TO_IDX.items()}

NT_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_NT: dict[int, str] = {v: k for k, v in NT_TO_IDX.items()}


def _get_alphabet_maps(alphabet: str) -> tuple[dict[str, int], dict[int, str], int, int]:
    if alphabet == "protein":
        return AA_TO_IDX, IDX_TO_AA, 20, 21
    elif alphabet == "nucleotide":
        return NT_TO_IDX, IDX_TO_NT, 4, 5
    else:
        raise ValueError(f"Unknown alphabet: {alphabet}")


def _tokenize_sequence(seq: str, char_to_idx: dict[str, int], gap_idx: int) -> list[int]:
    tokens = []
    for ch in seq.upper():
        if ch in ("-", "."):
            tokens.append(gap_idx)
        elif ch in char_to_idx:
            tokens.append(char_to_idx[ch])
        else:
            tokens.append(gap_idx)
    return tokens


def load_fasta(path: str, alphabet: str = "protein") -> tuple[torch.Tensor, list[str]]:
    char_to_idx, _, gap_idx, pad_idx = _get_alphabet_maps(alphabet)
    names = []
    sequences = []
    current_name = ""
    current_seq: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name:
                    names.append(current_name)
                    sequences.append("".join(current_seq))
                    current_seq = []
                current_name = line[1:].strip()
            else:
                current_seq.append(line)

    if current_name:
        names.append(current_name)
        sequences.append("".join(current_seq))

    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    lengths = [len(s) for s in sequences]
    if len(set(lengths)) > 1:
        raise ValueError(f"Sequences must have equal length. Got lengths: {sorted(set(lengths))}")

    L = lengths[0]
    N = len(sequences)
    msa = torch.full((N, L), pad_idx, dtype=torch.long)

    for i, seq in enumerate(sequences):
        tokens = _tokenize_sequence(seq, char_to_idx, gap_idx)
        for j, tok in enumerate(tokens):
            msa[i, j] = tok

    return msa, names


def load_phylip(path: str, alphabet: str = "protein") -> tuple[torch.Tensor, list[str]]:
    char_to_idx, _, gap_idx, pad_idx = _get_alphabet_maps(alphabet)

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid phylip header: {lines[0]}")
    n_taxa = int(header[0])
    n_sites = int(header[1])

    data_lines = lines[1:]
    seq_names = []
    seq_dict: dict[str, list[str]] = {}

    i = 0
    while i < len(data_lines):
        parts = data_lines[i].split()
        if parts and (parts[0].isdigit() and len(parts) == 1):
            i += 1
            continue
        if parts:
            name = parts[0]
            seq_part = parts[1] if len(parts) > 1 else ""
            if name not in seq_dict:
                seq_names.append(name)
                seq_dict[name] = []
            seq_dict[name].append(seq_part)
        i += 1

    for name in seq_names:
        full_seq = "".join(seq_dict[name])
        if len(full_seq) != n_sites:
            raise ValueError(f"Sequence {name} has length {len(full_seq)}, expected {n_sites}")

    msa = torch.full((n_taxa, n_sites), pad_idx, dtype=torch.long)
    for i, name in enumerate(seq_names):
        tokens = _tokenize_sequence("".join(seq_dict[name]), char_to_idx, gap_idx)
        for j, tok in enumerate(tokens):
            msa[i, j] = tok

    return msa, seq_names


def load_msa(path: str, alphabet: str = "protein") -> tuple[torch.Tensor, list[str]]:
    with open(path) as f:
        first_line = f.readline().strip()
    if first_line and not first_line.startswith(">"):
        parts = first_line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return load_phylip(path, alphabet)
    return load_fasta(path, alphabet)


class PreGeneratedDataset(Dataset):
    """
    Dataset that generates structurally-realistic MSAs on-the-fly.

    Each __getitem__ call generates a fresh MSA by:
    1. Randomly selecting a pre-generated ProteinMPNN sequence pool
    2. Sampling N leaves from the pool
    3. Running Pyvolve evolution along a random tree
    """

    def __init__(
        self,
        pregen_dir: str,
        n_examples_per_epoch: int = 5000,
        n_leaves_range: tuple[int, int] = (50, 500),
        n_sites_range: tuple[int, int] = (200, 1500),
        model: str = "JTT",
        alpha: float = 1.0,
        include_trees: bool = True,
        split: str = "train",
        train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        self.pregen_dir = pregen_dir
        self.n_examples_per_epoch = n_examples_per_epoch
        self.n_leaves_range = n_leaves_range
        self.n_sites_range = n_sites_range
        self.model = model
        self.alpha = alpha
        self.include_trees = include_trees
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        from .simulation import build_pregen_index

        self.pregen_index = build_pregen_index(pregen_dir)

    def __len__(self) -> int:
        return self.n_examples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        from .simulation import random_training_example

        seed = self.seed + idx if self.seed is not None else None
        n_leaves = int(self._rng.integers(self.n_leaves_range[0], self.n_leaves_range[1] + 1))
        n_sites = int(self._rng.integers(self.n_sites_range[0], self.n_sites_range[1] + 1))

        msa, seq_names, tree_newick, backbone_id = random_training_example(
            self.pregen_index,
            n_leaves_range=(n_leaves, n_leaves),
            n_sites_range=(n_sites, n_sites),
            model=self.model,
            alpha=self.alpha,
            seed=seed,
        )

        result = {"msa": msa, "mask": (msa != 21).bool()}
        if self.include_trees and tree_newick:
            from .tree_utils import patristic_distances

            pairwise_distances = patristic_distances(tree_newick, msa.shape[0])
            result["tree_newick"] = tree_newick
            result["leaf_names"] = seq_names
            result["pairwise_distances"] = torch.from_numpy(pairwise_distances).float()
            result["backbone_id"] = backbone_id

        return result


class PreGeneratedSubproblemDataset(PreGeneratedDataset):
    def __init__(self, *args, max_leaves: int = 1500, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_leaves = max_leaves

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        msa = result["msa"]
        mask = result["mask"]
        N = msa.shape[0]
        if N > self.max_leaves:
            indices = torch.randperm(N)[: self.max_leaves]
            msa = msa[indices]
            mask = mask[indices]
        else:
            indices = torch.arange(N)
        return {
            "msa": msa,
            "mask": mask,
            "true_tree": result.get("tree_newick", ""),
            "leaf_indices": indices,
        }


class PreGeneratedDecomposedDataset(PreGeneratedDataset):
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        tree = result.get("tree_newick", "")
        return {
            "msa": result["msa"],
            "mask": result["mask"],
            "true_tree": tree,
            "subproblems": [],
            "guide_tree": tree,
        }


class PreGeneratedErrorTreeDataset(PreGeneratedDataset):
    def __init__(self, *args, d_model: int = 256, n_nni_swaps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.n_nni_swaps = n_nni_swaps

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        true_tree = result.get("tree_newick", "")
        from .tree_utils import corrupt_tree

        corrupted = corrupt_tree(true_tree, n_swaps=self.n_nni_swaps, seed=idx)
        msa = result["msa"]
        return {
            "corrupted_tree": corrupted,
            "true_tree": true_tree,
            "msa": msa,
            "mask": result["mask"],
            "embeddings": torch.zeros(msa.shape[0], self.d_model),
        }


class _MSATreeDataset(Dataset):
    def __init__(self, msa_dir, tree_dir, alphabet="protein"):
        self.msa_dir = msa_dir
        self.tree_dir = tree_dir
        self.alphabet = alphabet
        self.files = sorted(
            [
                f
                for f in os.listdir(msa_dir)
                if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
            ]
        )

    def __len__(self):
        return len(self.files)

    def _load_msa_and_tree(self, idx):
        base = os.path.splitext(self.files[idx])[0]
        base = base.replace("msa_", "")
        msa_path = os.path.join(self.msa_dir, self.files[idx])
        tree_path = os.path.join(self.tree_dir, f"tree_{base}.nwk")
        msa, leaf_names = load_msa(msa_path, self.alphabet)
        mask = (msa != 21).bool()
        with open(tree_path) as f:
            tree_newick = f.read().strip()
        return msa, mask, leaf_names, tree_newick


class MSADataset(Dataset):
    def __init__(self, msa_dir: str, alphabet: str = "protein", max_seq_len: int = 2048):
        self.msa_dir = msa_dir
        self.alphabet = alphabet
        self.max_seq_len = max_seq_len
        self.files = sorted(
            [
                f
                for f in os.listdir(msa_dir)
                if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = os.path.join(self.msa_dir, self.files[idx])
        msa, _ = load_msa(path, self.alphabet)
        if msa.shape[1] > self.max_seq_len:
            start = torch.randint(0, msa.shape[1] - self.max_seq_len + 1, (1,)).item()
            msa = msa[:, start : start + self.max_seq_len]
        mask = (msa != 21).bool()
        return {"msa": msa, "mask": mask}


class ContrastiveDataset(_MSATreeDataset):
    def __getitem__(self, idx: int) -> dict:
        msa, mask, leaf_names, tree_newick = self._load_msa_and_tree(idx)
        from .tree_utils import patristic_distances

        pairwise_distances = patristic_distances(tree_newick, msa.shape[0])
        return {
            "msa": msa,
            "mask": mask,
            "tree_newick": tree_newick,
            "leaf_names": leaf_names,
            "pairwise_distances": torch.from_numpy(pairwise_distances).float(),
        }


class GraphDataset(_MSATreeDataset):
    def __getitem__(self, idx: int) -> dict:
        msa, mask, leaf_names, tree_newick = self._load_msa_and_tree(idx)
        from .tree_utils import newick_to_splits, patristic_distances

        pairwise_distances = patristic_distances(tree_newick, msa.shape[0])
        splits = newick_to_splits(tree_newick, msa.shape[0])
        return {
            "msa": msa,
            "mask": mask,
            "tree_newick": tree_newick,
            "leaf_names": leaf_names,
            "pairwise_distances": torch.from_numpy(pairwise_distances).float(),
            "splits": splits,
        }


class SubproblemDataset(_MSATreeDataset):
    def __init__(
        self, msa_dir: str, tree_dir: str, max_leaves: int = 1500, alphabet: str = "protein"
    ):
        super().__init__(msa_dir, tree_dir, alphabet)
        self.max_leaves = max_leaves

    def __getitem__(self, idx: int) -> dict:
        msa, mask, _, tree_newick = self._load_msa_and_tree(idx)
        N = msa.shape[0]
        if N > self.max_leaves:
            indices = torch.randperm(N)[: self.max_leaves].tolist()
            msa = msa[indices]
            mask = mask[indices]
        else:
            indices = list(range(N))
        return {
            "sub_msa": msa,
            "sub_mask": mask,
            "true_tree": tree_newick,
            "leaf_indices": torch.tensor(indices, dtype=torch.long),
        }


class DecomposedTreeDataset(_MSATreeDataset):
    def __getitem__(self, idx: int) -> dict:
        msa, mask, leaf_names, tree_newick = self._load_msa_and_tree(idx)
        from .tree_utils import get_leaf_order

        leaf_order = get_leaf_order(tree_newick)
        n_leaves = len(leaf_order)

        if n_leaves <= 500:
            subproblems = [(list(range(n_leaves)), msa, None)]
            guide_tree = tree_newick
        else:
            subproblems = [(list(range(n_leaves)), msa, None)]
            guide_tree = tree_newick

        return {
            "msa": msa,
            "mask": mask,
            "true_tree": tree_newick,
            "subproblems": subproblems,
            "guide_tree": guide_tree,
            "leaf_names": leaf_names,
        }


class ErrorTreeDataset(_MSATreeDataset):
    def __init__(self, msa_dir, tree_dir, alphabet="protein", n_nni_swaps=3):
        super().__init__(msa_dir, tree_dir, alphabet)
        self.n_nni_swaps = n_nni_swaps

    def __getitem__(self, idx: int) -> dict:
        msa, mask, _, true_tree = self._load_msa_and_tree(idx)
        from .tree_utils import corrupt_tree

        corrupted = corrupt_tree(true_tree, n_swaps=self.n_nni_swaps, seed=idx)
        return {
            "corrupted_tree": corrupted,
            "true_tree": true_tree,
            "msa": msa,
            "mask": mask,
            "embeddings": torch.zeros(msa.shape[0], 256),
        }


class MaxTokensCollator:
    def __init__(self, config, truncate_sites: bool = True):
        self.pad_idx = config.pad_idx
        self.max_tokens = config.training.max_tokens_per_batch
        self.truncate_sites = truncate_sites

    def __call__(self, batch):
        truncated_batch = []
        for item in batch:
            msa = item["msa"]
            mask = item["mask"]
            n, sites = msa.shape
            tokens = n * sites
            if tokens > self.max_tokens and self.truncate_sites and n > 0:
                max_sites = max(1, self.max_tokens // n)
                if max_sites < sites:
                    msa = msa[:, :max_sites].contiguous()
                    mask = mask[:, :max_sites].contiguous()
            truncated_batch.append({**item, "msa": msa, "mask": mask})

        truncated_batch = sorted(
            truncated_batch,
            key=lambda item: item["msa"].shape[0] * item["msa"].shape[1],
            reverse=True,
        )
        kept = []
        total = 0
        max_n = 0
        max_sites = 0
        for item in truncated_batch:
            n, sites = item["msa"].shape
            tokens = n * sites
            if total + tokens > self.max_tokens and kept:
                break
            kept.append(item)
            total += tokens
            max_n = max(max_n, n)
            max_sites = max(max_sites, sites)

        padded_msa = []
        padded_mask = []
        padded_extra: dict = {}
        for item in kept:
            n, sites = item["msa"].shape
            msa_pad = torch.full((max_n, max_sites), self.pad_idx, dtype=torch.long)
            mask_pad = torch.zeros(max_n, max_sites, dtype=torch.bool)
            msa_pad[:n, :sites] = item["msa"]
            mask_pad[:n, :sites] = item["mask"]
            padded_msa.append(msa_pad)
            padded_mask.append(mask_pad)
            for k, v in item.items():
                if k not in ("msa", "mask"):
                    padded_extra.setdefault(k, []).append(v)

        out = {"msa": torch.stack(padded_msa), "mask": torch.stack(padded_mask)}

        for k, v_list in padded_extra.items():
            if v_list and isinstance(v_list[0], torch.Tensor):
                shapes_match = all(t.shape == v_list[0].shape for t in v_list[1:])
                if shapes_match:
                    out[k] = torch.stack(v_list)
                    continue
                padded_n = max_n
                first = v_list[0]
                if first.ndim == 2 and first.shape[0] == first.shape[1]:
                    padded = torch.zeros(
                        len(v_list), padded_n, padded_n, dtype=first.dtype, device=first.device
                    )
                    for bi, t in enumerate(v_list):
                        tn = t.shape[0]
                        padded[bi, :tn, :tn] = t
                elif first.ndim == 2:
                    D = first.shape[1]
                    padded = torch.zeros(
                        len(v_list), padded_n, D, dtype=first.dtype, device=first.device
                    )
                    for bi, t in enumerate(v_list):
                        tn = t.shape[0]
                        padded[bi, :tn] = t
                elif first.ndim == 1:
                    padded = torch.zeros(
                        len(v_list), padded_n, dtype=first.dtype, device=first.device
                    )
                    for bi, t in enumerate(v_list):
                        tn = t.shape[0]
                        padded[bi, :tn] = t
                else:
                    extra_dims = first.shape[1:]
                    padded = torch.zeros(
                        len(v_list), padded_n, *extra_dims, dtype=first.dtype, device=first.device
                    )
                    for bi, t in enumerate(v_list):
                        tn = t.shape[0]
                        padded[bi, :tn] = t
                out[k] = padded
            else:
                out[k] = v_list

        return out
