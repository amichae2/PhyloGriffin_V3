"""
PhyloGriffin v3 -- Data loading and dataset classes.
"""

import os
import warnings
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

from .config import PhyloGriffinConfig

AA_TO_IDX: Dict[str, int] = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

IDX_TO_AA: Dict[int, str] = {v: k for k, v in AA_TO_IDX.items()}

NT_TO_IDX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_NT: Dict[int, str] = {v: k for k, v in NT_TO_IDX.items()}


def _get_alphabet_maps(alphabet: str) -> Tuple[Dict[str, int], Dict[int, str], int, int]:
    if alphabet == "protein":
        return AA_TO_IDX, IDX_TO_AA, 20, 21
    elif alphabet == "nucleotide":
        return NT_TO_IDX, IDX_TO_NT, 4, 5
    else:
        raise ValueError(f"Unknown alphabet: {alphabet}")


def _tokenize_sequence(seq: str, char_to_idx: Dict[str, int], gap_idx: int) -> List[int]:
    tokens = []
    for ch in seq.upper():
        if ch in ("-", "."):
            tokens.append(gap_idx)
        elif ch in char_to_idx:
            tokens.append(char_to_idx[ch])
        else:
            tokens.append(gap_idx)
    return tokens


def load_fasta(path: str, alphabet: str = "protein") -> Tuple[torch.Tensor, List[str]]:
    char_to_idx, _, gap_idx, pad_idx = _get_alphabet_maps(alphabet)
    names = []
    sequences = []
    current_name = ""
    current_seq: List[str] = []

    with open(path, "r") as f:
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


def load_phylip(path: str, alphabet: str = "protein") -> Tuple[torch.Tensor, List[str]]:
    char_to_idx, _, gap_idx, pad_idx = _get_alphabet_maps(alphabet)

    with open(path, "r") as f:
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
    seq_dict: Dict[str, List[str]] = {}

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


def load_msa(path: str, alphabet: str = "protein") -> Tuple[torch.Tensor, List[str]]:
    with open(path, "r") as f:
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
        n_leaves_range: Tuple[int, int] = (50, 500),
        n_sites_range: Tuple[int, int] = (200, 1500),
        model: str = "JTT",
        alpha: float = 1.0,
        include_trees: bool = True,
        split: str = "train",
        train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        self.pregen_dir = pregen_dir
        self.n_examples_per_epoch = n_examples_per_epoch
        self.n_leaves_range = n_leaves_range
        self.n_sites_range = n_sites_range
        self.model = model
        self.alpha = alpha
        self.include_trees = include_trees
        self.split = split
        self.train_val_test = train_val_test
        self.seed = seed

        from .simulation import build_pregen_index
        self.pregen_index = build_pregen_index(pregen_dir)

    def __len__(self) -> int:
        return self.n_examples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        from .simulation import random_training_example

        seed = self.seed + idx if self.seed is not None else None
        n_leaves = np.random.randint(self.n_leaves_range[0], self.n_leaves_range[1] + 1)
        n_sites = np.random.randint(self.n_sites_range[0], self.n_sites_range[1] + 1)

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


class MSADataset(Dataset):
    def __init__(self, msa_dir: str, alphabet: str = "protein", max_seq_len: int = 2048):
        self.msa_dir = msa_dir
        self.alphabet = alphabet
        self.max_seq_len = max_seq_len
        self.files = sorted([
            f for f in os.listdir(msa_dir)
            if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = os.path.join(self.msa_dir, self.files[idx])
        msa, _ = load_msa(path, self.alphabet)
        if msa.shape[1] > self.max_seq_len:
            start = torch.randint(0, msa.shape[1] - self.max_seq_len + 1, (1,)).item()
            msa = msa[:, start:start + self.max_seq_len]
        mask = (msa != 21).bool()
        return {"msa": msa, "mask": mask}


class ContrastiveDataset(Dataset):
    def __init__(self, msa_dir: str, tree_dir: str, alphabet: str = "protein"):
        self.msa_dir = msa_dir
        self.tree_dir = tree_dir
        self.alphabet = alphabet
        self.files = sorted([
            f for f in os.listdir(msa_dir)
            if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        base = os.path.splitext(self.files[idx])[0]
        base = base.replace("msa_", "")
        msa_path = os.path.join(self.msa_dir, self.files[idx])
        tree_path = os.path.join(self.tree_dir, f"tree_{base}.nwk")

        msa, leaf_names = load_msa(msa_path, self.alphabet)
        mask = (msa != 21).bool()

        with open(tree_path, "r") as f:
            tree_newick = f.read().strip()

        from .tree_utils import patristic_distances
        pairwise_distances = patristic_distances(tree_newick, msa.shape[0])

        return {
            "msa": msa,
            "mask": mask,
            "tree_newick": tree_newick,
            "leaf_names": leaf_names,
            "pairwise_distances": torch.from_numpy(pairwise_distances).float(),
        }


class GraphDataset(Dataset):
    def __init__(self, msa_dir: str, tree_dir: str, alphabet: str = "protein"):
        self.msa_dir = msa_dir
        self.tree_dir = tree_dir
        self.alphabet = alphabet
        self.files = sorted([
            f for f in os.listdir(msa_dir)
            if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        base = os.path.splitext(self.files[idx])[0]
        base = base.replace("msa_", "")
        msa_path = os.path.join(self.msa_dir, self.files[idx])
        tree_path = os.path.join(self.tree_dir, f"tree_{base}.nwk")

        msa, leaf_names = load_msa(msa_path, self.alphabet)
        mask = (msa != 21).bool()

        with open(tree_path, "r") as f:
            tree_newick = f.read().strip()

        from .tree_utils import patristic_distances, newick_to_splits

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


class SubproblemDataset(Dataset):
    def __init__(self, msa_dir: str, tree_dir: str, max_leaves: int = 1500,
                 alphabet: str = "protein"):
        self.msa_dir = msa_dir
        self.tree_dir = tree_dir
        self.max_leaves = max_leaves
        self.alphabet = alphabet
        self.files = sorted([
            f for f in os.listdir(msa_dir)
            if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        base = os.path.splitext(self.files[idx])[0]
        base = base.replace("msa_", "")
        msa_path = os.path.join(self.msa_dir, self.files[idx])
        tree_path = os.path.join(self.tree_dir, f"tree_{base}.nwk")

        msa, leaf_names = load_msa(msa_path, self.alphabet)
        mask = (msa != 21).bool()

        with open(tree_path, "r") as f:
            tree_newick = f.read().strip()

        N = msa.shape[0]
        if N > self.max_leaves:
            indices = torch.randperm(N)[:self.max_leaves].tolist()
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


class DecomposedTreeDataset(Dataset):
    def __init__(self, msa_dir: str, tree_dir: str, alphabet: str = "protein"):
        self.msa_dir = msa_dir
        self.tree_dir = tree_dir
        self.alphabet = alphabet
        self.files = sorted([
            f for f in os.listdir(msa_dir)
            if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        base = os.path.splitext(self.files[idx])[0]
        base = base.replace("msa_", "")
        msa_path = os.path.join(self.msa_dir, self.files[idx])
        tree_path = os.path.join(self.tree_dir, f"tree_{base}.nwk")

        msa, leaf_names = load_msa(msa_path, self.alphabet)
        mask = (msa != 21).bool()

        with open(tree_path, "r") as f:
            tree_newick = f.read().strip()

        from .tree_utils import parse_newick, get_leaf_order

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


class ErrorTreeDataset(Dataset):
    def __init__(self, msa_dir: str, tree_dir: str, alphabet: str = "protein"):
        self.msa_dir = msa_dir
        self.tree_dir = tree_dir
        self.alphabet = alphabet
        self.files = sorted([
            f for f in os.listdir(msa_dir)
            if f.endswith(".fa") or f.endswith(".fasta") or f.endswith(".aln")
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        from .tree_utils import patristic_distances

        base = os.path.splitext(self.files[idx])[0]
        base = base.replace("msa_", "")
        msa_path = os.path.join(self.msa_dir, self.files[idx])
        tree_path = os.path.join(self.tree_dir, f"tree_{base}.nwk")

        msa, _ = load_msa(msa_path, self.alphabet)
        mask = (msa != 21).bool()

        with open(tree_path, "r") as f:
            true_tree = f.read().strip()

        return {
            "corrupted_tree": true_tree,
            "true_tree": true_tree,
            "msa": msa,
            "mask": mask,
            "embeddings": torch.zeros(msa.shape[0], 256),
        }
