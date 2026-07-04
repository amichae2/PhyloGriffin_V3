"""
PhyloGriffin v3 -- Stage F: Refinement Pass.
Local NNI adjustments on the full tree using learned quartet scoring
and branch length prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Set, Optional

from ..config import PhyloGriffinConfig
from ..tree_utils import (
    TreeNode,
    parse_newick,
    tree_to_newick,
    get_leaf_order,
)


def _determine_quartet_topology(
    true_tree: "TreeNode",
    a_set: set,
    b_set: set,
    c_set: set,
    d_set: set,
    leaf_to_idx: dict,
) -> int:
    idx_to_name = {v: k for k, v in leaf_to_idx.items()}

    def _to_names(s):
        return {idx_to_name.get(i, str(i)) for i in s}

    a_names = _to_names(a_set)
    b_names = _to_names(b_set)
    c_names = _to_names(c_set)
    d_names = _to_names(d_set)

    def _get_leaf_set(node: "TreeNode") -> set:
        if node.is_leaf:
            return {node.name}
        leaves = set()
        for child in node.children:
            leaves.update(_get_leaf_set(child))
        return leaves

    def _is_clade(node: "TreeNode", group: set) -> bool:
        if not group:
            return False
        if _get_leaf_set(node) == group:
            return True
        for child in node.children:
            if _is_clade(child, group):
                return True
        return False

    if _is_clade(true_tree, a_names | b_names) or _is_clade(true_tree, c_names | d_names):
        return 0
    if _is_clade(true_tree, a_names | c_names) or _is_clade(true_tree, b_names | d_names):
        return 1
    if _is_clade(true_tree, a_names | d_names) or _is_clade(true_tree, b_names | c_names):
        return 2
    return 0


class QuartetScorer(nn.Module):
    """
    MLP that takes embeddings from 4 subtrees and scores topologies.

    Input: concatenated mean embeddings of the 4 subtrees -> (4*d_model,)
    Hidden: [quartet_hidden, quartet_hidden//2] = [256, 128]
    Output: (3,) - scores for the three quartet topologies:
        T1 = ((A,B),(C,D))
        T2 = ((A,C),(B,D))
        T3 = ((A,D),(B,C))
    """

    def __init__(self, d_model: int, quartet_hidden: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4 * d_model, quartet_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(quartet_hidden, quartet_hidden // 2),
            nn.GELU(),
            nn.Linear(quartet_hidden // 2, 3),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        emb_c: torch.Tensor,
        emb_d: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([emb_a, emb_b, emb_c, emb_d], dim=-1)
        return self.layers(x)


class BranchLengthPredictor(nn.Module):
    """
    MLP that predicts branch length from two subtree embeddings.

    Input: concat[emb_subtree_a, emb_subtree_b] -> (2*d_model,)
    Output: scalar (positive, via softplus)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb_a, emb_b], dim=-1)
        return self.layers(x).squeeze(-1)


class RefinementPass(nn.Module):
    """
    Stage F: Local NNI adjustments on the full tree.

    Iterates over internal nodes that form quartet configurations,
    scores alternative topologies with QuartetScorer, and applies
    NNI swaps when a better topology is found.
    """

    def __init__(self, config: PhyloGriffinConfig):
        super().__init__()
        self.config = config
        self.quartet_scorer = QuartetScorer(
            d_model=config.griffin.d_model,
            quartet_hidden=config.refinement.quartet_hidden,
        )
        self.branch_predictor = BranchLengthPredictor(
            d_model=config.griffin.d_model,
        )

    def forward(self, tree_newick: str, seq_embeddings: torch.Tensor):
        tree = parse_newick(tree_newick)
        leaf_names = get_leaf_order(tree_to_newick(tree))
        leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}

        quartet_scores_list = []
        quartet_metadata_list = []

        for _ in range(self.config.refinement.n_rounds):
            quartets = self._get_internal_nodes(tree)

            for node, c1, c2, a_set, b_set, c_set, d_set in quartets:
                if not a_set or not b_set or not c_set or not d_set:
                    continue

                a_indices = torch.tensor(sorted(a_set), device=seq_embeddings.device)
                b_indices = torch.tensor(sorted(b_set), device=seq_embeddings.device)
                c_indices = torch.tensor(sorted(c_set), device=seq_embeddings.device)
                d_indices = torch.tensor(sorted(d_set), device=seq_embeddings.device)

                emb_a = seq_embeddings[a_indices].mean(dim=0)
                emb_b = seq_embeddings[b_indices].mean(dim=0)
                emb_c = seq_embeddings[c_indices].mean(dim=0)
                emb_d = seq_embeddings[d_indices].mean(dim=0)

                scores = self.quartet_scorer(emb_a, emb_b, emb_c, emb_d)
                quartet_scores_list.append(scores if self.training else scores.detach())
                quartet_metadata_list.append((a_set, b_set, c_set, d_set))
                best_idx = scores.argmax(dim=-1).item()

                if best_idx != 0 and scores[best_idx] > scores[0] + self.config.refinement.nni_margin:
                    self._apply_nni_swap(node, c1, c2, best_idx)

        intermediates = {
            "quartet_scores": torch.stack(quartet_scores_list) if quartet_scores_list else torch.zeros(0, 3, device=seq_embeddings.device),
            "quartet_metadata": quartet_metadata_list,
            "leaf_to_idx": leaf_to_idx,
        }
        return tree_to_newick(tree), intermediates

    def _score_quartet(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        emb_c: torch.Tensor,
        emb_d: torch.Tensor,
    ) -> torch.Tensor:
        return self.quartet_scorer(emb_a, emb_b, emb_c, emb_d)

    def _get_internal_nodes(self, tree: TreeNode) -> List[Tuple]:
        leaf_names = get_leaf_order(tree_to_newick(tree))
        leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}

        def _get_leaves(node: TreeNode) -> Set[int]:
            if node.is_leaf:
                if node.name in leaf_to_idx:
                    return {leaf_to_idx[node.name]}
                return set()
            leaves = set()
            for child in node.children:
                leaves.update(_get_leaves(child))
            return leaves

        results = []

        def _find_quartets(node: TreeNode):
            if node.is_leaf:
                return
            if len(node.children) == 2:
                c1, c2 = node.children
                if (
                    not c1.is_leaf
                    and not c2.is_leaf
                    and len(c1.children) == 2
                    and len(c2.children) == 2
                ):
                    A = _get_leaves(c1.children[0])
                    B = _get_leaves(c1.children[1])
                    C = _get_leaves(c2.children[0])
                    D = _get_leaves(c2.children[1])
                    if A and B and C and D:
                        results.append((node, c1, c2, A, B, C, D))
            for child in node.children:
                _find_quartets(child)

        _find_quartets(tree)
        return results

    def _apply_nni_swap(
        self,
        node: TreeNode,
        c1: TreeNode,
        c2: TreeNode,
        alternative: int,
    ):
        if alternative == 1:
            b_node = c1.children[1]
            c_node = c2.children[0]
            c1.children[1] = c_node
            c_node.parent = c1
            c2.children[0] = b_node
            b_node.parent = c2
        elif alternative == 2:
            b_node = c1.children[1]
            d_node = c2.children[1]
            c1.children[1] = d_node
            d_node.parent = c1
            c2.children[1] = b_node
            b_node.parent = c2

    def compute_loss(self, intermediates, true_tree_newick, seq_embeddings, device):
        quartet_scores = intermediates["quartet_scores"]
        if quartet_scores.numel() == 0:
            return torch.zeros(1, device=device, requires_grad=True)

        quartet_metadata = intermediates.get("quartet_metadata", [])
        leaf_to_idx = intermediates.get("leaf_to_idx", {})
        if not quartet_metadata or len(quartet_metadata) != quartet_scores.shape[0]:
            target = torch.zeros_like(quartet_scores)
            target[:, 0] = 1.0
            return F.cross_entropy(quartet_scores, target.argmax(dim=-1))

        try:
            true_tree = parse_newick(true_tree_newick)
        except Exception:
            target = torch.zeros_like(quartet_scores)
            target[:, 0] = 1.0
            return F.cross_entropy(quartet_scores, target.argmax(dim=-1))

        targets = []
        for a_set, b_set, c_set, d_set in quartet_metadata:
            targets.append(_determine_quartet_topology(
                true_tree, a_set, b_set, c_set, d_set, leaf_to_idx
            ))

        target = torch.tensor(targets, dtype=torch.long, device=quartet_scores.device)
        return F.cross_entropy(quartet_scores, target)
