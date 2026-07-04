"""
PhyloGriffin v3 -- Stage E: Learned Supertree Reconciler.
Combines K subtrees into one global tree using a Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from ..config import PhyloGriffinConfig
from ..tree_utils import (
    parse_newick, tree_to_newick, newick_to_splits,
    get_leaf_order, splits_to_newick, TreeNode
)

D_TREE = 256
D_AFFINITY = 64


def _sinusoidal_encoding(depths: torch.Tensor, d_model: int) -> torch.Tensor:
    K = depths.shape[0]
    device = depths.device
    pe = torch.zeros(K, d_model, device=device)
    position = depths.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _compute_guide_depths(guide_tree_newick: str, K: int) -> torch.Tensor:
    tree = parse_newick(guide_tree_newick)
    depths: Dict[int, int] = {}

    def _dfs(node: TreeNode, depth: int):
        if node.is_leaf:
            depths[len(depths)] = depth
        for child in node.children:
            _dfs(child, depth + 1)

    _dfs(tree, 0)
    return torch.tensor([depths.get(i, 0) for i in range(K)], dtype=torch.float32)


def _guide_adjacency_mask(guide_tree_newick: str, K: int) -> Optional[torch.Tensor]:
    return None


class TreeLSTM(nn.Module):

    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden

        self.leaf_encoder = nn.Linear(1, d_input)

        self.W_i = nn.Linear(d_input, d_hidden)
        self.U_i = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_f = nn.Linear(d_input, d_hidden)
        self.U_f = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_input, d_hidden)
        self.U_o = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_u = nn.Linear(d_input, d_hidden)
        self.U_u = nn.Linear(d_hidden, d_hidden, bias=False)

    def forward(self, tree: TreeNode) -> torch.Tensor:
        h, _, _ = self._forward_recursive(tree)
        return h

    def _forward_recursive(self, node: TreeNode) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        if node.is_leaf:
            bl = torch.tensor([[node.branch_length]], dtype=torch.float32, device=device)
            x = self.leaf_encoder(bl).squeeze(0)
            h = torch.zeros(self.d_hidden, device=device)
            c = torch.zeros(self.d_hidden, device=device)
            return h, c, x

        child_results = [self._forward_recursive(c) for c in node.children]
        child_h = torch.stack([r[0] for r in child_results])
        child_c = torch.stack([r[1] for r in child_results])

        n_children = child_h.shape[0]
        if n_children == 0:
            h = torch.zeros(self.d_hidden, device=device)
            c = torch.zeros(self.d_hidden, device=device)
            return h, c, torch.zeros(self.d_input, device=device)

        h_sum = child_h.sum(dim=0)

        x = torch.zeros(self.d_input, device=device)

        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))
        f = torch.sigmoid(self.W_f(x).unsqueeze(0) + self.U_f(child_h))

        c = i * u + (f * child_c).sum(dim=0)
        h = o * torch.tanh(c)

        return h, c, x


class TransformerLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_in = x.unsqueeze(0)
        attn_out, _ = self.self_attn(x_in, x_in, x_in, attn_mask=mask)
        x = x + attn_out.squeeze(0)
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm(x)
        return x


class SupertreeReconciler(nn.Module):
    """
    Stage E: Combines K subtrees into one global tree.

    Uses a Transformer over K subproblem tokens, guided by a
    guide-tree positional encoding and adjacency mask, to predict
    branch-length scaling, cross-subproblem affinities, and per-leaf
    stay-or-move probabilities.  A non-learned reconciliation step
    assembles the final global Newick tree.
    """

    def __init__(self, config: PhyloGriffinConfig):
        super().__init__()
        sc = config.supertree
        gc = config.griffin

        self.d_supertree = sc.d_model
        self.d_tree = D_TREE
        self.d_affinity = D_AFFINITY
        self.d_leaf = self.d_supertree
        self.max_subtrees = sc.max_subtrees
        self.half_dim = self.d_supertree // 2

        self.sub_emb_proj = nn.Linear(gc.d_model, self.half_dim)
        self.tree_enc_proj = nn.Linear(self.d_tree, self.half_dim)
        self.token_proj = nn.Linear(self.d_supertree + 2, self.d_supertree)

        self.leaf_feature_proj = nn.Linear(gc.d_model, self.d_leaf)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.d_supertree, sc.n_heads, sc.d_feedforward, sc.dropout)
            for _ in range(sc.n_layers)
        ])

        self.scale_mlp = nn.Sequential(
            nn.Linear(self.d_supertree, self.d_supertree // 2),
            nn.GELU(),
            nn.Dropout(sc.dropout),
            nn.Linear(self.d_supertree // 2, 1),
        )

        self.affinity_proj_a = nn.Linear(self.d_supertree, self.d_affinity)
        self.affinity_proj_b = nn.Linear(self.d_supertree, self.d_affinity)
        self.affinity_scorer = nn.Linear(self.d_affinity, 1)

        self.stay_mlp = nn.Sequential(
            nn.Linear(self.d_supertree + self.d_leaf, self.d_supertree // 2),
            nn.GELU(),
            nn.Dropout(sc.dropout),
            nn.Linear(self.d_supertree // 2, 1),
        )

        self.tree_lstm = TreeLSTM(d_input=1, d_hidden=self.d_tree)

    def forward(
        self,
        subtrees: List[Tuple[torch.Tensor, str]],
        guide_tree_newick: str,
        global_embeddings: torch.Tensor,
    ) -> str:
        K = len(subtrees)
        device = global_embeddings.device
        d_model = global_embeddings.shape[1]

        leaf_features = self._compute_leaf_features(subtrees, global_embeddings)

        guide_depths = _compute_guide_depths(guide_tree_newick, K).to(device)
        pos_enc = _sinusoidal_encoding(guide_depths, self.d_supertree)

        max_size = max(len(indices) for indices, _ in subtrees)

        tokens = []
        for k, (leaf_indices, subtree_newick) in enumerate(subtrees):
            sub_emb = global_embeddings[leaf_indices].mean(dim=0)
            tree_enc = self._encode_tree_structure(subtree_newick)
            quality = self._estimate_quality(subtree_newick, sub_emb)
            size_norm = torch.tensor(len(leaf_indices) / max(max_size, 1),
                                     dtype=torch.float32, device=device)

            emb_proj = self.sub_emb_proj(sub_emb)
            tree_proj = self.tree_enc_proj(tree_enc)
            combined = torch.cat([
                emb_proj, tree_proj,
                quality.unsqueeze(0),
                size_norm.unsqueeze(0),
            ])
            token = self.token_proj(combined)
            tokens.append(token)

        x = torch.stack(tokens, dim=0)
        x = x + pos_enc

        mask = _guide_adjacency_mask(guide_tree_newick, K)
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)

        branch_scales = torch.exp(self.scale_mlp(x)).squeeze(-1)

        aff_a = self.affinity_proj_a(x)
        aff_b = self.affinity_proj_b(x)
        affinity = aff_a.unsqueeze(1) * aff_b.unsqueeze(0)
        affinity_scores = self.affinity_scorer(affinity).squeeze(-1)

        stay_probs = torch.zeros(global_embeddings.shape[0], device=device)
        for k, (leaf_indices, _) in enumerate(subtrees):
            for local_idx, global_idx in enumerate(leaf_indices):
                gi = global_idx.item()
                lf = leaf_features[gi]
                si = torch.cat([x[k], lf])
                stay_probs[gi] = torch.sigmoid(self.stay_mlp(si)).squeeze()

        global_tree_newick = self._reconcile(
            subtrees, branch_scales, stay_probs, affinity_scores, guide_tree_newick
        )
        return global_tree_newick

    def _encode_subtree(self, subtree_newick: str, n_leaves: int) -> torch.Tensor:
        tree = parse_newick(subtree_newick)
        return self.tree_lstm(tree)

    def _encode_tree_structure(self, subtree_newick: str) -> torch.Tensor:
        device = next(self.parameters()).device
        leaves = get_leaf_order(subtree_newick)
        n_leaves = len(leaves)
        splits = newick_to_splits(subtree_newick, n_leaves)

        if not splits:
            return torch.zeros(self.d_tree, device=device)

        features = []
        for mask, blen in splits[:64]:
            leaf_ratio = float(mask.sum()) / max(n_leaves, 1)
            features.append(torch.tensor(
                [leaf_ratio, min(blen, 5.0), 0.0],
                dtype=torch.float32, device=device
            ))

        if not features:
            return torch.zeros(self.d_tree, device=device)

        feat_tensor = torch.stack(features)
        pooled = feat_tensor.mean(dim=0)

        result = F.pad(pooled, (0, self.d_tree - pooled.shape[0]))
        return result

    def _estimate_quality(self, subtree_newick: str, mean_emb: torch.Tensor) -> torch.Tensor:
        device = mean_emb.device
        tree = parse_newick(subtree_newick)

        def _count_leaves(node: TreeNode) -> int:
            if node.is_leaf:
                return 1
            return sum(_count_leaves(c) for c in node.children)

        def _get_branch_lengths(node: TreeNode) -> List[float]:
            bls: List[float] = []
            for child in node.children:
                bls.append(abs(child.branch_length))
                bls.extend(_get_branch_lengths(child))
            return bls

        branch_lengths = _get_branch_lengths(tree)
        if branch_lengths:
            mean_bl = torch.tensor(sum(branch_lengths) / len(branch_lengths),
                                   dtype=torch.float32, device=device)
        else:
            mean_bl = torch.tensor(1.0, dtype=torch.float32, device=device)

        return torch.sigmoid(mean_bl / 5.0)

    def _compute_leaf_features(
        self,
        subtrees: List[Tuple[torch.Tensor, str]],
        global_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.leaf_feature_proj(global_embeddings)

    def _reconcile(
        self,
        subtrees: List[Tuple[torch.Tensor, str]],
        branch_scales: torch.Tensor,
        stay_probs: torch.Tensor,
        affinity_scores: torch.Tensor,
        guide_tree_newick: str,
    ) -> str:
        K = len(subtrees)
        device = branch_scales.device

        scaled_trees: List[TreeNode] = []
        leaf_to_info: Dict[int, Tuple[int, str]] = {}

        for k, (leaf_indices, subtree_newick) in enumerate(subtrees):
            tree = parse_newick(subtree_newick)
            self._scale_branch_lengths(tree, branch_scales[k].item())
            scaled_trees.append(tree)

            leaves = get_leaf_order(subtree_newick)
            for local_i, name in enumerate(leaves):
                if local_i < len(leaf_indices):
                    leaf_to_info[leaf_indices[local_i].item()] = (k, name)

        migrated: Dict[int, List[str]] = defaultdict(list)
        for global_idx, (from_k, leaf_name) in leaf_to_info.items():
            if global_idx < len(stay_probs) and stay_probs[global_idx].item() < 0.5:
                scores = affinity_scores[from_k].clone()
                scores[from_k] = -float('inf')
                dest = int(scores.argmax().item())
                migrated[dest].append(leaf_name)

                self._remove_leaf_from_tree(scaled_trees[from_k], leaf_name)

        for dest_k, leaf_names in migrated.items():
            if dest_k < K:
                for name in leaf_names:
                    new_leaf = TreeNode(name=name, is_leaf=True, branch_length=0.1)
                    scaled_trees[dest_k].children.append(new_leaf)

        guide_tree = parse_newick(guide_tree_newick)
        counter = [0]
        global_tree = self._graft_onto_guide(guide_tree, scaled_trees, counter)

        return tree_to_newick(global_tree)

    def _scale_branch_lengths(self, tree: TreeNode, scale: float):
        tree.branch_length *= scale
        for child in tree.children:
            self._scale_branch_lengths(child, scale)

    def _remove_leaf_from_tree(self, tree: TreeNode, leaf_name: str) -> bool:
        for i, child in enumerate(tree.children):
            if child.is_leaf and child.name == leaf_name:
                tree.children.pop(i)
                return True
            elif not child.is_leaf:
                if self._remove_leaf_from_tree(child, leaf_name):
                    if len(child.children) == 0:
                        for j, c in enumerate(tree.children):
                            if c is child:
                                tree.children.pop(j)
                                break
                    elif len(child.children) == 1:
                        grandchild = child.children[0]
                        grandchild.branch_length += child.branch_length
                        for j, c in enumerate(tree.children):
                            if c is child:
                                tree.children[j] = grandchild
                                break
                    return True
        return False

    def _graft_onto_guide(
        self,
        guide_node: TreeNode,
        subtrees: List[TreeNode],
        counter: List[int],
    ) -> TreeNode:
        if guide_node.is_leaf:
            k = counter[0]
            counter[0] += 1
            if k < len(subtrees):
                sub_root = subtrees[k]
                sub_root.branch_length = guide_node.branch_length
                return sub_root
            return guide_node

        new_node = TreeNode()
        new_node.branch_length = guide_node.branch_length
        new_node.support = guide_node.support
        new_node.is_leaf = False
        for child in guide_node.children:
            grafted = self._graft_onto_guide(child, subtrees, counter)
            new_node.children.append(grafted)
        return new_node
