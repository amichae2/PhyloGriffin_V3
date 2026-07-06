"""
PhyloGriffin v3 -- Tree utility functions.
Minimal Newick parser, split extraction, RF distance, patristic distances.
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict


class TreeNode:
    def __init__(self, name: str = "", branch_length: float = 0.0,
                 support: float = 0.0, is_leaf: bool = False):
        self.name = name
        self.branch_length = branch_length
        self.support = support
        self.is_leaf = is_leaf
        self.children: List["TreeNode"] = []
        self.parent: Optional["TreeNode"] = None


def _tokenize_newick(newick_str: str) -> List[str]:
    tokens = []
    i = 0
    s = newick_str.strip().rstrip(";")
    while i < len(s):
        ch = s[i]
        if ch in ("(", ")", ",", ":", ";"):
            tokens.append(ch)
            i += 1
        elif ch == "'":
            j = s.index("'", i + 1)
            tokens.append(s[i:j + 1])
            i = j + 1
        elif ch == '"':
            j = s.index('"', i + 1)
            tokens.append(s[i:j + 1])
            i = j + 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in ("(", ")", ",", ":", ";") and not s[j].isspace():
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_tokens(tokens: List[str], pos: int = 0) -> Tuple[TreeNode, int]:
    node = TreeNode()
    name = ""
    branch_length = 0.0
    support = 0.0

    if tokens[pos] == "(":
        pos += 1
        children: List[TreeNode] = []
        while True:
            child, pos = _parse_tokens(tokens, pos)
            children.append(child)
            child.parent = node
            if pos >= len(tokens) or tokens[pos] == ")":
                break
            if tokens[pos] == ",":
                pos += 1
            else:
                break
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        node.children = children
        node.is_leaf = False
    else:
        name = tokens[pos].strip("'\"")
        pos += 1
        node.name = name
        node.is_leaf = True

    if pos < len(tokens) and tokens[pos] != "," and tokens[pos] != ")" and tokens[pos] != ";":
        tag = tokens[pos]
        pos += 1
        try:
            support = float(tag)
            node.support = support
        except ValueError:
            node.name = tag

    if pos < len(tokens) and tokens[pos] == ":":
        pos += 1
        if pos < len(tokens):
            try:
                branch_length = float(tokens[pos])
                pos += 1
            except ValueError:
                pass
    node.branch_length = branch_length
    return node, pos


def parse_newick(newick_str: str) -> TreeNode:
    tokens = _tokenize_newick(newick_str)
    tree, _ = _parse_tokens(tokens)
    return tree


def _count_nodes(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_nodes(c) for c in node.children)


def _tree_to_newick_rec(node: TreeNode) -> str:
    if node.is_leaf:
        name = node.name.replace(" ", "_")
        if node.branch_length:
            return f"{name}:{node.branch_length:.6f}"
        return name
    child_strs = [_tree_to_newick_rec(c) for c in node.children]
    inner = ",".join(child_strs)
    result = f"({inner})"
    if node.support:
        result += f"{node.support:.6f}"
    if node.branch_length:
        result += f":{node.branch_length:.6f}"
    return result


def tree_to_newick(tree: TreeNode) -> str:
    return _tree_to_newick_rec(tree) + ";"


def _leaf_index_map(node: TreeNode, idx: int = 0) -> Tuple[Dict[str, int], int]:
    mapping = {}
    if node.is_leaf:
        mapping[node.name] = idx
        return mapping, idx + 1
    for child in node.children:
        child_map, idx = _leaf_index_map(child, idx)
        mapping.update(child_map)
    return mapping, idx


def newick_to_splits(newick_str: str, n_leaves: int) -> List[Tuple[np.ndarray, float]]:
    tree = parse_newick(newick_str)
    leaf_to_idx, _ = _leaf_index_map(tree)

    splits = []

    def _collect(node: TreeNode):
        if node.is_leaf:
            return set()

        child_sets = []
        for child in node.children:
            child_sets.append(_collect(child))

        all_leaves = set()
        for cs in child_sets:
            all_leaves.update(cs)

        mask = np.zeros(n_leaves, dtype=bool)
        for name in all_leaves:
            if name in leaf_to_idx:
                mask[leaf_to_idx[name]] = True

        n_true = mask.sum()
        if 1 < n_true < n_leaves - 1:
            splits.append((mask, node.branch_length))

        return all_leaves

    _collect(tree)
    return splits


def _splits_compatible(s1: np.ndarray, s2: np.ndarray) -> bool:
    i1 = set(np.where(s1)[0].tolist())
    i2 = set(np.where(s2)[0].tolist())
    return (i1.issubset(i2) or i2.issubset(i1) or
            len(i1.intersection(i2)) == 0)


def splits_to_newick(splits: List[Tuple[np.ndarray, float]],
                     leaf_names: List[str]) -> str:
    if not splits:
        return f"({','.join(leaf_names)});"

    n_leaves = len(leaf_names)
    for s1, _ in splits:
        for s2, _ in splits:
            if s1 is not s2 and not _splits_compatible(s1, s2):
                raise ValueError("Splits are not pairwise compatible")

    clusters = [{i} for i in range(n_leaves)]

    sorted_splits = sorted(
        [(mask.copy(), bl) for mask, bl in splits],
        key=lambda x: x[0].sum()
    )

    active_clusters = list(range(n_leaves))
    node_counter = n_leaves
    cluster_to_node: Dict[int, TreeNode] = {
        i: TreeNode(name=leaf_names[i], is_leaf=True)
        for i in range(n_leaves)
    }

    for mask, blen in sorted_splits:
        members = set(np.where(mask)[0].tolist())
        relevant = [c for c in active_clusters if clusters[c].issubset(members)]
        if len(relevant) < 2:
            continue
        new_node = TreeNode(branch_length=blen)
        for c in relevant:
            new_node.children.append(cluster_to_node[c])
            active_clusters.remove(c)
        cluster_to_node[node_counter] = new_node
        clusters.append(members)
        active_clusters.append(node_counter)
        node_counter += 1

    if len(active_clusters) > 1:
        root = TreeNode()
        for c in active_clusters:
            root.children.append(cluster_to_node[c])
        return tree_to_newick(root)

    if active_clusters:
        return tree_to_newick(cluster_to_node[active_clusters[0]])

    return f"({','.join(leaf_names)});"


def robinson_foulds(splits1, splits2) -> float:
    def split_key(mask):
        return bytes(mask.tobytes())

    keys1 = set()
    keys2 = set()
    for mask, _ in splits1:
        keys1.add(split_key(mask))
    for mask, _ in splits2:
        keys2.add(split_key(mask))

    if not keys1 and not keys2:
        return 0.0

    n_diff = len(keys1.symmetric_difference(keys2))
    n_total = len(keys1) + len(keys2)
    if n_total == 0:
        return 0.0
    return n_diff / n_total


def patristic_distances(tree_newick: str, n_leaves: int) -> np.ndarray:
    tree = parse_newick(tree_newick)
    leaf_to_idx, _ = _leaf_index_map(tree)

    node_list = []
    node_to_id = {}

    def _assign_ids(node):
        node_to_id[id(node)] = len(node_list)
        node_list.append(node)
        for child in node.children:
            _assign_ids(child)
    _assign_ids(tree)

    n_nodes = len(node_list)
    parent = np.full(n_nodes, -1, dtype=np.int32)
    bl_to_parent = np.zeros(n_nodes, dtype=np.float64)

    for nid, node in enumerate(node_list):
        for child in node.children:
            cid = node_to_id[id(child)]
            parent[cid] = nid
            bl_to_parent[cid] = child.branch_length

    leaf_ids = []
    leaf_indices = []
    for nid, node in enumerate(node_list):
        if node.is_leaf and node.name in leaf_to_idx:
            leaf_ids.append(nid)
            leaf_indices.append(leaf_to_idx[node.name])

    actual_n = len(leaf_ids)
    if actual_n == 0:
        return np.zeros((n_leaves, n_leaves), dtype=np.float32)

    dist_to_root = np.zeros(n_nodes, dtype=np.float64)
    for nid in range(n_nodes):
        total_bl = 0.0
        curr = nid
        while parent[curr] != -1:
            total_bl += bl_to_parent[curr]
            curr = parent[curr]
        dist_to_root[nid] = total_bl

    leaf_ancestors = []
    for lid in leaf_ids:
        ancestors = []
        curr = lid
        while curr != -1:
            ancestors.append(curr)
            curr = parent[curr]
        leaf_ancestors.append(set(ancestors))

    dist = np.zeros((n_leaves, n_leaves), dtype=np.float32)
    for i in range(actual_n):
        for j in range(i + 1, actual_n):
            common = leaf_ancestors[i] & leaf_ancestors[j]
            if not common:
                continue
            lca = max(common)
            d = dist_to_root[leaf_ids[i]] + dist_to_root[leaf_ids[j]] - 2.0 * dist_to_root[lca]
            gi = leaf_indices[i]
            gj = leaf_indices[j]
            dist[gi, gj] = d
            dist[gj, gi] = d

    return dist


def get_leaf_order(newick_str: str) -> List[str]:
    tree = parse_newick(newick_str)
    leaves = []

    def _traverse(node: TreeNode):
        if node.is_leaf:
            leaves.append(node.name)
        else:
            for child in node.children:
                _traverse(child)

    _traverse(tree)
    return leaves


def corrupt_tree(newick_str: str, n_swaps: int = 3, seed: int = None) -> str:
    if seed is not None:
        np.random.seed(seed)
    tree = parse_newick(newick_str)

    internal_nodes: List[TreeNode] = []
    def _collect(node):
        if not node.is_leaf and len(node.children) == 2:
            c1, c2 = node.children
            if not c1.is_leaf and not c2.is_leaf:
                if len(c1.children) == 2 and len(c2.children) == 2:
                    internal_nodes.append(node)
        for child in node.children:
            _collect(child)
    _collect(tree)

    if not internal_nodes:
        return newick_str

    for _ in range(n_swaps):
        node = internal_nodes[np.random.randint(len(internal_nodes))]
        c1, c2 = node.children
        b_node = c1.children[1]
        c_node = c2.children[0]
        c1.children[1] = c_node
        c_node.parent = c1
        c2.children[0] = b_node
        b_node.parent = c2

    return tree_to_newick(tree)


def is_binary(tree_newick: str) -> bool:
    tree = parse_newick(tree_newick)

    def _check(node: TreeNode) -> bool:
        if node.is_leaf:
            return True
        if len(node.children) != 2:
            return False
        return all(_check(c) for c in node.children)

    return _check(tree)


def collapse_low_support(tree_newick: str, threshold: float = 0.5) -> str:
    tree = parse_newick(tree_newick)

    def _collapse(node: TreeNode) -> TreeNode:
        if node.is_leaf:
            return node
        new_children = []
        for child in node.children:
            collapsed = _collapse(child)
            if not child.is_leaf and child.support < threshold:
                new_children.extend(collapsed.children)
            else:
                new_children.append(collapsed)
        node.children = new_children
        return node

    _collapse(tree)
    return tree_to_newick(tree)
