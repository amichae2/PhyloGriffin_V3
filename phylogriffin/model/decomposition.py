import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from ..config import DecompositionConfig

try:
    from scipy.sparse.linalg import eigsh
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False

try:
    from scipy.cluster.vq import kmeans2
    HAS_SCIPY_KMEANS = True
except ImportError:
    HAS_SCIPY_KMEANS = False


class HierarchicalDecomposition(nn.Module):
    def __init__(self, config: DecompositionConfig):
        super().__init__()
        self.config = config

    def forward(self, msa: torch.Tensor, embeddings: torch.Tensor,
                edge_index: torch.Tensor, edge_weights: torch.Tensor) -> Tuple[List[Dict], str]:
        N = msa.shape[0]
        device = msa.device

        if N <= self.config.max_subproblem_size:
            subproblem = {
                "indices": torch.arange(N, dtype=torch.long, device=device),
                "sub_msa": msa,
                "sub_embeddings": embeddings,
                "sub_edge_index": edge_index,
            }
            guide = self._build_guide_tree([subproblem], embeddings)
            return [subproblem], guide

        if self.config.clustering_method == "spectral":
            try:
                clusters = self._spectral_clustering(
                    edge_index, edge_weights, N,
                    self.config.max_subproblem_size,
                    self.config.min_subproblem_size,
                )
            except Exception:
                clusters = self._greedy_partitioning(
                    edge_index, edge_weights, N,
                    self.config.max_subproblem_size,
                    self.config.min_subproblem_size,
                    embeddings,
                )
        else:
            clusters = self._greedy_partitioning(
                edge_index, edge_weights, N,
                self.config.max_subproblem_size,
                self.config.min_subproblem_size,
                embeddings,
            )

        subproblems = []
        for cluster_indices in clusters:
            idx = cluster_indices.to(device)
            sub_msa = msa[idx]
            sub_embs = embeddings[idx]

            in_cluster = torch.zeros(N, dtype=torch.bool, device=device)
            in_cluster[idx] = True

            g2l = {g.item(): l for l, g in enumerate(idx)}

            src = edge_index[0]
            dst = edge_index[1]
            edge_mask = in_cluster[src] & in_cluster[dst]

            if edge_mask.any():
                local_src = torch.tensor(
                    [g2l[s.item()] for s in src[edge_mask].tolist()],
                    dtype=torch.long, device=device,
                )
                local_dst = torch.tensor(
                    [g2l[d.item()] for d in dst[edge_mask].tolist()],
                    dtype=torch.long, device=device,
                )
                sub_ei = torch.stack([local_src, local_dst], dim=0)
            else:
                sub_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

            subproblems.append({
                "indices": idx,
                "sub_msa": sub_msa,
                "sub_embeddings": sub_embs,
                "sub_edge_index": sub_ei,
            })

        guide = self._build_guide_tree(subproblems, embeddings)
        return subproblems, guide

    def _spectral_clustering(self, edge_index: torch.Tensor, edge_weights: torch.Tensor,
                              n_nodes: int, max_size: int, min_size: int) -> List[torch.Tensor]:
        if not HAS_SCIPY_SPARSE:
            return self._greedy_partitioning(edge_index, edge_weights, n_nodes, max_size, min_size)

        import scipy.sparse as sp

        k = max(2, int(np.ceil(n_nodes / max_size)))
        if k >= n_nodes:
            k = max(1, n_nodes - 1)

        rows = edge_index[0].cpu().numpy().astype(np.int64)
        cols = edge_index[1].cpu().numpy().astype(np.int64)
        vals = edge_weights.cpu().numpy().astype(np.float64)

        A = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        A = (A + A.T) * 0.5
        A = A.tocsr()

        d = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
        I = sp.eye(n_nodes, format='csr')
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', maxiter=500)
        except Exception:
            return self._greedy_partitioning(edge_index, edge_weights, n_nodes, max_size, min_size)

        X = eigenvectors.astype(np.float64)
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-10, row_norms, 1.0)
        X = X / row_norms

        if HAS_SCIPY_KMEANS:
            try:
                centroids, labels = kmeans2(X, k, minit='points', missing='warn')
            except Exception:
                labels = self._simple_kmeans(X, k)
        else:
            labels = self._simple_kmeans(X, k)

        clusters = []
        for c in range(k):
            mask = labels == c
            if mask.any():
                clusters.append(np.where(mask)[0].astype(np.int64))

        result = []
        for cluster in clusters:
            if len(cluster) > max_size:
                sub_nodes = cluster
                idx_map = {old: new for new, old in enumerate(sub_nodes)}
                in_sub = np.zeros(n_nodes, dtype=bool)
                in_sub[sub_nodes] = True

                edge_mask = in_sub[rows] & in_sub[cols]
                if edge_mask.any():
                    sub_rows = np.array([idx_map[r] for r in rows[edge_mask]], dtype=np.int64)
                    sub_cols = np.array([idx_map[c] for c in cols[edge_mask]], dtype=np.int64)
                    sub_vals = vals[edge_mask]

                    sub_ei = torch.tensor(np.stack([sub_rows, sub_cols]), dtype=torch.long)
                    sub_ew = torch.tensor(sub_vals, dtype=torch.float)

                    sub_result = self._spectral_clustering(sub_ei, sub_ew, len(sub_nodes), max_size, min_size)
                    for sr in sub_result:
                        result.append(torch.from_numpy(sub_nodes[sr.numpy()]).long())
                else:
                    result.append(torch.from_numpy(sub_nodes).long())
            else:
                result.append(torch.from_numpy(cluster).long())

        result = self._merge_small_clusters(result, edge_index, edge_weights, n_nodes, min_size)
        return result

    def _greedy_partitioning(self, edge_index: torch.Tensor, edge_weights: torch.Tensor,
                              n_nodes: int, max_size: int, min_size: int,
                              embeddings: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        device = edge_index.device
        degree = torch.zeros(n_nodes, device=device)
        degree.index_add_(0, edge_index[0], edge_weights.float())
        degree.index_add_(0, edge_index[1], edge_weights.float())

        sorted_nodes = torch.argsort(degree, descending=True)
        assigned = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        clusters = []

        adj = [[] for _ in range(n_nodes)]
        for e in range(edge_index.shape[1]):
            u = edge_index[0, e].item()
            v = edge_index[1, e].item()
            w = edge_weights[e].item()
            adj[u].append((v, w))
            adj[v].append((u, w))

        for start_node in sorted_nodes:
            start = start_node.item()
            if assigned[start]:
                continue
            cluster = []
            queue = [start]
            assigned[start] = True
            while queue and len(cluster) < max_size:
                u = queue.pop(0)
                cluster.append(u)
                neighbors = sorted(adj[u], key=lambda x: x[1], reverse=True)
                for v, w in neighbors:
                    if not assigned[v] and len(cluster) < max_size:
                        assigned[v] = True
                        queue.append(v)
            if cluster:
                clusters.append(torch.tensor(cluster, dtype=torch.long, device=device))

        orphan_mask = ~assigned
        if orphan_mask.any():
            orphan_indices = torch.where(orphan_mask)[0]
            if embeddings is not None and len(clusters) > 0:
                embs = embeddings.to(device).float()
                orphan_embs = embs[orphan_indices]
                cluster_means = torch.stack([embs[cl].mean(dim=0) for cl in clusters])
                for oi_val in orphan_indices:
                    oe = embs[oi_val]
                    dists = torch.norm(cluster_means - oe.unsqueeze(0), dim=1)
                    best = torch.argmin(dists).item()
                    clusters[best] = torch.cat([clusters[best], oi_val.unsqueeze(0)])
            else:
                clusters.append(torch.where(orphan_mask)[0])

        clusters = self._merge_small_clusters(clusters, edge_index, edge_weights, n_nodes, min_size)
        return clusters

    def _merge_small_clusters(self, clusters: List[torch.Tensor], edge_index: torch.Tensor,
                               edge_weights: torch.Tensor, n_nodes: int,
                               min_size: int) -> List[torch.Tensor]:
        if len(clusters) <= 1:
            return clusters

        device = edge_index.device
        cluster_assign = torch.zeros(n_nodes, dtype=torch.long, device=device)
        for i, cl in enumerate(clusters):
            cluster_assign[cl.to(device)] = i

        k = len(clusters)
        inter_weight = np.zeros((k, k), dtype=np.float64)
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        wgt = edge_weights.cpu().numpy()
        assign_np = cluster_assign.cpu().numpy()
        for e in range(edge_index.shape[1]):
            cu = assign_np[src[e]]
            cv = assign_np[dst[e]]
            if cu != cv:
                inter_weight[cu, cv] += wgt[e]
                inter_weight[cv, cu] += wgt[e]

        while True:
            k = len(clusters)
            small = [i for i in range(k) if len(clusters[i]) < min_size]
            if not small or k <= 1:
                break

            i = small[0]

            candidates = [j for j in range(k) if j != i]
            non_small = [j for j in candidates if len(clusters[j]) >= min_size]
            if non_small:
                candidates = non_small

            best_j = max(candidates, key=lambda j: inter_weight[i, j])

            a, b = (i, best_j) if i < best_j else (best_j, i)

            clusters[a] = torch.cat([clusters[a], clusters[b]])
            clusters.pop(b)

            for c in range(inter_weight.shape[0]):
                if c != a and c != b:
                    inter_weight[a, c] += inter_weight[b, c]
                    inter_weight[c, a] = inter_weight[a, c]

            mask = np.ones(inter_weight.shape[0], dtype=bool)
            mask[b] = False
            inter_weight = inter_weight[mask][:, mask]

        return clusters

    def _simple_kmeans(self, X: np.ndarray, k: int, max_iter: int = 100,
                        seed: int = 42) -> np.ndarray:
        n, d = X.shape
        if k >= n:
            return np.arange(n, dtype=np.int64)

        rng = np.random.RandomState(seed)
        centroids = X[rng.choice(n, k, replace=False)].copy()

        for _ in range(max_iter):
            dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centroids[c] = X[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(centroids, new_centroids, rtol=1e-6):
                break
            centroids = new_centroids

        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def _build_guide_tree(self, subproblems: List[Dict],
                           embeddings: torch.Tensor) -> str:
        K = len(subproblems)
        if K == 1:
            indices = subproblems[0]["indices"]
            n = len(indices)
            if n == 1:
                return f"(L{indices[0].item()});"
            show = min(n, 20)
            names = ",".join([f"L{indices[i].item()}" for i in range(show)])
            if n > 20:
                names += ",..."
            return f"({names});"

        device = embeddings.device
        means = torch.stack([
            sp["sub_embeddings"].to(device).float().mean(dim=0)
            for sp in subproblems
        ])

        D = torch.cdist(means, means, p=2).cpu().numpy().astype(np.float64)

        max_nodes = 2 * K
        big_D = np.full((max_nodes, max_nodes), np.inf, dtype=np.float64)
        big_D[:K, :K] = D

        active = set(range(K))
        labels = {i: f"S{i}" for i in range(K)}
        nid = K

        while len(active) > 1:
            active_list = sorted(active)
            min_val = np.inf
            min_i = min_j = -1
            for ai in range(len(active_list)):
                i = active_list[ai]
                for aj in range(ai + 1, len(active_list)):
                    j = active_list[aj]
                    if big_D[i, j] < min_val:
                        min_val = big_D[i, j]
                        min_i, min_j = i, j

            h = min_val / 2.0
            labels[nid] = f"({labels[min_i]},{labels[min_j]}):{h:.6f}"

            for k_idx in range(nid):
                if big_D[min_i, k_idx] < np.inf:
                    big_D[nid, k_idx] = (big_D[min_i, k_idx] + big_D[min_j, k_idx]) / 2.0
                    big_D[k_idx, nid] = big_D[nid, k_idx]

            del labels[min_i]
            del labels[min_j]
            active.discard(min_i)
            active.discard(min_j)
            active.add(nid)
            nid += 1

        root = next(iter(active))
        return labels[root] + ";"
