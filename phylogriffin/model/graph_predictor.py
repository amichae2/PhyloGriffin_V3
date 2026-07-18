"""
PhyloGriffin v3 -- Stage B: Learned Phylogenetic Graph.
Predicts whether two sequences are phylogenetically adjacent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPredictor(nn.Module):
    """
    Stage B: Predicts whether two sequences are phylogenetically adjacent.

    Given a pair of sequence embeddings, outputs a scalar probability
    in [0, 1] indicating whether the two sequences share a direct
    ancestor-descendant relationship in the true phylogenetic tree.
    """

    def __init__(self, d_model: int, hidden_dims: list[int]):
        super().__init__()
        self.d_model = d_model
        input_dim = 4 * d_model

        layers = []
        in_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            in_dim = hdim
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward_single(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(
            [
                emb_i,
                emb_j,
                emb_i * emb_j,
                torch.abs(emb_i - emb_j),
            ]
        )

        x = concat
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = self.output_layer(x)
        return torch.sigmoid(x).squeeze(-1)

    def forward_batch(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(
            [
                emb_i,
                emb_j,
                emb_i * emb_j,
                torch.abs(emb_i - emb_j),
            ],
            dim=-1,
        )

        x = concat
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = self.output_layer(x)
        return torch.sigmoid(x).squeeze(-1)

    @torch.no_grad()
    def build_graph(
        self,
        embeddings: torch.Tensor,
        edge_threshold: float = 0.5,
        k_candidates: int = 200,
        k_neighbors: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the full sparse phylogenetic graph.

        1. Compute L2 distance matrix between all pairs.
           If N > 10000, use chunked computation on GPU.
        2. For each node, take top k_candidates nearest neighbours.
        3. For each candidate edge, compute predictor probability.
        4. Keep edges where prob > edge_threshold.
        5. Enforce symmetry and remove self-loops.
        6. Optionally prune to at most k_neighbors per node.

        Args:
            embeddings: (N, d_model) sequence embeddings.
            edge_threshold: probability threshold for keeping edges.
            k_candidates: nearest neighbour candidates to score per node.
            k_neighbors: maximum edges per node after pruning.

        Returns:
            edge_index: (2, E) LongTensor of undirected edges (u < v).
            edge_weights: (E,) FloatTensor of edge probabilities.
        """
        was_training = self.training
        self.eval()
        N, d_model = embeddings.shape
        device = embeddings.device

        k_cand = min(k_candidates, N - 1)
        dist_chunk_size = 10000

        edges_i_parts = []
        edges_j_parts = []

        for start in range(0, N, dist_chunk_size):
            end = min(start + dist_chunk_size, N)
            chunk = embeddings[start:end]
            dists = torch.cdist(chunk, embeddings, p=2)

            if k_cand < N - 1:
                _, topk_idx = torch.topk(dists, k=k_cand + 1, dim=-1, largest=False)
            else:
                topk_idx = torch.arange(N, device=device).unsqueeze(0).expand(chunk.size(0), -1)

            for local_i in range(chunk.size(0)):
                global_i = start + local_i
                cands = topk_idx[local_i]
                mask = cands != global_i
                cands = cands[mask][:k_cand]
                edges_i_parts.append(
                    torch.full((cands.numel(),), global_i, dtype=torch.long, device=device)
                )
                edges_j_parts.append(cands)

        rows_i = torch.cat(edges_i_parts, dim=0)
        cols_i = torch.cat(edges_j_parts, dim=0)

        prediction_chunk_size = 4096
        num_pairs = rows_i.numel()
        probs = torch.zeros(num_pairs, device=device)

        for start in range(0, num_pairs, prediction_chunk_size):
            end = min(start + prediction_chunk_size, num_pairs)
            r = rows_i[start:end]
            c = cols_i[start:end]
            probs[start:end] = self.forward_batch(embeddings[r], embeddings[c])

        mask = probs > edge_threshold
        rows_i = rows_i[mask]
        cols_i = cols_i[mask]
        probs = probs[mask]

        if rows_i.numel() == 0:
            self.train(was_training)
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float, device=device),
            )

        u = torch.min(rows_i, cols_i)
        v = torch.max(rows_i, cols_i)
        self_loop_mask = u != v
        u, v, probs = (
            u[self_loop_mask],
            v[self_loop_mask],
            probs[self_loop_mask],
        )

        edge_key = u * N + v
        unique_keys, inverse = edge_key.unique(return_inverse=True)
        edge_weights = torch.zeros(unique_keys.numel(), device=device).scatter_reduce(
            0, inverse, probs, reduce="amax", include_self=False
        )
        u_unique = unique_keys // N
        v_unique = unique_keys % N
        edge_index = torch.stack([u_unique, v_unique], dim=0)

        if k_neighbors is not None and k_neighbors > 0 and edge_index.size(1) > 0:
            E = edge_index.size(1)
            sorted_w, sorted_idx = edge_weights.sort(descending=True)
            sorted_u = edge_index[0, sorted_idx].cpu()
            sorted_v = edge_index[1, sorted_idx].cpu()

            node_counts = torch.zeros(N, dtype=torch.long)
            keep_mask = torch.zeros(E, dtype=torch.bool)

            for e in range(E):
                u = sorted_u[e].item()
                v = sorted_v[e].item()
                if node_counts[u] < k_neighbors and node_counts[v] < k_neighbors:
                    keep_idx = sorted_idx[e].item()
                    keep_mask[keep_idx] = True
                    node_counts[u] += 1
                    node_counts[v] += 1

            edge_index = edge_index[:, keep_mask]
            edge_weights = edge_weights[keep_mask]

        self.train(was_training)
        return edge_index, edge_weights
