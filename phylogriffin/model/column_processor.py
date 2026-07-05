import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from ..config import PhyloGriffinConfig


class TokenEmbedding(nn.Module):
    def __init__(self, alphabet_size: int, pad_idx: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(alphabet_size + 1, d_model, padding_idx=pad_idx)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(dtype)


class RG_LRU(nn.Module):
    def __init__(self, d_model: int, d_rnn: int):
        super().__init__()
        self.d_rnn = d_rnn
        self.c = 8.0
        self.input_proj = nn.Linear(d_model, d_rnn)
        self.W_r = nn.Linear(d_rnn, d_rnn)
        self.W_i = nn.Linear(d_rnn, d_rnn)
        self.Lambda = nn.Parameter(torch.randn(d_rnn) * 0.01)
        self.output_proj = nn.Linear(d_rnn, d_model)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        if state is None:
            state = torch.zeros(N, self.d_rnn, device=x.device, dtype=x.dtype)

        x_proj = self.input_proj(x)
        r = torch.sigmoid(self.W_r(x_proj))
        i = torch.sigmoid(self.W_i(x_proj))

        log_a = self.c * r * F.logsigmoid(self.Lambda)
        a = torch.exp(log_a).clamp(0, 1)

        h = a * state + torch.sqrt(1.0 - a * a) * (i * x_proj)
        y = self.output_proj(h)

        return y, h


def _sequential_scan(a: torch.Tensor, input_term: torch.Tensor) -> torch.Tensor:
    N, L, D = a.shape
    h = torch.zeros(N, D, device=a.device, dtype=a.dtype)
    outputs: List[torch.Tensor] = []
    for t in range(L):
        h = a[:, t, :] * h + input_term[:, t, :]
        outputs.append(h.unsqueeze(1))
    return torch.cat(outputs, dim=1)


class ParallelRG_LRU(nn.Module):
    def __init__(self, d_model: int, d_rnn: int):
        super().__init__()
        self.d_rnn = d_rnn
        self.c = 8.0

        self.x_rnn_proj = nn.Linear(d_model, d_rnn)
        self.x_gate_proj = nn.Linear(d_model, d_rnn)

        self.r_proj = nn.Linear(d_rnn, d_rnn)
        self.i_proj = nn.Linear(d_rnn, d_rnn)

        self.Lambda = nn.Parameter(torch.randn(d_rnn) * 0.01)

        self.output_proj = nn.Linear(d_rnn, d_model)

        self.conv = nn.Conv1d(d_rnn, d_rnn, kernel_size=4, padding=3, groups=d_rnn)

        self._scan_compiled = None

    def _get_compiled_scan(self):
        if self._scan_compiled is None and hasattr(torch, "compile"):
            self._scan_compiled = torch.compile(
                _sequential_scan, fullgraph=True, dynamic=True, mode="reduce-overhead"
            )
        return self._scan_compiled

    def _run_scan(self, a: torch.Tensor, input_term: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "compile"):
            scan_fn = self._get_compiled_scan()
            if scan_fn is not None:
                return scan_fn(a, input_term)
        return _sequential_scan(a, input_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dim = False
        store_shape = None
        if x.ndim == 4:
            B, N, L, d_model = x.shape
            x = x.view(B * N, L, d_model)
            batch_dim = True
            store_shape = (B, N)
        elif x.ndim == 3:
            N, L, d_model = x.shape
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D")

        N, L, d_model = x.shape

        x_rnn = self.x_rnn_proj(x)
        x_gate = self.x_gate_proj(x)

        x_rnn_t = x_rnn.transpose(1, 2)
        x_rnn_conv = self.conv(x_rnn_t)
        x_rnn_conv = x_rnn_conv[..., :L]
        x_rnn_conv = x_rnn_conv.transpose(1, 2)

        r = torch.sigmoid(self.r_proj(x_rnn_conv))
        i = torch.sigmoid(self.i_proj(x_rnn_conv))

        log_a = self.c * r * F.logsigmoid(self.Lambda)
        a = torch.exp(log_a).clamp(0, 1)

        input_term = torch.sqrt(1.0 - a * a) * (i * x_rnn_conv)

        h_seq = self._run_scan(a, input_term)

        y = h_seq * F.gelu(x_gate)
        y = self.output_proj(y)

        if batch_dim:
            y = y.view(store_shape[0], store_shape[1], L, d_model)

        return y


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.gate_proj = nn.Linear(d_model, hidden)
        self.value_proj = nn.Linear(d_model, hidden)
        self.output_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        value = self.value_proj(x)
        out = self.output_proj(gate * value)
        return self.dropout(out)


class LocalMQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, window_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, n_heads * head_dim)
        self.k_proj = nn.Linear(d_model, head_dim)
        self.v_proj = nn.Linear(d_model, head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model)

    def _build_sliding_mask(self, L: int, device: torch.device) -> torch.Tensor:
        half_w = self.window_size // 2
        positions = torch.arange(L, device=device)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        mask = (distances.abs() <= half_w).float()
        mask = (1.0 - mask) * -1e9
        return mask

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        *batch, L, d = x.shape
        position = torch.arange(L, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(0, d, 2, device=x.device, dtype=torch.float32)
        theta = 1.0 / (10000.0 ** (dim_idx / d))

        freqs = torch.outer(position, theta)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        x_flat = x.reshape(*batch, L, d // 2, 2)
        x_even = x_flat[..., 0]
        x_odd = x_flat[..., 1]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        return x_rot.reshape(*batch, L, d)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N, L, _ = x.shape

        q = self.q_proj(x).view(N, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(N, L, 1, self.head_dim)
        v = self.v_proj(x).view(N, L, 1, self.head_dim)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        window_mask = self._build_sliding_mask(L, x.device)
        scores = scores + window_mask

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~key_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, L, self.n_heads * self.head_dim)
        out = self.out_proj(out)

        return out


class GriffinLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_rnn: int,
        n_heads: int,
        head_dim: int,
        local_window: int,
        mlp_expansion: int,
        dropout: float,
        is_recurrent: bool,
    ):
        super().__init__()
        self.is_recurrent = is_recurrent

        if is_recurrent:
            self.temporal_mixer = ParallelRG_LRU(d_model, d_rnn)
        else:
            self.temporal_mixer = LocalMQA(d_model, n_heads, head_dim, local_window)

        self.temporal_norm = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, mlp_expansion, dropout)
        self.mlp_norm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.is_recurrent:
            t_out = self.temporal_mixer(x)
        else:
            t_out = self.temporal_mixer(x, mask)

        x = self.temporal_norm(x + t_out)
        x = self.mlp_norm(x + self.mlp(x))

        return x


class TitansMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mem: int,
        n_slots: int,
        depth: int,
        surprise_threshold: float,
        momentum: float,
    ):
        super().__init__()
        self.d_mem = d_mem
        self.n_slots = n_slots
        self.depth = depth
        self.surprise_threshold = surprise_threshold
        self.momentum = momentum

        self.keys = nn.Parameter(torch.randn(n_slots, d_mem))
        self.values = nn.Parameter(torch.randn(n_slots, d_mem))
        self.register_buffer("usage", torch.zeros(n_slots))

        self.col_proj = nn.Linear(d_model, d_mem)
        self.query_proj = nn.Linear(d_mem, d_mem)
        self.read_proj = nn.Linear(d_mem, d_model)
        self.mem_mlp = nn.ModuleList(
            [nn.Linear(d_mem, d_mem) for _ in range(depth)]
        )

    def reset_state(self):
        self.usage.zero_()
        nn.init.normal_(self.keys, std=0.02)
        nn.init.normal_(self.values, std=0.02)

    def forward(
        self, col_repr: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        valid = col_repr[mask]
        if valid.shape[0] == 0:
            return col_repr

        c_col = valid.mean(dim=0)
        c_col = self.col_proj(c_col)

        query = self.query_proj(c_col)
        scores = F.softmax(
            torch.matmul(self.keys, query) / math.sqrt(self.d_mem), dim=0
        )
        predicted = torch.matmul(scores, self.values)

        surprise = F.mse_loss(c_col, predicted, reduction="mean")

        if self.training:
            idx = scores.argmax().item()
            new_key = (
                self.momentum * self.keys[idx]
                + (1.0 - self.momentum) * c_col
            )
            mem_val = c_col
            for layer in self.mem_mlp:
                mem_val = F.silu(layer(mem_val))
            new_val = (
                self.momentum * self.values[idx]
                + (1.0 - self.momentum) * mem_val
            )

            if surprise > self.surprise_threshold:
                self.keys.data[idx] = new_key.detach()
                self.values.data[idx] = new_val.detach()
                self.usage[idx] += 1
        else:
            if surprise > self.surprise_threshold:
                idx = scores.argmax().item()
                with torch.no_grad():
                    new_key = (
                        self.momentum * self.keys[idx]
                        + (1.0 - self.momentum) * c_col.detach()
                    )
                    mem_val = c_col.detach()
                    for layer in self.mem_mlp:
                        mem_val = F.silu(layer(mem_val))
                    new_val = (
                        self.momentum * self.values[idx]
                        + (1.0 - self.momentum) * mem_val
                    )
                self.keys.data[idx] = new_key
                self.values.data[idx] = new_val
                self.usage[idx] += 1

        memory_context = torch.matmul(scores, self.values)
        enriched = col_repr + self.read_proj(memory_context).unsqueeze(0)

        return enriched


class ColumnProcessor(nn.Module):
    def __init__(self, config: PhyloGriffinConfig):
        super().__init__()
        self.config = config

        cfg = config.griffin
        self.token_embed = TokenEmbedding(
            config.alphabet_size, config.pad_idx, cfg.d_model
        )

        pattern = cfg.pattern
        self.layers = nn.ModuleList()
        pattern_idx = 0
        for i in range(cfg.n_layers):
            is_recurrent = (pattern_idx < pattern[0])
            self.layers.append(
                GriffinLayer(
                    d_model=cfg.d_model,
                    d_rnn=cfg.d_rnn,
                    n_heads=cfg.n_heads,
                    head_dim=cfg.head_dim,
                    local_window=cfg.local_window,
                    mlp_expansion=cfg.mlp_expansion,
                    dropout=cfg.dropout,
                    is_recurrent=is_recurrent,
                )
            )
            pattern_idx += 1
            if pattern_idx == sum(pattern):
                pattern_idx = 0

        self.final_norm = RMSNorm(cfg.d_model)

        mem_cfg = config.titans
        self.titans = TitansMemory(
            d_model=cfg.d_model,
            d_mem=mem_cfg.d_mem,
            n_slots=mem_cfg.n_memory_slots,
            depth=mem_cfg.memory_depth,
            surprise_threshold=mem_cfg.surprise_threshold,
            momentum=mem_cfg.momentum,
        )

        self._original_layers = self.layers
        self._compiled = False

    def build_layer_sequence(self):
        self._original_layers = self.layers
        self._layer_sequence = nn.Sequential(*list(self.layers))

    def reset_layers(self):
        if hasattr(self, "_original_layers") and self._original_layers is not None:
            self.layers = self._original_layers
        self._compiled = False

    def _batched_titans(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, L, d = x.shape
        x_flat = x.view(N * L, d)
        mask_flat = mask.view(N * L)
        return self.titans(x_flat, mask_flat).view(N, L, d)

    def forward_hidden(
        self,
        msa: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if msa.ndim == 3:
            B, N, L = msa.shape
            msa = msa.view(B * N, L)
            batched = True
            store_shape = (B, N, L)
        elif msa.ndim == 2:
            N, L = msa.shape
            batched = False
        else:
            raise ValueError(f"Expected 2D or 3D MSA, got {msa.ndim}D")

        if mask is None:
            mask = msa != self.config.pad_idx

        if mask.ndim == 3:
            mask = mask.view(-1, L)

        x = self.token_embed(msa)

        self.titans.reset_state()

        if self._compiled:
            x = self.layers(x, mask)
            x = self._batched_titans(x, mask)
        else:
            for layer in self.layers:
                x = layer(x, mask)

        x = self.final_norm(x)
        return x

    def forward(
        self,
        msa: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if msa.ndim == 3:
            B, N, L = msa.shape
            batched = True
            store_shape = (B, N, L)
        elif msa.ndim == 2:
            batched = False
            N, L = msa.shape
        else:
            raise ValueError(f"Expected 2D or 3D MSA, got {msa.ndim}D")

        x = self.forward_hidden(msa, mask)

        if mask is None:
            mask = msa != self.config.pad_idx
        if mask.ndim == 3:
            mask = mask.view(-1, L)

        mask_float = mask.float().unsqueeze(-1)
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        seq_emb = (x * mask_float).sum(dim=1) / valid_count

        if batched:
            seq_emb = seq_emb.view(B, N, -1)

        col_memory = self.titans.values

        return seq_emb, col_memory
