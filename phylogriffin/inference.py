"""
PhyloGriffin v3 -- Inference Pipeline.
Complete model wrapping all stages.
"""

import torch
import torch.nn as nn
import math
import warnings
from typing import List, Tuple, Dict, Optional

from .config import PhyloGriffinConfig
from .model.column_processor import ColumnProcessor
from .model.graph_predictor import GraphPredictor
from .model.decomposition import HierarchicalDecomposition
from .model.diffusion import DiffusionTreeGenerator
from .model.supertree import SupertreeReconciler
from .model.refinement import RefinementPass


class LayerSequenceWrapper(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*list(layers))

    def forward(self, x, mask=None):
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                for sub in layer:
                    if hasattr(sub, "is_recurrent") and sub.is_recurrent:
                        x = sub(x)
                    else:
                        x = sub(x, mask)
            elif hasattr(layer, "is_recurrent") and layer.is_recurrent:
                x = layer(x)
            else:
                x = layer(x, mask)
        return x


class PhyloGriffinV3(nn.Module):
    def __init__(self, config: PhyloGriffinConfig):
        super().__init__()
        self.config = config

        self.column_processor = ColumnProcessor(config)
        self.graph_predictor = GraphPredictor(
            d_model=config.griffin.d_model,
            hidden_dims=config.graph.predictor_hidden,
        )
        self.decomposition = HierarchicalDecomposition(config.decomposition)
        self.diffusion = DiffusionTreeGenerator(config)
        self.supertree = SupertreeReconciler(config)
        self.refinement = RefinementPass(config)

        self._compiled = False

    def forward(self, msa: torch.Tensor, mask: Optional[torch.Tensor] = None,
                chunk_size: int = 5000) -> str:
        return infer_tree(msa, [], self.config, self, chunk_size=chunk_size)

    def compile(self, mode: str = "reduce-overhead", dynamic: bool = True) -> "PhyloGriffinV3":
        compile_model(self, mode=mode, dynamic=dynamic)
        return self

    def uncompile(self) -> "PhyloGriffinV3":
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        if hasattr(self.column_processor, "reset_layers"):
            self.column_processor.reset_layers()
        self._compiled = False
        return self


def compile_model(
    model: "PhyloGriffinV3",
    mode: str = "reduce-overhead",
    dynamic: bool = True,
    compile_griffin: bool = True,
    compile_diffusion: bool = True,
    compile_supertree: bool = False,
) -> "PhyloGriffinV3":
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile not available (requires PyTorch >= 2.0). "
            "Skipping compilation. Performance will be significantly slower."
        )
        return model

    compile_opts = {"dynamic": dynamic}
    if mode != "default":
        compile_opts["mode"] = mode

    if compile_griffin:
        processor = model.column_processor
        if hasattr(processor, "layers") and len(processor.layers) > 0:
            original_layers = list(processor.layers)
            wrapper = LayerSequenceWrapper(original_layers)
            compiled_seq = torch.compile(wrapper, fullgraph=True, **compile_opts)
            processor.layers = compiled_seq
            processor._compiled = True
            print(f"[compile_model] ColumnProcessor layers compiled "
                  f"(mode={mode}, dynamic={dynamic})")

    if compile_diffusion and hasattr(model.diffusion, "denoiser"):
        denoiser = model.diffusion.denoiser
        model.diffusion.denoiser = torch.compile(denoiser, fullgraph=True, **compile_opts)
        print(f"[compile_model] Diffusion denoiser compiled (mode={mode}, dynamic={dynamic})")

    if compile_supertree:
        reconciler = model.supertree
        model.supertree = torch.compile(reconciler, fullgraph=False, **compile_opts)
        print(f"[compile_model] Supertree reconciler compiled with fullgraph=False")

    return model


@torch.no_grad()
def infer_tree(
    msa: torch.Tensor,
    seq_names: List[str],
    config: PhyloGriffinConfig,
    model: PhyloGriffinV3,
    device: str = "cuda",
    chunk_size: int = 5000,
) -> str:
    model = model.to(device)
    model.eval()

    N, L = msa.shape
    msa = msa.to(device)
    pad_idx = config.pad_idx

    mask = (msa != pad_idx)

    all_embeddings = []
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_msa = msa[chunk_start:chunk_end]
        chunk_mask = mask[chunk_start:chunk_end]

        emb, mem = model.column_processor(chunk_msa, chunk_mask)
        all_embeddings.append(emb)

    seq_embeddings = torch.cat(all_embeddings, dim=0)

    edge_index, edge_weights = model.graph_predictor.build_graph(seq_embeddings)

    subproblems, guide_tree = model.decomposition(
        msa, seq_embeddings, edge_index, edge_weights
    )

    subtrees = []
    for idxs, sub_msa, sub_emb in subproblems:
        idxs = idxs.to(device)
        sub_msa = sub_msa.to(device)
        sub_emb = sub_emb.to(device)
        subtree = model.diffusion.generate(sub_msa, sub_emb)
        subtrees.append((idxs, subtree))

    full_tree = model.supertree(subtrees, guide_tree, seq_embeddings)

    full_tree = model.refinement(full_tree, seq_embeddings)

    return full_tree
