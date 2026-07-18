# PhyloGriffin v3

End-to-end neural phylogenetic tree inference. PhyloGriffin v3 takes a multiple sequence alignment (MSA) as input and produces a phylogenetic tree in Newick format using a six-stage neural pipeline: Griffin-style SSM encoder → learned graph → hierarchical decomposition → diffusion tree generation → supertree reconciliation → NNI refinement.

## Architecture

```
INPUT: MSA tensor (N, L) integers + sequence names list
  │
  ▼
STAGE A: Column Processor (Griffin SSM + Titans Memory)
  Produces: per-sequence embeddings (N, D_model)
  Produces: per-column memory signatures (L, D_mem)
  │
  ▼
STAGE B: Learned Phylogenetic Graph
  Produces: sparse edge list (E, 2), E ≈ k_neighbors × N
  │
  ▼
STAGE C: Hierarchical Decomposition
  Produces: K subproblems, each with N_k ≈ 500–1500 leaves
  │
  ▼
STAGE D: Per-Subproblem Diffusion Tree Generator
  Produces: K subtrees (Newick strings)
  │
  ▼
STAGE E: Learned Supertree Reconciler
  Produces: one global tree (Newick string)
  │
  ▼
STAGE F: Refinement Pass
  Produces: refined global tree (Newick string)
  │
  ▼
OUTPUT: Final phylogenetic tree (Newick string)
```

## Installation

```bash
git clone https://github.com/amichae2/PhyloGriffin_V3.git
cd PhyloGriffin_V3
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick Start

```python
from phylogriffin.config import PhyloGriffinConfig
from phylogriffin.inference import PhyloGriffinV3, infer_tree
from phylogriffin.data import load_msa

# Load your alignment
msa, names = load_msa("alignment.fasta", alphabet="protein")

# Configure model
config = PhyloGriffinConfig()
model = PhyloGriffinV3(config)

# Load trained checkpoint
import torch
checkpoint = torch.load("phylogriffin_v3_full.pt", map_location="cpu")
config = PhyloGriffinConfig.from_dict(checkpoint["config"])
model = PhyloGriffinV3(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Infer tree
tree = infer_tree(msa, names, config, model)
print(tree)
```

## Training Stages

| Stage | Component Trained | Objective | Data Required |
|-------|------------------|-----------|---------------|
| Train-1 | Column Processor | Masked column reconstruction (self-supervised) | Raw MSAs, no trees needed |
| Train-2 | Column Processor + Titans Memory | Contrastive phylogenetic embedding | Simulated MSAs with known trees |
| Train-3 | Graph Predictor | Binary edge classification | Simulated MSAs with known trees |
| Train-4 | Diffusion Denoiser | Denoising score matching | Simulated subtrees |
| Train-5 | Supertree Reconciler | Full tree reconstruction from subtrees | Simulated decomposed trees |
| Train-6 | Refinement Pass | NNI correctness prediction | Simulated trees with errors |

Training is performed on Google Colab. Open `phylogriffin_v3_colab.ipynb` in Colab with a GPU runtime. Total estimated training time: 13–25 hours across six stages.

## Configuration

All hyperparameters live in `phylogriffin/config.py` as dataclasses. See [`PHYLOGRIFFIN_V3_SPEC.md`](PHYLOGRIFFIN_V3_SPEC.md) for the complete architectural specification.

## Testing

```bash
python tests/test_smoke.py
```

## License

MIT License (see [LICENSE](LICENSE) file).

## Citation

```bibtex
@software{phylogriffin_v3,
  author = {Alex Michael},
  title = {PhyloGriffin v3: End-to-End Neural Phylogenetic Tree Inference},
  year = {2026},
  url = {https://github.com/amichae2/PhyloGriffin_V3},
}
```
